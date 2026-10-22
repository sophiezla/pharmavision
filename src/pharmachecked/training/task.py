"""Model tasks: one model, one fold, one seed, full protocol outputs (CLAUDE.md section 7).

``train``          gradient training (trainer.fit) of a registered torch model;
``handcrafted``    colour/HOG/LBP features + logistic regression or XGBoost (E1 tier 1);
``clip_zeroshot``  zero-shot CLIP prompt ensemble, no training (E1 tier 6).

Every task picks the decision threshold on the fold's val split (Youden), predicts the held-out
fold (OOF) and every external image (ADR-005), and writes train_log.csv, metrics.json,
oof_predictions.csv, external_predictions.csv and checkpoint_path.txt.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.datasets import LabelledConcat, ManifestDataset
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import (
    DevSplits,
    SplitSettings,
    inner_split,
    leave_type_out,
    load_dev_splits,
    subsample_groups,
)
from pharmachecked.data.toy import make_toy_dataset
from pharmachecked.data.transforms import (
    build_eval_geometry,
    build_tensorize,
    build_train_transform,
)
from pharmachecked.evaluation.metrics import binary_metrics, single_class_metric, youden_threshold
from pharmachecked.models.registry import build_model
from pharmachecked.training.trainer import PRED_COLUMNS, fit, predict
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import RunDir

log = get_logger(__name__)


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, DevSplits, Path]:
    """Development manifest, external manifest, development splits and images folder."""
    if cfg.data.dataset == "toy":
        toy = make_toy_dataset(Path(cfg.paths.toy_dir))
        return (
            read_manifest(toy.manifest),
            read_manifest(toy.external_manifest),
            load_dev_splits(toy.splits),
            toy.images_dir,
        )
    processed = Path(cfg.paths.processed_dir)
    return (
        read_manifest(processed / f"manifest_v{cfg.data.split_version}.csv"),
        read_manifest(processed / "manifest_external_v1.csv"),
        load_dev_splits(Path(cfg.paths.splits_dir) / f"v{cfg.data.split_version}.json"),
        processed / "images",
    )


def device_for(cfg: DictConfig) -> torch.device:
    """``cuda`` when available unless ``device`` is set explicitly."""
    name = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass(frozen=True)
class Partitions:
    """Image ids used by one run."""

    train: list[str]
    val: list[str]
    heldout: list[str]
    external: list[str]


def partitions(
    cfg: DictConfig, splits: DevSplits, external: pd.DataFrame, manifest: pd.DataFrame
) -> Partitions:
    """Partitions of one run under ``cfg.data.protocol``, with the E6 and smoke caps applied.

    Protocols: ``kfold`` (default; fold ``cfg.fold`` of the split file), ``whole_pool`` (E7: one
    inner train/val/cal split of the whole development pool, no held-out images; ADR-022) and
    ``leave_type_out`` (E8: held-out = every image of ``cfg.data.heldout_type``). Phase 4 splits
    use the protocol seed (``SplitSettings.seed``). ``cfg.data.train_fraction`` (E6) keeps that
    fraction of the training groups, stratified, drawn with seed ``cfg.seed``.

    Raises:
        ValueError: If the protocol is unknown.
    """
    protocol = cfg.data.get("protocol") or "kfold"
    dev = manifest[manifest["image_id"].isin(splits.all_ids())]
    if protocol == "kfold":
        train = list(splits.ids(cfg.fold, "train"))
        val, heldout = list(splits.ids(cfg.fold, "val")), list(splits.ids(cfg.fold, "heldout"))
    elif protocol == "whole_pool":
        parts = inner_split(dev, SplitSettings().seed)
        train, val, heldout = parts["train"], parts["val"], []
    elif protocol == "leave_type_out":
        parts = leave_type_out(dev, str(cfg.data.heldout_type), SplitSettings().seed)
        train, val, heldout = parts["train"], parts["val"], parts["heldout"]
    else:
        raise ValueError(f"unknown data.protocol {protocol!r}")
    fraction = cfg.data.get("train_fraction")
    if fraction is not None and float(fraction) < 1:
        train = subsample_groups(dev[dev["image_id"].isin(train)], float(fraction), int(cfg.seed))
    if cfg.data.max_train_images is not None:
        train = train[: int(cfg.data.max_train_images)]
    ext = external
    if cfg.data.max_external_per_source is not None:
        ext = external.groupby("source_dataset").head(int(cfg.data.max_external_per_source))
    return Partitions(train=train, val=val, heldout=heldout, external=list(ext["image_id"]))


def eval_dataset(
    cfg: DictConfig,
    tcfg: DictConfig,
    table: pd.DataFrame,
    ids: list[str],
    images: Path,
    mean: list[float],
    std: list[float],
    size: int,
) -> ManifestDataset:
    """Evaluation dataset whose deterministic geometry output is cached per image on disk.

    The cache key is the eval mode and size; normalization (model-specific) is applied after
    loading, so every model sharing a size reuses the cache. Results are identical to the
    uncached transform.
    """
    folder = Path(cfg.paths.results_dir) / "scratch" / "eval_cache" / f"{tcfg.eval.mode}_{size}"
    return ManifestDataset(
        table,
        ids,
        images,
        build_tensorize(mean, std),
        return_meta=True,
        geometry_cache=(build_eval_geometry(tcfg, size), folder),
    )


def _check_threshold_rule(cfg: DictConfig) -> None:
    if cfg.train.threshold_rule != "youden":
        raise ValueError(f"unsupported threshold rule {cfg.train.threshold_rule!r}")


def _path_text(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_outputs(
    run: RunDir,
    cfg: DictConfig,
    val: pd.DataFrame,
    held: pd.DataFrame,
    ext: pd.DataFrame,
    checkpoint: Path,
    extra: dict[str, Any],
) -> None:
    """Threshold on val, then write predictions, metrics.json and checkpoint_path.txt."""
    threshold = youden_threshold(val["y_true"], val["prob"])
    held[list(PRED_COLUMNS)].to_csv(
        run.file("oof_predictions.csv"), index=False, lineterminator="\n"
    )
    ext[list(PRED_COLUMNS)].to_csv(
        run.file("external_predictions.csv"), index=False, lineterminator="\n"
    )
    write_json(
        run.file("metrics.json"),
        {
            # E7 (whole_pool) has no held-out images: only the val-selected threshold is kept
            "heldout": (
                binary_metrics(held["y_true"], held["prob"], threshold)
                if len(held)
                else {"threshold": threshold}
            ),
            "external": {
                source: single_class_metric(part["y_true"], part["prob"], threshold)
                for source, part in ext.groupby("source_dataset")
            },
            "threshold_rule": cfg.train.threshold_rule,
            "n": {"val": len(val), "heldout": len(held), "external": len(ext)},
            **extra,
        },
    )
    run.file("checkpoint_path.txt").write_text(
        _path_text(checkpoint) + "\n", encoding="utf-8", newline="\n"
    )


def _write_single_row_log(run: RunDir, row: dict[str, float]) -> None:
    with run.file("train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


@dataclass
class FoldData:
    """Datasets of one fold for a torch model."""

    train: ManifestDataset | LabelledConcat
    val: ManifestDataset
    heldout: ManifestDataset
    external: ManifestDataset
    info: dict[str, Any] = field(default_factory=dict)


ExtraImages = Callable[[Partitions, pd.DataFrame], list[tuple[pd.DataFrame, list[str], Path]]]


def synthetic_extra(cfg: DictConfig) -> ExtraImages | None:
    """E14: synthetic counterfeits whose source is in the run's training partition.

    ``cfg.synthetic = {generator, ratio, only}``: ``round(ratio x real training counterfeits)``
    images of ``generator`` are added (drawn with the run seed); with ``only``, the real
    training counterfeits are removed (:func:`fold_datasets` drops them). None when not
    configured.
    """
    syn = cfg.get("synthetic")
    if syn is None or float(syn.get("ratio", 0)) <= 0:
        return None
    from pharmachecked.synthesis.generate import synthetic_store, training_images

    store = synthetic_store(cfg)

    def extra(parts: Partitions, manifest: pd.DataFrame) -> list:
        labels = manifest.set_index("image_id").loc[parts.train, "label"]
        n = round(float(syn.ratio) * int((labels == 1).sum()))
        table = store.read()
        ids = training_images(table, parts.train, str(syn.generator), n, int(cfg.seed))
        if len(ids) < n:
            log.warning("E14: %d synthetic images requested, %d available", n, len(ids))
        return [(table, ids, store.images)]

    return extra


def fold_datasets(cfg: DictConfig, extra: ExtraImages | None = None) -> FoldData:
    """Train (augmented) and evaluation datasets of fold ``cfg.fold`` for ``cfg.model``.

    ``extra`` (or the E14 ``cfg.synthetic`` block) adds images to the training set only; each
    item is ``(manifest-like table, image ids, images folder)``.
    """
    manifest, external, splits, images = load_data(cfg)
    parts = partitions(cfg, splits, external, manifest)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    mean, std = list(cfg.model.mean), list(cfg.model.std)
    train_tf = build_train_transform(tcfg, mean, std, size=size, augment=cfg.get("augment"))
    extra = extra or synthetic_extra(cfg)
    train_ids = list(parts.train)
    info: dict[str, Any] = {}
    train: ManifestDataset | LabelledConcat = ManifestDataset(
        manifest, train_ids, images, train_tf, True
    )
    if extra is not None:
        added = extra(parts, manifest)
        if cfg.get("synthetic") is not None and cfg.synthetic.get("only"):
            labels = manifest.set_index("image_id").loc[train_ids, "label"]
            train_ids = [i for i in train_ids if int(labels[i]) == 0]
        datasets = [ManifestDataset(manifest, train_ids, images, train_tf, True)]
        datasets += [ManifestDataset(t, ids, d, train_tf, True) for t, ids, d in added if ids]
        train = LabelledConcat(datasets)
        info = {"n_train_real": len(train_ids), "n_train_extra": sum(len(a[1]) for a in added)}
    return FoldData(
        train=train,
        val=eval_dataset(cfg, tcfg, manifest, parts.val, images, mean, std, size),
        heldout=eval_dataset(cfg, tcfg, manifest, parts.heldout, images, mean, std, size),
        external=eval_dataset(cfg, tcfg, external, parts.external, images, mean, std, size),
        info=info,
    )


def run_training(
    cfg: DictConfig,
    run: RunDir,
    extra: ExtraImages | None = None,
    model: torch.nn.Module | None = None,
    hooks: Callable[[torch.nn.Module, torch.device], dict[str, Any]] | None = None,
) -> None:
    """Train ``cfg.model`` on fold ``cfg.fold`` with seed ``cfg.seed`` and write all artifacts.

    Args:
        cfg: Composed run config.
        run: Run folder.
        extra: Extra training images (see :func:`fold_datasets`).
        model: A prebuilt model (default: ``build_model(cfg.model)``).
        hooks: ``(model, device) -> {"batch_hook": ..., "loss_hook": ...}`` passed to ``fit``
            (E13, E15, E16).
    """
    _check_threshold_rule(cfg)
    data = fold_datasets(cfg, extra)
    train_ds, val_ds, held_ds, ext_ds = data.train, data.val, data.heldout, data.external
    device = device_for(cfg)
    model = model if model is not None else build_model(cfg.model)
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / run.run_id
    if cfg.train.get("grad_checkpointing"):
        enabled = [
            m.set_grad_checkpointing(True) or type(m).__name__
            for m in model.modules()
            if hasattr(m, "set_grad_checkpointing")
        ]
        log.info("gradient checkpointing enabled on %s", enabled or "nothing (unsupported)")
    if int(cfg.train.get("supcon_epochs", 0)) > 0:  # E5 SupCon arm: pretrain, then linear head
        from pharmachecked.training.supcon import freeze_encoder, pretrain

        pretrain(model, train_ds, cfg.train, device, ckpt_dir / "supcon.pt", cfg.seed)
        freeze_encoder(model)
    log.info(
        "%s on %s: train %d, val %d, held-out %d, external %d",
        cfg.model.name,
        device,
        len(train_ds),
        len(val_ds),
        len(held_ds),
        len(ext_ds),
    )
    fit_hooks = hooks(model, device) if hooks is not None else {}
    result = fit(
        model,
        train_ds,
        val_ds,
        cfg.train,
        device,
        ckpt_dir,
        run.file("train_log.csv"),
        cfg.seed,
        **fit_hooks,
    )
    preds = [predict(model, ds, device, cfg.train, cfg.seed) for ds in (val_ds, held_ds, ext_ds)]
    write_outputs(
        run,
        cfg,
        *preds,
        checkpoint=ckpt_dir / "best.pt",
        extra={
            "val_auroc_best": result.best_val_auroc,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "train_seconds": result.seconds,
            "n_train": len(train_ds),
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else None,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(device),
            **data.info,
        },
    )
    log.info(
        "run complete: best epoch %d, val AUROC %.4f", result.best_epoch, result.best_val_auroc
    )


def _rows(table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    rows = table.set_index("image_id").loc[ids].reset_index()
    readable = rows["split"].isin(["dev", "external"])
    if not readable.all():  # same guard as ManifestDataset (Rule 2)
        raise PermissionError("requested images outside the dev/external partitions")
    return rows


def _predictions(rows: pd.DataFrame, prob: np.ndarray, seed: int) -> pd.DataFrame:
    p = np.clip(prob.astype(float), 1e-7, 1 - 1e-7)
    return pd.DataFrame(
        {
            "image_id": rows["image_id"],
            "fold": rows["fold"].fillna(-1).astype(int),
            "seed": seed,
            "y_true": rows["label"].astype(int),
            "logit": np.log(p / (1 - p)),
            "prob": prob.astype(float),
            "packaging_type": rows["packaging_type"],
            "source_dataset": rows["source_dataset"],
        }
    )


def handcrafted_features(ids: list[str], images: Path, cache_dir: Path, size: int) -> np.ndarray:
    """Feature matrix for ``ids``, cached per image (features are label-blind and fixed)."""
    from pharmachecked.models.handcrafted import extract_features

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for image_id in ids:
        cached = cache_dir / f"{image_id}.npy"
        if cached.exists():
            out.append(np.load(cached))
            continue
        with Image.open(images / f"{image_id}.png") as im:
            vec = extract_features(im.convert("RGB"), size)
        np.save(cached, vec)
        out.append(vec)
    return np.vstack(out)


def run_handcrafted(cfg: DictConfig, run: RunDir) -> None:
    """Fit the handcrafted pipeline on the fold's train split and write all artifacts."""
    from pharmachecked.models.handcrafted import build_classifier

    _check_threshold_rule(cfg)
    manifest, external, splits, images = load_data(cfg)
    parts = partitions(cfg, splits, external, manifest)
    size = int(cfg.model.img_size)
    cache = Path(cfg.paths.results_dir) / "scratch" / "features" / f"handcrafted_{size}"

    def features(table: pd.DataFrame, ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        rows = _rows(table, ids)
        return rows, handcrafted_features(ids, images, cache, size)

    train_rows, x_train = features(manifest, parts.train)
    start = time.perf_counter()
    pipeline = build_classifier(cfg.model, cfg.seed)
    pipeline.fit(x_train, train_rows["label"].to_numpy())
    seconds = time.perf_counter() - start

    def score(table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        if not ids:  # E7 (whole_pool) has no held-out images
            return pd.DataFrame(columns=list(PRED_COLUMNS))
        rows, x = features(table, ids)
        return _predictions(rows, pipeline.predict_proba(x)[:, 1], cfg.seed)

    val, held, ext = (
        score(manifest, parts.val),
        score(manifest, parts.heldout),
        score(external, parts.external),
    )
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / run.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ckpt_dir / "model.joblib")
    _write_single_row_log(run, {"epoch": 0, "fit_seconds": seconds, "n_features": x_train.shape[1]})
    write_outputs(
        run,
        cfg,
        val,
        held,
        ext,
        checkpoint=ckpt_dir / "model.joblib",
        extra={
            "train_seconds": seconds,
            "n_train": len(train_rows),
            "n_features": x_train.shape[1],
        },
    )


def run_clip_zeroshot(cfg: DictConfig, run: RunDir) -> None:
    """Score val, held-out and external images with zero-shot CLIP (no training)."""
    from pharmachecked.models.zeroshot import build_zeroshot

    _check_threshold_rule(cfg)
    manifest, external, splits, images = load_data(cfg)
    parts = partitions(cfg, splits, external, manifest)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    mean, std, size = list(cfg.model.mean), list(cfg.model.std), int(cfg.model.img_size)
    device = device_for(cfg)
    model = build_zeroshot(cfg.model).to(device)

    def scored(table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        ds = eval_dataset(cfg, tcfg, table, ids, images, mean, std, size)
        return predict(model, ds, device, cfg.train, cfg.seed)

    val, held, ext = (
        scored(manifest, parts.val),
        scored(manifest, parts.heldout),
        scored(external, parts.external),
    )
    _write_single_row_log(run, {"epoch": 0, "fit_seconds": 0.0})
    # No weights are trained; the "checkpoint" is the model config naming the public weights.
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / run.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "zeroshot_model.yaml").write_text(OmegaConf.to_yaml(cfg.model), encoding="utf-8")
    write_outputs(
        run,
        cfg,
        val,
        held,
        ext,
        checkpoint=ckpt_dir / "zeroshot_model.yaml",
        extra={"train_seconds": 0.0, "device": str(device)},
    )


TASK_FUNCTIONS: dict[str, Callable[[DictConfig, RunDir], None]] = {
    "train": run_training,
    "handcrafted": run_handcrafted,
    "clip_zeroshot": run_clip_zeroshot,
}
