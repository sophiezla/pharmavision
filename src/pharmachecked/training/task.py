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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import DevSplits, load_dev_splits
from pharmachecked.data.toy import make_toy_dataset
from pharmachecked.data.transforms import build_eval_transform, build_train_transform
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


def partitions(cfg: DictConfig, splits: DevSplits, external: pd.DataFrame) -> Partitions:
    """Fold partitions, with the smoke-run caps applied when configured."""
    train = list(splits.ids(cfg.fold, "train"))
    if cfg.data.max_train_images is not None:
        train = train[: int(cfg.data.max_train_images)]
    ext = external
    if cfg.data.max_external_per_source is not None:
        ext = external.groupby("source_dataset").head(int(cfg.data.max_external_per_source))
    return Partitions(
        train=train,
        val=list(splits.ids(cfg.fold, "val")),
        heldout=list(splits.ids(cfg.fold, "heldout")),
        external=list(ext["image_id"]),
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
            "heldout": binary_metrics(held["y_true"], held["prob"], threshold),
            "external": {
                source: single_class_metric(part["y_true"], part["prob"], threshold)
                for source, part in ext.groupby("source_dataset")
            },
            "threshold_rule": cfg.train.threshold_rule,
            "n": {"val": len(val), "heldout": len(held), "external": len(ext)},
            **extra,
        },
    )
    run.file("checkpoint_path.txt").write_text(_path_text(checkpoint) + "\n", encoding="utf-8")


def _write_single_row_log(run: RunDir, row: dict[str, float]) -> None:
    with run.file("train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def run_training(cfg: DictConfig, run: RunDir) -> None:
    """Train ``cfg.model`` on fold ``cfg.fold`` with seed ``cfg.seed`` and write all artifacts."""
    _check_threshold_rule(cfg)
    manifest, external, splits, images = load_data(cfg)
    parts = partitions(cfg, splits, external)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    mean, std = list(cfg.model.mean), list(cfg.model.std)
    eval_tf = build_eval_transform(tcfg, mean, std, size=size)

    def dataset(ids: list[str], tf: Any, table: pd.DataFrame = manifest) -> ManifestDataset:
        return ManifestDataset(table, ids, images, tf, return_meta=True)

    train_ds = dataset(parts.train, build_train_transform(tcfg, mean, std, size=size))
    val_ds, held_ds = dataset(parts.val, eval_tf), dataset(parts.heldout, eval_tf)
    ext_ds = dataset(parts.external, eval_tf, external)
    device = device_for(cfg)
    model = build_model(cfg.model)
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / run.run_id
    log.info(
        "%s on %s: train %d, val %d, held-out %d, external %d",
        cfg.model.name,
        device,
        len(train_ds),
        len(val_ds),
        len(held_ds),
        len(ext_ds),
    )
    result = fit(
        model, train_ds, val_ds, cfg.train, device, ckpt_dir, run.file("train_log.csv"), cfg.seed
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
            "device": str(device),
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
    parts = partitions(cfg, splits, external)
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
    parts = partitions(cfg, splits, external)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    eval_tf = build_eval_transform(
        tcfg, list(cfg.model.mean), list(cfg.model.std), size=int(cfg.model.img_size)
    )
    device = device_for(cfg)
    model = build_zeroshot(cfg.model).to(device)

    def scored(table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        ds = ManifestDataset(table, ids, images, eval_tf, return_meta=True)
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
