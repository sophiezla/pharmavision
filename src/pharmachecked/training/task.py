"""The ``train`` task: one model, one fold, one seed, full protocol outputs (CLAUDE.md section 7).

Trains on the fold's inner train split, early-stops and picks the decision threshold on val,
predicts the held-out fold (OOF) and every external image (ADR-005), and writes train_log.csv,
metrics.json, oof_predictions.csv, external_predictions.csv and checkpoint_path.txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

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


def _cap(ids: tuple[str, ...], limit: Any) -> list[str]:
    return list(ids if limit is None else ids[: int(limit)])


def run_training(cfg: DictConfig, run: RunDir) -> None:
    """Train ``cfg.model`` on fold ``cfg.fold`` with seed ``cfg.seed`` and write all artifacts."""
    if cfg.train.threshold_rule != "youden":
        raise ValueError(f"unsupported threshold rule {cfg.train.threshold_rule!r}")
    manifest, external, splits, images = load_data(cfg)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    mean, std = list(cfg.model.mean), list(cfg.model.std)
    train_tf = build_train_transform(tcfg, mean, std, size=size)
    eval_tf = build_eval_transform(tcfg, mean, std, size=size)

    def dataset(ids: list[str], tf: Any, table: pd.DataFrame = manifest) -> ManifestDataset:
        return ManifestDataset(table, ids, images, tf, return_meta=True)

    train_ds = dataset(_cap(splits.ids(cfg.fold, "train"), cfg.data.max_train_images), train_tf)
    val_ds = dataset(list(splits.ids(cfg.fold, "val")), eval_tf)
    held_ds = dataset(list(splits.ids(cfg.fold, "heldout")), eval_tf)
    ext_rows = external
    if cfg.data.max_external_per_source is not None:
        ext_rows = external.groupby("source_dataset").head(int(cfg.data.max_external_per_source))
    ext_ds = dataset(list(ext_rows["image_id"]), eval_tf, external)

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

    val = predict(model, val_ds, device, cfg.train, cfg.seed)
    threshold = youden_threshold(val["y_true"], val["prob"])
    held = predict(model, held_ds, device, cfg.train, cfg.seed)
    ext = predict(model, ext_ds, device, cfg.train, cfg.seed)
    held[list(PRED_COLUMNS)].to_csv(
        run.file("oof_predictions.csv"), index=False, lineterminator="\n"
    )
    ext[list(PRED_COLUMNS)].to_csv(
        run.file("external_predictions.csv"), index=False, lineterminator="\n"
    )
    external_metrics = {
        source: single_class_metric(part["y_true"], part["prob"], threshold)
        for source, part in ext.groupby("source_dataset")
    }
    write_json(
        run.file("metrics.json"),
        {
            "heldout": binary_metrics(held["y_true"], held["prob"], threshold),
            "external": external_metrics,
            "val_auroc_best": result.best_val_auroc,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "train_seconds": result.seconds,
            "threshold_rule": cfg.train.threshold_rule,
            "n": {
                "train": len(train_ds),
                "val": len(val_ds),
                "heldout": len(held_ds),
                "external": len(ext_ds),
            },
            "device": str(device),
        },
    )
    best = (ckpt_dir / "best.pt").resolve()
    try:
        text = best.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        text = best.as_posix()
    run.file("checkpoint_path.txt").write_text(text + "\n", encoding="utf-8")
    log.info(
        "run complete: best epoch %d, val AUROC %.4f", result.best_epoch, result.best_val_auroc
    )
