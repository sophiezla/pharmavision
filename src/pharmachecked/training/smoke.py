"""Smoke task (PLAN 1.11): one short training epoch that exercises the whole run-folder contract.

Deliberately minimal. The full trainer (PLAN 2.1) replaces this loop; what must stay identical is
the run-folder output: train_log.csv, metrics.json, oof_predictions.csv, external_predictions.csv
and checkpoint_path.txt next to config.yaml and env.json.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import load_dev_splits
from pharmachecked.data.toy import make_toy_dataset
from pharmachecked.data.transforms import build_eval_transform, build_train_transform
from pharmachecked.models.registry import build_model
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import RunDir
from pharmachecked.utils.seed import make_generator, worker_init_fn

log = get_logger(__name__)
PRED_COLUMNS = (
    "image_id",
    "fold",
    "seed",
    "y_true",
    "logit",
    "prob",
    "packaging_type",
    "source_dataset",
)


def _data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, Any, Path]:
    """Return (dev manifest, external manifest, dev splits, images dir) for real or toy data."""
    if cfg.smoke.dataset == "toy":
        toy = make_toy_dataset(Path(cfg.paths.toy_dir))
        return (
            read_manifest(toy.manifest),
            read_manifest(toy.external_manifest),
            load_dev_splits(toy.splits),
            toy.images_dir,
        )
    processed = Path(cfg.paths.processed_dir)
    return (
        read_manifest(processed / "manifest_v1.csv"),
        read_manifest(processed / "manifest_external_v1.csv"),
        load_dev_splits(Path(cfg.paths.splits_dir) / "v1.json"),
        processed / "images",
    )


@torch.inference_mode()
def _predict(model: torch.nn.Module, ds: ManifestDataset, batch_size: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in DataLoader(ds, batch_size=batch_size):
        logits = model(batch["image"])["logits"]
        for k in range(len(logits)):
            rows.append(
                {
                    "image_id": batch["image_id"][k],
                    "fold": int(batch["fold"][k]),
                    "y_true": int(batch["label"][k]),
                    "logit": float(logits[k]),
                    "prob": float(torch.sigmoid(logits[k])),
                    "packaging_type": batch["packaging_type"][k],
                    "source_dataset": batch["source_dataset"][k],
                }
            )
    return pd.DataFrame(rows)


def _auroc(df: pd.DataFrame) -> float | None:
    return float(roc_auc_score(df["y_true"], df["prob"])) if df["y_true"].nunique() == 2 else None


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity - 1 on validation (0.5 if one class)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def run_smoke(cfg: DictConfig, run: RunDir) -> None:
    """Train ``cfg.model`` for ``cfg.smoke.epochs`` epoch(s) and write every run artifact."""
    s = cfg.smoke
    manifest, external, splits, images = _data(cfg)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    mean, std = list(cfg.model.mean), list(cfg.model.std)
    train_tf = build_train_transform(tcfg, mean, std, size=s.img_size)
    eval_tf = build_eval_transform(tcfg, mean, std, size=s.img_size)
    train_ids = splits.ids(cfg.fold, "train")[: s.max_train_images]
    train_ds = ManifestDataset(manifest, train_ids, images, train_tf, return_meta=True)
    val_ds = ManifestDataset.from_split(manifest, splits, cfg.fold, "val", images, eval_tf, True)
    held_ds = ManifestDataset.from_split(
        manifest, splits, cfg.fold, "heldout", images, eval_tf, True
    )
    ext_rows = external.groupby("source_dataset").head(s.max_external_per_source)
    ext_ds = ManifestDataset(external, list(ext_rows["image_id"]), images, eval_tf, True)

    model = build_model(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=s.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        train_ds,
        batch_size=s.batch_size,
        shuffle=True,
        num_workers=s.num_workers,
        worker_init_fn=worker_init_fn,
        generator=make_generator(cfg.seed),
    )
    log_rows = []
    for epoch in range(s.epochs):
        model.train()
        losses = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch["image"])["logits"], batch["label"].float())
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = _predict(model, val_ds, s.batch_size)
        log_rows.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": _auroc(val)}
        )

    threshold = youden_threshold(val["y_true"].to_numpy(), val["prob"].to_numpy())
    held = _predict(model, held_ds, s.batch_size).assign(seed=cfg.seed)
    ext = _predict(model, ext_ds, s.batch_size).assign(seed=cfg.seed)
    held[list(PRED_COLUMNS)].to_csv(
        run.file("oof_predictions.csv"), index=False, lineterminator="\n"
    )
    ext[list(PRED_COLUMNS)].to_csv(
        run.file("external_predictions.csv"), index=False, lineterminator="\n"
    )
    with run.file("train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(log_rows)
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{run.run_id}.pt"
    torch.save(model.state_dict(), ckpt)
    try:
        ckpt_text = ckpt.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        ckpt_text = ckpt.resolve().as_posix()
    run.file("checkpoint_path.txt").write_text(ckpt_text + "\n", encoding="utf-8")
    predicted = (held["prob"] >= threshold).astype(int)
    write_json(
        run.file("metrics.json"),
        {
            "note": "smoke run: pipeline check only, not a result",
            "dataset": s.dataset,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_heldout": len(held_ds),
            "n_external": len(ext_ds),
            "threshold_val_youden": threshold,
            "heldout_auroc": _auroc(held),
            "heldout_accuracy": float((predicted == held["y_true"]).mean()),
        },
    )
    log.info("smoke run complete: %s", run.root)
