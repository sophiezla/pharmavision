"""Corrupted-held-out predictions and robustness tables (PLAN 4.9-4.10, E9).

Each trained fold model re-predicts its own held-out fold, without retraining, under every
corruption x severity of ``configs/eval/corruptions.yaml`` plus the clean condition. Corruptions
act on the evaluation geometry's output (model-input resolution), so the clean condition here
equals the model's ordinary evaluation input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.corruptions import CORRUPTIONS, corrupt
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.evaluation.metrics import auroc
from pharmachecked.evaluation.stats import bootstrap_ci

CORRUPTION_FILE = "corruption_predictions.csv"
CLEAN = ("clean", 0)


def conditions(cfg_corr: Any) -> list[tuple[str, int]]:
    """``("clean", 0)`` followed by every (corruption, severity 1-5)."""
    return [CLEAN] + [
        (name, s) for name in CORRUPTIONS for s in range(1, len(cfg_corr.severities[name]) + 1)
    ]


class CorruptedView(Dataset):
    """A metadata-returning evaluation dataset with one corruption applied after its geometry."""

    def __init__(
        self, base: ManifestDataset, corruption: str, severity: int, cfg_corr: Any
    ) -> None:
        self.base, self.cfg_corr = base, cfg_corr
        self.corruption, self.severity = corruption, severity

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        if self.severity > 0:
            image_id = item["image_id"]
            image = corrupt(
                self.base._load(image_id), image_id, self.corruption, self.severity, self.cfg_corr
            )
            item["image"] = self.base.transform(image)
        return item


def _rows(preds: pd.DataFrame, corruption: str, severity: int) -> pd.DataFrame:
    keep = preds[["image_id", "fold", "seed", "y_true", "prob"]]
    return keep.assign(corruption=corruption, severity=severity)


def corruption_predictions(
    run_root: Path, paths: DictConfig, checkpoint: Path, cfg_corr: Any, device: str | None = None
) -> pd.DataFrame:
    """Predict the run's held-out fold under every condition; write ``corruption_predictions.csv``.

    Supports runs of task ``train`` (torch models) and ``handcrafted`` (joblib pipelines).

    Returns:
        Long table ``image_id, fold, seed, y_true, prob, corruption, severity``.

    Raises:
        ValueError: For other tasks.
    """
    from pharmachecked.training.task import (  # heavy: data loading and models
        device_for,
        eval_dataset,
        fold_datasets,
        load_data,
        partitions,
    )

    cfg = OmegaConf.load(run_root / "config.yaml")
    cfg.paths = paths
    cfg.device = device
    frames = []
    if cfg.task == "train":
        from pharmachecked.models.registry import build_model
        from pharmachecked.training.trainer import predict

        if "pretrained" in cfg.model:
            cfg.model.pretrained = False  # the checkpoint holds every weight
        if "backbone" in cfg.model:
            cfg.model.backbone.pretrained = False
        dev = device_for(cfg)
        model = build_model(cfg.model)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(dev)
        held = fold_datasets(cfg).heldout
        for name, s in conditions(cfg_corr):
            view = CorruptedView(held, name, s, cfg_corr)
            frames.append(_rows(predict(model, view, dev, cfg.train, int(cfg.seed)), name, s))
    elif cfg.task == "handcrafted":
        import joblib

        from pharmachecked.models.handcrafted import extract_features

        pipeline = joblib.load(checkpoint)
        manifest, external, splits, images = load_data(cfg)
        ids = partitions(cfg, splits, external, manifest).heldout
        tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
        size = int(cfg.model.img_size)
        # Same short-side resize + centre crop as extract_features, so the clean condition
        # reproduces the training-time features up to resampling details.
        base = eval_dataset(cfg, tcfg, manifest, ids, images, [0.0] * 3, [1.0] * 3, size)
        meta = manifest.set_index("image_id").loc[ids]
        geoms = [base._load(i) for i in ids]
        for name, s in conditions(cfg_corr):
            x = np.stack(
                [
                    extract_features(corrupt(g, i, name, s, cfg_corr) if s else g, size)
                    for g, i in zip(geoms, ids, strict=True)
                ]
            )
            prob = pipeline.predict_proba(x)[:, 1]
            preds = pd.DataFrame(
                {
                    "image_id": ids,
                    "fold": meta["fold"].fillna(-1).astype(int).to_numpy(),
                    "seed": int(cfg.seed),
                    "y_true": meta["label"].astype(int).to_numpy(),
                    "prob": prob,
                }
            )
            frames.append(_rows(preds, name, s))
    else:
        raise ValueError(f"corruption evaluation does not support task {cfg.task!r}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(run_root / CORRUPTION_FILE, index=False, lineterminator="\n", float_format="%.6g")
    return out


def severity_table(
    frames: dict[str, list[pd.DataFrame]], n_boot: int, seed: int = 0
) -> pd.DataFrame:
    """Pooled AUROC (95% bootstrap CI) per configuration x condition.

    Args:
        frames: Configuration name -> its runs' corruption-prediction tables.
        n_boot: Bootstrap resamples (stratified, seed ``seed``).

    Returns:
        Columns ``config, corruption, severity, images, auroc, ci_low, ci_high``. Per image and
        condition the probabilities are averaged over seeds first (CLAUDE.md section 5).
    """
    rows = []
    for config, runs in frames.items():
        df = pd.concat(runs, ignore_index=True)
        pooled = df.groupby(["corruption", "severity", "image_id"], as_index=False).agg(
            y_true=("y_true", "first"), prob=("prob", "mean")
        )
        for (name, s), part in pooled.groupby(["corruption", "severity"]):
            y, p = part["y_true"].to_numpy(), part["prob"].to_numpy()
            ci = bootstrap_ci(y, (p,), auroc, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "config": config,
                    "corruption": name,
                    "severity": int(s),
                    "images": len(part),
                    "auroc": ci.estimate,
                    "ci_low": ci.low,
                    "ci_high": ci.high,
                }
            )
    return pd.DataFrame(rows).sort_values(["config", "corruption", "severity"], ignore_index=True)


def quality_limits(table: pd.DataFrame, config: str, threshold: float) -> pd.DataFrame:
    """Highest severity per corruption up to which ``config``'s AUROC stays >= ``threshold``.

    Conservative when AUROC is not monotone in severity: the limit is the largest s such that
    every severity 1..s meets the threshold (0 if severity 1 already fails).
    """
    part = table[(table["config"] == config) & (table["corruption"] != CLEAN[0])]
    rows = []
    for name, sub in part.groupby("corruption"):
        limit = 0
        for _, row in sub.sort_values("severity").iterrows():
            if row["auroc"] < threshold:
                break
            limit = int(row["severity"])
        rows.append({"corruption": name, "max_severity": limit, "threshold": threshold})
    return pd.DataFrame(rows)
