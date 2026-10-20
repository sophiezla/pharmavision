"""Per-image branch outputs of a trained multi-branch model (PLAN 3.7).

The training task writes only logits and probabilities. This module re-predicts a run's held-out
fold from its saved checkpoint and keeps the fusion gate weights and router probabilities, so
the mean gate weight per branch per packaging type can be tabulated from OOF outputs. The
recomputed probability is compared with the run's ``oof_predictions.csv`` to confirm that the
checkpoint and evaluation pipeline reproduce the original predictions.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pharmachecked.models.registry import build_model
from pharmachecked.models.router import ROUTER_CLASSES as CLASSES
from pharmachecked.training.trainer import autocast

BRANCH_FILE = "branch_outputs.csv"


def branch_names(cfg_model: Any) -> list[str]:
    """Fusion inputs in the model's order: global, then local and frequency when enabled."""
    names = ["global"]
    if cfg_model.use_local:
        names.append("local")
    if cfg_model.use_frequency:
        names.append("frequency")
    return names


def predict_branch_outputs(
    model: nn.Module,
    ds: Dataset[dict[str, Any]],
    device: torch.device,
    cfg_train: Any,
    seed: int,
    branches: list[str],
) -> pd.DataFrame:
    """Predictions (``PRED_COLUMNS``) plus ``gate_<branch>`` and ``router_<class>`` columns.

    Args:
        model: A multi-branch model returning ``gate_weights`` (and ``router_probs`` with a
            router).
        ds: Evaluation dataset returning metadata.
        device: Device to run on.
        cfg_train: Train config (``eval_batch_size``, ``amp``), as in ``trainer.predict``.
        seed: Seed of the run, copied into the output.
        branches: Names of the fusion inputs, in order (``branch_names``).

    Returns:
        One row per image.

    Raises:
        ValueError: If the number of gate weights does not match ``branches``.
    """
    model.eval()
    rows = []
    loader = DataLoader(ds, batch_size=cfg_train.eval_batch_size, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            with autocast(device, cfg_train.amp):
                out = model(batch["image"].to(device, non_blocking=True))
            logits = out["logits"].float().cpu()
            gates = out["gate_weights"].float().cpu()
            if gates.shape[1] != len(branches):
                raise ValueError(f"{gates.shape[1]} gate weights for branches {branches}")
            router = out["router_probs"].float().cpu() if "router_probs" in out else None
            for k in range(len(logits)):
                row = {
                    "image_id": batch["image_id"][k],
                    "fold": int(batch["fold"][k]),
                    "seed": seed,
                    "y_true": int(batch["label"][k]),
                    "logit": float(logits[k]),
                    "prob": float(torch.sigmoid(logits[k])),
                    "packaging_type": batch["packaging_type"][k],
                    "source_dataset": batch["source_dataset"][k],
                }
                row |= {f"gate_{b}": float(gates[k, j]) for j, b in enumerate(branches)}
                if router is not None:
                    row |= {f"router_{c}": float(router[k, j]) for j, c in enumerate(CLASSES)}
                rows.append(row)
    return pd.DataFrame(rows)


def max_prob_difference(recomputed: pd.DataFrame, original: pd.DataFrame) -> float:
    """Largest absolute probability difference per image between two prediction tables.

    Raises:
        ValueError: If the two tables do not cover the same images.
    """
    merged = recomputed[["image_id", "prob"]].merge(
        original[["image_id", "prob"]], on="image_id", suffixes=("", "_orig")
    )
    if len(merged) != len(recomputed) or len(merged) != len(original):
        raise ValueError("recomputed and original predictions cover different images")
    return float((merged["prob"] - merged["prob_orig"]).abs().max())


def find_checkpoint(run_root: Path, recorded_dir: str, extra_roots: Iterable[Path]) -> Path | None:
    """Locate a run's ``best.pt``.

    Looked for, in order: under the checkpoints folder recorded in the run's config (the machine
    that trained it), then ``<root>/checkpoints/<run_id>/best.pt`` for every extra root.
    """
    candidates = [Path(recorded_dir) / run_root.name / "best.pt"]
    candidates += [Path(r) / "checkpoints" / run_root.name / "best.pt" for r in extra_roots]
    return next((c for c in candidates if c.is_file()), None)


def run_branch_outputs(
    run_root: Path, paths: DictConfig, checkpoint: Path, device: str | None = None
) -> tuple[pd.DataFrame, float]:
    """Re-predict one run's held-out fold from ``checkpoint``; write ``branch_outputs.csv``.

    Args:
        run_root: The run folder (its ``config.yaml`` and ``oof_predictions.csv`` are read).
        paths: Paths config of this machine (replaces the training machine's paths).
        checkpoint: The run's ``best.pt`` (a ``state_dict``).
        device: ``cuda`` / ``cpu``; default: cuda when available.

    Returns:
        The branch-output table and its maximum absolute probability difference against the
        run's OOF predictions.
    """
    from pharmachecked.training.task import device_for, fold_datasets  # heavy: data loading

    cfg = OmegaConf.load(run_root / "config.yaml")
    cfg.paths = paths
    cfg.device = device
    cfg.model.backbone.pretrained = False  # the checkpoint holds every weight
    dev = device_for(cfg)
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(dev)
    out = predict_branch_outputs(
        model, fold_datasets(cfg).heldout, dev, cfg.train, int(cfg.seed), branch_names(cfg.model)
    )
    out.to_csv(run_root / BRANCH_FILE, index=False, lineterminator="\n")
    return out, max_prob_difference(out, pd.read_csv(run_root / "oof_predictions.csv"))


def gate_table(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Mean gate weight per branch per packaging type: mean ± sd over runs of per-run means.

    Args:
        frames: One branch-output table per run (``predict_branch_outputs``).

    Returns:
        Long table with columns ``packaging_type, branch, runs, images, mean, sd`` (``images``
        = held-out images of that type per seed, i.e. counted once per run's fold and summed
        over folds of one seed). ``packaging_type == "all"`` pools every type.
    """
    per_run = []
    for df in frames:
        gates = [c for c in df.columns if c.startswith("gate_")]
        both = pd.concat([df, df.assign(packaging_type="all")])
        means = both.groupby("packaging_type")[gates].mean()
        counts = both.groupby("packaging_type").size()
        for ptype, row in means.iterrows():
            for col in gates:
                per_run.append(
                    {
                        "packaging_type": ptype,
                        "branch": col.removeprefix("gate_"),
                        "fold": int(df["fold"].iloc[0]),
                        "seed": int(df["seed"].iloc[0]),
                        "n": int(counts[ptype]),
                        "gate": float(row[col]),
                    }
                )
    long = pd.DataFrame(per_run)
    images = (
        long.groupby(["packaging_type", "branch", "seed"])["n"].sum().groupby(level=[0, 1]).max()
    )
    out = long.groupby(["packaging_type", "branch"])["gate"].agg(["count", "mean", "std"])
    out = out.rename(columns={"count": "runs", "std": "sd"}).join(images.rename("images"))
    return out.reset_index()[["packaging_type", "branch", "runs", "images", "mean", "sd"]]
