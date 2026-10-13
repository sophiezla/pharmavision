"""Training loop (PLAN 2.1): AdamW, warm-up + cosine, bf16 AMP, clipping, early stopping, resume.

Plain PyTorch. Resume state (model, optimizer, scheduler, RNG, history) is written after every
epoch to ``resume.pt`` in the run's checkpoint folder and removed once training finishes; the best
weights (by validation AUROC) are kept as ``best.pt``.
"""

from __future__ import annotations

import csv
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pharmachecked.evaluation.metrics import auroc
from pharmachecked.models.router import router_targets
from pharmachecked.training.losses import build_loss, mix_batch
from pharmachecked.training.optim import build_optimizer
from pharmachecked.utils.logging import get_logger
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


@dataclass
class FitResult:
    """Outcome of :func:`fit`."""

    best_epoch: int
    best_val_auroc: float
    best_val_loss: float
    epochs_run: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def _autocast(device: torch.device, enabled: bool) -> Any:
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda"
    )


@torch.inference_mode()
def predict(
    model: nn.Module, ds: Dataset[dict[str, Any]], device: torch.device, cfg: Any, seed: int
) -> pd.DataFrame:
    """Per-image predictions (columns ``PRED_COLUMNS``); the dataset must return metadata."""
    model.eval()
    rows = []
    loader = DataLoader(ds, batch_size=cfg.eval_batch_size, num_workers=cfg.num_workers)
    for batch in loader:
        with _autocast(device, cfg.amp):
            logits = model(batch["image"].to(device, non_blocking=True))["logits"].float()
        logits = logits.cpu()
        probs = torch.sigmoid(logits)
        for k in range(len(logits)):
            rows.append(
                {
                    "image_id": batch["image_id"][k],
                    "fold": int(batch["fold"][k]),
                    "seed": seed,
                    "y_true": int(batch["label"][k]),
                    "logit": float(logits[k]),
                    "prob": float(probs[k]),
                    "packaging_type": batch["packaging_type"][k],
                    "source_dataset": batch["source_dataset"][k],
                }
            )
    return pd.DataFrame(rows, columns=list(PRED_COLUMNS))


SELECTION_RULES = ("auroc", "auroc_then_loss")
TIE_TOLERANCE = 1e-12


def is_improvement(
    auroc_value: float, loss: float, best_auroc: float, best_loss: float, rule: str
) -> bool:
    """Whether an epoch beats the best so far under the model-selection rule.

    ``auroc``: strictly higher validation AUROC. ``auroc_then_loss`` (ADR-012): higher AUROC,
    or equal AUROC (within floating-point tolerance) and strictly lower validation loss, so
    selection stays defined when validation AUROC saturates at 1.0. Early-stopping patience
    counts epochs without an improvement under the same rule.
    """
    if rule not in SELECTION_RULES:
        raise ValueError(f"unknown selection rule {rule!r}")
    if auroc_value > best_auroc + TIE_TOLERANCE:
        return True
    tied = abs(auroc_value - best_auroc) <= TIE_TOLERANCE
    return rule == "auroc_then_loss" and tied and loss < best_loss


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_log(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def fit(
    model: nn.Module,
    train_ds: Dataset[dict[str, Any]],
    val_ds: Dataset[dict[str, Any]],
    cfg: Any,
    device: torch.device,
    ckpt_dir: Path,
    log_path: Path,
    seed: int,
    on_epoch: Callable[[int, float, float], None] | None = None,
) -> FitResult:
    """Train with early stopping on validation AUROC and restore the best weights.

    Args:
        model: Model returning ``{"logits": (B,)}``; ``model.backbone`` (if any) gets the
            backbone learning-rate multiplier.
        train_ds: Training dataset (augmented).
        val_ds: Validation dataset (evaluation transform, with metadata).
        cfg: The ``train`` config.
        device: Device to train on.
        ckpt_dir: Folder for ``best.pt`` and the temporary ``resume.pt``.
        log_path: ``train_log.csv`` of the run (rewritten after every epoch).
        seed: Run seed (shuffling and augmentation).
        on_epoch: Optional hook ``(epoch, val_auroc, val_loss)`` called after each epoch's
            bookkeeping (e.g. Optuna reporting); an exception raised there stops training.

    Returns:
        A :class:`FitResult`; the model holds the best weights afterwards.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, resume_path = ckpt_dir / "best.pt", ckpt_dir / "resume.pt"
    model.to(device)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        generator=make_generator(seed),
        pin_memory=device.type == "cuda",
    )
    opt, sched = build_optimizer(model, cfg, len(loader))
    loss_fn = nn.BCEWithLogitsLoss()  # validation loss: always plain BCE (ADR-012 selection)
    train_loss = build_loss(cfg, getattr(train_ds, "labels", ()))
    mix_mode, mix_alpha = str(cfg.get("mix", "none")), float(cfg.get("mix_alpha", 1.0))
    router_weight = float(cfg.get("router_weight", 0.3))
    history: list[dict[str, float]] = []
    # -inf so the first epoch is always saved, even if its val AUROC is undefined (scored -1).
    start_epoch, best_auroc, best_epoch, elapsed = 0, float("-inf"), -1, 0.0
    best_loss = float("inf")
    rule = cfg.get("selection", "auroc")

    if cfg.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        loader.generator.set_state(state["loader_generator"])
        _set_rng_state(state["rng"])
        start_epoch, history = state["epoch"] + 1, state["history"]
        best_auroc, best_epoch, elapsed = state["best_auroc"], state["best_epoch"], state["seconds"]
        best_loss = state.get("best_loss", float("inf"))
        log.info("resumed at epoch %d (best val AUROC %.4f)", start_epoch, best_auroc)

    for epoch in range(start_epoch, cfg.max_epochs):
        if best_epoch >= 0 and epoch - best_epoch > cfg.patience:
            break
        t0 = time.perf_counter()
        model.train()
        losses = []
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()
            images, targets = mix_batch(images, labels, mix_mode, mix_alpha)
            opt.zero_grad(set_to_none=True)
            with _autocast(device, cfg.amp):
                out = model(images)
            loss = train_loss(out["logits"].float(), targets)
            if "router_logits" in out and mix_mode == "none":
                # Auxiliary packaging-type loss (PLAN 3.5); not_a_drug images are ignored (ADR-009).
                router_target = router_targets(batch["packaging_type"]).to(device)
                if (router_target >= 0).any():
                    loss = loss + router_weight * nn.functional.cross_entropy(
                        out["router_logits"].float(), router_target, ignore_index=-1
                    )
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = predict(model, val_ds, device, cfg, seed)
        val_auroc = auroc(val["y_true"], val["prob"])
        val_loss = float(
            loss_fn(
                torch.tensor(val["logit"].to_numpy()),
                torch.tensor(val["y_true"].to_numpy()).float(),
            )
        )
        elapsed += time.perf_counter() - t0
        score = -1.0 if np.isnan(val_auroc) else val_auroc
        improved = is_improvement(score, val_loss, best_auroc, best_loss, rule)
        if improved:
            best_auroc, best_loss, best_epoch = score, val_loss, epoch
            torch.save(model.state_dict(), best_path)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "lr_head": opt.param_groups[-1]["lr"],
                "seconds": elapsed,
                "best": int(improved),
            }
        )
        _write_log(log_path, history)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "loader_generator": loader.generator.get_state(),
                "rng": _rng_state(),
                "epoch": epoch,
                "history": history,
                "best_auroc": best_auroc,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "seconds": elapsed,
            },
            resume_path,
        )
        log.info(
            "epoch %d: train loss %.4f, val loss %.4f, val AUROC %.4f%s",
            epoch,
            history[-1]["train_loss"],
            val_loss,
            val_auroc,
            " (best)" if improved else "",
        )
        if on_epoch is not None:
            on_epoch(epoch, val_auroc, val_loss)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    resume_path.unlink(missing_ok=True)  # temporary resume state of this run only
    return FitResult(best_epoch, best_auroc, best_loss, len(history), elapsed, history)
