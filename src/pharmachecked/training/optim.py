"""Optimizer and learning-rate schedule."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def param_groups(
    model: nn.Module, lr: float, backbone_lr_mult: float, weight_decay: float
) -> list[dict[str, Any]]:
    """AdamW parameter groups: ``model.backbone`` at ``lr * backbone_lr_mult``, the rest at ``lr``.

    Biases and normalization parameters (1-D tensors) get no weight decay. Frozen parameters
    (``requires_grad=False``) are left out.
    """
    backbone = getattr(model, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = (id(p) in backbone_ids, p.ndim > 1)
        groups.setdefault(key, []).append(p)
    out = []
    for (is_backbone, decay), params in sorted(groups.items()):
        out.append(
            {
                "params": params,
                "lr": lr * (backbone_lr_mult if is_backbone else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
            }
        )
    return out


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warm-up from 0 to 1, then cosine decay to 0 at ``total_steps``."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(
    model: nn.Module, cfg: Any, steps_per_epoch: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + per-step warm-up/cosine schedule from the ``train`` config."""
    opt = torch.optim.AdamW(
        param_groups(model, cfg.lr, cfg.backbone_lr_mult, cfg.weight_decay), lr=cfg.lr
    )
    warmup = int(cfg.warmup_epochs * steps_per_epoch)
    total = int(cfg.max_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, warmup, total))
    return opt, sched
