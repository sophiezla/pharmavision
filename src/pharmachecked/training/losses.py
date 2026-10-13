"""Training losses and batch mixing for the E5 recipe arms (PLAN 3.2).

Validation always uses plain BCE, so model selection (ADR-012) is comparable across recipes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def focal_loss(gamma: float, alpha_pos: float) -> Loss:
    """Binary focal loss (Lin et al., 2017) with class weight ``alpha_pos`` for counterfeit.

    loss = -alpha_t * (1 - p_t)^gamma * log(p_t), with alpha_t = alpha_pos for label 1 and
    1 - alpha_pos for label 0. Targets may be soft (then p_t uses the target-weighted mix).
    """

    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * alpha_pos + (1 - targets) * (1 - alpha_pos)
        return (alpha_t * (1 - p_t) ** gamma * bce).mean()

    return loss


def bce_loss(label_smoothing: float = 0.0) -> Loss:
    """BCE on logits; label smoothing maps targets to y * (1 - s) + s / 2."""

    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1 - label_smoothing) + label_smoothing / 2
        return functional.binary_cross_entropy_with_logits(logits, smoothed)

    return loss


def build_loss(cfg: Any, train_labels: Sequence[int]) -> Loss:
    """Training loss from ``train.loss`` (bce | focal) and ``train.label_smoothing``.

    Focal's class-balanced alpha is the share of authentic images in the training split, so the
    rarer class (counterfeit) gets the larger weight.
    """
    name = cfg.get("loss", "bce")
    if name == "bce":
        return bce_loss(float(cfg.get("label_smoothing", 0.0)))
    if name == "focal":
        labels = np.asarray(train_labels)
        alpha_pos = float((labels == 0).mean())
        return focal_loss(float(cfg.get("focal_gamma", 2.0)), alpha_pos)
    raise ValueError(f"unknown loss {name!r}")


def mix_batch(
    images: torch.Tensor, targets: torch.Tensor, mode: str, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """MixUp (Zhang et al., 2018) or CutMix (Yun et al., 2019) on a batch; ``none`` passes through.

    Mixing partners are a random permutation of the batch; lambda ~ Beta(alpha, alpha), and for
    CutMix it is corrected to the pasted box's actual area. Uses the torch RNG (seeded per run).
    """
    if mode == "none" or images.shape[0] < 2:
        return images, targets
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(images.shape[0], device=images.device)
    if mode == "mixup":
        mixed = lam * images + (1 - lam) * images[perm]
    elif mode == "cutmix":
        h, w = images.shape[-2:]
        cut_h, cut_w = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
        y0, y1 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x0, x1 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        mixed = images.clone()
        mixed[..., y0:y1, x0:x1] = images[perm][..., y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    else:
        raise ValueError(f"unknown mix mode {mode!r}")
    return mixed, lam * targets + (1 - lam) * targets[perm]
