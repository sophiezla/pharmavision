"""Image transforms: default training augmentation and deterministic evaluation preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2


class PadToSquare(torch.nn.Module):
    """Pad a PIL image to a square canvas, centred, with a constant fill colour."""

    def __init__(self, fill: Sequence[int]) -> None:
        super().__init__()
        self.fill = tuple(int(v) for v in fill)

    def forward(self, im: Image.Image) -> Image.Image:
        """Return ``im`` centred on a square canvas of side ``max(width, height)``."""
        side = max(im.size)
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        return canvas


def _to_tensor(mean: Sequence[float], std: Sequence[float]) -> list[Any]:
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ]


def build_train_transform(
    cfg: Any, mean: Sequence[float], std: Sequence[float], size: int | None = None
) -> v2.Compose:
    """Default recipe: RandomResizedCrop, small rotation, brightness/contrast jitter; no flips.

    Args:
        cfg: The ``data/transforms.yaml`` config.
        mean: Per-channel normalization mean (from the model config).
        std: Per-channel normalization std (from the model config).
        size: Output side; defaults to ``cfg.size``.

    Returns:
        A transform mapping a PIL RGB image to a normalized float tensor (3, size, size).
    """
    side = int(size or cfg.size)
    fill = tuple(int(v) for v in cfg.fill)
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                side, scale=tuple(cfg.train.rrc_scale), interpolation=cfg.interpolation
            ),
            v2.RandomRotation(cfg.train.rotation_degrees, interpolation="bilinear", fill=fill),
            v2.ColorJitter(brightness=cfg.train.brightness, contrast=cfg.train.contrast),
            *_to_tensor(mean, std),
        ]
    )


def build_eval_transform(
    cfg: Any, mean: Sequence[float], std: Sequence[float], size: int | None = None
) -> v2.Compose:
    """Deterministic preprocessing: resize + center crop (``crop``) or pad to square (``pad``).

    Args:
        cfg: The ``data/transforms.yaml`` config.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.
        size: Output side; defaults to ``cfg.size``.

    Returns:
        A transform mapping a PIL RGB image to a normalized float tensor (3, size, size).

    Raises:
        ValueError: On an unknown eval mode.
    """
    side = int(size or cfg.size)
    if cfg.eval.mode == "crop":
        geometry: list[Any] = [
            v2.Resize(side, interpolation=cfg.interpolation),
            v2.CenterCrop(side),
        ]
    elif cfg.eval.mode == "pad":
        geometry = [
            PadToSquare(cfg.fill),
            v2.Resize((side, side), interpolation=cfg.interpolation),
        ]
    else:
        raise ValueError(f"unknown eval mode {cfg.eval.mode!r}")
    return v2.Compose([*geometry, *_to_tensor(mean, std)])
