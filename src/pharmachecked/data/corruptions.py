"""Image corruptions with five severity levels for the robustness study (PLAN 4.8, E9).

Corruption families follow Hendrycks & Dietterich (2019), "Benchmarking Neural Network
Robustness to Common Corruptions and Perturbations" (ImageNet-C), plus rotation, perspective
warp and a synthetic glare highlight for handheld photos. Parameters per severity come from
``configs/eval/corruptions.yaml``; severity 0 returns the image unchanged.

Randomness (noise, motion direction, rotation sign, perspective corners, glare position) is
drawn from a generator seeded by the image ID and the corruption name only, so every model sees
the same corrupted image and a higher severity changes the strength, never the direction.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import functional as tvf

from pharmachecked.config import CONFIG_DIR

WHITE = (255, 255, 255)


def load_corruption_config(path: Path | None = None) -> Any:
    """``configs/eval/corruptions.yaml`` (or ``path``)."""
    return OmegaConf.load(path or CONFIG_DIR / "eval" / "corruptions.yaml")


def image_rng(image_id: str, corruption: str) -> np.random.Generator:
    """Generator seeded from ``image_id`` and ``corruption`` (stable across runs and machines)."""
    digest = hashlib.sha256(f"{image_id}/{corruption}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def gaussian_blur(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Gaussian blur with standard deviation ``sigma`` px."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def motion_blur(image: Image.Image, length: float, rng: np.random.Generator) -> Image.Image:
    """Linear motion blur: a normalized line kernel of ``length`` px at a random angle."""
    size = int(length) | 1  # odd
    angle = rng.uniform(0.0, np.pi)
    kernel = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    for t in np.linspace(-c, c, 4 * size):
        kernel[round(c + t * np.sin(angle)), round(c + t * np.cos(angle))] = 1.0
    kernel /= kernel.sum()
    array = _to_array(image)
    pad = np.pad(array, ((c, c), (c, c), (0, 0)), mode="edge")
    out = np.zeros_like(array)
    for dy, dx in zip(*np.nonzero(kernel), strict=True):
        out += kernel[dy, dx] * pad[dy : dy + array.shape[0], dx : dx + array.shape[1]]
    return _to_image(out)


def gaussian_noise(image: Image.Image, sd: float, rng: np.random.Generator) -> Image.Image:
    """Additive Gaussian noise with standard deviation ``sd`` on the 0-1 intensity scale."""
    array = _to_array(image)
    unit = rng.standard_normal(array.shape).astype(np.float32)  # same draw at every severity
    return _to_image(array + sd * unit)


def jpeg(image: Image.Image, quality: float, rng: np.random.Generator) -> Image.Image:
    """Re-encode as JPEG at ``quality``."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def brightness(image: Image.Image, factor: float, rng: np.random.Generator) -> Image.Image:
    """PIL brightness enhancement by ``factor`` (< 1 darkens)."""
    return ImageEnhance.Brightness(image).enhance(factor)


def contrast(image: Image.Image, factor: float, rng: np.random.Generator) -> Image.Image:
    """PIL contrast enhancement by ``factor`` (< 1 lowers contrast)."""
    return ImageEnhance.Contrast(image).enhance(factor)


def rotation(image: Image.Image, degrees: float, rng: np.random.Generator) -> Image.Image:
    """Rotate by ``degrees`` with a random sign, same canvas, white corners (as in training)."""
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return tvf.rotate(
        image, sign * degrees, interpolation=tvf.InterpolationMode.BILINEAR, fill=list(WHITE)
    )


def perspective(image: Image.Image, scale: float, rng: np.random.Generator) -> Image.Image:
    """Perspective warp: each corner moves inwards by up to ``scale`` x the side length."""
    w, h = image.size
    start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    unit = rng.uniform(0.0, 1.0, size=(4, 2))  # same direction at every severity
    inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    end = (np.array(start) + inward * unit * scale * np.array([w, h])).round().astype(int)
    return tvf.perspective(
        image, start, end.tolist(), interpolation=tvf.InterpolationMode.BILINEAR, fill=list(WHITE)
    )


def glare(image: Image.Image, intensity: float, rng: np.random.Generator) -> Image.Image:
    """Additive elliptical highlight (soft-edged) of peak ``intensity`` at a random position."""
    array = _to_array(image)
    h, w = array.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    ry, rx = rng.uniform(0.1, 0.25) * h, rng.uniform(0.15, 0.35) * w
    theta = rng.uniform(0.0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = ((xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)) / rx
    v = (-(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)) / ry
    highlight = np.exp(-(u**2 + v**2) * 2.0)[..., None]
    return _to_image(array + intensity * highlight)


CORRUPTIONS: dict[str, Callable[[Image.Image, float, np.random.Generator], Image.Image]] = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg": jpeg,
    "brightness": brightness,
    "contrast": contrast,
    "rotation": rotation,
    "perspective": perspective,
    "glare": glare,
}


def corrupt(
    image: Image.Image, image_id: str, corruption: str, severity: int, cfg: Any
) -> Image.Image:
    """Apply ``corruption`` at ``severity`` (0 = identity, 1-5 from ``cfg.severities``).

    Raises:
        KeyError: If ``corruption`` is unknown.
        ValueError: If ``severity`` is outside 0-5.
    """
    if corruption not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {corruption!r}; known: {sorted(CORRUPTIONS)}")
    levels = list(cfg.severities[corruption])
    if not 0 <= severity <= len(levels):
        raise ValueError(f"severity {severity} outside 0-{len(levels)}")
    if severity == 0:
        return image
    rng = image_rng(image_id, corruption)
    return CORRUPTIONS[corruption](image.convert("RGB"), float(levels[severity - 1]), rng)
