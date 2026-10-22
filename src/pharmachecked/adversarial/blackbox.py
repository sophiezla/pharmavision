"""Score-only black-box attack for E15 (PLAN 7.9, 7.11): random search over procedural edits.

For a counterfeit image (model-input resolution, [0, 1]) the attacker draws up to ``QUERIES``
random local edits and queries only the detector's probability. Each candidate is one edit in a
random box (sides 15-40% of the image): hue rotation (10-45 degrees), Gaussian blur (sigma 1-4
px), halftone, JPEG (quality 5-30), sinusoidal warp (2-8%) or a brightness change (factor
0.7-1.3). The search stops at the first candidate scored below the detector's val-selected
threshold (evasion); otherwise the lowest-scoring candidate is kept. Queries are batched
(``BATCH``); a success inside a batch counts the queries up to that candidate.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from pharmachecked.adversarial.attacks import Normalized
from pharmachecked.synthesis.procedural import halftone

QUERIES = 200
BATCH = 50
KINDS = ("hue", "blur", "halftone", "jpeg", "warp", "brightness")


@dataclass
class SearchResult:
    """Outcome for one image."""

    success: bool
    queries: int
    best_prob: float
    best_image: np.ndarray  # uint8 (H, W, 3)
    kind: str


def _edit(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    w, h = image.size
    bw, bh = int(w * rng.uniform(0.15, 0.4)), int(h * rng.uniform(0.15, 0.4))
    x0, y0 = int(rng.integers(0, w - bw + 1)), int(rng.integers(0, h - bh + 1))
    box = (x0, y0, x0 + bw, y0 + bh)
    patch = image.crop(box)
    kind = KINDS[int(rng.integers(len(KINDS)))]
    if kind == "hue":
        hsv = np.asarray(patch.convert("HSV")).copy()
        shift = round(rng.uniform(10, 45) / 360 * 255) * (1 if rng.random() < 0.5 else -1)
        hsv[..., 0] = ((hsv[..., 0].astype(int) + shift) % 256).astype(np.uint8)
        patch = Image.fromarray(hsv, "HSV").convert("RGB")
    elif kind == "blur":
        patch = patch.filter(ImageFilter.GaussianBlur(float(rng.uniform(1, 4))))
    elif kind == "halftone":
        patch = halftone(patch)
    elif kind == "jpeg":
        buf = io.BytesIO()
        patch.save(buf, format="JPEG", quality=int(rng.integers(5, 31)))
        buf.seek(0)
        patch = Image.open(buf).convert("RGB")
    elif kind == "warp":
        arr = np.asarray(patch).astype(float)
        amp, phase = float(rng.uniform(0.02, 0.08)), float(rng.uniform(0, 2 * np.pi))
        yy, xx = np.mgrid[0:bh, 0:bw].astype(float)
        coords = [
            yy + amp * bh * np.sin(2 * np.pi * xx / max(bw, 1) + phase),
            xx + amp * bw * np.sin(2 * np.pi * yy / max(bh, 1) + phase),
        ]
        chans = [
            ndimage.map_coordinates(arr[..., c], coords, order=1, mode="nearest") for c in range(3)
        ]
        patch = Image.fromarray(np.stack(chans, axis=-1).clip(0, 255).astype(np.uint8))
    else:
        patch = ImageEnhance.Brightness(patch).enhance(float(rng.uniform(0.7, 1.3)))
    out = image.copy()
    out.paste(patch, box[:2])
    return out, kind


def random_search(
    wrapped: Normalized,
    image: np.ndarray,
    threshold: float,
    seed: int,
    queries: int = QUERIES,
    batch: int = BATCH,
) -> SearchResult:
    """Search an evasive edit of one image (uint8 ``(H, W, 3)``) with score-only queries."""
    rng = np.random.default_rng(seed)
    base = Image.fromarray(image)
    device = wrapped.mean.device
    best = SearchResult(False, 0, float("inf"), image, "")
    used = 0
    while used < queries:
        n = min(batch, queries - used)
        cands = [_edit(base, rng) for _ in range(n)]
        x = torch.stack(
            [torch.from_numpy(np.asarray(c)).permute(2, 0, 1).float() / 255 for c, _ in cands]
        ).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(wrapped.logit(x)).cpu().numpy()
        hits = np.flatnonzero(probs < threshold)
        if len(hits):
            k = int(hits[0])
            return SearchResult(
                True, used + k + 1, float(probs[k]), np.asarray(cands[k][0]), cands[k][1]
            )
        k = int(np.argmin(probs))
        if probs[k] < best.best_prob:
            best = SearchResult(False, 0, float(probs[k]), np.asarray(cands[k][0]), cands[k][1])
        used += n
    best.queries = used
    return best
