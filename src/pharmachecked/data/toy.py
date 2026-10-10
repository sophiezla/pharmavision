"""Procedurally generated toy dataset for tests and smoke runs (no real images involved).

Counterfeit (label 1) images carry a red disc and authentic (label 0) images a blue square, on a
noisy background, so a tiny model can learn the task in one epoch. The toy set has a valid
manifest, an external manifest and a split file, built with the real pipeline functions.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from pharmachecked.data.dedup import group_ids, phash_hex
from pharmachecked.data.manifest import COLUMNS, write_manifest
from pharmachecked.data.normalize import image_id_from_bytes
from pharmachecked.data.splits import SplitSettings, apply_to_manifest, make_splits, save_splits


@dataclass(frozen=True)
class ToyData:
    """Paths of a generated toy dataset."""

    root: Path
    images_dir: Path
    manifest: Path
    external_manifest: Path
    splits: Path


def _image(label: int, rng: np.random.Generator, size: int) -> bytes:
    noise = (rng.random((size, size, 3)) * 80 + 120).astype(np.uint8)
    im = Image.fromarray(noise)
    draw = ImageDraw.Draw(im)
    r = int(rng.integers(size // 6, size // 3))
    cx, cy = (int(v) for v in rng.integers(r, size - r, 2))
    box = (cx - r, cy - r, cx + r, cy + r)
    if label:
        draw.ellipse(box, fill=(200, 30, 30))
    else:
        draw.rectangle(box, fill=(30, 30, 200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _rows(
    n: int, rng: np.random.Generator, size: int, images_dir: Path, source: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in range(n):
        label = int(k % 3 == 0)  # about one third counterfeit, like the real pool
        data = _image(label, rng, size)
        image_id = image_id_from_bytes(data)
        path = images_dir / f"{image_id}.png"
        path.write_bytes(data)
        rows.append(
            {
                "image_id": image_id,
                "raw_path": f"toy/{source}/{k:03d}.png",
                "source_dataset": source,
                "label": label,
                "width": size,
                "height": size,
                "sha256": hashlib.sha256(data).hexdigest(),
                "phash": phash_hex(path),
                "packaging_type": "",
                "split": "",
            }
        )
    return rows


def make_toy_dataset(root: Path, n: int = 64, n_external: int = 16, seed: int = 0) -> ToyData:
    """Generate the toy dataset under ``root`` (deterministic for a given seed).

    Args:
        root: Output folder (created).
        n: Development-pool size.
        n_external: External-set size (half ``toy_ext_auth`` authentic-only, half
            ``toy_ext_cf`` counterfeit-only, mirroring the real single-class external sets).
        seed: Random seed.

    Returns:
        Paths of the generated files.
    """
    rng = np.random.default_rng(seed)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    dev = pd.DataFrame(_rows(n, rng, 64, images, "toy"))
    dev["group_id"] = group_ids(list(dev["image_id"]), list(dev["phash"]), 0)
    dev["fold"] = pd.array([pd.NA] * len(dev), dtype="Int64")
    spec = make_splits(dev, SplitSettings(), version=1, extra={"manifest": "toy"})
    dev = apply_to_manifest(dev[list(COLUMNS)], spec)

    ext_rows = []
    for source, label in (("toy_ext_auth", 0), ("toy_ext_cf", 1)):
        for row in _rows(n_external // 2, rng, 64, images, source):
            row["label"] = label  # single-class external sets
            ext_rows.append(row)
    ext = pd.DataFrame(ext_rows)
    ext["group_id"] = group_ids(list(ext["image_id"]), list(ext["phash"]), 0)
    ext["split"] = "external"
    ext["fold"] = pd.array([pd.NA] * len(ext), dtype="Int64")

    data = ToyData(
        root=root,
        images_dir=images,
        manifest=root / "manifest_v1.csv",
        external_manifest=root / "manifest_external_v1.csv",
        splits=root / "splits_v1.json",
    )
    write_manifest(dev[list(COLUMNS)], data.manifest)
    write_manifest(ext[list(COLUMNS)], data.external_manifest)
    save_splits(spec, data.splits)
    return data
