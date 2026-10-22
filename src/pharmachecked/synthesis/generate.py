"""Generate the E14 synthetic counterfeits and their manifest (PLAN 7.1-7.3).

Sources: every authentic development image of the split version (never the locked test set:
ids come from ``load_dev_splits``). Each source gets ``per_source`` edits per generator
(``procedural``, ``inpaint``), seeded by (generator, source, k). A synthetic image inherits the
source's ``group_id``, held-out ``fold`` and packaging type and has ``label = 1`` and
``split = "synthetic"``. Which synthetic images a run may train on is decided per run: only
those whose source is in that run's training partition (``training_images``), so validation,
calibration and held-out folds stay real and source-disjoint (Rule 3).

Layout under ``<data_dir>/processed/synthetic/``: ``images/<id>.png``, ``masks/<id>.png``
(255 = edited), ``ocr/<source>.json`` and ``synthetic_manifest.csv``. Generation is resumable.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.normalize import image_id_from_bytes
from pharmachecked.utils.io import atomic_write_text
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)

GENERATORS = ("procedural", "inpaint")
SYNTHETIC_SPLIT = "synthetic"
MANIFEST_NAME = "synthetic_manifest.csv"
COLUMNS = (
    "image_id",
    "source_image_id",
    "generator",
    "k",
    "edit_type",
    "strength",
    "prompt",
    "mask_fraction",
    "label",
    "split",
    "group_id",
    "fold",
    "source_dataset",
    "packaging_type",
    "width",
    "height",
)


def load_synthesis_config() -> DictConfig:
    """``configs/synthesis/e14.yaml``."""
    return OmegaConf.load(CONFIG_DIR / "synthesis" / "e14.yaml")


@dataclass(frozen=True)
class SyntheticStore:
    """Paths of the synthetic-image store."""

    root: Path

    @staticmethod
    def under(data_dir: Path) -> SyntheticStore:
        """The store of a data folder."""
        return SyntheticStore(Path(data_dir) / "processed" / "synthetic")

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def ocr(self) -> Path:
        return self.root / "ocr"

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_NAME

    def read(self) -> pd.DataFrame:
        """The synthetic manifest (empty with the right columns when missing)."""
        if not self.manifest.is_file():
            return pd.DataFrame(columns=list(COLUMNS))
        table = pd.read_csv(
            self.manifest, dtype={"image_id": str, "source_image_id": str, "group_id": str}
        )
        return table.assign(prompt=table["prompt"].fillna(""))

    def write(self, table: pd.DataFrame) -> None:
        """Write the manifest atomically, sorted by generator, source and k."""
        out = table[list(COLUMNS)].sort_values(["generator", "source_image_id", "k"])
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.manifest, out.to_csv(index=False, lineterminator="\n"))


def synthetic_store(cfg: DictConfig) -> SyntheticStore:
    """The store a run config uses (the toy data set keeps its own under ``toy_dir``)."""
    if cfg.data.dataset == "toy":
        return SyntheticStore(Path(cfg.paths.toy_dir) / "synthetic")
    return SyntheticStore.under(Path(cfg.paths.data_dir))


def source_table(cfg: DictConfig) -> tuple[pd.DataFrame, Path]:
    """Authentic development images of the run config's data (manifest rows) and their folder.

    Raises:
        PermissionError: If a selected row is not a development image (Rule 2).
    """
    from pharmachecked.training.task import load_data

    manifest, _, splits, images = load_data(cfg)
    dev_ids = splits.all_ids()
    rows = manifest[manifest["image_id"].isin(dev_ids) & (manifest["label"] == 0)]
    if not (rows["split"] == "dev").all():
        raise PermissionError("synthetic sources must be development images")
    return rows.sort_values("image_id").reset_index(drop=True), images


def edit_seed(generator: str, source_id: str, k: int, base: int) -> int:
    """Seed of one edit, stable across runs and machines."""
    digest = hashlib.sha256(f"{base}:{generator}:{source_id}:{k}".encode()).hexdigest()
    return int(digest[:12], 16)


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def generate(
    run_cfg: DictConfig,
    cfg: DictConfig,
    generators: tuple[str, ...] = GENERATORS,
    limit: int | None = None,
) -> pd.DataFrame:
    """Create every missing synthetic image; returns the full manifest.

    Args:
        run_cfg: A composed run config naming the data (paths, data set, split version).
        cfg: Synthesis config (``configs/synthesis/e14.yaml``).
        generators: Generators to run.
        limit: Only the first ``limit`` sources (self-test).
    """
    from pharmachecked.synthesis.inpaint import Inpainter
    from pharmachecked.synthesis.ocr import OcrReader, cached_boxes
    from pharmachecked.synthesis.procedural import random_edit

    store = synthetic_store(run_cfg)
    for folder in (store.images, store.masks, store.ocr):
        folder.mkdir(parents=True, exist_ok=True)
    sources, images_dir = source_table(run_cfg)
    if limit is not None:
        sources = sources.head(limit)
    table = store.read()
    done = set(
        zip(table["generator"], table["source_image_id"], table["k"].astype(int), strict=True)
    )
    reader = OcrReader(str(cfg.ocr_backend))
    inpainter: Any = None
    new_rows: list[dict[str, Any]] = []
    for n, src in enumerate(sources.itertuples(index=False), start=1):
        with Image.open(images_dir / f"{src.image_id}.png") as im:
            image = im.convert("RGB")
        boxes = cached_boxes(reader, src.image_id, image, store.ocr)
        for generator in generators:
            for k in range(int(cfg.per_source)):
                if (generator, src.image_id, k) in done:
                    continue
                rng = np.random.default_rng(edit_seed(generator, src.image_id, k, int(cfg.seed)))
                prompt = ""
                if generator == "procedural":
                    edit = random_edit(image, boxes, rng, tuple(cfg.procedural.types))
                elif generator == "inpaint":
                    if inpainter is None:
                        inpainter = Inpainter(cfg.inpaint)
                    edit, prompt = inpainter.edit(image, boxes, rng)
                else:
                    raise ValueError(f"unknown generator {generator!r}")
                data = _png(edit.image)
                image_id = image_id_from_bytes(data)
                (store.images / f"{image_id}.png").write_bytes(data)
                Image.fromarray(edit.mask.astype(np.uint8) * 255).save(
                    store.masks / f"{image_id}.png"
                )
                new_rows.append(
                    {
                        "image_id": image_id,
                        "source_image_id": src.image_id,
                        "generator": generator,
                        "k": k,
                        "edit_type": edit.edit_type,
                        "strength": round(float(edit.strength), 4),
                        "prompt": prompt,
                        "mask_fraction": round(float(edit.mask.mean()), 5),
                        "label": 1,
                        "split": SYNTHETIC_SPLIT,
                        "group_id": src.group_id,
                        "fold": src.fold,
                        "source_dataset": f"synthetic_{generator}",
                        "packaging_type": src.packaging_type,
                        "width": edit.image.width,
                        "height": edit.image.height,
                    }
                )
        if new_rows and (n % 25 == 0 or n == len(sources)):
            table = pd.concat([table, pd.DataFrame(new_rows)], ignore_index=True)
            store.write(table)
            new_rows = []
            log.info("synthesis: %d / %d sources", n, len(sources))
    if new_rows:
        table = pd.concat([table, pd.DataFrame(new_rows)], ignore_index=True)
        store.write(table)
    return store.read()


def training_images(
    table: pd.DataFrame,
    train_ids: list[str],
    generator: str,
    n: int,
    seed: int,
) -> list[str]:
    """``n`` synthetic image ids of ``generator`` whose source is in ``train_ids``.

    Drawn without replacement with a generator seeded by ``seed``; all candidates when fewer
    than ``n`` exist (the caller records the count).
    """
    pool = table[(table["generator"] == generator) & table["source_image_id"].isin(set(train_ids))]
    ids = sorted(pool["image_id"])
    if n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, size=n, replace=False))
