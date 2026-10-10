"""CLIP zero-shot packaging-type pre-labels for every development-pool image (PLAN 1.7).

Covers the whole Kaggle pool, locked-test images included: packaging type is label-blind
metadata (ADR-002), and no model is trained or evaluated here. Writes
data/annotations/packaging_type.csv with empty final labels for review in the app.

Usage:
    python scripts/prelabel_packaging.py
"""

from __future__ import annotations

import argparse
import sys

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.annotations import PackagingStore
from pharmachecked.data.clip_prelabel import prelabel
from pharmachecked.data.manifest import read_manifest
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("prelabel_packaging")


def main() -> int:
    """Pre-label and write the store; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=32, help="lower it when memory is short")
    args = parser.parse_args()
    configure_logging()
    out = REPO_ROOT / "data" / "annotations" / "packaging_type.csv"
    if out.exists():
        log.error("%s exists; refusing to overwrite annotations", out)
        return 1
    manifest = read_manifest(REPO_ROOT / "data" / "processed" / "manifest_v1.csv")
    ids = sorted(manifest["image_id"])
    log.info("pre-labelling %d images with CLIP ViT-B/16 (OpenAI)", len(ids))
    labels = prelabel(ids, REPO_ROOT / "data" / "processed" / "images", batch_size=args.batch_size)
    store = PackagingStore.create(out, labels)
    log.info("wrote %s (%d rows)", out, len(store.ids()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
