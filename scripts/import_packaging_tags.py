"""Import Sophie's earlier packaging-type tags as final labels (ADR-007).

The earlier tag file lists each image by its raw path relative to that project's metadata folder.
Rows are matched to v1 image ids by hashing those files (the image id is the SHA-256 prefix of
the original bytes). Only images without a final label are filled; nothing is overwritten.

Usage:
    python scripts/import_packaging_tags.py PATH/TO/manual_modality_tags.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.annotations import PACKAGING_TYPES, PackagingStore
from pharmachecked.data.normalize import image_id_from_bytes
from pharmachecked.utils.io import sha256_file
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("import_packaging_tags")
ANNOTATOR = "Sophie (imported from earlier manual_modality_tags.csv)"


def match_tags(tags: pd.DataFrame, base: Path) -> dict[str, str]:
    """Map v1 image id -> tag; raises on files that cannot be found or ids tagged twice."""
    out: dict[str, str] = {}
    for rel, tag in zip(tags["orig_relpath"], tags["modality"], strict=True):
        path = (base / rel).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        image_id = image_id_from_bytes(path.read_bytes())
        if out.get(image_id, tag) != tag:
            raise ValueError(f"conflicting tags for {image_id}")
        out[image_id] = tag
    return out


def main() -> int:
    """Import and save; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tags", type=Path)
    args = parser.parse_args()
    configure_logging()
    tags = pd.read_csv(args.tags, dtype=str, keep_default_na=False)
    unknown = set(tags["modality"]) - set(PACKAGING_TYPES)
    if unknown:
        log.error("tags outside the scheme %s: %s", PACKAGING_TYPES, unknown)
        return 1
    mapping = match_tags(tags, args.tags.parent)
    store = PackagingStore(REPO_ROOT / "data" / "annotations" / "packaging_type.csv")
    ids = set(store.ids())
    filled = 0
    for image_id, tag in sorted(mapping.items()):
        if image_id in ids and store.row(image_id)["final_label"] == "":
            store.set_label(image_id, tag, ANNOTATOR)
            filled += 1
    log.info(
        "source sha256 %s: %d tag rows, %d distinct images, %d in the pool, %d filled; "
        "%d of %d images still need a label",
        sha256_file(args.tags),
        len(tags),
        len(mapping),
        len(ids & set(mapping)),
        filled,
        len(ids) - store.n_done(),
        len(ids),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
