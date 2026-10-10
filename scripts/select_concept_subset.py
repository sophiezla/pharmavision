"""Select the stratified 300-image concept-annotation subset from the development pool (PLAN 1.8).

Only development images are eligible (never the locked test set). Stratified by label with a
fixed seed; writes data/annotations/concepts.csv with empty concept columns for the app.

Usage:
    python scripts/select_concept_subset.py [--n 300]
"""

from __future__ import annotations

import argparse
import sys

from sklearn.model_selection import train_test_split

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.annotations import ConceptStore
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import load_dev_splits
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("select_concept_subset")
SEED = 42


def select(dev_ids: list[str], labels: list[int], n: int, seed: int = SEED) -> list[str]:
    """Return ``n`` ids stratified by label (deterministic for a seed)."""
    if n >= len(dev_ids):
        return sorted(dev_ids)
    chosen, _ = train_test_split(dev_ids, train_size=n, stratify=labels, random_state=seed)
    return sorted(chosen)


def main() -> int:
    """Select and write the subset; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    configure_logging()
    out = REPO_ROOT / "data" / "annotations" / "concepts.csv"
    if out.exists():
        log.error("%s exists; refusing to overwrite annotations", out)
        return 1
    dev = sorted(load_dev_splits(REPO_ROOT / "splits" / "v1.json").all_ids())
    manifest = read_manifest(REPO_ROOT / "data" / "processed" / "manifest_v1.csv")
    label = dict(zip(manifest["image_id"], manifest["label"], strict=True))
    ids = select(dev, [label[i] for i in dev], args.n)
    ConceptStore.create(out, ids)
    log.info("wrote %s: %d of %d development images", out, len(ids), len(dev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
