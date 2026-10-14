"""Apply the E3 label audit (PLAN 2.14): manifest_v{N} -> manifest_v{N+1}.

Removed images are dropped and relabelled images flipped, as recorded in
``data/annotations/label_audit.csv``. Then run
``python scripts/make_splits.py --version N+1 --manifest data/processed/manifest_v{N+1}.csv``,
which copies locked-test membership unchanged (ADR-002) and re-folds the development pool.

Usage:
    python scripts/apply_label_audit.py --from-version 1
"""

from __future__ import annotations

import argparse
import sys

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.annotations import AuditStore
from pharmachecked.data.manifest import read_manifest, write_manifest
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-version", type=int, required=True)
    args = parser.parse_args()
    configure_logging()
    processed = REPO_ROOT / "data" / "processed"
    src = processed / f"manifest_v{args.from_version}.csv"
    out = processed / f"manifest_v{args.from_version + 1}.csv"
    if out.exists():
        log.error("%s exists; manifests are never overwritten", out)
        return 1
    store = AuditStore(REPO_ROOT / "data" / "annotations" / "label_audit.csv")
    manifest = read_manifest(src)
    nxt = store.apply(manifest)
    write_manifest(nxt, out)
    counts = store.df["decision"].value_counts().to_dict()
    log.info("wrote %s: %d -> %d images (%s)", out.name, len(manifest), len(nxt), counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
