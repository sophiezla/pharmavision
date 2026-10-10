"""Verify a results bundle from the GPU PC and install it into results/.

Every file's SHA-256 is checked against the bundle manifest before anything is written, and
nothing that already exists with different content is overwritten.

Usage:
    python scripts/unpack_results.py bundles/results_<host>_<stamp>.zip [--verify-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pharmachecked.config import REPO_ROOT
from pharmachecked.utils.bundle import BundleError, unpack, verify
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("unpack_results")


def main() -> int:
    """Parse arguments, verify and install the bundle; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    configure_logging()
    try:
        if args.verify_only:
            files = verify(args.bundle)
            log.info("bundle OK: %d files verified", len(files))
            return 0
        runs = unpack(args.bundle, REPO_ROOT / "results")
    except BundleError as exc:
        log.error("bundle rejected: %s", exc)
        return 1
    for run in runs:
        log.info("installed results/runs/%s", run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
