"""Pack run folders (without weights) and benchmark tables into a verified zip bundle.

Run on the GPU PC after a batch of runs, then bring the zip back and unpack it with
``scripts/unpack_results.py``.

Usage:
    python scripts/pack_results.py                       # every run folder in results/runs
    python scripts/pack_results.py --glob "e01_*"        # a subset
    python scripts/pack_results.py --out bundles/e01.zip
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path

from pharmachecked.config import REPO_ROOT
from pharmachecked.utils.bundle import pack
from pharmachecked.utils.logging import configure_logging, get_logger
from pharmachecked.utils.run_dir import slug

log = get_logger("pack_results")


def main() -> int:
    """Parse arguments and write the bundle; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glob", default="*", help="run-folder pattern under results/runs")
    parser.add_argument("--out", type=Path, default=None, help="zip path (default: bundles/...)")
    args = parser.parse_args()
    configure_logging()
    runs_root = REPO_ROOT / "results" / "runs"
    runs = sorted(p for p in runs_root.glob(args.glob) if p.is_dir())
    tables = sorted((REPO_ROOT / "results" / "tables").glob("bench_*.md"))
    if not runs and not tables:
        log.error("nothing to pack: no run folders match %r and no benchmark tables", args.glob)
        return 1
    host = slug(platform.node() or "host")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or REPO_ROOT / "bundles" / f"results_{host}_{stamp}.zip"
    result = pack(runs, tables, out, created_by=host)
    log.info(
        "wrote %s: %d run folders, %d files, %d weight files left out",
        result.path,
        result.n_runs,
        result.n_files,
        len(result.skipped_weights),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
