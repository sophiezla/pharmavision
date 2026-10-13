"""Run Phase 3 on the GPU kit: [E3 retrain], E5 recipe, pre-registered recipe decision, E4 arms.

Usage (the kit's 2_RUN.bat does this):
    python scripts/phase3_pipeline.py --profile full [--split-version 2 --e3-retrain]
    python scripts/phase3_pipeline.py --profile selftest
"""

from __future__ import annotations

import argparse
import sys

from pharmachecked.pipeline.phase3 import PROFILES, run
from pharmachecked.utils.logging import configure_logging


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--split-version", type=int, default=1)
    parser.add_argument("--e3-retrain", action="store_true")
    args = parser.parse_args()
    configure_logging()
    return run(args.profile, args.split_version, args.e3_retrain)


if __name__ == "__main__":
    sys.exit(main())
