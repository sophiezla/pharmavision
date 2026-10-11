"""Run Phase 2 on the GPU kit: E1, pre-registered decisions, Optuna, tuned re-runs, E2.

Usage (the kit's 2_RUN.bat does this):
    python scripts/phase2_pipeline.py --profile full
    python scripts/phase2_pipeline.py --profile selftest     (tiny CPU dry run on toy data)
"""

from __future__ import annotations

import argparse
import sys

from pharmachecked.pipeline.phase2 import PROFILES, run
from pharmachecked.utils.logging import configure_logging


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    configure_logging()
    return run(args.profile)


if __name__ == "__main__":
    sys.exit(main())
