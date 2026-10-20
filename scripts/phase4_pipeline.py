"""Run Phase 4 on the GPU kit: p4ref retrain, E6, E7, E8, then E9 corruption evaluation.

Usage (the kit's 2_RUN.bat does this):
    python scripts/phase4_pipeline.py --profile full
    python scripts/phase4_pipeline.py --profile selftest
"""

from __future__ import annotations

import argparse
import sys

from pharmachecked.pipeline.phase4 import PROFILES, run
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
