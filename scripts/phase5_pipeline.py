"""Run Phase 5 on the GPU kit: E10/E11 prediction-only pass, then the E9 extension.

Usage (the kit's 2_RUN.bat does this):
    python scripts/phase5_pipeline.py --profile full
    python scripts/phase5_pipeline.py --profile selftest
"""

from __future__ import annotations

import argparse
import sys

from pharmachecked.pipeline.phase5 import PROFILES, run
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
