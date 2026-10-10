"""Final, one-time evaluation on the locked test set (PLAN 10.2). The only code allowed to read it.

Refuses to run unless ``--i-understand-this-is-final`` is passed and ``results/LOCK`` does not
exist; on completion it writes ``results/LOCK`` so it can never run again (Rule 2).

The evaluation body is implemented in Phase 10. Until then the guards are live and the script
stops before opening the test set.

Usage:
    python scripts/final_eval.py --i-understand-this-is-final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pharmachecked.config import REPO_ROOT

EXIT_REFUSED = 2
EXIT_NOT_IMPLEMENTED = 3


def main(argv: list[str] | None = None) -> int:
    """Check the guards; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-understand-this-is-final", dest="final", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    lock = args.results_dir / "LOCK"
    if not args.final:
        sys.stderr.write("refused: pass --i-understand-this-is-final to open the locked test set\n")
        return EXIT_REFUSED
    if lock.exists():
        sys.stderr.write(f"refused: {lock} exists; the locked test set was already evaluated\n")
        return EXIT_REFUSED
    sys.stderr.write("not implemented until PLAN 10.2; the locked test set was not opened\n")
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    sys.exit(main())
