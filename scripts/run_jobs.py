"""Run a job list in order (skip finished runs, resume interrupted ones, log failures).

Usage:
    python scripts/run_jobs.py jobs/e01_cpu.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pharmachecked.config import REPO_ROOT
from pharmachecked.jobs import read_jobs, run_jobs
from pharmachecked.utils.logging import configure_logging


def main() -> int:
    """Run every job; exit 1 if any failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_file", type=Path)
    args = parser.parse_args()
    configure_logging()
    log_file = REPO_ROOT / "results" / "job_logs" / f"{args.job_file.stem}.log"
    outcomes = run_jobs(read_jobs(args.job_file), log_file)
    failed = [o for o in outcomes if o.status == "failed"]
    sys.stdout.write(
        f"{len(outcomes)} jobs: {sum(o.status == 'done' for o in outcomes)} done, "
        f"{sum(o.status == 'skipped' for o in outcomes)} skipped, {len(failed)} failed "
        f"(log: {log_file.relative_to(REPO_ROOT)})\n"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
