"""GPU-kit run (2_RUN.bat): benchmark once, run the kit's job list, pack the results.

Keeps Windows awake while running, skips finished runs, resumes interrupted ones, and always
packs whatever is finished into ``RESULTS_<name>_<timestamp>.zip`` in the kit folder.

Usage (done by 2_RUN.bat):
    python scripts/kit_run.py --kit <kit folder>
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pharmachecked.config import REPO_ROOT
from pharmachecked.jobs import read_jobs, run_jobs
from pharmachecked.utils.bundle import pack
from pharmachecked.utils.env import git_info
from pharmachecked.utils.io import read_json
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("kit_run")
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while jobs run (no effect elsewhere)."""
    if os.name == "nt":
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]


def banner(text: str) -> None:
    """Print a framed message for the person watching the window."""
    sys.stdout.write("\n" + "=" * 60 + "\n " + text.replace("\n", "\n ") + "\n" + "=" * 60 + "\n")
    sys.stdout.flush()


def main() -> int:
    """Run the kit; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    kit = read_json(REPO_ROOT / "KIT.json")
    if git_info(REPO_ROOT)["git_dirty"]:
        banner("Files in the kit were changed. Unzip a fresh copy of the kit and try again.")
        return 1
    keep_awake(True)
    try:
        bench = REPO_ROOT / "results" / "tables" / "bench_gpu-pc.md"
        if kit.get("bench", True) and not bench.exists():
            banner("Measuring the GPU speed (a few minutes)...")
            bench_log = REPO_ROOT / "results" / "job_logs" / "bench_gpu-pc.log"
            bench_log.parent.mkdir(parents=True, exist_ok=True)
            with bench_log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    [sys.executable, "scripts/bench_machine.py", "--name", "gpu-pc"],
                    cwd=REPO_ROOT,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode != 0:  # not fatal: estimates use fold-0 runs (CLAUDE.md §12)
                log.warning("GPU benchmark failed (exit %d); see %s", result.returncode, bench_log)
        jobs = read_jobs(REPO_ROOT / kit["job_file"])
        banner(f"Running {len(jobs)} jobs from {kit['job_file']}.\nYou can leave this window open.")
        log_file = REPO_ROOT / "results" / "job_logs" / f"{Path(kit['job_file']).stem}.log"
        outcomes = run_jobs(jobs, log_file)
    finally:
        keep_awake(False)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = sorted(p for p in (REPO_ROOT / "results" / "runs").glob("*") if p.is_dir())
    tables = sorted((REPO_ROOT / "results" / "tables").glob("bench_*.md"))
    logs = {
        f"{kit['name']}_{stamp}_{p.name}": p
        for p in (REPO_ROOT / "results" / "job_logs").glob("*.log")
    }
    setup = REPO_ROOT / "results" / "kit_setup.json"
    if setup.exists():
        logs[f"{kit['name']}_{stamp}_kit_setup.json"] = setup
    out = args.kit / f"RESULTS_{kit['name']}_{stamp}.zip"
    pack(runs, tables, out, created_by=f"gpu-kit {kit['name']} {kit['commit']}", log_files=logs)
    failed = sum(o.status == "failed" for o in outcomes)
    done = sum(o.status in ("done", "skipped") for o in outcomes)
    if failed:
        banner(
            f"{done} of {len(outcomes)} jobs finished, {failed} failed.\n"
            f"Results so far: {out.name}\n"
            "Double-click 2_RUN.bat again to retry the failed ones, or bring the zip back."
        )
        return 1
    banner(f"ALL {len(outcomes)} JOBS FINISHED.\nCopy this file back to the laptop:\n{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
