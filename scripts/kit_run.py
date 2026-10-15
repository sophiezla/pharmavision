"""GPU-kit run (2_RUN.bat): benchmark once, run the kit's work, pack the results.

The kit's work is either a job list (``KIT.json: job_file``) or a pipeline command
(``KIT.json: entry``, e.g. the Phase 2 pipeline). Keeps Windows awake while running, skips
finished runs, resumes interrupted ones, and always packs whatever is finished into
``RESULTS_<name>_<timestamp>.zip`` in the kit folder.

Usage (done by 2_RUN.bat):
    python scripts/kit_run.py --kit <kit folder>
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pharmachecked.config import REPO_ROOT
from pharmachecked.jobs import read_jobs, run_jobs
from pharmachecked.utils.bundle import pack
from pharmachecked.utils.env import SNAPSHOT_FILE, git_info
from pharmachecked.utils.io import read_json, sha256_file
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("kit_run")
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
RESULTS = REPO_ROOT / "results"


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while jobs run (no effect elsewhere)."""
    if os.name == "nt":
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]


def banner(text: str) -> None:
    """Print a framed message for the person watching the window."""
    sys.stdout.write("\n" + "=" * 60 + "\n " + text.replace("\n", "\n ") + "\n" + "=" * 60 + "\n")
    sys.stdout.flush()


def benchmark() -> None:
    """Measure the GPU once; output goes to a log that travels back with the results."""
    bench_log = RESULTS / "job_logs" / "bench_gpu-pc.log"
    bench_log.parent.mkdir(parents=True, exist_ok=True)
    banner("Measuring the GPU speed (a few minutes)...")
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


def do_work(kit: dict) -> bool:
    """Run the kit's job list or pipeline; True when everything finished."""
    if "entry" in kit:
        banner(f"Running {' '.join(kit['entry'])}.\nYou can leave this window open.")
        result = subprocess.run([sys.executable, *kit["entry"]], cwd=REPO_ROOT, check=False)
        return result.returncode == 0
    jobs = read_jobs(REPO_ROOT / kit["job_file"])
    banner(f"Running {len(jobs)} jobs from {kit['job_file']}.\nYou can leave this window open.")
    log_file = RESULTS / "job_logs" / f"{Path(kit['job_file']).stem}.log"
    return all(o.status != "failed" for o in run_jobs(jobs, log_file))


def snapshot_unchanged() -> Callable[[Path], bool]:
    """Predicate: is this file in the kit's source snapshot, byte-identical?

    Results committed before the kit was built travel inside its snapshot; packing them again
    only duplicates them (and renames job logs). Without a snapshot nothing is skipped.
    """
    root = next((p for p in (REPO_ROOT, *REPO_ROOT.parents) if (p / SNAPSHOT_FILE).is_file()), None)
    files = read_json(root / SNAPSHOT_FILE)["files"] if root is not None else {}

    def unchanged(path: Path) -> bool:
        if root is None:
            return False
        digest = files.get(path.resolve().relative_to(root.resolve()).as_posix())
        return digest is not None and sha256_file(path) == digest

    return unchanged


def pack_results(kit: dict, kit_dir: Path) -> Path:
    """Pack new run folders, benchmark tables, logs, pipeline decisions and Optuna studies."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    old = snapshot_unchanged()
    runs = sorted(
        p
        for p in (RESULTS / "runs").glob("*")
        if p.is_dir() and not all(old(f) for f in p.rglob("*") if f.is_file())
    )
    tables = sorted(p for p in (RESULTS / "tables").glob("bench_*.md") if not old(p))
    logs = {
        f"{kit['name']}_{stamp}_{p.name}": p
        for p in (RESULTS / "job_logs").glob("*.log")
        if not old(p)
    }
    setup = RESULTS / "kit_setup.json"
    if setup.exists():
        logs[f"{kit['name']}_{stamp}_kit_setup.json"] = setup
    extra = {
        top: sorted(p for p in (RESULTS / top).glob("*") if p.is_file() and not old(p))
        for top in ("pipeline", "optuna")
    }
    out = kit_dir / f"RESULTS_{kit['name']}_{stamp}.zip"
    pack(
        runs,
        tables,
        out,
        created_by=f"gpu-kit {kit['name']} {kit['commit']}",
        log_files=logs,
        extra_files=extra,
    )
    return out


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
        if kit.get("bench", True) and not (RESULTS / "tables" / "bench_gpu-pc.md").exists():
            benchmark()
        finished = do_work(kit)
    finally:
        keep_awake(False)
    out = pack_results(kit, args.kit)
    if not finished:
        banner(
            "Not everything finished.\n"
            f"Results so far: {out.name}\n"
            "Double-click 2_RUN.bat again to continue, or bring the zip back."
        )
        return 1
    banner(f"ALL WORK FINISHED.\nCopy this file back to the laptop:\n{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
