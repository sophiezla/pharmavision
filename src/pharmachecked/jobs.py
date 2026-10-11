"""Job lists: one run per line, as Hydra overrides; run in order, skip finished, resume interrupted.

A job file has one job per line: Hydra overrides such as
``experiment=e01_baselines model=resnet50 variant=ft fold=0 seed=0``. Blank lines and ``#``
comments are ignored. A job is skipped when a complete run folder with an identical resolved
config exists; an interrupted run resumes in its own folder (trainer resume). Failures are logged
and the list continues.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from pharmachecked.config import CONFIG_DIR
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import missing_files, slug

log = get_logger(__name__)


def read_jobs(path: Path) -> list[list[str]]:
    """Parse a job file into override lists."""
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            jobs.append(line.split())
    return jobs


def compose_job(overrides: Sequence[str]) -> DictConfig:
    """The resolved config a job would run with."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def find_complete(cfg: DictConfig) -> Path | None:
    """A complete run folder whose saved config equals ``cfg`` (resolved), if any."""
    runs = Path(cfg.paths.runs_dir)
    if not runs.is_dir():
        return None
    prefix = f"{slug(cfg.exp)}_{slug(cfg.model.name)}_{slug(cfg.variant)}_f{cfg.fold}_s{cfg.seed}_"
    wanted = OmegaConf.to_yaml(cfg, resolve=True)
    for folder in sorted(runs.glob(prefix + "*")):
        saved = folder / "config.yaml"
        if (
            not missing_files(folder)
            and saved.is_file()
            and saved.read_text(encoding="utf-8") == wanted
        ):
            return folder
    return None


@dataclass
class JobOutcome:
    """What happened to one job."""

    overrides: list[str]
    status: str  # done | skipped | failed
    seconds: float
    detail: str = ""


def run_jobs(
    jobs: list[list[str]], log_file: Path, python: str = sys.executable
) -> list[JobOutcome]:
    """Run jobs in order with ``python -m pharmachecked.run``; append progress to ``log_file``."""
    outcomes = []
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for k, overrides in enumerate(jobs, start=1):
        label = " ".join(overrides)
        done = find_complete(compose_job(overrides))
        if done is not None:
            outcomes.append(JobOutcome(overrides, "skipped", 0.0, done.name))
            _append(log_file, f"[{k}/{len(jobs)}] skipped (already complete: {done.name}) {label}")
            continue
        _append(log_file, f"[{k}/{len(jobs)}] start {label}")
        start = time.perf_counter()
        result = subprocess.run([python, "-m", "pharmachecked.run", *overrides], check=False)
        seconds = time.perf_counter() - start
        status = "done" if result.returncode == 0 else "failed"
        outcomes.append(JobOutcome(overrides, status, seconds, f"exit {result.returncode}"))
        _append(log_file, f"[{k}/{len(jobs)}] {status} in {seconds / 60:.1f} min {label}")
    return outcomes


def _append(path: Path, line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {line}\n")
    log.info(line)
