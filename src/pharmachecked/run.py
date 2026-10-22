"""Hydra entry point: ``python -m pharmachecked.run experiment=<name> [overrides]``.

Composes the config, creates the run folder (``config.yaml`` + ``env.json``), seeds every
generator, mirrors the run into MLflow and dispatches to the experiment's task.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.utils.env import capture_env
from pharmachecked.utils.logging import configure_logging, get_logger
from pharmachecked.utils.run_dir import RunDir, create_run_dir, find_resumable
from pharmachecked.utils.seed import seed_everything
from pharmachecked.utils.tracking import track

log = get_logger(__name__)


def noop(cfg: DictConfig, run: RunDir) -> None:
    """Do nothing beyond what :func:`execute` already wrote; tests the entry point."""
    log.info("noop task: run folder %s", run.root)


def _model_task(name: str) -> Callable[[DictConfig, RunDir], None]:
    """Task that imports the model-task module lazily (it pulls in torch and data loading)."""

    def task(cfg: DictConfig, run: RunDir) -> None:
        from pharmachecked.training.task import TASK_FUNCTIONS

        TASK_FUNCTIONS[name](cfg, run)

    return task


def _one_class(cfg: DictConfig, run: RunDir) -> None:
    """E8 one-class methods (PatchCore, prototype verifier); imported lazily."""
    from pharmachecked.training.one_class import run_one_class

    run_one_class(cfg, run)


def _lazy(module: str, function: str) -> Callable[[DictConfig, RunDir], None]:
    """Task imported on first use (Phases 6-8 tasks pull in heavy modules)."""

    def task(cfg: DictConfig, run: RunDir) -> None:
        import importlib

        getattr(importlib.import_module(module), function)(cfg, run)

    return task


TASKS: dict[str, Callable[[DictConfig, RunDir], None]] = {
    "noop": noop,
    "train": _model_task("train"),
    "handcrafted": _model_task("handcrafted"),
    "clip_zeroshot": _model_task("clip_zeroshot"),
    "one_class": _one_class,
    "cbm": _lazy("pharmachecked.training.cbm", "run_cbm"),  # E13
    "adv_train": _lazy("pharmachecked.adversarial.race", "run_adv_train"),  # E15 defences
    "race": _lazy("pharmachecked.adversarial.race", "run_race"),  # E15 arms race
    "distill": _lazy("pharmachecked.training.distill", "run_distill"),  # E16
}


def execute(cfg: DictConfig) -> RunDir:
    """Create the run folder for ``cfg`` and run its task.

    Args:
        cfg: Composed config (root config plus overrides).

    Returns:
        The run folder.

    Raises:
        KeyError: If ``cfg.task`` is not a registered task.
    """
    if cfg.task not in TASKS:
        raise KeyError(f"unknown task {cfg.task!r}; registered: {sorted(TASKS)}")
    env = capture_env(REPO_ROOT)
    fields = {
        "exp": cfg.exp,
        "model": cfg.model.name,
        "variant": cfg.variant,
        "fold": cfg.fold,
        "seed": cfg.seed,
    }
    resumable = None
    if cfg.get("train") is not None and cfg.train.get("resume"):
        resumable = find_resumable(Path(cfg.paths.runs_dir), cfg, **fields)
    if resumable is not None:
        run = resumable
        log.info("resuming interrupted run %s", run.run_id)
    else:
        run = create_run_dir(Path(cfg.paths.runs_dir), cfg, env, **fields)
    log.info("run %s (git %s, dirty=%s)", run.run_id, env["git_commit"], env["git_dirty"])
    seed_everything(cfg.seed)
    with track(cfg, run, env):
        TASKS[cfg.task](cfg, run)
    return run


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra main; see the module docstring."""
    configure_logging()
    execute(cfg)


if __name__ == "__main__":
    main()
