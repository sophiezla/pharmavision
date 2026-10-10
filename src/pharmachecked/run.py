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
from pharmachecked.utils.run_dir import RunDir, create_run_dir
from pharmachecked.utils.seed import seed_everything
from pharmachecked.utils.tracking import track

log = get_logger(__name__)


def noop(cfg: DictConfig, run: RunDir) -> None:
    """Do nothing beyond what :func:`execute` already wrote; tests the entry point."""
    log.info("noop task: run folder %s", run.root)


TASKS: dict[str, Callable[[DictConfig, RunDir], None]] = {"noop": noop}


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
    run = create_run_dir(
        Path(cfg.paths.runs_dir),
        cfg,
        env,
        exp=cfg.exp,
        model=cfg.model.name,
        variant=cfg.variant,
        fold=cfg.fold,
        seed=cfg.seed,
    )
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
