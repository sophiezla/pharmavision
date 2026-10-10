"""MLflow logging of runs. The run folder stays the source of truth; MLflow mirrors it."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import RunDir

log = get_logger(__name__)
# Config groups that describe the machine, not the experiment; not logged as params.
EXCLUDED_GROUPS = ("paths", "tracker")


def disable_mlflow_network_extras() -> None:
    """Turn off MLflow usage telemetry and console hints before mlflow is imported.

    Telemetry sends usage events over the network and flushes them at exit, which can block a
    run on a slow connection; nothing in this project needs it.
    """
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted keys (lists are kept as values).

    Args:
        data: Nested mapping.
        prefix: Key prefix used during recursion.

    Returns:
        Flat dict ``{"a.b.c": value}``.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def run_params(cfg: DictConfig) -> dict[str, Any]:
    """Return the experiment parameters of a resolved config as flat MLflow params.

    Args:
        cfg: Composed config.

    Returns:
        Dotted-key params without the machine-specific groups.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    kept = {k: v for k, v in container.items() if k not in EXCLUDED_GROUPS}
    return flatten(kept)


@contextmanager
def track(cfg: DictConfig, run: RunDir, env: Mapping[str, Any]) -> Iterator[None]:
    """Mirror a run into MLflow for the duration of the block (no-op if disabled).

    Args:
        cfg: Composed config with a ``tracker`` group.
        run: The run folder.
        env: Captured environment (git commit and dirty flag become tags).

    Yields:
        None; the MLflow run is active inside the block and marked FAILED on exceptions.
    """
    if not cfg.tracker.enabled:
        yield
        return
    disable_mlflow_network_extras()
    import mlflow

    Path(cfg.paths.mlruns_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.tracker.tracking_uri)
    name = cfg.tracker.experiment_name
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        artifacts = (Path(cfg.tracker.artifact_location) / name).resolve()
        experiment_id = mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    else:
        experiment_id = experiment.experiment_id
    tags = {
        "run_id": run.run_id,
        "run_dir": run.root.as_posix(),
        "git_commit": str(env.get("git_commit")),
        "git_dirty": str(env.get("git_dirty")),
    }
    with mlflow.start_run(experiment_id=experiment_id, run_name=run.run_id, tags=tags):
        mlflow.log_params(run_params(cfg))
        log.info("MLflow run %s in experiment %r", run.run_id, name)
        yield
