"""`python -m pharmachecked.run experiment=noop` writes a run folder and an MLflow run."""

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.utils.io import read_json, read_yaml


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("paths", ["local", "remote"])
def test_config_composes(paths: str) -> None:
    cfg = _compose(["experiment=noop", f"paths={paths}"])
    assert cfg.exp == "noop"
    assert cfg.task == "noop"
    assert cfg.model.name == "none"
    assert Path(cfg.paths.root) == REPO_ROOT
    assert Path(cfg.paths.runs_dir) == REPO_ROOT / "results" / "runs"
    assert cfg.tracker.tracking_uri.startswith("sqlite:///")


def test_experiment_is_mandatory() -> None:
    with pytest.raises(Exception, match="experiment"):
        _compose([])


@pytest.mark.parametrize("relative", [False, True])
def test_noop_cli_writes_run_folder(tmp_path: Path, relative: bool) -> None:
    runs = tmp_path / "runs"
    mlruns = tmp_path / "mlruns"
    # Relative overrides are resolved against the working directory (tmp_path here).
    runs_arg = "runs" if relative else runs.as_posix()
    mlruns_arg = "mlruns" if relative else mlruns.as_posix()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pharmachecked.run",
            "experiment=noop",
            f"paths.runs_dir={runs_arg}",
            f"paths.mlruns_dir={mlruns_arg}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (run,) = list(runs.iterdir())
    assert run.name.startswith("noop_none_default_f0_s0_")
    cfg = read_yaml(run / "config.yaml")
    assert cfg.exp == "noop"
    assert cfg.paths.runs_dir == runs_arg
    env = read_json(run / "env.json")
    assert {"git_commit", "git_dirty", "python", "packages"} <= env.keys()

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{mlruns.as_posix()}/mlflow.db")
    found = mlflow.search_runs(experiment_names=["noop"], output_format="list")
    assert [r.info.run_name for r in found] == [run.name]
