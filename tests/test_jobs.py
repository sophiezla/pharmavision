"""Job lists: parsing, finished-run detection and skipping."""

from pathlib import Path

from omegaconf import OmegaConf

from pharmachecked.jobs import compose_job, find_complete, read_jobs, run_jobs
from pharmachecked.utils.run_dir import COMPLETE_FILES, create_run_dir


def test_read_jobs_ignores_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "jobs.txt"
    path.write_text("# header\n\nexperiment=noop fold=1  # trailing\nexperiment=noop\n")
    assert read_jobs(path) == [["experiment=noop", "fold=1"], ["experiment=noop"]]


def _job(tmp_path: Path) -> list[str]:
    return ["experiment=noop", f"paths.runs_dir={(tmp_path / 'runs').as_posix()}"]


def test_finished_run_is_found_and_skipped(tmp_path: Path) -> None:
    cfg = compose_job(_job(tmp_path))
    assert find_complete(cfg) is None
    run = create_run_dir(
        Path(cfg.paths.runs_dir),
        cfg,
        {"git_dirty": False},
        exp=cfg.exp,
        model=cfg.model.name,
        variant=cfg.variant,
        fold=cfg.fold,
        seed=cfg.seed,
    )
    assert find_complete(cfg) is None  # incomplete folders are not finished
    for name in COMPLETE_FILES:
        run.file(name).touch()
    assert find_complete(cfg) == run.root
    other = compose_job([*_job(tmp_path), "seed=1"])
    assert find_complete(other) is None

    outcomes = run_jobs([_job(tmp_path)], tmp_path / "log.txt", python="python-that-does-not-exist")
    assert [o.status for o in outcomes] == ["skipped"]
    assert "skipped" in (tmp_path / "log.txt").read_text()


def test_composed_config_matches_saved_format(tmp_path: Path) -> None:
    cfg = compose_job(_job(tmp_path))
    assert OmegaConf.to_yaml(cfg, resolve=True).startswith("exp: noop")
