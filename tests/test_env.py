"""env.json captures git state, versions and device; the dirty flag tracks tracked files only."""

import shutil
import subprocess
from pathlib import Path

import pytest

from pharmachecked.utils.env import capture_env, git_info

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_repo(repo: Path) -> None:
    info = git_info(repo)
    assert len(info["git_commit"]) == 40
    assert info["git_dirty"] is False


def test_untracked_run_folder_is_not_dirty(repo: Path) -> None:
    run = repo / "results" / "runs" / "e01_m_v_f0_s0_20261009-000000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text("{}\n")
    assert git_info(repo)["git_dirty"] is False


def test_untracked_source_file_is_dirty(repo: Path) -> None:
    (repo / "new_module.py").write_text("x = 1\n")
    assert git_info(repo)["git_dirty"] is True


def test_ignored_file_is_not_dirty(repo: Path) -> None:
    (repo / ".gitignore").write_text("checkpoints/\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "ignore")
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints" / "w.pt").write_text("w\n")
    assert git_info(repo)["git_dirty"] is False


def test_modified_tracked_file_is_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("two\n")
    assert git_info(repo)["git_dirty"] is True


def test_not_a_repo(tmp_path: Path) -> None:
    assert git_info(tmp_path) == {"git_commit": None, "git_dirty": None}


def test_capture_env_keys(repo: Path) -> None:
    env = capture_env(repo)
    for key in (
        "git_commit",
        "git_dirty",
        "python",
        "platform",
        "packages",
        "cuda_version",
        "gpu_name",
        "cuda_available",
    ):
        assert key in env
    assert env["packages"]["torch"] is not None
