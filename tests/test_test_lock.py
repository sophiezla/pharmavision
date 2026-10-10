"""Rule 2: only scripts/final_eval.py may reach the locked test set, and it refuses by default."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.splits import load_locked_test

FINAL_EVAL = REPO_ROOT / "scripts" / "final_eval.py"
CODE_DIRS = ("src", "scripts", "app")

# Files allowed to name the locked test set at all:
#   splits.py defines it; make_splits.py carries membership to new versions (ADR-002);
#   final_eval.py is the one consumer.
LOCKED_TEST_ALLOWED = {
    "src/pharmachecked/data/splits.py",
    "scripts/make_splits.py",
    "scripts/final_eval.py",
}
# Files allowed to contain the literal "test" (the partition name): the split and manifest
# definitions, final_eval, make_splits, and the task runner's `test` target.
TEST_LITERAL_ALLOWED = LOCKED_TEST_ALLOWED | {
    "src/pharmachecked/data/manifest.py",
    "scripts/tasks.py",
}
LOCKED_TEST = re.compile(r"locked_test")
TEST_LITERAL = re.compile(r"""["']test["']""")


def _code_files() -> list[Path]:
    return sorted(p for d in CODE_DIRS for p in (REPO_ROOT / d).rglob("*.py"))


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


def test_only_allowed_files_name_the_locked_test_set() -> None:
    offenders = [
        _rel(p)
        for p in _code_files()
        if _rel(p) not in LOCKED_TEST_ALLOWED and LOCKED_TEST.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_only_allowed_files_select_the_test_partition() -> None:
    offenders = [
        _rel(p)
        for p in _code_files()
        if _rel(p) not in TEST_LITERAL_ALLOWED
        and TEST_LITERAL.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_load_locked_test_called_only_by_final_eval() -> None:
    callers = [
        _rel(p)
        for p in _code_files()
        if "load_locked_test(" in p.read_text(encoding="utf-8")
        and _rel(p) != "src/pharmachecked/data/splits.py"
    ]
    assert set(callers) <= {"scripts/final_eval.py"}


def test_load_locked_test_requires_acknowledgement(tmp_path: Path) -> None:
    (tmp_path / "v1.json").write_text('{"locked_test": ["a"]}')
    with pytest.raises(PermissionError):
        load_locked_test(tmp_path / "v1.json", acknowledge_final=False)


def _final_eval(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FINAL_EVAL), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_final_eval_refuses_without_flag(tmp_path: Path) -> None:
    result = _final_eval("--results-dir", str(tmp_path))
    assert result.returncode == 2
    assert "refused" in result.stderr
    assert not (tmp_path / "LOCK").exists()


def test_final_eval_refuses_when_locked(tmp_path: Path) -> None:
    (tmp_path / "LOCK").write_text("done\n")
    result = _final_eval("--i-understand-this-is-final", "--results-dir", str(tmp_path))
    assert result.returncode == 2
    assert "already evaluated" in result.stderr


def test_final_eval_stub_does_not_open_or_lock(tmp_path: Path) -> None:
    result = _final_eval("--i-understand-this-is-final", "--results-dir", str(tmp_path))
    assert result.returncode == 3
    assert not (tmp_path / "LOCK").exists()


def test_repository_is_not_locked() -> None:
    assert not (REPO_ROOT / "results" / "LOCK").exists()
