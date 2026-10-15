"""GPU kit pieces: image selection, snapshot provenance, batch-file line endings."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from pharmachecked.config import REPO_ROOT
from pharmachecked.utils.env import SNAPSHOT_FILE, snapshot_info

spec = importlib.util.spec_from_file_location(
    "make_gpu_kit", REPO_ROOT / "scripts" / "make_gpu_kit.py"
)
assert spec is not None and spec.loader is not None
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)


def test_kit_excludes_locked_test_and_includes_externals() -> None:
    dev = [{"image_id": "a", "split": "dev"}, {"image_id": "t", "split": "test"}]
    ext = [{"image_id": "e", "split": "external"}]
    assert kit.kit_image_ids(dev, ext) == ["a", "e"]


def test_snapshot_marks_modified_files_dirty(tmp_path: Path) -> None:
    files = {"src/a.py": b"x = 1\n", "configs/c.yaml": b"k: v\n"}
    for rel, data in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    (tmp_path / SNAPSHOT_FILE).write_bytes(kit.snapshot_json("abc123", files))
    assert snapshot_info(tmp_path / "src") == {"git_commit": "abc123", "git_dirty": False}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "new_run.txt").write_text("ok")  # new files do not count
    assert snapshot_info(tmp_path)["git_dirty"] is False
    (tmp_path / "src" / "a.py").write_bytes(b"x = 2\n")
    assert snapshot_info(tmp_path)["git_dirty"] is True


def test_no_snapshot_no_git(tmp_path: Path) -> None:
    assert snapshot_info(tmp_path) == {"git_commit": None, "git_dirty": None}


def test_snapshot_json_lists_hashes() -> None:
    data = json.loads(kit.snapshot_json("c0ffee", {"a.txt": b"hello"}))
    assert data["commit"] == "c0ffee"
    assert (
        data["files"]["a.txt"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_batch_files_get_crlf_and_torch_index_placeholder() -> None:
    setup = (REPO_ROOT / "kit" / "1_SETUP.bat").read_text(encoding="utf-8")
    assert "__TORCH_INDEX__" in setup
    converted = kit.crlf(setup)
    assert b"\r\n" in converted
    assert b"\n" not in converted.replace(b"\r\n", b"")


def _load_script(name: str) -> ModuleType:
    module_spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_results_already_in_the_snapshot_are_not_packed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kit_run = _load_script("kit_run")
    old_log = tmp_path / "results" / "job_logs" / "phase2_e01.log"
    old_log.parent.mkdir(parents=True)
    old_log.write_bytes(b"done\n")
    rel = old_log.relative_to(tmp_path).as_posix()
    (tmp_path / SNAPSHOT_FILE).write_bytes(kit.snapshot_json("abc123", {rel: b"done\n"}))
    new_log = old_log.with_name("phase3_e05.log")
    new_log.write_bytes(b"new\n")
    monkeypatch.setattr(kit_run, "REPO_ROOT", tmp_path)
    unchanged = kit_run.snapshot_unchanged()
    assert unchanged(old_log) and not unchanged(new_log)
    old_log.write_bytes(b"appended\n")  # a log that grew on the GPU PC is packed
    assert not unchanged(old_log)
