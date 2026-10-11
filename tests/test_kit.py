"""GPU kit pieces: image selection, snapshot provenance, batch-file line endings."""

import importlib.util
import json
from pathlib import Path

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
