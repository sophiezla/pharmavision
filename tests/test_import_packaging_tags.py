"""Earlier packaging tags are matched to image ids by file hash."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.normalize import image_id_from_bytes

spec = importlib.util.spec_from_file_location(
    "import_packaging_tags", REPO_ROOT / "scripts" / "import_packaging_tags.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_match_by_file_hash(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.png").write_bytes(b"aaa")
    (tmp_path / "raw" / "copy_of_a.png").write_bytes(b"aaa")
    (tmp_path / "meta").mkdir()
    tags = pd.DataFrame(
        {
            "orig_relpath": ["../raw/a.png", "../raw/copy_of_a.png"],
            "modality": ["blister", "blister"],
        }
    )
    assert module.match_tags(tags, tmp_path / "meta") == {image_id_from_bytes(b"aaa"): "blister"}


def test_conflicting_tags_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"aaa")
    (tmp_path / "b.png").write_bytes(b"aaa")
    tags = pd.DataFrame({"orig_relpath": ["a.png", "b.png"], "modality": ["blister", "other"]})
    with pytest.raises(ValueError, match="conflicting"):
        module.match_tags(tags, tmp_path)


def test_missing_file_rejected(tmp_path: Path) -> None:
    tags = pd.DataFrame({"orig_relpath": ["nope.png"], "modality": ["other"]})
    with pytest.raises(FileNotFoundError):
        module.match_tags(tags, tmp_path)
