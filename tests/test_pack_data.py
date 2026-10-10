"""Data bundles never contain non-portable sources and verify on unpack."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.manifest import COLUMNS, write_manifest


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pack_data", REPO_ROOT / "scripts" / "pack_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pack_data = _load_script()


@pytest.fixture
def processed(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for content, source in ((b"k1", "kaggle"), (b"k2", "kaggle"), (b"e1", "split_e")):
        sha = hashlib.sha256(content).hexdigest()
        (images / f"{sha[:16]}.png").write_bytes(content)
        rows.append(
            {
                "image_id": sha[:16],
                "raw_path": "x",
                "source_dataset": source,
                "label": 1,
                "width": 1,
                "height": 1,
                "sha256": sha,
                "phash": "",
                "group_id": "",
                "packaging_type": "",
                "split": "",
                "fold": pd.NA,
            }
        )
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df["fold"] = df["fold"].astype("Int64")
    write_manifest(df, tmp_path / "manifest_v1.csv")
    return images, tmp_path / "manifest_v1.csv"


PORTABLE = {"kaggle": True, "split_e": False}


def test_non_portable_images_and_rows_are_left_out(
    tmp_path: Path, processed: tuple[Path, Path]
) -> None:
    images, manifest = processed
    out = tmp_path / "data.zip"
    assert pack_data.pack(images, [manifest], PORTABLE, out) == 2
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        packed_manifest = zf.read("manifests/manifest_v1.csv").decode()
    split_e_id = hashlib.sha256(b"e1").hexdigest()[:16]
    assert not any(split_e_id in n for n in names)
    assert "split_e" not in packed_manifest


def test_unknown_source_refused(tmp_path: Path, processed: tuple[Path, Path]) -> None:
    images, manifest = processed
    with pytest.raises(pack_data.DataBundleError, match="missing"):
        pack_data.pack(images, [manifest], {"kaggle": True}, tmp_path / "d.zip")


def test_roundtrip_and_tamper_detection(tmp_path: Path, processed: tuple[Path, Path]) -> None:
    images, manifest = processed
    out = tmp_path / "data.zip"
    pack_data.pack(images, [manifest], PORTABLE, out)
    dest = tmp_path / "gpu_pc" / "processed"
    assert pack_data.unpack(out, dest) == 2
    assert (dest / "manifest_v1.csv").is_file()
    assert len(list((dest / "images").iterdir())) == 2

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(bad, "w") as dst:
        index = json.loads(src.read(pack_data.INDEX))
        for name in src.namelist():
            data = src.read(name)
            dst.writestr(name, b"tampered" if name.startswith("images/") else data)
        assert index
    with pytest.raises(pack_data.DataBundleError, match="hash"):
        pack_data.unpack(bad, tmp_path / "elsewhere")
