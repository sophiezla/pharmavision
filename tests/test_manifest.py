"""Manifest schema validation, duplicate collapse and CSV round trip."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from pharmachecked.data.manifest import (
    COLUMNS,
    collapse_duplicates,
    empty_partition_columns,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from pharmachecked.data.sources import RawFile


def _row(content: bytes, **overrides: object) -> dict[str, object]:
    sha = hashlib.sha256(content).hexdigest()
    row: dict[str, object] = {
        "image_id": sha[:16],
        "raw_path": f"data/raw/src/{sha[:6]}.png",
        "source_dataset": "kaggle_fake_real",
        "label": 1,
        "width": 10,
        "height": 20,
        "sha256": sha,
        "phash": "0123456789abcdef",
        "group_id": f"g_{sha[:16]}",
        "packaging_type": "",
        "split": "",
        "fold": pd.NA,
    }
    row.update(overrides)
    return row


def _frame(*rows: dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df["fold"] = df["fold"].astype("Int64")
    return df


def test_valid_manifest_roundtrip(tmp_path: Path) -> None:
    df = _frame(_row(b"a"), _row(b"b", label=0, split="dev", fold=3), _row(b"c", split="test"))
    write_manifest(df, tmp_path / "m.csv")
    back = read_manifest(tmp_path / "m.csv")
    assert list(back.columns) == list(COLUMNS)
    assert (
        back.sort_values("image_id")
        .reset_index(drop=True)
        .equals(df.sort_values("image_id").reset_index(drop=True))
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"label": 2}, "label"),
        ({"image_id": "XYZ"}, "image_id"),
        ({"image_id": "0" * 16}, "sha256"),
        ({"width": 0}, "positive"),
        ({"phash": "abc"}, "phash"),
        ({"group_id": ""}, "group_id"),
        ({"split": "train"}, "split"),
        ({"split": "dev"}, "fold"),
        ({"split": "dev", "fold": 5}, "fold"),
        ({"split": "test", "fold": 1}, "only dev"),
    ],
)
def test_schema_violations(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_manifest(_frame(_row(b"a", **overrides)))


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(_frame(_row(b"a"), _row(b"a")))


def test_wrong_columns_rejected() -> None:
    with pytest.raises(ValueError, match="columns"):
        validate_manifest(_frame(_row(b"a")).drop(columns="phash"))


def _raw(tmp_path: Path, rel: str, content: bytes, label: int, source: str = "s") -> RawFile:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return RawFile(path, source, "development", label, path.parent.name, True)


def test_collapse_keeps_one_row_per_content_and_drops_conflicts(tmp_path: Path) -> None:
    files = [
        _raw(tmp_path, "s/Fake/a.png", b"same", 1),
        _raw(tmp_path, "s/train/Fake/a.png", b"same", 1),
        _raw(tmp_path, "s/Real/b.jpg", b"other", 0),
        _raw(tmp_path, "s/Fake/c.png", b"clash", 1),
        _raw(tmp_path, "s/Real/c.jpg", b"clash", 0),
    ]
    out = collapse_duplicates(files, tmp_path)
    assert sorted(r["label"] for r in out.rows) == [0, 1]
    same = next(r for r in out.rows if r["sha256"] == hashlib.sha256(b"same").hexdigest())
    assert same["raw_path"] == "s/Fake/a.png"  # first path in sort order
    assert len(out.raw_paths) == 3
    assert out.conflicts == [["s/Fake/c.png", "s/Real/c.jpg"]]


def test_empty_partition_columns_produces_schema() -> None:
    sha = hashlib.sha256(b"a").hexdigest()
    base = pd.DataFrame(
        [
            {
                "image_id": sha[:16],
                "raw_path": "x",
                "source_dataset": "s",
                "label": 0,
                "sha256": sha,
                "width": 1,
                "height": 1,
            }
        ]
    )
    out = empty_partition_columns(base)
    assert tuple(out.columns) == COLUMNS
    validate_manifest(out)
