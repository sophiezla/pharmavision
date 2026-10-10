"""Split integrity: no group overlap, class balance, reproducibility, fixed test membership."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import (
    PARTS,
    SplitSettings,
    load_dev_splits,
    make_splits,
    pool_fingerprint,
)

MANIFEST = REPO_ROOT / "data" / "processed" / "manifest_v1.csv"
SPLIT_V1 = REPO_ROOT / "splits" / "v1.json"
TOLERANCE_PP = 3.0

needs_v1 = pytest.mark.skipif(not SPLIT_V1.exists(), reason="splits/v1.json not built yet")


def _synthetic(n_groups: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = int(rng.random() < 0.37)
        for k in range(int(rng.integers(1, 4))):
            rows.append(
                {"image_id": f"{g:06d}{k:02d}{'0' * 8}", "label": label, "group_id": f"g{g}"}
            )
    return pd.DataFrame(rows)


def _check(manifest: pd.DataFrame, spec: dict) -> None:
    group = dict(zip(manifest["image_id"], manifest["group_id"], strict=True))
    label = dict(zip(manifest["image_id"], manifest["label"], strict=True))
    pool = set(manifest["image_id"])
    test = set(spec["locked_test"])
    share = 100 * manifest["label"].mean()

    def groups(ids: set[str]) -> set[str]:
        return {group[i] for i in ids}

    heldouts = [set(f["heldout"]) for f in spec["folds"]]
    assert set().union(*heldouts) | test == pool
    assert sum(map(len, heldouts)) + len(test) == len(pool)
    for f, heldout in zip(spec["folds"], heldouts, strict=True):
        parts = {p: set(f[p]) for p in PARTS}
        assert set().union(*parts.values()) == pool - test
        assert sum(map(len, parts.values())) == len(pool - test)
        for name in parts:
            assert not groups(parts[name]) & groups(test), (
                f"fold {f['fold']} {name} shares a group with test"
            )
            for other in parts:
                if name < other:
                    assert not groups(parts[name]) & groups(parts[other])
        for name, ids in parts.items():
            part_share = 100 * np.mean([label[i] for i in ids])
            assert abs(part_share - share) <= TOLERANCE_PP, (f["fold"], name, part_share)
        assert parts["heldout"] == heldout
    assert abs(100 * np.mean([label[i] for i in test]) - share) <= TOLERANCE_PP


def test_synthetic_partitions_are_disjoint_by_group_and_balanced() -> None:
    df = _synthetic()
    _check(df, make_splits(df, SplitSettings(), version=1))


def test_reproducible_from_seed() -> None:
    df = _synthetic()
    assert make_splits(df, SplitSettings(), 1) == make_splits(
        df.sample(frac=1, random_state=3), SplitSettings(), 1
    )


def test_different_seed_changes_test_set() -> None:
    df = _synthetic()
    a = make_splits(df, SplitSettings(seed=42), 1)["locked_test"]
    b = make_splits(df, SplitSettings(seed=7), 1)["locked_test"]
    assert a != b


def test_later_version_keeps_test_membership() -> None:
    df = _synthetic()
    v1 = make_splits(df, SplitSettings(), 1)
    test = set(v1["locked_test"])
    dev_ids = sorted(set(df["image_id"]) - test)
    removed = set(dev_ids[:20]) | {sorted(test)[0]}  # label-audit removals
    v2 = make_splits(df[~df["image_id"].isin(removed)], SplitSettings(), 2, v1["locked_test"])
    assert set(v2["locked_test"]) == test - removed
    _check(df[~df["image_id"].isin(removed)], v2)


def test_new_image_in_a_test_group_is_refused() -> None:
    df = _synthetic()
    v1 = make_splits(df, SplitSettings(), 1)
    test_group = df.loc[df["image_id"] == v1["locked_test"][0], "group_id"].iloc[0]
    extra = pd.DataFrame([{"image_id": "f" * 16, "label": 0, "group_id": test_group}])
    with pytest.raises(ValueError, match="share a group"):
        make_splits(pd.concat([df, extra]), SplitSettings(), 2, v1["locked_test"])


@needs_v1
def test_committed_v1_is_valid_and_reproducible() -> None:
    manifest = read_manifest(MANIFEST)
    spec = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    _check(manifest, spec)
    assert spec["pool_fingerprint"] == pool_fingerprint(manifest)
    rebuilt = make_splits(
        manifest,
        SplitSettings(),
        1,
        extra={k: spec[k] for k in ("manifest", "phash_threshold")},
    )
    assert rebuilt == spec


@needs_v1
def test_manifest_partition_columns_match_v1() -> None:
    manifest = read_manifest(MANIFEST).set_index("image_id")
    spec = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    assert set(manifest.index[manifest["split"] == "test"]) == set(spec["locked_test"])
    for f in spec["folds"]:
        assert set(manifest.index[manifest["fold"] == f["fold"]]) == set(f["heldout"])


@needs_v1
def test_dev_splits_hold_no_test_ids() -> None:
    dev = load_dev_splits(SPLIT_V1)
    spec = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    assert not dev.all_ids() & set(spec["locked_test"])
    assert not hasattr(dev, "locked_test")
    with pytest.raises(KeyError):
        dev.ids(0, "test")


def test_split_file_round_trip(tmp_path: Path) -> None:
    from pharmachecked.data.splits import save_splits

    spec = make_splits(_synthetic(), SplitSettings(), 1)
    save_splits(spec, tmp_path / "v1.json")
    dev = load_dev_splits(tmp_path / "v1.json")
    assert dev.ids(2, "val") == tuple(spec["folds"][2]["val"])
