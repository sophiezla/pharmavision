"""Phase 4 partitions: E6 subsampling, E7 whole pool, E8 leave one packaging type out."""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import (
    inner_split,
    leave_type_out,
    load_dev_splits,
    subsample_groups,
)
from pharmachecked.data.toy import ToyData
from pharmachecked.training.task import partitions


def _pool(n_groups: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic pool: groups of 1-3 images, one label per group, three packaging types."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = int(rng.random() < 0.4)
        for k in range(int(rng.integers(1, 4))):
            ptype = ("blister", "packaging", "other")[int(rng.integers(0, 3))]
            rows.append(
                {
                    "image_id": f"g{g:03d}i{k}",
                    "group_id": f"g{g:03d}",
                    "label": label,
                    "packaging_type": ptype,
                }
            )
    return pd.DataFrame(rows)


def _groups(df: pd.DataFrame, ids: list[str]) -> set[str]:
    return set(df.set_index("image_id").loc[ids, "group_id"])


def test_inner_split_is_grouped_disjoint_and_deterministic() -> None:
    df = _pool()
    parts = inner_split(df, 42)
    ids = [set(parts[p]) for p in ("train", "val", "cal")]
    assert sum(len(s) for s in ids) == len(df) and set.union(*ids) == set(df["image_id"])
    groups = [_groups(df, parts[p]) for p in ("train", "val", "cal")]
    assert (
        not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    )
    assert 0.6 < len(parts["train"]) / len(df) < 0.9
    assert inner_split(df.sample(frac=1, random_state=1), 42) == parts


def test_subsample_groups_keeps_whole_groups_per_class() -> None:
    df = _pool()
    kept = subsample_groups(df, 0.25, seed=0)
    k = df[df["image_id"].isin(kept)]
    assert set(k["image_id"]) == set(df[df["group_id"].isin(set(k["group_id"]))]["image_id"])
    for label in (0, 1):
        n_all = df.loc[df["label"] == label, "group_id"].nunique()
        assert k.loc[k["label"] == label, "group_id"].nunique() == int(np.ceil(0.25 * n_all))
    assert subsample_groups(df, 0.25, seed=0) == kept
    assert subsample_groups(df, 0.25, seed=1) != kept
    assert subsample_groups(df, 1.0, seed=0) == sorted(df["image_id"])
    tiny = subsample_groups(df, 0.001, seed=0)
    assert set(df.set_index("image_id").loc[tiny, "label"]) == {0, 1}  # >= 1 group per class
    with pytest.raises(ValueError):
        subsample_groups(df, 0.0, seed=0)


def test_leave_type_out_never_shares_a_group_with_the_held_out_type() -> None:
    df = _pool()
    parts = leave_type_out(df, "blister", 42)
    assert set(parts["heldout"]) == set(df.loc[df["packaging_type"] == "blister", "image_id"])
    train_side = parts["train"] + parts["val"] + parts["cal"]
    assert not _groups(df, train_side) & _groups(df, parts["heldout"])
    assert set(df.set_index("image_id").loc[train_side, "packaging_type"]) <= {"packaging", "other"}
    with pytest.raises(ValueError):
        leave_type_out(df, "bottle", 42)


def _cfg(**data: object) -> OmegaConf:
    base = {
        "fold": 0,
        "seed": 0,
        "data": {
            "protocol": "kfold",
            "heldout_type": None,
            "train_fraction": None,
            "max_train_images": None,
            "max_external_per_source": None,
        },
    }
    base["data"].update(data)
    return OmegaConf.create(base)


def _toy(toy: ToyData) -> tuple:
    manifest = read_manifest(toy.manifest)
    types = ["blister", "packaging"]
    manifest["packaging_type"] = [types[i % 2] for i in range(len(manifest))]
    return manifest, read_manifest(toy.external_manifest), load_dev_splits(toy.splits)


def test_partitions_protocols_stay_in_the_development_pool(toy: ToyData) -> None:
    manifest, external, splits = _toy(toy)
    dev = splits.all_ids()
    kfold = partitions(_cfg(), splits, external, manifest)
    assert kfold.train == list(splits.ids(0, "train"))
    assert kfold.heldout == list(splits.ids(0, "heldout"))
    whole = partitions(_cfg(protocol="whole_pool"), splits, external, manifest)
    assert whole.heldout == [] and set(whole.train + whole.val) <= dev
    assert len(whole.train) > len(kfold.train)
    lto = partitions(
        _cfg(protocol="leave_type_out", heldout_type="blister"), splits, external, manifest
    )
    types = manifest.set_index("image_id")["packaging_type"]
    assert set(types.loc[lto.heldout]) == {"blister"} and set(lto.heldout) <= dev
    assert set(types.loc[lto.train + lto.val]) == {"packaging"}
    frac = partitions(_cfg(train_fraction=0.5), splits, external, manifest)
    assert set(frac.train) < set(kfold.train) and frac.val == kfold.val
    with pytest.raises(ValueError):
        partitions(_cfg(protocol="bootstrap"), splits, external, manifest)
