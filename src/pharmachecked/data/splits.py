"""Grouped, stratified partitions: locked test set, 5 outer folds, inner train/val/cal.

The split file (``splits/v{N}.json``) stores the locked-test image ids, but they are only
reachable through :func:`load_locked_test`, which only ``scripts/final_eval.py`` may call
(Rule 2, enforced by ``tests/test_test_lock.py``). Everything else uses :func:`load_dev_splits`,
whose result contains no test ids at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from pharmachecked.utils.io import atomic_write_text

PARTS = ("train", "val", "cal", "heldout")
INNER_FOLDS = 8  # 6/8 train = 75%, 1/8 val = 12.5%, 1/8 cal = 12.5%


@dataclass(frozen=True)
class SplitSettings:
    """Protocol constants (CLAUDE.md section 5)."""

    seed: int = 42
    test_fraction: float = 0.15
    n_folds: int = 5


def pool_fingerprint(manifest: pd.DataFrame) -> str:
    """SHA-256 over sorted ``image_id,label,group_id`` lines: identifies the pool a split is for."""
    lines = sorted(
        f"{i},{lab},{g}"
        for i, lab, g in zip(
            manifest["image_id"], manifest["label"], manifest["group_id"], strict=True
        )
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _group_labels(df: pd.DataFrame) -> pd.Series:
    """Label per group for stratification: majority label (ties -> counterfeit)."""
    return (df.groupby("group_id")["label"].mean() >= 0.5).astype(int)


def carve_test(df: pd.DataFrame, settings: SplitSettings) -> list[str]:
    """Pick ``test_fraction`` of the groups, stratified by group label; return their image ids."""
    labels = _group_labels(df)
    _, test_groups = train_test_split(
        labels.index.to_numpy(),
        test_size=settings.test_fraction,
        stratify=labels.to_numpy(),
        random_state=settings.seed,
    )
    return sorted(df.loc[df["group_id"].isin(set(test_groups)), "image_id"])


def _sgkf(df: pd.DataFrame, n_splits: int, seed: int) -> list[np.ndarray]:
    """Return the test-index arrays of a shuffled StratifiedGroupKFold over ``df``."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [test for _, test in sgkf.split(df, df["label"], df["group_id"])]


def make_splits(
    manifest: pd.DataFrame,
    settings: SplitSettings,
    version: int,
    previous_test: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a split specification.

    Args:
        manifest: Development-pool manifest (``image_id, label, group_id`` used).
        settings: Seed, test fraction and fold count.
        version: Split version number.
        previous_test: Locked-test ids of the previous version. When given, test membership
            is carried over unchanged (restricted to ids still in the pool) instead of being
            re-carved (ADR-002).
        extra: Additional metadata stored in the file.

    Returns:
        JSON-serializable dict with ``locked_test`` and ``folds``.
    """
    df = manifest[["image_id", "label", "group_id"]].sort_values("image_id").reset_index(drop=True)
    if previous_test is None:
        test_ids = carve_test(df, settings)
    else:
        test_ids = sorted(set(previous_test) & set(df["image_id"]))
        test_groups = set(df.loc[df["image_id"].isin(test_ids), "group_id"])
        leaked = df.loc[df["group_id"].isin(test_groups) & ~df["image_id"].isin(test_ids)]
        if len(leaked):
            raise ValueError(
                f"{len(leaked)} new pool images share a group with the locked test set"
            )
    dev = df[~df["image_id"].isin(set(test_ids))].reset_index(drop=True)
    folds = []
    for k, heldout_idx in enumerate(_sgkf(dev, settings.n_folds, settings.seed)):
        heldout = dev.iloc[heldout_idx]
        rest = dev.drop(index=heldout.index).reset_index(drop=True)
        inner = _sgkf(rest, INNER_FOLDS, settings.seed + k)
        val, cal = rest.iloc[inner[0]], rest.iloc[inner[1]]
        train = rest.drop(index=np.concatenate([inner[0], inner[1]]))
        folds.append(
            {
                "fold": k,
                "train": sorted(train["image_id"]),
                "val": sorted(val["image_id"]),
                "cal": sorted(cal["image_id"]),
                "heldout": sorted(heldout["image_id"]),
            }
        )
    return {
        "version": version,
        "seed": settings.seed,
        "test_fraction": settings.test_fraction,
        "n_folds": settings.n_folds,
        "inner_fractions": {"train": 0.75, "val": 0.125, "cal": 0.125},
        "pool_fingerprint": pool_fingerprint(manifest),
        **(extra or {}),
        "locked_test": sorted(test_ids),
        "folds": folds,
    }


def save_splits(spec: dict[str, Any], path: Path) -> None:
    """Write a split file (stable key order and formatting)."""
    atomic_write_text(path, json.dumps(spec, indent=1, sort_keys=False) + "\n")


@dataclass(frozen=True)
class DevSplits:
    """Development partitions of one split version. Holds no locked-test ids."""

    version: int
    pool_fingerprint: str
    folds: tuple[dict[str, tuple[str, ...]], ...]

    def ids(self, fold: int, part: str) -> tuple[str, ...]:
        """Image ids of one partition (``train``, ``val``, ``cal`` or ``heldout``) of a fold."""
        if part not in PARTS:
            raise KeyError(f"unknown partition {part!r}; use one of {PARTS}")
        return self.folds[fold][part]

    def all_ids(self) -> frozenset[str]:
        """Every development image id."""
        return frozenset(i for f in self.folds for i in f["heldout"])


def load_dev_splits(path: Path) -> DevSplits:
    """Load the development partitions of a split file (locked-test ids are discarded)."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    folds = tuple({p: tuple(f[p]) for p in PARTS} for f in spec["folds"])
    return DevSplits(spec["version"], spec["pool_fingerprint"], folds)


def load_locked_test(path: Path, *, acknowledge_final: bool) -> tuple[str, ...]:
    """Return the locked-test image ids. Only ``scripts/final_eval.py`` may call this (Rule 2).

    Args:
        path: Split file.
        acknowledge_final: Must be True; mirrors final_eval's command-line flag.

    Raises:
        PermissionError: If ``acknowledge_final`` is not True.
    """
    if acknowledge_final is not True:
        raise PermissionError("the locked test set is opened only by scripts/final_eval.py")
    return tuple(json.loads(path.read_text(encoding="utf-8"))["locked_test"])


def apply_to_manifest(manifest: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Return the manifest with ``split`` (``test``/``dev``) and ``fold`` (held-out fold) filled."""
    out = manifest.copy()
    test = set(spec["locked_test"])
    fold_of = {i: f["fold"] for f in spec["folds"] for i in f["heldout"]}
    out["split"] = ["test" if i in test else "dev" for i in out["image_id"]]
    out["fold"] = pd.array([fold_of.get(i, pd.NA) for i in out["image_id"]], dtype="Int64")
    missing = set(out["image_id"]) - test - set(fold_of)
    if missing:
        raise ValueError(f"{len(missing)} manifest images are in no partition")
    return out


# Phase 4 partitions (docs/experiments/E06.md, E07.md, E08.md). They are built at run time from
# the development pool (never the locked test set) and use the same inner 75/12.5/12.5 split.


def inner_split(df: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    """Grouped, stratified train / val / cal (75 / 12.5 / 12.5) of ``df``, as in the folds.

    Args:
        df: Rows with ``image_id, label, group_id``.
        seed: StratifiedGroupKFold seed.

    Returns:
        ``{"train", "val", "cal"}`` -> sorted image ids.
    """
    df = df[["image_id", "label", "group_id"]].sort_values("image_id").reset_index(drop=True)
    inner = _sgkf(df, INNER_FOLDS, seed)
    train = df.drop(index=np.concatenate([inner[0], inner[1]]))
    return {
        "train": sorted(train["image_id"]),
        "val": sorted(df.iloc[inner[0]]["image_id"]),
        "cal": sorted(df.iloc[inner[1]]["image_id"]),
    }


def subsample_groups(df: pd.DataFrame, fraction: float, seed: int) -> list[str]:
    """Keep ``fraction`` of the groups of ``df``, stratified by group label (E6).

    Per class, ``ceil(fraction * groups)`` groups (at least one) are drawn without replacement
    with a generator seeded by ``seed``; every image of a kept group is kept.

    Raises:
        ValueError: If ``fraction`` is not in (0, 1].
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    labels = _group_labels(df).sort_index()
    rng = np.random.default_rng(seed)
    kept: list[str] = []
    for label in sorted(labels.unique()):
        groups = labels.index[labels == label].to_numpy()
        n = max(1, int(np.ceil(fraction * len(groups))))
        kept += list(rng.choice(groups, size=n, replace=False))
    return sorted(df.loc[df["group_id"].isin(set(kept)), "image_id"])


def leave_type_out(df: pd.DataFrame, packaging_type: str, seed: int) -> dict[str, list[str]]:
    """Held-out = every image of ``packaging_type``; train/val/cal from groups without it (E8).

    A near-duplicate group that contains any image of the held-out type is removed from the
    training pool as a whole, so no group crosses train and test; its images of other types are
    used nowhere.

    Raises:
        ValueError: If no image has ``packaging_type``.
    """
    is_type = df["packaging_type"] == packaging_type
    if not is_type.any():
        raise ValueError(f"no images of packaging type {packaging_type!r}")
    blocked = set(df.loc[is_type, "group_id"])
    parts = inner_split(df[~df["group_id"].isin(blocked)], seed)
    return {**parts, "heldout": sorted(df.loc[is_type, "image_id"])}
