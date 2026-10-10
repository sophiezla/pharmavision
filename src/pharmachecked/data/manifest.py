"""Image manifests: one row per unique image, with provenance, label, hashes and partition."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pharmachecked.data.sources import RawFile
from pharmachecked.utils.io import sha256_file
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)

COLUMNS = (
    "image_id",
    "raw_path",
    "source_dataset",
    "label",
    "width",
    "height",
    "sha256",
    "phash",
    "group_id",
    "packaging_type",
    "split",
    "fold",
)
SPLITS = ("", "dev", "test", "external")
N_FOLDS = 5
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Collapsed:
    """Unique images after exact-duplicate collapse."""

    rows: list[dict[str, object]]  # one per unique sha256, raw_path = first path in sort order
    raw_paths: list[tuple[str, str]]  # (sha256, raw_path) for every raw file kept
    conflicts: list[list[str]]  # raw paths of dropped groups whose copies disagree on label


def collapse_duplicates(files: Iterable[RawFile], repo_root: Path) -> Collapsed:
    """Collapse byte-identical raw files to one row each (ADR-001).

    A group of identical files whose labels or sources disagree is dropped entirely and
    reported in ``conflicts``.

    Args:
        files: Included raw files of development or external sources.
        repo_root: Repository root (raw paths are stored relative to it).

    Returns:
        A :class:`Collapsed` result.
    """
    by_sha: dict[str, list[RawFile]] = defaultdict(list)
    for f in files:
        by_sha[sha256_file(f.path)].append(f)
    rows: list[dict[str, object]] = []
    raw_paths: list[tuple[str, str]] = []
    conflicts: list[list[str]] = []
    for sha, group in sorted(by_sha.items()):
        paths = sorted(f.path.relative_to(repo_root).as_posix() for f in group)
        if len({f.label for f in group}) > 1 or len({f.source for f in group}) > 1:
            conflicts.append(paths)
            log.warning("dropping %d identical files with conflicting labels/sources", len(paths))
            continue
        first = group[0]
        rows.append(
            {
                "image_id": sha[:16],
                "raw_path": paths[0],
                "source_dataset": first.source,
                "label": first.label,
                "sha256": sha,
            }
        )
        raw_paths.extend((sha, p) for p in paths)
    return Collapsed(rows, raw_paths, conflicts)


def empty_partition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns filled by later steps (phash, group, packaging type, split, fold)."""
    out = df.copy()
    for col in ("phash", "group_id", "packaging_type", "split"):
        if col not in out:
            out[col] = ""
    if "fold" not in out:
        out["fold"] = pd.array([pd.NA] * len(out), dtype="Int64")
    return out[list(COLUMNS)]


def validate_manifest(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` describing every schema violation in a manifest.

    Checks column order, unique hex ids consistent with the SHA-256, labels in {0, 1},
    positive sizes, 64-bit pHashes and group ids when present, known split values, and folds
    (0-4 for ``dev`` rows, empty otherwise).
    """
    errors: list[str] = []
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"columns {list(df.columns)} != {list(COLUMNS)}")
    if df["image_id"].duplicated().any():
        errors.append("duplicate image_id")
    if not df["image_id"].map(lambda s: bool(_HEX16.match(str(s)))).all():
        errors.append("image_id must be 16 lower-case hex characters")
    if not df["sha256"].map(lambda s: bool(_HEX64.match(str(s)))).all():
        errors.append("sha256 must be 64 lower-case hex characters")
    elif not (df["sha256"].str[:16] == df["image_id"]).all():
        errors.append("image_id must equal sha256[:16]")
    if not df["label"].isin([0, 1]).all():
        errors.append("label must be 0 or 1")
    if not ((df["width"] > 0) & (df["height"] > 0)).all():
        errors.append("width and height must be positive")
    phash = df["phash"].fillna("")
    if not phash.map(lambda s: s == "" or bool(_HEX16.match(s))).all():
        errors.append("phash must be empty or 16 hex characters")
    group = df["group_id"].fillna("")
    if ((phash != "") & (group == "")).any():
        errors.append("rows with a phash need a group_id")
    split = df["split"].fillna("")
    if not split.isin(SPLITS).all():
        errors.append(f"split must be one of {SPLITS}")
    fold = df["fold"]
    dev = split == "dev"
    dev_folds = fold[dev]
    # between() yields <NA> for missing folds and all() would skip them; check presence first.
    if dev.any() and (dev_folds.isna().any() or not dev_folds.between(0, N_FOLDS - 1).all()):
        errors.append("dev rows need fold 0-4")
    if fold[~dev].notna().any():
        errors.append("only dev rows may have a fold")
    if errors:
        raise ValueError("; ".join(errors))


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Validate and write a manifest CSV (LF line endings, sorted by image_id)."""
    validate_manifest(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("image_id").to_csv(path, index=False, lineterminator="\n")


def read_manifest(path: Path) -> pd.DataFrame:
    """Read and validate a manifest CSV with the column types the code expects."""
    df = pd.read_csv(
        path,
        dtype={
            "image_id": str,
            "raw_path": str,
            "source_dataset": str,
            "label": int,
            "width": int,
            "height": int,
            "sha256": str,
            "phash": str,
            "group_id": str,
            "packaging_type": str,
            "split": str,
            "fold": "Int64",
        },
        keep_default_na=False,
        na_values={"fold": [""]},
    )
    validate_manifest(df)
    return df
