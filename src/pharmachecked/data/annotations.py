"""Annotation stores (CSV under data/annotations/): packaging type and concept labels."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from pharmachecked.utils.io import atomic_write_text

# Packaging types (ADR-007, amended 2026-10-10): blister packs, other packaging (cartons, boxes,
# bottles, labels), both in one image, images that do not show a medicine, and anything else.
PACKAGING_TYPES = ("blister", "packaging", "both", "not_a_drug", "other")
PACKAGING_COLUMNS = ("image_id", "clip_label", "clip_conf", "final_label", "annotator", "timestamp")
CONCEPTS = (
    "text_error",
    "color_mismatch",
    "print_blur",
    "logo_distortion",
    "layout_misalignment",
    "missing_security_feature",
    "seal_or_package_damage",
)
CONCEPT_VALUES = ("present", "absent", "unsure")
CONCEPT_COLUMNS = ("image_id", *CONCEPTS, "annotator", "timestamp")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(df.columns) != tuple(columns):
        raise ValueError(f"{path}: columns {list(df.columns)} != {list(columns)}")
    return df


def _write(df: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, df.to_csv(index=False, lineterminator="\n"))


class PackagingStore:
    """``packaging_type.csv``: CLIP pre-label and the human final label per image."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.df = _read(path, PACKAGING_COLUMNS).set_index("image_id", drop=False)

    @staticmethod
    def create(path: Path, prelabels: pd.DataFrame) -> PackagingStore:
        """Write a new store from ``image_id, clip_label, clip_conf`` rows; never overwrites."""
        if path.exists():
            raise FileExistsError(f"{path} exists; annotations are never overwritten")
        bad = set(prelabels["clip_label"]) - set(PACKAGING_TYPES)
        if bad:
            raise ValueError(f"unknown packaging types {bad}")
        df = prelabels[["image_id", "clip_label", "clip_conf"]].copy()
        df["clip_conf"] = df["clip_conf"].map(lambda v: f"{float(v):.4f}")
        for col in ("final_label", "annotator", "timestamp"):
            df[col] = ""
        _write(df.sort_values("image_id"), path)
        return PackagingStore(path)

    def ids(self) -> list[str]:
        """All image ids, sorted."""
        return list(self.df.index)

    def row(self, image_id: str) -> dict[str, str]:
        """One row as a dict."""
        return self.df.loc[image_id].to_dict()

    def refresh_clip(self, prelabels: pd.DataFrame) -> None:
        """Replace the CLIP columns (after a scheme change); human labels are left untouched."""
        bad = set(prelabels["clip_label"]) - set(PACKAGING_TYPES)
        if bad:
            raise ValueError(f"unknown packaging types {bad}")
        if set(prelabels["image_id"]) != set(self.df.index):
            raise ValueError("pre-labels must cover exactly the images in the store")
        new = prelabels.set_index("image_id")
        self.df["clip_label"] = new["clip_label"]
        self.df["clip_conf"] = new["clip_conf"].map(lambda v: f"{float(v):.4f}")
        _write(self.df.reset_index(drop=True), self.path)

    def set_label(self, image_id: str, label: str, annotator: str) -> None:
        """Record a final label and save immediately."""
        if label not in PACKAGING_TYPES:
            raise ValueError(f"unknown packaging type {label!r}")
        if not annotator.strip():
            raise ValueError("annotator name required")
        self.df.loc[image_id, ["final_label", "annotator", "timestamp"]] = [
            label,
            annotator.strip(),
            _now(),
        ]
        _write(self.df.reset_index(drop=True), self.path)

    def n_done(self) -> int:
        """Number of images with a final label."""
        return int((self.df["final_label"] != "").sum())


class ConceptStore:
    """``concepts.csv``: seven concept labels (present/absent/unsure) per subset image."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.df = _read(path, CONCEPT_COLUMNS).set_index("image_id", drop=False)

    @staticmethod
    def create(path: Path, image_ids: Sequence[str]) -> ConceptStore:
        """Write an empty store for the selected subset (refuses to overwrite)."""
        if path.exists():
            raise FileExistsError(f"{path} exists; annotations are never overwritten")
        df = pd.DataFrame({"image_id": sorted(image_ids)})
        for col in CONCEPT_COLUMNS[1:]:
            df[col] = ""
        _write(df, path)
        return ConceptStore(path)

    def ids(self) -> list[str]:
        """All image ids, sorted."""
        return list(self.df.index)

    def row(self, image_id: str) -> dict[str, str]:
        """One row as a dict."""
        return self.df.loc[image_id].to_dict()

    def set_concepts(self, image_id: str, values: dict[str, str], annotator: str) -> None:
        """Record all seven concepts for an image and save immediately."""
        if set(values) != set(CONCEPTS):
            raise ValueError(f"need exactly the concepts {CONCEPTS}")
        if not set(values.values()) <= set(CONCEPT_VALUES):
            raise ValueError(f"concept values must be in {CONCEPT_VALUES}")
        if not annotator.strip():
            raise ValueError("annotator name required")
        for concept, value in values.items():
            self.df.loc[image_id, concept] = value
        self.df.loc[image_id, ["annotator", "timestamp"]] = [annotator.strip(), _now()]
        _write(self.df.reset_index(drop=True), self.path)

    def n_done(self) -> int:
        """Number of images with all concepts filled."""
        return int((self.df[list(CONCEPTS)] != "").all(axis=1).sum())
