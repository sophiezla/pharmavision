"""Raw data sources: which files belong to which source, their role and their label."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLES = ("development", "external", "excluded")


@dataclass(frozen=True)
class RawFile:
    """One raw image file with its source, role and label (None when unknown)."""

    path: Path
    source: str
    role: str
    label: int | None
    class_folder: str  # path of the parent folder relative to the source root ("." if top level)
    included: bool  # False when a membership file excludes it


@dataclass(frozen=True)
class SourceSpec:
    """Configuration of one source (see configs/data/sources.yaml)."""

    name: str
    role: str
    label: int | None = None
    folder_labels: dict[str, int] | None = None
    membership: Path | None = None
    membership_key: str = "image_stem"
    include_column: str = "include"

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any], repo_root: Path) -> SourceSpec:
        """Build a spec from one entry of ``sources.yaml``.

        Raises:
            ValueError: On an unknown role or a development/external source without labels.
        """
        role = cfg["role"]
        if role not in ROLES:
            raise ValueError(f"{name}: unknown role {role!r}")
        folder_labels = cfg.get("folder_labels")
        label = cfg.get("label")
        if role != "excluded" and label is None and not folder_labels:
            raise ValueError(f"{name}: a {role} source needs `label` or `folder_labels`")
        membership = cfg.get("membership")
        return cls(
            name=name,
            role=role,
            label=label,
            folder_labels={k.lower(): int(v) for k, v in folder_labels.items()}
            if folder_labels
            else None,
            membership=repo_root / membership if membership else None,
            membership_key=cfg.get("membership_key", "image_stem"),
            include_column=cfg.get("include_column", "include"),
        )

    def label_for(self, relative: Path) -> int | None:
        """Return the label of a file given its path relative to the source root."""
        if self.folder_labels:
            for part in relative.parts[:-1]:
                if part.lower() in self.folder_labels:
                    return self.folder_labels[part.lower()]
            return None
        return self.label

    def members(self) -> dict[str, bool] | None:
        """Return ``{file stem: included}`` from the membership file, or None if there is none."""
        if self.membership is None:
            return None
        with self.membership.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return {
            row[self.membership_key]: row[self.include_column].strip().lower() == "true"
            for row in rows
        }


def load_specs(cfg: dict[str, Any], repo_root: Path) -> list[SourceSpec]:
    """Build every source spec from the ``sources`` mapping of ``sources.yaml``."""
    return [SourceSpec.from_config(n, c, repo_root) for n, c in cfg["sources"].items()]


def list_raw_files(
    raw_dir: Path, specs: list[SourceSpec], extensions: tuple[str, ...]
) -> list[RawFile]:
    """List every image file of every configured source, sorted by path.

    Files with a membership entry of false are listed with ``included=False``; image files that
    a membership file does not mention are treated as excluded too.

    Args:
        raw_dir: ``data/raw``.
        specs: Source specs.
        extensions: Lower-case image extensions including the dot.

    Returns:
        Raw files in deterministic order.

    Raises:
        FileNotFoundError: If a configured source folder is missing.
    """
    out: list[RawFile] = []
    for spec in specs:
        root = raw_dir / spec.name
        if not root.is_dir():
            raise FileNotFoundError(f"source folder missing: {root}")
        members = spec.members()
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix.lower() not in extensions:
                continue
            relative = path.relative_to(root)
            included = True if members is None else members.get(path.stem, False)
            out.append(
                RawFile(
                    path=path,
                    source=spec.name,
                    role=spec.role,
                    label=spec.label_for(relative),
                    class_folder=relative.parent.as_posix(),
                    included=included,
                )
            )
    return out
