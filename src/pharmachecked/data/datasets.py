"""Datasets over the manifest. Images come only from development or external rows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from pharmachecked.data.splits import DevSplits

# Manifest partitions a dataset may read. The locked test set is not among them (Rule 2).
READABLE_SPLITS = frozenset({"dev", "external"})
META_COLUMNS = ("source_dataset", "packaging_type", "group_id", "fold")


class ManifestDataset(Dataset[dict[str, Any]]):
    """Images and labels for an explicit list of image ids.

    Each item is ``{"image", "label", "image_id"}`` plus metadata columns when
    ``return_meta`` is set. ``image`` is the transformed tensor, or the PIL image when
    ``transform`` is None.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_ids: Sequence[str],
        images_dir: Path,
        transform: Callable[[Image.Image], Any] | None = None,
        return_meta: bool = False,
    ) -> None:
        """Select ``image_ids`` from ``manifest``.

        Raises:
            KeyError: If an id is not in the manifest.
            PermissionError: If an id belongs to a partition other than dev or external.
        """
        table = manifest.set_index("image_id")
        missing = [i for i in image_ids if i not in table.index]
        if missing:
            raise KeyError(f"{len(missing)} ids not in the manifest, e.g. {missing[0]}")
        rows = table.loc[list(image_ids)]
        forbidden = ~rows["split"].isin(READABLE_SPLITS)
        if forbidden.any():
            raise PermissionError(
                f"{int(forbidden.sum())} requested images are outside the dev/external partitions"
            )
        self.image_ids = tuple(image_ids)
        self.labels = tuple(int(v) for v in rows["label"])
        self._meta = rows[list(META_COLUMNS)].to_dict("records")
        self.images_dir = images_dir
        self.transform = transform
        self.return_meta = return_meta

    @classmethod
    def from_split(
        cls,
        manifest: pd.DataFrame,
        splits: DevSplits,
        fold: int,
        part: str,
        images_dir: Path,
        transform: Callable[[Image.Image], Any] | None = None,
        return_meta: bool = False,
    ) -> ManifestDataset:
        """Dataset for one partition (``train``/``val``/``cal``/``heldout``) of one fold."""
        return cls(manifest, splits.ids(fold, part), images_dir, transform, return_meta)

    @classmethod
    def external(
        cls,
        manifest: pd.DataFrame,
        images_dir: Path,
        transform: Callable[[Image.Image], Any] | None = None,
        sources: Sequence[str] | None = None,
        return_meta: bool = False,
    ) -> ManifestDataset:
        """Dataset over the external manifest, optionally restricted to some sources."""
        rows = manifest[manifest["split"] == "external"]
        if sources is not None:
            rows = rows[rows["source_dataset"].isin(sources)]
        return cls(manifest, list(rows["image_id"]), images_dir, transform, return_meta)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        with Image.open(self.images_dir / f"{image_id}.png") as im:
            image = im.convert("RGB")
        item: dict[str, Any] = {
            "image": self.transform(image) if self.transform else image,
            "label": self.labels[index],
            "image_id": image_id,
        }
        if self.return_meta:
            meta = dict(self._meta[index])
            fold = meta.pop("fold")
            item.update({k: "" if pd.isna(v) else str(v) for k, v in meta.items()})
            item["fold"] = -1 if pd.isna(fold) else int(fold)  # -1: external (no fold)
        return item
