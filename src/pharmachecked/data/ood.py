"""OOD reference sets as manifest-like tables (PLAN 1.9; used by E11).

``data/ood/{near,far}/manifest.csv`` (committed) lists the images in
``data/ood/{near,far}/images/{image_id}.png`` (local only). The tables returned here carry the
columns ``ManifestDataset`` needs, with ``split = "ood"``, ``label = -1`` and ``fold = -1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OOD_SETS = ("near", "far")
OOD_LABEL = -1
SPLIT_SEED = 42  # ADR-026: the select/eval halves


def load_ood_set(data_dir: Path, name: str) -> tuple[pd.DataFrame, Path]:
    """Manifest-like table and images folder of OOD set ``name`` (``near`` or ``far``).

    Raises:
        ValueError: For an unknown set.
    """
    if name not in OOD_SETS:
        raise ValueError(f"unknown OOD set {name!r}; known: {OOD_SETS}")
    folder = Path(data_dir) / "ood" / name
    table = pd.read_csv(folder / "manifest.csv", dtype={"image_id": str})
    table = table.assign(
        label=OOD_LABEL,
        split="ood",
        fold=-1,
        group_id=table["image_id"],
        packaging_type="ood",
    )
    return table, folder / "images"


def select_eval_split(table: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """``select`` / ``eval`` per image: 50/50 within each (source_dataset, source_class).

    Within a stratum of odd size the extra image goes to ``eval``. Deterministic for a given
    table and seed, independent of row order.
    """
    rng = np.random.default_rng(seed)
    half = pd.Series("eval", index=table["image_id"].to_numpy(), name="ood_half")
    for _, part in table.sort_values("image_id").groupby(["source_dataset", "source_class"]):
        ids = part["image_id"].to_numpy()
        chosen = rng.permutation(ids)[: len(ids) // 2]
        half.loc[chosen] = "select"
    return half
