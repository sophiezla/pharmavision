"""Figures are built from tables only and saved as PNG + PDF."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from pharmachecked.config import REPO_ROOT
from pharmachecked.evaluation.figstyle import CATEGORICAL, color_for

spec = importlib.util.spec_from_file_location(
    "make_figures", REPO_ROOT / "scripts" / "make_figures.py"
)
assert spec is not None and spec.loader is not None
make_figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_figures)


def test_parse_ci() -> None:
    assert make_figures.parse_ci("0.912 [0.880, 0.940]") == (0.912, 0.88, 0.94)
    with pytest.raises(ValueError):
        make_figures.parse_ci("0.9 ± 0.1")


def test_colours_are_fixed_order_and_never_cycled() -> None:
    assert color_for(0) == CATEGORICAL[0]
    with pytest.raises(ValueError):
        color_for(len(CATEGORICAL))


def test_e01_figure_from_table(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "config": [
                "resnet50/default",
                "vit-b16/default",
                "dinov2-vits14/lp",
                "handcrafted-lr/default",
            ],
            "auroc (pooled)": [
                "0.90 [0.86, 0.93]",
                "0.93 [0.90, 0.95]",
                "0.88 [0.84, 0.91]",
                "0.8 [0.7, 0.9]",
            ],
            "params (M)": [23.5, 85.8, 21.6, float("nan")],
        }
    ).to_csv(tmp_path / "e01_summary.csv", index=False)
    paths = make_figures.e01_auroc_vs_params(tmp_path, tmp_path / "figs")
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert paths[0].stat().st_size < 1_000_000  # pre-commit large-file limit
