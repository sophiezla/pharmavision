"""Shared figure style (PLAN 2.4): one validated palette, recessive axes, PNG (300 dpi) + PDF.

Categorical colours are assigned in this fixed order and never cycled; the palette passed the
colour-vision checks (lightness band, chroma, CVD and normal-vision separation) on the light
surface. Three slots fall below 3:1 contrast with the surface, so figures direct-label their
marks and every figure has a table behind it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    """Set matplotlib defaults for every project figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
        }
    )


def color_for(index: int) -> str:
    """Categorical colour for the ``index``-th entity (fixed order; never cycled)."""
    if index >= len(CATEGORICAL):
        raise ValueError("more than 8 categories: fold into 'Other' or use small multiples")
    return CATEGORICAL[index]


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    """Save ``name``.png (300 dpi) and ``name``.pdf in ``out_dir``; close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths
