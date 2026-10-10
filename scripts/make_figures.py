"""Build an experiment's figures from results/tables/ only (PLAN 2.4).

Usage:
    python scripts/make_figures.py --exp e01
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from pharmachecked.config import REPO_ROOT
from pharmachecked.evaluation.figstyle import INK_SECONDARY, apply_style, color_for, save_figure
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("make_figures")
TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"
CI = re.compile(r"^\s*([-\d.]+)\s*\[\s*([-\d.]+),\s*([-\d.]+)\s*\]")
# E1 tier of each model, for colour (identity) and legend order.
TIERS = {
    "cnn-small": "2 small CNN",
    "resnet50": "3 CNN",
    "efficientnetv2-s": "3 CNN",
    "convnext-tiny": "3 CNN",
    "vit-b16": "4 ViT",
    "swin-tiny": "4 ViT",
    "dinov2-vits14": "5 foundation",
    "dinov2-vitb14": "5 foundation",
    "clip-vitb16": "5 foundation",
}


def parse_ci(text: str) -> tuple[float, float, float]:
    """``"0.912 [0.880, 0.940]"`` -> (0.912, 0.880, 0.940)."""
    match = CI.match(str(text))
    if not match:
        raise ValueError(f"not an estimate with CI: {text!r}")
    return float(match[1]), float(match[2]), float(match[3])


def e01_auroc_vs_params(tables: Path, out: Path) -> list[Path]:
    """Pooled AUROC (95% CI) against parameter count (log x), coloured by tier, direct-labelled."""
    df = pd.read_csv(tables / "e01_summary.csv")
    df = df[df["params (M)"].notna()].copy()
    df["model"] = df["config"].str.split("/").str[0]
    df["variant"] = df["config"].str.split("/").str[1]
    df["tier"] = df["model"].map(TIERS).fillna("other")
    est = df["auroc (pooled)"].map(parse_ci)
    df["auroc"], df["low"], df["high"] = zip(*est, strict=True)
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, (tier, part) in enumerate(sorted(df.groupby("tier"))):
        ax.errorbar(
            part["params (M)"],
            part["auroc"],
            yerr=[part["auroc"] - part["low"], part["high"] - part["auroc"]],
            fmt="o",
            color=color_for(k),
            ecolor=color_for(k),
            elinewidth=1,
            capsize=0,
            label=f"tier {tier}",
        )
        for _, row in part.iterrows():
            label = row["model"] + ("" if row["variant"] == "default" else f" ({row['variant']})")
            ax.annotate(
                label,
                (row["params (M)"], row["auroc"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7,
                color=INK_SECONDARY,
            )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    low_x, high_x = df["params (M)"].min(), df["params (M)"].max()
    ax.set_xlim(low_x / 1.5, high_x * 1.5)
    ax.set_xlabel("Trainable + frozen parameters (millions, log scale)")
    ax.set_ylabel("Pooled OOF AUROC (95% CI)")
    ax.set_title("E1: AUROC vs model size")
    ax.legend(loc="lower right")
    return save_figure(fig, out, "e01_auroc_vs_params")


FIGURE_SETS: dict[str, list[Callable[[Path, Path], list[Path]]]] = {"e01": [e01_auroc_vs_params]}


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True, choices=sorted(FIGURE_SETS))
    args = parser.parse_args()
    configure_logging()
    for make in FIGURE_SETS[args.exp]:
        for path in make(TABLES, FIGURES):
            log.info("wrote %s", path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
