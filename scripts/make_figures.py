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


def _arm_label(model: str, variant: str) -> str:
    return model + ("" if variant in ("default", "ft") else f" ({variant})")


def e01_auroc_vs_params(tables: Path, out: Path) -> list[Path]:
    """(a) Pooled AUROC vs parameter count (log x) by tier; (b) every arm ranked, with 95% CI.

    Panel (a) labels only points standing apart from the cluster near AUROC 1.0; panel (b)
    names every arm, so no label overlaps. Zero-shot CLIP (no trained parameters) appears in (b).
    """
    df = pd.read_csv(tables / "e01_summary.csv")
    df["model"] = df["config"].str.split("/").str[0]
    df["variant"] = df["config"].str.split("/").str[1]
    df["tier"] = df["model"].map(TIERS).fillna("1 handcrafted")
    df.loc[df["variant"] == "zeroshot", "tier"] = "6 zero-shot"
    est = df["auroc (pooled)"].map(parse_ci)
    df["auroc"], df["low"], df["high"] = zip(*est, strict=True)
    tiers = sorted(df["tier"].unique())
    colour = {t: color_for(k) for k, t in enumerate(tiers)}
    apply_style()
    fig, (ax, forest) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.0, 1.1]}
    )

    sized = df[df["params (M)"].notna()]
    for tier, part in sized.groupby("tier"):
        ax.errorbar(
            part["params (M)"],
            part["auroc"],
            yerr=[part["auroc"] - part["low"], part["high"] - part["auroc"]],
            fmt="o",
            color=colour[tier],
            elinewidth=1,
            label=f"tier {tier}",
        )
        for _, row in part[part["auroc"] < 0.99].iterrows():  # the cluster is named in (b)
            ax.annotate(
                _arm_label(row["model"], row["variant"]),
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
    ax.set_xlim(sized["params (M)"].min() / 1.5, sized["params (M)"].max() * 1.5)
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Pooled OOF AUROC (95% CI)")
    ax.set_title("(a) AUROC vs model size")

    ranked = df.sort_values("auroc").reset_index(drop=True)
    for y, row in ranked.iterrows():
        forest.errorbar(
            row["auroc"],
            y,
            xerr=[[row["auroc"] - row["low"]], [row["high"] - row["auroc"]]],
            fmt="o",
            color=colour[row["tier"]],
            elinewidth=1,
            markersize=5,
        )
    forest.set_yticks(range(len(ranked)))
    forest.set_yticklabels(
        [_arm_label(m, v) for m, v in zip(ranked["model"], ranked["variant"], strict=True)]
    )
    forest.tick_params(axis="y", labelsize=7, colors=INK_SECONDARY)
    forest.grid(axis="y", visible=False)
    forest.set_xlabel("Pooled OOF AUROC (95% CI)")
    forest.set_title("(b) All arms, ranked")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour[t], label=f"tier {t}")
        for t in tiers
    ]
    forest.legend(handles=handles, loc="upper left")  # the left of the axis is empty
    fig.suptitle("E1: baseline ladder (development pool, out-of-fold)", fontsize=11, weight="bold")
    fig.tight_layout()
    return save_figure(fig, out, "e01_auroc_vs_params")


E2_STRATEGIES = ("lp", "lora8", "lora16", "last2", "ft")  # few -> many trainable parameters


def e02_auroc_by_strategy(tables: Path, out: Path) -> list[Path]:
    """Pooled AUROC and log-loss (95% CI) per adaptation strategy, one colour per backbone."""
    df = pd.read_csv(tables / "e02_summary.csv")
    df["model"] = df["config"].str.split("/").str[0]
    df["variant"] = df["config"].str.split("/").str[1]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    panels = (("auroc (pooled)", "Pooled OOF AUROC"), ("log_loss (pooled)", "Pooled OOF log-loss"))
    xs = {s: k for k, s in enumerate(E2_STRATEGIES)}
    for ax, (column, ylabel) in zip(axes, panels, strict=True):
        for k, (model, part) in enumerate(sorted(df.groupby("model"))):
            part = part[part["variant"].isin(xs)].sort_values("variant", key=lambda v: v.map(xs))
            est, low, high = zip(*part[column].map(parse_ci), strict=True)
            x = [xs[v] + (k - 0.5) * 0.18 for v in part["variant"]]  # side by side per strategy
            ax.errorbar(
                x,
                est,
                yerr=[
                    [e - lo for e, lo in zip(est, low, strict=True)],
                    [h - e for e, h in zip(est, high, strict=True)],
                ],
                fmt="o",
                color=color_for(k),
                elinewidth=1,
                label=model,
            )
        ax.set_xticks(range(len(E2_STRATEGIES)))
        ax.set_xticklabels(
            ["linear\nprobe", "LoRA r8", "LoRA r16", "last 2\nblocks", "full\nfine-tune"]
        )
        ax.set_ylabel(f"{ylabel} (95% CI)")
        ax.grid(axis="x", visible=False)
    axes[0].set_title("E2: AUROC by strategy")
    axes[1].set_title("E2: log-loss by strategy")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out, "e02_auroc_by_strategy")


E4_BRANCHES = ("global", "local", "frequency")
E4_TYPES = ("blister", "packaging", "both", "other", "all")


def e04_gate_weights_by_packaging(tables: Path, out: Path) -> list[Path]:
    """Mean fusion gate weight (± sd over runs) per branch, grouped by packaging type."""
    df = pd.read_csv(tables / "e04_gate_weights.csv")
    types = [t for t in E4_TYPES if t in set(df["packaging_type"])]
    images = df.groupby("packaging_type")["images"].first()
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.8 / len(E4_BRANCHES)
    for k, branch in enumerate(E4_BRANCHES):
        part = df[df["branch"] == branch].set_index("packaging_type").reindex(types)
        x = [i + (k - (len(E4_BRANCHES) - 1) / 2) * width for i in range(len(types))]
        ax.bar(x, part["mean"], width, yerr=part["sd"], color=color_for(k), label=branch)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{t}\n(n={images[t]})" for t in types])
    ax.set_ylabel("Mean gate weight (± sd over runs)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title("E4 full model: fusion gate weight by packaging type (held-out folds)")
    ax.legend(loc="upper right", title="branch")
    fig.tight_layout()
    return save_figure(fig, out, "e04_gate_weights_by_packaging")


FRAC = re.compile(r"-frac(\d+)$")


def e06_learning_curves(tables: Path, out: Path) -> list[Path]:
    """Pooled AUROC vs training fraction (log x), line + 95% CI band per model."""
    df = pd.read_csv(tables / "e06_summary.csv")
    df["model"] = df["config"].str.split("/").str[0]
    variant = df["config"].str.split("/").str[1]
    df["fraction"] = [int(m[1]) if (m := FRAC.search(v)) else 100 for v in variant]
    est, low, high = zip(*df["auroc (pooled)"].map(parse_ci), strict=True)
    df["est"], df["low"], df["high"] = est, low, high
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, (model, part) in enumerate(sorted(df.groupby("model"))):
        part = part.sort_values("fraction")
        ax.plot(part["fraction"], part["est"], marker="o", color=color_for(k), label=model)
        ax.fill_between(part["fraction"], part["low"], part["high"], color=color_for(k), alpha=0.18)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10, 25, 50, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Share of each outer training portion (log scale)")
    ax.set_ylabel("Pooled OOF AUROC (95% CI)")
    ax.set_title("E6: data efficiency")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out, "e06_learning_curves")


def e08_seen_vs_unseen(tables: Path, out: Path) -> list[Path]:
    """Seen vs unseen AUROC (95% CI) per method, one panel per held-out packaging type."""
    df = pd.read_csv(tables / "e08_open_set.csv")
    types = list(dict.fromkeys(df["held-out type"]))
    apply_style()
    fig, axes = plt.subplots(1, len(types), figsize=(5.2 * len(types), 3.8), squeeze=False)
    for ax, t in zip(axes[0], types, strict=True):
        part = df[df["held-out type"] == t].reset_index(drop=True)
        for col, (label, offset, k) in {
            "seen AUROC (5-fold OOF)": ("seen", -0.15, 0),
            "unseen AUROC": ("unseen", 0.15, 1),
        }.items():
            for i, text in enumerate(part[col]):
                if text == "-" or pd.isna(text):
                    continue
                e, lo, hi = parse_ci(text)
                ax.errorbar(
                    i + offset,
                    e,
                    yerr=[[e - lo], [hi - e]],
                    fmt="o",
                    color=color_for(k),
                    elinewidth=1,
                    label=label,
                )
        ax.set_xticks(range(len(part)))
        ax.set_xticklabels(part["method"], rotation=30, ha="right")
        ax.set_title(f"held-out type: {t}")
        ax.set_ylabel("AUROC (95% CI)")
        ax.grid(axis="x", visible=False)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(unique.values(), unique.keys(), loc="lower left")
    fig.suptitle("E8: seen vs unseen packaging type")
    fig.tight_layout()
    return save_figure(fig, out, "e08_seen_vs_unseen")


def e09_severity_curves(tables: Path, out: Path) -> list[Path]:
    """AUROC vs severity (0 = clean) per model, one panel per corruption."""
    df = pd.read_csv(tables / "e09_severity.csv")
    clean = df[df["corruption"] == "clean"].set_index("config")["auroc"]
    names = sorted(set(df["corruption"]) - {"clean"})
    apply_style()
    ncol = 3
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.0 * nrow), sharey=True, squeeze=False)
    configs = sorted(df["config"].unique())
    for ax, name in zip(axes.flat, names, strict=False):
        part = df[df["corruption"] == name]
        for k, config in enumerate(configs):
            sub = part[part["config"] == config].sort_values("severity")
            ax.plot(
                [0, *sub["severity"]],
                [clean[config], *sub["auroc"]],
                marker="o",
                markersize=3,
                color=color_for(k),
                label=config,
            )
        ax.axhline(0.85, color=INK_SECONDARY, linestyle="--", linewidth=0.8)
        ax.set_title(name.replace("_", " "))
        ax.set_xticks(range(6))
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Pooled AUROC")
    for ax in axes[-1, :]:
        ax.set_xlabel("Severity (0 = clean)")
    axes.flat[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("E9: AUROC under corruption (dashed: 0.85 quality limit, ADR-023)")
    fig.tight_layout()
    return save_figure(fig, out, "e09_severity_curves")


FIGURE_SETS: dict[str, list[Callable[[Path, Path], list[Path]]]] = {
    "e01": [e01_auroc_vs_params],
    "e02": [e02_auroc_by_strategy],
    "e04": [e04_gate_weights_by_packaging],
    "e06": [e06_learning_curves],
    "e08": [e08_seen_vs_unseen],
    "e09": [e09_severity_curves],
}


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
