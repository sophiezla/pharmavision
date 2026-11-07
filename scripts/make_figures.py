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
    return _severity_curves(
        pd.read_csv(tables / "e09_severity.csv"),
        out,
        "e09_severity_curves",
        "E9: AUROC under corruption (dashed: 0.85 quality limit, ADR-023)",
    )


def e09_extension_curves(tables: Path, out: Path) -> list[Path]:
    """As ``e09_severity_curves`` over severities 1-8; dotted line: start of the extension."""
    return _severity_curves(
        pd.read_csv(tables / "e09_ext_severity.csv"),
        out,
        "e09_ext_severity_curves",
        "E9 extension: severities 1-8 (dotted: ADR-024 levels start; dashed: 0.85 limit)",
        extension_from=6,
    )


def _severity_curves(
    df: pd.DataFrame, out: Path, stem: str, title: str, extension_from: int | None = None
) -> list[Path]:
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
                [*([0] if config in clean else []), *sub["severity"]],  # 0 = clean, if present
                [*([clean[config]] if config in clean else []), *sub["auroc"]],
                marker="o",
                markersize=3,
                color=color_for(k),
                label=config,
            )
        ax.axhline(0.85, color=INK_SECONDARY, linestyle="--", linewidth=0.8)
        if extension_from is not None:
            ax.axvline(extension_from - 0.5, color=INK_SECONDARY, linestyle=":", linewidth=0.8)
        ax.set_title(name.replace("_", " "))
        ax.set_xticks(range(int(df["severity"].max()) + 1))
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Pooled AUROC")
    for ax in axes[-1, :]:
        ax.set_xlabel("Severity (0 = clean)")
    axes.flat[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, out, stem)


def _panels(n: int, width: float = 3.2, height: float = 3.0):
    ncol = min(n, 3)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(width * ncol, height * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def e10_reliability_diagrams(tables: Path, out: Path) -> list[Path]:
    """Pooled top-label reliability per model, before and after temperature scaling."""
    df = pd.read_csv(tables / "e10_reliability_bins.csv")
    configs = sorted(df["config"].unique())
    apply_style()
    fig, axes = _panels(len(configs))
    for ax, config in zip(axes.flat, configs, strict=False):
        ax.plot([0.5, 1], [0.5, 1], color=INK_SECONDARY, linestyle="--", linewidth=0.8)
        for k, stage in enumerate(("before", "after")):
            part = df[(df["config"] == config) & (df["stage"] == stage)].dropna()
            ax.plot(
                part["confidence"],
                part["accuracy"],
                marker="o",
                markersize=3,
                color=color_for(k),
                label=f"{stage} temperature scaling",
            )
        ax.set_title(config, fontsize=9)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Accuracy in bin")
    axes.flat[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("E10: reliability diagrams (held-out folds, pooled; 15 bins)")
    fig.tight_layout()
    return save_figure(fig, out, "e10_reliability_diagrams")


def e10_risk_coverage(tables: Path, out: Path) -> list[Path]:
    """Selective risk vs coverage per model: single fold models vs 3-seed ensembles (pooled)."""
    df = pd.read_csv(tables / "e10_risk_coverage.csv")
    configs = sorted(df["config"].unique())
    apply_style()
    fig, axes = _panels(len(configs))
    for ax, config in zip(axes.flat, configs, strict=False):
        for k, kind in enumerate(("single", "ensemble")):
            part = df[(df["config"] == config) & (df["kind"] == kind)]
            ax.step(part["coverage"], part["risk"], where="post", color=color_for(k), label=kind)
        ax.set_title(config, fontsize=9)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk (error rate)")
    axes.flat[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("E10: risk-coverage (calibrated probabilities)")
    fig.tight_layout()
    return save_figure(fig, out, "e10_risk_coverage")


def e12_examples(tables: Path, out: Path) -> list[Path]:
    """4 held-out images x the saliency methods of the three fold-0 / seed-0 models (E12.md).

    Shows dataset images, so it is written to ``results/figures/local/`` (ADR-006) and built
    from the run folders' ``saliency_examples.npz`` and the local processed images.
    """
    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image

    from pharmachecked.config import CONFIG_DIR
    from pharmachecked.data.transforms import build_eval_geometry

    runs = REPO_ROOT / "results" / "runs"
    columns = [
        ("e05_efficientnetv2-s_default_f0_s0_*", "gradcam", "final: Grad-CAM"),
        ("e05_efficientnetv2-s_default_f0_s0_*", "gradcampp", "final: Grad-CAM++"),
        ("e05_efficientnetv2-s_default_f0_s0_*", "ig", "final: IG"),
        ("p4ref_dinov2-vits14_ft_f0_s0_*", "rollout", "DINOv2-S: rollout"),
        ("e04_multibranch_full_f0_s0_*", "mil", "E4: MIL attention"),
        ("e05_efficientnetv2-s_default_f0_s0_*", "random", "random"),
    ]
    arrays = {g: np.load(next(runs.glob(g)) / "saliency_examples.npz") for g, _, _ in columns}
    ids = [str(i) for i in arrays[columns[0][0]]["image_ids"]]
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    apply_style()
    fig, axes = plt.subplots(
        len(ids), len(columns) + 1, figsize=(1.9 * (len(columns) + 1), 1.9 * len(ids))
    )
    for r, image_id in enumerate(ids):
        with Image.open(REPO_ROOT / "data" / "processed" / "images" / f"{image_id}.png") as im:
            img = np.asarray(build_eval_geometry(tcfg, 224)(im.convert("RGB")))
        label = "counterfeit" if r < 2 else "authentic"
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(label, fontsize=8)
        for c, (g, method, title) in enumerate(columns, start=1):
            sal = arrays[g][method][r].astype(float)
            if sal.shape[0] != 224:
                sal = np.asarray(Image.fromarray(sal.astype("float32")).resize((224, 224)))
            axes[r, c].imshow(img)
            axes[r, c].imshow(sal, cmap="inferno", alpha=0.55)
            if r == 0:
                axes[r, c].set_title(title, fontsize=8)
        axes[0, 0].set_title("image", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("E12: saliency examples (fold 0, seed 0; first 2 counterfeit + 2 authentic)")
    fig.tight_layout()
    local = out / "local"
    local.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, local, "e12_examples")


def e14_ratio_curve(tables: Path, out: Path) -> list[Path]:
    """Pooled OOF AUROC (95% CI) vs synthetic:real counterfeit ratio; synthetic-only as a band."""
    df = pd.read_csv(tables / "e14_summary.csv").set_index("config")
    ratios = {
        0: "efficientnetv2-s/default",
        1: "efficientnetv2-s/procedural-r1",
        4: "efficientnetv2-s/procedural-r4",
    }
    apply_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    pts = [(r, *parse_ci(df.loc[c, "auroc (pooled)"])) for r, c in ratios.items() if c in df.index]
    xs, es = [p[0] for p in pts], [p[1] for p in pts]
    ax.errorbar(
        xs,
        es,
        yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
        fmt="o-",
        color=color_for(0),
        label="procedural, with real counterfeits",
    )
    only = "efficientnetv2-s/procedural-only"
    if only in df.index:
        e, lo, hi = parse_ci(df.loc[only, "auroc (pooled)"])
        ax.axhspan(lo, hi, color=color_for(1), alpha=0.2)
        ax.axhline(e, color=color_for(1), linestyle="--", label="procedural only (no real)")
    ax.set_xticks(xs)
    ax.set_xlabel("synthetic : real counterfeit ratio")
    ax.set_ylabel("pooled OOF AUROC (95% CI)")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("E14: AUROC vs synthetic ratio")
    fig.tight_layout()
    return save_figure(fig, out, "e14_ratio_curve")


def _pm_value(text: str) -> tuple[float, float]:
    mean, sd = (float(x) for x in str(text).split("±"))
    return mean, sd


def e15_success_vs_eps(tables: Path, out: Path) -> list[Path]:
    """Evasion rate (mean ± sd over fold models) vs L-inf eps, FGSM and PGD, per defence."""
    df = pd.read_csv(tables / "e15_defences.csv")
    defences = list(dict.fromkeys(df["defence"]))
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, attack in zip(axes, ("fgsm", "pgd"), strict=True):
        for k, d in enumerate(defences):
            part = df[(df["defence"] == d) & (df["attack"] == attack)].sort_values("eps (x255)")
            vals = [_pm_value(v) for v in part["evasion rate"]]
            ax.errorbar(
                part["eps (x255)"],
                [v[0] for v in vals],
                yerr=[v[1] for v in vals],
                fmt="o-",
                capsize=2,
                color=color_for(k),
                label=d,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlabel("L-inf eps (x/255)")
        ax.set_title(attack.upper())
    axes[0].set_ylabel("evasion rate (mean ± sd)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("E15: targeted evasion vs perturbation budget")
    fig.tight_layout()
    return save_figure(fig, out, "e15_success_vs_eps")


def e15_arms_race(tables: Path, out: Path) -> list[Path]:
    """Held-out black-box evasion per round; round 0 = the undefended final model."""
    race = pd.read_csv(tables / "e15_arms_race.csv")
    d = pd.read_csv(tables / "e15_defences.csv")
    r0 = d[(d["defence"] == "none") & (d["attack"] == "blackbox")]["evasion rate"].iloc[0]
    rounds = [0, *race["round"].tolist()]
    vals = [_pm_value(r0), *(_pm_value(v) for v in race["evasion rate"])]
    apply_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.errorbar(
        rounds,
        [v[0] for v in vals],
        yerr=[v[1] for v in vals],
        fmt="o-",
        capsize=2,
        color=color_for(0),
    )
    ax.set_xticks(rounds)
    ax.set_xlabel("arms-race round (0 = final model)")
    ax.set_ylabel("held-out evasion rate (mean ± sd)")
    ax.set_ylim(bottom=0)
    ax.set_title("E15: arms race (200-query black-box search)")
    fig.tight_layout()
    return save_figure(fig, out, "e15_arms_race")


def e16_pareto(tables: Path, out: Path) -> list[Path]:
    """Held-out AUROC vs model size (and latency once measured) per exported model."""
    df = pd.read_csv(tables / "e16_deployment.csv")
    has_latency = "latency ms (laptop CPU)" in df.columns
    xs = [("size MB", "model size (MB, log)")]
    if has_latency:
        xs.append(("latency ms (laptop CPU)", "laptop CPU latency (ms, batch 1, log)"))
    apply_style()
    fig, axes = plt.subplots(1, len(xs), figsize=(4.8 * len(xs), 3.6), squeeze=False)
    markers = {"fp32": "o", "int8": "s"}
    models = list(dict.fromkeys(df["model"]))
    for ax, (col, label) in zip(axes[0], xs, strict=True):
        for _, row in df.iterrows():
            x = _pm_value(row[col])[0] if "±" in str(row[col]) else float(row[col])
            ax.scatter(
                x,
                row["AUROC (pooled, held-out)"],
                marker=markers[row["precision"]],
                color=color_for(models.index(row["model"])),
                label=f"{row['model']} {row['precision']}",
            )
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("pooled held-out AUROC")
    axes[0, 0].legend(fontsize=6, loc="lower right")
    fig.suptitle("E16: accuracy vs cost of the exported models")
    fig.tight_layout()
    return save_figure(fig, out, "e16_pareto")


FIGURE_SETS: dict[str, list[Callable[[Path, Path], list[Path]]]] = {
    "e01": [e01_auroc_vs_params],
    "e02": [e02_auroc_by_strategy],
    "e04": [e04_gate_weights_by_packaging],
    "e06": [e06_learning_curves],
    "e08": [e08_seen_vs_unseen],
    "e09": [e09_severity_curves],
    "e09ext": [e09_extension_curves],
    "e10": [e10_reliability_diagrams, e10_risk_coverage],
    "e12": [e12_examples],
    "e14": [e14_ratio_curve],
    "e15": [e15_success_vs_eps, e15_arms_race],
    "e16": [e16_pareto],
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
