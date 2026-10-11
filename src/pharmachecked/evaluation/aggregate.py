"""Combine run folders into per-configuration results (PLAN 2.2; CLAUDE.md section 5).

For each configuration (model + variant) of one experiment:

* per-run metrics on the held-out fold: mean ± std over the runs (protocol: 5 folds x 3 seeds);
* pooled OOF: each development image's probability averaged over seeds, then AUROC, AUPRC,
  sensitivity at 95% specificity, ECE and Brier with 95% stratified bootstrap CIs over images;
  the pooled hard decision is the majority vote of the per-run decisions (each at its own
  val-selected threshold);
* external block (ADR-005): for every external image, the fraction of runs deciding correctly;
  its mean is the run-averaged specificity (authentic sets) or recall (counterfeit sets), with a
  bootstrap CI over images.

Pairwise comparisons: paired DeLong of each configuration against the best pooled AUROC, Holm
corrected within the experiment, plus the bootstrap CI of the AUROC difference.
Dirty runs (``*_dirty``) and incomplete run folders are excluded.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pharmachecked.evaluation.metrics import (
    auprc,
    auroc,
    brier,
    expected_calibration_error,
    log_loss,
    sensitivity_at_specificity,
)
from pharmachecked.evaluation.stats import bootstrap_ci, delong_test, holm
from pharmachecked.utils.io import read_json
from pharmachecked.utils.run_dir import missing_files, parse_run_id

PER_RUN_METRICS = (
    "auroc",
    "auprc",
    "sens_at_95_spec",
    "balanced_accuracy",
    "macro_f1",
    "ece",
    "brier",
)
POOLED_METRICS = {
    "auroc": auroc,
    "auprc": auprc,
    "sens_at_95_spec": sensitivity_at_specificity,
    "ece": expected_calibration_error,
    "brier": brier,
    "log_loss": log_loss,
}


@dataclass(frozen=True)
class RunRecord:
    """One complete, clean run folder."""

    root: Path
    model: str
    variant: str
    fold: int
    seed: int
    metrics: dict
    threshold: float


def collect_runs(runs_dir: Path, exp: str) -> list[RunRecord]:
    """Complete, non-dirty run folders of experiment ``exp``."""
    records = []
    for folder in sorted(runs_dir.glob(f"{exp}_*")):
        try:
            fields = parse_run_id(folder.name)
        except ValueError:
            continue
        if fields["exp"] != exp or fields["dirty"] or missing_files(folder):
            continue
        metrics = read_json(folder / "metrics.json")
        records.append(
            RunRecord(
                folder,
                str(fields["model"]),
                str(fields["variant"]),
                int(fields["fold"]),
                int(fields["seed"]),
                metrics,
                float(metrics["heldout"]["threshold"]),
            )
        )
    keys = [(r.model, r.variant, r.fold, r.seed) for r in records]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"several complete runs for the same arm/fold/seed: {duplicates}")
    return records


def _predictions(records: Iterable[RunRecord], name: str) -> pd.DataFrame:
    frames = []
    for r in records:
        df = pd.read_csv(r.root / name)
        df["decision"] = (df["prob"] >= r.threshold).astype(int)
        df["run"] = r.root.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pooled_oof(records: list[RunRecord]) -> pd.DataFrame:
    """One row per development image: seed-averaged probability and majority decision."""
    df = _predictions(records, "oof_predictions.csv")
    return (
        df.groupby("image_id")
        .agg(
            y_true=("y_true", "first"),
            prob=("prob", "mean"),
            decision=("decision", lambda d: int(d.mean() >= 0.5)),
            packaging_type=("packaging_type", "first"),
            n_runs=("run", "nunique"),
        )
        .reset_index()
    )


def external_correctness(records: list[RunRecord]) -> pd.DataFrame:
    """One row per external image: fraction of runs whose decision matches the label."""
    df = _predictions(records, "external_predictions.csv")
    df["correct"] = (df["decision"] == df["y_true"]).astype(float)
    return (
        df.groupby(["source_dataset", "image_id"])
        .agg(y_true=("y_true", "first"), correct=("correct", "mean"))
        .reset_index()
    )


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def _ci(interval: object) -> str:
    return f"{interval.estimate:.3f} [{interval.low:.3f}, {interval.high:.3f}]"  # type: ignore[attr-defined]


def summarize(
    records: list[RunRecord], n_boot: int = 2000
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Summary rows per configuration and the pooled OOF frame of each configuration."""
    by_config: dict[str, list[RunRecord]] = {}
    for r in records:
        by_config.setdefault(f"{r.model}/{r.variant}", []).append(r)
    rows, pooled_frames = [], {}
    for config, recs in sorted(by_config.items()):
        heldout = pd.DataFrame([r.metrics["heldout"] for r in recs])
        row: dict[str, object] = {
            "config": config,
            "runs": len(recs),
            "folds": len({r.fold for r in recs}),
            "seeds": len({r.seed for r in recs}),
        }
        for m in PER_RUN_METRICS:
            row[f"{m} (runs)"] = _fmt(
                heldout[m].mean(), heldout[m].std(ddof=1) if len(recs) > 1 else 0.0
            )
        pooled = pooled_oof(recs)
        pooled_frames[config] = pooled
        for name, fn in POOLED_METRICS.items():
            row[f"{name} (pooled)"] = _ci(
                bootstrap_ci(pooled["y_true"], pooled["prob"], fn, n_boot=n_boot)
            )
        ext = external_correctness(recs)
        for source, part in ext.groupby("source_dataset"):
            label = "specificity" if part["y_true"].iloc[0] == 0 else "recall"
            ci = bootstrap_ci(
                np.zeros(len(part), dtype=int),
                part["correct"].to_numpy(),
                lambda _y, c: float(np.mean(c)),
                n_boot=n_boot,
                stratified=False,
            )
            row[f"{source} {label}"] = _ci(ci)
        train_s = [r.metrics.get("train_seconds", float("nan")) for r in recs]
        row["train min/run"] = f"{np.nanmean(train_s) / 60:.1f}"
        params = [r.metrics.get("n_params") for r in recs]
        trainable = [r.metrics.get("n_trainable_params") for r in recs]
        row["params (M)"] = params[0] / 1e6 if params[0] is not None else float("nan")
        row["trainable params (M)"] = (
            trainable[0] / 1e6 if trainable[0] is not None else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows), pooled_frames


def pairwise_vs_best(pooled: dict[str, pd.DataFrame], n_boot: int = 2000) -> pd.DataFrame:
    """DeLong (Holm-corrected) and bootstrap AUROC difference of each configuration vs the best."""
    aurocs = {c: auroc(df["y_true"], df["prob"]) for c, df in pooled.items()}
    best = max(aurocs, key=lambda c: aurocs[c])
    rows = []
    for config, df in pooled.items():
        if config == best:
            continue
        merged = pooled[best].merge(df, on="image_id", suffixes=("_best", ""))
        test = delong_test(merged["y_true"], merged["prob_best"], merged["prob"])
        diff = bootstrap_ci(
            merged["y_true"],
            (merged["prob_best"].to_numpy(), merged["prob"].to_numpy()),
            lambda y, a, b: auroc(y, a) - auroc(y, b),
            n_boot=n_boot,
        )
        rows.append(
            {"config": config, "best": best, "auroc_diff": _ci(diff), "delong_p": test.p_value}
        )
    out = pd.DataFrame(rows, columns=["config", "best", "auroc_diff", "delong_p"])
    if len(out):
        reject, corrected = holm(out["delong_p"].tolist())
        out["holm_p"] = corrected
        out["different"] = reject
    return out


def to_markdown(df: pd.DataFrame) -> str:
    """Markdown table without a pandas tabulate dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
