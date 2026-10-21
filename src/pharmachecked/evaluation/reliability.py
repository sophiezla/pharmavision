"""E10/E11 tables from the fold models' ``reliability_predictions.csv`` (PLAN 5.4, 5.6, 5.7).

Pre-registered in docs/experiments/E10.md and E11.md. Every fitted quantity (temperature,
conformal quantile, guard threshold, Mahalanobis/kNN statistics) comes from the fold model's
own ``cal``, ``val`` or ``train`` images (Rule 3); held-out folds and OOD images are only scored.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from pharmachecked.data.ood import OOD_SETS, load_ood_set, select_eval_split
from pharmachecked.evaluation.aggregate import RunRecord
from pharmachecked.evaluation.calibration import (
    apply_temperature,
    fit_temperature,
    temperature_identifiable,
)
from pharmachecked.evaluation.conformal import (
    MODES,
    SCORES,
    calibrate,
    evaluate,
    prediction_sets,
)
from pharmachecked.evaluation.metrics import (
    brier,
    expected_calibration_error,
    log_loss,
    youden_threshold,
)
from pharmachecked.evaluation.ood import (
    detection_auroc,
    energy_score,
    fpr_at_95_tpr,
    msp_score,
    threshold_at_acceptance,
)
from pharmachecked.evaluation.reliability_pass import RELIABILITY_FILE
from pharmachecked.evaluation.selective import aurc, risk_coverage
from pharmachecked.evaluation.stats import bootstrap_ci

RUN_SCORES = ("msp", "energy", "maha", "knn")  # per fold model
GUARD_TIE_BREAK = "knn"  # E11.md
CONFORMAL_ALPHA = 0.10  # E11.md: alpha of the end-to-end system (0.05 reported too)
CHANGE_PP = 0.02  # E10.md: per-class coverage may fall at most 2 pp below 1 - alpha


def _pm(values: Iterable[float], digits: int = 3) -> str:
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:  # e.g. error on issued when a system issues nothing
        return "n/a"
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return f"{v.mean():.{digits}f} ± {sd:.{digits}f}"


def load(records: list[RunRecord]) -> pd.DataFrame:
    """Every fold model's reliability predictions with ``config`` and ``run`` columns, plus
    ``temperature`` (fitted on the run's ``cal`` logits; 1 when not identifiable, E10.md),
    ``temperature_fitted`` and ``prob_cal`` = sigmoid(logit / temperature)."""
    frames = []
    for r in records:
        df = pd.read_csv(r.root / RELIABILITY_FILE, dtype={"image_id": str})
        cal = df[df["split"] == "cal"]
        z, y = cal["logit"].to_numpy(), cal["y_true"].to_numpy()
        fitted = temperature_identifiable(z, y)
        t = fit_temperature(z, y) if fitted else 1.0
        frames.append(
            df.assign(
                config=f"{r.model}/{r.variant}",
                run=r.root.name,
                temperature=t,
                temperature_fitted=fitted,
                prob_cal=apply_temperature(df["logit"].to_numpy(), t),
            )
        )
    return pd.concat(frames, ignore_index=True)


def _pooled(held: pd.DataFrame, column: str) -> pd.DataFrame:
    return held.groupby("image_id", as_index=False).agg(
        y_true=("y_true", "first"), prob=(column, "mean")
    )


# ---------------------------------------------------------------- E10


def calibration_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """ECE, Brier and log-loss before / after temperature scaling, pooled, with paired CIs."""
    rows = []
    for config, part in sorted(df[df["split"] == "heldout"].groupby("config")):
        before, after = _pooled(part, "prob"), _pooled(part, "prob_cal")
        y = before["y_true"].to_numpy()
        pb = before["prob"].to_numpy()
        pa = after.set_index("image_id").loc[before["image_id"], "prob"].to_numpy()
        per_run = part.groupby("run")[["temperature", "temperature_fitted"]].first()
        row: dict[str, object] = {"config": config, "runs": len(per_run), "images": len(y)}
        row["T fitted (identifiable)"] = int(per_run["temperature_fitted"].sum())
        fitted = per_run.loc[per_run["temperature_fitted"], "temperature"]
        row["temperature (fitted runs)"] = _pm(fitted, 2) if len(fitted) else "n/a"
        for name, fn in (
            ("ece", expected_calibration_error),
            ("brier", brier),
            ("log_loss", log_loss),
        ):
            diff = bootstrap_ci(y, (pa, pb), lambda yy, a, b, f=fn: f(yy, a) - f(yy, b), n_boot)
            row[f"{name} before"] = round(fn(y, pb), 4)
            row[f"{name} after"] = round(fn(y, pa), 4)
            row[f"{name} diff (after - before)"] = (
                f"{diff.estimate:.4f} [{diff.low:.4f}, {diff.high:.4f}]"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_diagram_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled top-label reliability bins per config, before and after temperature scaling."""
    from pharmachecked.evaluation.calibration import reliability_bins

    frames = []
    for config, part in sorted(df[df["split"] == "heldout"].groupby("config")):
        for stage, column in (("before", "prob"), ("after", "prob_cal")):
            pooled = _pooled(part, column)
            bins = reliability_bins(pooled["y_true"], pooled["prob"])
            frames.append(bins.assign(config=config, stage=stage))
    return pd.concat(frames, ignore_index=True)


def _ensemble(held: pd.DataFrame) -> pd.DataFrame:
    """Per fold, the mean calibrated probability of the three seeds per image."""
    return held.groupby(["config", "fold", "image_id"], as_index=False).agg(
        y_true=("y_true", "first"), prob_cal=("prob_cal", "mean")
    )


def selective_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AURC of a single fold model vs the 3-seed ensemble per fold, and pooled curves."""
    held = df[df["split"] == "heldout"]
    ens = _ensemble(held)
    rows, curves = [], []
    for config, part in sorted(held.groupby("config")):
        single = [aurc(g["y_true"], g["prob_cal"]) for _, g in part.groupby("run")]
        e_part = ens[ens["config"] == config]
        ensemble = [aurc(g["y_true"], g["prob_cal"]) for _, g in e_part.groupby("fold")]
        err_single = [
            float(((g["prob_cal"] >= 0.5).astype(int) != g["y_true"]).mean())
            for _, g in part.groupby("run")
        ]
        err_ens = [
            float(((g["prob_cal"] >= 0.5).astype(int) != g["y_true"]).mean())
            for _, g in e_part.groupby("fold")
        ]
        rows.append(
            {
                "config": config,
                "AURC single (15 fold models)": _pm(single, 4),
                "AURC ensemble (5 folds)": _pm(ensemble, 4),
                "error at full coverage, single": _pm(err_single, 4),
                "error at full coverage, ensemble": _pm(err_ens, 4),
            }
        )
        for kind, frame in (("single", part), ("ensemble", e_part)):
            curve = risk_coverage(frame["y_true"], frame["prob_cal"])
            curves.append(curve.assign(config=config, kind=kind))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def conformal_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Every score x mode x alpha per fold model: ``cal`` -> held-out, calibrated probs."""
    frames = []
    for (config, run), part in df.groupby(["config", "run"]):
        cal, held = part[part["split"] == "cal"], part[part["split"] == "heldout"]
        table = evaluate(cal["y_true"], cal["prob_cal"], held["y_true"], held["prob_cal"])
        frames.append(table.assign(config=config, run=run))
    return pd.concat(frames, ignore_index=True)


METRICS = (
    "coverage",
    "coverage_authentic",
    "coverage_counterfeit",
    "mean_set_size",
    "singleton_rate",
    "abstention_rate",
    "empty_rate",
)


def conformal_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean ± sd over fold models per config x score x mode x alpha."""
    rows = []
    for (config, score, mode, alpha), part in per_run.groupby(["config", "score", "mode", "alpha"]):
        row = {"config": config, "score": score, "mode": mode, "alpha": alpha, "runs": len(part)}
        row |= {m: _pm(part[m]) for m in METRICS}
        rows.append(row)
    return pd.DataFrame(rows)


def choose_conformal(per_run: pd.DataFrame, config: str) -> tuple[str, str, bool, pd.DataFrame]:
    """The pre-registered conformal choice for ``config`` (E10.md).

    Returns:
        ``(score, mode, criterion_met, summary)``; ``summary`` lists each candidate's mean
        per-class coverage shortfall and mean set size at alpha = 0.10.
    """
    part = per_run[per_run["config"] == config]
    rows = []
    for score in SCORES:
        for mode in MODES:
            sub = part[(part["score"] == score) & (part["mode"] == mode)]
            shortfall = 0.0
            for alpha, a_part in sub.groupby("alpha"):
                for col in ("coverage_authentic", "coverage_counterfeit"):
                    shortfall = max(shortfall, (1 - alpha) - a_part[col].mean())
            size = sub[sub["alpha"] == CONFORMAL_ALPHA]["mean_set_size"].mean()
            rows.append(
                {
                    "score": score,
                    "mode": mode,
                    "largest per-class shortfall (pp)": round(100 * shortfall, 2),
                    "mean set size (alpha 0.10)": round(size, 4),
                    "qualifies": shortfall <= CHANGE_PP,
                }
            )
    summary = pd.DataFrame(rows)
    ok = summary[summary["qualifies"]]
    met = not ok.empty
    pool = ok if met else summary
    key = "mean set size (alpha 0.10)" if met else "largest per-class shortfall (pp)"
    # ties: LAC before APS, then Mondrian before marginal
    pool = pool.assign(
        _s=pool["score"].map({"lac": 0, "aps": 1}),
        _m=pool["mode"].map({"mondrian": 0, "marginal": 1}),
    ).sort_values([key, "_s", "_m"])
    best = pool.iloc[0]
    return str(best["score"]), str(best["mode"]), met, summary


# ---------------------------------------------------------------- E11


def _with_scores(part: pd.DataFrame) -> pd.DataFrame:
    z = part["logit"].to_numpy()
    return part.assign(msp=msp_score(z), energy=energy_score(z))


def _ood_halves(data_dir: Path) -> pd.Series:
    halves = [select_eval_split(load_ood_set(data_dir, name)[0]) for name in OOD_SETS]
    return pd.concat(halves)


def ood_table(df: pd.DataFrame) -> pd.DataFrame:
    """AUROC, FPR@95TPR and acceptance at the val threshold, near and far, per config x score."""
    rows = []
    scored = _with_scores(df)
    for config, part in sorted(scored.groupby("config")):
        per_score: dict[str, dict[str, list[float]]] = {}
        for _, run in part.groupby("run"):
            for score in RUN_SCORES:
                if run[score].isna().all():
                    continue
                _ood_metrics(run, "run", score, per_score.setdefault(score, {}))
        # ensemble disagreement: one score per fold (std over the fold's three seeds)
        ens = part.assign(fold_model=part["run"].str.extract(r"_f(\d+)_s\d+_")[0])
        dis = ens.groupby(["fold_model", "split", "image_id"], as_index=False).agg(
            disagreement=("prob", "std")
        )
        if dis["disagreement"].notna().all():  # needs >= 2 seeds per fold (not in the self-test)
            for _, fold in dis.groupby("fold_model"):
                _ood_metrics(fold, "fold", "disagreement", per_score.setdefault("disagreement", {}))
        for score, metrics in per_score.items():
            row = {"config": config, "score": score, "units": len(metrics["near AUROC"])}
            row |= {k: _pm(v) for k, v in metrics.items()}
            rows.append(row)
    return pd.DataFrame(rows)


def _ood_metrics(frame: pd.DataFrame, _unit: str, score: str, out: dict[str, list[float]]) -> None:
    id_test = frame.loc[frame["split"] == "heldout", score].to_numpy()
    id_val = frame.loc[frame["split"] == "val", score].to_numpy()
    t = threshold_at_acceptance(id_val)
    for name in OOD_SETS:
        ood = frame.loc[frame["split"] == f"ood_{name}", score].to_numpy()
        out.setdefault(f"{name} AUROC", []).append(detection_auroc(id_test, ood))
        out.setdefault(f"{name} FPR@95TPR", []).append(fpr_at_95_tpr(id_test, ood))
        out.setdefault(f"{name} accepted at val threshold", []).append(float(np.mean(ood <= t)))
    out.setdefault("ID held-out accepted at val threshold", []).append(float(np.mean(id_test <= t)))


def choose_guard(df: pd.DataFrame, config: str, data_dir: Path) -> tuple[str, pd.DataFrame]:
    """The guard score for ``config``: best mean OOD AUROC (near + far) on the select half."""
    halves = _ood_halves(data_dir)
    part = _with_scores(df[df["config"] == config])
    is_ood = part["split"].str.startswith("ood_")
    keep = ~is_ood | (part["image_id"].map(halves) == "select")
    part = part[keep]
    rows = []
    for score in RUN_SCORES:
        if part[score].isna().all():
            continue
        values = []
        for _, run in part.groupby("run"):
            id_test = run.loc[run["split"] == "heldout", score]
            ood = run.loc[run["split"].str.startswith("ood_"), score]
            values.append(detection_auroc(id_test, ood))
        rows.append({"score": score, "select-half OOD AUROC": float(np.mean(values))})
    summary = pd.DataFrame(rows)
    best = summary["select-half OOD AUROC"].max()
    tied = summary[np.isclose(summary["select-half OOD AUROC"], best)]["score"].tolist()
    chosen = GUARD_TIE_BREAK if GUARD_TIE_BREAK in tied else tied[0]
    return chosen, summary


def end_to_end_table(
    df: pd.DataFrame,
    config: str,
    guard: str,
    score: str,
    mode: str,
    data_dir: Path,
    alphas: tuple[float, ...] = (0.10, 0.05),
) -> pd.DataFrame:
    """Full system vs bare classifier on held-out + OOD eval images, per fold model."""
    halves = _ood_halves(data_dir)
    part = _with_scores(df[df["config"] == config])
    per_system: dict[str, list[dict[str, float]]] = {}
    for _, run in part.groupby("run"):
        val, cal = run[run["split"] == "val"], run[run["split"] == "cal"]
        stream = run[
            (run["split"] == "heldout")
            | (run["split"].str.startswith("ood_") & (run["image_id"].map(halves) == "eval"))
        ]
        is_ood = stream["split"].str.startswith("ood_").to_numpy()
        y = stream["y_true"].to_numpy()
        # bare classifier: val-selected threshold on the raw probability, every image issued
        t_bare = youden_threshold(val["y_true"], val["prob"])
        pred = (stream["prob"].to_numpy() >= t_bare).astype(int)
        wrong = is_ood | (pred != y)
        per_system.setdefault("bare classifier", []).append(
            _stream_metrics(is_ood, np.zeros(len(y), bool), np.ones(len(y), bool), wrong)
        )
        accepted = stream[guard].to_numpy() <= threshold_at_acceptance(val[guard].to_numpy())
        for alpha in alphas:
            q = calibrate(cal["y_true"], cal["prob_cal"], score, mode, alpha)
            sets = prediction_sets(stream["prob_cal"].to_numpy(), score, q)
            issued = accepted & (sets.sum(axis=1) == 1)
            label = sets.argmax(axis=1)
            wrong = is_ood | (label != y)
            per_system.setdefault(f"full system (alpha {alpha})", []).append(
                _stream_metrics(is_ood, ~accepted, issued, wrong)
            )
    rows = []
    for system, runs in per_system.items():
        row: dict[str, object] = {"system": system, "runs": len(runs)}
        for key in runs[0]:
            row[key] = _pm([r[key] for r in runs])
        rows.append(row)
    return pd.DataFrame(rows)


def _stream_metrics(
    is_ood: np.ndarray, rejected: np.ndarray, issued: np.ndarray, wrong: np.ndarray
) -> dict[str, float]:
    abstained = ~rejected & ~issued
    n_issued = int(issued.sum())
    return {
        "ID rejected": float(rejected[~is_ood].mean()),
        "OOD rejected": float(rejected[is_ood].mean()),
        "abstained": float(abstained.mean()),
        "issued": float(issued.mean()),
        "error on issued": float(wrong[issued].mean()) if n_issued else float("nan"),
        "OOD among issued": float(is_ood[issued].mean()) if n_issued else float("nan"),
    }
