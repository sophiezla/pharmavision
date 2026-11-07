"""Tables for E12-E16 from the per-run files of the combined kit (ADR-027).

The standard training-arm tables (pooled OOF AUROC, external block, DeLong against the final
model) come from ``scripts/aggregate.py`` with ``--include e05:efficientnetv2-s/default``; the
functions here summarize the experiment-specific files written into the run folders.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pharmachecked.data.annotations import CONCEPTS
from pharmachecked.evaluation.aggregate import RunRecord
from pharmachecked.evaluation.metrics import auroc
from pharmachecked.evaluation.stats import holm
from pharmachecked.utils.io import read_json


def _pm(values: Iterable[float], digits: int = 3) -> str:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(v):
        return "n/a"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.{digits}f} ± {sd:.{digits}f}"


def _read(records: Iterable[RunRecord], name: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(r.root / name, dtype={"image_id": str}).assign(run=r.root.name)
        for r in records
        if (r.root / name).is_file()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def e12_faithfulness(models: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Deletion / insertion AUC per model x method; Wilcoxon vs random (Holm over all tests)."""
    rows, tests = [], []
    for label, recs in models.items():
        df = _read(recs, "faithfulness.csv")
        if df.empty:
            continue
        rand = df[df["method"] == "random"].set_index(["run", "image_id"])
        for method, part in df.groupby("method", sort=False):
            per_run = part.groupby("run")[["deletion_auc", "insertion_auc"]].mean()
            row = {
                "model": label,
                "method": method,
                "runs": len(per_run),
                "images": len(part),
                "deletion AUC (lower better)": _pm(per_run["deletion_auc"]),
                "insertion AUC (higher better)": _pm(per_run["insertion_auc"]),
            }
            if method != "random" and len(rand):
                paired = part.set_index(["run", "image_id"]).join(rand, rsuffix="_rand")
                for kind in ("deletion_auc", "insertion_auc"):
                    diff = paired[kind] - paired[f"{kind}_rand"]
                    p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
                    tests.append((len(rows), kind, p))
            rows.append(row)
    table = pd.DataFrame(rows)
    if tests:
        _, corrected = holm([t[2] for t in tests])
        for (i, kind, _p), q in zip(tests, corrected, strict=True):
            table.loc[i, f"{kind.split('_')[0]} vs random p (Holm)"] = f"{q:.3g}"
    return table


def e12_randomization(models: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Mean Spearman correlation with the original maps after each randomization step."""
    frames = [
        _read(recs, "randomization.csv").assign(model=label) for label, recs in models.items()
    ]
    frames = [f for f in frames if "spearman" in f]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[["model", "method", "step", "layer", "n_images", "spearman"]].round(4)


def _first_value(cell: object) -> object:
    """Recover the value of a cell that holds a stringified one-valued pandas Series."""
    if isinstance(cell, str) and "\nName: " in cell:
        values = set(re.findall(r"^\S+\s+(\S+)$", cell.split("\nName: ")[0], re.M))
        if len(values) == 1:
            return values.pop()
    return cell


def repair_localization(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the duplicate-id artefact of the Phases 6-8 kit's ``localization.csv``.

    The synthetic manifest held some image ids twice (ids are content hashes, so the images
    and masks are identical); the kit scored each copy and wrote the metadata lookup as a
    stringified Series. Keeps one row per run x method x image and restores the plain values.
    """
    out = df.copy()
    for col in ("generator", "edit_type", "source_image_id"):
        if col in out:
            out[col] = out[col].map(_first_value)
    return out.drop_duplicates(["run", "method", "image_id"]).reset_index(drop=True)


def e12_localization(models: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Pointing-game hit rate and top-15% IoU per model x method x generator x edit type."""
    rows = []
    for label, recs in models.items():
        df = _read(recs, "localization.csv")
        if df.empty:
            continue
        df = repair_localization(df)
        for keys, part in df.groupby(["method", "generator", "edit_type"]):
            per_run = part.groupby("run")[["hit", "iou"]].mean()
            rows.append(
                {
                    "model": label,
                    "method": keys[0],
                    "generator": keys[1],
                    "edit_type": keys[2],
                    "images": len(part) // max(1, part["run"].nunique()),
                    "pointing game": _pm(per_run["hit"]),
                    "IoU top 15%": _pm(per_run["iou"]),
                    "mean mask fraction": f"{part['mask_fraction'].mean():.3f}",
                }
            )
    return pd.DataFrame(rows)


def _concepts(records: list[RunRecord]) -> pd.DataFrame:
    """Seed-averaged concept predictions per held-out image."""
    df = _read(records, "concept_predictions.csv")
    return df.groupby("image_id").mean(numeric_only=True).reset_index()


def e13_concepts(arms: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Per-concept AUROC of the predicted concept on annotated held-out images."""
    rows = []
    for arm, recs in arms.items():
        df = _concepts(recs)
        for c in CONCEPTS:
            part = df[df[f"true_{c}"].notna()]
            y = part[f"true_{c}"].round().astype(int)
            value = auroc(y, part[f"pred_{c}"]) if y.nunique() == 2 else float("nan")
            rows.append(
                {
                    "arm": arm,
                    "concept": c,
                    "annotated images": len(part),
                    "present": int(y.sum()),
                    "AUROC": round(value, 4),
                }
            )
    return pd.DataFrame(rows)


def e13_intervention(arms: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Task AUROC on annotated held-out images after intervening on k concepts."""
    rows = []
    for arm, recs in arms.items():
        df = _concepts(recs)
        annotated = df[[f"true_{c}" for c in CONCEPTS]].notna().any(axis=1)
        part = df[annotated]
        for k in range(len(CONCEPTS) + 1):
            rows.append(
                {
                    "arm": arm,
                    "k": k,
                    "images": len(part),
                    "AUROC": round(auroc(part["y_true"].round(), part[f"prob_k{k}"]), 4),
                }
            )
    return pd.DataFrame(rows)


def e13_tcav(records: list[RunRecord], min_folds: int = 3) -> pd.DataFrame:
    """TCAV per concept x layer over the seed-0 fold models (E13.md: used if >= 3 of 5)."""
    df = _read(records, "tcav.csv")
    if df.empty:
        return df
    rows = []
    for (concept, layer), part in df.groupby(["concept", "layer"], sort=False):
        tested = part[part["status"] == "tested"]
        n_sig = int(tested["significant"].sum()) if len(tested) else 0
        rows.append(
            {
                "concept": concept,
                "layer": layer,
                "fold models tested": len(tested),
                "TCAV score": _pm(tested.get("tcav_mean", pd.Series(dtype=float))),
                "random score": _pm(tested.get("random_mean", pd.Series(dtype=float))),
                "significant (Holm)": n_sig,
                "used": "yes" if n_sig >= min_folds else "no",
            }
        )
    return pd.DataFrame(rows)


def e15_defences(defences: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Evasion rate per defence x attack among counterfeits detected when clean."""
    rows = []
    for label, recs in defences.items():
        df = _read(recs, "adversarial_predictions.csv")
        if df.empty:
            continue
        det = df[(df["detected_clean"] == 1) & (df["attack"] != "clean")]
        for (attack, eps), part in det.groupby(["attack", "eps"]):
            per_run = part.groupby("run")["evaded"].mean()
            row = {
                "defence": label,
                "attack": attack,
                "eps (x255)": round(eps * 255, 1) if eps else "",
                "runs": len(per_run),
                "evasion rate": _pm(per_run),
            }
            if attack == "blackbox":
                hits = part[part["evaded"] == 1]
                row["queries to evade (mean)"] = (
                    f"{hits['queries'].mean():.1f}" if len(hits) else "n/a"
                )
            rows.append(row)
        clean = df[df["attack"] == "clean"].groupby("run")["detected_clean"].mean()
        rows.append(
            {
                "defence": label,
                "attack": "clean (detected)",
                "eps (x255)": "",
                "runs": len(clean),
                "evasion rate": _pm(1 - clean),
            }
        )
    return pd.DataFrame(rows)


def e15_arms_race(rounds: dict[int, list[RunRecord]]) -> pd.DataFrame:
    """Held-out evasion rate per round over the lineages."""
    rows = []
    for r, recs in sorted(rounds.items()):
        stats = [read_json(x.root / "race.json") for x in recs if (x.root / "race.json").is_file()]
        if not stats:
            continue
        rows.append(
            {
                "round": r,
                "lineages": len(stats),
                "evasion rate": _pm(s["evasion_rate"] for s in stats),
                "evasive images added this round": _pm(
                    (s["evasive_added_per_round"][-1] for s in stats), 1
                ),
                "clean AUROC (runs)": _pm(x.metrics["heldout"]["auroc"] for x in recs),
            }
        )
    return pd.DataFrame(rows)


def e16_deployment(arms: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Held-out AUROC of the exported models (seed-averaged), size and export check."""
    rows = []
    for arm, recs in arms.items():
        preds = _read(recs, "deploy_predictions.csv")
        if preds.empty:
            continue
        info = [
            read_json(r.root / "deploy.json") for r in recs if (r.root / "deploy.json").is_file()
        ]
        held = preds[preds["split"] == "heldout"]
        passed = sum(bool(i["fp32_check_passed"]) for i in info)
        for precision, part in held.groupby("precision"):
            pooled = part.groupby("image_id").agg(y=("y_true", "first"), p=("prob", "mean"))
            rows.append(
                {
                    "model": arm,
                    "precision": precision,
                    "runs": part["run"].nunique(),
                    "AUROC (pooled, held-out)": round(auroc(pooled["y"], pooled["p"]), 4),
                    "size MB": _pm((i[f"size_mb_{precision}"] for i in info), 2),
                    "max |ONNX - torch| (FP32)": f"{max(i['max_abs_diff_fp32'] for i in info):.2e}",
                    "export checks passed": f"{passed}/{len(info)}",
                }
            )
    return pd.DataFrame(rows)


DEPLOY_ALPHA = 0.10  # E11.md: alpha of the deployed system
DEPLOY_CONFORMAL = ("lac", "mondrian")  # ADR-028


def e16_refit_runs(preds: pd.DataFrame) -> pd.DataFrame:
    """Re-fit temperature, threshold and conformal quantiles per exported model (PLAN 8.3).

    Every quantity is fitted on the run's own ``cal`` (temperature, conformal) or ``val``
    (Youden threshold) predictions of that precision and evaluated on its held-out fold, with
    the E10 methods: T = 1 when not identifiable, LAC Mondrian at alpha 0.10 (ADR-028).

    Args:
        preds: ``deploy_predictions.csv`` rows of several runs with a ``run`` column.

    Returns:
        One row per run x precision.
    """
    from pharmachecked.evaluation.calibration import (
        apply_temperature,
        fit_temperature,
        temperature_identifiable,
    )
    from pharmachecked.evaluation.conformal import calibrate, prediction_sets, set_metrics
    from pharmachecked.evaluation.metrics import expected_calibration_error, youden_threshold

    rows = []
    for (run, precision), part in preds.groupby(["run", "precision"]):
        split = {s: g for s, g in part.groupby("split")}
        cal, val, held = split["cal"], split["val"], split["heldout"]
        fitted = temperature_identifiable(cal["logit"], cal["y_true"])
        temp = fit_temperature(cal["logit"], cal["y_true"]) if fitted else 1.0
        p = {s: apply_temperature(g["logit"], temp) for s, g in split.items()}
        thr = youden_threshold(val["y_true"], p["val"])
        q = calibrate(cal["y_true"], p["cal"], *DEPLOY_CONFORMAL, DEPLOY_ALPHA)
        sets = prediction_sets(p["heldout"], DEPLOY_CONFORMAL[0], q)
        y = held["y_true"].to_numpy()
        rows.append(
            {
                "run": run,
                "precision": precision,
                "temperature": temp,
                "t_fitted": fitted,
                "auroc": auroc(y, p["heldout"]),
                "ece": expected_calibration_error(y, p["heldout"]),
                "error_at_threshold": float(((p["heldout"] >= thr).astype(int) != y).mean()),
            }
            | set_metrics(y, sets)
        )
    return pd.DataFrame(rows)


def e16_reliability(arms: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Mean ± sd over fold models of the re-fitted E10 metrics per model x precision."""
    rows = []
    for arm, recs in arms.items():
        preds = _read(recs, "deploy_predictions.csv")
        if preds.empty:
            continue
        per_run = e16_refit_runs(preds)
        for precision, part in per_run.groupby("precision"):
            rows.append(
                {
                    "model": arm,
                    "precision": precision,
                    "runs": len(part),
                    "T fitted (identifiable)": int(part["t_fitted"].sum()),
                    "AUROC (per run)": _pm(part["auroc"]),
                    "ECE after T": _pm(part["ece"], 4),
                    "error at val threshold": _pm(part["error_at_threshold"], 4),
                    "coverage (LAC Mondrian, alpha 0.10)": _pm(part["coverage"]),
                    "coverage authentic": _pm(part["coverage_authentic"]),
                    "coverage counterfeit": _pm(part["coverage_counterfeit"]),
                    "mean set size": _pm(part["mean_set_size"]),
                    "abstention rate": _pm(part["abstention_rate"]),
                }
            )
    return pd.DataFrame(rows)


def e14_open_set_gain(
    e14os: list[RunRecord], e08: list[RunRecord], types: Iterable[str], best: str, n_boot: int
) -> pd.DataFrame:
    """Unseen-type AUROC of the best E14 arm vs the E8 final model (E14.md E8 re-run).

    Paired over the held-out type's images (seed-averaged): AUROC difference (E14 minus E8)
    with a paired bootstrap CI and DeLong p, Holm over the types.
    """
    from pharmachecked.evaluation.aggregate import pooled_oof
    from pharmachecked.evaluation.stats import bootstrap_ci, delong_test

    rows, intervals = [], []
    for t in types:
        a = [r for r in e14os if r.variant == f"{best}-no-{t}"]
        b = [r for r in e08 if r.model == "efficientnetv2-s" and r.variant == f"default-no-{t}"]
        if not a or not b:
            continue
        m = pooled_oof(b).merge(pooled_oof(a), on="image_id", suffixes=("_e08", ""))
        y, pa, pb = m["y_true"].to_numpy(), m["prob"].to_numpy(), m["prob_e08"].to_numpy()
        ci = bootstrap_ci(y, (pa, pb), lambda yy, p, q: auroc(yy, p) - auroc(yy, q), n_boot)
        intervals.append(ci)
        rows.append(
            {
                "held-out type": t,
                "images": len(m),
                "runs (E14 / E8)": f"{len(a)} / {len(b)}",
                "E8 final model AUROC": round(auroc(y, pb), 4),
                f"E14 {best} AUROC": round(auroc(y, pa), 4),
                "AUROC diff (E14 - E8)": f"{ci.estimate:.3f} [{ci.low:.3f}, {ci.high:.3f}]",
                "delong_p": delong_test(y, pa, pb).p_value,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        reject, corrected = holm(out["delong_p"].tolist())
        out["holm_p"] = corrected
        out["improves"] = [bool(r and c.low > 0) for r, c in zip(reject, intervals, strict=True)]
        out["worse"] = [bool(r and c.high < 0) for r, c in zip(reject, intervals, strict=True)]
    return out
