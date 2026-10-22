"""Tables for E12-E16 from the per-run files of the combined kit (ADR-027).

The standard training-arm tables (pooled OOF AUROC, external block, DeLong against the final
model) come from ``scripts/aggregate.py`` with ``--include e05:efficientnetv2-s/default``; the
functions here summarize the experiment-specific files written into the run folders.
"""

from __future__ import annotations

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


def e12_localization(models: dict[str, list[RunRecord]]) -> pd.DataFrame:
    """Pointing-game hit rate and top-15% IoU per model x method x generator x edit type."""
    rows = []
    for label, recs in models.items():
        df = _read(recs, "localization.csv")
        if df.empty:
            continue
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
