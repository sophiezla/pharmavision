"""Phase 4 tables: E7 transfer, E8 open set, E9 robustness summaries (PLAN 4.4, 4.7, 4.10).

All numbers are computed from run folders: seed-averaged probabilities per image, 95%
stratified bootstrap CIs (CLAUDE.md section 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmachecked.evaluation.aggregate import (
    RunRecord,
    external_correctness,
    pooled_oof,
)
from pharmachecked.evaluation.metrics import auprc, auroc, sensitivity_at_specificity
from pharmachecked.evaluation.robustness import CLEAN, quality_limits
from pharmachecked.evaluation.stats import Interval, _resample_indices, bootstrap_ci


def _ci_text(i: Interval) -> str:
    return f"{i.estimate:.3f} [{i.low:.3f}, {i.high:.3f}]"


def difference_ci(
    y_a: np.ndarray, p_a: np.ndarray, y_b: np.ndarray, p_b: np.ndarray, n_boot: int, seed: int = 0
) -> Interval:
    """AUROC(a) - AUROC(b) for two independent image sets; stratified resampling of each."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia, ib = _resample_indices(y_a, rng, True), _resample_indices(y_b, rng, True)
        diffs.append(auroc(y_a[ia], p_a[ia]) - auroc(y_b[ib], p_b[ib]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return Interval(auroc(y_a, p_a) - auroc(y_b, p_b), float(low), float(high))


def pooled_external(records: list[RunRecord]) -> pd.DataFrame:
    """Seed-averaged probability per external image (``image_id, source_dataset, y_true, prob``)."""
    frames = [
        pd.read_csv(r.root / "external_predictions.csv").assign(run=r.root.name) for r in records
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["image_id", "source_dataset"], as_index=False).agg(
        y_true=("y_true", "first"), prob=("prob", "mean")
    )


def _by_config(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    out: dict[str, list[RunRecord]] = {}
    for r in records:
        out.setdefault(f"{r.model}/{r.variant}", []).append(r)
    return out


def e07_table(transfer: list[RunRecord], reference: list[RunRecord], n_boot: int) -> pd.DataFrame:
    """In-source (5-fold OOF) vs cross-source (pooled external) AUROC per model (ADR-022)."""
    refs = _by_config(reference)
    rows = []
    for config, recs in sorted(_by_config(transfer).items()):
        if config not in refs:
            raise KeyError(f"no in-source reference runs for {config}")
        ins, ext = pooled_oof(refs[config]), pooled_external(recs)
        yi, pi = ins["y_true"].to_numpy(), ins["prob"].to_numpy()
        ye, pe = ext["y_true"].to_numpy(), ext["prob"].to_numpy()
        # Mendeley is all authentic and Split E all counterfeit (ADR-005), so specificity and
        # recall are read off by label: fraction of runs deciding correctly, mean over images.
        correct = external_correctness(recs)
        specificity = correct.loc[correct["y_true"] == 0, "correct"]
        recall = correct.loc[correct["y_true"] == 1, "correct"]
        rows.append(
            {
                "config": config,
                "runs": len(recs),
                "in-source AUROC (5-fold OOF)": _ci_text(bootstrap_ci(yi, (pi,), auroc, n_boot)),
                "cross-source AUROC": _ci_text(bootstrap_ci(ye, (pe,), auroc, n_boot)),
                "drop (in - cross)": _ci_text(difference_ci(yi, pi, ye, pe, n_boot)),
                "external specificity (Mendeley)": f"{specificity.mean():.3f}",
                "external recall (Split E)": f"{recall.mean():.3f}",
                "external images": len(ext),
            }
        )
    return pd.DataFrame(rows)


def _ranking_row(y: np.ndarray, p: np.ndarray, n_boot: int) -> dict[str, object]:
    return {
        "auroc": bootstrap_ci(y, (p,), auroc, n_boot),
        "auprc": auprc(y, p),
        "sens_at_95_spec": sensitivity_at_specificity(y, p),
        "images": len(y),
    }


def e08_table(
    open_set: list[RunRecord], reference: list[RunRecord], types: list[str], n_boot: int
) -> pd.DataFrame:
    """Unseen-type AUROC per type x method, with the seen-type AUROC of the supervised models.

    ``open_set`` variants end in ``-no-<type>``; the method is the variant without that suffix.
    The seen reference is the type's subset of the method's within-source 5-fold OOF.
    """
    refs = _by_config(reference)
    rows = []
    for t in types:
        suffix = f"-no-{t}"
        for config, recs in sorted(_by_config(open_set).items()):
            if not config.endswith(suffix):
                continue
            method = config.removesuffix(suffix)
            unseen = pooled_oof(recs)
            u = _ranking_row(unseen["y_true"].to_numpy(), unseen["prob"].to_numpy(), n_boot)
            seen_text = "-"
            if method in refs:
                ref = pooled_oof(refs[method])
                ref = ref[ref["packaging_type"] == t]
                yi, pi = ref["y_true"].to_numpy(), ref["prob"].to_numpy()
                seen_text = _ci_text(bootstrap_ci(yi, (pi,), auroc, n_boot))
            rows.append(
                {
                    "held-out type": t,
                    "method": method,
                    "runs": len(recs),
                    "unseen AUROC": _ci_text(u["auroc"]),
                    "seen AUROC (5-fold OOF)": seen_text,
                    "unseen AUPRC": f"{u['auprc']:.3f}",
                    "unseen sens@95%spec": f"{u['sens_at_95_spec']:.3f}",
                    "images": u["images"],
                }
            )
    return pd.DataFrame(rows)


def e09_summaries(
    severity: pd.DataFrame, final: str, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean corruption AUROC per configuration, and the final model's quality limits.

    Args:
        severity: Output of ``robustness.severity_table``.
        final: Configuration name of the final model (ADR-020).
        threshold: Quality-limit AUROC (ADR-023).
    """
    corrupted = severity[severity["corruption"] != CLEAN[0]]
    clean = severity[severity["corruption"] == CLEAN[0]].set_index("config")["auroc"]
    mean = (
        corrupted.groupby("config")["auroc"]
        .agg(["mean", "min", "count"])
        .rename(
            columns={
                "mean": "mean corruption AUROC",
                "min": "worst condition AUROC",
                "count": "conditions",
            }
        )
        .join(clean.rename("clean AUROC"))
        .reset_index()
    )
    return mean, quality_limits(severity, final, threshold)
