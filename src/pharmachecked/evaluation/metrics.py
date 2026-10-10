"""Classification and calibration metrics (CLAUDE.md section 5); positive class = counterfeit."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

ArrayLike = Any


def _arrays(y_true: ArrayLike, prob: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and prob must have the same shape")
    return y, p


def auroc(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Area under the ROC curve (NaN when only one class is present)."""
    y, p = _arrays(y_true, prob)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def auprc(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Average precision for the counterfeit class (NaN without positives)."""
    y, p = _arrays(y_true, prob)
    return float(average_precision_score(y, p)) if y.any() else float("nan")


def sensitivity_at_specificity(
    y_true: ArrayLike, prob: ArrayLike, specificity: float = 0.95
) -> float:
    """Highest sensitivity reachable while specificity stays at or above the target.

    Read off the ROC curve: max TPR over operating points with FPR <= 1 - specificity.
    """
    y, p = _arrays(y_true, prob)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1.0 - specificity + 1e-12
    return float(tpr[ok].max())


def youden_threshold(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Threshold maximizing sensitivity + specificity - 1 (0.5 when one class is present)."""
    y, p = _arrays(y_true, prob)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def expected_calibration_error(y_true: ArrayLike, prob: ArrayLike, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins (Guo et al., 2017).

    Confidence = max(p, 1 - p); a prediction is correct when (p >= 0.5) equals the label.
    ECE = sum over bins of (bin size / n) * |accuracy - mean confidence|.
    """
    y, p = _arrays(y_true, prob)
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)  # top-label confidence is in [0.5, 1]
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Mean squared error of the counterfeit probability."""
    y, p = _arrays(y_true, prob)
    return float(np.mean((p - y) ** 2))


def threshold_metrics(y_true: ArrayLike, prob: ArrayLike, threshold: float) -> dict[str, float]:
    """Metrics of the hard decision ``prob >= threshold``."""
    y, p = _arrays(y_true, prob)
    pred = (p >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    return {
        "sensitivity": float(pred[pos].mean()) if pos.any() else float("nan"),
        "specificity": float((1 - pred[neg]).mean()) if neg.any() else float("nan"),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred))
        if pos.any() and neg.any()
        else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
    }


def binary_metrics(y_true: ArrayLike, prob: ArrayLike, threshold: float) -> dict[str, float]:
    """Every per-run metric of CLAUDE.md section 5 for a held-out set with both classes."""
    return {
        "auroc": auroc(y_true, prob),
        "auprc": auprc(y_true, prob),
        "sens_at_95_spec": sensitivity_at_specificity(y_true, prob, 0.95),
        "ece": expected_calibration_error(y_true, prob),
        "brier": brier(y_true, prob),
        "threshold": float(threshold),
        **threshold_metrics(y_true, prob, threshold),
    }


def single_class_metric(y_true: ArrayLike, prob: ArrayLike, threshold: float) -> dict[str, float]:
    """External-set metric (ADR-005): specificity (authentic-only) or recall (counterfeit-only)."""
    y, p = _arrays(y_true, prob)
    classes = set(np.unique(y).tolist())
    if classes == {0}:
        return {"specificity": float((p < threshold).mean()), "n": len(y)}
    if classes == {1}:
        return {"recall": float((p >= threshold).mean()), "n": len(y)}
    raise ValueError("expected a single-class set")
