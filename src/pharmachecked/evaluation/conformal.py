"""Split conformal prediction for the binary task (PLAN 5.3, E10).

Scores (higher = less conforming), for a calibrated counterfeit probability p and class y:
- LAC: 1 - p_y.
- APS (non-randomized): total probability of the classes at least as likely as y; for two
  classes that is p_y when y is the more likely class and 1 otherwise.

Marginal conformal uses one quantile over all calibration images; class-conditional (Mondrian)
one quantile per true class, computed on that class's calibration images. A label enters the
prediction set when its score is at most the (class's) quantile. Sets of size 0 or 2 are
abstentions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pharmachecked.evaluation.metrics import ArrayLike, _arrays

SCORES = ("lac", "aps")
MODES = ("marginal", "mondrian")


def class_scores(prob: ArrayLike, score: str) -> np.ndarray:
    """Nonconformity score of each class for each image: shape (n, 2), column = class.

    Raises:
        ValueError: For an unknown score.
    """
    p1 = np.asarray(prob, dtype=float)
    probs = np.stack([1.0 - p1, p1], axis=1)
    if score == "lac":
        return 1.0 - probs
    if score == "aps":
        other = probs[:, ::-1]
        # ties count as "at least as likely", so both classes get score 1 at p = 0.5
        return np.where(probs > other, probs, 1.0)
    raise ValueError(f"unknown conformal score {score!r}; known: {SCORES}")


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; +inf when that rank exceeds n.

    An infinite quantile admits every label (the finite-sample guarantee cannot be met with so
    few calibration scores).
    """
    s = np.sort(np.asarray(scores, dtype=float))
    rank = math.ceil((len(s) + 1) * (1.0 - alpha))
    return float(s[rank - 1]) if rank <= len(s) else float("inf")


def calibrate(
    y_cal: ArrayLike, p_cal: ArrayLike, score: str, mode: str, alpha: float
) -> np.ndarray:
    """Quantile per class (both equal for ``marginal``), shape (2,).

    Raises:
        ValueError: For an unknown mode.
    """
    y, p = _arrays(y_cal, p_cal)
    true_scores = class_scores(p, score)[np.arange(len(y)), y]
    if mode == "marginal":
        return np.full(2, conformal_quantile(true_scores, alpha))
    if mode == "mondrian":
        return np.array([conformal_quantile(true_scores[y == c], alpha) for c in (0, 1)])
    raise ValueError(f"unknown conformal mode {mode!r}; known: {MODES}")


def prediction_sets(prob: ArrayLike, score: str, quantiles: np.ndarray) -> np.ndarray:
    """Boolean (n, 2) membership: class c is in the set when its score <= quantiles[c]."""
    return class_scores(prob, score) <= quantiles[None, :]


def set_metrics(y_true: ArrayLike, sets: np.ndarray) -> dict[str, float]:
    """Coverage (overall and per class), mean set size, singleton/abstention/empty rates."""
    y = np.asarray(y_true, dtype=int)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    return {
        "coverage": float(covered.mean()),
        "coverage_authentic": float(covered[y == 0].mean()) if (y == 0).any() else math.nan,
        "coverage_counterfeit": float(covered[y == 1].mean()) if (y == 1).any() else math.nan,
        "mean_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "abstention_rate": float((size == 2).mean()),
        "empty_rate": float((size == 0).mean()),
    }


def evaluate(
    y_cal: ArrayLike,
    p_cal: ArrayLike,
    y_test: ArrayLike,
    p_test: ArrayLike,
    alphas: tuple[float, ...] = (0.05, 0.10),
) -> pd.DataFrame:
    """Every score x mode x alpha on one calibration/test split: one row each."""
    rows = []
    for score in SCORES:
        for mode in MODES:
            for alpha in alphas:
                q = calibrate(y_cal, p_cal, score, mode, alpha)
                sets = prediction_sets(p_test, score, q)
                rows.append(
                    {"score": score, "mode": mode, "alpha": alpha} | set_metrics(y_test, sets)
                )
    return pd.DataFrame(rows)
