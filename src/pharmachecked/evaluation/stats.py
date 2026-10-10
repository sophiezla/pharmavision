"""Statistics: bootstrap CIs, fast DeLong, exact McNemar, Holm correction (CLAUDE.md section 5)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ArrayLike = Any
N_BOOT = 2000


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile bootstrap confidence interval."""

    estimate: float
    low: float
    high: float


def _resample_indices(y: np.ndarray, rng: np.random.Generator, stratified: bool) -> np.ndarray:
    if not stratified:
        return rng.integers(0, len(y), len(y))
    parts = [rng.choice(np.flatnonzero(y == c), size=int((y == c).sum())) for c in np.unique(y)]
    return np.concatenate(parts)


def bootstrap_ci(
    y_true: ArrayLike,
    scores: ArrayLike | Sequence[ArrayLike],
    metric: Callable[..., float],
    n_boot: int = N_BOOT,
    seed: int = 0,
    level: float = 0.95,
    stratified: bool = True,
) -> Interval:
    """Percentile bootstrap CI of ``metric(y, *scores)`` with class-stratified resampling.

    Args:
        y_true: Labels.
        scores: One score array, or a tuple of aligned score arrays passed to ``metric``
            (e.g. two models' probabilities for a paired difference).
        metric: Function of ``(y, *scores)``.
        n_boot: Resamples (protocol: 2,000).
        seed: RNG seed (protocol: 0).
        level: Confidence level.
        stratified: Resample within each class (protocol default).

    Returns:
        The point estimate on the full data and the percentile interval.
    """
    y = np.asarray(y_true, dtype=int)
    arrays = [
        np.asarray(s, dtype=float) for s in (scores if isinstance(scores, tuple) else (scores,))
    ]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = _resample_indices(y, rng, stratified)
        stats[b] = metric(y[idx], *(a[idx] for a in arrays))
    alpha = (1.0 - level) / 2.0
    low, high = np.nanquantile(stats, [alpha, 1.0 - alpha])
    return Interval(float(metric(y, *arrays)), float(low), float(high))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_covariance(y_true: ArrayLike, *scores: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their DeLong covariance matrix (fast algorithm of Sun & Xu, 2014).

    Args:
        y_true: Binary labels.
        *scores: One or more score arrays aligned with ``y_true``.

    Returns:
        ``(aucs, cov)`` with shapes ``(k,)`` and ``(k, k)``.
    """
    y = np.asarray(y_true, dtype=int)
    preds = np.vstack([np.asarray(s, dtype=float) for s in scores])
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("DeLong needs at least two samples of each class")
    tx = np.vstack([_midrank(r) for r in pos])
    ty = np.vstack([_midrank(r) for r in neg])
    tz = np.vstack([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


@dataclass(frozen=True)
class DeLongResult:
    """Paired DeLong test of two AUCs on the same samples."""

    auc_a: float
    auc_b: float
    difference: float
    z: float
    p_value: float


def delong_test(y_true: ArrayLike, score_a: ArrayLike, score_b: ArrayLike) -> DeLongResult:
    """Two-sided paired DeLong test (DeLong et al., 1988). p = 1 when the scores coincide."""
    aucs, cov = delong_covariance(y_true, score_a, score_b)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 1e-15:
        return DeLongResult(float(aucs[0]), float(aucs[1]), diff, 0.0, 1.0)
    z = diff / np.sqrt(var)
    return DeLongResult(
        float(aucs[0]), float(aucs[1]), diff, float(z), float(2.0 * norm.sf(abs(z)))
    )


def mcnemar_exact(y_true: ArrayLike, pred_a: ArrayLike, pred_b: ArrayLike) -> float:
    """Exact McNemar p-value for paired errors of two hard classifiers (statsmodels)."""
    y = np.asarray(y_true, dtype=int)
    ok_a = np.asarray(pred_a, dtype=int) == y
    ok_b = np.asarray(pred_b, dtype=int) == y
    table = [
        [int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
        [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())],
    ]
    return float(mcnemar(table, exact=True).pvalue)


def holm(p_values: Sequence[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni correction within one experiment (statsmodels ``method="holm"``).

    Returns:
        ``(reject, corrected_p)`` in input order.
    """
    reject, corrected, _, _ = multipletests(list(p_values), alpha=alpha, method="holm")
    return [bool(r) for r in reject], [float(p) for p in corrected]
