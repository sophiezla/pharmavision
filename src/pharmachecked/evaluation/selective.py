"""Selective prediction: risk-coverage curves and AURC (PLAN 5.2, E10).

Predictions are made at 0.5 on calibrated probabilities; confidence = max(p, 1 - p). Images are
accepted from the most to the least confident; ties in confidence enter together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmachecked.evaluation.metrics import ArrayLike, _arrays


def risk_coverage(y_true: ArrayLike, prob: ArrayLike) -> pd.DataFrame:
    """Selective risk at every coverage level reachable by a confidence threshold.

    Returns:
        Columns ``coverage, risk, threshold`` sorted by coverage (ascending); ``risk`` is the
        error rate among accepted images, ``threshold`` the lowest accepted confidence.
    """
    y, p = _arrays(y_true, prob)
    conf = np.maximum(p, 1.0 - p)
    error = ((p >= 0.5).astype(int) != y).astype(float)
    order = np.argsort(-conf, kind="stable")
    conf, error = conf[order], error[order]
    cum_err = np.cumsum(error)
    n = np.arange(1, len(y) + 1)
    last = np.r_[conf[1:] != conf[:-1], True]  # end of each block of equal confidence
    return pd.DataFrame(
        {"coverage": n[last] / len(y), "risk": cum_err[last] / n[last], "threshold": conf[last]}
    )


def aurc(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Area under the risk-coverage curve (Geifman et al., 2019), step-wise per image.

    Every image contributes the selective risk at the coverage where it is accepted (with
    ties sharing their block's risk), averaged over images: 0 for a perfect ranking without
    errors, the error rate for a constant confidence.
    """
    curve = risk_coverage(y_true, prob)
    n = len(np.asarray(y_true))
    if n == 0:
        return float("nan")
    counts = np.diff(np.r_[0.0, curve["coverage"].to_numpy() * n])
    return float(np.sum(counts * curve["risk"].to_numpy()) / n)
