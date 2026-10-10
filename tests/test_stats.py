"""DeLong, bootstrap, McNemar and Holm behave as specified."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from pharmachecked.evaluation.metrics import auroc
from pharmachecked.evaluation.stats import (
    bootstrap_ci,
    delong_covariance,
    delong_test,
    holm,
    mcnemar_exact,
)


def _synthetic(n: int = 400, shift: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n)
    s = rng.normal(size=2 * n) + shift * y
    return y, s


def test_delong_auc_matches_sklearn() -> None:
    y, s = _synthetic()
    aucs, _ = delong_covariance(y, s)
    assert aucs[0] == pytest.approx(roc_auc_score(y, s))


def test_delong_identical_predictions_p_is_one() -> None:
    y, s = _synthetic()
    assert delong_test(y, s, s).p_value == 1.0


def test_delong_detects_a_real_difference() -> None:
    y, s = _synthetic(shift=1.5)
    rng = np.random.default_rng(5)
    weak = s + rng.normal(scale=3.0, size=len(s))
    assert delong_test(y, s, weak).p_value < 1e-6


def test_delong_variance_agrees_with_bootstrap() -> None:
    y, s = _synthetic(n=500)
    _, cov = delong_covariance(y, s)
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(4000):
        idx = np.concatenate([rng.choice(neg, len(neg)), rng.choice(pos, len(pos))])
        boots.append(roc_auc_score(y[idx], s[idx]))
    assert float(cov[0, 0]) == pytest.approx(np.var(boots), rel=0.10)


def test_bootstrap_ci_contains_estimate_and_is_reproducible() -> None:
    y, s = _synthetic(n=150)
    a = bootstrap_ci(y, s, auroc, n_boot=500)
    b = bootstrap_ci(y, s, auroc, n_boot=500)
    assert a == b
    assert a.low <= a.estimate <= a.high


def test_bootstrap_paired_difference() -> None:
    y, s = _synthetic(n=150)
    ci = bootstrap_ci(y, (s, s), lambda yy, p1, p2: auroc(yy, p1) - auroc(yy, p2), n_boot=200)
    assert ci.estimate == ci.low == ci.high == 0.0


def test_mcnemar_exact() -> None:
    y = np.zeros(20, dtype=int)
    a = np.zeros(20, dtype=int)  # always right
    b = np.zeros(20, dtype=int)
    b[:10] = 1  # wrong on 10 where a is right
    assert mcnemar_exact(y, a, b) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(y, a, a) == 1.0


def _holm_by_hand(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def test_holm_matches_statsmodels_and_hand_computation() -> None:
    p = [0.01, 0.04, 0.03, 0.2, 0.005]
    reject, corrected = holm(p)
    assert corrected == pytest.approx(multipletests(p, method="holm")[1].tolist())
    assert corrected == pytest.approx(_holm_by_hand(p))
    assert reject == [c < 0.05 for c in corrected]
