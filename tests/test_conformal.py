"""Split conformal prediction (PLAN 5.3 acceptance test)."""

import numpy as np
import pytest

from pharmachecked.evaluation.conformal import (
    calibrate,
    class_scores,
    conformal_quantile,
    evaluate,
    prediction_sets,
    set_metrics,
)


def _calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from calibrated probabilities, so the probabilities are exchangeable."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.7, 0.7, n)
    return (rng.random(n) < p).astype(int), p


def test_scores() -> None:
    p = np.array([0.9, 0.2, 0.5])
    np.testing.assert_allclose(class_scores(p, "lac"), [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    # APS: p_y for the more likely class, 1 for the less likely one; both 1 at a tie
    np.testing.assert_allclose(class_scores(p, "aps"), [[1.0, 0.9], [0.8, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        class_scores(p, "raps")


def test_quantile_rank_and_small_n() -> None:
    s = np.arange(1, 20) / 20  # n = 19
    assert conformal_quantile(s, 0.10) == pytest.approx(s[17])  # ceil(20 * 0.9) = 18
    assert conformal_quantile(s, 0.05) == pytest.approx(s[18])  # ceil(20 * 0.95) = 19
    assert conformal_quantile(s[:10], 0.05) == float("inf")  # ceil(11 * 0.95) = 11 > 10


@pytest.mark.parametrize("score", ["lac", "aps"])
@pytest.mark.parametrize("mode", ["marginal", "mondrian"])
@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_coverage_guarantee(score: str, mode: str, alpha: float) -> None:
    """Average coverage over many calibration/test draws is at least 1 - alpha."""
    covers, per_class = [], {0: [], 1: []}
    for rep in range(200):
        y_cal, p_cal = _calibrated(60, rep)
        y_te, p_te = _calibrated(200, 10_000 + rep)
        sets = prediction_sets(p_te, score, calibrate(y_cal, p_cal, score, mode, alpha))
        m = set_metrics(y_te, sets)
        covers.append(m["coverage"])
        per_class[0].append(m["coverage_authentic"])
        per_class[1].append(m["coverage_counterfeit"])
    assert np.mean(covers) >= 1 - alpha - 0.01
    if mode == "mondrian":  # class-conditional guarantee
        assert min(np.mean(per_class[0]), np.mean(per_class[1])) >= 1 - alpha - 0.01


def test_set_metrics_counts() -> None:
    sets = np.array([[True, False], [True, True], [False, False], [False, True]])
    m = set_metrics(np.array([0, 1, 0, 0]), sets)
    assert m["coverage"] == 0.5 and m["coverage_authentic"] == pytest.approx(1 / 3)
    assert m["coverage_counterfeit"] == 1.0
    assert m["singleton_rate"] == 0.5 and m["abstention_rate"] == 0.25
    assert m["empty_rate"] == 0.25 and m["mean_set_size"] == 1.0


def test_evaluate_grid() -> None:
    y, p = _calibrated(80, 1)
    table = evaluate(y[:40], p[:40], y[40:], p[40:])
    assert len(table) == 8 and set(table["alpha"]) == {0.05, 0.10}
    with pytest.raises(ValueError):
        calibrate(y, p, "lac", "per-type", 0.1)
