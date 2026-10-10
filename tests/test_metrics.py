"""Metrics agree with scikit-learn and behave as specified on hand-built cases."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from pharmachecked.evaluation.metrics import (
    auprc,
    auroc,
    binary_metrics,
    brier,
    expected_calibration_error,
    sensitivity_at_specificity,
    single_class_metric,
    threshold_metrics,
    youden_threshold,
)

RNG = np.random.default_rng(0)
Y = RNG.integers(0, 2, 300)
P = np.clip(Y * 0.3 + RNG.random(300) * 0.7, 0, 1)


def test_auroc_auprc_f1_match_sklearn() -> None:
    assert auroc(Y, P) == pytest.approx(roc_auc_score(Y, P))
    assert auprc(Y, P) == pytest.approx(average_precision_score(Y, P))
    pred = (P >= 0.5).astype(int)
    assert threshold_metrics(Y, P, 0.5)["macro_f1"] == pytest.approx(
        f1_score(Y, pred, average="macro")
    )


def test_sensitivity_at_95_specificity_hand_built() -> None:
    # 21 negatives: 20 scored 0.00-0.19 and one scored 0.95. Positives scored 0.5 and 0.97.
    neg = [*(np.arange(20) / 100), 0.95]
    y = [0] * len(neg) + [1, 1]
    p = [*neg, 0.5, 0.97]
    # Threshold 0.5 catches both positives with one false positive: FPR 1/21 = 0.048, so
    # specificity 0.952 >= 0.95 and sensitivity is 1.0.
    assert sensitivity_at_specificity(y, p, 0.95) == pytest.approx(1.0)
    # At 99% specificity no false positive is allowed: only the 0.97 positive is caught.
    assert sensitivity_at_specificity(y, p, 0.99) == pytest.approx(0.5)


def test_ece_near_zero_when_perfectly_calibrated() -> None:
    rng = np.random.default_rng(1)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_ece_large_when_overconfident() -> None:
    y = np.array([0, 1] * 500)
    p = np.where(y == 1, 0.01, 0.99)  # confidently wrong
    assert expected_calibration_error(y, p) > 0.9


def test_brier_and_threshold_metrics() -> None:
    assert brier([0, 1], [0.0, 1.0]) == 0.0
    assert brier([0, 1], [1.0, 0.0]) == 1.0
    m = threshold_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m["sensitivity"] == 0.5
    assert m["specificity"] == 0.5
    assert m["balanced_accuracy"] == 0.5


def test_youden_threshold_separates_perfect_scores() -> None:
    t = youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < t <= 0.8
    assert youden_threshold([1, 1], [0.3, 0.4]) == 0.5


def test_single_class_metrics() -> None:
    assert single_class_metric([0, 0, 0, 0], [0.1, 0.2, 0.7, 0.3], 0.5) == {
        "specificity": 0.75,
        "n": 4,
    }
    assert single_class_metric([1, 1], [0.9, 0.2], 0.5) == {"recall": 0.5, "n": 2}
    with pytest.raises(ValueError):
        single_class_metric([0, 1], [0.1, 0.9], 0.5)


def test_binary_metrics_keys_and_one_class_nan() -> None:
    m = binary_metrics(Y, P, 0.5)
    for key in ("auroc", "auprc", "sens_at_95_spec", "ece", "brier", "macro_f1", "threshold"):
        assert key in m
    assert np.isnan(auroc([1, 1, 1], [0.2, 0.5, 0.9]))
