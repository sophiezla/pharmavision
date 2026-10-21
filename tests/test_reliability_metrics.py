"""Temperature scaling, selective prediction and OOD scores (PLAN 5.1, 5.2, 5.5)."""

import numpy as np
import pytest

from pharmachecked.evaluation.calibration import (
    apply_temperature,
    fit_temperature,
    logit_from_prob,
    reliability_bins,
    temperature_identifiable,
)
from pharmachecked.evaluation.metrics import expected_calibration_error, log_loss
from pharmachecked.evaluation.ood import (
    KNN,
    Mahalanobis,
    detection_auroc,
    energy_score,
    fpr_at_95_tpr,
    msp_score,
    threshold_at_acceptance,
)
from pharmachecked.evaluation.selective import aurc, risk_coverage


def test_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    t = fit_temperature(3.0 * true_logit, y)  # logits 3x too confident
    assert t == pytest.approx(3.0, rel=0.1)
    before = log_loss(y, apply_temperature(3.0 * true_logit, 1.0))
    after = log_loss(y, apply_temperature(3.0 * true_logit, t))
    assert after < before


def test_temperature_identifiable_only_with_a_misclassified_image() -> None:
    y = np.array([0, 0, 1, 1])
    assert not temperature_identifiable(np.array([-3.0, -1.0, 2.0, 5.0]), y)
    assert temperature_identifiable(np.array([-3.0, 0.5, 2.0, 5.0]), y)
    assert temperature_identifiable(np.array([-3.0, -1.0, 0.0, 5.0]), y)  # on the boundary


def test_temperature_inputs() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros(0), np.zeros(0))
    np.testing.assert_allclose(apply_temperature(logit_from_prob([0.2, 0.7]), 1.0), [0.2, 0.7])


def test_reliability_bins_match_ece() -> None:
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p**2).astype(int)
    bins = reliability_bins(y, p).dropna()
    ece = (bins["count"] / 500 * (bins["accuracy"] - bins["confidence"]).abs()).sum()
    assert ece == pytest.approx(expected_calibration_error(y, p))
    assert reliability_bins(y, p)["count"].sum() == 500


def test_risk_coverage_and_aurc() -> None:
    y = np.array([1, 1, 0, 0])
    perfect = np.array([0.99, 0.9, 0.1, 0.4])  # all correct
    assert aurc(y, perfect) == 0.0
    wrong_last = np.array([0.99, 0.9, 0.05, 0.55])  # least confident image is the error
    curve = risk_coverage(y, wrong_last)
    assert list(curve["coverage"]) == [0.25, 0.5, 0.75, 1.0]
    assert curve["risk"].iloc[-1] == 0.25
    assert aurc(y, wrong_last) == pytest.approx((0 + 0 + 0 + 0.25) / 4)
    constant = np.full(4, 0.6)  # one block: every image gets the overall error rate
    assert aurc(y, constant) == pytest.approx(0.5)


def test_output_scores() -> None:
    z = np.array([-6.0, -1.0, 0.0, 1.0, 6.0])
    msp, energy = msp_score(z), energy_score(z)
    assert msp[2] == msp.max() and energy[2] == energy.max()  # z = 0 is most OOD
    np.testing.assert_allclose(energy[:2], energy[:-3:-1])  # symmetric in z
    order = np.argsort(msp)
    assert np.all(np.diff(energy[order]) >= -1e-12)  # same ranking as MSP


def test_feature_scores_rank_far_points_higher() -> None:
    rng = np.random.default_rng(2)
    # embeddings away from the origin, as real penultimate features are; kNN compares directions
    centre = np.full(8, 5.0)
    train = np.r_[rng.normal(0, 1, (100, 8)), rng.normal(1, 1, (100, 8))] + centre
    labels = np.r_[np.zeros(100, int), np.ones(100, int)]
    near = rng.normal(0, 1, (50, 8)) + centre
    far = rng.normal(0, 1, (50, 8)) + centre * np.r_[1, -1, 1, -1, 1, -1, 1, -1]
    maha = Mahalanobis().fit(train, labels)
    assert detection_auroc(maha.score(near), maha.score(far)) > 0.99
    knn = KNN(k=50).fit(train)
    assert detection_auroc(knn.score(near), knn.score(far)) > 0.95
    with pytest.raises(ValueError):
        KNN(k=50).fit(train[:10])
    with pytest.raises(ValueError):
        Mahalanobis().fit(train, np.zeros(200, int))


def test_threshold_and_fpr() -> None:
    id_scores = np.arange(100, dtype=float)
    t = threshold_at_acceptance(id_scores)
    assert np.mean(id_scores <= t) >= 0.95
    assert fpr_at_95_tpr(id_scores, np.array([0.0, 50.0, 200.0, 300.0])) == 0.5
