"""OOD scores and detection metrics (PLAN 5.5, E11). Every score: higher = more OOD.

Output-based scores use the binary logit z; feature-based scores use penultimate embeddings
and are fitted on the fold model's training images only (Rule 3).
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from pharmachecked.evaluation.metrics import ArrayLike, auroc

ACCEPT_ID = 0.95  # guard threshold: accept 95% of in-distribution val images


def msp_score(logits: ArrayLike) -> np.ndarray:
    """``-max(p, 1 - p)`` with p = sigmoid(z)."""
    z = np.asarray(logits, dtype=float)
    p = 1.0 / (1.0 + np.exp(-z))
    return -np.maximum(p, 1.0 - p)


def energy_score(logits: ArrayLike) -> np.ndarray:
    """Energy (T = 1) of the class logits (-z/2, z/2): ``-log(e^(-z/2) + e^(z/2))``.

    Monotone in -|z|, so for a binary model it ranks images like ``msp_score``.
    """
    z = np.asarray(logits, dtype=float)
    return -np.logaddexp(-z / 2.0, z / 2.0)


class Mahalanobis:
    """Class-conditional means, shared Ledoit-Wolf covariance; score = min squared distance."""

    def fit(self, features: ArrayLike, labels: ArrayLike) -> Mahalanobis:
        """Fit on training embeddings and labels.

        Raises:
            ValueError: Unless both classes are present.
        """
        x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Mahalanobis needs training embeddings of both classes")
        self.means = np.stack([x[y == c].mean(axis=0) for c in (0, 1)])
        centred = x - self.means[y]
        self.precision = LedoitWolf(assume_centered=True).fit(centred).precision_
        return self

    def score(self, features: ArrayLike) -> np.ndarray:
        """Minimum squared Mahalanobis distance to a class mean."""
        x = np.asarray(features, dtype=float)
        d = [np.einsum("ij,jk,ik->i", x - m, self.precision, x - m) for m in self.means]
        return np.minimum(*d)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class KNN:
    """Distance to the k-th nearest L2-normalized training embedding (Sun et al., 2022)."""

    def __init__(self, k: int = 50) -> None:
        self.k = k

    def fit(self, features: ArrayLike) -> KNN:
        """Store the normalized training embeddings.

        Raises:
            ValueError: With fewer than k training embeddings.
        """
        x = np.asarray(features, dtype=float)
        if len(x) < self.k:
            raise ValueError(f"kNN needs at least k={self.k} training embeddings, got {len(x)}")
        self.bank = _normalize(x)
        return self

    def score(self, features: ArrayLike, batch: int = 1024) -> np.ndarray:
        """Euclidean distance (on the unit sphere) to the k-th nearest training embedding."""
        x = _normalize(np.asarray(features, dtype=float))
        out = []
        for start in range(0, len(x), batch):
            sim = x[start : start + batch] @ self.bank.T
            kth = np.partition(-sim, self.k - 1, axis=1)[:, self.k - 1]
            out.append(np.sqrt(np.maximum(2.0 + 2.0 * kth, 0.0)))  # |a-b|^2 = 2 - 2 cos
        return np.concatenate(out) if out else np.zeros(0)


def threshold_at_acceptance(id_scores: ArrayLike, accept: float = ACCEPT_ID) -> float:
    """Score threshold accepting (score <= threshold) a fraction ``accept`` of ID images."""
    return float(np.quantile(np.asarray(id_scores, dtype=float), accept, method="higher"))


def detection_auroc(id_scores: ArrayLike, ood_scores: ArrayLike) -> float:
    """AUROC of separating OOD (positive) from ID images by the score."""
    s_id, s_ood = np.asarray(id_scores, dtype=float), np.asarray(ood_scores, dtype=float)
    y = np.r_[np.zeros(len(s_id), dtype=int), np.ones(len(s_ood), dtype=int)]
    return auroc(y, np.r_[s_id, s_ood])


def fpr_at_95_tpr(id_scores: ArrayLike, ood_scores: ArrayLike) -> float:
    """Fraction of OOD images accepted when the threshold accepts 95% of the ID images."""
    t = threshold_at_acceptance(id_scores, ACCEPT_ID)
    return float(np.mean(np.asarray(ood_scores, dtype=float) <= t))
