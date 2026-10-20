"""Prototype verifier for the open-set study (PLAN 4.6, E8).

One-class: it sees only authentic training images. Each image is embedded by a frozen DINOv2
backbone and L2-normalized; the counterfeit score of a query is the mean cosine distance to its
k nearest authentic training embeddings (a kNN novelty score, as in deep nearest-neighbour
anomaly detection; Bergman, Cohen & Hoshen, 2020, "Deep Nearest Neighbor Anomaly Detection").
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rows of ``x`` scaled to unit L2 norm."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


class PrototypeVerifier:
    """kNN cosine-distance scorer over authentic reference embeddings."""

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.reference: np.ndarray | None = None

    def fit(self, authentic: np.ndarray) -> PrototypeVerifier:
        """Store the L2-normalized authentic embeddings ``(n, d)``.

        Raises:
            ValueError: If fewer than ``k`` reference embeddings are given.
        """
        if len(authentic) < self.k:
            raise ValueError(f"{len(authentic)} reference embeddings for k = {self.k}")
        self.reference = l2_normalize(np.asarray(authentic, dtype=np.float32))
        return self

    def score(self, queries: np.ndarray) -> np.ndarray:
        """Mean cosine distance (0-2) of each query to its ``k`` nearest references.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.reference is None:
            raise RuntimeError("fit the verifier first")
        sims = l2_normalize(np.asarray(queries, dtype=np.float32)) @ self.reference.T
        top = -np.sort(-sims, axis=1)[:, : self.k]
        return (1.0 - top).mean(axis=1)
