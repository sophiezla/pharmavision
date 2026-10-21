"""Temperature scaling and reliability-diagram data (PLAN 5.1, E10).

One temperature per fold model, fitted on its ``cal`` logits only (Rule 3), then applied to its
held-out logits. ECE, Brier and log-loss come from ``evaluation.metrics``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pharmachecked.evaluation.metrics import ArrayLike, _arrays


def logit_from_prob(prob: ArrayLike, eps: float = 1e-7) -> np.ndarray:
    """``log(p / (1 - p))`` with p clipped to [eps, 1 - eps] (for the handcrafted model)."""
    p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def temperature_identifiable(logits: ArrayLike, y_true: ArrayLike) -> bool:
    """Whether the NLL over T has a finite minimizer: some image is on the wrong side of 0.

    When every logit has the correct sign (z > 0 for counterfeit, z < 0 for authentic), the NLL
    of ``sigmoid(z / T)`` decreases monotonically as T -> 0, so no temperature is optimal and
    any fitted value only reflects where the optimizer stopped (E10.md: T = 1 is used then).
    """
    y, z = _arrays(y_true, logits)
    return bool(np.any(np.where(y == 1, z <= 0, z >= 0)))


def fit_temperature(logits: ArrayLike, y_true: ArrayLike, max_iter: int = 200) -> float:
    """Temperature T > 0 minimizing the NLL of ``sigmoid(logits / T)`` (LBFGS on log T).

    Args:
        logits: Binary logits of the calibration images.
        y_true: Their labels (0 authentic, 1 counterfeit).
        max_iter: LBFGS iterations.

    Returns:
        The fitted temperature.

    Raises:
        ValueError: Without calibration images.
    """
    y, z = _arrays(y_true, logits)
    if len(z) == 0:
        raise ValueError("no calibration images")
    zt = torch.tensor(z, dtype=torch.float64)  # copies (inputs may be read-only views)
    yt = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn(zt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: ArrayLike, temperature: float) -> np.ndarray:
    """Calibrated counterfeit probability ``sigmoid(logits / T)``."""
    z = np.asarray(logits, dtype=float) / temperature
    return 1.0 / (1.0 + np.exp(-z))


def reliability_bins(y_true: ArrayLike, prob: ArrayLike, n_bins: int = 15) -> pd.DataFrame:
    """Top-label reliability-diagram data with the bins of ``expected_calibration_error``.

    Returns:
        One row per bin: ``bin, lower, upper, count, confidence, accuracy`` (NaN when empty).
    """
    y, p = _arrays(y_true, prob)
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append(
            {
                "bin": b,
                "lower": edges[b],
                "upper": edges[b + 1],
                "count": int(mask.sum()),
                "confidence": float(conf[mask].mean()) if mask.any() else float("nan"),
                "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)
