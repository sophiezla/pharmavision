"""PLAN 8.3: re-fitted reliability of exported models uses only that run's val/cal rows."""

import numpy as np
import pandas as pd

from pharmachecked.evaluation.phase6to8 import e16_refit_runs


def _preds(rng: np.random.Generator, held_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for split, n in (("val", 80), ("cal", 200), ("heldout", 200)):
        y = rng.integers(0, 2, n)
        logit = (
            (2 * y - 1) * 1.5 + rng.normal(0, 1.5, n) + (held_shift if split == "heldout" else 0)
        )
        for i in range(n):
            rows.append(
                {
                    "image_id": f"{split}{i}",
                    "split": split,
                    "precision": "fp32",
                    "y_true": int(y[i]),
                    "logit": float(logit[i]),
                    "run": "r0",
                }
            )
    df = pd.DataFrame(rows)
    df["prob"] = 1 / (1 + np.exp(-df["logit"]))
    return df


def test_refit_coverage_near_target() -> None:
    covs = [e16_refit_runs(_preds(np.random.default_rng(s)))["coverage"].item() for s in range(30)]
    assert abs(np.mean(covs) - 0.90) < 0.03


def test_heldout_does_not_change_fitted_temperature() -> None:
    a = e16_refit_runs(_preds(np.random.default_rng(0)))
    b = e16_refit_runs(_preds(np.random.default_rng(0), held_shift=5.0))
    assert a["temperature"].item() == b["temperature"].item()


def test_repair_localization_restores_values_and_dedups() -> None:
    from pharmachecked.evaluation.phase6to8 import repair_localization

    broken = "image_id\nab12    font_swap\nab12    font_swap\nName: edit_type, dtype: str"
    df = pd.DataFrame(
        {
            "run": ["r", "r", "r"],
            "method": ["m", "m", "m"],
            "image_id": ["ab12", "ab12", "cd"],
            "generator": ["procedural"] * 3,
            "edit_type": [broken, broken, "local_jpeg"],
            "hit": [1, 1, 0],
            "iou": [0.5, 0.5, 0.1],
        }
    )
    out = repair_localization(df)
    assert list(out["edit_type"]) == ["font_swap", "local_jpeg"]
