"""Phase 4 tables: independent-sample AUROC difference, E8 open-set table, E9 summaries."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmachecked.evaluation.aggregate import RunRecord
from pharmachecked.evaluation.phase4 import difference_ci, e08_table, e09_summaries


def test_difference_ci_covers_the_estimate() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    strong = y + rng.normal(0, 0.3, 100)
    weak = y + rng.normal(0, 1.5, 100)
    ci = difference_ci(y, strong, y, weak, n_boot=200)
    assert ci.low <= ci.estimate <= ci.high and ci.estimate > 0


def _run(tmp: Path, name: str, model: str, variant: str, seed: int, df: pd.DataFrame) -> RunRecord:
    root = tmp / name
    root.mkdir()
    df.assign(seed=seed).to_csv(root / "oof_predictions.csv", index=False)
    return RunRecord(root, model, variant, 0, seed, {}, 0.5)


def _oof(types: list[str], noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (len(types) // 2))
    return pd.DataFrame(
        {
            "image_id": [f"i{k}" for k in range(len(types))],
            "fold": 0,
            "y_true": y,
            "logit": 0.0,
            "prob": np.clip(y + rng.normal(0, noise, len(y)), 0, 1),
            "packaging_type": types,
            "source_dataset": "kaggle_fake_real",
        }
    )


def test_e08_table_pairs_unseen_with_the_seen_subset(tmp_path: Path) -> None:
    types = ["blister", "blister", "packaging", "packaging"] * 10  # both classes per type
    ref = [_run(tmp_path, f"r{s}", "convnext-tiny", "ft", s, _oof(types, 0.2, s)) for s in (0, 1)]
    blister = [t for t in types if t == "blister"]
    unseen = [
        _run(tmp_path, f"u{s}", "convnext-tiny", "ft-no-blister", s, _oof(blister, 0.6, 10 + s))
        for s in (0, 1)
    ]
    one_class = [_run(tmp_path, "pc", "patchcore", "oc-no-blister", 0, _oof(blister, 0.8, 20))]
    table = e08_table(unseen + one_class, ref, ["blister", "packaging"], n_boot=100)
    assert list(table["method"]) == ["convnext-tiny/ft", "patchcore/oc"]
    row = table.set_index("method").loc["convnext-tiny/ft"]
    assert row["images"] == 20 and row["runs"] == 2
    assert row["seen AUROC (5-fold OOF)"] != "-"
    assert table.set_index("method").loc["patchcore/oc", "seen AUROC (5-fold OOF)"] == "-"


def test_e09_summaries() -> None:
    severity = pd.DataFrame(
        {
            "config": ["m"] * 4,
            "corruption": ["clean", "jpeg", "jpeg", "glare"],
            "severity": [0, 1, 2, 1],
            "auroc": [0.99, 0.9, 0.8, 0.95],
        }
    )
    mean, limits = e09_summaries(severity, "m", 0.85)
    row = mean.set_index("config").loc["m"]
    assert row["mean corruption AUROC"] == pytest.approx((0.9 + 0.8 + 0.95) / 3)
    assert row["clean AUROC"] == 0.99 and row["conditions"] == 3
    assert limits.set_index("corruption")["max_severity"].to_dict() == {"glare": 1, "jpeg": 1}
