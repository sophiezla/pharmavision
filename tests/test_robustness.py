"""E9 robustness: conditions, pooled severity table, quality limits, end-to-end on toy runs."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from pharmachecked.data.corruptions import CORRUPTIONS, load_corruption_config
from pharmachecked.evaluation.robustness import (
    CORRUPTION_FILE,
    conditions,
    corruption_predictions,
    quality_limits,
    severity_table,
)

CFG = load_corruption_config()


def test_conditions_are_clean_plus_every_level() -> None:
    conds = conditions(CFG)
    assert conds[0] == ("clean", 0)
    assert len(conds) == 1 + 5 * len(CORRUPTIONS)


def _frame(seed: int, noise: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array([0] * 20 + [1] * 20)
    rows = []
    for name, s in [("clean", 0), ("jpeg", 1), ("jpeg", 2)]:
        prob = np.clip(y + rng.normal(0, noise * (1 + s), len(y)), 0, 1)
        rows.append(
            pd.DataFrame(
                {
                    "image_id": [f"i{k}" for k in range(40)],
                    "fold": 0,
                    "seed": seed,
                    "y_true": y,
                    "prob": prob,
                    "corruption": name,
                    "severity": s,
                }
            )
        )
    return pd.concat(rows)


def test_severity_table_pools_seeds_and_degrades() -> None:
    table = severity_table({"m": [_frame(0, 0.3), _frame(1, 0.3)]}, n_boot=50)
    assert set(table["images"]) == {40}
    by = table.set_index(["corruption", "severity"])["auroc"]
    assert by[("clean", 0)] >= by[("jpeg", 2)]
    assert (table["ci_low"] <= table["auroc"]).all() and (table["auroc"] <= table["ci_high"]).all()


def test_quality_limit_is_conservative() -> None:
    table = pd.DataFrame(
        {
            "config": "m",
            "corruption": ["clean"] + ["blur"] * 5 + ["jpeg"] * 5,
            "severity": [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "auroc": [0.99, 0.95, 0.9, 0.8, 0.86, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9],
        }
    )
    limits = quality_limits(table, "m", 0.85).set_index("corruption")["max_severity"]
    assert limits["blur"] == 2  # severity 4 recovers to 0.86 but 3 already failed
    assert limits["jpeg"] == 0
    assert "clean" not in limits


def _train(tmp_path: Path, *extra: str) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pharmachecked.run",
            "experiment=smoke",
            "data.dataset=toy",
            "tracker.enabled=false",
            f"paths.results_dir={tmp_path.as_posix()}",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return max((tmp_path / "scratch" / "runs").glob("smoke_*"), key=lambda p: p.stat().st_mtime)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("extra", "ckpt"),
    [((), "best.pt"), (("model=handcrafted_xgb", "task=handcrafted"), "model.joblib")],
)
def test_corruption_predictions_on_toy_runs(tmp_path: Path, extra: tuple, ckpt: str) -> None:
    run = _train(tmp_path, *extra)
    cfg = OmegaConf.load(run / "config.yaml")
    checkpoint = Path(cfg.paths.checkpoints_dir) / run.name / ckpt
    out = corruption_predictions(run, cfg.paths, checkpoint, CFG, "cpu")
    oof = pd.read_csv(run / "oof_predictions.csv")
    assert (run / CORRUPTION_FILE).is_file()
    assert len(out) == len(oof) * len(conditions(CFG))
    clean = out[out["corruption"] == "clean"].set_index("image_id")["prob"]
    if ckpt == "best.pt":  # the clean condition is the ordinary evaluation input
        assert np.allclose(clean.loc[oof["image_id"]].to_numpy(), oof["prob"].to_numpy(), atol=1e-5)
