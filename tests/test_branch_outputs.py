"""Branch outputs (PLAN 3.7): gate/router columns, OOF check, checkpoint search, gate table."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR
from pharmachecked.evaluation.branch_outputs import (
    BRANCH_FILE,
    branch_names,
    find_checkpoint,
    gate_table,
    max_prob_difference,
    predict_branch_outputs,
    run_branch_outputs,
)
from pharmachecked.models.registry import build_model

BASE = OmegaConf.load(CONFIG_DIR / "model" / "multibranch.yaml")
TRAIN = OmegaConf.create({"eval_batch_size": 2, "amp": False})


class _Images(torch.utils.data.Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {
            "image": torch.randn(3, 448, 448),
            "image_id": f"img{i}",
            "fold": 0,
            "label": i % 2,
            "packaging_type": ("blister", "packaging", "other")[i % 3],
            "source_dataset": "toy",
        }


@pytest.mark.parametrize(
    ("flags", "branches", "router"),
    [
        ({}, ["global", "local", "frequency"], True),
        ({"use_local": False}, ["global", "frequency"], True),
        ({"use_router": False, "use_frequency": False}, ["global", "local"], False),
    ],
)
def test_gate_and_router_columns_sum_to_one(flags: dict, branches: list, router: bool) -> None:
    cfg = OmegaConf.merge(BASE, {"backbone": {"pretrained": False}}, flags)
    assert branch_names(cfg) == branches
    out = predict_branch_outputs(
        build_model(cfg), _Images(3), torch.device("cpu"), TRAIN, 7, branches
    )
    gates = out[[f"gate_{b}" for b in branches]]
    assert len(out) == 3 and (out["seed"] == 7).all()
    assert gates.sum(axis=1).sub(1).abs().max() < 1e-5
    routers = [c for c in out.columns if c.startswith("router_")]
    assert bool(routers) == router
    if router:
        assert out[routers].sum(axis=1).sub(1).abs().max() < 1e-5


def test_max_prob_difference() -> None:
    a = pd.DataFrame({"image_id": ["x", "y"], "prob": [0.2, 0.9]})
    b = pd.DataFrame({"image_id": ["y", "x"], "prob": [0.85, 0.2]})
    assert max_prob_difference(a, b) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="different images"):
        max_prob_difference(a, b.iloc[:1])


def test_find_checkpoint_prefers_recorded_folder(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "e04_multibranch_full_f0_s0_x"
    recorded, other = tmp_path / "old_kit" / "checkpoints", tmp_path / "copy"
    assert find_checkpoint(run, str(recorded), [other]) is None
    (other / "checkpoints" / run.name).mkdir(parents=True)
    (other / "checkpoints" / run.name / "best.pt").write_bytes(b"")
    assert (
        find_checkpoint(run, str(recorded), [other]) == other / "checkpoints" / run.name / "best.pt"
    )
    (recorded / run.name).mkdir(parents=True)
    (recorded / run.name / "best.pt").write_bytes(b"")
    assert find_checkpoint(run, str(recorded), [other]) == recorded / run.name / "best.pt"


def _frame(fold: int, seed: int, local: list[float]) -> pd.DataFrame:
    types = ["blister", "blister", "other"]
    return pd.DataFrame(
        {
            "fold": fold,
            "seed": seed,
            "packaging_type": types,
            "gate_global": [1 - v for v in local],
            "gate_local": local,
        }
    )


def test_gate_table_means_over_runs() -> None:
    table = gate_table([_frame(0, 0, [0.2, 0.4, 0.5]), _frame(1, 0, [0.6, 0.6, 0.1])])
    row = table.set_index(["packaging_type", "branch"])
    assert row.loc[("blister", "local"), "mean"] == pytest.approx((0.3 + 0.6) / 2)
    assert row.loc[("blister", "local"), "runs"] == 2
    assert row.loc[("blister", "local"), "images"] == 4  # 2 per fold, folds of one seed summed
    assert row.loc[("all", "global"), "mean"] == pytest.approx(1 - (1.1 / 3 + 1.3 / 3) / 2)


@pytest.mark.slow
def test_reprediction_reproduces_oof_on_toy(tmp_path: Path) -> None:
    """Train a toy multi-branch run, then re-predict it from best.pt: same probabilities."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pharmachecked.run",
            "experiment=smoke",
            "data.dataset=toy",
            "model=multibranch",
            "model.backbone.pretrained=false",
            "data.img_size=448",
            "data.max_train_images=8",
            "train.batch_size=4",
            "train.eval_batch_size=8",
            f"paths.results_dir={tmp_path.as_posix()}",
            f"paths.mlruns_dir={(tmp_path / 'mlruns').as_posix()}",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    (run,) = (tmp_path / "scratch" / "runs").glob("smoke_multibranch_*")
    cfg = OmegaConf.load(run / "config.yaml")
    ckpt = find_checkpoint(run, str(cfg.paths.checkpoints_dir), [])
    assert ckpt is not None
    out, diff = run_branch_outputs(run, cfg.paths, ckpt, "cpu")
    assert (run / BRANCH_FILE).is_file()
    assert len(out) == len(pd.read_csv(run / "oof_predictions.csv"))
    assert diff < 1e-5
