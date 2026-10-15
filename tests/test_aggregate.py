"""Aggregation over synthetic run folders: counts, exclusions, pooling and the external block."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmachecked.evaluation.aggregate import (
    collect_runs,
    external_correctness,
    pairwise_vs_best,
    pairwise_vs_reference,
    pooled_oof,
    summarize,
    to_markdown,
)
from pharmachecked.utils.io import write_json
from pharmachecked.utils.run_dir import COMPLETE_FILES, make_run_id

RNG = np.random.default_rng(0)
IDS = [f"{k:016x}" for k in range(40)]
LABELS = np.array([k % 2 for k in range(40)])
EXT_IDS = [f"e{k:015x}" for k in range(10)]


def _make_run(
    runs: Path, model: str, fold: int, seed: int, strength: float, dirty: bool = False
) -> None:
    run_id = make_run_id(
        "e99", model, "default", fold, seed, datetime(2026, 10, 10, 0, fold, seed), dirty
    )
    root = runs / run_id
    root.mkdir(parents=True)
    held = [i for k, i in enumerate(IDS) if k % 2 == fold % 2] if fold < 2 else IDS
    held_labels = LABELS[[IDS.index(i) for i in held]]
    prob = np.clip(held_labels * strength + RNG.random(len(held)) * (1 - strength), 0, 1)
    common = {"seed": seed, "packaging_type": "box", "source_dataset": "dev"}
    pd.DataFrame(
        {
            "image_id": held,
            "fold": fold,
            "y_true": held_labels,
            "logit": 0.0,
            "prob": prob,
            **common,
        }
    ).to_csv(root / "oof_predictions.csv", index=False)
    ext_prob = RNG.random(10) * 0.4  # mostly below threshold 0.5
    pd.DataFrame(
        {
            "image_id": EXT_IDS,
            "fold": -1,
            "y_true": 0,
            "logit": 0.0,
            "prob": ext_prob,
            "seed": seed,
            "packaging_type": "",
            "source_dataset": "ext_auth",
        }
    ).to_csv(root / "external_predictions.csv", index=False)
    write_json(
        root / "metrics.json",
        {
            "heldout": {
                "auroc": 0.8,
                "auprc": 0.8,
                "sens_at_95_spec": 0.5,
                "balanced_accuracy": 0.7,
                "macro_f1": 0.7,
                "ece": 0.1,
                "brier": 0.2,
                "threshold": 0.5,
            },
            "train_seconds": 60.0,
        },
    )
    for name in COMPLETE_FILES:
        (root / name).touch(exist_ok=True)


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    for fold in (0, 1):
        for seed in (0, 1):
            _make_run(tmp_path, "strong", fold, seed, 0.6)
            _make_run(tmp_path, "weak", fold, seed, 0.1)
    _make_run(tmp_path, "strong", 0, 2, 0.6, dirty=True)  # excluded
    (tmp_path / "e99_incomplete_default_f0_s0_20261010-000000").mkdir()  # excluded
    return tmp_path


def test_collect_excludes_dirty_and_incomplete(runs: Path) -> None:
    records = collect_runs(runs, "e99")
    assert len(records) == 8
    assert {r.model for r in records} == {"strong", "weak"}


def test_pooled_oof_averages_over_seeds(runs: Path) -> None:
    records = [r for r in collect_runs(runs, "e99") if r.model == "strong"]
    pooled = pooled_oof(records)
    assert len(pooled) == len(IDS)
    assert set(pooled["n_runs"]) == {2}


def test_external_correctness_is_run_average(runs: Path) -> None:
    records = [r for r in collect_runs(runs, "e99") if r.model == "strong"]
    ext = external_correctness(records)
    assert len(ext) == 10
    assert ext["correct"].between(0, 1).all()


def test_summary_and_pairwise(runs: Path) -> None:
    summary, pooled = summarize(collect_runs(runs, "e99"), n_boot=200)
    assert list(summary["config"]) == ["strong/default", "weak/default"]
    assert list(summary["runs"]) == [4, 4]
    assert "ext_auth specificity" in summary.columns
    pairwise = pairwise_vs_best(pooled, n_boot=200)
    assert list(pairwise["config"]) == ["weak/default"]
    assert pairwise["holm_p"].iloc[0] < 0.05
    assert "| config |" in to_markdown(pairwise)

    vs = pairwise_vs_reference(pooled, "strong/default", n_boot=200)
    assert list(vs["config"]) == ["weak/default"]
    assert bool(vs["worse"].iloc[0]) and not bool(vs["improves"].iloc[0])
    assert {"auroc_diff", "ece_diff", "log_loss_diff", "images"} <= set(vs.columns)
    assert vs["auroc_diff"].iloc[0].startswith("-")  # configuration minus reference
    flipped = pairwise_vs_reference(pooled, "weak/default", n_boot=200)
    assert bool(flipped["improves"].iloc[0])
    with pytest.raises(KeyError):
        pairwise_vs_reference(pooled, "missing/default")


def test_duplicate_runs_for_one_arm_are_refused(runs: Path) -> None:
    import shutil

    original = next(runs.glob("e99_strong_default_f0_s0_*"))
    shutil.copytree(original, runs / "e99_strong_default_f0_s0_20261011-000000")  # newer copy
    with pytest.raises(ValueError, match="several complete runs"):
        collect_runs(runs, "e99")
