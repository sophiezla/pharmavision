"""Handcrafted and zero-shot tasks write the same complete run folder as the train task."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pharmachecked.models.handcrafted import extract_features
from pharmachecked.utils.io import read_json
from pharmachecked.utils.run_dir import missing_files


def _run(tmp_path: Path, *overrides: str) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pharmachecked.run",
            "experiment=smoke",
            "data.dataset=toy",
            f"paths.results_dir={tmp_path.as_posix()}",
            f"paths.mlruns_dir={(tmp_path / 'mlruns').as_posix()}",
            *overrides,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    (run,) = list((tmp_path / "scratch" / "runs").iterdir())
    assert missing_files(run) == []
    return run


@pytest.mark.parametrize("model", ["handcrafted_lr", "handcrafted_xgb"])
def test_handcrafted_task(tmp_path: Path, model: str) -> None:
    run = _run(tmp_path, f"model={model}", "task=handcrafted")
    metrics = read_json(run / "metrics.json")
    assert metrics["n_features"] > 6000
    assert "recall" in metrics["external"]["toy_ext_cf"]
    oof = pd.read_csv(run / "oof_predictions.csv")
    assert oof["prob"].between(0, 1).all()
    assert Path((run / "checkpoint_path.txt").read_text().strip()).name == "model.joblib"
    # Features are cached per image and reused.
    assert any((tmp_path / "scratch" / "features").rglob("*.npy"))


def test_clip_zeroshot_task(tmp_path: Path) -> None:
    run = _run(tmp_path, "model=clip_vitb16", "task=clip_zeroshot", "model.pretrained=false")
    metrics = read_json(run / "metrics.json")
    assert metrics["train_seconds"] == 0.0
    assert set(metrics["external"]) == {"toy_ext_auth", "toy_ext_cf"}


def test_handcrafted_feature_layout() -> None:
    rng = np.random.default_rng(0)
    im = Image.fromarray((rng.random((300, 200, 3)) * 255).astype(np.uint8))
    vec = extract_features(im, size=224)
    hog_len = 13 * 13 * 2 * 2 * 9  # 14x14 cells of 16 px -> 13x13 blocks of 2x2 cells, 9 bins
    assert vec.shape == (96 + hog_len + 10,)
    assert vec[:96].sum() == pytest.approx(3.0)  # three normalized colour histograms
    assert vec[-10:].sum() == pytest.approx(1.0)  # normalized LBP histogram
