"""E10/E11 prediction-only pass on toy runs, with a tiny fake OOD folder (ADR-026)."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from pharmachecked.data.ood import load_ood_set, select_eval_split
from pharmachecked.evaluation.reliability_pass import RELIABILITY_FILE, reliability_predictions


def _ood_folder(tmp_path: Path, toy_images: Path) -> Path:
    """data/ood/{near,far} with four toy images each, in the committed manifest format."""
    data_dir = tmp_path / "data"
    sources = sorted(toy_images.glob("*.png"))
    for k, name in enumerate(("near", "far")):
        folder = data_dir / "ood" / name
        (folder / "images").mkdir(parents=True)
        rows = []
        for j, src in enumerate(sources[4 * k : 4 * k + 4]):
            image_id = f"ood{name}{j:02d}"
            shutil.copy(src, folder / "images" / f"{image_id}.png")
            rows.append(
                {
                    "image_id": image_id,
                    "set": name,
                    "source_dataset": f"fake_{name}",
                    "source_class": "a" if j % 2 else "b",
                }
            )
        pd.DataFrame(rows).to_csv(folder / "manifest.csv", index=False)
    return data_dir


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


def test_select_eval_split_is_stratified_and_stable(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "image_id": [f"i{k:02d}" for k in range(9)],
            "source_dataset": ["a"] * 4 + ["b"] * 5,
            "source_class": ["x"] * 9,
        }
    )
    half = select_eval_split(table)
    assert (half == "select").sum() == 2 + 2  # floor(4/2) + floor(5/2)
    assert half.equals(select_eval_split(table.sample(frac=1, random_state=3)).loc[half.index])
    with pytest.raises(ValueError):
        load_ood_set(tmp_path, "medium")


@pytest.mark.slow
@pytest.mark.parametrize("extra", [(), ("model=handcrafted_xgb", "task=handcrafted")])
def test_reliability_pass_on_toy_runs(tmp_path: Path, extra: tuple) -> None:
    run = _train(tmp_path, *extra)
    cfg = OmegaConf.load(run / "config.yaml")
    ckpt = "model.joblib" if extra else "best.pt"
    checkpoint = Path(cfg.paths.checkpoints_dir) / run.name / ckpt
    paths = cfg.paths
    paths.data_dir = _ood_folder(tmp_path, Path(cfg.paths.toy_dir) / "images").as_posix()
    out, diff = reliability_predictions(run, paths, checkpoint, "cpu")
    assert (run / RELIABILITY_FILE).is_file()
    counts = out["split"].value_counts()
    assert set(counts.index) == {"train", "val", "cal", "heldout", "ood_near", "ood_far"}
    assert counts["ood_near"] == counts["ood_far"] == 4
    assert (out.loc[out["split"].str.startswith("ood"), "y_true"] == -1).all()
    assert diff < 1e-4  # same machine, same code path as training
    if extra:
        assert out["maha"].isna().all() and out["knn"].isna().all()
    else:
        assert np.isfinite(out[["maha", "knn"]].to_numpy()).all()
