"""The smoke experiment (real `train` task, tiny settings) writes a complete run folder."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from pharmachecked.utils.io import read_json
from pharmachecked.utils.run_dir import missing_files


def _smoke(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pharmachecked.run",
            "experiment=smoke",
            "data.dataset=toy",
            f"paths.results_dir={tmp_path.as_posix()}",
            f"paths.mlruns_dir={(tmp_path / 'mlruns').as_posix()}",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_smoke_on_toy_writes_complete_run_folder(tmp_path: Path) -> None:
    result = _smoke(tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    (run,) = list((tmp_path / "scratch" / "runs").iterdir())
    assert run.name.startswith("smoke_cnn-small_toy_f0_s0_")
    assert missing_files(run) == []
    oof = pd.read_csv(run / "oof_predictions.csv")
    ext = pd.read_csv(run / "external_predictions.csv")
    assert list(oof.columns) == list(ext.columns)
    assert set(oof["fold"]) == {0}
    assert set(ext["fold"]) == {-1}
    assert oof["prob"].between(0, 1).all()
    metrics = read_json(run / "metrics.json")
    assert set(metrics["external"]) == {"toy_ext_auth", "toy_ext_cf"}
    assert "specificity" in metrics["external"]["toy_ext_auth"]
    assert "recall" in metrics["external"]["toy_ext_cf"]
    assert "auroc" in metrics["heldout"]
    log = pd.read_csv(run / "train_log.csv")
    assert len(log) == 1
    ckpt = Path((run / "checkpoint_path.txt").read_text().strip())
    assert ckpt.name == "best.pt"
    assert ckpt.is_file()
    assert not (ckpt.parent / "resume.pt").exists()
