"""E8 one-class methods: prototype verifier maths and the one_class task on the toy set."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pharmachecked.models.prototype import PrototypeVerifier, l2_normalize
from pharmachecked.utils.run_dir import missing_files


def test_prototype_score_is_mean_cosine_distance_to_k_nearest() -> None:
    ref = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    v = PrototypeVerifier(k=2).fit(ref)
    q = np.array([[2.0, 0.0]])  # cosine sims 1, 0, -1 -> two nearest distances 0 and 1
    assert v.score(q) == pytest.approx([0.5])
    assert np.allclose(np.linalg.norm(l2_normalize(np.array([[3.0, 4.0]])), axis=1), 1)


def test_prototype_ranks_outliers_higher() -> None:
    rng = np.random.default_rng(0)
    centre = rng.standard_normal(16)
    inliers = centre + 0.1 * rng.standard_normal((50, 16))
    v = PrototypeVerifier(k=5).fit(inliers[:40])
    outliers = rng.standard_normal((10, 16))
    assert v.score(inliers[40:]).max() < v.score(outliers).min()


def test_prototype_rejects_bad_use() -> None:
    with pytest.raises(ValueError):
        PrototypeVerifier(k=0)
    with pytest.raises(ValueError):
        PrototypeVerifier(k=5).fit(np.ones((3, 4)))
    with pytest.raises(RuntimeError):
        PrototypeVerifier().score(np.ones((1, 4)))


def _one_class(tmp_path: Path, *model: str) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pharmachecked.run",
            "experiment=smoke",
            "data.dataset=toy",
            "task=one_class",
            "variant=oc",
            "tracker.enabled=false",
            f"paths.results_dir={tmp_path.as_posix()}",
            *model,
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    (run,) = (tmp_path / "scratch" / "runs").glob("smoke_*")
    assert missing_files(run) == []
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert 0 <= metrics["heldout"]["auroc"] <= 1
    return run


@pytest.mark.slow
def test_prototype_task_on_toy(tmp_path: Path) -> None:
    _one_class(tmp_path, "model=prototype_dinov2", "model.backbone.pretrained=false")


@pytest.mark.slow
def test_patchcore_task_on_toy(tmp_path: Path) -> None:
    pytest.importorskip("anomalib")
    _one_class(tmp_path, "model=patchcore", "model.pretrained=false")
