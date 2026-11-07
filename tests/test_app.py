"""PLAN 8.6: the app's decision chain and preprocessing match the offline pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.deploy.app_pipeline import (
    AUTHENTIC,
    COUNTERFEIT,
    UNCERTAIN,
    AppParams,
    decide,
    fit_params,
    offline_decisions,
    quality_stats,
)

PARAMS = REPO_ROOT / "results" / "app" / "app_params.json"


def _rel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in (("val", 100), ("cal", 300), ("heldout", 100)):
        y = rng.integers(0, 2, n)
        z = (2 * y - 1) * 3.0 + rng.normal(0, 2.0, n)
        rows += [
            {"image_id": f"{split}{i}", "split": split, "y_true": int(y[i]), "logit": float(z[i])}
            for i in range(n)
        ]
    return pd.DataFrame(rows)


def test_decide_labels_and_guard() -> None:
    p = fit_params(_rel(), "r")
    assert decide(20.0, p).label == COUNTERFEIT
    assert decide(-20.0, p).label == AUTHENTIC
    assert decide(0.0, p).label == UNCERTAIN  # least confident input is rejected by the guard
    assert decide(20.0, p, ["sharpness"]).prob_counterfeit is None


def test_fit_uses_only_val_and_cal() -> None:
    a, b = _rel(), _rel()
    b.loc[b["split"] == "heldout", "logit"] *= -5
    assert fit_params(a, "r") == fit_params(b, "r")


def test_offline_matches_decide() -> None:
    rel = _rel()
    p = fit_params(rel, "r")
    off = offline_decisions(rel, p, "heldout")
    assert all(decide(z, p).label == lab for z, lab in zip(off["logit"], off["label"], strict=True))


def test_quality_stats_monotone() -> None:
    rng = np.random.default_rng(0)
    sharp = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    dark = Image.fromarray((np.asarray(sharp) * 0.2).astype(np.uint8))
    s, d = quality_stats(sharp), quality_stats(dark)
    assert d["brightness"] < s["brightness"] and d["contrast"] < s["contrast"]


def test_app_preprocessing_matches_offline_dataset(toy, tmp_path: Path) -> None:
    """Uploaded file -> AppModel gives the logit of the offline evaluation dataset."""
    import torch

    from pharmachecked.data.datasets import ManifestDataset
    from pharmachecked.data.transforms import build_eval_transform
    from pharmachecked.deploy.app_model import AppModel

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(48, 1)
    )

    class Wrap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = model

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": self.m(x).squeeze(-1)}

    cfg = OmegaConf.create(
        {"data": {"img_size": 64}, "model": {"img_size": 64, "mean": [0.5] * 3, "std": [0.25] * 3}}
    )
    app = AppModel.from_model(Wrap(), cfg)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    manifest = pd.read_csv(toy.manifest, dtype={"image_id": str})
    ids = list(manifest.loc[manifest["split"] == "dev", "image_id"][:20])
    ds = ManifestDataset(
        manifest, ids, toy.images_dir, build_eval_transform(tcfg, [0.5] * 3, [0.25] * 3, 64)
    )
    for k, iid in enumerate(ids):
        with torch.no_grad():
            offline = float(Wrap()(ds[k]["image"].unsqueeze(0))["logits"][0])
        with Image.open(toy.images_dir / f"{iid}.png") as im:
            online = app.logit(app.view(im))
        assert online == pytest.approx(offline, abs=1e-6)


@pytest.mark.slow
def test_app_matches_offline_on_20_heldout_images() -> None:
    """Acceptance test with the real fold model (needs the MODELS bundle)."""
    from pharmachecked.deploy.app_model import AppModel
    from pharmachecked.deploy.app_pipeline import load_reliability
    from pharmachecked.models.loading import run_checkpoint

    if not PARAMS.is_file():
        pytest.skip("no results/app/app_params.json")
    raw = json.loads(PARAMS.read_text())
    raw["quantiles"] = tuple(raw["quantiles"])
    params = AppParams(**raw)
    run_root = REPO_ROOT / "results" / "runs" / params.run
    ckpt = run_checkpoint(run_root)
    if ckpt is None:
        pytest.skip("checkpoint not installed (MODELS bundle)")
    app = AppModel.from_run(run_root, ckpt, OmegaConf.load(CONFIG_DIR / "paths" / "local.yaml"))
    rel = load_reliability(run_root)
    off = offline_decisions(rel, params, "heldout").sort_values("image_id").head(20)
    for row in off.itertuples():
        with Image.open(REPO_ROOT / "data" / "processed" / "images" / f"{row.image_id}.png") as im:
            logit = app.logit(app.view(im))
        # the offline logits come from fp16 autocast on the GPU (every value is fp16-exact);
        # the app runs fp32 on the CPU, so logits agree to fp16 precision and labels exactly
        assert logit == pytest.approx(row.logit, abs=0.5)
        assert decide(logit, params).label == row.label
