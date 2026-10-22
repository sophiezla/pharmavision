"""E12 saliency methods and faithfulness metrics; E13 TCAV helpers (PLAN 6.1-6.3, 6.7)."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from pharmachecked.config import CONFIG_DIR
from pharmachecked.evaluation.faithfulness import (
    FRACTIONS,
    cascade_groups,
    curve_auc,
    perturbation_curves,
    pointing_hit,
    randomization_curve,
    spearman,
    top_iou,
)
from pharmachecked.interpret.cams import blur, random_map, saliency
from pharmachecked.interpret.tcav import cav, tcav_score
from pharmachecked.models.registry import build_model


def _model(name: str, **overrides: object) -> nn.Module:
    cfg = OmegaConf.load(CONFIG_DIR / "model" / f"{name}.yaml")
    cfg = OmegaConf.merge(cfg, {"pretrained": False, **overrides})
    torch.manual_seed(0)
    return build_model(cfg).eval().requires_grad_(False)


class _Bright(nn.Module):
    """Counterfeit logit = 20 x (mean of the left half - 0.5); the evidence sits on the left."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": 20 * (x[..., : x.shape[-1] // 2].mean(dim=(1, 2, 3)) - 0.5)}


@pytest.mark.parametrize("method", ["gradcam", "gradcampp", "ig", "random"])
def test_cnn_maps(method: str) -> None:
    model = _model("efficientnet_lite0")
    x = torch.randn(2, 3, 64, 64)
    maps = saliency(method, model, x, ["a", "b"])
    assert maps.shape == (2, 64, 64) and (maps >= 0).all() and torch.isfinite(maps).all()


def test_rollout_map_on_a_vit() -> None:
    model = _model("dinov2_vits14", img_size=56)
    maps = saliency("rollout", model, torch.randn(2, 3, 56, 56), ["a", "b"])
    assert maps.shape == (2, 56, 56) and (maps > 0).all()
    blocks = [b.attn.fused_attn for b in model.backbone.blocks]
    assert all(blocks)  # restored after the pass


def test_random_map_is_reproducible_per_image() -> None:
    x = torch.zeros(2, 3, 32, 32)
    a, b = random_map(x, ["p", "q"]), random_map(x, ["p", "q"])
    assert torch.equal(a, b) and not torch.equal(a[0], a[1])


def test_deletion_and_insertion_reward_the_right_map() -> None:
    model = _Bright()
    x = torch.full((3, 16, 16), 0.5)
    x[..., :8] = 1.0  # counterfeit evidence on the left
    good = torch.zeros(16, 16)
    good[:, :8] = 1.0
    bad = 1 - good
    d_good, i_good = perturbation_curves(model, x, good)
    d_bad, i_bad = perturbation_curves(model, x, bad)
    assert len(d_good) == len(FRACTIONS) == 21
    assert curve_auc(d_good) < curve_auc(d_bad)
    assert curve_auc(i_good) > curve_auc(i_bad)


def test_blur_is_linear_in_normalization() -> None:
    x = torch.rand(1, 3, 32, 32)
    assert torch.allclose(blur(2 * x + 1), 2 * blur(x) + 1, atol=1e-5)


def test_localization_metrics() -> None:
    sal = torch.zeros(10, 10)
    sal[2, 3] = 1.0
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[1:4, 2:5] = True
    assert pointing_hit(sal, mask)
    assert not pointing_hit(sal, ~mask)
    full = torch.zeros(10, 10)
    full[1:4, 2:5] = 1.0
    assert top_iou(full, mask, fraction=0.09) == pytest.approx(1.0)


def test_cascade_groups_and_randomization() -> None:
    model = _model("efficientnet_lite0")
    names = [n for n, _ in cascade_groups(model)]
    assert names[0] == "head" and names[-1] == "stem" and "block0" in names
    x = torch.randn(2, 3, 32, 32)

    def grad_map(m: nn.Module, inp: torch.Tensor) -> torch.Tensor:
        return saliency("gradcam", m, inp, ["a", "b"])

    before = [p.clone() for p in model.parameters()]
    rows = randomization_curve(model, x, grad_map)
    assert [r["layer"] for r in rows] == names
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert np.isnan(spearman(torch.ones(4), torch.arange(4.0)))


def test_tcav_helpers() -> None:
    rng = np.random.default_rng(0)
    pos = rng.normal(1, 0.1, (20, 5))
    neg = rng.normal(-1, 0.1, (20, 5))
    v = cav(pos, neg, 0)
    assert np.isclose(np.linalg.norm(v), 1) and v @ np.ones(5) > 0
    assert tcav_score(np.ones((4, 5)), v) == 1.0
    assert tcav_score(-np.ones((4, 5)), v) == 0.0
