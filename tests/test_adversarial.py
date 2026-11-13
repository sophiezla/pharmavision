"""E15 attacks, adversarial-training hook and E16 distillation loss (PLAN 7.7-7.10, 8.1)."""

import numpy as np
import pytest
import torch
from torch import nn

from pharmachecked.adversarial.attacks import (
    Normalized,
    adversarial_batch_hook,
    gradient_attack,
    pgd_untargeted,
    to_unit,
)
from pharmachecked.adversarial.blackbox import random_search
from pharmachecked.adversarial.physical import (
    BOUNDS,
    apply_color,
    bounded,
    color_attack,
    patch_attack,
    patch_side,
)
from pharmachecked.training.distill import binary_kl

MEAN, STD = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]


class _Linear(nn.Module):
    """Counterfeit logit = 10 x (mean red - mean blue) of the normalized input."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": 10 * (x[:, 0].mean(dim=(1, 2)) - x[:, 2].mean(dim=(1, 2)))}


def _red(n: int = 3, size: int = 16) -> torch.Tensor:
    x = torch.full((n, 3, size, size), 0.3)
    x[:, 0] = 0.7
    return x


@pytest.mark.parametrize("name", ["fgsm", "pgd"])
def test_targeted_gradient_attack_lowers_the_counterfeit_logit(name: str) -> None:
    pytest.importorskip("torchattacks")  # [adversarial] extra; CI installs [dev] only
    wrapped = Normalized(_Linear(), MEAN, STD)
    x = _red()
    eps = 8 / 255
    adv = gradient_attack(wrapped, x, name, eps)
    assert (adv - x).abs().max() <= eps + 1e-6 and adv.min() >= 0 and adv.max() <= 1
    assert (wrapped.logit(adv) < wrapped.logit(x)).all()


def test_two_class_logits_match_the_sigmoid() -> None:
    wrapped = Normalized(_Linear(), MEAN, STD)
    x = _red()
    p = torch.softmax(wrapped(x), dim=1)[:, 1]
    assert torch.allclose(p, torch.sigmoid(wrapped.logit(x)))


def test_untargeted_pgd_and_batch_hook() -> None:
    wrapped = Normalized(_Linear(), MEAN, STD)
    x = _red()
    adv = pgd_untargeted(wrapped, x, torch.ones(3), 4 / 255, 7, 1 / 255)
    assert (adv - x).abs().max() <= 4 / 255 + 1e-6
    assert (wrapped.logit(adv) < wrapped.logit(x)).all()  # BCE toward label 1 goes up
    model = _Linear()
    images = (x - 0.5) / 0.25
    out, _targets = adversarial_batch_hook(MEAN, STD, 4 / 255, 3, 0.5)(model, images, torch.ones(3))
    changed = (out != images).flatten(1).any(dim=1)
    assert changed.tolist() == [True, True, False] or changed.tolist() == [True, False, False]
    assert torch.allclose(to_unit(images, MEAN, STD), x, atol=1e-6)


def test_color_parameters_stay_bounded_and_identity_is_neutral() -> None:
    raw = torch.tensor([[50.0, -50.0, 0.0, 50.0]])
    p = bounded(raw)
    assert p[0, 0] <= 1 + BOUNDS["brightness"] + 1e-6 and p[0, 1] >= 1 - BOUNDS["contrast"] - 1e-6
    assert abs(float(p[0, 3])) <= BOUNDS["hue"] + 1e-6
    x = torch.rand(2, 3, 8, 8)
    ident = torch.tensor([[1.0, 1.0, 1.0, 0.0]] * 2)
    assert torch.allclose(apply_color(x, ident), x, atol=1e-4)


def test_physical_attacks_reduce_the_logit() -> None:
    wrapped = Normalized(_Linear(), MEAN, STD)
    x = _red(2, 20)
    assert (wrapped.logit(color_attack(wrapped, x, steps=20)) < wrapped.logit(x)).all()
    side = patch_side(20, 20)
    assert side * side <= 0.05 * 400
    adv = patch_attack(wrapped, x, steps=20)
    assert torch.equal(adv[:, :, side:, :], x[:, :, side:, :])
    assert (wrapped.logit(adv) < wrapped.logit(x)).all()


class _BrightnessDetector(nn.Module):
    """Counterfeit when the image is dark: logit = 10 x (0.5 - mean)."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": 10 * (0.5 - x.mean(dim=(1, 2, 3)))}


def test_random_search_finds_and_counts_queries() -> None:
    wrapped = Normalized(_BrightnessDetector(), [0.0] * 3, [1.0] * 3)
    dark = np.full((32, 32, 3), 126, dtype=np.uint8)  # mean 0.494: counterfeit, close to 0.5
    found = random_search(wrapped, dark, threshold=0.5, seed=0, queries=200, batch=50)
    assert found.success and 1 <= found.queries <= 200 and found.best_prob < 0.5
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    missed = random_search(wrapped, black, threshold=0.5, seed=0, queries=20, batch=10)
    assert not missed.success and missed.queries == 20


def test_binary_kl() -> None:
    z = torch.tensor([-2.0, 0.0, 3.0])
    assert torch.isclose(binary_kl(z, z, 2.0), torch.tensor(0.0), atol=1e-6)
    assert binary_kl(z, -z, 2.0) > 0
