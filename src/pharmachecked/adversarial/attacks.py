"""Gradient attacks for E15 (PLAN 7.7): FGSM and PGD-20 via ``torchattacks``.

Attacks act on model-input images in [0, 1] (after the evaluation geometry, before
normalization). The one-logit detector is exposed to ``torchattacks`` as a two-class model with
logits ``[0, z]``, so softmax(class 1) = sigmoid(z) and cross-entropy equals the BCE of the
detector. Targeted toward "authentic" (class 0), L-inf, eps in {1, 2, 4, 8}/255; PGD: 20 steps of
eps/8 from a random start (Madry et al., 2018); FGSM: one step (Goodfellow et al., 2015).

``pgd_untargeted`` is the training-time attack of the PGD defence (PLAN 7.10).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

EPSILONS = (1 / 255, 2 / 255, 4 / 255, 8 / 255)
PGD_STEPS = 20


class Normalized(nn.Module):
    """``x01 -> model((x01 - mean) / std)["logits"]`` as two-class logits ``[0, z]``."""

    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(list(mean)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(std)).view(1, 3, 1, 1))

    def logit(self, x01: torch.Tensor) -> torch.Tensor:
        """The detector's counterfeit logit ``(B,)``."""
        return self.model((x01 - self.mean) / self.std)["logits"].float()

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        z = self.logit(x01)
        return torch.stack([torch.zeros_like(z), z], dim=1)


def to_unit(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo the model normalization: normalized tensor -> [0, 1] image."""
    m = torch.tensor(list(mean), device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(list(std), device=x.device).view(1, 3, 1, 1)
    return (x * s + m).clamp(0, 1)


def gradient_attack(wrapped: Normalized, x01: torch.Tensor, name: str, eps: float) -> torch.Tensor:
    """Targeted (toward authentic) FGSM or PGD-20 adversarial images in [0, 1].

    Raises:
        ValueError: On an unknown attack name.
    """
    import torchattacks

    if name == "fgsm":
        attack = torchattacks.FGSM(wrapped, eps=eps)
    elif name == "pgd":
        attack = torchattacks.PGD(wrapped, eps=eps, alpha=eps / 8, steps=PGD_STEPS)
    else:
        raise ValueError(f"unknown gradient attack {name!r}")
    attack.set_mode_targeted_by_label(quiet=True)
    target = torch.zeros(len(x01), dtype=torch.long, device=x01.device)
    return attack(x01, target).detach()


def pgd_untargeted(
    wrapped: Normalized,
    x01: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    alpha: float,
) -> torch.Tensor:
    """Untargeted L-inf PGD on the BCE loss (random start), for adversarial training."""
    adv = (x01 + torch.empty_like(x01).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = nn.functional.binary_cross_entropy_with_logits(wrapped.logit(adv), y.float())
        (grad,) = torch.autograd.grad(loss, adv)
        adv = adv.detach() + alpha * grad.sign()
        adv = (x01 + (adv - x01).clamp(-eps, eps)).clamp(0, 1).detach()
    return adv


def adversarial_batch_hook(
    mean: Sequence[float], std: Sequence[float], eps: float, steps: int, fraction: float
) -> object:
    """``fit`` batch hook: replace the first ``fraction`` of each batch by PGD examples.

    The attack runs with the model in eval mode (no batch-norm statistic updates) and restores
    training mode afterwards. Step size eps / 4.
    """

    def hook(model: nn.Module, images: torch.Tensor, targets: torch.Tensor) -> tuple:
        n = round(fraction * len(images))
        if n == 0:
            return images, targets
        wrapped = Normalized(model, mean, std).to(images.device)
        was_training = model.training
        model.eval()
        x01 = to_unit(images[:n], mean, std)
        adv = pgd_untargeted(wrapped, x01, targets[:n].round(), eps, steps, eps / 4)
        model.train(was_training)
        out = images.clone()
        out[:n] = (adv - wrapped.mean) / wrapped.std
        return out, targets

    return hook
