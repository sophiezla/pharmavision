"""Saliency maps for E12 (PLAN 6.1): Grad-CAM, Grad-CAM++, integrated gradients, MIL tile
attention and a random-map baseline; attention rollout lives in ``attention_rollout.py``.

Every method maps a batch of model inputs ``(B, 3, H, W)`` (normalized, as the model sees them)
to non-negative importance maps ``(B, H, W)`` for the class the model predicts at logit 0
(counterfeit if ``logit >= 0``, else authentic). Methods:

* ``gradcam`` (Selvaraju et al., 2017) and ``gradcampp`` (Chattopadhay et al., 2018) via
  ``pytorch-grad-cam`` on the backbone's last feature map;
* ``ig``: integrated gradients (Sundararajan et al., 2017) via Captum, 32 steps from the blurred
  image (the faithfulness baseline), channel-summed absolute attribution;
* ``rollout``: attention rollout (Abnar & Zuidema, 2020), ViT backbones only;
* ``mil``: the multi-branch model's gated-attention MIL weight of each tile, painted over the
  tile (Ilse et al., 2018);
* ``random``: smooth uniform noise (14 x 14, bilinear), seeded per image.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvf

METHODS = ("gradcam", "gradcampp", "ig", "rollout", "mil", "random")
BLUR_SIGMA = 5.0  # px at model-input resolution; also the faithfulness baseline (E12.md)
IG_STEPS = 32


def blur(x: torch.Tensor, sigma: float = BLUR_SIGMA) -> torch.Tensor:
    """Gaussian blur of normalized inputs (blur commutes with per-channel normalization)."""
    kernel = int(4 * sigma) | 1
    return tvf.gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def predicted_sign(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """+1 where the model predicts counterfeit (logit >= 0), else -1; shape ``(B,)``."""
    with torch.no_grad():
        logits = model(x)["logits"].float()
    return torch.where(logits >= 0, 1.0, -1.0)


def unwrap_backbone(model: nn.Module) -> nn.Module:
    """The timm model inside ``model.backbone`` (peft LoRA wrappers removed)."""
    backbone = getattr(model, "backbone", model)
    get_base = getattr(backbone, "get_base_model", None)
    return get_base() if callable(get_base) else backbone


def cam_layer(model: nn.Module) -> nn.Module:
    """The backbone's last spatial feature map (EfficientNet ``bn2``, ConvNeXt last stage...)."""
    base = unwrap_backbone(model)
    for name in ("bn2", "norm_head", "act2"):
        layer = getattr(base, name, None)
        if isinstance(layer, nn.Module) and not isinstance(layer, nn.Identity):
            return layer
    for name in ("stages", "layer4"):
        layer = getattr(base, name, None)
        if layer is not None:
            return layer[-1] if isinstance(layer, nn.Sequential) else layer
    raise ValueError(f"no CAM layer found in {type(base).__name__}")


class _Logit(nn.Module):
    """``model(x)["logits"]`` as a ``(B, 1)`` tensor, the shape pytorch-grad-cam expects."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["logits"].float().unsqueeze(1)


class _Signed:
    """Grad-CAM target: the logit times +-1 (explains the predicted class)."""

    def __init__(self, sign: float) -> None:
        self.sign = sign

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        return self.sign * output[..., 0]


def _cam(kind: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus

    signs = predicted_sign(model, x)
    cls = GradCAM if kind == "gradcam" else GradCAMPlusPlus
    with cls(model=_Logit(model), target_layers=[cam_layer(model)]) as cam:
        cam.compute_input_gradient = True  # LoRA: frozen base weights still pass gradients
        maps = cam(input_tensor=x, targets=[_Signed(float(s)) for s in signs])
    return torch.from_numpy(np.asarray(maps, dtype=np.float32))


def integrated_gradients(model: nn.Module, x: torch.Tensor, steps: int = IG_STEPS) -> torch.Tensor:
    """Channel-summed |IG| from the blurred image, for the predicted class."""
    from captum.attr import IntegratedGradients

    signs = predicted_sign(model, x).to(x.device)

    def forward(inp: torch.Tensor) -> torch.Tensor:
        # captum expands the batch to (steps * B); repeat the signs to match
        reps = inp.shape[0] // len(signs)
        return model(inp)["logits"].float() * signs.repeat(reps)

    ig = IntegratedGradients(forward)
    attr = ig.attribute(x, baselines=blur(x), n_steps=steps, internal_batch_size=len(x))
    return attr.abs().sum(dim=1).detach().float().cpu()


def mil_map(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Tile attention of a multi-branch model, constant over each tile."""
    with torch.no_grad():
        attention = model(x)["mil_attention"].float().cpu()  # (B, g*g), row-major tiles
    g = round(attention.shape[1] ** 0.5)
    grid = attention.reshape(-1, 1, g, g)
    return nn.functional.interpolate(grid, size=x.shape[-2:], mode="nearest")[:, 0]


def _seed(image_id: str) -> int:
    return int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16)


def random_map(x: torch.Tensor, image_ids: list[str]) -> torch.Tensor:
    """Smooth uniform noise, reproducible per image id."""
    maps = []
    for image_id in image_ids:
        g = torch.Generator().manual_seed(_seed(image_id))
        maps.append(torch.rand(1, 1, 14, 14, generator=g))
    noise = torch.cat(maps)
    return nn.functional.interpolate(noise, size=x.shape[-2:], mode="bilinear")[:, 0]


def saliency(method: str, model: nn.Module, x: torch.Tensor, image_ids: list[str]) -> torch.Tensor:
    """Importance maps ``(B, H, W)`` (float32, CPU) of one method for a batch.

    Raises:
        ValueError: On an unknown method.
    """
    from pharmachecked.interpret.attention_rollout import attention_rollout

    fns: dict[str, Callable[[], torch.Tensor]] = {
        "gradcam": lambda: _cam("gradcam", model, x),
        "gradcampp": lambda: _cam("gradcampp", model, x),
        "ig": lambda: integrated_gradients(model, x),
        "rollout": lambda: attention_rollout(model, x),
        "mil": lambda: mil_map(model, x),
        "random": lambda: random_map(x, image_ids),
    }
    if method not in fns:
        raise ValueError(f"unknown saliency method {method!r}; known: {METHODS}")
    return fns[method]().float().clamp_min(0)
