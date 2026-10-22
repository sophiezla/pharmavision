"""Faithfulness and localization of saliency maps (PLAN 6.2-6.3, E12).

* Deletion / insertion (Petsiuk et al., 2018): pixels are ranked by importance; in 5% steps the
  top-ranked pixels are replaced by the blurred image (deletion) or copied from the image into
  the blurred image (insertion). The score is the probability of the class predicted on the
  clean image; the AUC is the trapezoid area over the 21 fractions 0, 0.05, ..., 1. Lower
  deletion AUC and higher insertion AUC mean a more faithful map.
* Model randomization (Adebayo et al., 2018): layers are re-initialized cumulatively from the
  top (head first, then backbone blocks from the last to the first, then the stem); after each
  step the maps are recomputed and compared with the original maps by Spearman rank
  correlation. A map that stays similar does not depend on the learned weights.
* Localization on synthetic counterfeits (PLAN 6.3): pointing game (the map's maximum lies inside
  the edit mask) and IoU of the top-15% map pixels with the mask.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from pharmachecked.interpret.cams import blur, unwrap_backbone

FRACTIONS = np.round(np.arange(0, 1.0001, 0.05), 2)
TOP_FRACTION = 0.15


def _order(saliency: torch.Tensor) -> torch.Tensor:
    """Pixel indices by decreasing importance (stable: ties keep raster order)."""
    return torch.argsort(saliency.flatten(), descending=True, stable=True)


def perturbation_curves(
    model: nn.Module, x: torch.Tensor, saliency: torch.Tensor, chunk: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Deletion and insertion curves of one image.

    Args:
        model: Model returning ``{"logits": (B,)}``, in eval mode.
        x: One normalized input ``(3, H, W)`` on the model's device.
        saliency: Its importance map ``(H, W)``.
        chunk: Forward-pass batch size.

    Returns:
        ``(deletion, insertion)``: predicted-class probabilities at each of ``FRACTIONS``.
    """
    h, w = x.shape[-2:]
    order = _order(saliency).to(x.device)
    base = blur(x.unsqueeze(0))[0]
    with torch.no_grad():
        sign = 1.0 if float(model(x.unsqueeze(0))["logits"][0]) >= 0 else -1.0
    variants = []
    for kind in ("deletion", "insertion"):
        for frac in FRACTIONS:
            mask = torch.zeros(h * w, dtype=torch.bool, device=x.device)
            mask[order[: round(frac * h * w)]] = True
            mask = mask.reshape(1, h, w)
            if kind == "deletion":
                variants.append(torch.where(mask, base, x))
            else:
                variants.append(torch.where(mask, x, base))
    batch = torch.stack(variants)
    probs = []
    with torch.no_grad():
        for start in range(0, len(batch), chunk):
            logits = model(batch[start : start + chunk])["logits"].float()
            probs.append(torch.sigmoid(sign * logits).cpu())
    p = torch.cat(probs).numpy()
    return p[: len(FRACTIONS)], p[len(FRACTIONS) :]


def curve_auc(curve: np.ndarray) -> float:
    """Trapezoid area under a curve sampled at ``FRACTIONS``."""
    return float(np.trapezoid(curve, FRACTIONS))


def cascade_groups(model: nn.Module) -> list[tuple[str, list[nn.Module]]]:
    """Layer groups for cascading randomization, top first.

    ``head`` = every child of the model except the backbone(s); then the backbone's modules after
    its block list (``neck``), each block from the last to the first, and the modules before
    the blocks (``stem``).
    """
    groups: list[tuple[str, list[nn.Module]]] = []
    head = [m for n, m in model.named_children() if n not in ("backbone", "local_backbone")]
    if head:
        groups.append(("head", head))
    base = unwrap_backbone(model)
    children = list(base.named_children())
    names = [n for n, _ in children]
    key = next((k for k in ("blocks", "stages", "layers") if k in names), None)
    if key is None:
        return [*groups, ("backbone", [base])]
    split = names.index(key)
    neck = [m for _, m in children[split + 1 :] if any(True for _ in m.parameters())]
    if neck:
        groups.append(("neck", neck))
    blocks = list(getattr(base, key))
    for k in range(len(blocks) - 1, -1, -1):
        groups.append((f"block{k}", [blocks[k]]))
    stem = [m for _, m in children[:split] if any(True for _ in m.parameters())]
    if stem:
        groups.append(("stem", stem))
    return groups


def reinitialize(modules: list[nn.Module], seed: int) -> None:
    """Re-initialize every submodule that defines ``reset_parameters`` (seeded)."""
    torch.manual_seed(seed)
    for module in modules:
        for sub in module.modules():
            reset = getattr(sub, "reset_parameters", None)
            if callable(reset):
                reset()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation of two maps (nan if either is constant)."""
    x, y = a.flatten().numpy(), b.flatten().numpy()
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def randomization_curve(
    model: nn.Module,
    x: torch.Tensor,
    maps_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    seed: int = 0,
) -> list[dict[str, object]]:
    """Mean Spearman correlation with the original maps after each cumulative randomization.

    Works on a deep copy; ``model`` is not changed.

    Returns:
        One row per step: ``{"step", "layer", "spearman"}`` (step 0 = no layer randomized).
    """
    original = maps_fn(model, x)
    work = copy.deepcopy(model).eval()
    rows: list[dict[str, object]] = []
    for step, (name, modules) in enumerate(cascade_groups(work), start=1):
        reinitialize(modules, seed + step)
        work.eval()
        maps = maps_fn(work, x)
        values = [spearman(o, m) for o, m in zip(original, maps, strict=True)]
        rows.append({"step": step, "layer": name, "spearman": float(np.nanmean(values))})
    return rows


def pointing_hit(saliency: torch.Tensor, mask: torch.Tensor) -> bool:
    """Whether the map's maximum (first in raster order on ties) lies inside the mask."""
    index = int(torch.argmax(saliency.flatten()))
    return bool(mask.flatten()[index])


def top_iou(saliency: torch.Tensor, mask: torch.Tensor, fraction: float = TOP_FRACTION) -> float:
    """IoU of the top ``fraction`` of map pixels with the (boolean) mask."""
    flat = saliency.flatten()
    top = torch.zeros_like(flat, dtype=torch.bool)
    top[_order(saliency)[: max(1, round(fraction * flat.numel()))]] = True
    m = mask.flatten().bool()
    union = int((top | m).sum())
    return float(int((top & m).sum()) / union) if union else float("nan")
