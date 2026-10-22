"""ONNX export of a trained fold model (PLAN 8.2).

The exported graph maps a normalized ``(B, 3, H, W)`` float32 input ``image`` to the counterfeit
logit ``logit`` ``(B,)``. Exporter: the TorchScript-based ``torch.onnx.export(dynamo=False)``
(the dynamo exporter needs ``onnxscript``, which is not a dependency), opset 17. LoRA adapters
are merged into the base weights first (``peft`` ``merge_and_unload``, numerically equivalent).
The export is accepted when the largest absolute logit difference against PyTorch (FP32, CPU)
on real inputs is at most ``MAX_ABS_DIFF``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

OPSET = 17
MAX_ABS_DIFF = 1e-4


class LogitOnly(nn.Module):
    """``model(x)["logits"]`` as the single graph output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["logits"]


def merged(model: nn.Module) -> nn.Module:
    """A CPU FP32 copy of ``model`` with LoRA adapters merged (unchanged otherwise)."""
    out = copy.deepcopy(model).float().cpu().eval()
    merge = getattr(getattr(out, "backbone", None), "merge_and_unload", None)
    if callable(merge):
        out.backbone = merge()
    return out


def export(model: nn.Module, path: Path, size: int) -> None:
    """Write ``model`` (merged, FP32) to ``path`` with a dynamic batch dimension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        LogitOnly(merged(model)),
        (dummy,),
        str(path),
        dynamo=False,
        opset_version=OPSET,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
    )


def session(path: Path) -> object:
    """An ONNX Runtime CPU session."""
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_logits(sess: object, x: np.ndarray, batch: int = 32) -> np.ndarray:
    """Logits of normalized inputs ``(N, 3, H, W)`` float32."""
    out = [sess.run(None, {"image": x[k : k + batch]})[0] for k in range(0, len(x), batch)]
    return np.concatenate(out).reshape(-1) if out else np.zeros(0)


def max_abs_diff(model: nn.Module, sess: object, x: np.ndarray) -> float:
    """Largest |logit_torch - logit_onnx| on ``x`` (PyTorch FP32 on CPU, merged copy)."""
    ref = merged(model)
    with torch.no_grad():
        torch_logits = ref(torch.from_numpy(x))["logits"].float().numpy().reshape(-1)
    return float(np.abs(torch_logits - onnx_logits(sess, x)).max())
