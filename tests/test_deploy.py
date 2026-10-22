"""E16 deployment: ONNX export check, INT8 quantization, latency (PLAN 8.2, 8.4)."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR
from pharmachecked.deploy.export_onnx import (
    MAX_ABS_DIFF,
    export,
    max_abs_diff,
    onnx_logits,
    session,
)
from pharmachecked.deploy.latency import latency_ms
from pharmachecked.deploy.quantize import quantize
from pharmachecked.models.registry import build_model


def _model(name: str, **overrides: object) -> torch.nn.Module:
    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / "model" / f"{name}.yaml"), {"pretrained": False, **overrides}
    )
    torch.manual_seed(0)
    return build_model(cfg).eval()


def test_export_quantize_and_time(tmp_path: Path) -> None:
    model = _model("efficientnet_lite0")
    fp32, int8 = tmp_path / "m.onnx", tmp_path / "m8.onnx"
    export(model, fp32, 64)
    x = np.random.default_rng(0).normal(size=(4, 3, 64, 64)).astype(np.float32)
    assert max_abs_diff(model, session(fp32), x) <= MAX_ABS_DIFF
    quantize(fp32, int8, x)
    assert int8.stat().st_size < fp32.stat().st_size
    logits = onnx_logits(session(int8), x)
    assert logits.shape == (4,) and np.isfinite(logits).all()
    timing = latency_ms(fp32, 64, warmup=2, runs=5)
    assert timing["median_ms"] > 0 and timing["runs"] == 5


def test_lora_model_exports_with_merged_adapters(tmp_path: Path) -> None:
    model = _model("efficientnet_lite0", mode="lora")
    with torch.no_grad():  # make the adapters matter
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.1)
    export(model, tmp_path / "lora.onnx", 64)
    x = np.random.default_rng(1).normal(size=(2, 3, 64, 64)).astype(np.float32)
    assert max_abs_diff(model, session(tmp_path / "lora.onnx"), x) <= MAX_ABS_DIFF
