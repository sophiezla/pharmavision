"""Static INT8 quantization with ONNX Runtime (PLAN 8.2).

QDQ format, per-channel weights, QInt8 weights and QUInt8 activations (ONNX Runtime's CPU
recommendation), MinMax calibration on 200 training images of the fold (evaluation transform),
after ``quant_pre_process`` (shape inference and graph optimization).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_CALIBRATION = 200


class _Reader:
    """Calibration data reader over a fixed array of normalized inputs."""

    def __init__(self, x: np.ndarray, batch: int = 16) -> None:
        self.batches = [x[k : k + batch] for k in range(0, len(x), batch)]
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.batches):
            return None
        self.index += 1
        return {"image": self.batches[self.index - 1]}

    def rewind(self) -> None:
        self.index = 0


def quantize(fp32: Path, out: Path, calibration: np.ndarray) -> None:
    """Write the INT8 model of ``fp32`` to ``out``, calibrated on ``calibration``."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(_Reader, CalibrationDataReader):
        pass

    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    quantize_static(
        str(pre),
        str(out),
        Reader(calibration),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    pre.unlink(missing_ok=True)
