"""CPU latency of an ONNX model (PLAN 8.4): median of 100 runs after 10 warm-ups, batch 1."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

WARMUP = 10
RUNS = 100


def latency_ms(path: Path, size: int, warmup: int = WARMUP, runs: int = RUNS) -> dict[str, float]:
    """Median and interquartile range of single-image inference time (ms) on this CPU."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, 3, size, size), dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {"image": x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {"image": x})
        times.append((time.perf_counter() - start) * 1000)
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    return {"median_ms": float(med), "q1_ms": float(q1), "q3_ms": float(q3), "runs": runs}
