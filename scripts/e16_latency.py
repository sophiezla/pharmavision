"""Laptop CPU latency of the exported fold-0 / seed-0 models (PLAN 8.4).

Median of 100 single-image runs after 10 warm-ups (``deploy/latency.py``) for the FP32 and
INT8 ONNX files of each E16 student and of the final model, which come with the MODELS bundle
(``scripts/unpack_models.py``). Writes ``results/pipeline/e16_latency.json`` with the machine
description; ``scripts/phase6to8_tables.py --exp e16`` adds it to ``e16_deployment``.

Usage:
    python scripts/e16_latency.py
"""

from __future__ import annotations

import argparse
import platform
import sys

from pharmachecked.config import REPO_ROOT
from pharmachecked.deploy.latency import latency_ms
from pharmachecked.models.loading import run_checkpoint
from pharmachecked.utils.io import read_json, write_json
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("e16_latency")
ARMS = {
    "efficientnet-lite0/kd-t4": "e16_efficientnet-lite0_kd-t4_f0_s0_*",
    "mobilenetv3-large/kd-t4": "e16_mobilenetv3-large_kd-t4_f0_s0_*",
    "final model": "e05_efficientnetv2-s_default_f0_s0_*",
}
OUT = REPO_ROOT / "results" / "pipeline" / "e16_latency.json"


def main() -> int:
    """CLI entry point; returns a process exit code."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    configure_logging()
    import onnxruntime

    record: dict[str, object] = {
        "machine": f"{platform.platform()} / {platform.processor() or platform.machine()}",
        "onnxruntime": onnxruntime.__version__,
        "python": platform.python_version(),
        "models": {},
    }
    missing = 0
    for arm, pattern in ARMS.items():
        run_root = next((REPO_ROOT / "results" / "runs").glob(pattern))
        ckpt = run_checkpoint(run_root)
        size = int(read_json(run_root / "deploy.json")["img_size"])
        for precision in ("fp32", "int8"):
            onnx = ckpt.parent / f"model_{precision}.onnx" if ckpt else None
            if onnx is None or not onnx.is_file():
                log.error("missing ONNX file for %s %s", arm, precision)
                missing += 1
                continue
            res = latency_ms(onnx, size)
            record["models"][f"{arm}|{precision}"] = {"run": run_root.name, **res}  # type: ignore[index]
            log.info("%s %s: median %.2f ms", arm, precision, res["median_ms"])
    if missing:
        log.error("%d ONNX files missing; install the MODELS bundle first", missing)
        return 1
    write_json(OUT, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
