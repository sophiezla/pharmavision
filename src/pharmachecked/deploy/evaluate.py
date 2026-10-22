"""E16 deployment pass over one trained fold model (PLAN 8.2-8.3; docs/experiments/E16.md).

Exports the model to ONNX (FP32, checked against PyTorch), quantizes it to static INT8
(calibration: 200 training images of the fold, drawn with the run seed), and predicts the
fold's ``val``, ``cal`` and held-out images with both ONNX models on the CPU, so temperature,
threshold and conformal quantiles can be re-fitted per exported model on the laptop (8.3).
Writes ``deploy_predictions.csv`` and ``deploy.json`` into the run folder; the ONNX files go
next to the checkpoint (``model_fp32.onnx``, ``model_int8.onnx``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from pharmachecked.config import CONFIG_DIR
from pharmachecked.deploy.export_onnx import (
    MAX_ABS_DIFF,
    OPSET,
    export,
    max_abs_diff,
    onnx_logits,
    session,
)
from pharmachecked.deploy.quantize import N_CALIBRATION, quantize
from pharmachecked.models.loading import load_torch_model, run_config
from pharmachecked.utils.io import write_json

DEPLOY_FILE = "deploy_predictions.csv"
DEPLOY_JSON = "deploy.json"
PARTS = ("val", "cal", "heldout")


def _inputs(cfg: DictConfig, ids: list[str]) -> tuple[np.ndarray, list[int]]:
    from pharmachecked.training.task import eval_dataset, load_data

    manifest, _, _, images = load_data(cfg)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    ds = eval_dataset(
        cfg, tcfg, manifest, ids, images, list(cfg.model.mean), list(cfg.model.std), size
    )
    items = [ds[k] for k in range(len(ds))]
    x = np.stack([it["image"].numpy() for it in items]).astype(np.float32)
    return x, [int(it["label"]) for it in items]


def deploy_pass(run_root: Path, paths: DictConfig, checkpoint: Path) -> dict[str, object]:
    """Export, check, quantize and predict one fold model; returns the ``deploy.json`` record."""
    from pharmachecked.training.task import load_data

    cfg = run_config(run_root, paths, "cpu")
    if (cfg.data.get("protocol") or "kfold") != "kfold":
        raise ValueError("the deployment pass needs a k-fold run")
    model = load_torch_model(cfg, checkpoint)
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    _, _, splits, _ = load_data(cfg)
    fold = int(cfg.fold)
    fp32, int8 = checkpoint.parent / "model_fp32.onnx", checkpoint.parent / "model_int8.onnx"
    export(model, fp32, size)
    x_val, _ = _inputs(cfg, list(splits.ids(fold, "val")))
    diff = max_abs_diff(model, session(fp32), x_val)
    train = sorted(splits.ids(fold, "train"))
    rng = np.random.default_rng(int(cfg.seed))
    calib_ids = sorted(rng.choice(train, size=min(N_CALIBRATION, len(train)), replace=False))
    x_cal, _ = _inputs(cfg, list(calib_ids))
    quantize(fp32, int8, x_cal)
    rows = []
    sessions = {"fp32": session(fp32), "int8": session(int8)}
    for part in PARTS:
        ids = list(splits.ids(fold, part))
        x, labels = _inputs(cfg, ids)
        for precision, sess in sessions.items():
            logits = onnx_logits(sess, x)
            rows.append(
                pd.DataFrame(
                    {
                        "image_id": ids,
                        "split": part,
                        "precision": precision,
                        "fold": fold,
                        "seed": int(cfg.seed),
                        "y_true": labels,
                        "logit": logits,
                        "prob": 1 / (1 + np.exp(-logits)),
                    }
                )
            )
    pd.concat(rows, ignore_index=True).to_csv(
        run_root / DEPLOY_FILE, index=False, lineterminator="\n", float_format="%.7g"
    )
    record: dict[str, object] = {
        "opset": OPSET,
        "exporter": f"torch.onnx.export(dynamo=False), torch {torch.__version__}",
        "max_abs_diff_fp32": diff,
        "fp32_check_passed": diff <= MAX_ABS_DIFF,
        "size_mb_fp32": fp32.stat().st_size / 1e6,
        "size_mb_int8": int8.stat().st_size / 1e6,
        "size_mb_checkpoint": checkpoint.stat().st_size / 1e6,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_calibration": len(calib_ids),
        "img_size": size,
    }
    write_json(run_root / DEPLOY_JSON, record)
    return record
