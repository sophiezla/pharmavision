"""PharmaChecked app (PLAN 8.6): photo -> quality checks -> OOD guard -> model -> calibrated
probability + conformal set -> label -> Grad-CAM++ heatmap.

Research prototype; not a medical device. Run on the laptop and open it from a phone on the
same network:

    python app/app.py [--host 0.0.0.0] [--port 7860]

Needs ``results/app/app_params.json`` (``scripts/app_params.py``) and the chosen fold model's
checkpoint (MODELS bundle, ``scripts/unpack_models.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.deploy.app_model import AppModel
from pharmachecked.deploy.app_pipeline import AppParams, decide, quality_failures
from pharmachecked.models.loading import run_checkpoint

PARAMS = REPO_ROOT / "results" / "app" / "app_params.json"
FOOTER = (
    "**Intended use:** research demonstration of counterfeit-packaging screening from photos "
    "of medicine packaging similar to the training data (Kaggle `kaggle_fake_real`). It is not "
    "a medical device and must not decide whether a medicine is taken; send suspected "
    "counterfeits to a pharmacist or the national regulator. The heatmap is indicative only "
    "(E12). Known failure modes: unseen packaging types, deliberate adversarial edits (E15), "
    "other sources (E7). See the [model card](https://github.com/sophiezla/pharmavision/blob/"
    "main/docs/MODEL_CARD.md)."
)


def load_params(path: Path = PARAMS) -> AppParams:
    """The fitted decision parameters."""
    raw = json.loads(path.read_text())
    raw["quantiles"] = tuple(raw["quantiles"])
    return AppParams(**raw)


def overlay(view: Image.Image, heat: np.ndarray) -> Image.Image:
    """The heatmap in red over the model-input view."""
    base = np.asarray(view, dtype=np.float32) / 255.0
    red = np.stack([np.ones_like(heat), np.zeros_like(heat), np.zeros_like(heat)], axis=-1)
    a = 0.55 * heat[..., None]
    return Image.fromarray(((1 - a) * base * 255 + a * red * 255).clip(0, 255).astype(np.uint8))


def analyse(image: Image.Image | None, model: AppModel, params: AppParams):
    """One request: returns (label markdown, details markdown, heatmap image)."""
    if image is None:
        return "Upload or take a photo of the packaging.", "", None
    view = model.view(image)
    failures = quality_failures(view, params)
    logit = model.logit(view) if not failures else float("nan")
    d = decide(logit, params, failures)
    details = f"Reason: {d.reason}."
    if d.prob_counterfeit is not None:
        names = {0: "authentic", 1: "counterfeit"}
        pset = ", ".join(names[c] for c in d.prediction_set) or "empty"
        details += (
            f"\n\nCalibrated probability of counterfeit: {d.prob_counterfeit:.3f}. "
            f"Conformal set (90% target coverage): {{{pset}}}."
        )
    heat = overlay(view, model.heatmap(view)) if d.prob_counterfeit is not None else None
    return f"## {d.label}", details, heat


def build(model: AppModel, params: AppParams) -> gr.Blocks:
    """The Gradio interface."""
    with gr.Blocks(title="PharmaChecked") as demo:
        gr.Markdown("# PharmaChecked\nPhotograph the medicine packaging (box or blister).")
        with gr.Row():
            inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Photo")
            heat = gr.Image(type="pil", label="Where the model looked (Grad-CAM++)")
        label, details = gr.Markdown(), gr.Markdown()
        gr.Button("Check").click(lambda im: analyse(im, model, params), inp, [label, details, heat])
        gr.Markdown(FOOTER)
    return demo


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    params = load_params()
    run_root = REPO_ROOT / "results" / "runs" / params.run
    ckpt = run_checkpoint(run_root)
    if ckpt is None:
        sys.stderr.write(f"no checkpoint for {params.run}; install the MODELS bundle first\n")
        return 1
    paths = OmegaConf.load(CONFIG_DIR / "paths" / "local.yaml")
    build(AppModel.from_run(run_root, ckpt, paths), params).launch(
        server_name=args.host, server_port=args.port
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
