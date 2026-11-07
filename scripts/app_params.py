"""Fit the app's decision parameters for one fold model and save them (PLAN 8.6).

Guard, temperature and conformal quantiles come from the run's val / cal predictions; the
quality thresholds from the run's val images corrupted at the ``e09_quality_limits``
severities. Writes ``results/app/app_params.json``.

Usage:
    python scripts/app_params.py [--run e05_efficientnetv2-s_default_f0_s0_*]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.corruptions import load_corruption_config
from pharmachecked.deploy.app_pipeline import fit_params, fit_quality_thresholds, load_reliability
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("app_params")
DEFAULT_RUN = "e05_efficientnetv2-s_default_f0_s0_*"  # ADR-030


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN)
    args = parser.parse_args()
    configure_logging()
    run_root = next((REPO_ROOT / "results" / "runs").glob(args.run))
    rel = load_reliability(run_root)
    cfg = OmegaConf.load(run_root / "config.yaml")
    from pharmachecked.data.transforms import build_eval_geometry

    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    geometry = build_eval_geometry(OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml"), size)
    images = []
    for iid in rel.loc[rel["split"] == "val", "image_id"]:
        with Image.open(REPO_ROOT / "data" / "processed" / "images" / f"{iid}.png") as im:
            images.append((iid, geometry(im.convert("RGB"))))
    limits = pd.read_csv(REPO_ROOT / "results" / "tables" / "e09_ext_quality_limits.csv")
    limits = limits[limits["model"] == "efficientnetv2-s/default"] if "model" in limits else limits
    quality = fit_quality_thresholds(images, limits, load_corruption_config())
    params = fit_params(rel, run_root.name, quality)
    out = REPO_ROOT / "results" / "app" / "app_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, params.to_dict())
    log.info("wrote %s: %s", out.relative_to(REPO_ROOT), params.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
