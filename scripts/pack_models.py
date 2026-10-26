"""Pack the weights and samples the laptop needs after the Phases 6-8 kit (ADR-027).

Results bundles never contain weights. This second, PRIVATE bundle carries:

* ``checkpoints/<run_id>/best.pt`` of the final model (15), the E1 best model retrained on data
  v2 (``p4ref`` ConvNeXt-T, 15) and every E16 student (Phase 10 freeze and locked-test
  evaluation run on the laptop, which holds the locked test set);
* ``checkpoints/<run_id>/model_{fp32,int8}.onnx`` of the fold-0 / seed-0 run of each E16 arm and
  of the final model (laptop and phone latency, PLAN 8.4);
* ``synthetic_qc/synthetic_manifest.csv`` (the PLAN 7.4 review sample was dropped, ADR-029).

``MANIFEST.json`` lists the SHA-256 of every file; ``scripts/unpack_models.py`` verifies and
installs it.

Usage (run by kit_run.py when KIT.json has "models": true):
    python scripts/pack_models.py --out <kit folder>/MODELS_<name>_<stamp>.zip [--profile full]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

from pharmachecked.config import REPO_ROOT
from pharmachecked.evaluation.aggregate import collect_runs
from pharmachecked.jobs import compose_job
from pharmachecked.models.loading import run_checkpoint
from pharmachecked.pipeline.combined import PROFILES, refs
from pharmachecked.synthesis.generate import synthetic_store
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("pack_models")


def model_files(profile: str) -> dict[str, Path]:
    """``{archive name: path}`` of every weight file to pack."""
    p = PROFILES[profile]
    runs = refs(p, "final") + refs(p, "e1_best")
    students = collect_runs(REPO_ROOT / "results" / "runs", f"{p.base.prefix}e16")
    out: dict[str, Path] = {}
    for r in runs + students:
        ckpt = run_checkpoint(r.root)
        if ckpt is None:
            continue
        out[f"checkpoints/{r.root.name}/{ckpt.name}"] = ckpt
        if r.fold == 0 and r.seed == 0 and (r in students or r in refs(p, "final")):
            for name in ("model_fp32.onnx", "model_int8.onnx"):
                onnx = ckpt.parent / name
                if onnx.is_file():
                    out[f"checkpoints/{r.root.name}/{name}"] = onnx
    return out


def qc_files(profile: str) -> dict[str, Path]:
    """The PLAN 7.4 review sample (images, masks) and the synthetic manifest."""
    p = PROFILES[profile]
    cfg = compose_job(["experiment=e14_synthetic", "model=efficientnetv2_s", *p.base.extra])
    store = synthetic_store(cfg)
    sample = REPO_ROOT / "results" / "pipeline" / f"{p.base.prefix}e14_qc_sample.csv"
    out: dict[str, Path] = {}
    if store.manifest.is_file():
        out["synthetic_qc/synthetic_manifest.csv"] = store.manifest
    if sample.is_file():
        out["synthetic_qc/qc_sample.csv"] = sample
        for image_id in pd.read_csv(sample, dtype={"image_id": str})["image_id"]:
            out[f"synthetic_qc/images/{image_id}.png"] = store.images / f"{image_id}.png"
            out[f"synthetic_qc/masks/{image_id}.png"] = store.masks / f"{image_id}.png"
    return out


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Write the bundle; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", default="full", choices=sorted(PROFILES))
    args = parser.parse_args()
    configure_logging()
    files = {**model_files(args.profile), **qc_files(args.profile)}
    files = {k: v for k, v in files.items() if v.is_file()}
    if not files:
        log.error("nothing to pack")
        return 1
    manifest = {
        "files": {
            k: {"sha256": sha256(v), "size": v.stat().st_size} for k, v in sorted(files.items())
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_STORED) as zf:
        for name, path in sorted(files.items()):
            zf.write(path, name)
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=1) + "\n")
    total = sum(v.stat().st_size for v in files.values()) / 1e9
    log.info("wrote %s: %d files, %.2f GB", args.out, len(files), total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
