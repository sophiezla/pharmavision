"""Phase 5 on one GPU kit: prediction-only pass for E10/E11, then the E9 extension (ADR-024).

Pre-registered in docs/experiments/E10.md, E11.md and ADR-024-ADR-026. No training: every
stage re-predicts the 75 fold models of the five comparison models (ADR-021) from their
checkpoints. Finished files are kept, so the pipeline is resumable; one fold model's failure
does not stop the others.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.corruptions import load_corruption_config
from pharmachecked.evaluation.branch_outputs import find_checkpoint
from pharmachecked.evaluation.reliability_pass import RELIABILITY_FILE, reliability_predictions
from pharmachecked.evaluation.robustness import EXTENSION_FILE, corruption_predictions
from pharmachecked.pipeline import phase4
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)
PROFILES = phase4.PROFILES


def _checkpoint(run) -> Path | None:
    recorded = str(OmegaConf.load(run.root / "config.yaml").paths.checkpoints_dir)
    name = "model.joblib" if run.model.startswith("handcrafted") else "best.pt"
    ckpt = find_checkpoint(run.root, recorded, [REPO_ROOT], name)
    if ckpt is None:
        log.error("no %s for %s; copy it into %s", name, run.root.name, REPO_ROOT / "checkpoints")
    return ckpt


def _stage(name: str, runs: list, filename: str, work: Callable[[object, Path], str]) -> bool:
    """Run ``work(run, checkpoint)`` for every run without ``filename``; write a status file."""
    ok, report = True, {}
    for r in runs:
        if (r.root / filename).is_file():
            report[r.root.name] = "done earlier"
            continue
        ckpt = _checkpoint(r)
        if ckpt is None:
            report[r.root.name] = "missing checkpoint"
            ok = False
            continue
        try:
            report[r.root.name] = work(r, ckpt)
            log.info("%s: %s done", name, r.root.name)
        except Exception as exc:  # one fold model's failure must not stop the others
            log.exception("%s: %s failed", name, r.root.name)
            report[r.root.name] = f"failed: {exc}"
            ok = False
    write_json(phase4.PIPELINE_DIR / f"{name}_status.json", report)
    return ok


def run(profile_name: str) -> int:
    """Run Phase 5 end to end; returns a process exit code (0 = everything finished)."""
    p = PROFILES[profile_name]
    ok = True
    if p.name == "selftest":  # toy fold models to re-predict (the full kit uses existing runs)
        ok = phase4._run("st_p5_stage0_toyref", phase4.stage_jobs(p)["stage0_p4ref"])
    runs = phase4.e9_runs(p)
    paths = OmegaConf.load(CONFIG_DIR / "paths" / "remote.yaml")
    cfg_corr = load_corruption_config()

    def reliability(r, ckpt: Path) -> str:
        _, diff = reliability_predictions(r.root, paths, ckpt)
        return f"done (max held-out prob diff {diff:.2e})"

    def extension(r, ckpt: Path) -> str:
        corruption_predictions(r.root, paths, ckpt, cfg_corr, extension=True)
        return "done"

    prefix = p.prefix
    ok = _stage(f"{prefix}p5_reliability", runs, RELIABILITY_FILE, reliability) and ok
    ok = _stage(f"{prefix}e09_extension", runs, EXTENSION_FILE, extension) and ok
    log.info("Phase 5 pipeline finished (%s)", "all ok" if ok else "with failures")
    return 0 if ok else 1
