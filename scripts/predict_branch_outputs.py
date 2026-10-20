"""Re-predict the held-out folds of a multi-branch arm and save gate weights (PLAN 3.7).

Loads each run's ``best.pt`` (no training), writes ``branch_outputs.csv`` into the run folder
and ``results/pipeline/<exp>_<variant>_branch_outputs.json`` (per-run maximum probability
difference against the run's ``oof_predictions.csv``). The checkpoints are looked for where the
run's config recorded them and under ``<root>/checkpoints`` for every ``--checkpoint-root``.

Usage (the e04-gates GPU kit runs this):
    python scripts/predict_branch_outputs.py --exp e04 --model multibranch --variant full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.evaluation.aggregate import collect_runs
from pharmachecked.evaluation.branch_outputs import BRANCH_FILE, find_checkpoint, run_branch_outputs
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("predict_branch_outputs")


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--paths", default="remote", choices=("local", "remote"))
    parser.add_argument("--checkpoint-root", type=Path, action="append", default=[])
    parser.add_argument("--device", default=None, help="default: cuda when available")
    args = parser.parse_args()
    configure_logging()
    runs = [
        r
        for r in collect_runs(REPO_ROOT / "results" / "runs", args.exp)
        if (r.model, r.variant) == (args.model, args.variant)
    ]
    if not runs:
        log.error("no complete runs for %s %s/%s", args.exp, args.model, args.variant)
        return 1
    paths = OmegaConf.load(CONFIG_DIR / "paths" / f"{args.paths}.yaml")
    roots = [REPO_ROOT, *args.checkpoint_root]
    found = {}
    for r in runs:
        cfg = OmegaConf.load(r.root / "config.yaml")
        found[r.root.name] = find_checkpoint(r.root, str(cfg.paths.checkpoints_dir), roots)
    missing = sorted(name for name, path in found.items() if path is None)
    if missing:
        log.error(
            "%d of %d checkpoints not found, e.g. %s/best.pt. Copy the 'checkpoints' folder of "
            "the kit that trained them into %s and run again.",
            len(missing),
            len(runs),
            missing[0],
            REPO_ROOT,
        )
        return 1
    report: dict[str, dict[str, object]] = {}
    for r in runs:
        ckpt = found[r.root.name]
        assert ckpt is not None
        out, diff = run_branch_outputs(r.root, paths, ckpt, args.device)
        report[r.root.name] = {"images": len(out), "max_abs_prob_diff": diff}
        log.info("%s: %d images, max |prob - OOF prob| = %.2e", r.root.name, len(out), diff)
    out_json = REPO_ROOT / "results" / "pipeline" / f"{args.exp}_{args.variant}_branch_outputs.json"
    meta = {"exp": args.exp, "model": args.model, "variant": args.variant}
    write_json(out_json, {**meta, "device": args.device or "auto", "runs": report})
    log.info("wrote %s for %d runs", BRANCH_FILE, len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
