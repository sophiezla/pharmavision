"""Phase 2 on one GPU kit: E1 -> decisions -> Optuna -> tuned re-runs -> E2 (ADR-013).

Every stage is resumable: finished runs are skipped, Optuna studies continue from SQLite, and
decisions are recomputed deterministically from the run folders. The pipeline stops (without
deciding anything) if an earlier stage left runs missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pharmachecked.config import REPO_ROOT
from pharmachecked.evaluation.aggregate import collect_runs
from pharmachecked.jobs import run_jobs
from pharmachecked.pipeline import arms as arm_defs
from pharmachecked.pipeline.decisions import (
    E1_TIERS,
    as_table,
    score_arms,
    tier_best,
    top_backbones,
)
from pharmachecked.pipeline.tuning import kind_of, run_study
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import slug

log = get_logger(__name__)


@dataclass(frozen=True)
class Profile:
    """How big the pipeline runs: the real protocol, or a tiny laptop self-test."""

    name: str
    e1_exp: str
    e2_exp: str
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    extra: tuple[str, ...]
    n_trials: int
    e1_jobs: tuple[arm_defs.Arm, ...]  # arms the kit trains for E1
    e1_decide: tuple[arm_defs.Arm, ...]  # arms the decisions consider (includes laptop-run tier 1)


PROFILES = {
    "full": Profile(
        "full",
        "e01",
        "e02",
        (0, 1, 2, 3, 4),
        (0, 1, 2),
        ("paths=remote", "train.num_workers=4"),
        30,
        arm_defs.E1_GPU + arm_defs.E1_ZEROSHOT,
        arm_defs.E1_CPU + arm_defs.E1_GPU,
    ),
    "selftest": Profile(
        "selftest",
        "st01",
        "st02",
        (0, 1),
        (0,),
        (
            "paths=remote",
            "data.dataset=toy",
            "data.img_size=64",
            "data.max_train_images=32",
            "data.max_external_per_source=8",
            "train.max_epochs=2",
            "train.batch_size=8",
            "device=cpu",
            "++model.pretrained=false",
        ),
        2,
        (
            arm_defs.Arm("handcrafted_lr", "default", "handcrafted"),
            arm_defs.Arm("cnn_small", "ft"),
            arm_defs.Arm("resnet50", "ft"),
            arm_defs.Arm("convnext_tiny", "ft"),
        ),
        (
            arm_defs.Arm("handcrafted_lr", "default", "handcrafted"),
            arm_defs.Arm("cnn_small", "ft"),
            arm_defs.Arm("resnet50", "ft"),
            arm_defs.Arm("convnext_tiny", "ft"),
        ),
    ),
}
PIPELINE_DIR = REPO_ROOT / "results" / "pipeline"
OPTUNA_DIR = REPO_ROOT / "results" / "optuna"


def _lines(arms: list[arm_defs.Arm], experiment: str, exp: str, p: Profile) -> list[list[str]]:
    out = []
    for arm in arms:
        seeds = (0,) if arm.task == "clip_zeroshot" else p.seeds  # zero-shot is deterministic
        for fold in p.folds:
            for seed in seeds:
                out.append(
                    arm.job(experiment, (*p.extra, f"exp={exp}", f"fold={fold}", f"seed={seed}"))
                )
    return out


def _write_jobs(name: str, jobs: list[list[str]]) -> Path:
    path = PIPELINE_DIR / f"{name}.txt"
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(j) for j in jobs) + "\n", encoding="utf-8", newline="\n")
    return path


def _run(name: str, jobs: list[list[str]]) -> bool:
    _write_jobs(name, jobs)
    outcomes = run_jobs(jobs, REPO_ROOT / "results" / "job_logs" / f"{name}.log")
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        log.error(
            "%s: %d of %d jobs failed; stopping before any decision", name, len(failed), len(jobs)
        )
    return not failed


def _expected(arm: arm_defs.Arm, p: Profile) -> int:
    return len(p.folds) * (1 if arm.task == "clip_zeroshot" else len(p.seeds))


def run(profile_name: str) -> int:
    """Run Phase 2 end to end; returns a process exit code (0 = everything finished)."""
    p = PROFILES[profile_name]
    runs_dir = REPO_ROOT / "results" / "runs"

    # Stage 1: E1.
    if not _run(f"{p.e1_exp}_stage1_e1", _lines(list(p.e1_jobs), "e01_baselines", p.e1_exp, p)):
        return 1
    records = collect_runs(runs_dir, p.e1_exp)
    missing = {}
    for arm in p.e1_decide:
        n = sum((r.model, r.variant) == arm.key for r in records)
        if n < _expected(arm, p):
            missing[f"{arm.key[0]}/{arm.key[1]}"] = n
    if missing:
        log.error("E1 incomplete (runs per arm: %s); no decisions taken", missing)
        return 1

    # Stage 2: decisions (development pool only).
    tiers = {a.key: E1_TIERS[a.key] for a in p.e1_decide}
    scores = score_arms(records, tiers)
    best = tier_best(scores)
    top2 = top_backbones(scores, 2)
    as_table(scores).to_csv(
        PIPELINE_DIR / f"{p.e1_exp}_ranking.csv", index=False, lineterminator="\n"
    )
    decisions = {
        "rule": "pooled OOF AUROC (seed-averaged), exact ties -> lower pooled OOF log-loss; "
        "development pool only (ADR-005, ADR-013)",
        "tier_best": {str(t): f"{s.model}/{s.variant}" for t, s in best.items()},
        "e2_backbones": top2,
    }
    write_json(PIPELINE_DIR / f"{p.e1_exp}_decisions.json", decisions)
    log.info("decisions: %s", json.dumps(decisions))

    # Stage 3: Optuna on fold 0 for each tier's best arm.
    by_key = {a.key: a for a in p.e1_decide}
    tuned_arms = []
    for _tier, score in best.items():
        arm = by_key[(score.model, score.variant)]
        name = f"{p.e1_exp}_{score.model}_{score.variant}"
        base = arm.job("e01_baselines", (*p.extra, f"exp={p.e1_exp}", "fold=0", "seed=0"))
        record = run_study(
            name,
            base,
            kind_of(arm.model, arm.variant),
            p.n_trials,
            OPTUNA_DIR,
            REPO_ROOT / "results" / "scratch" / "optuna",
        )
        params = tuple(f"{k}={v}" for k, v in sorted(record["params"].items()))
        tuned_arms.append(
            arm_defs.Arm(arm.model, f"{arm.variant}-tuned", arm.task, arm.overrides + params)
        )

    # Stage 4: tuned settings under the full protocol.
    if not _run(f"{p.e1_exp}_stage4_tuned", _lines(tuned_arms, "e01_baselines", p.e1_exp, p)):
        return 1

    # Stage 5: E2 adaptation strategies on the two best backbones.
    model_of = {slug(a.model): a.model for a in p.e1_decide}
    e2 = []
    for backbone in top2:
        has_lp = (backbone, "lp") in {a.key for a in p.e1_decide}
        e2 += arm_defs.e2_arms(model_of[backbone], has_lp)
    write_json(
        PIPELINE_DIR / f"{p.e2_exp}_arms.json",
        {
            "backbones": top2,
            "arms": [f"{slug(a.model)}/{a.variant}" for a in e2],
            "from_e1": "full fine-tuning (and linear probe where E1 ran it)",
        },
    )
    if not _run(f"{p.e2_exp}_stage5_e2", _lines(e2, "e02_adaptation", p.e2_exp, p)):
        return 1
    log.info("Phase 2 pipeline finished")
    return 0
