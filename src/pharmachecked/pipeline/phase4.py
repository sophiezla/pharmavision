"""Phase 4 on one GPU kit: p4ref retrain -> E6 -> E7 -> E8 -> E9 corruption evaluation.

Pre-registered in docs/experiments/E06.md-E09.md and ADR-021-ADR-023. Every stage is resumable
(finished runs are skipped by the job runner; finished corruption files are kept). A failing
stage does not stop the later ones, except that E9 needs the p4ref fold models.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.corruptions import load_corruption_config
from pharmachecked.evaluation.aggregate import collect_runs
from pharmachecked.evaluation.branch_outputs import find_checkpoint
from pharmachecked.evaluation.robustness import CORRUPTION_FILE, corruption_predictions
from pharmachecked.jobs import run_jobs
from pharmachecked.pipeline.arms import Arm
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)
PIPELINE_DIR = REPO_ROOT / "results" / "pipeline"

# ADR-021: the five comparison models. The final model (EfficientNetV2-S LoRA r8, default
# recipe) already exists on data v2 as e05_efficientnetv2-s_default_*, so p4ref retrains four.
FINAL = Arm(
    "efficientnetv2_s",
    "default",
    overrides=("model.mode=lora", "model.lora_rank=8", "train.backbone_lr_mult=1.0"),
)
P4REF = (
    Arm("handcrafted_xgb", "default", "handcrafted"),
    Arm("convnext_tiny", "ft"),
    Arm("swin_tiny", "ft"),
    Arm("dinov2_vits14", "ft"),
)
COMPARISON = (*P4REF, FINAL)
FRACTIONS = (0.10, 0.25, 0.50)  # E6; 100% = p4ref (and e05 default for the final model)
E8_TYPES = ("blister", "packaging")  # >= 30 images per class (E08.md)
ONE_CLASS = (
    Arm("prototype_dinov2", "oc", "one_class"),
    Arm("patchcore", "oc", "one_class"),
)


@dataclass(frozen=True)
class Profile:
    """Full protocol or a tiny laptop self-test."""

    name: str
    prefix: str  # "" (p4ref/e06/...) or "st-" (st-p4ref/st06/...)
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    extra: tuple[str, ...]
    e8_types: tuple[str, ...]
    final_exp: str  # experiment holding the final model's full-protocol runs


PROFILES = {
    "full": Profile(
        "full",
        "",
        (0, 1, 2, 3, 4),
        (0, 1, 2),
        ("paths=remote", "train.num_workers=4", "data.split_version=2"),
        E8_TYPES,
        "e05",
    ),
    "selftest": Profile(
        "selftest",
        "st",
        (0, 1),
        (0,),
        (
            "paths=remote",
            "data.dataset=toy",
            "data.max_train_images=16",
            "data.max_external_per_source=4",
            "train.max_epochs=1",
            "train.batch_size=4",
            "device=cpu",
            "++model.pretrained=false",
            "++model.backbone.pretrained=false",
        ),
        (),  # the toy set has no packaging types; E8 partitions are unit-tested instead
        "stp4ref",
    ),
}


def _exp(p: Profile, name: str) -> str:
    return f"{p.prefix}{name}"


def _variant(arm: Arm, suffix: str) -> Arm:
    return Arm(arm.model, f"{arm.variant}-{suffix}", arm.task, arm.overrides)


def stage_jobs(p: Profile) -> dict[str, list[list[str]]]:
    """Job lines of every training stage, in run order."""
    common = p.extra
    ref_arms = (*P4REF, FINAL) if p.name == "selftest" else P4REF
    jobs: dict[str, list[list[str]]] = {
        "stage0_p4ref": [
            a.job("p4ref", (*common, f"exp={_exp(p, 'p4ref')}", f"fold={f}", f"seed={s}"))
            for a in ref_arms
            for f in p.folds
            for s in p.seeds
        ],
        "stage1_e6": [
            _variant(a, f"frac{round(frac * 100)}").job(
                "e06_data_efficiency",
                (
                    *common,
                    f"exp={_exp(p, 'e06')}",
                    f"data.train_fraction={frac}",
                    f"fold={f}",
                    f"seed={s}",
                ),
            )
            for a in COMPARISON
            for frac in FRACTIONS
            for f in p.folds
            for s in p.seeds
        ],
        "stage2_e7": [
            a.job("e07_transfer", (*common, f"exp={_exp(p, 'e07')}", "fold=0", f"seed={s}"))
            for a in COMPARISON
            for s in p.seeds
        ],
        "stage3_e8": [
            _variant(a, f"no-{t}").job(
                "e08_open_set",
                (*common, f"exp={_exp(p, 'e08')}", f"data.heldout_type={t}", "fold=0", f"seed={s}"),
            )
            for t in p.e8_types
            for a in (*COMPARISON, *ONE_CLASS)
            for s in (p.seeds if a.task != "one_class" else p.seeds[:1])  # one fit per type
        ],
    }
    return jobs


def _run(name: str, jobs: list[list[str]]) -> bool:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    (PIPELINE_DIR / f"{name}.txt").write_text(
        "\n".join(" ".join(j) for j in jobs) + "\n", encoding="utf-8", newline="\n"
    )
    outcomes = run_jobs(jobs, REPO_ROOT / "results" / "job_logs" / f"{name}.log")
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        log.error("%s: %d of %d jobs failed", name, len(failed), len(jobs))
    return not failed


def e9_runs(p: Profile) -> list:
    """Fold-model runs of the five comparison models (ADR-021) for E9."""
    runs_dir = REPO_ROOT / "results" / "runs"
    keys = {a.key for a in COMPARISON}
    runs = [r for r in collect_runs(runs_dir, _exp(p, "p4ref")) if (r.model, r.variant) in keys]
    if p.final_exp != _exp(p, "p4ref"):
        runs += [
            r for r in collect_runs(runs_dir, p.final_exp) if (r.model, r.variant) == FINAL.key
        ]
    return runs


def run_e9(p: Profile) -> bool:
    """Corruption predictions for every comparison fold model (skips finished runs)."""
    paths = OmegaConf.load(CONFIG_DIR / "paths" / "remote.yaml")
    cfg_corr = load_corruption_config()
    ok, report = True, {}
    for r in e9_runs(p):
        if (r.root / CORRUPTION_FILE).is_file():
            report[r.root.name] = "done earlier"
            continue
        recorded = str(OmegaConf.load(r.root / "config.yaml").paths.checkpoints_dir)
        name = "model.joblib" if r.model.startswith("handcrafted") else "best.pt"
        ckpt = find_checkpoint(r.root, recorded, [REPO_ROOT], name)
        if ckpt is None:
            log.error(
                "E9: no %s for %s; copy it into %s", name, r.root.name, REPO_ROOT / "checkpoints"
            )
            report[r.root.name] = "missing checkpoint"
            ok = False
            continue
        try:
            corruption_predictions(r.root, paths, ckpt, cfg_corr)
            report[r.root.name] = "done"
            log.info("E9: %s done", r.root.name)
        except Exception as exc:  # one fold model's failure must not stop the others
            log.exception("E9: %s failed", r.root.name)
            report[r.root.name] = f"failed: {exc}"
            ok = False
    write_json(PIPELINE_DIR / f"{_exp(p, 'e09')}_status.json", report)
    return ok


def run(profile_name: str) -> int:
    """Run Phase 4 end to end; returns a process exit code (0 = everything finished)."""
    p = PROFILES[profile_name]
    ok = True
    for name, jobs in stage_jobs(p).items():
        ok = _run(f"{_exp(p, 'p4')}_{name}", jobs) and ok
    ok = run_e9(p) and ok
    log.info("Phase 4 pipeline finished (%s)", "all ok" if ok else "with failures")
    return 0 if ok else 1
