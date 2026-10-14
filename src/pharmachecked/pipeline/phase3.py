"""Phase 3 on one GPU kit: [E3 retrain] -> E5 recipe -> recipe decision -> E4 arms (ADR-013).

Pre-registered in docs/experiments/E05.md and E04.md. Every stage is resumable; the recipe
decision is recomputed deterministically from the E5 run folders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pharmachecked.config import REPO_ROOT
from pharmachecked.evaluation.aggregate import collect_runs, pooled_oof
from pharmachecked.evaluation.metrics import auroc
from pharmachecked.evaluation.stats import bootstrap_ci, delong_test, holm
from pharmachecked.jobs import run_jobs
from pharmachecked.pipeline.arms import Arm
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)
PIPELINE_DIR = REPO_ROOT / "results" / "pipeline"

GLOBAL = (
    "model=efficientnetv2_s",
    "model.mode=lora",
    "model.lora_rank=8",
    "train.backbone_lr_mult=1.0",
)
# E5 components: one change each relative to the default recipe (PLAN 3.2).
E5_COMPONENTS: dict[str, tuple[str, ...]] = {
    "hflip": ("augment.hflip=true",),
    "colorjitter": ("augment.color_strong=true",),
    "randaugment": ("augment.randaugment=true",),
    "mixup": ("train.mix=mixup", "train.mix_alpha=0.2"),
    "cutmix": ("train.mix=cutmix", "train.mix_alpha=1.0"),
    "focal": ("train.loss=focal", "train.focal_gamma=2.0"),
    "smoothing": ("train.label_smoothing=0.1",),
    "supcon": (),  # + train.supcon_epochs=<profile.supcon_epochs> (20 in the full protocol)
}
# Components that cannot be combined: the larger AUROC gain wins (pre-registered).
EXCLUSIVE = ({"mixup", "cutmix"}, {"focal", "smoothing"})
# E4 arms: flags relative to the full model (PLAN 3.6, ADR-017).
E4_ARMS: dict[str, tuple[str, ...]] = {
    "full": (),
    "no-local": ("model.use_local=false",),
    "no-frequency": ("model.use_frequency=false",),
    "no-router": ("model.use_router=false",),
    "concat": ("model.fusion=concat",),
    "separate-local": ("model.shared_local_backbone=false",),
    "hard-routing": ("model.routing=hard",),
    # Last: about 17 backbone passes per image, so it is the slowest arm by far.
    "tiles4-896": (
        "model.tile_grid=4",
        "model.img_size=896",
        "data.img_size=896",
        "train.grad_checkpointing=true",
    ),
}
E4_BASE = (
    "model=multibranch",
    "data.img_size=448",
    "train.backbone_lr_mult=1.0",
    "train.batch_size=8",
    "train.eval_batch_size=16",
)


@dataclass(frozen=True)
class Profile:
    """Full protocol or a tiny laptop self-test."""

    name: str
    prefix: str  # experiment-name prefix: "e" (e03/e04/e05) or "st" (st03/...)
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    extra: tuple[str, ...]
    e4_arms: tuple[str, ...]
    n_boot: int
    supcon_epochs: int


PROFILES = {
    "full": Profile(
        "full",
        "e",
        (0, 1, 2, 3, 4),
        (0, 1, 2),
        ("paths=remote", "train.num_workers=4"),
        tuple(E4_ARMS),
        2000,
        20,
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
        ("full", "no-frequency"),
        200,
        1,
    ),
}


def component_overrides(p: Profile) -> dict[str, tuple[str, ...]]:
    """E5 component overrides with the profile's SupCon length filled in."""
    return {
        name: (f"train.supcon_epochs={p.supcon_epochs}",) if name == "supcon" else o
        for name, o in E5_COMPONENTS.items()
    }


def _lines(
    arms: list[Arm], experiment: str, exp: str, p: Profile, data: tuple[str, ...]
) -> list[list[str]]:
    return [
        arm.job(experiment, (*p.extra, *data, f"exp={exp}", f"fold={f}", f"seed={s}"))
        for arm in arms
        for f in p.folds
        for s in p.seeds
    ]


def _run(name: str, jobs: list[list[str]], stop_on_failure: bool = True) -> bool:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    (PIPELINE_DIR / f"{name}.txt").write_text(
        "\n".join(" ".join(j) for j in jobs) + "\n", encoding="utf-8", newline="\n"
    )
    outcomes = run_jobs(jobs, REPO_ROOT / "results" / "job_logs" / f"{name}.log")
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        log.error("%s: %d of %d jobs failed", name, len(failed), len(jobs))
    return not failed or not stop_on_failure


def improvement(
    default: np.ndarray, arm: np.ndarray, y: np.ndarray, n_boot: int
) -> tuple[float, float, float, float]:
    """(AUROC gain, CI low, CI high, DeLong p) of an arm over the default (paired, pooled OOF)."""
    test = delong_test(y, arm, default)
    ci = bootstrap_ci(
        y, (arm, default), lambda yy, a, b: auroc(yy, a) - auroc(yy, b), n_boot=n_boot
    )
    return ci.estimate, ci.low, ci.high, test.p_value


def select_components(rows: list[dict]) -> list[str]:
    """Apply the E5 rule to per-component rows (``component, auroc_gain, ci_low, delong_p``).

    Adds ``holm_p`` and ``improves`` to each row; returns the kept components in row order,
    keeping only the larger AUROC gain within each exclusive pair.
    """
    reject, corrected = holm([r["delong_p"] for r in rows])
    for row, rej, cp in zip(rows, reject, corrected, strict=True):
        row["holm_p"] = cp
        row["improves"] = bool(rej and row["ci_low"] > 0 and row["auroc_gain"] > 0)
    kept = [r["component"] for r in rows if r["improves"]]
    for group in EXCLUSIVE:
        clash = [r for r in rows if r["component"] in group and r["component"] in kept]
        if len(clash) > 1:
            best = max(clash, key=lambda r: r["auroc_gain"])
            kept = [k for k in kept if k not in group or k == best["component"]]
    return kept


def decide_recipe(exp: str, n_boot: int, components: dict[str, tuple[str, ...]]) -> dict:
    """Pre-registered E5 rule: keep components that improve pooled AUROC over the default."""
    records = collect_runs(REPO_ROOT / "results" / "runs", exp)
    pooled = {
        name: pooled_oof([r for r in records if r.variant == name])
        for name in ("default", *E5_COMPONENTS)
    }
    base = pooled["default"].set_index("image_id")
    rows = []
    for name in E5_COMPONENTS:
        arm = pooled[name].set_index("image_id").loc[base.index]
        gain, low, high, p = improvement(
            base["prob"].to_numpy(), arm["prob"].to_numpy(), base["y_true"].to_numpy(), n_boot
        )
        rows.append(
            {"component": name, "auroc_gain": gain, "ci_low": low, "ci_high": high, "delong_p": p}
        )
    kept = select_components(rows)
    overrides = [o for k in kept for o in components[k]]
    return {
        "rule": "component improves if Holm-corrected DeLong p < 0.05 and its paired bootstrap "
        "AUROC-gain CI excludes 0 (CLAUDE.md section 5); exclusive pairs keep the larger gain",
        "components": rows,
        "kept": kept,
        "recipe_overrides": overrides,
        "needs_verification_run": len(kept) >= 2,
    }


def run(profile_name: str, split_version: int = 1, e3_retrain: bool = False) -> int:
    """Run Phase 3 end to end; returns a process exit code (0 = everything finished)."""
    p = PROFILES[profile_name]
    data = (f"data.split_version={split_version}",)
    e03, e05, e04 = f"{p.prefix}03", f"{p.prefix}05", f"{p.prefix}04"

    # Stage 0 (PLAN 2.14): retrain the E1-best arm on the audited data version.
    if e3_retrain:
        best = Arm(
            "swin_tiny",
            "ft-tuned-v2",
            overrides=(
                "train.backbone_lr_mult=0.8459",
                "train.lr=7.502e-05",
                "train.weight_decay=0.1261",
            ),
        )
        if not _run(f"{e03}_stage0_retrain", _lines([best], "e01_baselines", e03, p, data)):
            return 1

    # Stage 1: E5 recipe arms on the global-only model (ADR-015).
    e5_arms = [Arm("efficientnetv2_s", "default", overrides=GLOBAL[1:])]
    components = component_overrides(p)
    e5_arms += [Arm("efficientnetv2_s", n, overrides=GLOBAL[1:] + o) for n, o in components.items()]
    if not _run(f"{e05}_stage1_e5", _lines(e5_arms, "e05_recipe", e05, p, data)):
        return 1

    # Stage 2: recipe decision (+ verification run of a combination).
    decision = decide_recipe(e05, p.n_boot, components)
    write_json(PIPELINE_DIR / f"{e05}_decisions.json", decision)
    recipe = tuple(decision["recipe_overrides"])
    log.info("E5 recipe: %s", decision["kept"] or "default")
    if decision["needs_verification_run"]:
        combo = Arm("efficientnetv2_s", "combo", overrides=GLOBAL[1:] + recipe)
        if not _run(f"{e05}_stage2_combo", _lines([combo], "e05_recipe", e05, p, data)):
            return 1

    # Stage 3: E4 arms with the E5 recipe; one arm's failure does not stop the others.
    e4 = [Arm("multibranch", n, overrides=E4_BASE[1:] + E4_ARMS[n] + recipe) for n in p.e4_arms]
    ok = _run(
        f"{e04}_stage3_e4", _lines(e4, "e04_multibranch", e04, p, data), stop_on_failure=False
    )
    write_json(
        PIPELINE_DIR / f"{e04}_arms.json",
        {
            "arms": [a.variant for a in e4],
            "recipe": list(recipe),
            "global_reference": f"{e05}: "
            + (
                "combo"
                if decision["needs_verification_run"]
                else (decision["kept"][0] if decision["kept"] else "default")
            ),
        },
    )
    log.info("Phase 3 pipeline finished")
    return 0 if ok else 1
