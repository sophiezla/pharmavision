"""Phases 6-8 on one GPU kit (ADR-027): every remaining GPU experiment, in dependency order.

Pre-registered in docs/experiments/E12.md-E16.md and ADR-027. Phase 5 ran in its own kit
(ADR-027 amendment, 2026-10-25); only the self-test still runs ``pipeline.phase5`` for its toy
reference models. Stages:

1. (self-test only) toy reference models via ``pipeline.phase5``.
2. E14 synthesis: OCR, procedural edits, SD inpainting (``synthesis.generate``).
3. E12 saliency pass: final model, DINOv2-S (rollout), E4 multi-branch (MIL attention).
4. E13 concept bottleneck (3 modes x 15 runs), then TCAV on the final model's seed-0 models.
5. E14 arms (2 generators x ratios 0.5/1/2/4, synthetic-only), the best-arm decision, and the
   E8 open-set re-run with it.
6. E16 distillation (2 students x T in {2, 4}), then the
   ONNX export / INT8 / CPU prediction pass of every student and the final model.
7. E15 defences (PGD training, PGD + E14 best), the attack pass over the final model, the E14
   best arm and both defences, then 5 arms-race rounds.

Every stage is resumable: finished runs are skipped by the job runner, finished per-run files
are kept. A failing stage does not stop independent later stages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.evaluation.aggregate import RunRecord, collect_runs
from pharmachecked.jobs import compose_job
from pharmachecked.pipeline import phase4, phase5
from pharmachecked.pipeline.arms import Arm
from pharmachecked.pipeline.decisions import rank, score_arms
from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)
PIPELINE_DIR = phase4.PIPELINE_DIR
FINAL = phase4.FINAL
LORA = FINAL.overrides
E14_RATIOS = {"r05": 0.5, "r1": 1.0, "r2": 2.0, "r4": 4.0}
GENERATORS = ("procedural", "inpaint")
STUDENTS = ("mobilenetv3_large", "efficientnet_lite0")
TEMPERATURES = {"kd-t2": 2.0, "kd-t4": 4.0}
RACE_ROUNDS = 5
E12_MODELS = {  # (experiment, model slug, variant) -> saliency methods (E12.md)
    "final": ("gradcam", "gradcampp", "ig", "random"),
    "vit": ("rollout", "ig", "random"),
    "mil": ("mil", "ig", "random"),
}


@dataclass(frozen=True)
class Profile:
    """Full protocol or a laptop self-test (toy data, stubs for OCR and SD)."""

    name: str
    base: phase4.Profile
    e14_ratios: dict[str, float]
    students: tuple[str, ...]
    temperatures: dict[str, float]
    race_rounds: int
    synthesis_limit: int | None
    synthesis_overrides: dict[str, object] = field(default_factory=dict)
    refs: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    extra: tuple[str, ...] = ()


PROFILES = {
    "full": Profile(
        "full",
        phase4.PROFILES["full"],
        E14_RATIOS,
        STUDENTS,
        TEMPERATURES,
        RACE_ROUNDS,
        None,
        refs={
            "final": ("e05", "efficientnetv2-s", "default"),
            "vit": ("p4ref", "dinov2-vits14", "ft"),
            "mil": ("e04", "multibranch", "full"),
            "e1_best": ("p4ref", "convnext-tiny", "ft"),
        },
    ),
    "selftest": Profile(
        "selftest",
        phase4.PROFILES["selftest"],
        {"r1": 1.0},
        STUDENTS[:1],
        {"kd-t2": 2.0},
        2,
        6,
        synthesis_overrides={"ocr_backend": "stub", "inpaint.backend": "stub", "per_source": 2},
        refs={
            "final": ("stp4ref", "efficientnetv2-s", "default"),
            "vit": ("stp4ref", "dinov2-vits14", "ft"),
            "mil": ("ste04", "multibranch", "full"),
            "e1_best": ("stp4ref", "convnext-tiny", "ft"),
        },
    ),
}


def _exp(p: Profile, name: str) -> str:
    return f"{p.base.prefix}{name}"


def _jobs(
    p: Profile, arm: Arm, experiment: str, exp: str, extra: tuple[str, ...] = ()
) -> list[list[str]]:
    b = p.base
    return [
        arm.job(experiment, (*b.extra, f"exp={exp}", *extra, f"fold={f}", f"seed={s}"))
        for f in b.folds
        for s in b.seeds
    ]


def refs(p: Profile, key: str) -> list[RunRecord]:
    """Complete runs of a reference model (``Profile.refs``)."""
    exp, model, variant = p.refs[key]
    return [
        r
        for r in collect_runs(REPO_ROOT / "results" / "runs", exp)
        if (r.model, r.variant) == (model, variant)
    ]


def arm_runs(exp: str, model: str, variant: str) -> list[RunRecord]:
    """Complete runs of one arm."""
    return [
        r
        for r in collect_runs(REPO_ROOT / "results" / "runs", exp)
        if (r.model, r.variant) == (model, variant)
    ]


def _pass(
    name: str, runs: list[RunRecord], filename: str, work: Callable[[RunRecord, Path], str]
) -> bool:
    """``phase5._stage`` over run records (skips runs that already have ``filename``)."""

    class _R:  # phase5._stage reads .root and .model
        def __init__(self, r: RunRecord) -> None:
            self.root, self.model, self.rec = r.root, r.model, r

    def wrapped(r: _R, ckpt: Path) -> str:
        return work(r.rec, ckpt)

    return phase5._stage(name, [_R(r) for r in runs], filename, wrapped)


def _paths() -> object:
    return OmegaConf.load(CONFIG_DIR / "paths" / "remote.yaml")


# --- stage 2: synthesis ----------------------------------------------------------------------


def synthesis(p: Profile) -> bool:
    """Generate the E14 synthetic images and the QC sample list (PLAN 7.1-7.4)."""
    from pharmachecked.synthesis.generate import generate, load_synthesis_config

    cfg = load_synthesis_config()
    for key, value in p.synthesis_overrides.items():
        OmegaConf.update(cfg, key, value)
    run_cfg = compose_job(
        ["experiment=e14_synthetic", "model=efficientnetv2_s", *p.base.extra, *p.extra]
    )
    try:
        table = generate(run_cfg, cfg, limit=p.synthesis_limit)
    except Exception:
        log.exception("E14 synthesis failed")
        return False
    qc = (
        table.groupby("generator", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), int(cfg.qc_per_generator)), random_state=42))
        .sort_values(["generator", "image_id"])
    )
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    qc[["image_id", "generator", "edit_type"]].to_csv(
        PIPELINE_DIR / f"{_exp(p, 'e14')}_qc_sample.csv", index=False, lineterminator="\n"
    )
    counts = table.groupby(["generator", "edit_type"]).size().reset_index(name="images")
    counts.to_csv(
        PIPELINE_DIR / f"{_exp(p, 'e14')}_synthesis_counts.csv", index=False, lineterminator="\n"
    )
    return True


# --- stage 3: E12 ------------------------------------------------------------------------------


def e12(p: Profile) -> bool:
    """Saliency pass over the three E12 models."""
    from pharmachecked.interpret.e12_pass import FAITH_FILE, e12_pass

    paths = _paths()
    ok = True
    for key, methods in E12_MODELS.items():

        def work(r: RunRecord, ckpt: Path, methods: tuple[str, ...] = methods) -> str:
            first = r.fold == 0 and r.seed == 0
            e12_pass(
                r.root,
                paths,
                ckpt,
                methods,
                randomization=first,
                examples=first,
                localization=True,
            )
            return "done"

        ok = _pass(f"{_exp(p, 'e12')}_{key}", refs(p, key), FAITH_FILE, work) and ok
    return ok


# --- stage 4: E13 ------------------------------------------------------------------------------


def e13_jobs(p: Profile) -> list[list[str]]:
    """Concept-bottleneck runs: 3 modes x folds x seeds."""
    concepts = []
    if p.name == "selftest":
        concepts = [f"cbm.concepts_file={toy_concepts().as_posix()}"]
    return [
        job
        for mode in ("independent", "sequential", "joint")
        for job in _jobs(
            p,
            Arm("cbm_efficientnetv2_s", mode, "cbm"),
            "e13_cbm",
            _exp(p, "e13"),
            (f"cbm.mode={mode}", *concepts, *p.extra),
        )
    ]


def toy_concepts() -> Path:
    """Random concept labels for the toy development pool (self-test only)."""
    import numpy as np

    from pharmachecked.data.annotations import CONCEPT_COLUMNS, CONCEPTS
    from pharmachecked.data.toy import make_toy_dataset

    paths = _paths()
    toy = make_toy_dataset(Path(paths.toy_dir))
    out = Path(paths.toy_dir) / "concepts.csv"
    ids = pd.read_csv(toy.manifest, dtype={"image_id": str})["image_id"]
    rng = np.random.default_rng(0)
    values = np.array(["present", "absent", "unsure"])
    table = pd.DataFrame({"image_id": ids})
    for c in CONCEPTS:
        table[c] = values[rng.integers(0, 3, len(ids))]
    table["annotator"], table["timestamp"] = "toy", ""
    table[list(CONCEPT_COLUMNS)].to_csv(out, index=False, lineterminator="\n")
    return out


def tcav(p: Profile) -> bool:
    """TCAV on the final model's seed-0 fold models."""
    from pharmachecked.interpret.tcav import TCAV_FILE, tcav_pass

    paths = _paths()
    concepts = (
        toy_concepts() if p.name == "selftest" else REPO_ROOT / "data/annotations/concepts.csv"
    )
    runs = [r for r in refs(p, "final") if r.seed == 0]

    def work(r: RunRecord, ckpt: Path) -> str:
        tcav_pass(r.root, paths, ckpt, concepts)
        return "done"

    return _pass(f"{_exp(p, 'e13')}_tcav", runs, TCAV_FILE, work)


# --- stage 5: E14 ------------------------------------------------------------------------------


def e14_arms(p: Profile) -> dict[str, tuple[str, float, bool]]:
    """Variant -> (generator, ratio, synthetic-only)."""
    arms = {f"{g}-{k}": (g, r, False) for g in GENERATORS for k, r in p.e14_ratios.items()}
    arms |= {f"{g}-only": (g, 1.0, True) for g in GENERATORS}
    return arms


def _syn_overrides(gen: str, ratio: float, only: bool) -> tuple[str, ...]:
    return (
        f"synthetic.generator={gen}",
        f"synthetic.ratio={ratio}",
        f"synthetic.only={str(only).lower()}",
    )


def e14_jobs(p: Profile) -> list[list[str]]:
    """E14 training runs (final-model recipe with synthetic images)."""
    return [
        job
        for variant, (gen, ratio, only) in e14_arms(p).items()
        for job in _jobs(
            p,
            Arm("efficientnetv2_s", variant, "train", LORA),
            "e14_synthetic",
            _exp(p, "e14"),
            (*_syn_overrides(gen, ratio, only), *p.extra),
        )
    ]


def e14_best(p: Profile) -> dict[str, object] | None:
    """Pre-registered decision (E14.md): the best ratio arm by pooled OOF AUROC (ties: log-loss).

    Synthetic-only arms are not candidates. Recorded in ``results/pipeline``.
    """
    ratio_arms = {v: a for v, a in e14_arms(p).items() if not a[2]}
    records = [
        r
        for r in collect_runs(REPO_ROOT / "results" / "runs", _exp(p, "e14"))
        if r.variant in ratio_arms
    ]
    if not records:
        return None
    tiers = {("efficientnetv2-s", v): 0 for v in ratio_arms}
    ranked = rank(score_arms(records, tiers))
    best = ranked[0]
    gen, ratio, _ = ratio_arms[best.variant]
    decision = {
        "variant": best.variant,
        "generator": gen,
        "ratio": ratio,
        "auroc": best.auroc,
        "log_loss": best.log_loss,
        "runs": best.runs,
        "ranking": [
            {"variant": s.variant, "runs": s.runs, "auroc": s.auroc, "log_loss": s.log_loss}
            for s in ranked
        ],
    }
    write_json(PIPELINE_DIR / f"{_exp(p, 'e14')}_best.json", decision)
    return decision


def e14_open_set_jobs(p: Profile, best: dict[str, object]) -> list[list[str]]:
    """E8 leave-one-packaging-type-out re-run with the best E14 arm (fold unused, 3 seeds)."""
    arm = Arm("efficientnetv2_s", f"{best['variant']}-no-x", "train", LORA)
    jobs = []
    for t in p.base.e8_types:
        a = Arm(arm.model, f"{best['variant']}-no-{t}", arm.task, arm.overrides)
        for s in p.base.seeds:
            jobs.append(
                a.job(
                    "e14_synthetic",
                    (
                        *p.base.extra,
                        f"exp={_exp(p, 'e14os')}",
                        *_syn_overrides(str(best["generator"]), float(best["ratio"]), False),
                        "data.protocol=leave_type_out",
                        f"data.heldout_type={t}",
                        "fold=0",
                        f"seed={s}",
                    ),
                )
            )
    return jobs


# --- stage 6: E16 ------------------------------------------------------------------------------


def e16_jobs(p: Profile) -> list[list[str]]:
    """Students distilled at each temperature (the cross-entropy-only arm was dropped, ADR-027)."""
    teacher = p.refs["final"][0]
    jobs = []
    for student in p.students:
        for variant, t in p.temperatures.items():
            jobs += _jobs(
                p,
                Arm(student, variant, "distill"),
                "e16_distill",
                _exp(p, "e16"),
                (f"distill.temperature={t}", f"distill.teacher_exp={teacher}", *p.extra),
            )
    return jobs


def deploy(p: Profile) -> bool:
    """ONNX export, INT8 quantization and CPU predictions of every student and the final model."""
    from pharmachecked.deploy.evaluate import DEPLOY_JSON, deploy_pass

    paths = _paths()
    runs = collect_runs(REPO_ROOT / "results" / "runs", _exp(p, "e16")) + refs(p, "final")

    def work(r: RunRecord, ckpt: Path) -> str:
        rec = deploy_pass(r.root, paths, ckpt)
        return f"done (max abs diff {rec['max_abs_diff_fp32']:.2e})"

    return _pass(f"{_exp(p, 'e16')}_deploy", runs, DEPLOY_JSON, work)


# --- stage 7: E15 ------------------------------------------------------------------------------


def e15_defence_jobs(p: Profile, best: dict[str, object] | None) -> list[list[str]]:
    """PGD adversarial training alone and on top of the best E14 arm."""
    jobs = _jobs(
        p, Arm("efficientnetv2_s", "adv", "adv_train", LORA), "e15_defence", _exp(p, "e15"), p.extra
    )
    if best is not None:
        jobs += _jobs(
            p,
            Arm("efficientnetv2_s", "both", "adv_train", LORA),
            "e15_defence",
            _exp(p, "e15"),
            (*_syn_overrides(str(best["generator"]), float(best["ratio"]), False), *p.extra),
        )
    return jobs


def attacks(p: Profile, best: dict[str, object] | None) -> bool:
    """E15 attack pass: final model, E14 best arm, PGD defence, PGD + E14 defence."""
    from pharmachecked.adversarial.evaluate import ADV_FILE, adversarial_pass

    paths = _paths()
    runs = refs(p, "final")
    runs += arm_runs(_exp(p, "e15"), "efficientnetv2-s", "adv")
    runs += arm_runs(_exp(p, "e15"), "efficientnetv2-s", "both")
    if best is not None:
        runs += arm_runs(_exp(p, "e14"), "efficientnetv2-s", str(best["variant"]))

    def work(r: RunRecord, ckpt: Path) -> str:
        out = adversarial_pass(r.root, paths, ckpt)
        return f"done ({out['image_id'].nunique()} counterfeits)"

    return _pass(f"{_exp(p, 'e15')}_attacks", runs, ADV_FILE, work)


def race_jobs(p: Profile, r: int) -> list[list[str]]:
    """One arms-race round over every fold and seed."""
    base_exp, base_model, base_variant = p.refs["final"]
    return _jobs(
        p,
        Arm("efficientnetv2_s", f"round{r}", "race", LORA),
        "e15_race",
        _exp(p, "e15race"),
        (
            f"race.round={r}",
            f"race.base_exp={base_exp}",
            f"race.base_model={base_model}",
            f"race.base_variant={base_variant}",
            *p.extra,
        ),
    )


# --- driver ------------------------------------------------------------------------------------


def selftest_e04(p: Profile) -> bool:
    """Self-test: toy E4 multi-branch models (the toy p4ref arms come from ``phase5.run``)."""
    b = p.base
    mb = Arm("multibranch", "full")
    jobs = [
        mb.job("e04_multibranch", (*b.extra, "exp=ste04", f"fold={f}", "seed=0")) for f in b.folds
    ]
    return phase4._run("st_c_stage0_e04", jobs)


def run(profile_name: str) -> int:
    """Run Phases 6-8 end to end; returns a process exit code (0 = everything finished)."""
    p = PROFILES[profile_name]
    pre = _exp(p, "c")
    ok = True
    if p.name == "selftest":  # toy reference models; the full profile reuses the Phase 5 runs
        ok = phase5.run(p.base.name) == 0 and ok
        ok = selftest_e04(p) and ok
    syn_ok = synthesis(p)
    ok = syn_ok and ok
    ok = e12(p) and ok
    ok = phase4._run(f"{pre}_e13", e13_jobs(p)) and ok
    ok = tcav(p) and ok
    best = None
    if syn_ok:
        ok = phase4._run(f"{pre}_e14", e14_jobs(p)) and ok
        best = e14_best(p)
        if best is not None and p.base.e8_types:
            ok = phase4._run(f"{pre}_e14os", e14_open_set_jobs(p, best)) and ok
    ok = phase4._run(f"{pre}_e16", e16_jobs(p)) and ok
    ok = deploy(p) and ok
    ok = phase4._run(f"{pre}_e15_defences", e15_defence_jobs(p, best)) and ok
    ok = attacks(p, best) and ok
    for r in range(1, p.race_rounds + 1):
        ok = phase4._run(f"{pre}_e15_race_round{r}", race_jobs(p, r)) and ok
    log.info("combined pipeline finished (%s)", "all ok" if ok else "with failures")
    return 0 if ok else 1
