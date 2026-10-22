"""Combined Phases 5-8 pipeline (ADR-027): job lists match the pre-registered arms."""

from collections import Counter

from pharmachecked.jobs import compose_job
from pharmachecked.pipeline import combined


def _key(job: list[str], field: str) -> str:
    return next(o.split("=", 1)[1] for o in job if o.startswith(f"{field}="))


def test_full_profile_job_counts() -> None:
    p = combined.PROFILES["full"]
    assert len(combined.e13_jobs(p)) == 45
    e14 = combined.e14_jobs(p)
    assert len(e14) == 150
    assert Counter(_key(j, "variant") for j in e14) == dict.fromkeys(
        [f"{g}-{r}" for g in ("procedural", "inpaint") for r in ("r05", "r1", "r2", "r4", "only")],
        15,
    )
    assert len(combined.e16_jobs(p)) == 90
    best = {"variant": "inpaint-r2", "generator": "inpaint", "ratio": 2.0}
    assert len(combined.e15_defence_jobs(p, best)) == 30
    assert len(combined.e15_defence_jobs(p, None)) == 15
    assert len(combined.race_jobs(p, 3)) == 15
    open_set = combined.e14_open_set_jobs(p, best)
    assert len(open_set) == 6
    assert {_key(j, "data.heldout_type") for j in open_set} == {"blister", "packaging"}


def test_jobs_compose_to_the_intended_runs() -> None:
    p = combined.PROFILES["full"]
    cfg = compose_job(next(j for j in combined.e14_jobs(p) if "variant=procedural-only" in j))
    assert cfg.exp == "e14" and cfg.synthetic.only and cfg.synthetic.ratio == 1.0
    assert cfg.model.mode == "lora" and cfg.train.backbone_lr_mult == 1.0
    cfg = compose_job(combined.race_jobs(p, 2)[0])
    assert cfg.task == "race" and cfg.race.round == 2 and cfg.race.base_exp == "e05"
    cfg = compose_job(combined.e16_jobs(p)[0])
    assert cfg.task == "distill" and cfg.distill.teacher_exp == "e05"
    assert cfg.model.name == "mobilenetv3_large" and cfg.model.mode == "full"
    cfg = compose_job(combined.e13_jobs(p)[0])
    assert cfg.task == "cbm" and cfg.cbm.mode == "independent"


def test_ce_students_use_plain_training() -> None:
    p = combined.PROFILES["full"]
    ce = [j for j in combined.e16_jobs(p) if "variant=ce" in j]
    assert len(ce) == 30 and all("task=train" in j for j in ce)
