"""Combined Phases 6-8 pipeline (ADR-027): job lists match the pre-registered arms."""

from collections import Counter

from pharmachecked.jobs import compose_job
from pharmachecked.pipeline import combined


def _key(job: list[str], field: str) -> str:
    return next(o.split("=", 1)[1] for o in job if o.startswith(f"{field}="))


def test_full_profile_job_counts() -> None:
    p = combined.PROFILES["full"]
    e14 = combined.e14_jobs(p)
    assert len(e14) == 45
    assert Counter(_key(j, "variant") for j in e14) == dict.fromkeys(
        ["procedural-r1", "procedural-r4", "procedural-only"], 15
    )
    assert len(combined.e16_jobs(p)) == 30
    assert {_key(j, "variant") for j in combined.e16_jobs(p)} == {"kd-t4"}
    assert len(combined.e15_defence_jobs(p)) == 15
    assert p.race_rounds == 3 and len(combined.race_jobs(p, 3)) == 15
    best = {"variant": "procedural-r4", "generator": "procedural", "ratio": 4.0}
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


def test_students_are_all_distilled() -> None:
    p = combined.PROFILES["full"]
    jobs = combined.e16_jobs(p)
    assert all("task=distill" in j for j in jobs) and not any("variant=ce" in j for j in jobs)
