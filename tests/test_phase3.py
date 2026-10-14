"""Phase 3 pipeline: the pre-registered E5 rule and the job lists."""

from pharmachecked.pipeline.phase3 import (
    E4_ARMS,
    E5_COMPONENTS,
    PROFILES,
    component_overrides,
    select_components,
)


def _row(name: str, gain: float, low: float, p: float) -> dict:
    return {"component": name, "auroc_gain": gain, "ci_low": low, "delong_p": p}


def test_select_components_rule_and_exclusive_pairs() -> None:
    rows = [
        _row("hflip", 0.010, 0.002, 0.0001),  # improves
        _row("mixup", 0.004, 0.001, 0.0002),  # improves, but cutmix gains more
        _row("cutmix", 0.008, 0.003, 0.0003),  # improves
        _row("focal", 0.006, -0.001, 0.0001),  # CI includes 0
        _row("smoothing", 0.005, 0.001, 0.2),  # not significant after Holm
        _row("supcon", -0.010, -0.020, 0.0001),  # significantly worse
    ]
    assert select_components(rows) == ["hflip", "cutmix"]
    assert [r["improves"] for r in rows] == [True, True, True, False, False, False]
    assert all(r["holm_p"] >= r["delong_p"] for r in rows)


def test_select_components_none() -> None:
    assert select_components([_row(n, 0.0, 0.0, 1.0) for n in E5_COMPONENTS]) == []


def test_profiles() -> None:
    full, selftest = PROFILES["full"], PROFILES["selftest"]
    assert component_overrides(full)["supcon"] == ("train.supcon_epochs=20",)
    assert component_overrides(selftest)["supcon"] == ("train.supcon_epochs=1",)
    assert not any("supcon" in o for o in full.extra + selftest.extra)
    assert full.e4_arms == tuple(E4_ARMS) and full.e4_arms[-1] == "tiles4-896"
    assert len(full.folds) * len(full.seeds) == 15
