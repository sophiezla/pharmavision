"""Phase 2 decision rules, fixed before the runs (docs/experiments/E01.md, E02.md; ADR-013).

Ranking rule (all decisions): highest pooled OOF AUROC (seed-averaged probabilities over the
development pool); exact ties (|difference| <= 1e-9) go to the lower pooled OOF log-loss. Only
development-pool predictions are used; external sets never enter a decision (ADR-005).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from pharmachecked.evaluation.aggregate import RunRecord, pooled_oof
from pharmachecked.evaluation.metrics import auroc, log_loss

TIE = 1e-9
# E1 tiers of each arm (model slug, variant). Tier 6 (zero-shot) is not tuned or ranked.
E1_TIERS: dict[tuple[str, str], int] = {
    ("handcrafted-lr", "default"): 1,
    ("handcrafted-xgb", "default"): 1,
    ("cnn-small", "ft"): 2,
    ("resnet50", "ft"): 3,
    ("efficientnetv2-s", "ft"): 3,
    ("convnext-tiny", "ft"): 3,
    ("vit-b16", "ft"): 4,
    ("swin-tiny", "ft"): 4,
    ("dinov2-vits14", "ft"): 5,
    ("dinov2-vits14", "lp"): 5,
    ("dinov2-vitb14", "ft"): 5,
    ("dinov2-vitb14", "lp"): 5,
    ("clip-vitb16", "ft"): 5,
    ("clip-vitb16", "lp"): 5,
}
PRETRAINED_TIERS = (3, 4, 5)


@dataclass(frozen=True)
class ArmScore:
    """Pooled development-pool score of one arm."""

    model: str
    variant: str
    tier: int
    runs: int
    auroc: float
    log_loss: float


def score_arms(records: list[RunRecord], tiers: dict[tuple[str, str], int]) -> list[ArmScore]:
    """Pooled OOF AUROC and log-loss of every arm listed in ``tiers``."""
    by_arm: dict[tuple[str, str], list[RunRecord]] = {}
    for r in records:
        if (r.model, r.variant) in tiers:
            by_arm.setdefault((r.model, r.variant), []).append(r)
    scores = []
    for (model, variant), recs in sorted(by_arm.items()):
        pooled = pooled_oof(recs)
        scores.append(
            ArmScore(
                model,
                variant,
                tiers[(model, variant)],
                len(recs),
                auroc(pooled["y_true"], pooled["prob"]),
                log_loss(pooled["y_true"], pooled["prob"]),
            )
        )
    return scores


def rank(scores: list[ArmScore]) -> list[ArmScore]:
    """Best first: higher AUROC, then (for ties within 1e-9) lower log-loss, then name."""

    def key(s: ArmScore) -> tuple[float, float, str]:
        return (-round(s.auroc / TIE) * TIE, s.log_loss, f"{s.model}/{s.variant}")

    return sorted(scores, key=key)


def tier_best(scores: list[ArmScore]) -> dict[int, ArmScore]:
    """The top-ranked arm of each tier."""
    best: dict[int, ArmScore] = {}
    for s in rank(scores):
        best.setdefault(s.tier, s)
    return dict(sorted(best.items()))


def top_backbones(scores: list[ArmScore], n: int = 2) -> list[str]:
    """The ``n`` best distinct pretrained backbones (tiers 3-5), each judged by its best arm."""
    out: list[str] = []
    for s in rank([s for s in scores if s.tier in PRETRAINED_TIERS]):
        if s.model not in out:
            out.append(s.model)
        if len(out) == n:
            break
    return out


def as_table(scores: list[ArmScore]) -> pd.DataFrame:
    """Ranked scores as a DataFrame (for the decisions record)."""
    return pd.DataFrame([asdict(s) for s in rank(scores)])
