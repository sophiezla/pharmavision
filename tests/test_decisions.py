"""Pre-registered ranking: AUROC first, exact ties by log-loss; tier best; top backbones."""

from pharmachecked.pipeline.decisions import ArmScore, rank, tier_best, top_backbones


def _s(model: str, variant: str, tier: int, a: float, loss: float) -> ArmScore:
    return ArmScore(model, variant, tier, 15, a, loss)


SCORES = [
    _s("resnet50", "ft", 3, 0.99, 0.10),
    _s("convnext-tiny", "ft", 3, 1.0, 0.08),
    _s("vit-b16", "ft", 4, 1.0, 0.05),
    _s("swin-tiny", "ft", 4, 0.995, 0.01),
    _s("clip-vitb16", "lp", 5, 1.0, 0.20),
    _s("clip-vitb16", "ft", 5, 1.0, 0.03),
    _s("cnn-small", "ft", 2, 0.95, 0.30),
    _s("handcrafted-xgb", "default", 1, 0.977, 0.12),
]


def test_rank_uses_log_loss_only_for_exact_auroc_ties() -> None:
    ranked = [f"{s.model}/{s.variant}" for s in rank(SCORES)]
    # AUROC 1.0 group ordered by loss, then swin (higher loss but lower AUROC is still behind).
    assert ranked[:5] == [
        "clip-vitb16/ft",
        "vit-b16/ft",
        "convnext-tiny/ft",
        "clip-vitb16/lp",
        "swin-tiny/ft",
    ]


def test_tier_best() -> None:
    best = {t: f"{s.model}/{s.variant}" for t, s in tier_best(SCORES).items()}
    assert best == {
        1: "handcrafted-xgb/default",
        2: "cnn-small/ft",
        3: "convnext-tiny/ft",
        4: "vit-b16/ft",
        5: "clip-vitb16/ft",
    }


def test_top_backbones_are_distinct_and_pretrained_only() -> None:
    assert top_backbones(SCORES) == ["clip-vitb16", "vit-b16"]
    assert "cnn-small" not in top_backbones(SCORES, n=10)
