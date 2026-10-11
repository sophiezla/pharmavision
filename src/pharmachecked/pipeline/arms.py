"""Experiment arms: which config, task, variant and overrides each arm runs with."""

from __future__ import annotations

from dataclasses import dataclass, field

from pharmachecked.utils.run_dir import slug


@dataclass(frozen=True)
class Arm:
    """One arm of an experiment (one row of its summary table)."""

    model: str  # config name under configs/model/
    variant: str
    task: str = "train"
    overrides: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str]:
        """``(model slug, variant slug)`` as they appear in run ids."""
        return slug(self.model), slug(self.variant)

    def job(self, experiment: str, extra: tuple[str, ...] = ()) -> list[str]:
        """Hydra overrides of this arm (without fold/seed)."""
        return [
            f"experiment={experiment}",
            f"model={self.model}",
            f"task={self.task}",
            f"variant={self.variant}",
            *self.overrides,
            *extra,
        ]


LP = ("model.mode=linear_probe",)
E1_CPU = (
    Arm("handcrafted_lr", "default", "handcrafted"),
    Arm("handcrafted_xgb", "default", "handcrafted"),
)
E1_GPU = (
    Arm("cnn_small", "ft"),
    Arm("resnet50", "ft"),
    Arm("efficientnetv2_s", "ft"),
    Arm("convnext_tiny", "ft"),
    Arm("vit_b16", "ft"),
    Arm("swin_tiny", "ft"),
    Arm("dinov2_vits14", "ft"),
    Arm("dinov2_vits14", "lp", overrides=LP),
    Arm("dinov2_vitb14", "ft"),
    Arm("dinov2_vitb14", "lp", overrides=LP),
    Arm("clip_vitb16", "ft"),
    Arm("clip_vitb16", "lp", overrides=LP),
)
E1_ZEROSHOT = (Arm("clip_vitb16", "zeroshot", "clip_zeroshot"),)
E1_ALL = E1_CPU + E1_GPU + E1_ZEROSHOT


def e1_arm(key: tuple[str, str]) -> Arm:
    """The E1 arm with this ``(model slug, variant)`` key."""
    for arm in E1_ALL:
        if arm.key == key:
            return arm
    raise KeyError(key)


def e2_arms(backbone: str, has_lp_in_e1: bool) -> list[Arm]:
    """E2 strategies for one backbone that E1 has not already run (PLAN 2.10).

    Full fine-tuning (and the linear probe, when E1 ran it) are taken from E1. LoRA trains its
    adapters at the head learning rate (backbone_lr_mult 1.0); last-2-blocks keeps the full
    fine-tuning multiplier (0.1). Pre-registered in docs/experiments/E02.md.
    """
    arms = []
    if not has_lp_in_e1:
        arms.append(Arm(backbone, "lp", overrides=LP))
    arms.append(Arm(backbone, "last2", overrides=("model.mode=last_k", "model.trainable_blocks=2")))
    for rank in (8, 16):
        arms.append(
            Arm(
                backbone,
                f"lora{rank}",
                overrides=(
                    "model.mode=lora",
                    f"model.lora_rank={rank}",
                    "train.backbone_lr_mult=1.0",
                ),
            )
        )
    return arms
