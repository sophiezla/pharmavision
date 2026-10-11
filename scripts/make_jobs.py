"""Write job lists for an experiment (one run per line, Hydra overrides).

Usage:
    python scripts/make_jobs.py --exp e01   # jobs/e01_cpu.txt, e01_gpu_timing.txt, e01_gpu.txt
"""

from __future__ import annotations

import argparse
import sys

from pharmachecked.config import REPO_ROOT
from pharmachecked.utils.io import atomic_write_text

FOLDS, SEEDS = range(5), range(3)


def e01_arms() -> tuple[list[str], list[str], list[str]]:
    """(cpu arms, gpu arms, gpu zero-shot arms) as override prefixes."""
    exp = "experiment=e01_baselines"
    cpu = [
        f"{exp} model=handcrafted_lr task=handcrafted variant=default",
        f"{exp} model=handcrafted_xgb task=handcrafted variant=default",
    ]
    gpu = [f"{exp} model=cnn_small variant=ft"]
    for model in ("resnet50", "efficientnetv2_s", "convnext_tiny", "vit_b16", "swin_tiny"):
        gpu.append(f"{exp} model={model} variant=ft")
    for model in ("dinov2_vits14", "dinov2_vitb14", "clip_vitb16"):
        gpu.append(f"{exp} model={model} variant=ft")
        gpu.append(f"{exp} model={model} variant=lp model.mode=linear_probe")
    zeroshot = [f"{exp} model=clip_vitb16 task=clip_zeroshot variant=zeroshot"]
    return cpu, gpu, zeroshot


def expand(arms: list[str], folds: range, seeds: range, extra: str = "") -> list[str]:
    """One line per arm x fold x seed."""
    tail = f" {extra}" if extra else ""
    return [f"{a} fold={f} seed={s}{tail}" for a in arms for f in folds for s in seeds]


def main() -> int:
    """Write the job files; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True, choices=["e01"])
    args = parser.parse_args()
    arms = {"e01": e01_arms}
    cpu, gpu, zeroshot = arms[args.exp]()
    remote = "paths=remote train.num_workers=4"
    files = {
        "e01_cpu.txt": ("# E1 tier 1 (laptop CPU)", expand(cpu, FOLDS, SEEDS)),
        "e01_gpu_timing.txt": (
            "# E1 timing batch: fold 0 / seed 0 of every GPU arm (CLAUDE.md section 12)",
            expand(gpu + zeroshot, range(1), range(1), remote),
        ),
        "e01_gpu.txt": (
            "# E1 full protocol on the GPU PC (zero-shot: seed 0 only, it is deterministic)",
            expand(gpu, FOLDS, SEEDS, remote) + expand(zeroshot, FOLDS, range(1), remote),
        ),
    }
    out = REPO_ROOT / "jobs"
    for name, (header, lines) in files.items():
        atomic_write_text(out / name, header + "\n" + "\n".join(lines) + "\n")
        sys.stdout.write(f"jobs/{name}: {len(lines)} jobs\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
