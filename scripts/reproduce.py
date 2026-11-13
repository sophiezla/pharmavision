"""Regenerate every committed table and figure from the committed run folders (PLAN 10.3).

The commands are those recorded in docs/EXPERIMENT_LOG.md, oldest first. Steps that need
model weights (E4 branch outputs) or dataset images (local-only E12 examples) are skipped
unless ``--with-local`` is given. The locked-test table is never regenerated (Rule 2), nor
``e01_timing`` (its timing-batch run folders were removed by design in 9f23bac, ADR-012).

Usage:
    python scripts/reproduce.py [--with-local]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pharmachecked.config import REPO_ROOT

REF = "efficientnetv2-s/default"
STEPS: list[list[str]] = [
    ["aggregate.py", "--exp", "e01"],
    ["make_figures.py", "--exp", "e01"],
    [
        "aggregate.py",
        "--exp",
        "e02",
        "--include",
        "e01:convnext-tiny/ft",
        "--include",
        "e01:efficientnetv2-s/ft",
    ],
    ["make_figures.py", "--exp", "e02"],
    [
        "aggregate.py",
        "--exp",
        "e03",
        "--include",
        "e01:swin-tiny/ft-tuned",
        "--reference",
        "swin-tiny/ft-tuned",
    ],
    ["aggregate.py", "--exp", "e05", "--reference", REF],
    ["aggregate.py", "--exp", "e04", "--include", f"e05:{REF}", "--reference", "multibranch/full"],
    ["gate_table.py", "--exp", "e04", "--model", "multibranch", "--variant", "full"],
    ["make_figures.py", "--exp", "e04"],
    [
        "external_auroc.py",
        "--exp",
        "e01",
        "--exp",
        "e02",
        "--exp",
        "e03",
        "--exp",
        "e04",
        "--exp",
        "e05",
    ],
    [
        "aggregate.py",
        "--exp",
        "e06",
        "--include",
        "p4ref:handcrafted-xgb/default",
        "--include",
        "p4ref:convnext-tiny/ft",
        "--include",
        "p4ref:swin-tiny/ft",
        "--include",
        "p4ref:dinov2-vits14/ft",
        "--include",
        f"e05:{REF}",
    ],
    ["make_figures.py", "--exp", "e06"],
    ["phase4_tables.py", "--exp", "e07"],
    ["phase4_tables.py", "--exp", "e08"],
    ["make_figures.py", "--exp", "e08"],
    ["phase4_tables.py", "--exp", "e09"],
    ["make_figures.py", "--exp", "e09"],
    ["phase5_tables.py", "--exp", "e09ext"],
    ["make_figures.py", "--exp", "e09ext"],
    ["phase5_tables.py", "--exp", "e10"],
    ["make_figures.py", "--exp", "e10"],
    ["phase5_tables.py", "--exp", "e11"],
    ["phase6to8_tables.py", "--exp", "e12"],
    ["aggregate.py", "--exp", "e14", "--include", f"e05:{REF}", "--reference", REF],
    ["phase6to8_tables.py", "--exp", "e14"],
    ["make_figures.py", "--exp", "e14"],
    [
        "aggregate.py",
        "--exp",
        "e15",
        "--include",
        f"e05:{REF}",
        "--include",
        "e14:efficientnetv2-s/procedural-r1",
        "--reference",
        REF,
    ],
    ["aggregate.py", "--exp", "e15race", "--include", f"e05:{REF}", "--reference", REF],
    ["phase6to8_tables.py", "--exp", "e15"],
    ["make_figures.py", "--exp", "e15"],
    ["aggregate.py", "--exp", "e16", "--include", f"e05:{REF}", "--reference", REF],
    ["phase6to8_tables.py", "--exp", "e16"],
    ["make_figures.py", "--exp", "e16"],
    ["release_assets.py"],
]
LOCAL_STEPS: list[list[str]] = [
    ["make_figures.py", "--exp", "e12"],
]


def main() -> int:
    """Run every step in order; returns the first failing step's exit code, else 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-local", action="store_true")
    args = parser.parse_args()
    for step in STEPS + (LOCAL_STEPS if args.with_local else []):
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / step[0]), *step[1:]]
        print("+", " ".join(["python", f"scripts/{step[0]}", *step[1:]]), flush=True)
        code = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
