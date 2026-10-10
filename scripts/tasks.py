"""Cross-platform task runner; `make <target>` delegates here.

Usage:
    python scripts/tasks.py setup [--torch-index URL] [--no-hooks] [--no-constraints]
    python scripts/tasks.py lint | test | smoke | freeze
    python scripts/tasks.py aggregate --exp e01
    python scripts/tasks.py figures --exp e01

Every target except ``setup`` re-executes itself with the ``.venv`` interpreter when one exists.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
CONSTRAINTS = ROOT / "constraints" / "cpu.txt"
# Packages installed from the PyTorch index; their local version labels (+cpu, +cu128) are
# stripped in the constraints file so the same pins accept CPU and CUDA builds.
TORCH_PACKAGES = ("torch", "torchvision")


def venv_python() -> Path:
    """Return the path of the venv interpreter for this platform."""
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return VENV / sub


def run(cmd: list[str | Path]) -> None:
    """Run a command from the repo root, echoing it; exit with its code on failure."""
    printable = " ".join(str(c) for c in cmd)
    sys.stdout.write(f"+ {printable}\n")
    sys.stdout.flush()
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def in_venv() -> bool:
    """True when the running interpreter is the repo venv."""
    return Path(sys.prefix).resolve() == VENV.resolve()


def reexec_in_venv() -> None:
    """Re-run this script with the venv interpreter if it exists and is not already active."""
    if not in_venv() and venv_python().exists():
        sys.exit(subprocess.run([str(venv_python()), *sys.argv], check=False).returncode)


def setup(args: argparse.Namespace) -> None:
    """Create .venv, install torch, the package with [dev], and pre-commit hooks."""
    # Runs before the venv exists, under whatever interpreter the user invoked.
    if sys.version_info[:2] < (3, 13):  # noqa: UP036
        sys.exit(f"Python 3.13+ required, found {sys.version.split()[0]}")
    if not venv_python().exists():
        run([sys.executable, "-m", "venv", VENV])
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    use_constraints = CONSTRAINTS.exists() and not args.no_constraints
    constraint_args: list[str | Path] = ["-c", CONSTRAINTS] if use_constraints else []
    run(
        [
            py,
            "-m",
            "pip",
            "install",
            *TORCH_PACKAGES,
            "--index-url",
            args.torch_index,
            *constraint_args,
        ]
    )
    run([py, "-m", "pip", "install", "-e", ".[dev]", *constraint_args])
    if not args.no_hooks:
        run([py, "-m", "pre_commit", "install"])


def freeze(_: argparse.Namespace) -> None:
    """Write constraints/cpu.txt from the current environment (without the editable package)."""
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = []
    for line in out.splitlines():
        name = re.split(r"[=<>@ ]", line, maxsplit=1)[0].lower()
        if name in TORCH_PACKAGES:
            line = re.sub(r"\+[A-Za-z0-9.]+$", "", line)
        lines.append(line)
    CONSTRAINTS.parent.mkdir(exist_ok=True)
    header = f"# pip freeze, Python {sys.version.split()[0]}, {sys.platform}\n"
    CONSTRAINTS.write_text(header + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(f"wrote {CONSTRAINTS.relative_to(ROOT)} ({len(lines)} pins)\n")


def lint(_: argparse.Namespace) -> None:
    """ruff check and ruff format --check."""
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "ruff", "format", "--check", "."])


def test(args: argparse.Namespace) -> None:
    """pytest -q (fast tests unless extra pytest args are passed)."""
    run([sys.executable, "-m", "pytest", "-q", *args.pytest_args])


SMOKE_LIMIT_SECONDS = 180


def smoke(_: argparse.Namespace) -> None:
    """PLAN 1.11: tiny CNN, 1 epoch, fold 0; real data if processed, else the toy set (CI).

    Fails if the run folder is incomplete or the run takes longer than 3 minutes.
    """
    import time

    real = (ROOT / "splits" / "v1.json").exists() and any(
        (ROOT / "data" / "processed" / "images").glob("*.png")
    )
    dataset = "real" if real else "toy"
    runs = ROOT / "results" / "scratch" / "runs"
    before = set(runs.glob("smoke_*")) if runs.exists() else set()
    start = time.perf_counter()
    run([sys.executable, "-m", "pharmachecked.run", "experiment=smoke", f"data.dataset={dataset}"])
    elapsed = time.perf_counter() - start
    from pharmachecked.utils.run_dir import missing_files

    (new,) = set(runs.glob("smoke_*")) - before
    missing = missing_files(new)
    sys.stdout.write(
        f"smoke ({dataset}): {new.name}, {elapsed:.0f} s, missing: {missing or 'none'}\n"
    )
    if missing or elapsed > SMOKE_LIMIT_SECONDS:
        sys.exit(1)


def not_yet(step: str) -> Callable[[argparse.Namespace], None]:
    """Build a task that exits with a pointer to the plan step that implements it."""

    def task(_: argparse.Namespace) -> None:
        sys.exit(f"not implemented yet: arrives in docs/PLAN.md step {step}")

    return task


TASKS = {
    "setup": setup,
    "freeze": freeze,
    "lint": lint,
    "test": test,
    "smoke": smoke,
    "aggregate": not_yet("2.2"),
    "figures": not_yet("2.4"),
}


def main() -> None:
    """Parse the target and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=sorted(TASKS))
    parser.add_argument("--torch-index", default=CPU_TORCH_INDEX)
    parser.add_argument("--no-hooks", action="store_true")
    parser.add_argument("--no-constraints", action="store_true")
    parser.add_argument("--exp", default=None)
    args, extra = parser.parse_known_args()
    args.pytest_args = extra
    if args.target != "setup":
        reexec_in_venv()
    TASKS[args.target](args)


if __name__ == "__main__":
    main()
