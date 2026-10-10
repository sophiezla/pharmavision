# PharmaChecked v2

Research codebase for detecting counterfeit medicine packaging from photographs. It benchmarks a
ladder of baselines (handcrafted features, CNNs, ViTs, foundation models), builds a multi-branch
model (global branch, local attention-MIL branch, frequency branch, packaging-type router, gated
fusion) and adds a reliability layer (temperature scaling, conformal prediction, OOD rejection).
Every experiment follows one fixed evaluation protocol; see `docs/PLAN.md`.

Status: Phase 0 (environment and repository).

## Install

Requires Python 3.13.

```bash
python scripts/tasks.py setup          # or: make setup
```

This creates `.venv/`, installs CPU PyTorch, installs the package in editable mode with the
`[dev]` extra under the pinned versions in `constraints/cpu.txt`, and installs the pre-commit
hooks. For a CUDA machine, see `docs/RUNNING_ELSEWHERE.md`.

## Common tasks

| Command | Purpose |
|---|---|
| `python scripts/tasks.py lint` | ruff check + format check |
| `python scripts/tasks.py test` | fast tests (`-m slow` for slow ones) |
| `python -m pharmachecked.run experiment=noop` | compose the config and write a run folder |

`make <target>` is equivalent where `make` is available.

## Reproducing results

To be completed in Phase 10.
