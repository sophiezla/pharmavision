# PharmaChecked v2

Research codebase for detecting counterfeit medicine packaging from photographs. It benchmarks a
ladder of baselines (handcrafted features, CNNs, ViTs, foundation models), builds a multi-branch
model (global branch, local attention-MIL branch, frequency branch, packaging-type router, gated
fusion) and adds a reliability layer (temperature scaling, conformal prediction, OOD rejection).
It then studies generalization, interpretability, synthetic counterfeits, adversarial robustness
and on-device deployment. Every experiment follows one fixed evaluation protocol; see
`docs/PLAN.md`.

Status: Phases 0-8 and 10 complete (E13, SD inpainting and the synthetic-image review dropped,
ADR-029). The locked test set was evaluated once (`results/tables/final_test.md`,
`results/LOCK`). Open: phone-browser latency and the phone run of the app (PLAN 8.4, 8.6) and
the optional Phase 9 extensions.

- Results: `results/tables/` (CSV + Markdown), `results/figures/`, one folder per run in
  `results/runs/`.
- What was run and what it showed: `docs/EXPERIMENT_LOG.md` (newest first),
  `docs/experiments/E*.md`, `docs/phase_reports/`.
- Decisions: `docs/DECISIONS.md`. Data: `docs/DATA_CARD.md`. Models: `docs/MODEL_CARD.md`.
- Leaderboard: `results/tables/leaderboard.md`. Paper figures and tables: `results/paper/`
  (`README.md` maps each to its source; `numbers.json` holds every table cell with its source).

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
| `python scripts/tasks.py smoke` | tiny CPU run producing a complete run folder |
| `python scripts/tasks.py reproduce` | regenerate every table and figure from the run folders |
| `python -m pharmachecked.run experiment=... fold=0 seed=0` | one training run |

`make <target>` is equivalent where `make` is available.

## Reproducing results

### Tables and figures from the committed run folders

No images or weights are needed:

```bash
python scripts/tasks.py reproduce      # runs scripts/reproduce.py
git diff --stat results/tables          # empty when everything reproduces
```

or in Docker (CPU):

```bash
docker build -t pharmachecked .
docker run --rm -v "$PWD/results:/repo/results" pharmachecked
```

`scripts/reproduce.py` lists every command in the order the experiment log records them.
Figures that show dataset images stay local (ADR-006) and need `--with-local` plus the images.

### Training runs

GPU runs were executed as kits on a separate machine: `scripts/make_gpu_kit.py` builds a kit
from a clean commit, the kit's pipeline (`scripts/phase*_pipeline.py`,
`scripts/combined_pipeline.py`) trains and evaluates every arm, `scripts/pack_results.py`
bundles the run folders, and `scripts/unpack_results.py` verifies (SHA-256) and installs them.
Each run folder records its resolved config, git commit and environment (`config.yaml`,
`env.json`). See `docs/RUNNING_ELSEWHERE.md`.

### Data

Images are not redistributed. Place the original sources under `data/raw/` (see
`docs/DATA_CARD.md` for sources and licences), then rebuild the processed images and manifest
(`scripts/build_manifest.py`). Image ids are the first 16 hex digits of each original file's
SHA-256, and `data/processed/manifest_v2.csv` lists the full hash, so a different copy of a source
shows up as missing ids. `splits/v1.json` and `splits/v2.json` fix the folds and the locked test
set.

### Weights

The frozen models (ADR-030) are packed one zip per configuration with a SHA-256 `MANIFEST.json`
(`python scripts/release_assets.py --weights <folder outside the repo>`). Unzip into
`checkpoints/`; each run folder's `checkpoint_path.txt` points there.

### Locked test set (done once)

The locked test set was evaluated on 2026-11-12 (`results/LOCK`); `final_eval.py` now refuses to
run again. The commands were:

```bash
python scripts/final_eval.py --i-understand-this-is-final --check-only   # opens nothing
python scripts/final_eval.py --i-understand-this-is-final                 # writes results/LOCK
python scripts/model_card.py                                             # adds the results
python scripts/release_assets.py                                         # leaderboard, paper
```

## App

```bash
python scripts/app_params.py           # fits guard, temperature, conformal and quality checks
python app/app.py --host 0.0.0.0       # then open http://<laptop-ip>:7860 on a phone
```

Photo -> quality checks (`e09_quality_limits`) -> OOD guard -> final model -> calibrated
probability and conformal set -> `Likely authentic | Likely counterfeit | Uncertain — verify
manually` -> Grad-CAM++ heatmap. Research prototype only; see the model card for intended use.

Latency: `python scripts/e16_latency.py` (laptop CPU) and `app/phone_latency.html` (ONNX Runtime
Web on a phone browser).
