# Phase 10 checkpoint report

Date: 2026-11-12
Branch: `phase-6-8-combined` (local; not pushed). Locked-test run at f47bc9e (`results/LOCK`).

## 1. Steps completed

| Step | Commits | Notes |
|---|---|---|
| 10.1 freeze | 54bba97 (ADR-030); tag `v1.0-rc` at c70b089 | final model, E1 best on data v2, MobileNetV3-L student |
| 10.2 locked-test evaluation | run at f47bc9e; results 79294c5 | raw data restored from `Projects.zip`; all raw SHA-256 matched; dev images re-normalized byte-identical |
| 10.3 benchmark release | 54bba97 (Dockerfile, reproduce), ba66e1a (leaderboard, weight packing) | weight zips built outside the repo; GitHub release upload waits for Sophie (Kaggle licence not confirmed, DATA_CARD) |
| 10.4 model card | 54bba97, 4a2df8f | locked-test table and coverage note added |
| 10.5 paper assets | ba66e1a | Fig 1-8 / Tab 1-4 mapping chosen here (the plan doc's list is not on this Mac); `numbers.json` = every cell of Tab 1-4 with its source |
| 10.6 README | this commit | reproduction guide (data, weights, locked test); tag `v1.0` waits for Sophie |

Skipped: 8.4 phone-browser latency and the 8.6 phone run (Sophie, 2026-11-12: skip the phone test for now).

## 2. Results tables

### `results/tables/final_test.md`

Generated once by `scripts/final_eval.py` (PLAN 10.2) on the locked test set (95 images); frozen models in ADR-030. Ensemble = mean probability of the 15 fold models with 95% stratified bootstrap CIs (2000 resamples, seed 0); per-run = mean ± sd over the 15 fold models, each with its own val threshold, cal temperature and LAC Mondrian conformal quantiles (alpha 0.10).

| model | runs | images | counterfeit | AUROC (ensemble) | AUPRC (ensemble) | sens@95%spec (ensemble) | ECE (ensemble, uncalibrated) | AUROC (per run) | balanced acc (per run, val threshold) | macro-F1 (per run) | sensitivity (per run) | specificity (per run) | ECE after T (per run) | Brier after T (per run) | conformal coverage (per run, alpha 0.10) | coverage authentic | coverage counterfeit | mean set size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final model (EfficientNetV2-S) | 15 | 95 | 36 | 0.987 [0.968, 0.999] | 0.978 [0.948, 0.999] | 0.833 | 0.0529 | 0.987 ± 0.005 | 0.944 ± 0.031 | 0.940 ± 0.025 | 0.944 ± 0.069 | 0.944 ± 0.017 | 0.0521 ± 0.0108 | 0.0481 ± 0.0099 | 0.845 ± 0.049 | 0.852 ± 0.068 | 0.833 ± 0.055 | 0.867 ± 0.051 |
| E1 best (ConvNeXt-T, data v2) | 15 | 95 | 36 | 0.998 [0.992, 1.000] | 0.996 [0.986, 1.000] | 1.000 | 0.0225 | 0.998 ± 0.003 | 0.961 ± 0.026 | 0.962 ± 0.025 | 0.948 ± 0.059 | 0.974 ± 0.038 | 0.0223 ± 0.0122 | 0.0190 ± 0.0119 | 0.849 ± 0.047 | 0.837 ± 0.068 | 0.869 ± 0.070 | 0.852 ± 0.046 |
| E16 student (MobileNetV3-L, KD T=4) | 15 | 95 | 36 | 0.992 [0.977, 1.000] | 0.986 [0.965, 1.000] | 0.861 | 0.0320 | 0.990 ± 0.003 | 0.939 ± 0.015 | 0.938 ± 0.011 | 0.930 ± 0.044 | 0.949 ± 0.022 | 0.0509 ± 0.0095 | 0.0436 ± 0.0070 | 0.872 ± 0.057 | 0.860 ± 0.082 | 0.893 ± 0.055 | 0.898 ± 0.057 |


## Paired DeLong vs the final model (ensembles, Holm)

| model | reference | AUROC diff | delong_p | holm_p | better | worse |
|---|---|---|---|---|---|---|
| E1 best (ConvNeXt-T, data v2) | final model (EfficientNetV2-S) | 0.010 [0.000, 0.028] | 0.1403 | 0.2805 | False | False |
| E16 student (MobileNetV3-L, KD T=4) | final model (EfficientNetV2-S) | 0.004 [-0.005, 0.018] | 0.476 | 0.476 | False | False |

## 3. Figures produced

`results/paper/fig1-8.{png,pdf}` (copies; mapping in `results/paper/README.md`). No new figures.

## 4. Anomalies, failed runs, open problems

- Conformal coverage on the locked test set (per run, alpha 0.10) is below 0.90 for all three frozen models and below each model's out-of-fold development coverage (logged 2026-11-12 in `docs/EXPERIMENT_LOG.md`; model card "Known failure modes").
- `tests/test_test_lock.py::test_repository_is_not_locked` failed after the run as designed; replaced by a post-lock check.
- The full `scripts/reproduce.py` was not re-run after adding the `release_assets.py` step (the E9-ext step takes hours); the new step was run on its own and its tests pass.

## 5. Decisions needed from Sophie

1. Publish? Options: (a) merge `phase-6-8-combined` into `main`, push branches and tags, tag `v1.0`; (b) keep everything local for now.
2. Weights: (a) attach the three zips in `~/Hard Drive Copy/PharmaVision/release_v1.0/` to a GitHub release; (b) keep them private until the Kaggle licence is confirmed.
3. Paper figure/table list: keep the mapping in `results/paper/README.md`, or swap entries in `FIGURES` / `TABLES` of `scripts/release_assets.py`.

## 6. Proposed plan for the next phase

Phone test (8.4, 8.6) when convenient; Phase 9 only if approved. GPU-hour estimate: none.
