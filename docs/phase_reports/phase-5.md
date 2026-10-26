# Phase 5 checkpoint report

Date: 2026-10-25
Branch / last commit: `phase-5-reliability` (not pushed); this report is committed on top of `3006ecc`.

## 1. Steps completed

| Step | Commits | Notes |
|---|---|---|
| 5.1 E10.md, calibration | 5a6cdc8 (originally 8425d1e, fd1e9e4) | Temperature on `cal` (NLL, LBFGS); T = 1 when not identifiable. |
| 5.2 Uncertainty, selective | 5a6cdc8 (originally 8425d1e) | Softmax confidence vs 3-seed ensemble; MC dropout dropped (ADR-025). |
| 5.3 Conformal | 5a6cdc8 (originally 8425d1e) | LAC/APS × marginal/Mondrian, alpha 0.05/0.10; `test_conformal.py` passes. |
| 5.4 E10 tables + figures | 863278a, 101644f | `e10_calibration`, `e10_selective`, `e10_conformal`, `e10_conformal_choice`; `e10_reliability_diagrams`, `e10_risk_coverage`. ADR-028 (LAC Mondrian). |
| 5.5 OOD scores | 5a6cdc8 (originally 8425d1e) | MSP, energy, Mahalanobis, kNN, ensemble disagreement. |
| 5.6 E11 OOD table | 101644f | `e11_ood`. |
| 5.7 End-to-end | 101644f, 3006ecc | `e11_guard_choice` (MSP), `e11_end_to_end`. Table-note wording fixed in 3006ecc (numbers unchanged). |
| 4.11 E9 severities 6–8 | 863278a, 101644f | `e09_ext_severity`, `e09_ext_quality_limits`, `e09_ext_severity_curves`. |
| Results install | 863278a | 75 `reliability_predictions.csv` + 75 `corruption_predictions_ext.csv`, job log, two status files. |

Skipped / incomplete: none of the Phase 5 steps. **1.8 concept labels** (Phase 1) are still open; ADR-027 needs them before the combined kit is built.

GPU run: the Phase 5 kit (`PRIVATE_gpu_kit_phase5_46d0a9f.zip`, RTX 5060, `results/job_logs/phase5_20261025-082954_kit_setup.json`), 75 + 75 jobs, 0 failures, source snapshot unmodified. The bundle has no stage logs with timings, so the GPU time is not recorded.

## 2. Results tables

### `results/tables/e10_calibration.md`

| config | runs | images | T fitted (identifiable) | temperature (fitted runs) | ece before | ece after | ece diff (after - before) | brier before | brier after | brier diff (after - before) | log_loss before | log_loss after | log_loss diff (after - before) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext-tiny/ft | 15 | 549 | 3 | 1.87 ± 0.94 | 0.0069 | 0.0069 | -0.0000 [-0.0007, 0.0006] | 0.0061 | 0.0061 | -0.0000 [-0.0001, 0.0001] | 0.0245 | 0.0246 | 0.0001 [-0.0003, 0.0005] |
| dinov2-vits14/ft | 15 | 549 | 8 | 2.00 ± 0.93 | 0.0137 | 0.017 | 0.0034 [-0.0042, 0.0095] | 0.0102 | 0.0102 | -0.0001 [-0.0007, 0.0004] | 0.0485 | 0.0525 | 0.0040 [0.0003, 0.0063] |
| efficientnetv2-s/default | 15 | 549 | 7 | 1.29 ± 0.55 | 0.0103 | 0.0099 | -0.0004 [-0.0030, 0.0030] | 0.0075 | 0.0074 | -0.0000 [-0.0003, 0.0001] | 0.0296 | 0.028 | -0.0016 [-0.0074, 0.0015] |
| handcrafted-xgb/default | 15 | 549 | 15 | 1.06 ± 0.18 | 0.0319 | 0.0331 | 0.0012 [-0.0117, 0.0153] | 0.0606 | 0.061 | 0.0004 [-0.0008, 0.0015] | 0.1963 | 0.1996 | 0.0033 [-0.0038, 0.0100] |
| swin-tiny/ft | 15 | 549 | 3 | 1.04 ± 0.44 | 0.0039 | 0.004 | 0.0001 [-0.0002, 0.0005] | 0.0031 | 0.0031 | 0.0000 [-0.0000, 0.0001] | 0.0175 | 0.0176 | 0.0000 [-0.0002, 0.0003] |

### `results/tables/e10_selective.md`

| config | AURC single (15 fold models) | AURC ensemble (5 folds) | error at full coverage, single | error at full coverage, ensemble |
|---|---|---|---|---|
| convnext-tiny/ft | 0.0003 ± 0.0005 | 0.0002 ± 0.0002 | 0.0085 ± 0.0064 | 0.0055 ± 0.0050 |
| dinov2-vits14/ft | 0.0018 ± 0.0023 | 0.0005 ± 0.0007 | 0.0195 ± 0.0158 | 0.0109 ± 0.0119 |
| efficientnetv2-s/default | 0.0006 ± 0.0011 | 0.0003 ± 0.0005 | 0.0116 ± 0.0136 | 0.0092 ± 0.0130 |
| handcrafted-xgb/default | 0.0140 ± 0.0047 | 0.0136 ± 0.0048 | 0.0855 ± 0.0229 | 0.0819 ± 0.0219 |
| swin-tiny/ft | 0.0002 ± 0.0004 | 0.0002 ± 0.0005 | 0.0043 ± 0.0059 | 0.0037 ± 0.0050 |

### `results/tables/e10_conformal_choice.md`

| score | mode | largest per-class shortfall (pp) | mean set size (alpha 0.10) | qualifies |
|---|---|---|---|---|
| lac | marginal | 1.63 | 0.9176 | True |
| lac | mondrian | 0.67 | 0.9169 | True |
| aps | marginal | 2.51 | 0.9176 | False |
| aps | mondrian | 2.36 | 0.9114 | False |

Final model rows of `results/tables/e10_conformal.md` (all five models in the file):

| config | score | mode | alpha | runs | coverage | coverage_authentic | coverage_counterfeit | mean_set_size | singleton_rate | abstention_rate | empty_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/default | aps | marginal | 0.05 | 15 | 0.964 ± 0.043 | 0.954 ± 0.057 | 0.982 ± 0.025 | 1.105 ± 0.365 | 0.838 ± 0.343 | 0.133 ± 0.352 | 0.028 ± 0.040 |
| efficientnetv2-s/default | aps | marginal | 0.1 | 15 | 0.906 ± 0.064 | 0.875 ± 0.077 | 0.959 ± 0.061 | 0.918 ± 0.063 | 0.918 ± 0.063 | 0.000 ± 0.000 | 0.082 ± 0.063 |
| efficientnetv2-s/default | aps | mondrian | 0.05 | 15 | 0.962 ± 0.033 | 0.974 ± 0.043 | 0.941 ± 0.048 | 1.197 ± 0.278 | 0.762 ± 0.240 | 0.217 ± 0.258 | 0.021 ± 0.029 |
| efficientnetv2-s/default | aps | mondrian | 0.1 | 15 | 0.900 ± 0.063 | 0.913 ± 0.080 | 0.876 ± 0.081 | 0.911 ± 0.061 | 0.911 ± 0.061 | 0.000 ± 0.000 | 0.089 ± 0.061 |
| efficientnetv2-s/default | lac | marginal | 0.05 | 15 | 0.956 ± 0.034 | 0.964 ± 0.035 | 0.943 ± 0.052 | 0.963 ± 0.038 | 0.963 ± 0.038 | 0.000 ± 0.000 | 0.037 ± 0.038 |
| efficientnetv2-s/default | lac | marginal | 0.1 | 15 | 0.914 ± 0.034 | 0.933 ± 0.035 | 0.884 ± 0.068 | 0.918 ± 0.036 | 0.918 ± 0.036 | 0.000 ± 0.000 | 0.082 ± 0.036 |
| efficientnetv2-s/default | lac | mondrian | 0.05 | 15 | 0.961 ± 0.031 | 0.970 ± 0.033 | 0.944 ± 0.044 | 0.965 ± 0.032 | 0.965 ± 0.032 | 0.000 ± 0.000 | 0.035 ± 0.032 |
| efficientnetv2-s/default | lac | mondrian | 0.1 | 15 | 0.916 ± 0.027 | 0.930 ± 0.038 | 0.893 ± 0.062 | 0.917 ± 0.028 | 0.917 ± 0.028 | 0.000 ± 0.000 | 0.083 ± 0.028 |

### `results/tables/e11_ood.md`

| config | score | units | near AUROC | near FPR@95TPR | near accepted at val threshold | far AUROC | far FPR@95TPR | far accepted at val threshold | ID held-out accepted at val threshold |
|---|---|---|---|---|---|---|---|---|---|
| convnext-tiny/ft | msp | 15 | 0.979 ± 0.023 | 0.106 ± 0.206 | 0.093 ± 0.135 | 0.983 ± 0.017 | 0.070 ± 0.128 | 0.111 ± 0.211 | 0.946 ± 0.037 |
| convnext-tiny/ft | energy | 15 | 0.979 ± 0.023 | 0.106 ± 0.206 | 0.093 ± 0.135 | 0.983 ± 0.017 | 0.070 ± 0.128 | 0.111 ± 0.211 | 0.946 ± 0.037 |
| convnext-tiny/ft | maha | 15 | 0.993 ± 0.006 | 0.028 ± 0.036 | 0.026 ± 0.025 | 0.999 ± 0.001 | 0.003 ± 0.006 | 0.002 ± 0.005 | 0.947 ± 0.028 |
| convnext-tiny/ft | knn | 15 | 0.975 ± 0.030 | 0.141 ± 0.200 | 0.171 ± 0.225 | 0.997 ± 0.004 | 0.011 ± 0.026 | 0.013 ± 0.025 | 0.927 ± 0.038 |
| convnext-tiny/ft | disagreement | 5 | 0.976 ± 0.021 | 0.118 ± 0.153 | 0.155 ± 0.296 | 0.969 ± 0.026 | 0.166 ± 0.221 | 0.209 ± 0.389 | 0.942 ± 0.033 |
| dinov2-vits14/ft | msp | 15 | 0.923 ± 0.036 | 0.474 ± 0.211 | 0.428 ± 0.271 | 0.938 ± 0.025 | 0.351 ± 0.169 | 0.314 ± 0.204 | 0.935 ± 0.042 |
| dinov2-vits14/ft | energy | 15 | 0.923 ± 0.036 | 0.474 ± 0.211 | 0.429 ± 0.272 | 0.938 ± 0.025 | 0.352 ± 0.169 | 0.314 ± 0.204 | 0.935 ± 0.042 |
| dinov2-vits14/ft | maha | 15 | 0.966 ± 0.024 | 0.174 ± 0.121 | 0.244 ± 0.206 | 0.993 ± 0.007 | 0.033 ± 0.041 | 0.032 ± 0.031 | 0.945 ± 0.055 |
| dinov2-vits14/ft | knn | 15 | 0.854 ± 0.081 | 0.623 ± 0.230 | 0.671 ± 0.199 | 0.965 ± 0.034 | 0.121 ± 0.082 | 0.163 ± 0.116 | 0.946 ± 0.044 |
| dinov2-vits14/ft | disagreement | 5 | 0.951 ± 0.016 | 0.313 ± 0.132 | 0.240 ± 0.286 | 0.937 ± 0.030 | 0.407 ± 0.266 | 0.295 ± 0.295 | 0.915 ± 0.065 |
| efficientnetv2-s/default | msp | 15 | 0.979 ± 0.010 | 0.102 ± 0.078 | 0.201 ± 0.228 | 0.973 ± 0.018 | 0.166 ± 0.174 | 0.259 ± 0.218 | 0.948 ± 0.051 |
| efficientnetv2-s/default | energy | 15 | 0.979 ± 0.010 | 0.102 ± 0.078 | 0.201 ± 0.228 | 0.973 ± 0.018 | 0.166 ± 0.174 | 0.259 ± 0.218 | 0.949 ± 0.052 |
| efficientnetv2-s/default | maha | 15 | 0.963 ± 0.021 | 0.282 ± 0.242 | 0.353 ± 0.268 | 0.982 ± 0.004 | 0.103 ± 0.040 | 0.167 ± 0.085 | 0.962 ± 0.017 |
| efficientnetv2-s/default | knn | 15 | 0.926 ± 0.051 | 0.477 ± 0.275 | 0.553 ± 0.320 | 0.993 ± 0.011 | 0.046 ± 0.124 | 0.020 ± 0.020 | 0.944 ± 0.045 |
| efficientnetv2-s/default | disagreement | 5 | 0.965 ± 0.017 | 0.299 ± 0.320 | 0.270 ± 0.255 | 0.957 ± 0.021 | 0.388 ± 0.358 | 0.353 ± 0.304 | 0.936 ± 0.063 |
| handcrafted-xgb/default | msp | 15 | 0.787 ± 0.058 | 0.859 ± 0.105 | 0.818 ± 0.167 | 0.770 ± 0.034 | 0.851 ± 0.085 | 0.809 ± 0.133 | 0.924 ± 0.061 |
| handcrafted-xgb/default | energy | 15 | 0.787 ± 0.058 | 0.859 ± 0.105 | 0.818 ± 0.167 | 0.770 ± 0.034 | 0.851 ± 0.085 | 0.809 ± 0.133 | 0.924 ± 0.061 |
| handcrafted-xgb/default | disagreement | 5 | 0.782 ± 0.065 | 0.845 ± 0.080 | 0.810 ± 0.137 | 0.756 ± 0.038 | 0.881 ± 0.049 | 0.826 ± 0.085 | 0.909 ± 0.053 |
| swin-tiny/ft | msp | 15 | 0.986 ± 0.008 | 0.036 ± 0.057 | 0.077 ± 0.098 | 0.986 ± 0.007 | 0.048 ± 0.035 | 0.099 ± 0.094 | 0.959 ± 0.027 |
| swin-tiny/ft | energy | 15 | 0.986 ± 0.008 | 0.036 ± 0.057 | 0.078 ± 0.099 | 0.986 ± 0.007 | 0.048 ± 0.035 | 0.099 ± 0.094 | 0.960 ± 0.027 |
| swin-tiny/ft | maha | 15 | 0.985 ± 0.012 | 0.081 ± 0.120 | 0.133 ± 0.225 | 0.985 ± 0.008 | 0.077 ± 0.060 | 0.129 ± 0.109 | 0.960 ± 0.023 |
| swin-tiny/ft | knn | 15 | 0.964 ± 0.023 | 0.243 ± 0.217 | 0.263 ± 0.260 | 0.997 ± 0.002 | 0.014 ± 0.010 | 0.021 ± 0.028 | 0.945 ± 0.035 |
| swin-tiny/ft | disagreement | 5 | 0.982 ± 0.009 | 0.061 ± 0.045 | 0.048 ± 0.038 | 0.974 ± 0.009 | 0.103 ± 0.041 | 0.097 ± 0.068 | 0.945 ± 0.017 |

### `results/tables/e11_guard_choice.md`

| score | select-half OOD AUROC |
|---|---|
| msp | 0.9761 |
| energy | 0.9761 |
| maha | 0.973 |
| knn | 0.9593 |

### `results/tables/e11_end_to_end.md`

| system | runs | ID rejected | OOD rejected | abstained | issued | error on issued | OOD among issued |
|---|---|---|---|---|---|---|---|
| bare classifier | 15 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.829 ± 0.003 | 0.826 ± 0.001 |
| full system (alpha 0.1) | 15 | 0.052 ± 0.051 | 0.766 ± 0.221 | 0.159 ± 0.184 | 0.199 ± 0.045 | 0.184 ± 0.139 | 0.183 ± 0.139 |
| full system (alpha 0.05) | 15 | 0.052 ± 0.051 | 0.766 ± 0.221 | 0.058 ± 0.084 | 0.300 ± 0.180 | 0.337 ± 0.236 | 0.335 ± 0.237 |

### `results/tables/e09_ext_quality_limits.md`

| corruption | max_severity | threshold |
|---|---|---|
| brightness | 8 | 0.85 |
| contrast | 8 | 0.85 |
| gaussian_blur | 8 | 0.85 |
| gaussian_noise | 8 | 0.85 |
| glare | 8 | 0.85 |
| jpeg | 8 | 0.85 |
| motion_blur | 8 | 0.85 |
| perspective | 8 | 0.85 |
| rotation | 8 | 0.85 |

Per-condition AUROC for severities 1–8: `results/tables/e09_ext_severity.md`.

## 3. Figures produced

- `results/figures/e10_reliability_diagrams.{png,pdf}`
- `results/figures/e10_risk_coverage.{png,pdf}`
- `results/figures/e09_ext_severity_curves.{png,pdf}`

## 4. Anomalies, failed runs, open problems

1. **No failed runs.** 75 reliability and 75 extension jobs, all `done` (`results/pipeline/p5_reliability_status.json`, `e09_extension_status.json`).
2. **Git history recovered; three commits squashed.** The repository copy moved to the laptop had a truncated `.git` (object folders `00`–`28` only, no packs), and `phase-5-reliability` / `phase-6-8-combined` had never been pushed. The trees of 46d0a9f (from the kit's `SNAPSHOT.json`) and 96bae50 (from the saved index) were rebuilt and verified by reproducing both commit hashes exactly, but 8425d1e, fd1e9e4 and b11f6dd could not be rebuilt. The branches were recreated as 5a6cdc8 (tree of 46d0a9f) and 4464f59 (tree of 96bae50); each commit message lists the original commits. Run `env.json` files and older PLAN ticks still name the original hashes.
3. **Conformal sets are empty, not size 2.** For the deep models nearly every LAC set is a singleton or empty (final model, LAC Mondrian, alpha 0.10: empty-set rate 0.083 ± 0.028, size-2 rate 0.000 ± 0.000). Coverage is met by dropping labels on images the classifier mostly gets right, so "abstain" in the system means "empty set". At alpha 0.05 the end-to-end system issues more and errs more (0.300 issued, 0.337 error) than at alpha 0.10 (0.199, 0.184).
4. **The guard is the part that matters.** The final model's MSP guard rejects 0.766 of OOD `eval` images and 0.052 of ID images; nearly every issued error is an accepted OOD image (OOD among issued 0.183 vs error 0.184 at alpha 0.10). The stream is mostly OOD by design (the bare classifier's OOD-among-issued is 0.826), so stream-level rates depend on that mix.
5. **Temperature mostly not identifiable.** Only 3–8 of 15 fold models of each deep model had a `cal` image on the wrong side of 0.5; the rest keep T = 1 (pre-registered, E10.md). No calibration change has a CI excluding 0 except DINOv2's log-loss increase (0.0040 [0.0003, 0.0063]).
6. **Feature-space OOD hypothesis not supported for the final model.** MSP beats Mahalanobis and kNN for the final model on near OOD and in the `select`-half guard choice; Mahalanobis is best for ConvNeXt and DINOv2.
7. **E9 extension: still no failure point for the final model** (limit 8 for all nine corruptions; lowest 0.8743, perspective warp at severity 8). DINOv2 fails under brightness from severity 6 and ConvNeXt at severity 8. Severities 1–5 reproduce `e09_severity` exactly.
8. **Small re-prediction differences.** Re-predicted held-out probabilities differ from the original OOF by up to 1.08e-02 (efficientnetv2-s) and 4.14e-03 (convnext-tiny); ≤ 6e-08 for the others. Not investigated; all Phase 5 tables use the re-predicted values consistently.
9. **The Phase 5 kit ran on its own, not as part of the combined kit.** ADR-027 (Proposed, on `phase-6-8-combined`) says this kit is superseded and must not be run, and its `combined_pipeline.py` starts with Phase 5. The returned results are valid (same code, 46d0a9f), but ADR-027 and the combined pipeline do not yet reflect that Phase 5 is done.
10. **PLAN wording:** 5.7 names `evaluation/pipeline.py`; the chain is `end_to_end_table` in `evaluation/reliability.py`. 4.11 asked to extend the E9 tables; the extension writes separate `e09_ext_*` tables and keeps the pre-registered ones unchanged. Both noted at the ticks.

## 5. Decisions needed from Sophie

1. **Conformal method (ADR-028).** Accept LAC Mondrian, chosen by the pre-registered rule? Options: (a) accept as specified, reporting the empty-set behaviour (recommended); (b) add a smaller alpha (e.g. 0.01) or count empty sets separately from abstentions in the system: a protocol change that needs a new ADR.
2. **ADR-027 after the standalone Phase 5 run.** Options: (a) amend ADR-027 and remove the Phase 5 stage from `combined_pipeline.py`, so the combined kit carries only Phases 6–8 (recommended); (b) leave the combined kit as written and run Phase 5 again.
3. **ADR-027's open points** (still "to confirm"): Stable Diffusion inpainting weights downloaded on the GPU PC (about 2.1 GB); a run time over the 4-hour rule (about 20 h estimated in ADR-027); the cross-entropy-only student arm added to PLAN 8.1.
4. **Git recovery.** Accept the recreated commits 5a6cdc8 and 4464f59 in place of the lost originals, and push both branches to GitHub?
5. **CHECKPOINT 5.** Approve Phase 5, merge `phase-5-reliability` into `main` and tag `v0.3-reliable`?

## 6. Proposed plan for the next phase

Phases 6–8 run as one GPU kit (ADR-027, code on `phase-6-8-combined`). Before building it: rebase or merge `phase-6-8-combined` onto the Phase 5 results, apply the answers to questions 2–3, and complete PLAN 1.8 (concept labels) on the laptop. Then build the kit with `--models`, run it on the GPU PC, and review the results one phase at a time (CHECKPOINTS 6, 7, 8).

GPU-hour estimate: about 20 h on the RTX 5060 per ADR-027 (including its Phase 5 stage, which would be dropped under question 2a); above the 4-hour rule, so it needs your approval.
