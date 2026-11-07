# Phases 6-8 checkpoint report (combined kit)

Date: 2026-11-07
Branch: `phase-6-8-combined` (not pushed). GPU run: `PRIVATE_gpu_kit_phases6-8_e7905b0.zip` on the RTX 5060 (`results/job_logs/phases6-8_20261107-041837_kit_setup.json`: commit e7905b0, torch 2.14.1+cu130, snapshot unmodified); results `RESULTS_phases6-8_20261107-041837.zip`, 186 run folders, every job `done` in the four status files.

## 1. Steps completed

| Step | Commits | Notes |
|---|---|---|
| 6.1-6.2 saliency methods, faithfulness | 4464f59, a62ecf9; results e69a6de | Grad-CAM, Grad-CAM++, IG, rollout, MIL, random; deletion/insertion; cascading randomization |
| 6.3 localization | e69a6de, e085e71 | procedural edits only (ADR-029); duplicate synthetic ids repaired at table time |
| 6.4 tables, figure | e085e71 | `e12_examples` local only (ADR-006) |
| 6.5-6.8 E13 | dropped | ADR-029 (no annotation) |
| 7.1-7.2 OCR, procedural synthesis | 4464f59, a62ecf9; e69a6de | 1,038 synthetic images (`e14_synthesis`) |
| 7.3-7.4 SD inpainting, quality review | dropped | ADR-029 |
| 7.5-7.6 E14 arms, tables, figure | e69a6de, e085e71 | ratios 1, 4, synthetic-only; best `procedural-r1`; E8 re-run |
| 7.7-7.12 E15 | e69a6de, e085e71 | no `both` arm, 3 race rounds (ADR-029) |
| 8.1-8.3 E16 distillation, export, re-fit | e69a6de, e085e71 | T = 4 only; `e16_reliability` |
| 8.4 latency | **open** | `scripts/e16_latency.py`, `app/phone_latency.html` ready; ONNX files are in the MODELS bundle, which was not on the drive |
| 8.5 Pareto figure, deployment table | partial | size axis done; latency column added by `e16_latency.py` |
| 8.6 app | code done | `app/app.py`, `scripts/app_params.py`, `tests/test_app.py`; the 20-image acceptance test with the real fold model skips until its checkpoint is installed; phone-browser run not yet done |

## 2. Results tables

### `results/tables/e12_faithfulness.md`

| model | method | runs | images | deletion AUC (lower better) | insertion AUC (higher better) | deletion vs random p (Holm) | insertion vs random p (Holm) |
|---|---|---|---|---|---|---|---|
| final | gradcam | 15 | 1647 | 0.949 ± 0.022 | 0.979 ± 0.011 | 1.2e-54 | 8.34e-160 |
| final | gradcampp | 15 | 1647 | 0.953 ± 0.023 | 0.978 ± 0.011 | 1.39e-64 | 1.59e-152 |
| final | ig | 15 | 1647 | 0.940 ± 0.019 | 0.976 ± 0.015 | 7.88e-26 | 7.7e-196 |
| final | random | 15 | 1647 | 0.904 ± 0.057 | 0.901 ± 0.057 | nan | nan |
| DINOv2-S | rollout | 15 | 1647 | 0.924 ± 0.048 | 0.937 ± 0.057 | 0.401 | 2.05e-10 |
| DINOv2-S | ig | 15 | 1647 | 0.834 ± 0.086 | 0.977 ± 0.018 | 6.4e-124 | 7.91e-135 |
| DINOv2-S | random | 15 | 1647 | 0.929 ± 0.056 | 0.928 ± 0.058 | nan | nan |
| E4 multi-branch | mil | 15 | 1647 | 0.948 ± 0.025 | 0.958 ± 0.020 | 2.65e-11 | 2.67e-33 |
| E4 multi-branch | ig | 15 | 1647 | 0.914 ± 0.025 | 0.951 ± 0.021 | 2.1e-22 | 5.21e-26 |
| E4 multi-branch | random | 15 | 1647 | 0.941 ± 0.024 | 0.941 ± 0.023 | nan | nan |

### `results/tables/e14_vs_reference.md`

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/procedural-only | efficientnetv2-s/default | 549 | -0.277 [-0.318, -0.234] | 0.075 [0.057, 0.106] | 0.580 [0.552, 0.603] | 1.569e-37 | 4.706e-37 | False | True |
| efficientnetv2-s/procedural-r1 | efficientnetv2-s/default | 549 | -0.001 [-0.003, -0.000] | 0.074 [0.058, 0.092] | 0.151 [0.127, 0.173] | 0.03356 | 0.03356 | False | True |
| efficientnetv2-s/procedural-r4 | efficientnetv2-s/default | 549 | -0.012 [-0.019, -0.006] | 0.328 [0.304, 0.350] | 0.667 [0.634, 0.698] | 0.001174 | 0.002349 | False | True |

### `results/tables/e14_open_set_gain.md`

| held-out type | images | runs (E14 / E8) | E8 final model AUROC | E14 procedural-r1 AUROC | AUROC diff (E14 - E8) | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| blister | 229 | 3 / 3 | 0.9541 | 0.9121 | -0.042 [-0.069, -0.017] | 0.001727 | 0.003453 | False | True |
| packaging | 175 | 3 / 3 | 0.9947 | 0.9907 | -0.004 [-0.012, 0.002] | 0.2161 | 0.2161 | False | False |

### `results/tables/e15_defences.md`

| defence | attack | eps (x255) | runs | evasion rate | queries to evade (mean) |
|---|---|---|---|---|---|
| none | blackbox |  | 15 | 0.046 ± 0.042 | 48.2 |
| none | patch |  | 15 | 0.836 ± 0.169 | nan |
| none | pgd | 4 | 15 | 0.987 ± 0.038 | nan |
| synth (E14 best) | blackbox |  | 15 | 0.040 ± 0.039 | 50.5 |
| synth (E14 best) | patch |  | 15 | 0.830 ± 0.177 | nan |
| synth (E14 best) | pgd | 4 | 15 | 1.000 ± 0.000 | nan |
| adv | blackbox |  | 15 | 0.036 ± 0.066 | 33.0 |
| adv | patch |  | 15 | 0.620 ± 0.337 | nan |
| adv | pgd | 4 | 15 | 0.643 ± 0.464 | nan |

### `results/tables/e15_arms_race.md`

| round | lineages | evasion rate | evasive images added this round | clean AUROC (runs) |
|---|---|---|---|---|
| 1 | 15 | 0.028 ± 0.039 | 2.9 ± 4.0 | 0.999 ± 0.001 |
| 2 | 15 | 0.034 ± 0.039 | 1.5 ± 1.7 | 0.999 ± 0.001 |
| 3 | 15 | 0.032 ± 0.047 | 1.9 ± 2.6 | 0.999 ± 0.001 |

### `results/tables/e16_vs_reference.md`

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | efficientnetv2-s/default | 549 | -0.002 [-0.005, 0.000] | 0.025 [0.018, 0.038] | 0.052 [0.037, 0.068] | 0.246 | 0.4759 | False | False |
| mobilenetv3-large/kd-t4 | efficientnetv2-s/default | 549 | -0.002 [-0.005, 0.000] | 0.001 [-0.005, 0.009] | 0.015 [-0.002, 0.035] | 0.238 | 0.4759 | False | False |

### `results/tables/e16_deployment.md`

| model | precision | runs | AUROC (pooled, held-out) | size MB | max |ONNX - torch| (FP32) | export checks passed |
|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | fp32 | 15 | 0.9973 | 13.45 ± 0.00 | 4.79e-04 | 13/15 |
| efficientnet-lite0/kd-t4 | int8 | 15 | 0.9977 | 3.81 ± 0.00 | 4.79e-04 | 13/15 |
| mobilenetv3-large/kd-t4 | fp32 | 15 | 0.9978 | 16.81 ± 0.00 | 1.63e-04 | 13/15 |
| mobilenetv3-large/kd-t4 | int8 | 15 | 0.9945 | 4.73 ± 0.00 | 1.63e-04 | 13/15 |
| final model | fp32 | 15 | 0.9993 | 80.65 ± 0.00 | 1.60e-05 | 15/15 |
| final model | int8 | 15 | 0.9994 | 22.93 ± 0.00 | 1.60e-05 | 15/15 |

### `results/tables/e16_reliability.md`

| model | precision | runs | T fitted (identifiable) | AUROC (per run) | ECE after T | error at val threshold | coverage (LAC Mondrian, alpha 0.10) | coverage authentic | coverage counterfeit | mean set size | abstention rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | fp32 | 15 | 12 | 0.997 ± 0.006 | 0.0359 ± 0.0346 | 0.0286 ± 0.0301 | 0.907 ± 0.037 | 0.890 ± 0.058 | 0.936 ± 0.059 | 0.913 ± 0.033 | 0.000 ± 0.000 |
| efficientnet-lite0/kd-t4 | int8 | 15 | 11 | 0.997 ± 0.007 | 0.0282 ± 0.0193 | 0.0322 ± 0.0279 | 0.919 ± 0.050 | 0.905 ± 0.073 | 0.943 ± 0.049 | 0.927 ± 0.048 | 0.000 ± 0.000 |
| mobilenetv3-large/kd-t4 | fp32 | 15 | 11 | 0.999 ± 0.001 | 0.0207 ± 0.0089 | 0.0188 ± 0.0116 | 0.923 ± 0.043 | 0.916 ± 0.071 | 0.934 ± 0.064 | 0.927 ± 0.045 | 0.000 ± 0.000 |
| mobilenetv3-large/kd-t4 | int8 | 15 | 15 | 0.995 ± 0.003 | 0.1014 ± 0.0659 | 0.0553 ± 0.0264 | 0.915 ± 0.073 | 0.915 ± 0.097 | 0.915 ± 0.068 | 0.937 ± 0.081 | 0.005 ± 0.017 |
| final model | fp32 | 15 | 7 | 0.999 ± 0.001 | 0.0159 ± 0.0122 | 0.0183 ± 0.0159 | 0.917 ± 0.028 | 0.931 ± 0.036 | 0.893 ± 0.062 | 0.918 ± 0.029 | 0.000 ± 0.000 |
| final model | int8 | 15 | 6 | 0.999 ± 0.001 | 0.0175 ± 0.0121 | 0.0116 ± 0.0117 | 0.921 ± 0.031 | 0.915 ± 0.058 | 0.931 ± 0.056 | 0.922 ± 0.033 | 0.000 ± 0.000 |

Full tables: `e12_randomization`, `e12_localization`, `e14_summary`, `e14_synthetic_only`, `e14_best`, `e14_synthesis`, `e15_summary`, `e15_vs_reference`, `e15race_vs_reference`, `e16_summary`.

## 3. Figures

`results/figures/e14_ratio_curve`, `e15_success_vs_eps`, `e15_arms_race`, `e16_pareto` (PNG + PDF); `results/figures/local/e12_examples` (local only).

## 4. Anomalies and open problems

- **`adv` defence is bimodal** (E15): 6 of 15 fold models are robust to FGSM/PGD, 8 are fully evaded even by FGSM, f2_s0 is in between. Not re-run (Rule 9); details in the log.
- **Synthetic manifest has 14 duplicated image ids** (content hashes): `localization.csv` rows held stringified Series; repaired in the table builder and fixed in the kit code. E14 training may have used these images twice.
- **ONNX FP32 check failed for 2 of 15 fold models per student** (max difference 4.79e-04 / 1.63e-04 vs the 1e-4 limit); their exports were still scored.
- **Synthetic data does not help** (E14): every ratio arm is significantly worse in pooled OOF AUROC than the final model, and the E8 re-run is worse on unseen blister.
- **The MODELS bundle is missing** from the drive, which blocks latency (8.4), the app's real-model test and phone run (8.6), and Phase 10.2.

## 5. Decisions needed from Sophie

1. Bring the `MODELS_*.zip` written next to the RESULTS zip by the kit on the GPU PC. Options: (a) copy it to the drive (recommended); (b) re-run `python scripts/pack_models.py --profile full` in the kit folder if it is gone.
2. Once it is installed, confirm the one-time locked-test evaluation (`scripts/final_eval.py --i-understand-this-is-final`), which cannot be undone. Options: (a) run it with the ADR-030 freeze; (b) change the freeze first.
3. Pushing `phase-6-8-combined` to GitHub and merging to `main` (tags `v0.4-deploy`, `v1.0-rc`): (a) yes; (b) keep local.

## 6. Next

Phase 10 (ADR-030; Phase 9 not run): install the MODELS bundle, run `scripts/e16_latency.py` and the phone page, run `pytest -m slow tests/test_app.py`, then the locked-test evaluation, `scripts/model_card.py`, and tag `v1.0`. No GPU time needed.
