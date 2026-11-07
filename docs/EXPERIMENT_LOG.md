# Experiment log

One entry per experiment batch, newest first. Results are pasted from `results/tables/`, never
retyped. Template:

```
## YYYY-MM-DD · E{N} · {short title}
- Commit: {hash}
- Command(s): {exact command lines}
- Hypothesis: {from docs/experiments/E{N}.md}
- Runs: {count}, {run_id glob}
- Result: {table pasted from results/tables/..., never retyped}
- Interpretation: {2–4 sentences, claims limited to what the table shows}
- Issues / anomalies: {crashes, divergences, surprises}
- Next: {what this changes}
```

---

## 2026-11-07 · Phase 10 · Locked-test evaluation attempt stopped: test images absent
- Commit: c70b089 (tag `v1.0-rc`); fix in the following commit.
- Command(s): `python scripts/final_eval.py --i-understand-this-is-final` (after Sophie's confirmation)
- Hypothesis: none (one-time evaluation of the ADR-030 freeze).
- Runs: none completed.
- Result: crashed with `FileNotFoundError` on the first locked-test image (`data/processed/images/00e72ba432c2b4e1.png`) of the first fold model, after its `val` and `cal` images were scored. None of the 95 locked-test images is on this Mac: `data/processed/images` was restored from the Phase 5 kit, which carries development images only, and `data/raw` is not here. No test image was read, no test prediction exists, no table was written and `results/LOCK` was not created, so the one-time evaluation is still unused.
- Interpretation: none (no result).
- Issues / anomalies: the `--check-only` preflight checked checkpoints but not image files. It now also checks that every locked-test image file exists (existence only, nothing is read) and refuses otherwise.
- Next: normalize the raw Kaggle images (`scripts/build_manifest.py`) once `data/raw` is back on this machine, confirm the regenerated development images are byte-identical to the existing ones, then run the evaluation.

## 2026-11-07 · E16 / app · Laptop latency and app acceptance test (MODELS bundle)
- Commit: MODELS bundle `MODELS_phases6-8_20261107-041849.zip` installed with `scripts/unpack_models.py` (67 files, SHA-256 verified; checkpoints git-ignored); results in the following commit.
- Command(s): `python scripts/e16_latency.py`; `python scripts/phase6to8_tables.py --exp e16`; `python scripts/make_figures.py --exp e16`; `python -m pytest -m slow tests/test_app.py`; `python scripts/final_eval.py --i-understand-this-is-final --check-only` (45 checkpoints found, test set not opened)
- Hypothesis: INT8 shrinks the model about 4x (E16.md); the app reproduces the offline pipeline (PLAN 8.6).
- Runs: no training; fold-0 / seed-0 ONNX files of the two students and the final model; machine in `results/pipeline/e16_latency.json` (macOS 27.0.1 arm64, onnxruntime 1.31.0).
- Result (from `results/tables/e16_deployment.md`):

| model | precision | runs | AUROC (pooled, held-out) | size MB | max |ONNX - torch| (FP32) | export checks passed | latency ms (laptop CPU) |
|---|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | fp32 | 15 | 0.9973 | 13.45 ± 0.00 | 4.79e-04 | 13/15 | 3.72 |
| efficientnet-lite0/kd-t4 | int8 | 15 | 0.9977 | 3.81 ± 0.00 | 4.79e-04 | 13/15 | 1.23 |
| mobilenetv3-large/kd-t4 | fp32 | 15 | 0.9978 | 16.81 ± 0.00 | 1.63e-04 | 13/15 | 2.78 |
| mobilenetv3-large/kd-t4 | int8 | 15 | 0.9945 | 4.73 ± 0.00 | 1.63e-04 | 13/15 | 1.65 |
| final model | fp32 | 15 | 0.9993 | 80.65 ± 0.00 | 1.60e-05 | 15/15 | 11.45 |
| final model | int8 | 15 | 0.9994 | 22.93 ± 0.00 | 1.60e-05 | 15/15 | 9.55 |

- Interpretation: on this laptop CPU the students run in 1.23-3.72 ms per image and the final model in 9.55-11.45 ms; INT8 is 3.0x faster than FP32 for EfficientNet-Lite0, 1.7x for MobileNetV3-L and 1.2x for the final model. The app gives the same label as the offline pipeline on all 110 held-out images of its fold model (the acceptance test checks the first 20).
- Issues / anomalies: the offline logits in `reliability_predictions.csv` were computed under fp16 autocast on the GPU (every value is fp16-exact); the app runs fp32 on the CPU, so logits differ by up to 0.295 (median 0.045). The acceptance test first used a 1e-3 logit tolerance and failed; it now requires identical labels and |logit difference| <= 0.5. Phone-browser latency (`app/phone_latency.html`) and the app on a phone are not done yet: they need Sophie's phone.
- Next: phone run; locked-test evaluation after Sophie's confirmation.

## 2026-11-07 · E16 · Distillation and on-device export (Phases 6-8 kit)
- Commit: Phases 6-8 GPU kit `PRIVATE_gpu_kit_phases6-8_e7905b0.zip` (RTX 5060, source e7905b0, snapshot unmodified), results `RESULTS_phases6-8_20261107-041837.zip` installed in e69a6de; tables and figures e085e71.
- Command(s): pipeline stage `c_e16`; `python scripts/aggregate.py --exp e16 --include e05:efficientnetv2-s/default --reference efficientnetv2-s/default`; `python scripts/phase6to8_tables.py --exp e16`; `python scripts/make_figures.py --exp e16`
- Hypothesis: distilled students lose little pooled OOF AUROC; INT8 changes AUROC by < 0.01 and shrinks the model about 4x (E16.md).
- Runs: 30, `e16_{mobilenetv3-large,efficientnet-lite0}_kd-t4_*`; deployment pass on these and the 15 final-model fold models (`deploy.json`, `deploy_predictions.csv`); 0 failures (`results/pipeline/e16_deploy_status.json`).
- Result (from `results/tables/e16_vs_reference.md`):

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | efficientnetv2-s/default | 549 | -0.002 [-0.005, 0.000] | 0.025 [0.018, 0.038] | 0.052 [0.037, 0.068] | 0.246 | 0.4759 | False | False |
| mobilenetv3-large/kd-t4 | efficientnetv2-s/default | 549 | -0.002 [-0.005, 0.000] | 0.001 [-0.005, 0.009] | 0.015 [-0.002, 0.035] | 0.238 | 0.4759 | False | False |

  Exported models (from `results/tables/e16_deployment.md`):

| model | precision | runs | AUROC (pooled, held-out) | size MB | max |ONNX - torch| (FP32) | export checks passed |
|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | fp32 | 15 | 0.9973 | 13.45 ± 0.00 | 4.79e-04 | 13/15 |
| efficientnet-lite0/kd-t4 | int8 | 15 | 0.9977 | 3.81 ± 0.00 | 4.79e-04 | 13/15 |
| mobilenetv3-large/kd-t4 | fp32 | 15 | 0.9978 | 16.81 ± 0.00 | 1.63e-04 | 13/15 |
| mobilenetv3-large/kd-t4 | int8 | 15 | 0.9945 | 4.73 ± 0.00 | 1.63e-04 | 13/15 |
| final model | fp32 | 15 | 0.9993 | 80.65 ± 0.00 | 1.60e-05 | 15/15 |
| final model | int8 | 15 | 0.9994 | 22.93 ± 0.00 | 1.60e-05 | 15/15 |

  Re-fitted reliability, PLAN 8.3 (from `results/tables/e16_reliability.md`):

| model | precision | runs | T fitted (identifiable) | AUROC (per run) | ECE after T | error at val threshold | coverage (LAC Mondrian, alpha 0.10) | coverage authentic | coverage counterfeit | mean set size | abstention rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | fp32 | 15 | 12 | 0.997 ± 0.006 | 0.0359 ± 0.0346 | 0.0286 ± 0.0301 | 0.907 ± 0.037 | 0.890 ± 0.058 | 0.936 ± 0.059 | 0.913 ± 0.033 | 0.000 ± 0.000 |
| efficientnet-lite0/kd-t4 | int8 | 15 | 11 | 0.997 ± 0.007 | 0.0282 ± 0.0193 | 0.0322 ± 0.0279 | 0.919 ± 0.050 | 0.905 ± 0.073 | 0.943 ± 0.049 | 0.927 ± 0.048 | 0.000 ± 0.000 |
| mobilenetv3-large/kd-t4 | fp32 | 15 | 11 | 0.999 ± 0.001 | 0.0207 ± 0.0089 | 0.0188 ± 0.0116 | 0.923 ± 0.043 | 0.916 ± 0.071 | 0.934 ± 0.064 | 0.927 ± 0.045 | 0.000 ± 0.000 |
| mobilenetv3-large/kd-t4 | int8 | 15 | 15 | 0.995 ± 0.003 | 0.1014 ± 0.0659 | 0.0553 ± 0.0264 | 0.915 ± 0.073 | 0.915 ± 0.097 | 0.915 ± 0.068 | 0.937 ± 0.081 | 0.005 ± 0.017 |
| final model | fp32 | 15 | 7 | 0.999 ± 0.001 | 0.0159 ± 0.0122 | 0.0183 ± 0.0159 | 0.917 ± 0.028 | 0.931 ± 0.036 | 0.893 ± 0.062 | 0.918 ± 0.029 | 0.000 ± 0.000 |
| final model | int8 | 15 | 6 | 0.999 ± 0.001 | 0.0175 ± 0.0121 | 0.0116 ± 0.0117 | 0.921 ± 0.031 | 0.915 ± 0.058 | 0.931 ± 0.056 | 0.922 ± 0.033 | 0.000 ± 0.000 |

- Interpretation: neither student is significantly different from the final model in pooled OOF AUROC (Holm p 0.4759 for both); EfficientNet-Lite0 has a higher ECE and log-loss than the final model (CIs above 0). INT8 shrinks every model about 3.5x; the pooled held-out AUROC of the INT8 MobileNetV3 is 0.0033 below its FP32 version, the other two change by less than that. After re-fitting, overall conformal coverage at alpha 0.10 is 0.907-0.923 for every model and precision (lowest per-class mean 0.890, Lite0 FP32 authentic); the INT8 MobileNetV3 has the highest re-fitted ECE (0.1014 ± 0.0659).
- Issues / anomalies: the FP32 ONNX check (max |ONNX - PyTorch| <= 1e-4 on val) failed in 2 of 15 fold models per student (largest difference 4.79e-04 Lite0, 1.63e-04 MobileNetV3); the final model passed 15/15. The failed exports were still quantized and scored; their predictions are in the tables. Latency (PLAN 8.4) is not measured yet: the ONNX files come in the MODELS bundle, which was not on the drive.
- Next: latency on the laptop and phone once the MODELS bundle is installed; the E16 student for Phase 10 is chosen then (ADR-030).

## 2026-11-07 · E15 · Adaptive adversary (Phases 6-8 kit)
- Commit: Phases 6-8 GPU kit `PRIVATE_gpu_kit_phases6-8_e7905b0.zip` (RTX 5060, source e7905b0, snapshot unmodified), results `RESULTS_phases6-8_20261107-041837.zip` installed in e69a6de; tables and figures e085e71.
- Command(s): pipeline stages `c_e15_defences`, `c_e15_race_round1..3`; `python scripts/aggregate.py --exp e15 --include e05:efficientnetv2-s/default --include e14:efficientnetv2-s/procedural-r1 --reference efficientnetv2-s/default`; `python scripts/aggregate.py --exp e15race --include e05:efficientnetv2-s/default --reference efficientnetv2-s/default`; `python scripts/phase6to8_tables.py --exp e15`; `python scripts/make_figures.py --exp e15`
- Hypothesis: targeted PGD at 4/255 evades the undefended final model on most held-out counterfeits; PGD training lowers PGD success at a small clean cost; arms-race evasion falls over the rounds (E15.md).
- Runs: 15 `e15_efficientnetv2-s_adv_*`, 45 `e15race_efficientnetv2-s_round{1,2,3}_*` (3 rounds, ADR-029); attack pass on the 45 fold models of `none`, `synth` (E14 `procedural-r1`) and `adv`; 0 failures (`results/pipeline/e15_attacks_status.json`).
- Result (from `results/tables/e15_defences.md`):

| defence | attack | eps (x255) | runs | evasion rate | queries to evade (mean) |
|---|---|---|---|---|---|
| none | blackbox |  | 15 | 0.046 ± 0.042 | 48.2 |
| none | color |  | 15 | 0.031 ± 0.025 | nan |
| none | fgsm | 1 | 15 | 0.252 ± 0.142 | nan |
| none | fgsm | 2 | 15 | 0.300 ± 0.169 | nan |
| none | fgsm | 4 | 15 | 0.327 ± 0.188 | nan |
| none | fgsm | 8 | 15 | 0.339 ± 0.182 | nan |
| none | patch |  | 15 | 0.836 ± 0.169 | nan |
| none | pgd | 1 | 15 | 0.872 ± 0.159 | nan |
| none | pgd | 2 | 15 | 0.972 ± 0.054 | nan |
| none | pgd | 4 | 15 | 0.987 ± 0.038 | nan |
| none | pgd | 8 | 15 | 0.997 ± 0.009 | nan |
| none | clean (detected) |  | 15 | 0.041 ± 0.043 | nan |
| synth (E14 best) | blackbox |  | 15 | 0.040 ± 0.039 | 50.5 |
| synth (E14 best) | color |  | 15 | 0.035 ± 0.041 | nan |
| synth (E14 best) | fgsm | 1 | 15 | 0.284 ± 0.111 | nan |
| synth (E14 best) | fgsm | 2 | 15 | 0.342 ± 0.131 | nan |
| synth (E14 best) | fgsm | 4 | 15 | 0.344 ± 0.145 | nan |
| synth (E14 best) | fgsm | 8 | 15 | 0.281 ± 0.148 | nan |
| synth (E14 best) | patch |  | 15 | 0.830 ± 0.177 | nan |
| synth (E14 best) | pgd | 1 | 15 | 0.956 ± 0.047 | nan |
| synth (E14 best) | pgd | 2 | 15 | 1.000 ± 0.000 | nan |
| synth (E14 best) | pgd | 4 | 15 | 1.000 ± 0.000 | nan |
| synth (E14 best) | pgd | 8 | 15 | 1.000 ± 0.000 | nan |
| synth (E14 best) | clean (detected) |  | 15 | 0.043 ± 0.033 | nan |
| adv | blackbox |  | 15 | 0.036 ± 0.066 | 33.0 |
| adv | color |  | 15 | 0.117 ± 0.066 | nan |
| adv | fgsm | 1 | 15 | 0.555 ± 0.499 | nan |
| adv | fgsm | 2 | 15 | 0.555 ± 0.499 | nan |
| adv | fgsm | 4 | 15 | 0.550 ± 0.502 | nan |
| adv | fgsm | 8 | 15 | 0.545 ± 0.505 | nan |
| adv | patch |  | 15 | 0.620 ± 0.337 | nan |
| adv | pgd | 1 | 15 | 0.641 ± 0.398 | nan |
| adv | pgd | 2 | 15 | 0.658 ± 0.433 | nan |
| adv | pgd | 4 | 15 | 0.643 ± 0.464 | nan |
| adv | pgd | 8 | 15 | 0.669 ± 0.453 | nan |
| adv | clean (detected) |  | 15 | 0.034 ± 0.026 | nan |

  Clean cost (from `results/tables/e15_vs_reference.md`):

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/adv | efficientnetv2-s/default | 549 | -0.007 [-0.012, -0.002] | 0.082 [0.071, 0.100] | 0.138 [0.108, 0.166] | 0.01201 | 0.02402 | False | True |
| efficientnetv2-s/procedural-r1 | efficientnetv2-s/default | 549 | -0.001 [-0.003, -0.000] | 0.074 [0.058, 0.092] | 0.151 [0.127, 0.173] | 0.03356 | 0.03356 | False | True |

  Arms race (from `results/tables/e15_arms_race.md`; clean AUROC vs the final model in `results/tables/e15race_vs_reference.md`, no round differs):

| round | lineages | evasion rate | evasive images added this round | clean AUROC (runs) |
|---|---|---|---|---|
| 1 | 15 | 0.028 ± 0.039 | 2.9 ± 4.0 | 0.999 ± 0.001 |
| 2 | 15 | 0.034 ± 0.039 | 1.5 ± 1.7 | 0.999 ± 0.001 |
| 3 | 15 | 0.032 ± 0.047 | 1.9 ± 2.6 | 0.999 ± 0.001 |

- Interpretation: PGD at 4/255 evades the undefended final model on 0.987 ± 0.038 of the held-out counterfeits it detects when clean, and the 5% patch on 0.836 ± 0.169; colour shifts and the 200-query black-box search evade 0.031-0.046. PGD adversarial training lowers mean PGD evasion to 0.641-0.669, with very large spread across fold models, and costs clean AUROC (-0.007 [-0.012, -0.002], Holm p 0.024, `worse`); synthetic-edit training (`synth`) changes no evasion rate by more than 0.084 in either direction (PGD 1/255: 0.872 to 0.956). The black-box evasion rate does not fall over the 3 arms-race rounds (0.028, 0.034, 0.032), and clean AUROC is unchanged.
- Issues / anomalies: the `adv` defence is bimodal. Pooled over the four eps values (`adversarial_predictions.csv`), 6 of 15 fold models (f0_s0, f2_s1, f2_s2, f4_s0-2) have FGSM evasion 0 and PGD 0.03-0.44; f2_s0 has FGSM 0.29 but PGD 1.0; the other 8 have FGSM 1.0 and PGD 0.92-1.0, above the undefended model's FGSM rate (0.252 ± 0.142). In `train_log.csv` the 6 robust runs ran 16-30 epochs and end at a clean val loss of 0.106-0.471, the 8 others 29-30 epochs ending at 0.0015-0.199; early stopping on clean val loss does not select for robustness, but the logs do not show the cause. Not re-run (Rule 9); a robust-validation stopping rule would be a new arm.
- Next: the model card lists white-box gradient and patch attacks as a known failure mode; no defence is adopted for the final model.

## 2026-11-07 · E14 · Procedural counterfeit synthesis (Phases 6-8 kit)
- Commit: Phases 6-8 GPU kit `PRIVATE_gpu_kit_phases6-8_e7905b0.zip` (RTX 5060, source e7905b0, snapshot unmodified), results `RESULTS_phases6-8_20261107-041837.zip` installed in e69a6de; tables and figures e085e71.
- Command(s): pipeline stages `c_e14`, `c_e14os`; `python scripts/aggregate.py --exp e14 --include e05:efficientnetv2-s/default --reference efficientnetv2-s/default`; `python scripts/phase6to8_tables.py --exp e14`; `python scripts/make_figures.py --exp e14`
- Hypothesis: a moderate synthetic ratio does not lower pooled OOF AUROC; synthetic-only is far worse; the best ratio arm improves E8 open-set AUROC (E14.md).
- Runs: 45 `e14_efficientnetv2-s_procedural-{r1,r4,only}_*` (ADR-029), 6 `e14os_efficientnetv2-s_procedural-r1-no-{blister,packaging}_*`.
- Result (from `results/tables/e14_vs_reference.md`):

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/procedural-only | efficientnetv2-s/default | 549 | -0.277 [-0.318, -0.234] | 0.075 [0.057, 0.106] | 0.580 [0.552, 0.603] | 1.569e-37 | 4.706e-37 | False | True |
| efficientnetv2-s/procedural-r1 | efficientnetv2-s/default | 549 | -0.001 [-0.003, -0.000] | 0.074 [0.058, 0.092] | 0.151 [0.127, 0.173] | 0.03356 | 0.03356 | False | True |
| efficientnetv2-s/procedural-r4 | efficientnetv2-s/default | 549 | -0.012 [-0.019, -0.006] | 0.328 [0.304, 0.350] | 0.667 [0.634, 0.698] | 0.001174 | 0.002349 | False | True |

  Decision (from `results/tables/e14_best.md`, rule fixed in E14.md):

| auroc | log_loss | runs | variant |
|---|---|---|---|
| 0.9979 | 0.1804 | 15 | procedural-r1 |
| 0.9877 | 0.6965 | 15 | procedural-r4 |

  E8 re-run (from `results/tables/e14_open_set_gain.md`):

| held-out type | images | runs (E14 / E8) | E8 final model AUROC | E14 procedural-r1 AUROC | AUROC diff (E14 - E8) | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| blister | 229 | 3 / 3 | 0.9541 | 0.9121 | -0.042 [-0.069, -0.017] | 0.001727 | 0.003453 | False | True |
| packaging | 175 | 3 / 3 | 0.9947 | 0.9907 | -0.004 [-0.012, 0.002] | 0.2161 | 0.2161 | False | False |

  Synthetic images (from `results/tables/e14_synthesis.md`):

| generator | edit_type | images |
|---|---|---|
| procedural | char_substitution | 102 |
| procedural | dose_digit_change | 70 |
| procedural | font_swap | 72 |
| procedural | local_blur_or_rehalftone | 213 |
| procedural | local_hue_shift | 185 |
| procedural | local_jpeg | 207 |
| procedural | logo_warp | 189 |

- Interpretation: every synthetic arm is significantly worse than the final model in pooled OOF AUROC (`worse` = True for all three), though for `procedural-r1` the difference is -0.001 [-0.003, -0.000]; ECE and log-loss rise in every arm. Synthetic-only training reaches a pooled AUROC of 0.722 [0.681, 0.765] (`e14_synthetic_only.md`). The best ratio arm does not improve open-set AUROC: it is worse on unseen blister (-0.042 [-0.069, -0.017]) and not different on unseen packaging. The first hypothesis is not supported; the second and the negative open-set result are.
- Issues / anomalies: the synthetic manifest holds 14 image ids twice (ids are content hashes, so these are identical images); E12 scored them twice and wrote the metadata lookup as a stringified Series. The table builder now drops the duplicates and restores the values (`repair_localization`, test in `tests/test_phase6to8_tables.py`); the kit code drops duplicate ids for future runs. Training arms may have seen these 14 images twice; not re-run. The plausibility review (PLAN 7.4) was dropped (ADR-029), so synthetic quality is unchecked.
- Next: `procedural-r1` was the `synth` defence in E15; no synthetic data in the final model.

## 2026-11-07 · E12 · Faithfulness of saliency maps (Phases 6-8 kit)
- Commit: Phases 6-8 GPU kit `PRIVATE_gpu_kit_phases6-8_e7905b0.zip` (RTX 5060, source e7905b0, snapshot unmodified), results `RESULTS_phases6-8_20261107-041837.zip` installed in e69a6de; tables and figures e085e71.
- Command(s): pipeline stages E12 (`results/pipeline/e12_{final,vit,mil}_status.json`); `python scripts/phase6to8_tables.py --exp e12`; `python scripts/make_figures.py --exp e12` (local-only figure, ADR-006)
- Hypothesis: every method beats the random map on deletion and insertion; Grad-CAM-type maps pass the randomization check, IG may not; localization is weak for small text edits (E12.md).
- Runs: no training; 45 fold models (final model, DINOv2-S `p4ref`, E4 multi-branch) wrote `faithfulness.csv` and `localization.csv`; the three fold-0 / seed-0 runs wrote `randomization.csv` and `saliency_examples.npz`; 0 failures.
- Result (from `results/tables/e12_faithfulness.md`):

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

  Localization of the final model (from `results/tables/e12_localization.md`, rows for `final`):

| model | method | generator | edit_type | images | pointing game | IoU top 15% | mean mask fraction |
|---|---|---|---|---|---|---|---|
| final | gradcam | procedural | char_substitution | 19 | 0.007 ± 0.018 | 0.006 ± 0.004 | 0.009 |
| final | gradcam | procedural | dose_digit_change | 14 | 0.000 ± 0.000 | 0.008 ± 0.005 | 0.014 |
| final | gradcam | procedural | font_swap | 14 | 0.105 ± 0.074 | 0.063 ± 0.020 | 0.038 |
| final | gradcam | procedural | local_blur_or_rehalftone | 41 | 0.198 ± 0.056 | 0.092 ± 0.015 | 0.174 |
| final | gradcam | procedural | local_hue_shift | 36 | 0.241 ± 0.089 | 0.116 ± 0.026 | 0.167 |
| final | gradcam | procedural | local_jpeg | 41 | 0.244 ± 0.062 | 0.123 ± 0.022 | 0.174 |
| final | gradcam | procedural | logo_warp | 37 | 0.160 ± 0.039 | 0.092 ± 0.025 | 0.149 |
| final | gradcampp | procedural | char_substitution | 19 | 0.010 ± 0.020 | 0.008 ± 0.005 | 0.009 |
| final | gradcampp | procedural | dose_digit_change | 14 | 0.005 ± 0.020 | 0.008 ± 0.005 | 0.014 |
| final | gradcampp | procedural | font_swap | 14 | 0.105 ± 0.075 | 0.065 ± 0.022 | 0.038 |
| final | gradcampp | procedural | local_blur_or_rehalftone | 41 | 0.227 ± 0.073 | 0.100 ± 0.016 | 0.174 |
| final | gradcampp | procedural | local_hue_shift | 36 | 0.263 ± 0.084 | 0.123 ± 0.029 | 0.167 |
| final | gradcampp | procedural | local_jpeg | 41 | 0.270 ± 0.068 | 0.134 ± 0.021 | 0.174 |
| final | gradcampp | procedural | logo_warp | 37 | 0.210 ± 0.060 | 0.113 ± 0.029 | 0.149 |
| final | ig | procedural | char_substitution | 19 | 0.003 ± 0.013 | 0.014 ± 0.005 | 0.009 |
| final | ig | procedural | dose_digit_change | 14 | 0.032 ± 0.041 | 0.012 ± 0.003 | 0.014 |
| final | ig | procedural | font_swap | 14 | 0.030 ± 0.053 | 0.046 ± 0.008 | 0.038 |
| final | ig | procedural | local_blur_or_rehalftone | 41 | 0.163 ± 0.067 | 0.098 ± 0.017 | 0.174 |
| final | ig | procedural | local_hue_shift | 36 | 0.285 ± 0.080 | 0.126 ± 0.017 | 0.167 |
| final | ig | procedural | local_jpeg | 41 | 0.312 ± 0.071 | 0.131 ± 0.015 | 0.174 |
| final | ig | procedural | logo_warp | 37 | 0.239 ± 0.076 | 0.130 ± 0.019 | 0.149 |
| final | random | procedural | char_substitution | 19 | 0.000 ± 0.000 | 0.008 ± 0.003 | 0.009 |
| final | random | procedural | dose_digit_change | 14 | 0.033 ± 0.069 | 0.009 ± 0.005 | 0.014 |
| final | random | procedural | font_swap | 14 | 0.032 ± 0.045 | 0.027 ± 0.005 | 0.038 |
| final | random | procedural | local_blur_or_rehalftone | 41 | 0.119 ± 0.034 | 0.063 ± 0.008 | 0.174 |
| final | random | procedural | local_hue_shift | 36 | 0.125 ± 0.044 | 0.064 ± 0.008 | 0.167 |
| final | random | procedural | local_jpeg | 41 | 0.129 ± 0.029 | 0.063 ± 0.009 | 0.174 |
| final | random | procedural | logo_warp | 37 | 0.137 ± 0.068 | 0.063 ± 0.012 | 0.149 |

- Interpretation: on insertion every method beats the random map (Holm p < 1e-9). On deletion only DINOv2-S IG (0.834 ± 0.086 vs random 0.929 ± 0.056) and E4 IG (0.914 vs 0.941) are clearly below their random map (DINOv2-S rollout 0.924, Holm p 0.401); all three final-model methods have a higher deletion AUC than random, so the first hypothesis holds for insertion only. In the randomization check (`e12_randomization.md`) Grad-CAM and Grad-CAM++ correlations fall to around 0 by the middle blocks, while IG stays at 0.776-0.865 through all layers of the final model, so IG fails the sanity check. The final model's maps hit the edit mask (pointing game) in at most 0.312 of the images for region edits (IG, `local_jpeg`; random map 0.119-0.137) and at most 0.105 for text edits (Grad-CAM / Grad-CAM++, `font_swap`).
- Issues / anomalies: the duplicate synthetic ids (E14 entry) affected `localization.csv`; repaired at table-build time.
- Next: the app shows the Grad-CAM++ map (passes the randomization check, unlike IG, and is at or above Grad-CAM in every localization row); maps are labelled as indicative only.

## 2026-10-25 · E11 · OOD guard and end-to-end system (Phase 5 kit)
- Commit: Phase 5 GPU kit `PRIVATE_gpu_kit_phase5_46d0a9f.zip` (RTX 5060; source 46d0a9f, recreated as 5a6cdc8), results `RESULTS_phase5_20261025-082954.zip` installed in 863278a; tables 101644f, note fix 3006ecc.
- Command(s): pipeline `p5_reliability`; `python scripts/phase5_tables.py --exp e11` (after `--exp e10`)
- Hypothesis: feature-space scores beat output scores, especially near OOD; near is harder than far; the full system has a much lower error on issued predictions (E11.md).
- Runs: no training; the 75 fold models (`p4ref_*`, `e05_efficientnetv2-s_default_*`) each wrote `reliability_predictions.csv`; 0 failures (`results/pipeline/p5_reliability_status.json`).
- Result (from `results/tables/e11_end_to_end.md`):

| system | runs | ID rejected | OOD rejected | abstained | issued | error on issued | OOD among issued |
|---|---|---|---|---|---|---|---|
| bare classifier | 15 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.829 ± 0.003 | 0.826 ± 0.001 |
| full system (alpha 0.1) | 15 | 0.052 ± 0.051 | 0.766 ± 0.221 | 0.159 ± 0.184 | 0.199 ± 0.045 | 0.184 ± 0.139 | 0.183 ± 0.139 |
| full system (alpha 0.05) | 15 | 0.052 ± 0.051 | 0.766 ± 0.221 | 0.058 ± 0.084 | 0.300 ± 0.180 | 0.337 ± 0.236 | 0.335 ± 0.237 |

  Guard (from `results/tables/e11_guard_choice.md`):

| score | select-half OOD AUROC |
|---|---|
| msp | 0.9761 |
| energy | 0.9761 |
| maha | 0.973 |
| knn | 0.9593 |

- Interpretation: for the final model MSP is the best guard on the `select` half; Mahalanobis is best on near OOD only for ConvNeXt and DINOv2 (`e11_ood.md`). The full system's error on issued predictions is 0.184 at alpha 0.10 against 0.829 for the bare classifier, mainly because the guard rejects 0.766 of OOD images; almost all remaining errors are accepted OOD images.
- Issues / anomalies: alpha 0.05 issues more and errs more than alpha 0.10 (E11.md findings). The table note mislabelled the last two columns as fractions of the stream; corrected in 3006ecc (numbers unchanged).
- Next: CHECKPOINT 5.

## 2026-10-25 · E10 · Calibration, selective prediction, conformal (Phase 5 kit)
- Commit: as E11 (same kit and results bundle); tables 101644f.
- Command(s): pipeline `p5_reliability`; `python scripts/phase5_tables.py --exp e10`; `python scripts/make_figures.py --exp e10`
- Hypothesis: temperature scaling does not raise ECE; the 3-seed ensemble has lower AURC; Mondrian keeps per-class coverage within 2 pp where marginal may not (E10.md).
- Runs: as E11 (75 fold models, 0 failures).
- Result (from `results/tables/e10_calibration.md`):

| config | runs | images | T fitted (identifiable) | temperature (fitted runs) | ece before | ece after | ece diff (after - before) | brier before | brier after | brier diff (after - before) | log_loss before | log_loss after | log_loss diff (after - before) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext-tiny/ft | 15 | 549 | 3 | 1.87 ± 0.94 | 0.0069 | 0.0069 | -0.0000 [-0.0007, 0.0006] | 0.0061 | 0.0061 | -0.0000 [-0.0001, 0.0001] | 0.0245 | 0.0246 | 0.0001 [-0.0003, 0.0005] |
| dinov2-vits14/ft | 15 | 549 | 8 | 2.00 ± 0.93 | 0.0137 | 0.017 | 0.0034 [-0.0042, 0.0095] | 0.0102 | 0.0102 | -0.0001 [-0.0007, 0.0004] | 0.0485 | 0.0525 | 0.0040 [0.0003, 0.0063] |
| efficientnetv2-s/default | 15 | 549 | 7 | 1.29 ± 0.55 | 0.0103 | 0.0099 | -0.0004 [-0.0030, 0.0030] | 0.0075 | 0.0074 | -0.0000 [-0.0003, 0.0001] | 0.0296 | 0.028 | -0.0016 [-0.0074, 0.0015] |
| handcrafted-xgb/default | 15 | 549 | 15 | 1.06 ± 0.18 | 0.0319 | 0.0331 | 0.0012 [-0.0117, 0.0153] | 0.0606 | 0.061 | 0.0004 [-0.0008, 0.0015] | 0.1963 | 0.1996 | 0.0033 [-0.0038, 0.0100] |
| swin-tiny/ft | 15 | 549 | 3 | 1.04 ± 0.44 | 0.0039 | 0.004 | 0.0001 [-0.0002, 0.0005] | 0.0031 | 0.0031 | 0.0000 [-0.0000, 0.0001] | 0.0175 | 0.0176 | 0.0000 [-0.0002, 0.0003] |

  Conformal choice (from `results/tables/e10_conformal_choice.md`):

| score | mode | largest per-class shortfall (pp) | mean set size (alpha 0.10) | qualifies |
|---|---|---|---|---|
| lac | marginal | 1.63 | 0.9176 | True |
| lac | mondrian | 0.67 | 0.9169 | True |
| aps | marginal | 2.51 | 0.9176 | False |
| aps | mondrian | 2.36 | 0.9114 | False |

- Interpretation: temperature scaling neither helps nor hurts ECE or Brier (every CI includes 0); the temperature was identifiable for only 3–8 of 15 fold models of each deep model. The ensemble's mean AURC is equal or lower for all five models (`e10_selective.md`, descriptive). LAC Mondrian is chosen for the final model (ADR-028).
- Issues / anomalies: conformal sets are mostly singletons or empty; size-2 sets (abstentions) are almost absent for the deep models, so coverage 1 - alpha is reached through empty sets. Re-predicted held-out probabilities differ from the original OOF by up to 1.08e-02 (efficientnetv2-s) and 4.14e-03 (convnext-tiny); not investigated.
- Next: E11 uses LAC Mondrian.

## 2026-10-25 · E9 · Extension to severities 6–8 (Phase 5 kit, ADR-024)
- Commit: as E11 (same kit and results bundle); tables and figure 101644f.
- Command(s): pipeline `e09_extension`; `python scripts/phase5_tables.py --exp e09ext`; `python scripts/make_figures.py --exp e09ext`
- Hypothesis: the stronger severities find a failure point at the 0.85 limit (ADR-024).
- Runs: no training; the same 75 fold models each wrote `corruption_predictions_ext.csv` (27 conditions); 0 failures (`results/pipeline/e09_extension_status.json`).
- Result (from `results/tables/e09_ext_quality_limits.md`):

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

- Interpretation: the final model still has no failure point; its lowest condition is perspective warp at severity 8 (0.8743). DINOv2 (brightness from severity 6) and ConvNeXt (brightness at severity 8) do fall below 0.85 (`e09_ext_severity.md`).
- Issues / anomalies: the 1–5 rows equal `e09_severity` exactly. Handcrafted rotation AUROC is not monotonic in severity (0.8815 at 45°, 0.9602 at 90°).
- Next: none for E9.

## 2026-10-20 · E9 · Robustness to image corruptions (Phase 4 kit)
- Commit: Phase 4 GPU kit `PRIVATE_gpu_kit_phase4_d0d2288.zip` (RTX 5060), results `RESULTS_phase4_20261020-125833.zip`; tables committed with this entry.
- Command(s): pipeline `run_e9` (after stage 3); `python scripts/phase4_tables.py --exp e09`; `python scripts/make_figures.py --exp e09`
- Hypothesis: AUROC falls with severity; blur and JPEG hurt most (E09.md).
- Runs: no training; the 75 fold models (`p4ref_*` and `e05_efficientnetv2-s_default_*`) each wrote `corruption_predictions.csv` (45 conditions + clean); 0 failures (`results/pipeline/e09_status.json`).
- Result (from `results/tables/e09_mean_corruption_auroc.md`):

| config | mean corruption AUROC | worst condition AUROC | conditions | clean AUROC |
|---|---|---|---|---|
| convnext-tiny/ft | 0.9983 | 0.9831 | 45 | 0.9996 |
| dinov2-vits14/ft | 0.9953 | 0.961 | 45 | 0.9975 |
| efficientnetv2-s/default | 0.9984 | 0.9814 | 45 | 0.9993 |
| handcrafted-xgb/default | 0.9451 | 0.8142 | 45 | 0.9768 |
| swin-tiny/ft | 0.9988 | 0.992 | 45 | 0.9995 |

- Interpretation: the deep models lose at most 0.04 AUROC in any condition; the handcrafted model falls below 0.85 under Gaussian noise (severity 5) and perspective warp (severity 5). Blur and JPEG do not hurt most. The final model's quality limit is severity 5 for every corruption (`e09_quality_limits.md`), so the 0.85 threshold (ADR-023) does not discriminate at the tested severities.
- Issues / anomalies: none in the runs. The severity grid did not reach a failure point for any deep model.
- Next: report at CHECKPOINT 4; a stronger severity grid would be a protocol change (Sophie's call).

## 2026-10-20 · E8 · Open set, leave one packaging type out (Phase 4 kit + PatchCore rerun)
- Commit: Phase 4 GPU kit (d0d2288); PatchCore fix 9ca2980; tables committed with this entry.
- Command(s): pipeline stage 3; after the fix, on the laptop CPU: `python -m pharmachecked.run experiment=e08_open_set model=patchcore task=one_class variant=oc-no-<T> paths=local data.split_version=2 exp=e08 data.heldout_type=<T> fold=0 seed=0` for T = blister, packaging; `python scripts/phase4_tables.py --exp e08`; `python scripts/make_figures.py --exp e08`
- Hypothesis: supervised models lose AUROC on the unseen type (E08.md).
- Runs: 34 under `results/runs/e08_*`; stage wall time 2026-10-20 12:02 to 12:30 on the GPU.
- Result (from `results/tables/e08_open_set.md`):

| held-out type | method | runs | unseen AUROC | seen AUROC (5-fold OOF) | unseen AUPRC | unseen sens@95%spec | images |
|---|---|---|---|---|---|---|---|
| blister | convnext-tiny/ft | 3 | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 | 1.000 | 229 |
| blister | dinov2-vits14/ft | 3 | 0.918 [0.882, 0.948] | 1.000 [1.000, 1.000] | 0.914 | 0.629 | 229 |
| blister | efficientnetv2-s/default | 3 | 0.954 [0.927, 0.976] | 1.000 [1.000, 1.000] | 0.943 | 0.638 | 229 |
| blister | handcrafted-xgb/default | 3 | 0.970 [0.948, 0.987] | 0.987 [0.977, 0.995] | 0.961 | 0.829 | 229 |
| blister | patchcore/oc | 1 | 0.907 [0.867, 0.942] | - | 0.874 | 0.648 | 229 |
| blister | prototype-dinov2/oc | 1 | 0.944 [0.915, 0.970] | - | 0.943 | 0.733 | 229 |
| blister | swin-tiny/ft | 3 | 0.994 [0.986, 0.998] | 1.000 [1.000, 1.000] | 0.992 | 0.943 | 229 |
| packaging | convnext-tiny/ft | 3 | 0.996 [0.990, 0.999] | 0.999 [0.998, 1.000] | 0.997 | 0.969 | 175 |
| packaging | dinov2-vits14/ft | 3 | 0.975 [0.956, 0.991] | 0.993 [0.978, 1.000] | 0.980 | 0.773 | 175 |
| packaging | efficientnetv2-s/default | 3 | 0.995 [0.988, 0.999] | 0.998 [0.993, 1.000] | 0.996 | 0.959 | 175 |
| packaging | handcrafted-xgb/default | 3 | 0.879 [0.825, 0.927] | 0.964 [0.937, 0.985] | 0.880 | 0.412 | 175 |
| packaging | patchcore/oc | 1 | 0.940 [0.903, 0.971] | - | 0.944 | 0.784 | 175 |
| packaging | prototype-dinov2/oc | 1 | 0.970 [0.943, 0.990] | - | 0.972 | 0.784 | 175 |
| packaging | swin-tiny/ft | 3 | 0.996 [0.990, 0.999] | 0.998 [0.994, 1.000] | 0.997 | 0.959 | 175 |

- Interpretation: unseen < seen for 9 of 10 supervised pairs; large losses for DINOv2 and EfficientNetV2-S on blisters and the handcrafted model on packaging; ConvNeXt and Swin lose ≤ 0.006. The prototype verifier beats PatchCore on both types.
- Issues / anomalies: both kit PatchCore runs had AUROC 0.500 [0.500, 0.500]: `prob = 1 - exp(-score)` was exactly 1.0 for every image (logit clipped at 16.118). Fixed in 9ca2980 (`score / (1 + score)`, regression test) and rerun on CPU (fp32, same config); the kit runs and one rerun that was marked `_dirty` (docs edited while it started) are in `results/scratch/superseded/`. The prototype runs were not affected (cosine distance / 2 does not saturate).
- Next: none for Phase 4.

## 2026-10-20 · E7 · Cross-dataset transfer, Kaggle -> external (Phase 4 kit)
- Commit: Phase 4 GPU kit (d0d2288); tables committed with this entry.
- Command(s): pipeline stage 2; `python scripts/phase4_tables.py --exp e07`
- Hypothesis: cross-source AUROC well below in-source for every model (E07.md, ADR-022).
- Runs: 15 under `results/runs/e07_*` (5 models × seeds 0–2), trained on the whole development pool; 0 failures; stage wall time 2026-10-20 11:50 to 12:02.
- Result (from `results/tables/e07_transfer.md`):

| config | runs | in-source AUROC (5-fold OOF) | cross-source AUROC | drop (in - cross) | external specificity (Mendeley) | external recall (Split E) | external images |
|---|---|---|---|---|---|---|---|
| convnext-tiny/ft | 3 | 1.000 [0.999, 1.000] | 0.626 [0.573, 0.678] | 0.373 [0.321, 0.427] | 0.577 | 0.558 | 449 |
| dinov2-vits14/ft | 3 | 0.998 [0.993, 1.000] | 0.847 [0.809, 0.883] | 0.151 [0.114, 0.190] | 0.953 | 0.269 | 449 |
| efficientnetv2-s/default | 3 | 0.999 [0.998, 1.000] | 0.860 [0.826, 0.891] | 0.139 [0.108, 0.172] | 0.835 | 0.600 | 449 |
| handcrafted-xgb/default | 3 | 0.976 [0.964, 0.986] | 0.605 [0.549, 0.663] | 0.371 [0.314, 0.433] | 0.476 | 0.660 | 449 |
| swin-tiny/ft | 3 | 1.000 [0.998, 1.000] | 0.538 [0.478, 0.597] | 0.461 [0.401, 0.521] | 0.821 | 0.211 | 449 |

- Interpretation: every model drops, and every drop CI excludes 0. The final model (0.860) and DINOv2 (0.847) transfer best; Swin, joint-best in-source, is lowest cross-source (0.538, CI includes 0.5). The cross-source AUROC also separates two acquisition sources (ADR-005).
- Issues / anomalies: none.
- Next: none for Phase 4.

## 2026-10-20 · E6 · Data efficiency (Phase 4 kit)
- Commit: Phase 4 GPU kit (d0d2288); tables committed with this entry.
- Command(s): pipeline stages 0–1; `python scripts/aggregate.py --exp e06 --include p4ref:handcrafted-xgb/default --include p4ref:convnext-tiny/ft --include p4ref:swin-tiny/ft --include p4ref:dinov2-vits14/ft --include e05:efficientnetv2-s/default`; `python scripts/make_figures.py --exp e06`
- Hypothesis: pretrained models stay near the ceiling at small fractions; the handcrafted model degrades earlier (E06.md).
- Runs: 60 `p4ref_*` (the 100% arm of the four retrained comparison models, ADR-021) + 225 `e06_*` (5 models × 3 fractions × 15); 0 failures. Stage 0 2026-10-20 08:28 to 09:14, stage 1 to 11:50.
- Result: full table in `results/tables/e06_summary.md`; pooled AUROC by fraction in `docs/experiments/E06.md` (copied from `e06_summary.csv`).
- Interpretation: ConvNeXt and Swin stay ≥ 0.998 at 10%; EfficientNetV2-S (0.987) and DINOv2 (0.972) lose a little at 10%; the handcrafted model falls from 0.976 to 0.895. Pooled ECE rises sharply at 10% (up to 0.158).
- Issues / anomalies: for some arms the pooled ECE point estimate lies just outside its bootstrap CI (e.g. `convnext-tiny/ft-frac25` 0.016 [0.017, 0.030]); bootstrap resamples with duplicated images bias binned ECE upwards. The same aggregator was used in earlier phases.
- Next: none for Phase 4.

## 2026-10-19 · E4 · Gate weights of the `full` model (PLAN 3.7)
- Commit: 0ec1fe3 (scripts); results committed with this entry.
- Command(s): `python scripts/predict_branch_outputs.py --exp e04 --model multibranch --variant full --paths local --checkpoint-root C:/Projects --device cpu`; `python scripts/gate_table.py --exp e04 --model multibranch --variant full`; `python scripts/make_figures.py --exp e04`
- Hypothesis: none (descriptive figure pre-registered in E04.md).
- Runs: no new training; the 15 `e04_multibranch_full_*` checkpoints (copied from the GPU PC to `C:\Projects\checkpoints`) re-predicted their held-out folds on the laptop CPU (about 8 minutes). Each run folder gained `branch_outputs.csv`.
- Result (from `results/tables/e04_gate_weights.md`):

| packaging_type | frequency | global | local | images |
|---|---|---|---|---|
| all | 0.047 ± 0.088 | 0.607 ± 0.104 | 0.346 ± 0.149 | 549 |
| blister | 0.057 ± 0.098 | 0.553 ± 0.060 | 0.389 ± 0.134 | 229 |
| both | 0.002 ± 0.003 | 0.837 ± 0.370 | 0.161 ± 0.370 | 21 |
| other | 0.002 ± 0.003 | 0.854 ± 0.329 | 0.143 ± 0.329 | 124 |
| packaging | 0.068 ± 0.145 | 0.471 ± 0.078 | 0.461 ± 0.154 | 175 |

- Interpretation: the frequency branch gets little gate weight for every packaging type, consistent with its ablation making no difference. Global and local weights are close for `packaging`; for `both` and `other` the global branch dominates on average but varies strongly between runs (sd 0.33-0.37).
- Issues / anomalies: the re-prediction ran in fp32 on CPU, the original runs in bf16 autocast on the GPU; the largest per-image probability difference against the original OOF predictions is 5.82e-02 (`results/pipeline/e04_full_branch_outputs.json`). The GPU kit `PRIVATE_gpu_kit_e04-gates_0ec1fe3.zip` built for this was not used.
- Next: PLAN 3.7 complete; Phase 3 done.

## 2026-10-19 · E4 · Multi-branch model and ablations (Phase 3 kit)
- Commit: Phase 3 GPU kit `PRIVATE_gpu_kit_phase3_0d31953.zip` (RTX 5060, data v2), results `RESULTS_phase3_20261014-062137.zip` installed in a6de29d; tables in b060204.
- Command(s): pipeline stage 3; `python scripts/aggregate.py --exp e04 --include e05:efficientnetv2-s/default --reference multibranch/full`; `python scripts/external_auroc.py --exp e01 --exp e02 --exp e03 --exp e04 --exp e05`
- Hypothesis: ablations indistinguishable from `full` in the development pool; the frequency branch may show on the external block (E04.md).
- Runs: 120 under `results/runs/e04_multibranch_*` (8 arms × 15), default recipe (ADR-019); 8.5 h on the GPU (2026-10-13 21:49 to 2026-10-14 06:22).
- Result: full table in `results/tables/e04_summary.md`. Each arm vs `full` (from `results/tables/e04_vs_reference.md`):

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/default | multibranch/full | 549 | 0.000 [-0.000, 0.001] | -0.004 [-0.014, 0.004] | -0.021 [-0.042, -0.003] | 0.3493 | 1 | False | False |
| multibranch/concat | multibranch/full | 549 | -0.001 [-0.003, 0.000] | -0.001 [-0.008, 0.009] | -0.001 [-0.014, 0.015] | 0.2819 | 1 | False | False |
| multibranch/hard-routing | multibranch/full | 549 | 0.000 [0.000, 0.001] | -0.000 [-0.009, 0.008] | -0.017 [-0.030, -0.007] | 0.1171 | 0.8875 | False | False |
| multibranch/no-frequency | multibranch/full | 549 | -0.000 [-0.001, 0.000] | 0.001 [-0.006, 0.006] | -0.011 [-0.021, -0.003] | 0.6562 | 1 | False | False |
| multibranch/no-local | multibranch/full | 549 | -0.003 [-0.008, -0.000] | -0.003 [-0.007, 0.011] | 0.014 [-0.008, 0.038] | 0.1109 | 0.8875 | False | False |
| multibranch/no-router | multibranch/full | 549 | 0.001 [0.000, 0.001] | 0.003 [-0.005, 0.008] | -0.019 [-0.032, -0.009] | 0.1426 | 0.8875 | False | False |
| multibranch/separate-local | multibranch/full | 549 | -0.001 [-0.004, 0.001] | -0.006 [-0.016, -0.001] | -0.025 [-0.042, -0.008] | 0.4959 | 1 | False | False |
| multibranch/tiles4-896 | multibranch/full | 549 | -0.000 [-0.002, 0.001] | 0.014 [0.004, 0.023] | 0.009 [-0.013, 0.029] | 0.7151 | 1 | False | False |

- Interpretation: no arm differs significantly from `full`, including the global-only reference, so by the pre-registered rule no branch is kept and the final model falls back to the global-only EfficientNetV2-S (ADR-020, proposed). On the external block all multi-branch arms have much lower Mendeley specificity and higher Split E recall than the global-only reference; `no-local` moves furthest back towards it, `no-frequency` does not. Exploratory threshold-free external AUROC (`external_auroc_exploratory.md`): global-only 0.851 [0.815, 0.884], `full` 0.529 [0.470, 0.589].
- Issues / anomalies: no job failed. `e04_gate_weights_by_packaging` (PLAN 3.7) could not be built: gate weights are not written to the run folders and the checkpoints are on the GPU PC. `tiles4-896` took 12.0 min per run vs 2.8 for `full`.
- Next: Sophie confirms ADR-020 at CHECKPOINT 3; decide how to produce the gate-weight figure.

## 2026-10-19 · E5 · Training recipe (Phase 3 kit)
- Commit: Phase 3 GPU kit `PRIVATE_gpu_kit_phase3_0d31953.zip` (RTX 5060, data v2), results `RESULTS_phase3_20261014-062137.zip` installed in a6de29d; tables in b060204.
- Command(s): pipeline stage 1 (decision in `results/pipeline/e05_decisions.json`); `python scripts/aggregate.py --exp e05 --reference efficientnetv2-s/default`
- Hypothesis: no component meets the improvement criterion at the ceiling (E05.md).
- Runs: 135 under `results/runs/e05_efficientnetv2-s_*` (9 arms × 15); 2.8 h on the GPU.
- Result: full table in `results/tables/e05_summary.md`. Each component vs `default` (from `results/tables/e05_vs_reference.md`):

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/colorjitter | efficientnetv2-s/default | 549 | -0.001 [-0.002, 0.000] | 0.010 [0.004, 0.014] | 0.012 [0.003, 0.022] | 0.2167 | 1 | False | False |
| efficientnetv2-s/cutmix | efficientnetv2-s/default | 549 | 0.000 [-0.000, 0.001] | 0.043 [0.033, 0.048] | 0.038 [0.024, 0.049] | 0.2386 | 1 | False | False |
| efficientnetv2-s/focal | efficientnetv2-s/default | 549 | 0.000 [-0.000, 0.001] | 0.028 [0.024, 0.036] | 0.030 [0.014, 0.041] | 0.4834 | 1 | False | False |
| efficientnetv2-s/hflip | efficientnetv2-s/default | 549 | 0.000 [-0.000, 0.001] | -0.001 [-0.004, 0.003] | -0.002 [-0.010, 0.005] | 0.6564 | 1 | False | False |
| efficientnetv2-s/mixup | efficientnetv2-s/default | 549 | 0.000 [-0.000, 0.001] | 0.019 [0.013, 0.025] | 0.017 [0.004, 0.026] | 0.9516 | 1 | False | False |
| efficientnetv2-s/randaugment | efficientnetv2-s/default | 549 | -0.000 [-0.001, 0.000] | 0.000 [-0.005, 0.003] | 0.004 [-0.004, 0.017] | 0.6583 | 1 | False | False |
| efficientnetv2-s/smoothing | efficientnetv2-s/default | 549 | -0.000 [-0.001, 0.000] | 0.056 [0.047, 0.065] | 0.062 [0.049, 0.073] | 0.2747 | 1 | False | False |
| efficientnetv2-s/supcon | efficientnetv2-s/default | 549 | -0.003 [-0.008, -0.001] | 0.072 [0.057, 0.084] | 0.106 [0.090, 0.123] | 0.07076 | 0.5661 | False | False |

- Interpretation: no component improves pooled AUROC, so the recipe stays the default and no combination arm ran (ADR-019). ECE differences are above 0 (worse calibration) for colorjitter, cutmix, focal, mixup, smoothing and supcon.
- Issues / anomalies: none; no job failed.
- Next: E4 used the default recipe.

## 2026-10-19 · E3 · Retrain of the E1-best arm on data v2 (PLAN 2.14)
- Commit: Phase 3 GPU kit `PRIVATE_gpu_kit_phase3_0d31953.zip` (RTX 5060, data v2), results `RESULTS_phase3_20261014-062137.zip` installed in a6de29d; tables in b060204.
- Command(s): pipeline stage 0; `python scripts/aggregate.py --exp e03 --include e01:swin-tiny/ft-tuned --reference swin-tiny/ft-tuned`
- Hypothesis: removing the 2 `not_a_drug` images does not change pooled AUROC.
- Runs: 15 under `results/runs/e03_swin-tiny_ft-tuned-v2_*`; 0.2 h on the GPU.
- Result: before (v1, E1) vs after (v2), from `results/tables/e03_vs_reference.md`:

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| swin-tiny/ft-tuned-v2 | swin-tiny/ft-tuned | 549 | 0.000 [-0.000, 0.000] | 0.000 [-0.009, 0.002] | -0.007 [-0.020, 0.007] | 0.3764 | 0.3764 | False | False |

- Interpretation: no difference on the 549 images both versions share. The two runs also differ in augmentation draw (ADR-018), so the external columns in `e03_summary.md` (Mendeley specificity 0.761/0.660 → 0.540/0.424, Split E recall up) cannot be attributed to the data change alone.
- Issues / anomalies: none.
- Next: data v2 stays in use.

## 2026-10-13 · E2 · Adaptation strategy (Phase 2 kit)
- Commit: 053e606 (same kit as E1); tables in 1714be0.
- Command(s): pipeline stage 5; `python scripts/aggregate.py --exp e02 --include e01:convnext-tiny/ft --include e01:efficientnetv2-s/ft`; `python scripts/make_figures.py --exp e02`
- Hypothesis: none directional (E02.md).
- Runs: 120 under `results/runs/e02_*` (2 backbones × lp, last2, lora8, lora16 × 15) plus the 30 E1 full fine-tuning runs of the two backbones.
- Result: full table in `results/tables/e02_summary.md`; figure `results/figures/e02_auroc_by_strategy.png`. Pairwise (from `results/tables/e02_pairwise.md`):

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| convnext-tiny/ft | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.002] | 0.2762 | 1 | False |
| convnext-tiny/last2 | efficientnetv2-s/lora8 | 0.002 [-0.000, 0.005] | 0.3342 | 1 | False |
| convnext-tiny/lora16 | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.004] | 0.3889 | 1 | False |
| convnext-tiny/lora8 | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.004] | 0.3805 | 1 | False |
| convnext-tiny/lp | efficientnetv2-s/lora8 | 0.006 [0.002, 0.012] | 0.01057 | 0.08455 | False |
| efficientnetv2-s/ft | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.002] | 0.2802 | 1 | False |
| efficientnetv2-s/last2 | efficientnetv2-s/lora8 | 0.002 [-0.000, 0.004] | 0.1827 | 1 | False |
| efficientnetv2-s/lora16 | efficientnetv2-s/lora8 | 0.000 [-0.000, 0.001] | 0.3978 | 1 | False |
| efficientnetv2-s/lp | efficientnetv2-s/lora8 | 0.010 [0.005, 0.017] | 0.0008855 | 0.007969 | True |

- Interpretation: against the best strategy (efficientnetv2-s/lora8), only efficientnetv2-s/lp is significantly worse; all other strategies of both backbones are statistically indistinguishable in the development pool. The summary table gives trainable parameters, log-loss, calibration and the external block for the E4 choice.
- Issues / anomalies: none; no job failed.
- Next: ADR choosing the E4 global-branch backbone and strategy (pending Sophie).

## 2026-10-13 · E1 · Baseline ladder (full protocol, Phase 2 kit)
- Commit: 053e606 (kit `PRIVATE_gpu_kit_phase2_053e606.zip`, RTX 5060); tier 1 on the laptop at ba6a80d. Results installed and verified (2,900 files), committed in 0a7cdbb; tables in 1714be0.
- Command(s): GPU kit `2_RUN.bat` → `python scripts/phase2_pipeline.py --profile full`; `python scripts/aggregate.py --exp e01`; `python scripts/make_figures.py --exp e01`
- Hypothesis: pretrained backbones beat handcrafted features and the from-scratch CNN on pooled OOF AUROC; in-distribution AUROC may saturate, so the external block separates models (E01.md).
- Runs: 290 under `results/runs/e01_*` (19 trained arms × 15, zero-shot 5): 185 default GPU runs, 30 laptop tier-1 runs, 75 tuned re-runs. 5 Optuna studies in `results/optuna/`; decisions in `results/pipeline/e01_decisions.json`.
- Result: full table in `results/tables/e01_summary.md` (per-run, pooled with CIs, external block); figure `results/figures/e01_auroc_vs_params.png`. Pairwise (from `results/tables/e01_pairwise.md`):

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| clip-vitb16/ft | swin-tiny/ft-tuned | 0.005 [-0.000, 0.015] | 0.1798 | 1 | False |
| clip-vitb16/lp | swin-tiny/ft-tuned | 0.005 [0.002, 0.009] | 0.01067 | 0.128 | False |
| clip-vitb16/zeroshot | swin-tiny/ft-tuned | 0.297 [0.255, 0.342] | 1.991e-41 | 3.783e-40 | True |
| cnn-small/ft | swin-tiny/ft-tuned | 0.054 [0.039, 0.073] | 1.692e-09 | 2.876e-08 | True |
| cnn-small/ft-tuned | swin-tiny/ft-tuned | 0.084 [0.062, 0.107] | 1.204e-12 | 2.168e-11 | True |
| convnext-tiny/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.001] | 0.1329 | 1 | False |
| convnext-tiny/ft-tuned | swin-tiny/ft-tuned | 0.001 [-0.000, 0.003] | 0.267 | 1 | False |
| dinov2-vitb14/ft | swin-tiny/ft-tuned | 0.003 [-0.000, 0.011] | 0.3136 | 1 | False |
| dinov2-vitb14/lp | swin-tiny/ft-tuned | 0.004 [0.001, 0.007] | 0.02342 | 0.2576 | False |
| dinov2-vits14/ft | swin-tiny/ft-tuned | 0.003 [0.000, 0.009] | 0.2396 | 1 | False |
| dinov2-vits14/ft-tuned | swin-tiny/ft-tuned | 0.003 [0.000, 0.008] | 0.1338 | 1 | False |
| dinov2-vits14/lp | swin-tiny/ft-tuned | 0.006 [0.002, 0.012] | 0.03798 | 0.3798 | False |
| efficientnetv2-s/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.002] | 0.1542 | 1 | False |
| handcrafted-lr/default | swin-tiny/ft-tuned | 0.048 [0.033, 0.067] | 1.201e-08 | 1.922e-07 | True |
| handcrafted-xgb/default | swin-tiny/ft-tuned | 0.023 [0.014, 0.034] | 4.527e-06 | 6.79e-05 | True |
| handcrafted-xgb/default-tuned | swin-tiny/ft-tuned | 0.023 [0.014, 0.034] | 5.46e-06 | 7.644e-05 | True |
| resnet50/ft | swin-tiny/ft-tuned | 0.012 [0.006, 0.020] | 0.0002731 | 0.003551 | True |
| swin-tiny/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.003] | 0.3569 | 1 | False |
| vit-b16/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.003] | 0.264 | 1 | False |

- Interpretation: against the best arm (swin-tiny/ft-tuned), only zero-shot CLIP, both small-CNN arms, the three handcrafted arms and resnet50/ft are significantly worse (Holm p < 0.05, CI excluding 0). All other pretrained arms are statistically indistinguishable in the development pool. On the external sets (summary table), arms with high Mendeley specificity have low Split E recall and vice versa; no arm is high on both.
- Issues / anomalies: no job failed. At install, 30 laptop tier-1 `checkpoint_path.txt` files differed from the bundle only by CRLF line endings; they were restored from git (LF) and the bundle installed unchanged.
- Next: the pre-registered E2 backbones (convnext-tiny, efficientnetv2-s) ran in the same kit; E3 label audit on the E1-best model; decision on freezing tuned settings.

## 2026-10-10 · E1 · Timing batch on the GPU PC (fold 0, seed 0 of every GPU arm)
- Commit: ba6a80d (kit `PRIVATE_gpu_kit_e01-timing_ba6a80d.zip`); results installed with `scripts/unpack_results.py` (94 files verified).
- Command(s): GPU kit `2_RUN.bat` running `jobs/e01_gpu_timing.txt`; `python scripts/timing_table.py --exp e01 --log results/job_logs/e01-timing_20261010-204318_e01_gpu_timing.log --jobs jobs/e01_gpu.txt`
- Hypothesis: none (timing; CLAUDE.md §12).
- Runs: 13, `results/runs/e01_*_f0_s0_202610102*` (GPU: NVIDIA GeForce RTX 5060, torch 2.14.1+cu130).
- Result (from `results/tables/e01_timing.md`):

Generated by `scripts/timing_table.py` from `results/job_logs/e01-timing_20261010-204318_e01_gpu_timing.log` (wall minutes per job, including data loading and evaluation) and the fold-0/seed-0 run folders. Estimate = wall minutes x runs in `jobs/e01_gpu.txt`.

| Arm | Wall min (fold 0, seed 0) | Epochs run | Best epoch | Train min | Peak GPU GB | Runs | Estimated GPU-hours |
|---|---|---|---|---|---|---|---|
| clip-vitb16 / ft | 2.9 | 8 | 0 | 2.0 | 3.64 | 15 | 0.7 |
| clip-vitb16 / lp | 6.0 | 22 | 14 | 5.3 | 0.78 | 15 | 1.5 |
| clip-vitb16 / zeroshot | 0.6 | - | - | 0.0 | - | 5 | 0.1 |
| cnn-small / ft | 5.8 | 20 | 12 | 5.0 | 1.34 | 15 | 1.4 |
| convnext-tiny / ft | 3.0 | 8 | 0 | 2.3 | 2.48 | 15 | 0.8 |
| dinov2-vitb14 / ft | 2.9 | 8 | 0 | 2.1 | 4.46 | 15 | 0.7 |
| dinov2-vitb14 / lp | 6.2 | 23 | 15 | 5.5 | 0.85 | 15 | 1.6 |
| dinov2-vits14 / ft | 2.8 | 8 | 0 | 2.1 | 2.00 | 15 | 0.7 |
| dinov2-vits14 / lp | 6.2 | 24 | 16 | 5.7 | 0.39 | 15 | 1.6 |
| efficientnetv2-s / ft | 3.5 | 10 | 2 | 2.8 | 2.89 | 15 | 0.9 |
| resnet50 / ft | 8.3 | 29 | 21 | 7.5 | 1.85 | 15 | 2.1 |
| swin-tiny / ft | 3.1 | 9 | 1 | 2.5 | 2.75 | 15 | 0.8 |
| vit-b16 / ft | 3.1 | 8 | 0 | 2.2 | 3.50 | 15 | 0.8 |

Total estimated GPU-hours: 13.5

- Interpretation: all arms fit in 8 GB (peak 4.46 GB). The full E1 GPU list is estimated at 13.5 GPU-hours with the current settings.
- Issues / anomalies:
    - For five full fine-tuning arms (clip-vitb16, convnext-tiny, dinov2-vitb14, vit-b16, and dinov2-vits14 at epoch 0), validation AUROC is 1.000 after the first (warm-up) epoch. Early stopping on val AUROC cannot improve on it, so the epoch-0 weights are selected and training stops at epoch 8. Model selection is degenerate at the in-distribution ceiling (ADR-005).
    - The GPU benchmark (`bench_machine.py`) produced no table on the GPU PC, and its output was not captured; kit_run does not stop on a benchmark error.
- Next: decide the selection rule at the ceiling before the full E1 batch (decision pending with Sophie).
