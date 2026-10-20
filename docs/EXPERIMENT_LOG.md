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
