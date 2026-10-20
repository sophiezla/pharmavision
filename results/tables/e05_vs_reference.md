# e05: each configuration vs `efficientnetv2-s/default`

Paired over the images both configurations have (pooled seed-averaged OOF predictions). Differences are configuration minus reference with paired bootstrap CIs (2000 resamples); DeLong p Holm-corrected over the comparisons. 'improves' / 'worse' = Holm p < 0.05 and an AUROC-difference CI above / below 0 (CLAUDE.md section 5). Lower ECE and log-loss are better.

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
