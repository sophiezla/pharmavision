# e15race: each configuration vs `efficientnetv2-s/default`

Paired over the images both configurations have (pooled seed-averaged OOF predictions). Differences are configuration minus reference with paired bootstrap CIs (2000 resamples); DeLong p Holm-corrected over the comparisons. 'improves' / 'worse' = Holm p < 0.05 and an AUROC-difference CI above / below 0 (CLAUDE.md section 5). Lower ECE and log-loss are better.

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/round1 | efficientnetv2-s/default | 549 | -0.000 [-0.001, 0.001] | 0.002 [-0.004, 0.006] | 0.003 [-0.008, 0.017] | 0.6879 | 1 | False | False |
| efficientnetv2-s/round2 | efficientnetv2-s/default | 549 | 0.000 [-0.000, 0.001] | -0.000 [-0.006, 0.003] | -0.004 [-0.018, 0.010] | 0.5041 | 1 | False | False |
| efficientnetv2-s/round3 | efficientnetv2-s/default | 549 | 0.000 [-0.000, 0.001] | 0.001 [-0.004, 0.006] | 0.001 [-0.010, 0.011] | 0.6745 | 1 | False | False |
