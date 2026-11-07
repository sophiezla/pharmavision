# e16: each configuration vs `efficientnetv2-s/default`

Paired over the images both configurations have (pooled seed-averaged OOF predictions). Differences are configuration minus reference with paired bootstrap CIs (2000 resamples); DeLong p Holm-corrected over the comparisons. 'improves' / 'worse' = Holm p < 0.05 and an AUROC-difference CI above / below 0 (CLAUDE.md section 5). Lower ECE and log-loss are better.

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | efficientnetv2-s/default | 549 | -0.002 [-0.005, 0.000] | 0.025 [0.018, 0.038] | 0.052 [0.037, 0.068] | 0.246 | 0.4759 | False | False |
| mobilenetv3-large/kd-t4 | efficientnetv2-s/default | 549 | -0.002 [-0.005, 0.000] | 0.001 [-0.005, 0.009] | 0.015 [-0.002, 0.035] | 0.238 | 0.4759 | False | False |
