# e14: each configuration vs `efficientnetv2-s/default`

Paired over the images both configurations have (pooled seed-averaged OOF predictions). Differences are configuration minus reference with paired bootstrap CIs (2000 resamples); DeLong p Holm-corrected over the comparisons. 'improves' / 'worse' = Holm p < 0.05 and an AUROC-difference CI above / below 0 (CLAUDE.md section 5). Lower ECE and log-loss are better.

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/procedural-only | efficientnetv2-s/default | 549 | -0.277 [-0.318, -0.234] | 0.075 [0.057, 0.106] | 0.580 [0.552, 0.603] | 1.569e-37 | 4.706e-37 | False | True |
| efficientnetv2-s/procedural-r1 | efficientnetv2-s/default | 549 | -0.001 [-0.003, -0.000] | 0.074 [0.058, 0.092] | 0.151 [0.127, 0.173] | 0.03356 | 0.03356 | False | True |
| efficientnetv2-s/procedural-r4 | efficientnetv2-s/default | 549 | -0.012 [-0.019, -0.006] | 0.328 [0.304, 0.350] | 0.667 [0.634, 0.698] | 0.001174 | 0.002349 | False | True |
