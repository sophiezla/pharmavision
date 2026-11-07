# e15: each configuration vs `efficientnetv2-s/default`

Paired over the images both configurations have (pooled seed-averaged OOF predictions). Differences are configuration minus reference with paired bootstrap CIs (2000 resamples); DeLong p Holm-corrected over the comparisons. 'improves' / 'worse' = Holm p < 0.05 and an AUROC-difference CI above / below 0 (CLAUDE.md section 5). Lower ECE and log-loss are better.

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| efficientnetv2-s/adv | efficientnetv2-s/default | 549 | -0.007 [-0.012, -0.002] | 0.082 [0.071, 0.100] | 0.138 [0.108, 0.166] | 0.01201 | 0.02402 | False | True |
| efficientnetv2-s/procedural-r1 | efficientnetv2-s/default | 549 | -0.001 [-0.003, -0.000] | 0.074 [0.058, 0.092] | 0.151 [0.127, 0.173] | 0.03356 | 0.03356 | False | True |
