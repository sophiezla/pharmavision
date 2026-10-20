# e03: each configuration vs `swin-tiny/ft-tuned`

Paired over the images both configurations have (pooled seed-averaged OOF predictions). Differences are configuration minus reference with paired bootstrap CIs (2000 resamples); DeLong p Holm-corrected over the comparisons. 'improves' / 'worse' = Holm p < 0.05 and an AUROC-difference CI above / below 0 (CLAUDE.md section 5). Lower ECE and log-loss are better.

| config | reference | images | auroc_diff | ece_diff | log_loss_diff | delong_p | holm_p | improves | worse |
|---|---|---|---|---|---|---|---|---|---|
| swin-tiny/ft-tuned-v2 | swin-tiny/ft-tuned | 549 | 0.000 [-0.000, 0.000] | 0.000 [-0.009, 0.002] | -0.007 [-0.020, 0.007] | 0.3764 | 0.3764 | False | False |
