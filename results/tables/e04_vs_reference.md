# e04: each configuration vs `multibranch/full`

Paired over the images both configurations have (pooled seed-averaged OOF predictions). Differences are configuration minus reference with paired bootstrap CIs (2000 resamples); DeLong p Holm-corrected over the comparisons. 'improves' / 'worse' = Holm p < 0.05 and an AUROC-difference CI above / below 0 (CLAUDE.md section 5). Lower ECE and log-loss are better.

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
