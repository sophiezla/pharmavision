# e04: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| efficientnetv2-s/default | multibranch/no-router | 0.000 [-0.000, 0.001] | 0.5088 | 1 | False |
| multibranch/concat | multibranch/no-router | 0.001 [0.000, 0.004] | 0.212 | 1 | False |
| multibranch/full | multibranch/no-router | 0.001 [0.000, 0.001] | 0.1426 | 0.9981 | False |
| multibranch/hard-routing | multibranch/no-router | 0.000 [-0.000, 0.001] | 0.6049 | 1 | False |
| multibranch/no-frequency | multibranch/no-router | 0.001 [-0.000, 0.002] | 0.2433 | 1 | False |
| multibranch/no-local | multibranch/no-router | 0.004 [0.000, 0.009] | 0.1091 | 0.8724 | False |
| multibranch/separate-local | multibranch/no-router | 0.002 [-0.000, 0.006] | 0.3841 | 1 | False |
| multibranch/tiles4-896 | multibranch/no-router | 0.001 [-0.000, 0.002] | 0.149 | 0.9981 | False |
