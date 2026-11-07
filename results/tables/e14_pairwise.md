# e14: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| efficientnetv2-s/procedural-only | efficientnetv2-s/default | 0.277 [0.234, 0.318] | 1.569e-37 | 4.706e-37 | True |
| efficientnetv2-s/procedural-r1 | efficientnetv2-s/default | 0.001 [0.000, 0.003] | 0.03356 | 0.03356 | True |
| efficientnetv2-s/procedural-r4 | efficientnetv2-s/default | 0.012 [0.006, 0.019] | 0.001174 | 0.002349 | True |
