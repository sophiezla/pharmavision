# e15: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| efficientnetv2-s/adv | efficientnetv2-s/default | 0.007 [0.002, 0.012] | 0.01201 | 0.02402 | True |
| efficientnetv2-s/procedural-r1 | efficientnetv2-s/default | 0.001 [0.000, 0.003] | 0.03356 | 0.03356 | True |
