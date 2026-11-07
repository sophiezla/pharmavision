# e15race: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| efficientnetv2-s/default | efficientnetv2-s/round2 | 0.000 [-0.000, 0.001] | 0.5041 | 0.5041 | False |
| efficientnetv2-s/round1 | efficientnetv2-s/round2 | 0.000 [0.000, 0.001] | 0.1614 | 0.4843 | False |
| efficientnetv2-s/round3 | efficientnetv2-s/round2 | 0.000 [-0.000, 0.000] | 0.1992 | 0.4843 | False |
