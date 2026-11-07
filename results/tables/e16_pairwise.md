# e16: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| efficientnet-lite0/kd-t4 | efficientnetv2-s/default | 0.002 [-0.000, 0.005] | 0.246 | 0.4759 | False |
| mobilenetv3-large/kd-t4 | efficientnetv2-s/default | 0.002 [-0.000, 0.005] | 0.238 | 0.4759 | False |
