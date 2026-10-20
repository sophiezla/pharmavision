# e05: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| efficientnetv2-s/colorjitter | efficientnetv2-s/focal | 0.001 [-0.000, 0.003] | 0.2145 | 1 | False |
| efficientnetv2-s/cutmix | efficientnetv2-s/focal | 0.000 [-0.000, 0.001] | 0.8544 | 1 | False |
| efficientnetv2-s/default | efficientnetv2-s/focal | 0.000 [-0.000, 0.001] | 0.4834 | 1 | False |
| efficientnetv2-s/hflip | efficientnetv2-s/focal | 0.000 [-0.000, 0.001] | 0.6266 | 1 | False |
| efficientnetv2-s/mixup | efficientnetv2-s/focal | 0.000 [-0.000, 0.001] | 0.3145 | 1 | False |
| efficientnetv2-s/randaugment | efficientnetv2-s/focal | 0.000 [-0.000, 0.001] | 0.3163 | 1 | False |
| efficientnetv2-s/smoothing | efficientnetv2-s/focal | 0.000 [-0.000, 0.001] | 0.2882 | 1 | False |
| efficientnetv2-s/supcon | efficientnetv2-s/focal | 0.004 [0.000, 0.008] | 0.0942 | 0.7536 | False |
