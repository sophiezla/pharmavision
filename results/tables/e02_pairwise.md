# e02: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| convnext-tiny/ft | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.002] | 0.2762 | 1 | False |
| convnext-tiny/last2 | efficientnetv2-s/lora8 | 0.002 [-0.000, 0.005] | 0.3342 | 1 | False |
| convnext-tiny/lora16 | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.004] | 0.3889 | 1 | False |
| convnext-tiny/lora8 | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.004] | 0.3805 | 1 | False |
| convnext-tiny/lp | efficientnetv2-s/lora8 | 0.006 [0.002, 0.012] | 0.01057 | 0.08455 | False |
| efficientnetv2-s/ft | efficientnetv2-s/lora8 | 0.001 [-0.000, 0.002] | 0.2802 | 1 | False |
| efficientnetv2-s/last2 | efficientnetv2-s/lora8 | 0.002 [-0.000, 0.004] | 0.1827 | 1 | False |
| efficientnetv2-s/lora16 | efficientnetv2-s/lora8 | 0.000 [-0.000, 0.001] | 0.3978 | 1 | False |
| efficientnetv2-s/lp | efficientnetv2-s/lora8 | 0.010 [0.005, 0.017] | 0.0008855 | 0.007969 | True |
