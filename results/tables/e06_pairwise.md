# e06: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| convnext-tiny/ft-frac10 | convnext-tiny/ft | 0.002 [0.000, 0.004] | 0.09201 | 0.9201 | False |
| convnext-tiny/ft-frac25 | convnext-tiny/ft | 0.001 [0.000, 0.002] | 0.08229 | 0.9052 | False |
| convnext-tiny/ft-frac50 | convnext-tiny/ft | 0.000 [-0.000, 0.001] | 0.6591 | 1 | False |
| dinov2-vits14/ft | convnext-tiny/ft | 0.002 [-0.000, 0.006] | 0.274 | 1 | False |
| dinov2-vits14/ft-frac10 | convnext-tiny/ft | 0.028 [0.016, 0.040] | 8.992e-06 | 0.0001349 | True |
| dinov2-vits14/ft-frac25 | convnext-tiny/ft | 0.005 [0.001, 0.010] | 0.03288 | 0.3945 | False |
| dinov2-vits14/ft-frac50 | convnext-tiny/ft | 0.003 [0.000, 0.008] | 0.1234 | 0.9873 | False |
| efficientnetv2-s/default | convnext-tiny/ft | 0.000 [-0.000, 0.001] | 0.4088 | 1 | False |
| efficientnetv2-s/default-frac10 | convnext-tiny/ft | 0.013 [0.007, 0.020] | 0.0001614 | 0.00226 | True |
| efficientnetv2-s/default-frac25 | convnext-tiny/ft | 0.003 [0.001, 0.006] | 0.006415 | 0.08339 | False |
| efficientnetv2-s/default-frac50 | convnext-tiny/ft | 0.000 [-0.000, 0.001] | 0.3309 | 1 | False |
| handcrafted-xgb/default | convnext-tiny/ft | 0.023 [0.014, 0.035] | 4.828e-06 | 7.725e-05 | True |
| handcrafted-xgb/default-frac10 | convnext-tiny/ft | 0.105 [0.081, 0.132] | 6.682e-16 | 1.27e-14 | True |
| handcrafted-xgb/default-frac25 | convnext-tiny/ft | 0.066 [0.046, 0.087] | 3.406e-10 | 6.131e-09 | True |
| handcrafted-xgb/default-frac50 | convnext-tiny/ft | 0.032 [0.020, 0.045] | 5.849e-07 | 9.943e-06 | True |
| swin-tiny/ft | convnext-tiny/ft | 0.000 [-0.001, 0.001] | 0.8373 | 1 | False |
| swin-tiny/ft-frac10 | convnext-tiny/ft | 0.002 [0.000, 0.005] | 0.104 | 0.9359 | False |
| swin-tiny/ft-frac25 | convnext-tiny/ft | 0.001 [-0.000, 0.002] | 0.213 | 1 | False |
| swin-tiny/ft-frac50 | convnext-tiny/ft | 0.000 [-0.000, 0.001] | 0.6539 | 1 | False |
