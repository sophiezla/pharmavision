# e01: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| clip-vitb16/ft | swin-tiny/ft-tuned | 0.005 [-0.000, 0.015] | 0.1798 | 1 | False |
| clip-vitb16/lp | swin-tiny/ft-tuned | 0.005 [0.002, 0.009] | 0.01067 | 0.128 | False |
| clip-vitb16/zeroshot | swin-tiny/ft-tuned | 0.297 [0.255, 0.342] | 1.991e-41 | 3.783e-40 | True |
| cnn-small/ft | swin-tiny/ft-tuned | 0.054 [0.039, 0.073] | 1.692e-09 | 2.876e-08 | True |
| cnn-small/ft-tuned | swin-tiny/ft-tuned | 0.084 [0.062, 0.107] | 1.204e-12 | 2.168e-11 | True |
| convnext-tiny/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.001] | 0.1329 | 1 | False |
| convnext-tiny/ft-tuned | swin-tiny/ft-tuned | 0.001 [-0.000, 0.003] | 0.267 | 1 | False |
| dinov2-vitb14/ft | swin-tiny/ft-tuned | 0.003 [-0.000, 0.011] | 0.3136 | 1 | False |
| dinov2-vitb14/lp | swin-tiny/ft-tuned | 0.004 [0.001, 0.007] | 0.02342 | 0.2576 | False |
| dinov2-vits14/ft | swin-tiny/ft-tuned | 0.003 [0.000, 0.009] | 0.2396 | 1 | False |
| dinov2-vits14/ft-tuned | swin-tiny/ft-tuned | 0.003 [0.000, 0.008] | 0.1338 | 1 | False |
| dinov2-vits14/lp | swin-tiny/ft-tuned | 0.006 [0.002, 0.012] | 0.03798 | 0.3798 | False |
| efficientnetv2-s/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.002] | 0.1542 | 1 | False |
| handcrafted-lr/default | swin-tiny/ft-tuned | 0.048 [0.033, 0.067] | 1.201e-08 | 1.922e-07 | True |
| handcrafted-xgb/default | swin-tiny/ft-tuned | 0.023 [0.014, 0.034] | 4.527e-06 | 6.79e-05 | True |
| handcrafted-xgb/default-tuned | swin-tiny/ft-tuned | 0.023 [0.014, 0.034] | 5.46e-06 | 7.644e-05 | True |
| resnet50/ft | swin-tiny/ft-tuned | 0.012 [0.006, 0.020] | 0.0002731 | 0.003551 | True |
| swin-tiny/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.003] | 0.3569 | 1 | False |
| vit-b16/ft | swin-tiny/ft-tuned | 0.001 [-0.000, 0.003] | 0.264 | 1 | False |
