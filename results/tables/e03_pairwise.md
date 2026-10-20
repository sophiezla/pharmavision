# e03: each configuration vs the best pooled AUROC

Paired DeLong on pooled OOF predictions, Holm-corrected within the experiment; AUROC difference (best minus configuration) with a paired bootstrap CI. 'Improvement' requires Holm p < 0.05 and a CI excluding 0.

| config | best | auroc_diff | delong_p | holm_p | different |
|---|---|---|---|---|---|
| swin-tiny/ft-tuned | swin-tiny/ft-tuned-v2 | 0.000 [-0.000, 0.000] | 0.3764 | 0.3764 | False |
