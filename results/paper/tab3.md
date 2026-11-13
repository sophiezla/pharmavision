# tab3: E11 end-to-end system vs bare classifier

| system | runs | ID rejected | OOD rejected | abstained | issued | error on issued | OOD among issued |
|---|---|---|---|---|---|---|---|
| bare classifier | 15 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.829 ± 0.003 | 0.826 ± 0.001 |
| full system (alpha 0.1) | 15 | 0.052 ± 0.051 | 0.766 ± 0.221 | 0.159 ± 0.184 | 0.199 ± 0.045 | 0.184 ± 0.139 | 0.183 ± 0.139 |
| full system (alpha 0.05) | 15 | 0.052 ± 0.051 | 0.766 ± 0.221 | 0.058 ± 0.084 | 0.300 ± 0.180 | 0.337 ± 0.236 | 0.335 ± 0.237 |
