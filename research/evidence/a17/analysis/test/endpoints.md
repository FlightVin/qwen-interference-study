# Endpoints (sealed test, pass@1, 128 problems × 32 attempts)

Base model: pass@1 0.526, pass@8 0.835.

## Primary: original evaluations

| Seed | H | R | U | R − U | R − H | U − H | pass@8 H / R / U |
|---|---|---|---|---|---|---|---|
| 101 | 0.770 | 0.819 | 0.777 | 0.819 − 0.777 = +0.042 [+0.021, +0.063] | 0.819 − 0.770 = +0.049 [+0.020, +0.080] | 0.777 − 0.770 = +0.007 [-0.020, +0.035] | 0.937 / 0.961 / 0.945 |
| 202 | missing | 0.748 | 0.792 | 0.748 − 0.792 = -0.044 [-0.072, -0.017] | missing | missing | missing / 0.922 / 0.949 |
| Mean | missing | 0.783 | 0.785 | 0.783 − 0.785 = -0.001 [-0.018, +0.015] | missing | missing | – |

## Reproducibility check: common GPU (L40S, original request seeds)

| Seed | H | R | U | R − U | R − H | U − H | pass@8 H / R / U |
|---|---|---|---|---|---|---|---|
| 101 | 0.770 | missing | missing | missing | missing | missing | 0.937 / missing / missing |
| 202 | missing | missing | missing | missing | missing | missing | missing / missing / missing |
| Mean | missing | missing | missing | missing | missing | missing | – |

Common-GPU cell sources: H-s101: original evaluation (ran on L40S); H-s202: endpoint /data/user_data/vineethb/qis/test-eval/qis-test-H-s202-sym (common-GPU); R-s101: qis-cgpu-R-s101; R-s202: qis-cgpu-R-s202; U-s101: qis-cgpu-U-s101; U-s202: qis-cgpu-U-s202; base: original evaluation (ran on L40S).

## Reproducibility per checkpoint (cgpu vs original)

| Checkpoint | Identical reward | Identical request seed | Identical token ids | Identical c/32 | max abs Δc | pass@1 cgpu − original |
|---|---|---|---|---|---|---|
| R-s101 | missing | | | | | |
| R-s202 | missing | | | | | |
| U-s101 | missing | | | | | |
| U-s202 | missing | | | | | |

95% paired bootstrap over the 128 problems (10,000 draws, seed 2027; streams a9/<contrast>/s<seed> for the original table as in report_tables, a9/a17-cgpu/... and a9/a17-repro/... otherwise); conditional on these trained policies.
