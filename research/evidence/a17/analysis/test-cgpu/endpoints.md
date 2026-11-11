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
| 101 | 0.770 | 0.823 | 0.778 | 0.823 − 0.778 = +0.044 [+0.024, +0.065] | 0.823 − 0.770 = +0.053 [+0.025, +0.082] | 0.778 − 0.770 = +0.009 [-0.018, +0.035] | 0.937 / 0.962 / 0.951 |
| 202 | missing | 0.761 | 0.792 | 0.761 − 0.792 = -0.031 [-0.060, -0.004] | missing | missing | missing / 0.934 / 0.949 |
| Mean | missing | 0.792 | 0.785 | 0.792 − 0.785 = +0.007 [-0.010, +0.023] | missing | missing | – |

Common-GPU cell sources: H-s101: original evaluation (ran on L40S); H-s202: endpoint /data/user_data/vineethb/qis/test-eval/qis-test-H-s202-sym (common-GPU); R-s101: qis-cgpu-R-s101; R-s202: qis-cgpu-R-s202; U-s101: qis-cgpu-U-s101; U-s202: qis-cgpu-U-s202; base: original evaluation (ran on L40S).

## Reproducibility per checkpoint (cgpu vs original)

| Checkpoint | Identical reward | Identical request seed | Identical token ids | Identical c/32 | max abs Δc | pass@1 cgpu − original |
|---|---|---|---|---|---|---|
| R-s101 | 3556/4096 | 4096/4096 | 48/4096 | 62/128 | 8 | 0.823 − 0.819 = +0.004 [-0.008, +0.016] |
| R-s202 | 3455/4096 | 4096/4096 | 133/4096 | 46/128 | 8 | 0.761 − 0.748 = +0.013 [+0.002, +0.025] |
| U-s101 | 3691/4096 | 4096/4096 | 1964/4096 | 79/128 | 8 | 0.778 − 0.777 = +0.001 [-0.010, +0.012] |
| U-s202 | 4096/4096 | 4096/4096 | 4096/4096 | 128/128 | 0 | 0.792 − 0.792 = +0.000 [+0.000, +0.000] |

95% paired bootstrap over the 128 problems (10,000 draws, seed 2027; streams a9/<contrast>/s<seed> for the original table as in report_tables, a9/a17-cgpu/... and a9/a17-repro/... otherwise); conditional on these trained policies.
