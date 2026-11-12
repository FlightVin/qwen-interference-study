# Sampling null: problems that look lost or gained at an unchanged checkpoint

Lost = group A has ≥ 1 success and group B has none; gained = the reverse. Counts are out of 128 problems.

| Policy | m | Expected lost = expected gained (hypergeometric) | Binomial plug-in (p = c/32) | Empirical lost per split | Empirical gained per split |
|---|---|---|---|---|---|
| base | 8 | 5.72 | 5.36 | 5, 5 | 4, 5 |
| base | 16 | 3.37 | 3.02 | 2 | 3 |
| H-s101 | 8 | 3.13 | 2.81 | 2, 1 | 6, 3 |
| H-s101 | 16 | 1.93 | 1.75 | 4 | 2 |
| H-s202 | | missing | | | |
| R-s101 | 8 | 1.69 | 1.81 | 1, 2 | 0, 0 |
| R-s101 | 16 | 0.26 | 0.57 | 1 | 0 |
| R-s202 | 8 | 2.66 | 2.45 | 2, 3 | 2, 4 |
| R-s202 | 16 | 2.32 | 1.56 | 1 | 3 |
| U-s101 | 8 | 2.66 | 2.51 | 2, 2 | 1, 4 |
| U-s101 | 16 | 1.43 | 1.31 | 3 | 0 |
| U-s202 | 8 | 1.73 | 1.74 | 0, 1 | 2, 1 |
| U-s202 | 16 | 0.86 | 0.80 | 1 | 1 |

P_loss(p, m) under independent sampling (per problem; maximum 0.25 at p = 1 − 2^(−1/m)):

| p | 0.010 | 0.020 | 0.050 | 0.042 | 0.083 | 0.100 | 0.200 | 0.300 | 0.500 | 0.700 | 0.900 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m = 8 | 0.071 | 0.127 | 0.223 | 0.207 | 0.250 | 0.245 | 0.140 | 0.054 | 0.004 | 0.000 | 0.000 |
| m = 16 | 0.126 | 0.200 | 0.246 | 0.250 | 0.187 | 0.151 | 0.027 | 0.003 | 0.000 | 0.000 | 0.000 |

- Hypergeometric: expectation over random disjoint splits of the observed 32 attempts (conditional on c); lost and gained are equal by symmetry.
- Empirical: fixed index splits. The two m = 8 splits use disjoint attempts; the m = 16 split reuses the same 32 attempts, so the m = 8 and m = 16 counts are not independent.
- Pooled bins combine all policies; the same problem appears once per policy, so pooled rows are not independent.
- Dev panel: 53 problems × 8 attempts per evaluation (reports/tables/report_data.json: all 28 panel=dev eval rows have n=8, problems=53). The per-problem rate depends on p, and the dev panel has a different p distribution, so test-panel counts must not be rescaled to the dev panel; use P_loss(p, 8).
