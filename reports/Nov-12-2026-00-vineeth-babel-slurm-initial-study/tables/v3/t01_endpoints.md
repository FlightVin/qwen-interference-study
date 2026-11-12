# Sealed test: base and round-128 endpoints

Sealed-test pass@1 (128 problems × 32 attempts, cap 12,288) of the base policy and the six round-128 endpoints, with the round-116 fallback and the four L40S reruns as supplementary rows (not endpoints). Interval for pass@1: problem bootstrap of the mean of c/32. Gain over base: paired problem bootstrap (audit pass B).

Sources: `research/audit-a26/pass-b/data/eval_test.json`; `reports/tables/v3/perproblem_counts.json`.

| Policy | Correct / attempts | pass@1 (%) | 95% interval | Gain over base (points) | 95% interval | Truncated (%) | pass@8 (%) |
|---|---|---|---|---|---|---|---|
| base | 2,153 / 4,096 | 52.6 | [46.5, 58.6] | — | — | 0.0 | 83.5 |
| H-s101 r128 | 3,152 / 4,096 | 77.0 | [71.5, 82.2] | +24.4 | [+19.7, +29.2] | 10.9 | 93.7 |
| R-s101 r128 | 3,353 / 4,096 | 81.9 | [76.7, 86.7] | +29.3 | [+24.5, +34.2] | 10.7 | 96.1 |
| U-s101 r128 | 3,182 / 4,096 | 77.7 | [72.4, 82.7] | +25.1 | [+20.5, +29.8] | 12.2 | 94.5 |
| H-s202 r128 | 2,965 / 4,096 | 72.4 | [66.4, 78.0] | +19.8 | [+15.3, +24.5] | 15.4 | 90.8 |
| R-s202 r128 | 3,063 / 4,096 | 74.8 | [69.1, 80.1] | +22.2 | [+18.0, +26.4] | 13.1 | 92.2 |
| U-s202 r128 | 3,245 / 4,096 | 79.2 | [74.0, 84.0] | +26.7 | [+21.9, +31.5] | 8.1 | 94.9 |
| H-s202 r116 fallback | 3,044 / 4,096 | 74.3 | [68.5, 79.8] | — | — | 14.8 | 92.8 |
| R-s101 L40S rerun | 3,369 / 4,096 | 82.3 | [77.2, 87.0] | — | — | 10.9 | 96.2 |
| R-s202 L40S rerun | 3,118 / 4,096 | 76.1 | [70.6, 81.4] | — | — | 12.4 | 93.4 |
| U-s101 L40S rerun | 3,187 / 4,096 | 77.8 | [72.6, 82.7] | — | — | 12.3 | 95.1 |
| U-s202 L40S rerun | 3,245 / 4,096 | 79.2 | [74.0, 84.0] | — | — | 8.1 | 94.9 |

pass@8 = unbiased estimate from 32 attempts per problem (DEFINITIONS §3.C).

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
