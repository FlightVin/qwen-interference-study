# Stress panels: pass@1 per tier

pass@1 on the stress panels (8 attempts per problem, cap 12,288): all 221 problems, Harder (95) and SuperHard (126). Tiers come from one annotation of solution structure, not from model success. Interval: problem bootstrap within the tier. The two round-64 cells are the only intermediate checkpoints evaluated; H-s202 r64 is trajectory a2-r60.

Sources: `research/audit-a26/pass-b/data/eval_stress.json`; `reports/tables/v3/perproblem_counts.json`.

| Policy | All (%) | 95% interval | Harder (%) | 95% interval | SuperHard (%) | 95% interval | Truncated, all (%) |
|---|---|---|---|---|---|---|---|
| base | 37.3 | [32.6, 42.0] | 52.5 | [45.1, 59.9] | 25.8 | [20.5, 31.3] | 0.1 |
| H-s101 r128 | 64.1 | [59.1, 69.1] | 79.5 | [73.2, 85.3] | 52.6 | [45.8, 59.4] | 17.7 |
| R-s101 r128 | 69.5 | [64.5, 74.4] | 84.2 | [78.6, 89.2] | 58.4 | [51.1, 65.7] | 20.5 |
| U-s101 r128 | 65.3 | [60.4, 70.2] | 80.1 | [74.1, 85.8] | 54.2 | [47.1, 60.9] | 23.0 |
| H-s202 r128 | 60.1 | [54.9, 65.2] | 74.3 | [67.6, 80.9] | 49.4 | [42.2, 56.5] | 25.5 |
| R-s202 r128 | 63.2 | [58.1, 68.1] | 76.3 | [69.7, 82.4] | 53.3 | [46.2, 60.3] | 23.0 |
| U-s202 r128 | 65.8 | [60.9, 70.7] | 79.5 | [73.3, 85.0] | 55.6 | [48.4, 62.5] | 15.2 |
| R-s101 r64 | 64.3 | [59.2, 69.3] | 78.6 | [72.2, 84.5] | 53.5 | [46.4, 60.4] | 26.1 |
| H-s202 r64 (a2-r60) | 62.4 | [57.5, 67.3] | 76.1 | [69.5, 82.1] | 52.1 | [45.3, 58.7] | 15.1 |

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
