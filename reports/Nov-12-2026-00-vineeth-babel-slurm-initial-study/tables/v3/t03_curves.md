# In-run evaluations over training rounds

pass@1 of each run at rounds 0, 32, 64, 96 and 128 on dev and the three probes (8 attempts per problem). Round 0 is the same base policy in every run; its spread is cross-job sampling noise. The evaluation cap is 8,192 at rounds 0 and 32 and 12,288 from round 64. H-s202 rounds 64–128 are trajectory a2-r60 (resumed from round 60). Intervals in the JSON: problem bootstrap.

Sources: `research/audit-a26/pass-b/data/train_curves.json`; `research/audit-a26/pass-b/data/train_rewards.json`.

## pass@1 (%) on dev (53 problems), 8 attempts per problem

| Run | r0 | r32 | r64 | r96 | r128 |
|---|---|---|---|---|---|
| H-s101 | 41.7 | 54.0 | 63.7 | 63.9 | 66.3 |
| H-s202 | 40.3 | 56.6 | 63.7 | 66.0 | 67.0 |
| R-s101 | 41.7 | 54.2 | 73.6 | 74.8 | 72.9 |
| R-s202 | 40.3 | 59.9 | 61.3 | 64.2 | 65.6 |
| U-s101 | 42.2 | 60.1 | 71.0 | 68.4 | 69.1 |
| U-s202 | 42.2 | 61.1 | 67.9 | 67.9 | 70.3 |
| cap (tokens) | 8,192 | 8,192 | 12,288 | 12,288 | 12,288 |

## pass@1 (%) on J_H (32 hard dev problems = first 32 of dev), 8 attempts per problem

| Run | r0 | r32 | r64 | r96 | r128 |
|---|---|---|---|---|---|
| H-s101 | 42.6 | 57.0 | 65.6 | 67.2 | 68.8 |
| H-s202 | 41.0 | 56.6 | 61.3 | 66.8 | 68.0 |
| R-s101 | 41.8 | 52.7 | 73.8 | 75.8 | 76.6 |
| R-s202 | 41.0 | 60.5 | 61.7 | 67.6 | 67.6 |
| U-s101 | 43.4 | 62.1 | 73.4 | 68.8 | 70.3 |
| U-s202 | 43.4 | 62.5 | 69.5 | 68.0 | 73.4 |
| cap (tokens) | 8,192 | 8,192 | 12,288 | 12,288 | 12,288 |

## pass@1 (%) on J_R (28 factoring probe problems), 8 attempts per problem

| Run | r0 | r32 | r64 | r96 | r128 |
|---|---|---|---|---|---|
| H-s101 | 80.8 | 82.1 | 83.9 | 84.4 | 89.3 |
| H-s202 | 80.8 | 82.6 | 86.2 | 92.0 | 91.1 |
| R-s101 | 81.7 | 83.9 | 91.5 | 92.9 | 91.1 |
| R-s202 | 80.8 | 87.9 | 82.1 | 87.9 | 90.2 |
| U-s101 | 81.7 | 83.0 | 87.5 | 91.5 | 92.4 |
| U-s202 | 81.7 | 82.6 | 87.9 | 90.6 | 92.0 |
| cap (tokens) | 8,192 | 8,192 | 12,288 | 12,288 | 12,288 |

## pass@1 (%) on J_U (32 modular-arithmetic problems), 8 attempts per problem

| Run | r0 | r32 | r64 | r96 | r128 |
|---|---|---|---|---|---|
| H-s101 | 80.5 | 83.6 | 88.3 | 88.3 | 93.8 |
| H-s202 | 80.5 | 87.9 | 91.0 | 88.7 | 90.2 |
| R-s101 | 81.2 | 87.1 | 95.7 | 96.1 | 96.1 |
| R-s202 | 80.5 | 86.3 | 90.6 | 87.9 | 91.8 |
| U-s101 | 81.6 | 89.1 | 92.2 | 96.9 | 94.1 |
| U-s202 | 81.6 | 87.9 | 94.5 | 96.1 | 94.1 |
| cap (tokens) | 8,192 | 8,192 | 12,288 | 12,288 | 12,288 |

## Cumulative hard training attempts after each round

| Run | r0 | r32 | r64 | r96 | r128 |
|---|---|---|---|---|---|
| H-s101 | 0 | 4,096 | 8,192 | 12,288 | 16,384 |
| H-s202 | 0 | 4,096 | 8,192 | 12,288 | 16,384 |
| R-s101 | 0 | 2,048 | 4,096 | 6,144 | 8,192 |
| R-s202 | 0 | 2,048 | 4,096 | 6,144 | 8,192 |
| U-s101 | 0 | 2,048 | 4,096 | 6,144 | 8,192 |
| U-s202 | 0 | 2,048 | 4,096 | 6,144 | 8,192 |

H trains on 128 hard attempts per round (16 slots × 8), R and U on 64 (8 shared slots × 8).

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
