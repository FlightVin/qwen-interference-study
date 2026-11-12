# Length extension 12,288 → 16,384 tokens

Every attempt that hit the 12,288-token cap was continued for up to 4,096 more tokens and regraded; other attempts keep their score. Change interval: paired problem bootstrap of the per-problem change. Shares in brackets are of the capped attempts. H-s202 was not extended.

Sources: `research/audit-a26/pass-b/data/eval_length.json`; `reports/tables/v3/perproblem_counts.json`.

## Sealed test (128 problems × 32 attempts)

| Checkpoint | pass@1 at 12,288 (%) | pass@1 at 16,384 (%) | Change (points) | 95% interval | Capped at 12,288 | Recovered (correct) | Finished wrong | Still capped at 16,384 |
|---|---|---|---|---|---|---|---|---|
| H-s101 r128 | 77.0 | 79.2 | +2.3 | [+1.5, +3.1] | 445 | 93 (20.9%) | 141 (31.7%) | 211 (47.4%) |
| R-s101 r128 | 81.9 | 84.2 | +2.3 | [+1.4, +3.3] | 437 | 94 (21.5%) | 50 (11.4%) | 293 (67.0%) |
| U-s101 r128 | 77.7 | 80.6 | +2.9 | [+1.9, +4.1] | 501 | 118 (23.6%) | 67 (13.4%) | 316 (63.1%) |
| R-s202 r128 | 74.8 | 76.9 | +2.1 | [+1.5, +2.8] | 538 | 87 (16.2%) | 104 (19.3%) | 347 (64.5%) |
| U-s202 r128 | 79.2 | 80.8 | +1.6 | [+0.9, +2.4] | 333 | 66 (19.8%) | 65 (19.5%) | 202 (60.7%) |

## Stress panels (221 problems × 8 attempts)

| Checkpoint | pass@1 at 12,288 (%) | pass@1 at 16,384 (%) | Change (points) | 95% interval | Capped at 12,288 | Recovered (correct) | Finished wrong | Still capped at 16,384 |
|---|---|---|---|---|---|---|---|---|
| H-s101 r128 | 64.1 | 67.3 | +3.1 | [+2.0, +4.4] | 313 | 55 (17.6%) | 126 (40.3%) | 132 (42.2%) |
| R-s101 r128 | 69.5 | 72.3 | +2.8 | [+1.8, +3.8] | 362 | 49 (13.5%) | 61 (16.9%) | 252 (69.6%) |
| U-s101 r128 | 65.3 | 68.2 | +2.9 | [+1.9, +4.0] | 406 | 51 (12.6%) | 64 (15.8%) | 291 (71.7%) |
| R-s202 r128 | 63.2 | 66.0 | +2.8 | [+1.9, +3.8] | 407 | 50 (12.3%) | 96 (23.6%) | 261 (64.1%) |
| U-s202 r128 | 65.8 | 68.2 | +2.3 | [+1.5, +3.3] | 268 | 41 (15.3%) | 81 (30.2%) | 146 (54.5%) |

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
