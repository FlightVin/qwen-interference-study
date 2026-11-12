# Sealed test: paired contrasts per training seed

Difference in sealed-test pass@1 between two round-128 endpoints of the same training seed. Interval: paired problem bootstrap over the 128 problems (same resampled problems for both policies). It covers problem and attempt sampling only, not training-run variation.

Sources: `research/audit-a26/pass-b/data/eval_test.json`.

| Contrast A − B | Seed | A pass@1 (%) | B pass@1 (%) | A − B (points) | 95% interval |
|---|---|---|---|---|---|
| R − U | 101 | 81.9 | 77.7 | +4.2 | [+2.0, +6.3] |
| R − U | 202 | 74.8 | 79.2 | −4.4 | [−7.3, −1.8] |
| R − H | 101 | 81.9 | 77.0 | +4.9 | [+2.1, +7.9] |
| R − H | 202 | 74.8 | 72.4 | +2.4 | [−0.5, +5.3] |
| U − H | 101 | 77.7 | 77.0 | +0.7 | [−1.9, +3.6] |
| U − H | 202 | 79.2 | 72.4 | +6.8 | [+3.6, +10.2] |
| R − U (L40S reruns) | 101 | 82.3 | 77.8 | +4.4 | [+2.4, +6.6] |
| R − U (L40S reruns) | 202 | 76.1 | 79.2 | −3.1 | [−6.0, −0.4] |

The L40S rows re-evaluate the R and U endpoints on one GPU type (reproducibility check).

## Two-seed summary (descriptive only)

| Contrast | Mean of the two seeds (points) | Half the difference between seeds (points) |
|---|---|---|
| R − U | −0.1 | 4.3 |
| R − H | +3.6 | 1.3 |
| U − H | +3.8 | 3.1 |

Two seeds cannot estimate the spread of effects across future runs.

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
