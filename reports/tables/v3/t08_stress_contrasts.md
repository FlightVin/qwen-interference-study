# Stress panels: paired contrasts per seed and tier

Difference in stress pass@1 between round-128 endpoints of the same seed. Interval: paired problem bootstrap within the tier (audit pass B).

Sources: `research/audit-a26/pass-b/data/eval_stress.json`.

| Contrast A − B | Seed | Tier | A (%) | B (%) | A − B (points) | 95% interval |
|---|---|---|---|---|---|---|
| R − U | 101 | all | 69.5 | 65.3 | +4.2 | [+1.5, +6.8] |
| R − U | 101 | harder | 84.2 | 80.1 | +4.1 | [+0.5, +7.6] |
| R − U | 101 | superhard | 58.4 | 54.2 | +4.3 | [+0.5, +8.0] |
| R − U | 202 | all | 63.2 | 65.8 | −2.7 | [−5.0, −0.2] |
| R − U | 202 | harder | 76.3 | 79.5 | −3.2 | [−6.7, +0.3] |
| R − U | 202 | superhard | 53.3 | 55.6 | −2.3 | [−5.7, +1.1] |
| R − H | 101 | all | 69.5 | 64.1 | +5.4 | [+2.6, +8.1] |
| R − H | 101 | harder | 84.2 | 79.5 | +4.7 | [+1.2, +8.6] |
| R − H | 101 | superhard | 58.4 | 52.6 | +5.9 | [+1.8, +10.0] |
| R − H | 202 | all | 63.2 | 60.1 | +3.1 | [−0.1, +6.3] |
| R − H | 202 | harder | 76.3 | 74.3 | +2.0 | [−2.6, +6.7] |
| R − H | 202 | superhard | 53.3 | 49.4 | +3.9 | [−0.3, +8.5] |
| U − H | 101 | all | 65.3 | 64.1 | +1.2 | [−1.5, +3.8] |
| U − H | 101 | harder | 80.1 | 79.5 | +0.7 | [−2.8, +4.1] |
| U − H | 101 | superhard | 54.2 | 52.6 | +1.6 | [−2.3, +5.5] |
| U − H | 202 | all | 65.8 | 60.1 | +5.7 | [+2.4, +9.1] |
| U − H | 202 | harder | 79.5 | 74.3 | +5.1 | [+0.5, +9.7] |
| U − H | 202 | superhard | 55.6 | 49.4 | +6.2 | [+1.5, +11.1] |

Under the any-box grading rule (sensitivity check, discrepancies.md §3.2) the seed-202 R − U and R − H intervals on all problems change their relation to 0.

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
