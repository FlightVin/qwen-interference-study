# Response audit (D144, D153; EXPLORATORY, arm-blind, two reads per response)

Counts are reads (2 per response). Failures = verifier WRONG at attempt 0.

| Policy | Failures (of 16) | missed_opportunity | factored_incorrectly | factored_then_failed_later | other_method | hit_cap | grader_issue | Fixable by more factoring practice: yes / no / unclear |
|---|---|---|---|---|---|---|---|---|
| base | 8 | 6 | 5 | 4 | 1 | 0 | 0 | 8 / 5 / 3 |
| H-s101-r128 | 5 | 4 | 2 | 1 | 0 | 0 | 3 | 4 / 3 / 1 |
| R-s101-r128 | 4 | 2 | 1 | 4 | 0 | 1 | 0 | 2 / 5 / 1 |
| U-s101-r128 | 4 | 6 | 0 | 0 | 0 | 2 | 0 | 5 / 1 / 2 |
| R-s202-r128 | 8 | 4 | 0 | 5 | 0 | 4 | 3 | 6 / 8 / 0 |
| U-s202-r128 | 6 | 4 | 1 | 3 | 0 | 4 | 0 | 4 / 5 / 3 |
| All | 35 | 26 | 9 | 17 | 1 | 11 | 6 | 29 / 27 / 10 |

Category agreement between the two reads: 88/96 responses; 29/35 failures. Fixability agreement on failures: 26/35.
Adjudication: `math:precalculus/train/00152` (gold `2,4`): responses that box 2 and 4 separately are extracted as `4` and marked wrong — a confirmed grader limitation. `math:algebra/test/00352`: `0, 1` lists the integers instead of their count (gold 2) — a wrong answer, not a grader issue.
