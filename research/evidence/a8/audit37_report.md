# Audit 37: first-error labels for 64 incorrect baseline attempts

**Purpose.** Describe where Qwen3-1.7B (non-thinking) first goes wrong on 64 hard training candidates, one incorrect baseline attempt per problem, before any training. This audit is descriptive only. It does not select, drop or reclassify problems, and it does not change the grading.

- Input: `research/evidence/a8/audit37_sample.jsonl.gz` (64 records).
- Labels: `research/evidence/a8/audit37_labels.jsonl` (64 lines, one per record, same order as the input).
- Technique tags F1–F8 follow rubric v1.3 in `research/RUBRIC.md`.
- Reviewer: one agent annotator, 2026-11-07. Each record was judged from the problem statement and the full response.

## Labelling rules applied

1. The label records the first **uncorrected** error. An early slip that the response itself fixes is not counted. This rule affected `omni/02804`, `omni/01788` and `algebra/train/00288`.
2. A record is `other_error_after_correct_factoring` when the factoring is executed correctly but the answer fails for another reason. This includes three records where the other error comes **before** the factoring, so the factoring is applied correctly to a wrong expression (`omni/00524`, `intermediate_algebra/train/00340`, `omni/01012`).
3. A record is `missed_factoring_opportunity` when the response never performs the tagged factoring step. This includes five records where a setup or misreading error comes first and prevents the response from reaching the factoring step (`omni/03346`, `omni/01355`, `omni/02868`, `omni/03387`, `omni/01966`).
4. The 8 records from rules 2 and 3 are fits to the nearest category, not clean fits. Each has "borderline" in its `note`.
5. Solving a quadratic with the quadratic formula, or using LTE instead of repeated difference of squares, counts as a valid alternative route.

## Counts by category

Command:

```
jq -s -r 'group_by(.category) | map("\(.[0].category)\t\(length)") | .[]' research/evidence/a8/audit37_labels.jsonl
```

| category | records |
|---|---|
| `missed_factoring_opportunity` | 30 |
| `other_error_after_correct_factoring` | 20 |
| `incorrect_execution` | 6 |
| `valid_alternative_then_other_error` | 6 |
| `extraction_failure` | 2 |
| `insufficient_evidence` | 0 |
| **total** | **64** |

## Counts by primary_tag × category

Command:

```
jq -s -r '(map(.category)|unique) as $c | group_by(.primary_tag) | map([.[0].primary_tag] + [$c[] as $k | (map(select(.category==$k))|length)] + [length]) | (["tag"]+$c+["total"]), .[] | @tsv' research/evidence/a8/audit37_labels.jsonl
```

| tag | missed | other after correct | incorrect execution | valid alt. then other | extraction | total |
|---|---|---|---|---|---|---|
| F1 | 4 | 5 | 0 | 3 | 0 | 12 |
| F2 | 4 | 4 | 3 | 2 | 1 | 14 |
| F3 | 7 | 6 | 1 | 1 | 0 | 15 |
| F4 | 10 | 5 | 0 | 0 | 1 | 16 |
| F8 | 5 | 0 | 2 | 0 | 0 | 7 |
| **total** | **30** | **20** | **6** | **6** | **2** | **64** |

No record has tag F5, F6 or F7. Under rubric v1.3, F5 is not a primary tag.

## Verified extraction misses (reported separately; not re-scored)

The frozen grading rule reads only the last `\boxed{}`. These records stay graded as incorrect.

| measure | count | records |
|---|---|---|
| `category = extraction_failure` | 2 | `intermediate_algebra/train/00062`, `intermediate_algebra/train/00074` |
| `split_box_complete = true` | 2 | the same two records |

Neither miss is a true split across boxes. In `00062` the last box holds `x \geq 8`, which is the gold `[8,\infty)` in different notation. In `00074` the last box holds `(f(0), f(1)) = (-1, 1)`, which has a label in front of the gold pair. In the other 62 records, no combination of boxes forms the gold answer.

Commands:

```
jq -s '[.[]|select(.split_box_complete)]|length' research/evidence/a8/audit37_labels.jsonl
jq -s '[.[]|select(.category=="extraction_failure")]|map(.id)' research/evidence/a8/audit37_labels.jsonl
```

## Findings

1. The most common first error is a missed factoring opportunity: 30 of 64 records (47%). In these, the response tests values, guesses, cites an invented identity, or trial-divides large numbers instead of using the structure that factoring would expose.
2. F4 (factor theorem) has the highest share of missed opportunities, at 10 of 16 records. In these, the response rarely establishes a root and extracts the matching linear factor.
3. In 20 records (31%), the response does the factoring correctly and fails at a later step. The later failures are mostly a missed case, a sign error, a wrong telescoping step, or a misread condition.
4. Errors inside the factoring step are rare: 6 records (9%). Examples are a wrong SFFT form, a mis-evaluated sum-of-cubes factor, a dividing step that loses the root `x = 0`, and a correct Sophie Germain factorization that the response then rejects.
5. Only 2 of 64 failures (3%) have correct mathematics with an extraction miss, so last-box grading accounts for few of the failures in this sample.

## Limitations

- **One sample per problem.** Each label describes one attempt only. It does not estimate how often the model makes this error on that problem.
- **Agent reviewer, not human-reviewed.** An agent assigned all labels, and no human has checked them.
- **Single annotator.** No second annotator labelled the records, so inter-annotator agreement is unknown.
- **Nearest-category fits.** 8 records do not fit one category cleanly (see "Labelling rules applied"). Moving them would change the counts by up to 8.
- **No reference solutions.** The sample has gold answers but no reference solutions. The reviewer located each first error with its own working of the problem. Some gold answers were not fully re-derived (for example `omni/00792`, `omni/01343`, `omni/02130`). For those records, the label rests on a clear error in the visible text, not on a full solution.
