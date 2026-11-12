# A6 failure inspection (cap 8,192 tokens)

Purpose: decide, for 32 failed Qwen3-1.7B attempts on hard algebra items, whether the item's focal factoring operation (rubric v1.3 primary tag) is implicated in the failure. Per-record labels and evidence notes are in `failure_inspection.jsonl`.

Sample: 16 failures in non-thinking mode (`off`; all 16 status `wrong`) and 16 in thinking mode (`on`; 15 `truncated`, 1 `wrong`). Source: `failure_sample_cap8192.json`.

## Counts: mode x focal_implicated

Counted with `jq -r '"\(.mode)\t\(.focal_implicated)"' failure_inspection.jsonl | sort | uniq -c`.

| mode | yes | no | unclear | total |
|---|---|---|---|---|
| off | 11 | 2 | 3 | 16 |
| on | 3 | 11 | 2 | 16 |

## Counts: mode x locus

Counted with `jq -r '"\(.mode)\t\(.locus)"' failure_inspection.jsonl | sort | uniq -c`.

| locus | off | on |
|---|---|---|
| factoring_not_attempted | 9 | 3 |
| factoring_step_error | 2 | 0 |
| truncated_after_focal_step | 0 | 10 |
| truncated_before_focal_step | 0 | 0 |
| other_algebra_error | 2 | 1 |
| arithmetic_error | 1 | 0 |
| answer_form_or_incomplete_answer | 1 | 1 |
| misread_problem | 0 | 0 |
| other | 1 | 1 |
| total | 16 | 16 |

## Findings

In non-thinking mode, the focal factoring operation is implicated in most failures (11 of 16). The usual pattern is that the model never finds the factoring step. It tests values, solves a coefficient system, or guesses an answer. Only 2 of these 11 show a wrong factorization: a wrong SFFT set-up, and a telescoped product that lost a factor of 2. In thinking mode, the focal step is usually done correctly (11 of 16 "no"). The failure is then truncation: the thinking continues to verify, re-derive or loop after the key step until it reaches the 8,192-token cap. In several of these truncated records the thinking had already reached the gold answer. One thinking-mode "wrong" record had the correct answer (8 and 12) in two separate boxes, and only one box was extracted.

## Limitations

- Small sample: 16 failures per mode, one sample per item, and 15 of the 16 items appear in both modes (`algebra/train/01269` only in `off`, `intermediate_algebra/train/00301` only in `on`). The counts are descriptive, not estimates.
- Partial text: only the last 800 characters of the final answer and the last 1,500 characters of the thinking are visible. Earlier work, including a correct or wrong factorization, may be hidden. "factoring_not_attempted" means that no factoring step is visible and the visible route avoids it.
- Problem statements are not stored in the repository files searched (`pass1.jsonl`, `pass2_hard.jsonl`, `tags_v13.jsonl`, `main_v1_manifest.json`). Labels rely on the model text, the gold answer and the tag's reference step (`primary_quote`), so a misreading of the problem could go undetected.
- "unclear" also covers upstream errors where the focal step is never reached (for example, a wrong equation set-up), because neither "yes" nor "no" fits them.
- Labels are one agent's annotation. No human has reviewed them, and there is no second annotator.
