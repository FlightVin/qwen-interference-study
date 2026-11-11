# Blind read of 12 preselected continuations (A19 §2, D154; CHECKLIST 9.3e) — 2026-11-11 17:07Z

Sample: per checkpoint, the 3 tranche-1 attempts capped at 12,288 with the smallest sha256("a19-inspect/<id>/<sample>")
(rule fixed in D154 before generation). Reader: one Sonnet subagent, arm-blind (random 4-letter codes, shuffled; key in
`inspect_key.json`), shown the problem, gold, last 3,000 characters before the cut, and the continuation.

| Code | Checkpoint | Problem | Sample | Class | Finish at 16,384 | Graded |
|---|---|---|---|---|---|---|
| VBED | R-s101 | math:precalculus/train/00145 | 3 | delayed_wrong | stop | 0 |
| XJCG | R-s101 | math:intermediate_algebra/test/00780 | 1 | repaired_error | length | 0 |
| YYDG | R-s101 | math:intermediate_algebra/train/00631 | 1 | loop | length | 0 |
| GBSC | R-s202 | omni:omni/01597 | 1 | loop | length | 0 |
| UKKR | R-s202 | omni:omni/03609 | 3 | loop | length | 0 |
| UYQV | R-s202 | omni:omni/02215 | 2 | delayed_wrong | stop | 0 |
| GGLX | U-s101 | omni:omni/01597 | 3 | other (productive scan, cut off) | length | 0 |
| JLUN | U-s101 | math:intermediate_algebra/train/00631 | 1 | loop | length | 0 |
| FWGH | U-s101 | math:precalculus/train/00502 | 1 | loop | length | 0 |
| ZWCM | U-s202 | omni:omni/02215 | 2 | delayed_wrong | stop | 0 |
| PVNM | U-s202 | math:intermediate_algebra/test/00780 | 1 | completed_sound | stop | 1 |
| ZUCB | U-s202 | math:intermediate_algebra/train/00631 | 1 | loop | length | 0 |

Counts: loop 6 (R 3, U 3); delayed_wrong 3 (R 2, U 1); repaired_error 1 (R); completed_sound 1 (U); other 1 (U).
Before the cut, the reader judged 9 of 12 not on track, 3 on track (PVNM, GBSC, GGLX).

Grader doubt (XJCG): the continuation wrote `\boxed{0}` (= gold) inside its reasoning but reached 16,384 tokens before the
final write-up. The frozen rule gives a capped answer reward 0 whatever its text says (`qis.verify`), so the 0 is the
rule, not an extraction error. It shows one more way extra room fails: a correct answer found, then not finished.

"Loop" here is the reader's judgement (re-deriving the same steps or systems without converging). D148's earlier
"unconverged searches, not loops" used exact repeated-text detection on the 12,288 prefixes; the two are compatible.
With 3 attempts per checkpoint, these counts describe the sample and cannot rank the arms.
