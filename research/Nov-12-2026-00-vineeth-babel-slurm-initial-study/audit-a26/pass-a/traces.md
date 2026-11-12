# A26 pass A — traced cases (agent a-evals)

Purpose: follow seven representative records from raw input to the number in report v2, with the path and value at every step. All values were read on compute on 2026-11-11 (EST) by `tools/a-evals/traces.py` (output `$P/audit-a26/a-evals/traces.json`) and `tools/a-evals/scan_evals.py`. The tokenizer (transformers, `70d244cc`) was used only to decode stored token ids. `$P = /data/user_data/vineethb/qis`. Every step below is **verified** unless marked otherwise.

## (a) Ordinary correct sealed-test attempt — R-s101, `intermediate_algebra/test/00874`, attempt 3

| Step | Path | Value |
|---|---|---|
| Input problem | `$P/data/main-v1/target_test_sealed.json` (sha256 `2ffe70dd…`), pool `test` | "Find $q(x)$ if the graph of $\frac{4x-x^3}{q(x)}$ has a hole at $x=-2$, a vertical asymptote at $x=1$, no horizontal asymptote, and $q(3) = -30$." Gold `-3(x+2)(x-1)`. |
| Request seed | formula `int_seed(4242, "eval/test/intermediate_algebra/test/00874/test-final/n32")` | 1478459057 = record `seed` (same for all 32 attempts) |
| Record | `$P/test-eval/qis-test-R-s101-sym/records.jsonl.gz` | sample 3, n_tokens 2,345, finish `stop`, extracted `-3(x + 2)(x - 1)`, status `correct`, reward 1. Response ends "### Final Answer: $$\boxed{ -3(x + 2)(x - 1) }$$" (two identical boxes; the last is graded). |
| Problem count | same file, 32 records for this problem | 16 correct of 32 (sample 0 was wrong: extracted `(x + 2)(x - 1)(x - 6)`) |
| Summary | `…/qis-test-R-s101-sym/summary.json` `per_problem` | c 16, c_within_8192 15, n 32 |
| Aggregate | pass@1 = mean over 128 problems of c/32 | this problem adds 16/32/128 = 0.00390625; summary pass1 0.818603515625 (= records) |
| Report | §2 table, Q1, opening 1 | R-s101 81.9% |

## (b) Capped response and its length extension — R-s202 sealed test

Recovered example, `algebra/test/00348`, attempt 10 (gold `3`):

| Step | Path | Value |
|---|---|---|
| Original record | `$P/test-eval/qis-test-R-s202-sym/records.jsonl.gz` | seed 1807596766, n_tokens **12,288**, finish `length`, extracted null, status `truncated`, reward **0**. Text ends mid-derivation ("### Step 15: Correct approach using algebra …"). |
| Eligibility | `eval_extend.py:43-54` | finish `length` and n_tokens = 12,288 → extend |
| Extension row | `$P/length-test/R-s202-r128/all/rows.jsonl.gz` | n_tokens_1 12,288, finish_1 length, reward_1 0 (= original), extended true, new_tokens 1,068, n_tokens_2 13,356, finish_2 `stop`, extracted_2 `3`, status_2 correct, reward_2 **1** |
| Prefix | prompt = original prompt ids + the original 12,288 response ids (`eval_extend.py:138`) | decoded full response = original text + continuation; it closes with "$\boxed{3}$" |
| Summary | `…/length-test/R-s202-r128/all/summary.json` | capped_1 538, recovered 87, finished_wrong 104, still_capped 347, new_tokens 1,781,371, GPU L40S, seed label `extend-test-v1` |
| Report | §5.2 row R-s202 | 538 capped, 87 recovered (16.2%), 104 finished wrong, 347 still capped; pass@1 74.8% → 76.9% (0.747803 → 0.769043) |

Still-capped example, `algebra/test/00054`, attempt 8 (gold `69`): original 12,288 tokens, `length`, reward 0 → extended by 4,096 tokens to exactly 16,384, finish_2 `length`, reward_2 0 although the text contains `\boxed{60}` (`extracted_2` "60"): a capped response scores 0 by rule (`verify.py:142-144`), and 60 is also wrong.

Note on the report column "Extra tokens per recovered answer": 1,781,371 / 87 = 20,476 is all new tokens spent on the 538 extended attempts divided by the 87 recoveries. The 87 recovered attempts themselves used 1,767 new tokens each on average (153,746 / 87).

## (c) Multiple-box answer — stress `math:precalculus/train/00152` (gold `2,4`)

| Step | Path | Value |
|---|---|---|
| Input | `$P/data/stress/stress_v1.json` | gold `2,4` |
| Audited record | `$P/stress-eval/R-s202-r128/records.jsonl.gz`, sample 0 (audit set uses attempt 0) | finish `stop`, 8 `\boxed{}` in the text, the last four `2`, `4`, `2`, `4`; extracted `4` (last box only, `math_source.py:16-29`); status `wrong`, reward 0 |
| Same pattern | `$P/stress-eval/H-s101-r128/records.jsonl.gz`, sample 0 | 6 boxes, last four `2`, `4`, `2`, `4`; extracted `4`; reward 0 |
| Correct form | U-s101 r128 sample 0 | one box `2 \text{ and } 4` → correct |
| Audit | `research/evidence/a17/audit/audit_merged.json` | both reads of both responses: category `grader_issue`, fixable `na` (the 4 "n.a." reads) |
| Cell counts (lower bound; rule: finished, reward 0, last four boxes contain both `2` and `4`) | `traces.json` → `c_multibox` | H-s101 r128 4, R-s202 r128 3, R-s101 r128 2, U-s101 r128 2, H-s202 r128 2, R-s101 r64 5, H-s202 r64 1, U-s202 r128 0, base 0 (of 8 attempts each) |
| Report | §7.3 grader limitation | "at most 7 of 1,768 attempts in any stress cell; at most 3 between R and U" — consistent: R − U difference here is 0 (s101) and 3 (s202). The sealed-test count (9 attempts) comes from `twobox.py` and was not rerun (**claimed**). |

## (d) Resumed runs — which rows the analyses use

**H-s202** (`$P/runs/qis-main-H-s202-sym/metrics.jsonl`, 755 rows):

| Row | Event | Job / host / GPU / code | Meaning |
|---|---|---|---|
| 0 | start, resume_round 0 | 11082192, babel-w5-24, L40, 7c44fa5 | a2-orig |
| 360 | eval dev @64 = 0.6580 | (same job) | a2-orig round 64 (cap 12,288) |
| 365 | start, resume_round 60 | 11112377, babel-u5-28, L40S, 09f5080 | a2-r60 |
| 367 | resume, round 60 | — | rows of rounds > 60 logged before this row are superseded |
| 388 | eval dev @64 = 0.6368 | 11112377 | a2-r60 round 64 |
| 568 / 748 | eval dev @96 = 0.6604, @128 = 0.6698 | 11112377 | — |
| 754 | complete (21:45:07Z) | — | round 128 trained at 20:42:05Z (round row), r128 evaluations 21:28–21:41Z |

Round rows 61–64 appear twice; all other rounds once (128 distinct). The original rows are also in `metrics.pre_resume_a17.jsonl` (365 rows, max round 64; config `ckpt_every` 4, `retain_rounds` []). The original round-60…63 training records and round-64 eval records are kept in `records_a2orig_r60-64/` (with `SHA256SUMS`); `records/` holds the a2-r60 versions. `resume_audit_a17.txt` (2026-11-11 01:37Z) shows `LATEST.json` = round 60 and only `round_0064.tmp` from a2-orig, so a2-orig's round-64 weights were never completed. Production `a17_train.py:7-17` builds H-s202-r60 = rows ≤ 60 before the resume + all rows after; `curves.json` H-s202-r60 dev = 0.4033 / 0.5660 / 0.6368 / 0.6604 / 0.6698 at rounds 0/32/64/96/128 (equal to rows 2, 180, 388, 568, 748). Report §3.1 "63.7% / 66.0% / 67.0%" = a2-r60 (**verified**). The H-s202 r64 stress cell is a2-r60 (eval-map §7).

**U-s202** (852 rows): start row 0 (job 11094037, dd71b20, round 0); start row 423 + resume row 425 (11097811, 6724cce, round 64); start row 512 + resume row 514 (11100159, d90dd8c, round 64); complete row 851. Round rows 65–80 appear 3 times, 81 twice. The last-resume rule keeps rounds 1–64 from the first segment and rounds 65–128 from job 11100159. Dev rows used: @64 row 336 (0.6792, first segment), @96 row 677 (0.6792), @128 row 845 (0.7028): report §3.2 67.9 / 67.9 / 70.3 (**verified**). R-s202 (resume at 16, rounds 17–19 twice) and R-s101 (resume at 124; two dev@128 rows, 0.7099 superseded and 0.7288 canonical; report uses 72.9) follow the same rule.

## (e) All-correct group in training — R-s101 round 101, slot 0

| Step | Path | Value |
|---|---|---|
| Records | `$P/runs/qis-main-R-s101-sym/records/train_r0100.jsonl.gz` (0-based index 100 = round 101) | slot 0, role `hard_common`, `omni/02804`, rewards 1,1,1,1,1,1,1,1; adv 0.0 × 8; n_tokens 3,051–6,386, all `stop` |
| Advantage rule | `grpo.py:16-23` | all-equal group → advantage exactly 0 |
| Round row | `metrics.jsonl` round 101 | groups_all0 0, groups_all1 11, groups_mixed 5 = records (11 all-correct, 5 mixed); 88 attempts with adv 0 (= 11 × 8) |
| Gradient | `run.py:300-303`, `grpo.py:26-41` | each attempt's objective is min(q·0, clip(q)·0) = 0, so it adds no gradient; it still counts in the 1/32 divisor of its minibatch |
| Report | §4.4 "only mixed groups carry a GRPO gradient"; R-s101 shared-hard all-correct 128 of 256 in rounds 97–128 | matches `signal.json` (production transcription check, `numbers.json`) |

## (f) Momentum-only step — U-s202 opt_step 57 (first zero-gradient step)

| Step | Path | Value |
|---|---|---|
| Step row | `$P/runs/qis-main-U-s202-sym/metrics.jsonl` row 76 | opt_step 57, round index 14 (= round 15), minibatch 0, grad_norm_preclip **0.0**, loss 0.0, grad_clipped false, valid_tokens 69,351, responses 32 |
| Minibatch membership | re-implemented `grpo.minibatch_plan` with stream `minibatch/r14` of seed 202 on the roles in `train_r0014.jsonl.gz` | plan [[0,6,9,11],[1,7,12,13],[2,4,10,14],[3,5,8,15]]; minibatch 0 = slots 0, 6 (`hard_common`, 0/8 each) and 9, 11 (`aux` `syn-U2-008`, `syn-U3-017`, 8/8 each); 0 of 32 attempts have nonzero advantage |
| Neighbouring steps | rows 77–79, 81 | opt_steps 58–60 grad norm 0.607, 0.631, 0.494; opt_step 61 (round 16) 0.769 |
| Weight change | step rows have no update-norm field | **not logged**. AdamW (wd 0) still applies lr·m̂/(√v̂ + ε) from the stored moments of steps 1–56, so the weights move; the size of that move is **unknown** from the records. |
| All zero-gradient steps | same recomputation for every canonical step row with grad norm 0 | U-s202 65/65 and H-s202 20/20 have all 32 attempts at advantage 0 (H-s202 first at opt_step 290, round 73, minibatch 1: four 8/8 groups) |
| Report / DEFINITIONS | DEFINITIONS §3.D "first at round 15 (U-s202)", 65 and 20 zero steps | **verified** for U-s202 and H-s202; other runs not traced here |

## (g) Sharded endpoint — H-s202 r128, `intermediate_algebra/test/00758`

| Step | Path | Value |
|---|---|---|
| Split | `eval_merge --split 6`: shard k gets panel items `[k::6]` | panel index 3 → shard k3, its first item (`k3/panel_3.json`: 00758, 00703, algebra/00416, 00315, …) |
| Shard job | 11120590 @ b77f066, babel-w9-32, **RTX A6000** (`gpu.txt`, `gpu_name.txt`) | `ckpt_identity.txt` recorded = on-disk `25134562…`; provenance round 128, opt_step 512, a2 ok |
| Shard records | `$P/test-eval/qis-test-H-s202-sym-shards/k3/records.jsonl.gz` | 32 records, samples 0–31, one seed 201401505 (= formula for key `eval/test/…/00758/test-final/n32`), 32 correct, 0 truncated |
| Shard summary | `k3/summary.json` | c 32, c_within_8192 25, n 32 |
| Merge | `eval_merge` by shard k5 (job 11120595, `merge.log`) into `$P/test-eval/qis-test-H-s202-sym/` | the 32 merged lines are byte-identical (canonical JSON) to the shard lines; merged summary c 32 |
| Aggregate | merged pass@1 | adds 32/32/128 = 0.0078125; pass1 0.723876953125 |
| Report | §2, Q1 | H-s202 72.4% |
