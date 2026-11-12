# A26 pass A — executed code: revisions, training and evaluation verification, tests (agent a-code)

Purpose: establish which code each job ran and whether the training and evaluation code does what `DEFINITIONS v1` and
`PROTOCOL.md` say (A26 §2). Written 2026-11-11, ~11 PM EST. Job inventory: `jobs.md`. Labels: **verified** (primary
evidence checked), **claimed** (a document says so), **unknown**. Compute checks ran read-only on CPU standby 11122432;
their outputs are in `tools/a-code/out/`; outputs on compute are under `$P/audit-a26/a-code/`.

## 1. Main results
| Area | Result | Label |
|---|---|---|
| Training code across a2 jobs | Identical except `run.py:368` (counter copy, 5f4b4d1); no change alters weights or data | verified (git diff) |
| Schedule, request seeds, minibatch plan | An independent re-implementation reproduces every executed attempt's prompt, role and seed (98,304/98,304) and every optimizer step's token count (3,072/3,072) | verified (compute) |
| Grading, truncation, advantages | All 98,304 training attempts obey the rules; advantages equal the formula exactly; 0 unknown | verified (compute) |
| Superseded segments | Every canonical training and in-run evaluation file was written by the retained segment | verified (compute) |
| Evaluation outputs | 46 evaluate_ckpt outputs and 15 eval_extend outputs satisfy every code invariant checked | verified (compute) |
| Environment and code trees | venv-v1 unchanged during the study; 264/264 pinned worktrees intact; 265/265 job scripts kept | verified |
| Project tests | 54/54 pass at `31c0036` (origin/main) on CPU; no test covers the run loop, checkpoint metadata or evaluate/merge | verified |
| Defects | 1 latent code hazard (eval-file skip), 1 bookkeeping defect (counters, R-s202 only), 2 negligible unknowns, 3 wording errors, GPU-type mixing | see §5 |

## 2. Code revisions per family
Commits are the ledger SHAs (`tools/a-code/revisions.py` → `out/a-code-revisions.json`). Paths per family are listed there.

| Family | Commits used by jobs that ran | Relevant code changes between them | Can a change alter results? |
|---|---|---|---|
| Training (a2) | 9d6349a, 7c44fa5, 0afd6e8, dd71b20, 6724cce, d90dd8c, 09f5080 (5f4b4d1: no output) | `run.py:368` only (`counters` → `dict(counters)`, 0151edc/5f4b4d1) | No: bookkeeping only (D127). H-s202 rounds 61–128 are the only canonical rounds after the fix |
| Training (a1, superseded) | b8f36ca, efdafdd, 0a83a74 | → 9d6349a: cap schedule (`run.py`, `rollout.py`), LM-head recompute (`learner.py`, same arithmetic) | Superseded; not used |
| Calibration | a64f832 … (6) | → main: cap plumbing and LM-head recompute only (`git diff a64f832 9d6349a`) | No arithmetic change to the update |
| Sealed test | 779c54d (base), a96dc7d, 95ff406, 73597a9, f9b294b, 815a9a6, b77f066 | none in `evaluate_ckpt`, `rollout`, `prompts`, `verify`, `seeds`, `grpo`; `eval_merge` added (22aa5e2) before the sharded jobs; `run.py:368` | No |
| Fallback r116, cgpu | 7a8190f; bcba941 | none after 779c54d except as above | No |
| Stress | 577aa9f, bcdc620, e4deb79, d06c7a5, b77f066, bcf3e47 | `eval_merge` added between 577aa9f and bcdc620 (used only by H-s202 shards) | No |
| Length | c894f2b, a674984, 0416921 (stress); ef68bf8, 0416921 (test) | `eval_extend`: `--select all` added (ef68bf8); `select`/`plan`/`combine` unchanged | No |
| Cue | 6d4e33a | none | No |
| Diagnostic | 4b47386 (r64), 594d408 (r0), 042cebd (ref) | `diag_local.py` identical for r64 and r0; 042cebd adds `--reference-only` only | No |

Jobs that ran before a fix: every a2 training segment except H-s202 rounds 61–128 ran with the live-counter save (D127);
only R-s202 resumed from an affected checkpoint (§5 D5). Every sealed-test job ran after the vLLM weight read-back check
(637689b). The completed diag r64 ran after the OOM fix (4b47386).

## 3. Training code (code at the job's commit; line numbers identical at HEAD)
| Item | Definition / PROTOCOL | Code | Executed evidence | Status |
|---|---|---|---|---|
| Model, template | Qwen3-1.7B `70d244cc…`, non-thinking (§2) | `prompts.py:10-13` `enable_thinking=False`; recipe `MODEL` = snapshot `70d244cc…` | fixture `research/evidence/g2/template_fixture.json` | verified (code), claimed (fixture) |
| Decoding | T 1.0, top_p 1.0, top_k off, rep 1.0, cap, stop = EOS | `rollout.py:32-34`; no stop strings; vLLM `dtype=bfloat16`, `seed=0`, no prefix cache (`:23-25`) | every finished attempt ends in token 151645/151643 (92,521/92,521) | verified |
| Response cap | 8,192 rounds 1–32, 12,288 rounds 33–128 | `run.py:98-108,254`; engine context 1,024 + 12,288 (`:168-170`) | round rows: cap = schedule in 768/768; every capped attempt has exactly the cap (5,783/5,783) | verified |
| Reward | last `\boxed{}`; capped → 0; unknown holds round | `math_source.py:16-29`; `verify.py:128-148`; `run.py:266-271` | status/reward consistent 98,304/98,304; 0 unknown; 0 `round_held` events | verified |
| Advantage | (r − mean)/(pop std + 1e-6); equal groups 0 | `grpo.py:16-23`; `run.py:272-273` | stored `adv` = float32 recomputation, max abs diff 0.0 | verified |
| Group counts | all-0 / mixed / all-1 | `run.py:338-340` | records = round rows in 768/768 rounds | verified |
| π_old | round-start learner, teacher forced (D11) | `run.py:275-284` | code only | verified (code) |
| Objective, clipping | min(qA, clip(q, 0.8, 1.2)A), no dual clip | `grpo.py:26-41`; config eps 0.2/0.2 in all 6 runs | config rows | verified |
| Loss reduction | token mean per attempt, ÷ 32 per minibatch | `grpo.py:41`; `run.py:295-303` (`-obj/len(members)`, 32 members) | step `responses` = 32 | verified |
| Minibatches | 4 × 4 complete groups, 2 per role | `grpo.py:44-64` (roles in `sorted` order, `:55`); `run.py:288-289` | predicted plan reproduces step `valid_tokens` 512/512 in every run; insertion-order plan only 20–21/512 for R/U (H: orders coincide) → DEFINITIONS §4 item 4 confirmed | verified |
| Zero-gradient steps | gradient 0 ⇔ 4 non-mixed groups | `run.py:302,310-319` | predicted = observed 512/512 per run; counts 18, 66, 53, 20, 25, 65 (= DEFINITIONS §3.D) | verified |
| Optimizer | AdamW β (0.9, 0.999), ε 1e-8, wd 0, lr 2e-6 const, clip 1.0 | `learner.py:50-51,88-96`; `run.py:310` (`grad_clip` absent in config → 1.0) | config lr 2e-6 in all segments | verified |
| Precision | FP32 weights and moments; BF16 body; FP32 head; BF16 rollout | `learner.py:42-46,17-36,64`; `prompts.py:33` | code only | verified (code) |
| Masking | prompt excluded; no padding; EOS in loss; capped attempts in loss with reward 0 | `learner.py:63-70` (single sequence, slice `P-1:-1`) | records: EOS present; capped attempts carry adv | verified |
| Seeds | streams of training seed; request seed `gen/r{rnd}/p{slot}` | `seeds.py:14-29`; `schedule.py:16-56`; `run.py:248-252,289` | prompt/role/slot 98,304/98,304; request seeds 98,304/98,304 | verified |
| Checkpoints | every 4 rounds (16 for R-s202 from round 17 and U-s202, D105); host snapshot, then background write | `ckpt.py:139-192`; `run.py:366-371` | configs: `ckpt_every` 4 / 16 as stated | verified |
| Cap transition | round 33 = 0-based 32 | `run.py:98-108` | round row 33 has cap 12,288 in all runs | verified |
| Resume | from `LATEST.json`; rounds re-run; records overwritten; metrics append | `run.py:174-192,243,355` | canonical train files written within 900 s of their canonical row (768/768) | verified |
| Eval-file skip | — | `run.py:216-220`: skips an evaluation whose records file exists | mitigated by moving files (D103, D143); all 90 canonical eval files written by the retained segment (mtime 0–0.2 s before row) | verified; latent hazard (§5 D6) |
| Counters | — | `ckpt.py:145,171-172` serialise metadata at write time; `run.py:184` loads them | distinct `opt_step` labels: R-s202 524; other runs 512 | verified (§5 D5) |
| Superseded exclusion | rows after last resume; superseded records excluded | `report_data.py:39` last occurrence; `a17_train.py:161-195` resume rule; `interference.py:37`, `curves.py:29`, `validate_endpoint.py:68` glob top-level `records/` only, so `records/superseded-*/` are not read; `a17_train.py:255` reads `records_a2orig_r60-64` only for the a2-orig trajectory | code reading | verified (code); `train_summary.py` does not exclude (§5 D7) |

## 4. Evaluation code
**evaluate_ckpt.py** (779c54d = HEAD). One request per problem with n samples and seed
`int_seed(4242, "eval/{panel}/{id}/{label}/n{n}")` (`:68,70`). FP32 `model.pt` is cast to BF16 and pushed into vLLM (`:50-52`);
a read-back check compares sums of 4 non-fused tensors with tolerance 1e-3 relative (`:54-63`). Unknown verifier status counts
as incorrect but stays in the denominator (`:80,87`). Success within 8,192 = correct and `n_tokens ≤ 8192` (`:81`). No partial
resume: a preempted job regenerates everything (cgpu U-s101, U-s202, stress H-s101, R-s101 r64 were preempted, `sacct`).
Executed (`evals_check.py`, 46 outputs: 24 sealed-test incl. 12 shards and 4 cgpu reruns, 12 stress incl. 3 shards, 10 cue): every record seed
equals the key seed; (prompt_id, sample) unique; ids equal the panel (or shard panel); samples 0..n−1 complete; every capped
record has `n_tokens` = 12,288 and reward 0; every finished record ends in EOS; summary per-problem counts equal the records
(100%); `unknown` field equals the recount. Unknowns: 2 in `qis-test-base`, 0 in the other 45 (§5 D4).

**eval_merge.py** (22aa5e2 = HEAD). Shard s takes `items[s::K]` (`:23-28`); per-item seeds do not depend on the shard, and the
records confirm it. Merge asserts equal `ckpt`, `ckpt_model_sha256`, `panel`, `n`, `max_tokens`, `seed_label` (`:20,34-36`),
exact id coverage (`:43`), samples 0..n−1 per id, which rejects missing and duplicate attempts (`:51`), and agreement with
each shard's counts (`:56`). It does not check record seeds or the panel file hash. Merged outputs: H-s202 endpoint 4,096,
fallback 4,096, stress H-s202 r128 1,768 records, all invariants pass.

**eval_extend.py** (5e70344 → ef68bf8; only `--select all` added). Eligible = finish `length`; the code asserts `n_tokens` =
12,288 and reward 0 for each (`:43-54`), so a capped record of another length aborts the job rather than being skipped.
Continuation prompt = original prompt ids + original response ids (`:138`), so the prefix is preserved token for token;
up to 4,096 new tokens; context 1,024 + 16,384 (`:117`). Every selected original reward is regraded and must match (`:107-111`);
all 10 completed length jobs passed this assert (20,480 sealed-test and 8,840 stress records regraded). Score at 16,384 = verifier
on the full response; still capped → 0 (`:141-143`). Seed per attempt `extend/{panel}/{id}/{label}/s{j}` (`:137`): a new stream,
not the original sampler state. Executed (15 outputs): extended ⇔ finish_1 = length (100%); every extended row has
`n_tokens_1` = 12,288; `reward_1` equals the source record (100%); still-capped rows have 16,384 tokens and reward 0;
`--select all` covers all 4,096 source attempts once; tranche 1 + rest = 256 + 1,512 = 1,768 per stress policy.

## 5. Defects and deviations
| # | Finding | Severity | Evidence |
|---|---|---|---|
| D1 | GPU type changes inside two training trajectories: H-s202 L40 (rounds 1–60) → L40S (61–128); R-s202 L40 (1–16) → L40S (17–128). H-s101 trained on L40; R-s101, U-s101, U-s202 on L40S. Not in DEFINITIONS. Sampling is not bitwise reproducible anyway, so no systematic effect is expected, but arm contrasts are not hardware-matched. | interpretation (minor) | `start` events in each `metrics.jsonl`; `sacct` AllocTRES (`jobs.md` §3) |
| D2 | DEFINITIONS v1 §4 item 12: "Shards k2–k5 ran on any 48 GB GPU". Executed: k0–k2 L40S (65 problems), k3–k5 RTX A6000 (63). Report v2 line 245–246 is correct. | wording | `sacct` 11120634 (l40s), 11120590/1/5 (a6000) |
| D3 | Report v2 line 710: every stress cell "on L40S". H-s202 r128 stress shards ran on L40 (k0, k2) and RTX A6000 (k1). | wording | `sacct` 11120596, 11120598, 11121110 |
| D4 | Base sealed test: 2 verifier-unknown attempts (`intermediate_algebra/test/00030`, samples 8 and 29; timeouts on `(1+274i)^{2002}`-type answers), scored incorrect. All other evaluation outputs and all training rounds: 0. Resolves DEFINITIONS §5 item 4 for test, cgpu, stress and cue summaries. | numbers (≤ 2/4,096 = 0.05 points, base only) | `$P/test-eval/qis-test-base/records.jsonl.gz` |
| D5 | Stale counters (D127): the async saver serialises `counters` at write time (`ckpt.py:145,171`; fixed `run.py:368`). Only R-s202 resumed from an affected checkpoint: 524 opt_step labels, and its `gen_tokens`/`prompt_tokens`/`hard_responses` round-row counters are offset after round 16. All other runs 512. `a17_train.py:391-393` uses counters only when a train file is missing; no file is missing (128/128 per run), so no reported number uses them. | numbers only if counters are used (none found) | `out/a-code-runs/R-s202.json` |
| D6 | `run.py:216-220` skips any evaluation whose records file exists. After a resume past an evaluation round this pairs a stale evaluation with the replayed policy. The code was never fixed; both affected resumes were handled by moving the files first (D103, D143). Verified: every canonical eval file was written by the retained segment. | latent; no effect on current numbers | `out/a-code-runs/*.json` `evals` |
| D7 | `train_summary.py:19-20,27,45` averages all `step` rows (superseded ones included) and lists duplicate eval rows. It feeds `qis-monitor` and `qis-train-summary`, not `qis-report-final`. Any grad-norm or eval figure taken from it mixes trajectories for R-s101, R-s202, U-s202, H-s202. | wording / monitoring only | code; `exp/qis-report-final.sh:21-27` |
| D8 | Analyses run on standbys with `srun --overlap` (a17_test, a17_train, a17_stress, a19_*, stress-panel selection) are outside the ledger and pinning; their commit is recorded only in DECISIONS text (e.g. D143 "at 4b8b88e"; line 957 "a17_test at 434b439, run in worktree d3fa9d5"). | provenance | `research/DECISIONS.md:832,957` |
| D9 | Weight-load check in `evaluate_ckpt.py:54-63` and `eval_extend.py:125-133` covers 4 non-fused tensors by sum with 1e-3 relative tolerance; a wrong fused q/k/v or gate/up load, or a small corruption, would pass. Checkpoint file identity is checked by sha256 in every recipe precheck. | latent (low) | code |
| D10 | Diagnostic reference (D138): "two evaluations of the same model differ by 0.025–0.031 in J_H" compares runs on different GPU types (r0: L40 vs L40S; r64: RTX 6000 Ada vs L40S), so that noise floor includes a hardware change. r0 branches use a fresh AdamW (first step ≈ lr per coordinate), r64 the saved moments (documented in report v2 line 637). | interpretation | `sacct` 11102499, 11101908, 11105174; D138 |
| D11 | 2 recipes were edited after their last run (`qis-data-v1`, `qis-probe-gpu-v1`: precheck predicate `"false"` → `"true"`). The as-run versions survive only in `~/ae-qis/jobs/` (45 versions in total exist only there). | provenance | `jobs.md` §5 |

No defect found that changes a training update, a grade or a sealed-test, stress, cue or length count.

## 6. Tests
Run: `env CUDA_VISIBLE_DEVICES= PYTHONPATH=<wt>/src $P/env/venv-v1/bin/python -m pytest -q -rA tests/` in worktree
`~/ae-qis/audit-a26/a-code/wt` (detached at origin/main `31c0036`), standby 11122432, 2026-11-11 ~9 PM EST. Result 54 passed,
0 failed, 26.6 s (`out/pytest-main.txt`; JUnit XML `$P/audit-a26/a-code/pytest-main.xml`).

| File | Passed | What it establishes |
|---|---|---|
| test_grpo.py | 14/14 | population-std advantage; zero for equal groups; 6 clip cases; per-response accumulation = full-batch token-mean/sequence-mean (matches verl); minibatch balance; pass@k; exact checkpoint checksums |
| test_verify.py | 8/8 | last box only; reasoning numbers ignored; missing box → no_answer; equivalent forms; capped correct → 0; timeout → unknown after retry |
| test_schedule.py | 5/5 | distinct, deterministic draws; R/U share streams; cycles cover pool; H extra balanced; calibration arm |
| test_analysis.py | 10/10 | contrasts and t interval; paired deterministic bootstrap; calibration tie rule; box split; report tables, per-problem bands, interference windows, cue DiD on synthetic data |
| test_diag_local.py | 4/4 | paired deterministic prompt draws; model + optimizer snapshot/restore; fresh-optimizer restore; seed-major resume |
| test_capsched.py | 3/3 | training and evaluation caps at the 0-based round-32 boundary; fallback without schedule |
| test_eval_extend.py | 3/3 | frozen tranche selection; plan boundaries (exactly 12,288 tokens); scoring at both caps |
| test_splits.py | 3/3 | nested `last_boxed`; stable, distinct seed streams; deterministic grouped splits |
| test_mode_diag.py | 3/3 | thinking-mode diagnostic grading (box inside reasoning earns nothing; incomplete → truncation) |
| test_lmhead.py | 1/1 | recomputed FP32 LM head equals stored (CPU) |

Not covered by any test: the `run.py` loop (resume, eval-file skip, record writing, minibatch role order), `ckpt.Saver`
metadata timing, `evaluate_ckpt`, `eval_merge.merge`, and anything that needs vLLM or a GPU.

## 7. Regression checks
Implemented as standalone scripts (numpy only, no project imports) and run on compute; all pass on current data:
- `tools/a-code/runs_check.py`: per run, schedule, request seeds, grading invariants, advantages, group counts, minibatch plan
  (zero-gradient and `valid_tokens`, with a non-discriminating alternative as control), file ownership after resumes,
  in-run eval seeds and counts. Outputs `out/a-code-runs/<run>.json`.
- `tools/a-code/evals_check.py`: evaluate_ckpt, merge and eval_extend invariants (§4). Output `out/a-code-evals.json`.
- `tools/a-code/env_check.sh`: venv mtimes and versions; pinned worktree HEAD and cleanliness.
Proposed (not implemented in `src/`):
1. Unit test: mutate `counters` after `Saver.save()`; `state.json` must hold the values at save time (catches D127).
2. Resume test with `QIS_EXIT_AFTER_ROUND` on a tiny model: after a resume past an evaluation round, assert no eval file
   predates the resume; better, key the skip on an `eval` row after the last `resume` event instead of file existence (D6).
3. `eval_merge`: assert each record's seed equals its key seed and add the panel sha256 to the `SAME` fields.
4. `evaluate_ckpt`: record the GPU name in `summary.json` (as `eval_extend` does) and check all tensors, including fused
   ones, or compare logits on fixed prompts against the HF BF16 model (D9).
5. `train_summary.py`: take the last occurrence per (round, minibatch) and per (panel, round) (D7).
6. Copy `~/ae-qis/jobs/` into the workspace and ledger the standby analyses or record their commit in the output (D8, D11).

## 8. What can and cannot be reproduced
**Can be reproduced exactly** (verified): which code, recipe and environment every job used (pinned worktrees, job scripts,
unchanged venv); the per-round prompts, roles, request seeds and minibatch plans of all six runs; the grade of every stored
response (regrade asserts passed on 29,320 records); every summary from its records; shard splits and merges.
**Cannot be reproduced token for token**: any vLLM generation. Outputs depend on batch composition and GPU kernels:
same-seed round-1 attempts were token-identical in 0–11 of 64 across arms, round-0 dev in 19–42 of 424 across jobs
(DEFINITIONS §3.A/§3.B, claimed), and the L40S rerun matched 1,964 of 4,096 U-s101 sequences (D151 correction, claimed).
Hence training trajectories, in-run evaluations, the sealed-test attempts and the length continuations reproduce only in
distribution. Training is further non-deterministic through GPU changes inside H-s202 and R-s202 (D1) and backward kernels.
**Cannot be re-run at all** (claimed in DEFINITIONS §3.B; checkpoints audited by another agent): in-run evaluations at
rounds 32 and 96 and any intermediate state other than R-s101 r64, H-s202 r64 (a2-r60) and the r116 fallback (rolling
deletion); the H-s202 a2-orig round-64 policy (weights never saved); training continued from R-s101 or U-s101 r128
(`optim.pt` deleted, D117). Standby analyses reproduce only if the commit recorded in DECISIONS is right (D8).
