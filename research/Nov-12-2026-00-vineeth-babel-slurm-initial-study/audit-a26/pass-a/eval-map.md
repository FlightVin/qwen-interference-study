# A26 pass A — evaluation map (agent a-evals)

Purpose: map every number cluster in report v2 (`reports/archive/REPORT-v2.md`) to the evaluation that produced it, and record the integrity checks of each evaluation's raw records. Machine-readable form: `data/eval_map.json`. Times are EST. Terms follow `research/DEFINITIONS.md` v1.

**Evidence base.** All record checks were made on 2026-11-11 (EST) on standby job 11122432 with `tools/a-evals/scan_evals.py` (stdlib + numpy, no project imports; reads `$P` only, writes `$P/audit-a26/a-evals/`). Job facts come from `slurm-agentic-engineering/state/LEDGER.jsonl`, `sacct -D` (allocation node lists) and `sinfo -N -o "%N %G"` (node → GPU type), and from each job's own `~/ae-qis/results/<expid>/<sha-expsha>/gpu.txt` and `ckpt_identity.txt`. **a-evals did not hash or load any checkpoint.** Hashes below are recorded values: `summary.json` `ckpt_model_sha256` (written by `evaluate_ckpt` from `state.json`), plus each job's start-time `ckpt_identity.txt` line "recorded=… on-disk=…" (equal in every job). They agree with `research/evidence/a17/manifest/manifest.json`.

## 1. Frozen inputs and code

| Input | Path | sha256 (re-hashed here) | Problems |
|---|---|---|---|
| Sealed test | `$P/data/main-v1/target_test_sealed.json` | `2ffe70dd…ec29ca` | 128 |
| Stress panel | `$P/data/stress/stress_v1.json` | `757393be…76405d` | 221 (Harder 95, SuperHard 126) |
| Cue panels | `~/ae-qis/code/qis-cue-eval-6d4e33a/data/cue/cue_v1_{uncued,cued}.json` | `c3399a42…` / `1b3aa7bf…` (= repo copies) | 24 each |

| Code path | Revision used by the jobs | Check |
|---|---|---|
| `evaluate_ckpt.py`, `verify.py`, `prompts.py`, `math_source.py`, `train/rollout.py`, `seeds.py` | job commits 779c54d … bcf3e47 (table 2) | `git diff 779c54d <job sha>` over these files: empty for every evaluation job (only `eval_merge.py` is added later). |
| `eval_merge.py` | f9b294b, 815a9a6, b77f066, bcf3e47, 7a8190f | unchanged to HEAD (DEFINITIONS §1). |
| `eval_extend.py` | stress t1: 5e70344 / c894f2b; stress rest: a674984 / c894f2b / 0416921; test: ef68bf8 / 0416921 | 5e70344, c894f2b, a674984 differ from ef68bf8 only by adding `--select all` (no change to the paths they ran). |
| `diag_local.py` | r0 594d408 (job 11102499), r64 4b47386 (job 11101908), reference reruns 042cebd (11105174) | 594d408 adds 21 lines to 4b47386 (not audited here). |
| Seeds | per problem `int_seed(4242, "eval/<panel_key>/<id>/<label>/n<n>")`; one vLLM request with n samples | re-implemented in `scan_evals.py`; matched every record (table 3). |

## 2. Evaluation → checkpoint → jobs → hardware

`unk/trunc` = verifier-unknown / truncated attempts in the records. Job = ledger job ID @ commit, then each started allocation `node:GPU` (PREE = preempted and requeued). Records: `$P/<dir>/records.jsonl.gz`; summaries next to them.

| Evaluation (dir) | Checkpoint | model.pt sha256 (recorded) | n × problems | Attempts | unk/trunc | Jobs and nodes |
|---|---|---|---|---|---|---|
| test/base (`test-eval/qis-test-base`) | base `70d244cc` | — | 32 × 128 | 4,096 | **2**/2 | 11089196@779c54d0 q5-20:L40S |
| test/H-s101 | `$P/ckpt/qis-main-H-s101-sym/round_0128` | `052c8a62b300` | 32 × 128 | 4,096 | 0/445 | 11095486@95ff4063 q5-20:L40S |
| test/R-s101 | `…/qis-main-R-s101-sym/round_0128` | `aedcca6e6a44` | 32 × 128 | 4,096 | 0/437 | 11095106@a96dc7dc w5-24:L40 |
| test/R-s202 | `…/qis-main-R-s202-sym/round_0128` | `22fd8364b310` | 32 × 128 | 4,096 | 0/538 | 11100782@73597a93 w5-28:L40 |
| test/U-s101 | `…/qis-main-U-s101-sym/round_0128` | `92aef0d2a53a` | 32 × 128 | 4,096 | 0/501 | 11095118@a96dc7dc q5-20:L40S |
| test/U-s202 | `…/qis-main-U-s202-sym/round_0128` | `7158b0952ee3` | 32 × 128 | 4,096 | 0/333 | 11100785@73597a93 o9-24:L40S |
| **test/H-s202 r128** (6 shards → `qis-test-H-s202-sym`) | `…/qis-main-H-s202-sym/round_0128` (a2-r60) | `2513456224d4` | 32 × 128 | 4,096 | 0/632 | k0 11119210@f9b294b3 p9-20:L40S; k1 11119212@f9b294b3 o5-20:L40S; k2 11120634@815a9a6f u5-28:L40S; k3 11120590@b77f0662 w9-32:**A6000**; k4 11120591@b77f0662 n5-16:**A6000**; k5 11120595@b77f0662 s9-32:**A6000** |
| test/H-s202 r116 fallback (6 shards → `qis-test-H-s202-sym-fallback`) | `$P/ckpt/qis-main-H-s202-sym-fallback/ckpt` (round 116, a2-r60) | `7b9ff3a40080` | 32 × 128 | 4,096 | 0/605 | 11118845–11118851@7a8190f2 on q9-16, q9-16, q9-24, o5-32, t5-24, v9-28: all L40S |
| cgpu/R-s101 (`test-eval-cgpu/…`) | R-s101 r128 | `aedcca6e6a44` | 32 × 128 | 4,096 | 0/448 | 11112395@bcba9413 q5-16:L40S |
| cgpu/R-s202 | R-s202 r128 | `22fd8364b310` | 32 × 128 | 4,096 | 0/506 | 11112396@bcba9413 t5-24:L40S |
| cgpu/U-s101 | U-s101 r128 | `92aef0d2a53a` | 32 × 128 | 4,096 | 0/505 | 11112397@bcba9413 o9-32(PREE) → **p5-16** (PREE; records written 06:18:28, 9 s before this allocation ended) → m5-28 (9 s, done-predicate); all L40S |
| cgpu/U-s202 | U-s202 r128 | `7158b0952ee3` | 32 × 128 | 4,096 | 0/333 | 11112398@bcba9413 n9-32(PREE) → **p9-20** (PREE; records 04:01:35) → q9-20 (37 s); all L40S |
| stress/base | base | — | 8 × 221 | 1,768 | 0/1 | 11112932@577aa9f6 p9-28:L40S |
| stress/H-s101-r128 | H-s101 r128 | `052c8a62b300` | 8 × 221 | 1,768 | 0/313 | 11113928@bcdc620e n9-32(PREE) → **m5-20** (PREE; records 04:44:22) → o9-28 (26 s); all L40S |
| stress/R-s101-r128 | R-s101 r128 | `aedcca6e6a44` | 8 × 221 | 1,768 | 0/362 | 11112937@577aa9f6 n9-32:L40S |
| stress/U-s101-r128 | U-s101 r128 | `92aef0d2a53a` | 8 × 221 | 1,768 | 0/406 | 11112938@577aa9f6 q5-24:L40S |
| stress/R-s202-r128 | R-s202 r128 | `22fd8364b310` | 8 × 221 | 1,768 | 0/407 | 11112940@577aa9f6 u5-28:L40S |
| stress/U-s202-r128 | U-s202 r128 | `7158b0952ee3` | 8 × 221 | 1,768 | 0/268 | 11115052@e4deb797 u5-28:L40S |
| stress/R-s101-r64 | `…/qis-main-R-s101-sym/round_0064` | `6fe89540d80d` | 8 × 221 | 1,768 | 0/462 | 11115377@d06c7a5e o9-28, o9-32 (PREE) → **p9-16** (PREE; records 05:13:32) → p9-28 (22 s); all L40S |
| **stress/H-s202-r64** | `…/qis-main-H-s202-sym/round_0064` (**a2-r60**) | `37963facc1cc` | 8 × 221 | 1,768 | 0/267 | 11115378@d06c7a5e q5-24:L40S |
| **stress/H-s202-r128** (3 shards → `stress-eval/H-s202-r128`) | H-s202 r128 | `2513456224d4` | 8 × 221 | 1,768 | 0/451 | k0 11120596@b77f0662 w5-28:**L40**; k1 11120598@b77f0662 n5-16:**A6000**; k2 11121110@bcf3e475 w5-28:**L40** |
| cue (10 cells, `cue-eval/v1/<pol>-<form>`) | base; R/U × s101/s202 r128 (same hashes as above) | as above | 8 × 24 per cell | 192 per cell | 0 unknown; 0–42 truncated | 11104738@6d4e33a4 p9-20:L40S (one job, all cells) |

Cancelled ledger rows that never started (state CANCELLED, "None assigned"): 11079315, 11079548 (test-base); 11119213/5/6/7 (H-s202 test k2–k5, resubmitted); 11119925/6/7 (H-s202 stress k0–k2, resubmitted); 11112846/8/9/50 (first stress submissions); 11102440 (cue).

**GPU findings (verified).** Report v2 states H-s202 r128 sealed-test shards "k0–k2 L40S, k3–k5 RTX A6000" and stress shards "L40 × 2, A6000 × 1": both match sacct and each shard's `gpu.txt`. DEFINITIONS v1 §4 item 12 says "shards k2–k5 ran on any 48 GB GPU": the k2–k5 recipes allowed any 48 GB GPU (`gpu_name.txt` written per D159), but k2 landed on L40S, so the pure-A6000 shards are k3–k5. Original R endpoints ran on L40, U on L40S (D141 claim verified). All five length-test jobs, all five stress length jobs (including the two cancelled jobs 11118157 and 11118247 whose `t1` outputs are used) and every common-GPU allocation ran on L40S. Report §7.2 says every stress cell ran "on L40S"; the H-s202 r128 stress cell did not (L40 × 2, A6000 × 1), as its own table states.

## 3. Record integrity per evaluation (verified)

For all 31 evaluation directories (8 sealed test incl. fallback, 4 common-GPU, 9 stress, 10 cue), `scan_evals.py` found:

| Check | Result |
|---|---|
| Problem IDs in records = panel IDs | 31/31; no missing, no extra |
| Attempts per problem = n (32 test, 8 stress/cue); sample indices exactly 0…n−1 | 31/31; 0 problems with ≠ n attempts |
| One request seed per problem, equal to `int_seed(4242, "eval/<key>/<id>/<label>/n<n>")` | 31/31; 0 mismatches (key `test`, `stress` or `cue`; labels `test-final`, `stress-v1`, `cue-v1`) |
| `finish == "length"` ⇒ n_tokens = 12,288 and reward 0; no n_tokens > 12,288; len(ids) = n_tokens | 0 violations |
| `summary.json` per-problem c and c_within_8192, pass@1, pass@8, truncated, unknown, completions, tokens = values recomputed from records | 31/31 equal |
| Verifier-unknown attempts | **base sealed test: 2 of 4,096** (counted as incorrect, DEFINITIONS §4 item 14); every other cell 0. Stress base and R-s202 r128 each have 1 `no_answer` attempt (finished, no box; reward 0). |

Because the seed key does not contain the shard, the sharded H-s202 evaluations used exactly the seeds an unsharded run would use: e.g. `intermediate_algebra/test/00758` has seed 201401505 in shard k3 and in the formula.

## 4. Shards and merges (verified)

| Sharded evaluation | Shard panels | Partition | Merge |
|---|---|---|---|
| H-s202 r128 sealed test, 6 shards | each `k<s>/panel_<s>.json` = full panel items `[s::6]` (identical item dicts); every shard dir holds all six split files and each follows the rule | IDs disjoint, union = 128; 22/22/21/21/21/21 problems; 704/704/672/672/672/672 records; 0 duplicate (problem, sample) keys across shards | merged 4,096 lines = union of shard lines (sha256 of each canonical record line equal for all 4,096); merged summary per-problem c equals shard summaries; written by k5 at 17:58:23 after the last shard file (17:57:32) |
| H-s202 r116 fallback, 6 shards | same rule | same counts | merged = union, 4,096/4,096 lines identical |
| H-s202 r128 stress, 3 shards | `[s::3]`: 74/74/73 problems | disjoint, union = 221; 592/592/584 records | merged = union, 1,768/1,768 lines identical |

Shard pass@1 differs by shard (H-s202 r128: k0 0.764, k1 0.682, k2 0.781, k3 0.650, k4 0.771, k5 0.695); shards hold different problems, so these are not GPU effects.

## 5. Length extension 12,288 → 16,384 (verified; DEFINITIONS §5 item 6)

Rows: sealed test `$P/length-test/<ck>-r128/all/rows.jsonl.gz`; stress `$P/length-diag/<ck>-r128/{t1,rest}/rows.jsonl.gz`. Each row was joined to its original record on (prompt_id, sample) by `scan_evals.py`.

| Check (10 row sets: 5 checkpoints × 2 panels) | Result |
|---|---|
| Rows cover every original attempt exactly once (test 4,096; stress t1 256 + rest 1,512 = 1,768) | 10/10; 0 duplicates |
| n_tokens_1, finish_1, reward_1, status_1 equal the original record | 0 mismatches |
| Every extended row: n_tokens_1 = 12,288 and finish_1 = length | 0 violations |
| Every original capped attempt was extended | 0 capped attempts not extended, in every set |
| Original capped attempts with n_tokens ≠ 12,288 | 0 |
| n_tokens_2 = 12,288 + new_tokens; still-capped rows end at exactly 16,384 with reward 0 | 0 violations |
| Non-extended rows unchanged (n_tokens, finish, reward) | 0 violations |
| Regrade of every selected original attempt matched | **claimed by code, not by log**: `eval_extend.py:107-111` asserts equality for every selected attempt before generation and aborts otherwise; every job that produced a used `rows.jsonl.gz` wrote its `summary.json` after that assert (the jobs' `tests.txt` show the 3 `test_eval_extend` tests passed). No per-attempt regrade log exists. |

Not extended, and why: only finished attempts (finish = stop), by design (`plan()`); they keep their original score. No capped attempt was left out.

| Panel | Checkpoint | Capped at 12,288 | Recovered | Finished wrong | Still capped | pass@1 12,288 → 16,384 |
|---|---|---|---|---|---|---|
| Sealed test | R-s101 | 437 | 94 | 50 | 293 | 0.818604 → 0.841553 |
| Sealed test | U-s101 | 501 | 118 | 67 | 316 | 0.776855 → 0.805664 |
| Sealed test | R-s202 | 538 | 87 | 104 | 347 | 0.747803 → 0.769043 |
| Sealed test | U-s202 | 333 | 66 | 65 | 202 | 0.792236 → 0.808350 |
| Sealed test | H-s101 | 445 | 93 | 141 | 211 | 0.769531 → 0.792236 |
| Stress (221) | R-s101 | 362 | 49 | 61 | 252 | 0.695136 → 0.722851 |
| Stress (221) | U-s101 | 406 | 51 | 64 | 291 | 0.653281 → 0.682127 |
| Stress (221) | R-s202 | 407 | 50 | 96 | 261 | 0.631787 → 0.660068 |
| Stress (221) | U-s202 | 268 | 41 | 81 | 146 | 0.658371 → 0.681561 |
| Stress (221) | H-s101 | 313 | 55 | 126 | 132 | 0.641403 → 0.672511 |

Per-tier and tranche-1 counts are in `data/numbers.json` (`claims` 5.3.*, 5.4.*). New tokens come from a new seed stream `extend/<key>/<id>/<label>/s<sample>`, so the continuation is not the path a 16,384-cap run would have sampled (DEFINITIONS §3.C).

## 6. Report clusters → evaluations

| Report v2 cluster | Evaluation(s) | Analysis script → production output |
|---|---|---|
| Opening 1–2, Q1 table, §2 primary table, base 52.6% / 83.5% | test/base, test/{H,R,U}-s{101,202} | `a17_test.py --cgpu-root` → `research/evidence/a17/analysis/test-cgpu/endpoints.json` (committed 934a1ea). Note: `…/analysis/test/` is an older run without H-s202 and the reruns; it is not the source of report v2. |
| §2 round-116 paragraph | test/H-s202 r116 fallback | summary.json; `manifest.json` |
| §2 common-GPU tables, opening 1 rerun values | cgpu/{R,U}-s{101,202} + test/H-s101, test/H-s202 | `test-cgpu/endpoints.json` (reproducibility block) |
| Opening 3, Q2 rows 1 and 3, §4.1–4.3 | test/{R,U}-s{101,202}, test/base (bands: base attempts 0–15; coverage baseline: attempts 16–31) | `test-cgpu/{reversal,coverage,truncation,loops}.json` |
| §4.4–4.6, opening 2 cost/learning signal, §3.2 training success and lengths | training records `$P/runs/<run>/records/train_r*.jsonl.gz`, `metrics.jsonl` (not evaluations) | `a17_train.py` → `…/analysis/train/{signal,persistence,dev_transitions,curves}.json`; `a19_divergence.py` → `research/evidence/a19/divergence/divergence.json` (fd94374) |
| §3.1–3.2 dev and probe values, Q3 dev values | in-run evaluations: `metrics.jsonl` `eval` rows (dev 53 × 8, J_R 28 × 8, J_U 32 × 8; J_H = first 32 dev problems), records `runs/<run>/records/eval_*_rNNNN.jsonl.gz` | `a17_train.py` → `curves.json`; checked against canonical rows by `traces.py` |
| Opening 5, Q3 stress table, §7.2 | stress/* (9 cells) | `a17_stress.py` → `research/evidence/a17/analysis/stress/stress.json` (1a9b49b) |
| §5.2, Q2 column "at 16,384", Q4 | length-test/{R,U}-s{101,202}, H-s101 | `a19_length.py` → `research/evidence/a19/length/test/length.json` (86bbe77) |
| §5.3, §5.4, Q2 stress rows | length-diag/* (t1 + rest) | `a19_length.py` → `…/length/full/length.json` (6360e1f), `…/length/t1/length.json` (d3dbd22) |
| §5.5 continuation reading | 12 extended attempts of tranche 1 (hash rule) | `…/length/t1/inspect*.json`, `inspect_read.md` (not re-read here) |
| §6 local diagnostic, opening 4 link 1 | `diag-local/{r0,r64}` (J_H 32, J_R 28, J_U 32 × 16 attempts, cap 8,192, label r7064); no-update reruns `diag_r*_ref.jsonl` | `research/evidence/a15/diag_r*.jsonl`; `…/analysis/train/{branches,delta_j_table}` |
| §7.3 response audit | stress attempt 0 of 16 items × 6 policies (base, H-s101, R/U × s101/s202 at r128) | `research/evidence/a17/audit/audit_merged.json` (4029ee2) |
| §7.3 grader limitation | stress records of `math:precalculus/train/00152`; sealed-test records via `twobox.py` | `research/evidence/a17/audit/twobox.py` (sealed-test count not rerun here) |
| §8 cue | cue/* (10 cells) | `cue.py` → `research/evidence/a15/cue_analysis.json` (65d53a6) |
| §9 cost | `reports/tables/sacct.txt` → `tables/cost.json` | not an evaluation; not checked by a-evals |
| Reproduction hash table | summary.json `ckpt_model_sha256` of every evaluation | recorded values agree with the manifest and with every job's `ckpt_identity.txt` (not re-hashed) |

## 7. Special cases

| Case | Finding | Label |
|---|---|---|
| H-s202 r128 endpoint | 6 shards, 3 GPU types per §2 above; seeds = unsharded seeds; merged = union; provenance check in every shard job `provenance round 128 opt_step 512 attempt a2 ok` | verified |
| H-s202 r116 fallback | `ROUND.txt` = 116 (`round.txt` "fallback round 116" in all six shard results); `state.json` round 116, a2, opt_step 464, model.pt `7b9ff3a4…`; all shards L40S | verified (state.json read; weights not hashed) |
| **H-s202 "r64" stress cell** | Trajectory **a2-r60**. The a2-orig job 11082192 was cancelled at 14:16 Nov 10 with only `round_0064.tmp` on disk (`resume_audit_a17.txt`: LATEST = round_0060, `round_0064.tmp` dated Nov 10 14:16). The resumed job 11112377 (a2-r60, started 20:51 Nov 10) saved `round_0064` at 2026-11-11T03:26:38Z (22:26 Nov 10 EST; `state.json saved_utc`, opt_step 256, model.pt `37963fac…`). The stress job 11115378 started 02:05 Nov 11, after that save, and its summary and `ckpt_identity.txt` both carry `37963fac…`. | verified |
| R-s101 r64 stress | `round_0064` hash `6fe89540…` in summary and job identity file | verified (hash recorded) |
| Common-GPU reruns | identical request seeds 4,096/4,096 in each pair; identical token sequences R-s101 48, R-s202 133, U-s101 1,964, U-s202 4,096 of 4,096; identical c per problem 62, 46, 79, 128 of 128 | verified |
| Cue study | same request seed for uncued and cued forms of a problem (key `eval/cue/<id>/cue-v1/n8`) | verified |
| Local diagnostic | record dirs `diag-local/{r0,r64}/records/{none, none_rerun, none_cycled, <branch>-<seed>}`: 3 files per branch (J_H, J_R, J_U), n = 16 per problem, max n_tokens ≤ 8,192 (`data/eval_map.json` → `diagnostic.record_checks`) | verified |
