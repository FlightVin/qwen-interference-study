# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-11 19:13Z (2:13 PM EST Wed; A19 ~16:03Z; A20–A22: 7 general GPUs until the final report). Timeline and GPU tables per A18 (`project.md` §32).

## Phase
A17 completion plus A19 (`project.md` §31, §33; D154). Deadline: verified `reports/REPORT.md` by **Thu 12 Nov 00:30Z =
Wed 11 Nov 7:30 PM EST** (NON-NEGOTIABLE). Scope (A9): H, R, U × seeds 101, 202; no new seeds, arms or domains. The
original sealed test (128 problems × 32 attempts) is primary; reruns, stress panels, audits and the A19 length diagnostic
are exploratory. Keep-first order if time tightens: H-s202 sealed test > four-checkpoint length diagnostic > verification.
Design frozen at `b8f36ca`: Qwen3-1.7B `70d244cc`, full-parameter GRPO, lr 2e-6, clip (0.2, 0.2), 128 rounds, cap 8,192
for rounds 1–32 and 12,288 after (D57, D81). Five runs complete; H-s202 is training (resumed from round 60, D143).

## Timeline (EST; UTC in brackets; from observed throughput)
| When | Event | Status |
|---|---|---|
| 10:40 AM [15:40Z] | H-s202 round 109; LATEST checkpoint round 108 | done |
| 11:03 AM [16:03Z] | A19 received; D154 design fixed; length canary 11118157 started 16:12Z | done |
| 11:22 AM [16:22Z] | Canary tranche 1: 56 of 256 capped, 9 recovered, 13 finished wrong, 34 still capped; re-grade exact | done |
| 11:32 AM [16:32Z] | U-s202 tranche 1: 41 capped, 9 recovered; s202 R − U on t1 −2.0 points at both caps | done |
| 11:34 AM [16:34Z] | H-s202 round 112 saved (LATEST); round 113 at 16:48Z | done |
| 12:02 PM [17:02Z] | Tranche 1 done for all four: recovered 6–11 of 41–56 capped; R − U at 12,288 → 16,384: s101 +3.9 → +5.9, s202 −2.0 → −2.0 | done |
| 12:33 PM [17:33Z] | H-s202 round 116 saved (LATEST); round 117 at 17:46Z | done |
| 1:00 PM [18:00Z] | Fallback submitted: copy 11118844 done 18:02Z (round 116, hash ok); shards 11118845–51 | done |
| 1:20 PM [18:20Z] | Stress full panels done (all four); R − U unchanged at 16,384 | done |
| 1:31 PM [18:31Z] | H-s202 round 120 saved (LATEST); round 121 at 18:46Z | done |
| 1:24–1:47 PM [18:24–18:47Z] | Sealed-test continuation U-s202 (79.2 → 80.8%), U-s101 (77.7 → 80.6%) | done |
| 2:03 PM [19:03Z] | Endpoint shards pre-submitted `--after` training (11119210–11119217) | done |
| 2:30 PM [19:30Z] | Sealed-test continuation done (all four R/U): R − U unchanged within 1 point | done |
| 2:34 PM [19:34Z] | H-s202 round 124 saved (LATEST) | done |
| 2:54 PM [19:54Z] | Fallback merged: H-s202 round 116 (incomplete run) sealed test 74.3% pass@1, 14.8% capped | done |
| ≈ 2:30 PM [19:30Z] | Fallback shards done; merge → "H-s202 round 116" | forecast |
| 1:00–6:00 PM [18:00–23:00Z] | Analyses (9.3d, 9.4), REPORT four questions (9.6), verification (8.8, 9.7) | planned |
| ≈ 3:40 PM [20:40Z] (20:37–20:56Z) | H-s202 round 128; then dev eval ≈ 1 h; save; RUN_COMPLETE seconds later | forecast |
| ≈ 4:40 PM [21:40Z] (21:37–21:56Z) | H-s202 RUN_COMPLETE → endpoint shards (pre-submitted `--after` when slots allow) | forecast |
| ≈ 5:30 PM [22:30Z] (22:27–22:46Z) | H-s202 round-128 sealed test (shards in parallel) | forecast; before the deadline |
| ≈ 6:20 PM [23:20Z] | H-s202 stress r128 (3 shards, D156) | forecast |
| 7:30 PM [00:30Z] | Deadline: final cost run, final commit, deliver | fixed |

## Work and GPUs (19:52Z; limits: general 7 until the final report (A22), preempt 6, 30 CPUs)
| State | Pool | Job | Work |
|---|---|---|---|
| running | general 1 | 11112377 | `qis-main-H-s202-sym` training (round 125) |
| running | general 3–4 | 11119262, 11119264 | `qis-lentest-H-s101`, `qis-len-H-s101` (D157) |
| queued (dependency: training ends) | general | 11119210, 11119212, 11119213, 11119215, 11119216 | endpoint shards k0–k4 |
| queued (dependency; preempt for A22) | preempt | 11119217 | endpoint shard k5 → general when a general job ends |
| running | CPU | 11112647 | CPU standby |
| blocked (by D156 order) | — | — | stress shards `qis-stress-H-s202-r128-k0..k2`: submit when endpoint shards start |
Unused: general 3 of 7 until RUN_COMPLETE (no other ready work); peak at RUN_COMPLETE 7.

## Question
Does practice on polynomial factorization (R) help a model solve harder problems more than matched modular-arithmetic
practice (U) or more hard practice (H)? Primary contrast: seed-wise R − U pass@1 on the sealed test.

## Established (sealed test pass@1; 95% paired problem bootstrap)
- Original: base 0.526; s101 H 0.770, R 0.819, U 0.777; s202 R 0.748, U 0.792, H pending.
  R − U: s101 +0.042 [+0.021, +0.063]; s202 −0.044 [−0.072, −0.017]; mean −0.001. The sign reverses between seeds.
- Common GPU (L40S, D151): s101 R − U 0.823 − 0.778 = +0.044 [+0.024, +0.065]; s202 −0.031 [−0.060, −0.004]. Hardware
  does not explain the reversal. Reruns moved pass@1 ≤ 1.3 points; same GPU type is not always bit-identical.
- Truncation (D142): s202 R − U −0.044 = truncation term −0.043 + accuracy-on-finished −0.001; s101 +0.042 = +0.014 + +0.028.
- Stress panels (D152; 221 problems × 8): s101 R − U +0.042 [+0.015, +0.068]; s202 −0.027 [−0.051, −0.002], all from
  truncation (−0.063; accuracy +0.036). Arm order follows the seed, as on the sealed test.
- Response audit (D153): 35 failed responses, 70 reads: fixable yes 29 / no 27 / unclear 10 / n.a. 4 (2 grader errors).
  Per response both readers said fixable for 13 of 33. Two-box grader limitation ≤ 0.07 points on the sealed test.
- GRPO signal: practice pools saturate after round 32 (all-correct groups R 62–90%, U 74–92%).

## Uncertain
- Seed-level length drift against an arm effect (A9.4). Two seeds cannot separate them; no task within scope can.
- Would capped answers succeed with more room? Few do: 16–24% of capped sealed-test answers, 12–15% on stress. R − U
  at 12,288 → 16,384: sealed test s101 +4.2 → +3.6, s202 −4.4 → −3.9; stress +4.2 → +4.1, −2.7 → −2.1 (D154, D155).
- When R and U diverged within a seed: no matched intermediate checkpoint pair survives; dev evals and logs only.
- H-s202 r128 endpoint and stress cell: likely after the deadline → reported as incomplete cells.
- Why U-s101 reran non-identically on the same GPU type (unknown; scores unaffected).

## Decisions in force
D146 (3 general GPUs carry A17), D149 (H-s202 test as 6 shards), D150 (fallback rule, 18:00Z), D151 (+ correction),
D154 (A19: gate kept; length diagnostic by exact continuation; sample fixed).

## Next actions
1. Sealed-test continuation (4 jobs) → fill Q2 sealed-test rows; fallback shards → "H-s202 round 116" row.
2. 18:00Z: apply D150; submit fallback copy (or `-g`) and `qis-test-H-s202-fb-k0..k5` (8.1b).
3. Analyses: length diagnostic (9.3d, 9.3e); divergence timing and per-tier stress table (9.4).
4. REPORT: four questions (9.6); incomplete-cell list; verification (8.8, 9.7).
5. When slots allow: endpoint shards `--after qis-main-H-s202-sym`; then stress r128; final cost run; deliver by 00:30Z.
