# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-11 17:37Z (12:37 PM EST Wed; A19 ~16:03Z; A20 ~17:15Z: 8 general GPUs until 21:00Z). Timeline and GPU tables per A18 (`project.md` §32).

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
| 12:33 PM [17:33Z] | H-s202 round 116 saved (LATEST); round 117 at 17:46Z | done |
| 12:02 PM [17:02Z] | Tranche 1 done for all four: recovered 6–11 of 41–56 capped; R − U at 12,288 → 16,384: s101 +3.9 → +5.9, s202 −2.0 → −2.0 | done |
| 1:00 PM [18:00Z] | D150 fallback: copy LATEST (112 or 116) on CPU + 6 shards on preempt; move shards to general as length jobs end | planned |
| ≈ 1:15 PM [18:15Z] | Stress full panels done (all four) | forecast |
| ≈ 2:30 PM [19:30Z] | Sealed-test continuation done (four checkpoints, D155) | forecast |
| 4:00 PM [21:00Z] | A20 ends: general back to 3 | fixed |
| ≈ 3:00–4:15 PM [20:00–21:15Z] | Fallback test done | forecast |
| 1:00–6:00 PM [18:00–23:00Z] | Analyses (9.3d, 9.4), REPORT four questions (9.6), verification (8.8, 9.7) | planned |
| 3:15–4:50 PM [20:15–21:50Z] | H-s202 round 128; then dev eval ≈ 1 h; save; RUN_COMPLETE seconds later | forecast |
| 4:15–5:50 PM [21:15–22:50Z] | H-s202 RUN_COMPLETE → endpoint shards (pre-submitted `--after` when slots allow) | forecast |
| 5:15–6:50 PM [22:15–23:50Z] | H-s202 round-128 sealed test; then stress r128 | forecast; likely after deadline |
| 7:30 PM [00:30Z] | Deadline: final cost run, final commit, deliver | fixed |

## Work and GPUs (17:37Z; limits: general 8 until 21:00Z then 3 (A20), preempt 6, 30 CPUs)
| State | Pool | Job | Work |
|---|---|---|---|
| running | general 1 | 11112377 | `qis-main-H-s202-sym` training (round ≈ 116) |
| running | general 2 | 11118248 | `qis-len-U-s101` stress full panel |
| running | general 3 | 11118444 | `qis-len-R-s202` stress full panel |
| running | general 4 | 11118448 | `qis-len-R-s101` stress full panel (moved from preempt 17:30Z) |
| queued (Slurm priority) | general 5–8 | 11118658, 11118660, 11118661, 11118662 | `qis-lentest-{R,U}-s{202,101}`: sealed-test continuation (D155) |
| running | CPU | 11112647 | CPU standby (babel-n5-20) |
| done | — | 11118245 | `qis-len-U-s202` stress full panel (17:04Z) |
| blocked (by time) | — | — | fallback at 18:00Z (D150): copy (CPU) + 6 shards on preempt (general full); endpoint shards and stress r128 at RUN_COMPLETE |
Unused: preempt 6 of 6 until 18:00Z (no ready GPU work before the fallback).

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
- Would capped answers succeed with more room? Tranche 1 (64 × 4 per checkpoint): few do (13–22% of capped); s202 R − U
  unchanged at 16,384 (−2.0 → −2.0, change 0.0 [−3.1, +3.1]); s101 +3.9 → +5.9. Full panel running.
- When R and U diverged within a seed: no matched intermediate checkpoint pair survives; dev evals and logs only.
- H-s202 r128 endpoint and stress cell: likely after the deadline → reported as incomplete cells.
- Why U-s101 reran non-identically on the same GPU type (unknown; scores unaffected).

## Decisions in force
D146 (3 general GPUs carry A17), D149 (H-s202 test as 6 shards), D150 (fallback rule, 18:00Z), D151 (+ correction),
D154 (A19: gate kept; length diagnostic by exact continuation; sample fixed).

## Next actions
1. Full-panel length phases (U-s202 11118245, U-s101 11118248 running; R-s202 11118444, R-s101 11118448 queued); blind read of 12 preselected continuations (running).
2. 18:00Z: apply D150; submit fallback copy (or `-g`) and `qis-test-H-s202-fb-k0..k5` (8.1b).
3. Analyses: length diagnostic (9.3d, 9.3e); divergence timing and per-tier stress table (9.4).
4. REPORT: four questions (9.6); incomplete-cell list; verification (8.8, 9.7).
5. When slots allow: endpoint shards `--after qis-main-H-s202-sym`; then stress r128; final cost run; deliver by 00:30Z.
