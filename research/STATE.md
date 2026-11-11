# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-11 16:13Z (12:13 PM EST Wed; A19 received ~16:03Z). Timeline and GPU tables per A18 (`project.md` §32).

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
| ≈ 11:35 AM–12:45 PM [16:35–17:45Z] | H-s202 round 112: checkpoint save + dev evaluation | expected |
| ≈ 12:00 PM [17:00Z] | Tranche 1 for U-s202, R-s101, U-s101 (≈ 10 min each once started) | forecast |
| 1:00 PM [18:00Z] | D150 fallback: copy round 112 (CPU job, or `-g` if the CPU pool is blocked) + 6 shards | planned |
| ≈ 2:00 PM [19:00Z] | Length diagnostic, all four checkpoints, tranche 1 + full panel | forecast |
| ≈ 3:00–4:15 PM [20:00–21:15Z] | Fallback test done | forecast |
| 1:00–6:00 PM [18:00–23:00Z] | Analyses (9.3d, 9.4), REPORT four questions (9.6), verification (8.8, 9.7) | planned |
| 4:30–5:30 PM [21:30–22:30Z] | H-s202 round 128; then dev eval ≈ 1 h; save; RUN_COMPLETE seconds later | forecast |
| 5:30–6:30 PM [22:30–23:30Z] | H-s202 RUN_COMPLETE → endpoint shards (pre-submitted `--after` when slots allow) | forecast |
| 6:30–7:30 PM [23:30–00:30Z] | H-s202 round-128 sealed test; then stress r128 | forecast; likely after deadline |
| 7:30 PM [00:30Z] | Deadline: final cost run, final commit, deliver | fixed |

## Work and GPUs (16:52Z; limits 3 general + 6 preempt GPUs, 30 CPUs)
| State | Pool | Job | Work |
|---|---|---|---|
| running | general 1 | 11112377 | `qis-main-H-s202-sym` training (round 113) |
| running | general 2 | 11118245 | `qis-len-U-s202`: t1 done; full panel since 16:32Z |
| running | general 3 | 11118247 | `qis-len-R-s101`: t1 generating |
| queued (Slurm priority) | preempt 1 | 11118248 | `qis-len-U-s101` → move to general when R-s101 t1 ends (equal coverage) |
| running | CPU | 11112647 | CPU standby (babel-n5-20) |
| blocked (by rule) | — | — | `qis-len-R-s202` full panel (t1 done): resubmit after all four t1 |
| blocked (by time) | — | — | fallback until 18:00Z (D150); endpoint shards and stress r128 until RUN_COMPLETE |
Unused: preempt 5 of 6 — no other ready GPU work (fallback waits for 18:00Z by D150).

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
- Would capped answers succeed with more room? Length diagnostic (D154) running: 12,288 → 16,384 continuation.
- When R and U diverged within a seed: no matched intermediate checkpoint pair survives; dev evals and logs only.
- H-s202 r128 endpoint and stress cell: likely after the deadline → reported as incomplete cells.
- Why U-s101 reran non-identically on the same GPU type (unknown; scores unaffected).

## Decisions in force
D146 (3 general GPUs carry A17), D149 (H-s202 test as 6 shards), D150 (fallback rule, 18:00Z), D151 (+ correction),
D154 (A19: gate kept; length diagnostic by exact continuation; sample fixed).

## Next actions
1. Tranche 1 for U-s202, R-s101, U-s101 (11118245, 11118247, 11118248) → then full-panel phases incl. R-s202 resubmit.
2. 18:00Z: apply D150; submit fallback copy (or `-g`) and `qis-test-H-s202-fb-k0..k5` (8.1b).
3. Analyses: length diagnostic (9.3d, 9.3e); divergence timing and per-tier stress table (9.4).
4. REPORT: four questions (9.6); incomplete-cell list; verification (8.8, 9.7).
5. When slots allow: endpoint shards `--after qis-main-H-s202-sym`; then stress r128; final cost run; deliver by 00:30Z.
