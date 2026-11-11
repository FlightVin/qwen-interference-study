# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-11 15:58Z (10:58 AM EST Wed; after Review 104). Timeline and GPU tables per A18 (`project.md` §32).

## Phase
A17 completion (`project.md` §31). Deadline: verified `reports/REPORT.md` by **Thu 12 Nov 00:30Z = Wed 11 Nov 7:30 PM
EST** (NON-NEGOTIABLE). Scope (A9): H, R, U × seeds 101, 202; no new seeds, arms or domains. The original sealed test
(128 problems × 32 attempts) is primary; common-GPU reruns, stress panels and audits are exploratory.
Design frozen at `b8f36ca`: Qwen3-1.7B `70d244cc`, full-parameter GRPO, lr 2e-6, clip (0.2, 0.2), 128 rounds, cap 8,192
for rounds 1–32 and 12,288 after (D57, D81). Five runs complete; H-s202 is training (resumed from round 60, D143).

## Timeline (EST; UTC in brackets; from observed throughput)
| When | Event | Status |
|---|---|---|
| 10:40 AM [15:40Z] | H-s202 round 109; LATEST checkpoint round 108 | done |
| ≈ 11:35 AM–12:45 PM [16:35–17:45Z] | H-s202 round 112: checkpoint save + dev evaluation | expected |
| 1:00 PM [18:00Z] | D150 decision (fallback if projected RUN_COMPLETE > 5:15 PM). All forecasts meet it → submit fallback copy (CPU job, or `-g` variant if the CPU pool is blocked) + 6 shards on round 112 | planned |
| ≈ 3:00–4:15 PM [20:00–21:15Z] | Fallback test done (earlier if preempt GPUs start; 4:15 PM with 2 general GPUs only) | forecast |
| 1:00–5:00 PM [18:00–22:00Z] | Verification audit (CHECKLIST 8.8) and REPORT fill (8.7) on CPU work | planned |
| 5:05–5:45 PM [22:05–22:45Z] | H-s202 round 128 (×1.18–1.30 of H-s101 pace) | forecast |
| 6:05–6:45 PM [23:05–23:45Z] | H-s202 RUN_COMPLETE → submit 6 endpoint shards | forecast |
| 7:05–7:45 PM [00:05–00:45Z] | H-s202 round-128 sealed test done | forecast; likely after deadline |
| 7:30 PM [00:30Z] | Deadline: final cost run, final commit, deliver | fixed |

## GPUs in use (15:58Z; limits 3 general + 6 preempt GPUs, 30 CPUs)
| Pool | In use | Jobs |
|---|---|---|
| general | 1 / 3 | 11112377 `qis-main-H-s202-sym` (training) |
| preempt GPU | 0 / 6 | — |
| CPU (`preempt_cpu_qos`) | 0 running | standby 11112647 PENDING (preempted; queue 77 deep) |
Planned at 18:00Z: general 3/3 (H-s202 + fallback k0, k1); preempt 4/6 (fallback k2–k5) if they start.

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
- Response audit (D153): 29 of 70 failure reads judged fixable by more factoring practice. Two-box grader limitation
  affects ≤ 3/4,096 sealed-test attempts per cell (≤ 0.07 points).
- GRPO signal: practice pools saturate after round 32 (all-correct groups R 62–90%, U 74–92%).

## Uncertain
- Seed-level length drift against an arm effect (A9.4). Two seeds cannot separate them; no task within scope can.
- H-s202 r128 endpoint and stress cell: likely after the deadline → reported as incomplete cells.
- Why U-s101 reran non-identically on the same GPU type (unknown; scores unaffected).

## Decisions in force
D146 (3 general GPUs carry A17), D149 (H-s202 test as 6 shards), D150 (fallback rule, 18:00Z), D151 (+ correction).

## Next actions
1. 18:00Z: apply D150; submit `qis-h202-fallback-copy` (or `-g`) and `qis-test-H-s202-fb-k0..k5` (CHECKLIST 8.1b).
2. REPORT (8.7): incomplete-cell list; §8 run table; What we learned items 2 and 6; H-s202 cells as they land.
3. Verification (8.8): recompute headline tables from records; cost vs ledger; figures vs tables; clean-checkout render.
4. At RUN_COMPLETE: submit endpoint shards `qis-test-H-s202-sym-k0..k5`; final cost run; deliver by 00:30Z.
