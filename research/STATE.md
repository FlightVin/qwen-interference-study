# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-08 02:10Z (review 24; A11 pause, smoke passed).

## Phase
Main study running since 2026-11-07 19:07Z. Gates G1, G2, G3 passed; PROTOCOL frozen at `b8f36ca` (D57, D58).
**Scope (A9, D71): six primary runs — H, R, U × seeds 101, 202**; clip (0.2, 0.2); lr 2e-6; 128 rounds.
**Main matrix PAUSED (A11, D77).** New schedule: rounds 1–32 at 8,192, rounds 33–128 at the amended cap (16,384 unless
the smoke test/deadline force one intermediate cap, frozen before resuming). No round-32 state survived → all s101 runs
replay rounds 1–32 from round 0; old attempts (13.9 GPU-h) superseded. Truncation study + smoke test running.
Seed 303 and clipping sensitivity deferred. Deadline: report by 10 Nov 01:00Z (target), 14:30Z (hard).
No endpoint result exists yet (none viewed at the A9 amendment).

## Question
Does direct practice on polynomial factorization (E_related) help the learner recognize and use factoring in
harder problems (H_train → held-out MATH targets), compared with matched modular-arithmetic practice
(E_unrelated)? Primary contrast: seed-mean R − U endpoint pass@1 on target_test (sealed until endpoints).
Qwen3-1.7B @ `70d244cc`, non-thinking.

## Established
- Trainer and environment: custom single-GPU GRPO (D10, D11); tests 33/33 + 5/5 + 3/3 (jobs 11075445, 11078552, 11078115); precision audit passed (A1; `research/evidence/g2/precision_audit.md`).
- Data (G1, cap 8,192): H_train 46 (31 core / 15 moderate; native 25 + Omni-MATH 21; D47); 63 matched R/U pairs (A5 coverage met; D48); target_dev 53, target_test 128 (sealed), probes 32/28/32 (D34). Native funnel: 184 at 4,096 → 117 excluded by 4,096 score, 55 re-screened with 32 fresh attempts at 8,192, 12 annotation-ineligible (D67).
- Stage-3 baseline (D52): H mean pass 0.202, mixed groups 0.77; R 0.589 / U 0.569.
- Baseline error audit (§3.7, D69; 64 attempts, one per problem; observational): first error = missed factoring opportunity 30, error after correct factoring 20, incorrect execution 6, valid alternative 6, extraction 2 (none split across boxes).
- Calibration (G3): lr 2e-6 / clip (0.2, 0.2) selected by the frozen rule (mean gain 0.159 vs 0.106; D57). Panels 59/56 = all in-band items; deviation from the §3 "64 per split" reading recorded (D67). Eval noise sd ≈ 0.020 per within-run gain (D50, D51).
- Main runs healthy so far (D61, D70): rounds 13–26, mismatch p99 ≈ 0.11, memory ≤ 41.6 GiB; responses 1,743 → ~2,550 tokens.

## Owner amendments in force
A1 precision + report figures (§15); A2 aux pool rules (§16); A3 ≤ 30 CPUs (§17); A4 H_train authority (§18);
A5 technique coverage (§19); A7 ceilings **3 general + 6 preempt GPUs** (§21); A9 deadline + six-run scope (§23); A10 cap raise (§24) with A11 boundary at round 32 + pause (§25); A8 priorities (§22): cap fixed,
truncation by arm/role reported, 5% crossing → inspect only. A6 thinking-mode diagnostic (§20) **closed** by A8
(evidence D62, D63; D63 narrowed in D65). D21: `registry/` and `ops/` untracked.

## Uncertain
- Preempt starts are rare (D37); wall time depends mostly on 3 general GPUs.
- Response growth at lr 2e-6 over 128 rounds. Truncation so far: hard 0.2–2.3% per run; shared hard schedule R 1.0% vs U 2.3% (21 rounds).
- Two seeds give no real estimate of training variability; conclusions rest on per-seed agreement plus problem-resampling intervals conditional on the trained policies (A9.4).
- Storage: 195/500 GB used; ≤ 7 concurrent main runs (D70).

## Next actions
1. Truncation study on dev at 16,384 from R r52 / U r48 / H r36 (crossing 8,192, crossing-then-correct, extra GPU cost) + fixed-sample inspection (productive / repetition / extraction).
2. Smoke test (11079543, general). Freeze the exact cap; record it.
3. Move superseded a1 run/ckpt dirs to `superseded/`; regenerate recipes (cap schedule 0:8192,32:CAP, attempt a2, new W&B ids); resume all six (s101 on general, s202 queued; move as slots free).
4. Verify from logs that round 33 uses the new cap in rollout, training, grading. Curves mark the transition.
5. Forecast from measured 16k throughput and the replay; report by 10 Nov 01:00Z, hard deadline 14:30Z.
