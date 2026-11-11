# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-10 20:10Z (Review 90; A11 resume; cap 12,288).

## Phase
Main study running since 2026-11-07 19:07Z. Gates G1, G2, G3 passed; PROTOCOL frozen at `b8f36ca` (D57, D58).
**Scope (A9, D71): six primary runs — H, R, U × seeds 101, 202**; clip (0.2, 0.2); lr 2e-6; 128 rounds.
**Schedule (A11, D81): 8k-to-12k** — rounds 1–32 at 8,192, rounds 33–128 at **12,288** (16,384 fit in memory but not the
deadline). All six runs restarted at round 0 as attempt a2 (D82); old s101 attempts superseded (13.9 GPU-h + 3.6 GPU-h of
cancelled evaluations). Questions screened at 8k; evaluations after round 32 and test at 12,288 with an 8,192 view.
Seed 303 and clipping sensitivity deferred. Deadline (A15, supersedes A12–A14): audited Markdown report by Tue 10 Nov 7:30 p.m. EST = Wed 11 Nov 00:30Z (hard); A15 analyses exploratory (D122); H-s202 cannot finish — partial run at lowest priority for an appendix (A16, D128); 01:00Z target (A9) will be missed.
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

- Round-32 dev (D99): canonical a2 values R 0.542, U 0.601, H 0.540; earlier 0.507/0.547/0.552 were the superseded a1 trajectory (A11 replay is not bitwise reproducible).

## Uncertain
- Preempt starts are rare (D37); wall time depends mostly on 3 general GPUs.
- Response growth at lr 2e-6 over 128 rounds. Truncation so far: hard 0.2–2.3% per run; shared hard schedule R 1.0% vs U 2.3% (21 rounds).
- Two seeds give no real estimate of training variability; conclusions rest on per-seed agreement plus problem-resampling intervals conditional on the trained policies (A9.4).
- Storage: 195/500 GB used; ≤ 7 concurrent main runs (D70).

## A17 (owner, 11 Nov ~01:25Z): complete the study by Thu 12 Nov 00:30Z (Wed 7:30 PM EST)
- Primary result unchanged (original sealed test): R − U s101 +0.042 [+0.021, +0.063]; s202 −0.044 [−0.072, −0.017]; mean −0.001.
- **Leading explanation of the reversal (D142, exploratory): truncation.** s202 R − U −0.044 = truncation −0.043 + accuracy-on-finished −0.001;
  s101 +0.042 = +0.014 + +0.028. R−U persists within 8,192 tokens. Reversal sits in B1 (−0.057 of −0.086); per-problem R−U uncorrelated across seeds.
- GRPO signal: aux pools saturate after round 32 (all-correct groups R 62–90%, U 74–92%); shared-hard signal similar across arms.
- Original test hardware confounded (R on L40, U on L40S, both seeds) → common-GPU reruns on L40S with original seeds (D141).
- Stress panels frozen (D143): SuperHard-128, Harder-98 (pool exhausted); audit sample fixed (D144); blind second read running.
- H-s202 resumed from round 60 (a2-r60); original r60–64 records backed up (D143).

## Next actions
1. Watch H-s202 (job 11112377; round 86 at 08:42Z; responses 6,000–6,800 tokens (H-s101 3,500–4,500) → RUN_COMPLETE ≈ 23:10Z–01:15Z, after the deadline; D150 fallback test of the latest checkpoint at ≈ 18:00Z); at RUN_COMPLETE submit the 6 test shards.
2. Common-GPU (D151): U-s202 L40S rerun identical to original (4,096/4,096 sequences). s202 R − U on L40S 0.761 − 0.792 = −0.031 [−0.060, −0.004]; hardware moved R by ≤ 1.3 points. U-s101 rerun (optional) running.
3. Stress evals (D152): s101 R − U +0.042 [+0.015, +0.068]; s202 −0.027 [−0.051, −0.002] = truncation −0.063 + accuracy +0.036. Done: base, R/U × s101/s202; R-s202-r128 running (general); H-s101, U-s202 r128, two r64 cells and the U-s101 rerun running on preempt since ≈ 07:10Z. H-s202-r128 after RUN_COMPLETE.
4. Response audit (D144) after stress evals; second-read agreement.
5. REPORT rewrite (What we learned; figures embedded; absolute values with deltas; dev n=8 fix); verification audit; deliver.
