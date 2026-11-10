# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-10 10:10Z (Review 80; A11 resume; cap 12,288).

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

## Next actions
1. All five planned complete runs done and validated (D132). Sealed outcomes read 09:45Z for five tests (D133): seed 101 R − U +0.042 [+0.021, +0.063], R − H +0.049; R-s202 0.748 (0.071 below R-s101). Seed 202 pair waits for test U-s202 (11100785, ends ≈ 13:40Z).
2. H-s202 partial running (A16; stop 19:00Z); appendix figure `a16_partial.png`; H/R shared prompt sets identical.
3. Diag r64 (11101908): last row ends ≈ 10:25Z; then cue 11104738 (≈ 2.5 h), then reference check 11105174 (D131).
4. After test U-s202: rerun report-final, final interference, cost table, independent recomputation + audit, finalise `reports/REPORT.md` (A13 table, three A15 questions, cue separate, A16 appendix, omissions). Deadline Wed 11 Nov 00:30Z (Tue 7:30 PM EST).
