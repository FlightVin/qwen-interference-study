# PROTOCOL — frozen experimental design (draft until the freeze line is signed)

Purpose: the single reference for the main study. Canonical instruction: root `project.md` §1–§19.
Every entry cites the decision (`research/DECISIONS.md`) that fixed it. Items marked PENDING are
filled only from the calibration result and the cost rule; nothing else may change at freeze.

## 1. Question and estimand
Does easier practice on polynomial factorization (E_related) improve GRPO learning on harder
problems whose audited solutions use it, compared with matched modular-arithmetic practice
(E_unrelated)? Primary estimand: seed-mean R − U endpoint pass@1 on held-out target_test (128 MATH
test problems, equal weight per problem, 32 samples/problem). Secondary: R − H, U − H; dev curves;
training-target success; per-problem changes; pass@8 (`1 − C(32−c,8)/C(32,8)`).

## 2. Model, prompt, decoding (D14, D18→D45)
Qwen/Qwen3-1.7B @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, full-parameter, non-thinking template
(`enable_thinking=False`; fixture `research/evidence/g2/template_fixture.json`). User prefix:
"Solve the following problem. Show your working and put your final answer in \boxed{}." Prompt cap
1,024 tokens. **Response cap 8,192 tokens** (D45). T=1.0, top_p=1.0, top_k disabled (vLLM 0),
repetition penalty 1.0, vLLM native sampler. Same decoding for screening, training, evaluation.

## 3. Reward (§6, verifier `src/qis/verify.py`)
Math-Verify 0.9.0 on the last `\boxed{}` only; reward 1 verified correct, 0 otherwise; capped responses
score 0; verifier timeout 5 s, retry 15 s in a fresh process; unresolved = unknown → round held.

## 4. Algorithm (D10, D11, D24–D26; `src/qis/train/`)
Synchronous GRPO; 16 prompts × 8 responses per round; 4 optimizer steps of 4 complete groups (32
responses), micro-batch 1; A=(r−mean)/(pop_std+1e-6); objective min(qA, clip(q,1−ε_low,1+ε_high)A),
no dual clip; mean over valid tokens per response, then over the 32 responses; π_old = round-start
learner log probs (teacher forced) fixed for the round; AdamW β=(0.9,0.999), ε=1e-8, wd 0, constant lr,
grad-norm clip 1.0; no KL, no entropy bonus; FP32 weights and moments, BF16 compute and rollout
(precision audit `research/evidence/g2/precision_audit.md`). Mixed arms: each minibatch 2 hard + 2
auxiliary groups. Checkpoint every 4 rounds (background, validated).

## 5. Datasets (frozen; `data/main_v1_manifest.json`, bundle sha256 `0e255243…`)
| pool | n | construction | decision |
|---|---|---|---|
| H_train | 46 | 31 core (2–8/32) + 15 moderate (9–12/32) at 8k; native MATH 25 + Omni-MATH 21; F1 11, F2 7, F3 11, F4 10, F8 7 | D47 |
| E_related | 63 | direct factoring practice, band 12–24/32 at 8k; 9 natural + 54 authored | D48 |
| E_unrelated | 63 | direct modular arithmetic, band 12–24/32; matched 1:1 to E_related within provenance | D48 |
| target_dev | 53 | study_dev, rubric only (central, multi-step, L4–5) | D34 |
| target_test | 128 | MATH test, rubric only; sealed file sha256 `2ffe70dd…` | D34 |
| probes | J_H 32 (⊂ dev), J_R 28, J_U 32 | study_dev rubric panels, IDs frozen before any probe result | D34 |
Matching balance: |SMD| success 0.025, response length 0.19; A5 coverage met (D48).
Annotation: agent-annotated and audited (pass 1, blind pass 2, third read), not human-reviewed.

## 6. Optimizer calibration (§7; D46)
Panels: calib_train 59, calib_val 56 at 8k (bundle `a90a0d71…`). Six configs lr {5e-7, 1e-6, 2e-6} ×
clip {(0.2,0.2),(0.2,0.28)}, seed 11, 32 rounds; validation 16/problem at 0/16/32; reject non-finite,
broken verification, > 5% capped validation responses; rank by endpoint gain; top 2 repeat with seed
22; mean gain; ties ≤ 0.01 broken lr 1e-6 > 5e-7 > 2e-6, symmetric > asymmetric; require positive mean
gain and neither seed < baseline − 0.02. Evaluation noise: sd ≈ 0.020 per within-run gain; cross-job reproduction partial (D50, D51).
**Result (D54, D57):** seed 11 — lr2e-6/asym +0.196 rejected (validation truncation 7.14%); lr2e-6/sym +0.146;
lr1e-6/sym +0.122; lr1e-6/asym +0.066; lr5e-7/sym +0.031; lr5e-7/asym −0.003. Seed 22 — lr2e-6/sym +0.172,
lr1e-6/sym +0.090. Means 0.159 vs 0.106 (no tie). Selection computed by `src/qis/analysis/calib_select.py` in CPU job
11078053; record `research/evidence/calibration/selection.json`.
**Selected lr / clip: lr 2e-6, clip (0.2, 0.2).** Sensitivity clip: (0.2, 0.28).

## 7. Matrix and schedule
Primary: H, R, U × seeds 101, 202, 303 (selected clip). Sensitivity: R, U × 3 seeds (other clip).
Shuffled-cycle schedules (`src/qis/train/schedule.py`): R and U share the hard stream and the pair
stream; H = common 8 hard + 8 extra hard (equal per-item exposure over cycles, so source/difficulty
strata are represented in proportion, A4.6). All runs start from the original checkpoint.
**Endpoint T = 128** (measured-cost projection ≤ 21 GPU-h per run; `research/evidence/cost/main_cost_estimate.md`). Recipe wall 40 h.
Dev evaluation at 0, T/4, T/2, 3T/4, T (8/problem; dev, J_R, J_U). Test at baseline and endpoint
(32/problem), sealed until endpoint.

## 8. Local update diagnostic (§7, A1.2)
Source: R, seed 101, selected clip at rounds 0 and T/2 (retained). Branches from identical model +
optimizer snapshots: none, hard, related, unrelated; 4 prompts × 8 responses; minibatch seeds
4101–4104; evaluate a fixed 32-problem dev subset (J_H) plus J_R, J_U with 16 responses/problem and
paired seeds. Reported as finite changes after an AdamW/GRPO update.

## 9. Analysis (§9; to be implemented in `src/qis/analysis/` before endpoint data)
Per seed R−U, R−H, U−H on target_test pass@1; two-sided 95% paired t interval over 3 seeds;
crossed bootstrap over seeds and problems preserving pairing, 10,000 draws, seed 2027; pass@8; per-
problem changes; dev curves vs rounds, hard rollouts, tokens, GPU-h; POPE-style (J_R,J_H), (J_U,J_H)
trajectories; coverage table (A5); costs. No equivalence claim from a null pilot.

## 10. Limitations known at freeze
H_train 46 (< 64) with 21 Omni-MATH items (distribution shift to MATH targets; transfer reported
separately); auxiliary pools 86% authored (equal across arms); cross-job sampling nondeterminism;
two general GPUs (preempt mostly unavailable at design time); selected lr 2e-6 lengthens responses
(validation truncation 4.7% at 32 rounds, seed 11) — truncation per arm is monitored every round and reported;
the asymmetric-clip sensitivity arm exceeded 5% truncation in calibration and runs as authorized.
A6: a separate thinking-mode baseline diagnostic (D56) does not alter this protocol.

## Freeze
Frozen 2026-11-07 by the commit that adds this line (hash recorded in DECISIONS D57 follow-up and in every main run's config `code_sha`). Main runs: `qis-main-{H,R,U}-s{101,202,303}-sym`, `qis-sens-{R,U}-s{101,202,303}-asym`, lr 2e-6, 128 rounds.

## Amendment A8 (2026-11-07, after freeze; project.md §22)
- A6 thinking-mode diagnostic closed after the same-cap comparison (D62) and failure sample (D63); no longer-cap profile.
- The 8,192 cap and the completion rule (capped → reward 0) stay fixed for every main and sensitivity run. The 5%
  truncation threshold applied to calibration selection only. During training, capped counts with denominators are
  reported per arm separately for hard and auxiliary prompts (from the per-response training records), and R vs U are
  compared on their shared hard schedule; a crossing of 5% triggers inspection and reporting, never a cap change.
  Scheduled evaluations report truncation alongside pass@1.
- Baseline error audit (§3.7) on saved non-thinking screening attempts, with the categories of project.md §22.3;
  extraction misses (e.g., complete answers split across boxes) are reported separately and never regraded.

## Amendment A9 (2026-11-07 ~22:00Z, after freeze, before any endpoint result; project.md §23)
- **Scope:** six primary runs — H, R, U × seeds 101, 202; clip (0.2, 0.2); lr 2e-6; 128 rounds; cap 8,192; frozen
  grading. Seed 303 and the clip (0.2, 0.28) sensitivity runs are **deferred** (recipes moved to
  `slurm-agentic-engineering/exp-deferred/`; slot guard refuses them).
- **Endpoint results viewed at amendment time: none** (no endpoint reached; target_test unopened).
- **Primary report table:** one row per seed (101, 202): H, R, U target_test pass@1 (32 samples/problem, 128
  problems), paired R−U and paired R−H; then means. Direction and magnitude agreement across seeds is reported.
- **Uncertainty:** problem-resampling (paired bootstrap over the 128 problems, seed 2027) is reported per seed and for
  the mean, labelled **conditional on these trained policies**. The two-seed t interval (df = 1) is shown only with an
  explicit warning that it is not strong evidence about training variability; no claim rests on it. This replaces the
  three-seed t interval and crossed bootstrap of §9 for the current scope.
- **Local update diagnostic** (R-s101, rounds 0 and 64) reported separately. **Clipping sensitivity: deferred.**
- Deadline: complete audited report by 2026-11-10 01:00 UTC; hard deadline 2026-11-10 14:30 UTC.

## Amendment A11 (2026-11-08 ~00:10Z; project.md §25) — supersedes the A10 round-64 boundary
- Cap schedule: rounds 1–32 at 8,192; rounds 33–128 at the frozen amended cap (16,384 unless the smoke test or the
  deadline forces one intermediate cap, frozen before any arm resumes). Evaluation cap = cap of the last trained round
  (round-32 evaluation at 8,192; rounds 64, 96, 128 at the amended cap); test evaluations at the amended cap with an
  8,192 view from the same traces.
- No end-of-round-32 state exists for the seed-101 runs (rolling checkpoints kept only rounds 52/48/36); the nearest
  verified full state is round 0, so all three replay rounds 1–32 at 8,192 from the base model with a fresh optimizer.
  Replayed rounds are a new trajectory (sampling is not bitwise reproducible across jobs, D24, D50). Old attempts
  (a1) are superseded; their directories move to `superseded/` and their compute is charged.
- The report describes an 8k-to-amended-cap training schedule with questions screened at 8k.

### A11 cap frozen (D81, 2026-11-08 ~03:00Z)
**Amended cap = 12,288** (rounds 33–128, all six runs; engine context 13,312). Evaluation cap after round 32 and for
all sealed test evaluations (base and endpoints) = 12,288, with success within 8,192 reported from the same traces.
The study uses an **8k-to-12k training schedule**; its questions were screened at 8k.

## Amendment A15 (2026-11-10 01:45Z; project.md §29) — deadline and EXPLORATORY analyses
Deadline Wed 11 Nov 00:30Z (19:30 EST Tue). Primary estimand, matrix, decoding, evaluation and A9 analysis unchanged.
Added, all **exploratory** (rules frozen in D122 before any test outcome was read): per-problem reliability and difficulty
bands on the existing sealed-test attempts (pass@1, pass@8, paired R−U and R−H, base-split 16/16 bands); interference
trace on shared hard prompts by 16-round window (all-wrong/mixed/all-correct, finished vs capped, mastery vs signal loss,
previously-solvable problems) with a dev-panel comparison; the prespecified local-update diagnostic (base and R-s101 r64;
the optional U-s101 r64 extension is skipped: checkpoint not retained); an optional strategy-cue panel (24 items, cued vs
uncued, 1,920 completions), dropped first if time tightens. H-s202 cannot finish by the deadline (D122).
