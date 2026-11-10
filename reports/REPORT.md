# Does related practice help GRPO learn hard math? Qwen3-1.7B, H vs R vs U

DRAFT (10 Nov 12:40Z). All sealed-test values are final. Values marked `[PENDING …]` arrive with the reference check,
the cue panel, the H-s202 stop (19:00Z) and the final cost table. Every other number is final. Sections 3–5 are EXPLORATORY analyses added by owner
amendment A15 after the primary design was fixed; section 2 is the primary result.

## 1. Lead table (A13)

Sealed target_test, 128 MATH problems × 32 samples, pass@1 (each problem weighted equally). Intervals: 95% paired
bootstrap over the 128 problems; they describe these trained models, not run-to-run training variation.

| Seed | H | R | U | R − U | R − H |
|---|---|---|---|---|---|
| 101 | 0.770 | 0.819 | 0.777 | +0.042 [+0.021, +0.063] | +0.049 [+0.020, +0.080] |
| 202 | not run (partial, A16) | 0.748 | 0.792 | −0.044 [−0.072, −0.018] | not available |
| Mean | seed 101 only | 0.783 | 0.785 | −0.001 [−0.018, +0.015] | seed 101 only |

Base model: pass@1 0.526, pass@8 0.835.

The two seeds disagree. R beat U by 4.2 points in seed 101; U beat R by 4.4 points in seed 202. The mean is −0.001.
The R score differs by 0.071 between seeds, more than either R − U effect. Training variation between seeds is
therefore as large as the effect under test, and two seeds cannot separate them. Each interval excludes zero only
because it holds the trained models fixed.

## 2. Did the choice of auxiliary practice affect held-out success?

**Primary answer (R − U, paired, per seed).** Seed 101: R beat U by 4.2 points, and the interval excludes zero. Seed
202: U beat R by 4.4 points, and the interval excludes zero. The seeds disagree in sign, so this study does not show
that the choice of auxiliary practice changes held-out success. It also does not show that the choice has no effect:
an effect smaller than the seed-to-seed spread (about 4 points) cannot be detected with two seeds.

**pass@8 (exploratory, A15.1).** Seed 101: R − U +0.017 [−0.000, +0.036]; R − H +0.024 [+0.005, +0.046]. Seed 202:
R − U −0.026 [−0.057, +0.001]. In both seeds the R − U difference is smaller at pass@8 than at pass@1. The winning
arm mainly made already-reachable problems more reliable, not newly reachable.

**Where the gain sits (exploratory).** Problems are banded by the base model's successes in 16 of its 32 attempts; the
other 16 give the band's baseline, so band and baseline do not share noise.

| Band (base k of 16) | Problems | Baseline | H-s101 | R-s101 | U-s101 | R-s202 | U-s202 |
|---|---|---|---|---|---|---|---|
| B0 (0) | 15 | 0.021 | 0.210 | 0.306 | 0.298 | 0.181 | 0.283 |
| B1 (1–5) | 34 | 0.232 | 0.696 | 0.755 | 0.658 | 0.592 | 0.710 |
| B2 (6–10) | 26 | 0.555 | 0.833 | 0.881 | 0.851 | 0.855 | 0.883 |
| B3 (11–15) | 32 | 0.773 | 0.913 | 0.959 | 0.935 | 0.938 | 0.912 |
| B4 (16) | 21 | 0.979 | 0.991 | 0.997 | 0.979 | 0.982 | 0.994 |

In seed 101, R's lead over U comes from B1 (+0.097), B2 (+0.030) and B3 (+0.024); in B0 they are equal (+0.008). In seed 202, U's lead
comes from the hard bands: B0 (+0.102), B1 (+0.118) and B2 (+0.028); R leads only in B3 (+0.026). The band where the
winner gains most (B1) is the same in both seeds, but the winner is not. Problems improved / unchanged / worse versus
base: H 94/21/13, R-s101 102/20/6, U-s101 101/18/9, R-s202 94/26/8, U-s202 100/17/11.
Per-problem counts are in `tables/per_problem.json` (written as c/32; "0/32" for no success). Figure
`figures/per_problem_change.png`.

## 3. Did an unrelated update reduce hard success at a measured checkpoint?

Design: from one saved state (model and optimizer), apply ONE training step on 4 prompts × 8 responses of one kind —
hard, related (factoring) or unrelated (modular arithmetic) — then measure J_H (32 hard-dev problems), J_R and J_U at 16
samples per problem. Four minibatch seeds per branch. Sources: the base model (fresh optimizer) and R-s101 at round 64
(its saved optimizer). The U-s101 round-64 source was not retained, so that extension was skipped.

**Base model (r0), change versus the no-update row:**

| Branch | ΔJ_H mean | ΔJ_H per seed | ΔJ_R mean | ΔJ_U mean |
|---|---|---|---|---|
| hard | +0.014 | +0.010, +0.020, +0.006, +0.022 | −0.026 | −0.013 |
| related | +0.027 | +0.016, +0.039, +0.033, +0.020 | −0.025 | −0.022 |
| unrelated | +0.005 | +0.006, +0.025, +0.004, −0.016 | −0.027 | +0.001 |

All 12 update rows score 0.020–0.040 below the no-update row on J_R, including the related branch. A shift of that
size in every branch may come from the reference row, which is evaluated before the save–restore–reload cycle that
every update branch goes through. A reference check measures this (D131): [PENDING result]. Comparisons between
branches do not depend on it, because every branch shares the same reference.

**R-s101 round 64** (its saved optimizer state), ΔJ_H versus the no-update row, per minibatch seed:

| Branch | 4101 | 4102 | 4103 | 4104 | Mixed groups (of 4) |
|---|---|---|---|---|---|
| hard | −0.006 | +0.002 | +0.012 | −0.004 | 2, 3, 2, 1 |
| related | +0.006* | +0.006* | +0.016 | +0.014 | 0, 0, 2, 1 |
| unrelated | −0.006 | +0.027 | +0.006* | −0.014 | 1, 2, 0, 2 |

\* No mixed group, so the gradient is exactly 0. AdamW still steps from its stored moments, so these three rows are one
"step without signal". They give identical J_H, J_R and J_U, so evaluation is exactly repeatable for identical weights.
That momentum-only step changes J_H by +0.006, J_R by −0.007 and J_U by −0.008. By round 64, R-s101 solved most
practice prompts in all 8 answers, so half the related draws carried no learning signal.

**Answer.** [PENDING: state whether the unrelated branch lowered J_H at either source, against the reference check
and the per-seed spread.] One step of 32 responses is small; a change below ≈ 0.02 on 512 samples is within evaluation
noise.

## 4. Did previously solvable hard problems produce more all-wrong groups?

Source: training records on the 8 shared hard prompts per round (role `hard_common`; same prompts for H, R and U in
every round of a seed, verified 128/128 rounds). A group is the 8 answers to one prompt. "Previously solvable" means at
least one correct answer at an earlier appearance in the same run.

| Run | Later appearances | All wrong | of which ≥ 1 answer capped | After round 32 |
|---|---|---|---|---|
| H-s101 | 957 | 24 (2.5%) | 4 | 13 / 768 (1.7%) |
| R-s101 | 960 | 40 (4.2%) | 22 | 29 / 765 (3.8%) |
| U-s101 | 966 | 48 (5.0%) | 34 | 38 / 768 (4.9%) |
| R-s202 | 963 | 33 (3.4%) | 10 | 19 / 765 (2.5%) |
| U-s202 | 963 | 39 (4.0%) | 11 | 26 / 768 (3.4%) |

**Answer.** Yes, rarely, and more often with practice prompts than with hard-only training. U had more all-wrong
groups than R in both seeds (5.0% vs 4.2%; 4.0% vs 3.4%). In U-s101, 34 of its 48 such groups included a response
cut off at the length cap, so much of the loss is truncation, not wrong reasoning. Groups that left the "mixed" state
went mostly to all-correct (mastery), not all-wrong (signal loss): H 124 vs 14, R-s101 121 vs 22, U-s101 106 vs 29,
R-s202 94 vs 25, U-s202 122 vs 23. H trains on twice as many hard prompts per round, so it is not a dose-matched
comparison with R and U. Figure `figures/interference_windows.png`; the cap change after round 32 is marked.

**Dev panel comparison.** [PENDING: lost/gained counts between scheduled dev evaluations per run.]

**Which links the evidence supports.** [PENDING final paragraph: practice choice → test success (section 2); update →
local hard change (section 3); signal loss on solvable problems (section 4); with uncertainty for each.]

## 5. Strategy-cue panel (separate from the interference evidence)

24 unused MATH test problems where factoring is central, asked as written and with one fixed sentence appended
("Consider whether a factorization could simplify the problem."). 8 samples per problem per form, paired seeds.
This panel asks whether naming the strategy helps R more than U. It does not test interference. Intervals: 95% item
bootstrap over the 24 problems (10,000 draws).

| Policy | pass@1 uncued | pass@1 cued | Cue gain (95% CI) | Truncated uncued → cued | Mentions "factor" uncued → cued |
|---|---|---|---|---|---|
| Base | 0.422 | 0.411 | −0.010 [−0.078, +0.057] | 0.000 → 0.000 | 0.40 → 0.49 |
| R-s101 | 0.682 | 0.693 | +0.010 [−0.036, +0.068] | 0.188 → 0.177 | 0.56 → 0.71 |
| U-s101 | 0.724 | 0.714 | −0.010 [−0.073, +0.042] | 0.182 → 0.203 | 0.52 → 0.60 |
| R-s202 | 0.620 | 0.573 | −0.047 [−0.099, +0.000] | 0.188 → 0.219 | 0.56 → 0.69 |
| U-s202 | 0.646 | 0.693 | +0.047 [−0.005, +0.109] | 0.156 → 0.078 | 0.57 → 0.70 |

Difference in cue gain, (R cued − R uncued) − (U cued − U uncued): seed 101 +0.021 [−0.063, +0.120]; seed 202 −0.094
[−0.193, −0.016].

**Answer.** The cue changes what the models write but not how often they succeed. Every policy mentions "factor" more
often when cued (+9 to +15 points). No policy's cue gain has an interval that excludes zero. The R-versus-U difference
points in opposite directions in the two seeds, so naming the strategy does not help R more than U. Uncued, U
scores above R on this panel in both seeds, although R led on the sealed test in seed 101. With 24
problems × 8 samples, the panel detects only differences of about 10 points or more.

**Inspection.** The fixed sample (problems at panel positions 0, 6, 12 and 18, sample 0, all 10 policy-forms; 40
responses) is in `research/evidence/a15/cue_analysis.txt`. The grader was consistent in all 40: on problem 0 every
\boxed{10} scored 1 and every \boxed{28} scored 0 (10 is correct); on problem 12 every \boxed{8} scored 1 and every 4
or 6 scored 0. Problem 18 hit the 12,288-token cap in 4 of the 8 trained-policy responses.

## 6. Question and design

**Question.** When a model learns hard competition problems with GRPO, does easier practice on a technique those problems
use (polynomial factorization) help more than equally easy practice on an unrelated technique (modular arithmetic)?

**Model.** Qwen3-1.7B (`70d244cc`), full-parameter training, non-thinking mode, T = 1.0.

| Arm | Prompts per round (16 × 8 responses) | Purpose |
|---|---|---|
| H (hard only) | 8 shared hard + 8 extra hard | control: hard practice only |
| R (related practice) | 8 shared hard + 8 factoring practice | treatment |
| U (unrelated practice) | 8 shared hard + 8 modular-arithmetic practice | control: same practice dose, unrelated skill |

R and U see the same hard prompts in the same rounds, and their practice items are matched one-to-one on difficulty and
length (D48). The primary comparison is therefore paired: R − U.

**Data.** H_train 46 hard problems (25 MATH + 21 Omni-MATH; 2–12 of 32 base attempts correct). Practice pools 63 + 63
items. target_dev 53 problems (curves only). target_test 128 MATH test problems, sealed until all training ended.

**Training.** GRPO, lr 2e-6, symmetric clip 0.2, no KL, 4 optimizer steps per round, 128 rounds, FP32 weights and Adam
moments. Settings chosen by a frozen calibration rule (D57).

**Length cap.** 8,192 tokens for rounds 1–32, then 12,288 for rounds 33–128 in every run (A10/A11). Every comparison
across round 32 includes this change.

**Evaluation.** Test: 128 problems × 32 samples, cap 12,288. Dev: 53 problems × 8 samples at rounds 0, 32, 64, 96, 128.

## 7. What ran

Scope (A9): H/R/U × seeds 101 and 202. All runs are attempt a2; attempt a1 data are excluded (D99; asserted in code).

| Run | Status | Notes |
|---|---|---|
| H-s101 | complete, validated | — |
| R-s101 | complete, validated | final save failed once (storage); one retry from round 124 (D103) |
| U-s101 | complete, validated | — |
| R-s202 | complete, validated | restarted once at round 16 for checkpoint cadence (D105) |
| U-s202 | complete, validated | failed twice on storage write errors; resumed from round 64 (D111, D117) |
| H-s202 | partial (A16) | did not fit before the deadline; runs until 19:00Z; no test; appendix A |

**Storage incident.** The account's 500 GB quota (shared with other projects) reached 495 GB on 9 Nov. Writes became
slow and failed with `Errno 512`/`EINTR` (D116). This caused every checkpoint-save failure and both U-s202 crashes.
Pruning freed ≈ 214 GB (all deletions logged). Committed checkpoints are checksum-verified; no result depends on a
failed write.

**Bookkeeping defect (D127).** In resumed runs, saved step counters came from later rounds, because the background
saver serialised a live dictionary. Model weights, optimizer state and random state were saved correctly. Token counts
in this report are therefore computed from the training records, not from those counters.

## 8. Learning curves and probes

Figures `figures/learning_rounds.png`, `figures/learning_exposure.png` (dev pass@1 vs round and vs cumulative hard
training responses) and `figures/pope_R.png`, `figures/pope_U.png` (J_R or J_U against J_H). The dotted line marks the
cap change. The untrained model scores 0.403–0.422 on dev across runs because vLLM sampling is not bitwise reproducible;
dev differences below ≈ 0.02 are noise. At round 128 the dev panel favours R in seed 101 (0.729 vs 0.691) and U in seed
202 (0.703 vs 0.656).

## 9. Truncation

Share of training responses cut off at the cap (a cut-off response scores 0):

| Run | Shared hard, r1–32 (8,192) | Shared hard, r33–128 (12,288) | Practice / extra hard, r33–128 |
|---|---|---|---|
| H-s101 | 6.6% | 6.9% | 7.3% (extra hard) |
| R-s101 | 6.1% | 10.6% | 1.6% |
| U-s101 | 6.1% | 14.0% | 0.3% |
| R-s202 | 6.2% | 9.3% | 1.0% |
| U-s202 | 4.5% | 10.3% | 0.1% |

After round 32, U truncated more on shared hard prompts than R in both seeds. Truncation is part of the outcome, not a
separate error.

## 10. Compute

[PENDING `qis.analysis.cost_table`] GPU-hours per category, every allocation charged, including failed, cancelled and
superseded attempts. Active training time on the canonical trajectory: H-s101 26.3 h, R-s101 21.0 h, U-s101 19.3 h,
R-s202 18.0 h, U-s202 16.0 h.

## 11. Omitted or partial items

- H-s202: partial run only (A16); no test result, so R − H and the H mean rest on seed 101.
- Local-update extension from U-s101 round 64: skipped; that checkpoint and optimizer state were not retained.
- [PENDING any further omission.]

## 12. Limitations

- Two training seeds; intervals do not cover training variation. Seed 202 lacks H.
- H_train has 46 problems, 21 from Omni-MATH; the test set is MATH only.
- Practice pools are 86% authored items (equal across R and U).
- The cap changed after round 32 in every run.
- vLLM sampling is not bitwise reproducible; replays differ (D99).
- The agent annotated techniques and audited them; no human reviewed them.
- Sections 3–5 were added after the design was fixed (A15) and are exploratory.

## Appendix A. Partial H-s202 run (A16)

[PENDING at 19:00Z stop: rounds reached; dev pass@1 at rounds 0 and 32 (if reached) next to R-s202 and U-s202 at the
same rounds; shared-hard success in 8-round bins the partial run completed. Figure `figures/a16_partial.png`.] H, R and
U see identical shared hard prompts in every round (verified on the rounds run so far). This comparison is not an
endpoint and has no test result.

## Reproduction

Code: commit [PENDING]. Tables and figures: `qis.analysis.report_data`, `report_tables`, `per_problem`,
`interference`, `figures`, `cue`, `cost_table`; source tables in `reports/tables/`. Decisions: `research/DECISIONS.md`.
