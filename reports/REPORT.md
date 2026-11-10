# Does related practice help GRPO learn hard math? Qwen3-1.7B, H vs R vs U

Final, 10 Nov 2026 (2:40 PM EST). Sections 3–5 are EXPLORATORY analyses added by owner
amendment A15 after the primary design was fixed; section 2 is the primary result.

## 1. Lead table (A13)

Sealed target_test, 128 MATH problems × 32 samples, pass@1 (each problem weighted equally). Intervals: 95% paired
bootstrap over the 128 problems; they describe these trained models, not run-to-run training variation.

| Seed | H | R | U | R − U | R − H |
|---|---|---|---|---|---|
| 101 | 0.770 | 0.819 | 0.777 | +0.042 [+0.021, +0.063] | +0.049 [+0.020, +0.080] |
| 202 | not run (partial, A16) | 0.748 | 0.792 | −0.044 [−0.072, −0.018] | not available |
| Mean | not available (seed 101 only: 0.770) | 0.783 | 0.785 | −0.001 [−0.018, +0.015] | not available (seed 101 only: +0.049) |

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

All 12 update rows score 0.020–0.040 below the no-update row on J_R, including the related branch. A reference check
(D131, job 11105174) tested two explanations. First, the save–restore–reload cycle that every update branch goes
through: evaluating the base model before and after the cycle gives identical scores (J_H 0.4316, J_R 0.808, J_U 0.793
both times), so the cycle has no effect. Second, chance in the single no-update evaluation: a fresh no-update evaluation
on another GPU type (L40S; the original ran on an L40) gives J_R 0.808, close to the update rows' mean of 0.802, and
J_H 0.432 instead of 0.400. The original no-update row was therefore a high J_R draw and a low J_H draw. Every Δ in the
table above shares that one draw, so the uniform J_R drop is not a measured effect of the update.

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

**Reference at round 64.** A fresh no-update evaluation of R-s101 round 64 on an L40S gives J_H 0.631, J_R 0.893, J_U
0.951; the original (RTX 6000 Ada) gave 0.606, 0.902, 0.940. Evaluating it again after the save–restore–reload cycle gives the same three scores exactly, as at r0.

**How large a change can this design detect?** On one GPU, identical weights give identical scores (the starred rows).
Any change of weights or hardware gives a new sample of 512 responses per probe. Two such samples of the same model
differed by 0.025 (r64) and 0.031 (r0) in J_H. The 12 update rows at r64 span 0.592–0.633. A single ΔJ_H smaller than
about 0.03 cannot be told apart from this sampling variation.

**Answer.** No. The unrelated update did not measurably lower hard success at either checkpoint. Its mean ΔJ_H is
+0.005 at r0 and +0.003 at r64. Its largest single drop (−0.016 at r0, −0.014 at r64) is smaller than the 0.025–0.031
difference between two evaluations of the same model. At r0 its mean is 0.009 below the hard branch and 0.022 below
the related branch; at r64 it is 0.003 above hard and 0.007 below related. These gaps are smaller still. One step on 32 responses
is a small update; this design can detect only a drop of about 0.03 or more.

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
R-s202 94 vs 25, U-s202 122 vs 23. On matched cells (same prompt, same round, R and U), all-wrong in R only versus U
only: 39 vs 41 (seed 101) and 30 vs 36 (seed 202), with 19 and 18 all-wrong in both. The U excess is small. H trains on twice as many hard prompts per round, so it is not a dose-matched
comparison with R and U. Figure `figures/interference_windows.png`; the cap change after round 32 is marked.

**Dev panel comparison.** On the 53 dev problems, a problem is "lost" between two scheduled evaluations if it went
from ≥ 1 correct of 16 to 0, and "gained" if the reverse. Totals over rounds 0 → 32 → 64 → 96 → 128:

| Run | Lost | Gained | Problems with 0 correct, round 0 → 128 |
|---|---|---|---|
| H-s101 | 10 | 20 | 18 → 8 |
| R-s101 | 4 | 15 | 17 → 6 |
| U-s101 | 5 | 17 | 18 → 6 |
| R-s202 | 6 | 18 | 19 → 7 |
| U-s202 | 7 | 19 | 18 → 6 |

Every run gained 2 to 4 problems for each one it lost (H 2, the others 2.7–3.8). R and U lose within one problem of
each other in each seed and gain within two. The dev panel
shows the same picture as the training records: losses happen, but gains outnumber them, and the arms do not differ.

**Which links the evidence supports.**
1. Practice choice → test success: not supported. R − U is +0.042 in seed 101 and −0.044 in seed 202 (section 2).
2. One unrelated update → lower hard success: not detected. Single-step changes stay below the ≈ 0.03 noise floor at
   both checkpoints (section 3).
3. Training → lost signal on solvable hard problems: present but rare (2.5–5.0% of later appearances), outnumbered by
   problems mastered, and similar for R and U on matched cells. Much of U-s101's excess is length-cap truncation.

The study therefore finds no interference effect large enough to separate the arms at this scale (two seeds, 128 rounds,
1.7B model). Effects below about 4 test points, or 0.03 on the probes, remain possible.

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
| H-s202 | partial (A16) | did not fit before the deadline; stopped at round 64 (19:16Z, D140); no test; appendix A |

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

Every Slurm allocation in the project ledger is charged, including failed, cancelled and superseded attempts (`sacct`,
job 11109623, 19:22Z on 10 Nov; table `tables/cost.md`). GPU-hours = elapsed time × GPUs allocated.

| Category | Jobs | Allocations | GPU-h | of which canonical | CPU-h | Final states |
|---|---|---|---|---|---|---|
| CPU jobs | 126 | 131 | 0.0 | 0.0 | 31.0 | CANCELLED 12, COMPLETED 103, FAILED 11, PREEMPTED 5 |
| local-update diagnostic | 8 | 8 | 14.9 | 0.0 | 118.9 | CANCELLED 4, COMPLETED 3, FAILED 1 |
| main H-s101 | 2 | 2 | 33.6 | 29.5 | 268.9 | CANCELLED 1, COMPLETED 1 |
| main H-s202 | 2 | 2 | 11.0 | 0.0 | 88.1 | CANCELLED 2 |
| main H-s303 | 2 | 2 | 0.0 | 0.0 | 0.0 | CANCELLED 2 |
| main R-s101 | 3 | 3 | 32.9 | 27.9 | 263.6 | CANCELLED 1, COMPLETED 1, FAILED 1 |
| main R-s202 | 3 | 3 | 22.8 | 22.8 | 182.3 | CANCELLED 2, COMPLETED 1 |
| main R-s303 | 1 | 1 | 0.0 | 0.0 | 0.0 | CANCELLED 1 |
| main U-s101 | 2 | 2 | 27.3 | 22.6 | 218.6 | CANCELLED 1, COMPLETED 1 |
| main U-s202 | 5 | 5 | 25.6 | 25.6 | 205.0 | CANCELLED 2, COMPLETED 1, FAILED 2 |
| main U-s303 | 1 | 1 | 0.0 | 0.0 | 0.0 | CANCELLED 1 |
| other GPU jobs (screening, calibration, smoke, A10/A11 studies) | 41 | 41 | 42.0 | 0.0 | 335.6 | CANCELLED 6, COMPLETED 28, FAILED 7 |
| sealed test evaluation | 8 | 8 | 21.5 | 0.0 | 171.8 | CANCELLED 2, COMPLETED 6 |
| strategy-cue panel (A15.4) | 2 | 2 | 2.2 | 0.0 | 17.7 | CANCELLED 1, COMPLETED 1 |
| **Total** | 206 | 211 | 233.8 | | 1901.4 | |

**Total: 233.8 GPU-h** (211 allocations) **and 1,901 CPU-h.** The canonical trajectories of the five complete runs used
128.4 GPU-h (55%). The rest went to sealed-test evaluation (21.5), the partial H-s202 run (11.0), the exploratory A15
analyses (local-update diagnostic 14.9, cue panel 2.2), screening, calibration and protocol studies (42.0), and
superseded or failed attempts of the main runs (13.8). Active training time on the canonical trajectories: H-s101
26.3 h, R-s101 21.0 h, U-s101 19.3 h, R-s202 18.0 h, U-s202 16.0 h. The cost job itself (≈ 0.01 GPU-h) is not in the
table. GPU limit (A15: 3 general + 6 preempt): exceeded once, for 99 s at 02:04Z on 10 Nov, when a moved job started as
a fourth general job; it was cancelled at once and replaced by a dependency chain (D124). Otherwise general use stayed
at or below 3 GPUs and preempt at or below 1 GPU after A15 (from `tables/sacct.txt`).

## 11. Omitted or partial items

- H-s202: partial run only (A16); no test result, so R − H and the H mean rest on seed 101.
- Local-update extension from U-s101 round 64: skipped; that checkpoint and optimizer state were not retained.
- Seed 303 and the clipping sensitivity runs: deferred under A9; not run.
- Two seeds only: the study cannot estimate training variation, and the seeds disagree on the primary contrast.
- The A15.2 interference trace uses training records; it has no independent evaluation per round.

## 12. Limitations

- Two training seeds; intervals do not cover training variation. Seed 202 lacks H.
- H_train has 46 problems, 21 from Omni-MATH; the test set is MATH only.
- Practice pools are 86% authored items (equal across R and U).
- The cap changed after round 32 in every run.
- vLLM sampling is not bitwise reproducible; replays differ (D99).
- The agent annotated techniques and audited them; no human reviewed them.
- Sections 3–5 were added after the design was fixed (A15) and are exploratory.

## Appendix A. Partial H-s202 run (A16)

H-s202 (job 11082192) started late: under the 3-GPU limit, a full H-s202 run did not fit before the deadline (D100), so it ran last, as a
partial run (D128). It was stopped at 19:16Z on 10 Nov
after its round-64 evaluations (D139, D140), so it covers rounds 1–64 of 128. It has no test evaluation, so this
comparison is not an endpoint. H, R and U saw identical shared hard prompts in all 64 rounds (64/64 verified).

**Dev panel** (53 problems × 8 samples, pass@1):

| Round | H-s202 (partial) | R-s202 | U-s202 | For reference: H / R / U seed 101 |
|---|---|---|---|---|
| 0 | 0.403 | 0.403 | 0.422 | 0.417 / 0.417 / 0.422 |
| 32 | 0.566 | 0.599 | 0.611 | 0.540 / 0.542 / 0.601 |
| 64 | 0.658 | 0.613 | 0.679 | 0.637 / 0.736 / 0.710 |

Round 0 is the same base model in every run. Its scores still differ by up to 0.019, which is the evaluation noise
found in section 3. Differences of that size between arms are not meaningful.

**Shared hard prompts in training** (success over the 8 shared prompts × 8 answers, 8-round bins):

| Rounds | 1–8 | 9–16 | 17–24 | 25–32 | 33–40 | 41–48 | 49–56 | 57–64 |
|---|---|---|---|---|---|---|---|---|
| H-s202 | 0.266 | 0.488 | 0.584 | 0.605 | 0.693 | 0.740 | 0.779 | 0.777 |
| R-s202 | 0.238 | 0.416 | 0.520 | 0.572 | 0.559 | 0.631 | 0.691 | 0.699 |
| U-s202 | 0.217 | 0.383 | 0.500 | 0.574 | 0.619 | 0.676 | 0.684 | 0.680 |
| H-s101 | 0.213 | 0.318 | 0.424 | 0.578 | 0.625 | 0.680 | 0.730 | 0.727 |
| R-s101 | 0.260 | 0.393 | 0.449 | 0.486 | 0.596 | 0.734 | 0.713 | 0.771 |
| U-s101 | 0.258 | 0.412 | 0.529 | 0.559 | 0.635 | 0.678 | 0.719 | 0.734 |

**Reading.** In seed 202, H leads both R and U on the shared hard prompts in every bin, by 0.03–0.09 over the better of the two. H trains on twice as
many hard prompts per round, so this lead is expected and is not a dose-matched comparison. In seed 101, H led in only 2
of the 8 bins. On the dev panel at round 64, H-s202 sits between R-s202 and U-s202; in seed 101, H was lowest. The
partial run therefore repeats the main finding: the order of the arms changes between seeds. Figure
`figures/a16_partial.png`.

## Reproduction

Code: commit `41f055f` (the tree that produced the final tables and figures, job 11109597). Tables and figures: `qis.analysis.report_data`, `report_tables`, `per_problem`,
`interference`, `figures`, `cue`, `cost_table`; source tables in `reports/tables/`. Decisions: `research/DECISIONS.md`.
