# Does related practice help GRPO learn hard math? Qwen3-1.7B, H vs R vs U

DRAFT — sections marked `[PENDING …]` are filled only after all training has ended and the sealed test set is opened
(A13). Everything else is from non-sealed records and is final unless marked.

## 1. Answer

[PENDING sealed test] Lead table (A13): target_test pass@1 for H, R and U, paired R − U and R − H, per seed (101, 202),
then the mean over seeds. One sentence on what it shows; one on how far it can be trusted (two seeds).

## 2. Question and design

**Question.** When a model learns hard competition problems with GRPO, does easier practice on a technique those problems
use (polynomial factorization) help more than equally easy practice on an unrelated technique (modular arithmetic)?

**Model.** Qwen3-1.7B (`70d244cc`), full-parameter training, non-thinking mode, T = 1.0.

**Arms.** Every arm trains for 128 rounds; each round has 16 prompts × 8 responses.

| Arm | Prompts per round | Purpose |
|---|---|---|
| H (hard only) | 8 shared hard + 8 extra hard | control: hard practice only |
| R (related practice) | 8 shared hard + 8 factoring practice | treatment |
| U (unrelated practice) | 8 shared hard + 8 modular-arithmetic practice | control: same practice dose, unrelated skill |

R and U see the same hard prompts in the same rounds, and their practice items are matched one-to-one on difficulty and
length (D48). The primary comparison is therefore paired: R − U.

**Data.** H_train 46 hard problems (25 MATH + 21 Omni-MATH; 2–12 of 32 base attempts correct). Practice pools 63 + 63
items. target_dev 53 problems (curves only). target_test 128 MATH test problems, sealed until all training ended.

**Training.** GRPO, lr 2e-6, symmetric clip 0.2, no KL, 4 optimizer steps per round, FP32 weights and Adam moments
(precision audit passed). Settings chosen by a frozen calibration rule (D57).

**Response length cap.** 8,192 tokens for rounds 1–32, then 12,288 tokens for rounds 33–128 in every run (owner
amendments A10/A11). Every comparison across round 32 includes this change.

**Evaluation.** Test: 128 problems × 32 samples per model, cap 12,288; pass@1 is the mean per-problem success, each
problem weighted equally. Intervals resample the 128 problems (10,000 draws); they describe sampling and problem variation
for these trained models, not training variability. Dev: 53 problems × 8 samples at rounds 0, 32, 64, 96, 128.

## 3. What ran

Scope (owner amendment A9): six runs, H/R/U × seeds 101 and 202. Seed 303 and the clipping-sensitivity arm were deferred
before any result. All six runs are attempt a2 (restarted from round 0 under the A11 cap schedule); attempt a1 data are
excluded from every table and figure (D99; asserted in code).

| Run | Status | Notes |
|---|---|---|
| H-s101 | complete, validated | — |
| R-s101 | complete, validated | final save failed once (storage); one retry from round 124 (D103) |
| U-s101 | complete, validated | — |
| H-s202 | not run | did not fit before the deadline (≈ 29 h per H run); H rows for seed 202 are missing |
| R-s202 | [PENDING] | restarted once at round 16 for checkpoint cadence (D105) |
| U-s202 | [PENDING] | failed twice on storage write errors; resumed from round 64 (D111, D117) |

**Storage incident.** By 9 Nov the account's 500 GB quota (shared with other projects) had filled to 495 GB (measured 21:11Z). Writes then
became slow and failed with `Errno 512`/`EINTR` (D116). This caused all checkpoint-save failures and both U-s202 crashes.
Pruning failed partial writes and superseded checkpoints and retiring two verified optimizer states freed ≈ 214 GB (all deletions logged). Committed checkpoints are
checksum-verified; no result depends on a failed write.

## 4. Main result

[PENDING sealed test] Table from `qis.analysis.report_tables` (pass@1 per arm and seed; R − U, R − H, U − H with 95%
problem-resampling intervals; mean; seed agreement; base model). Figure `figures/main_result.png`.

pass@8 table (secondary). 8,192-token view of the same traces (secondary).

## 5. Learning curves and exposure

Figures `figures/learning_rounds.png` (dev pass@1 vs round) and `figures/learning_exposure.png` (vs cumulative
hard-prompt training responses). Each seed is drawn separately. The dotted line marks the cap change after round 32.

Note on evaluation noise: the untrained model scores 0.403–0.422 on dev across runs with identical seeds, because vLLM
generation is not bitwise reproducible (D99). Differences on dev smaller than ≈ 0.02 are within evaluation noise.

## 6. Probe trajectories (POPE-style)

Figures `figures/pope_R.png` (J_R vs J_H) and `figures/pope_U.png` (J_U vs J_H). J_H: 32 fixed hard-dev problems; J_R:
28 fixed factoring probes; J_U: 32 fixed modular-arithmetic probes; every arm on the same panels, IDs frozen before
results; square = round 0; arrows join successive evaluations (rounds 0, 32, 64, 96, 128).

## 7. Local update effects

[PENDING diagnostics 11099083 (source round 0) and 11099088 (source R-s101 round 64)] From identical model + optimizer
snapshots, one AdamW/GRPO step on hard, related or unrelated prompts (4 prompts × 8 responses, minibatch seeds
4101–4104), then J_H, J_R, J_U at 16 samples per problem, against a no-update branch. Reported as finite changes after one
update, not as gradients. Figure `figures/local_update.png`. A run that times out reports only complete seed sets.

## 8. Truncation

Share of training responses that hit the cap (hard prompts), from training records:

| Run | rounds 1–32 (cap 8,192) | rounds 33–128 (cap 12,288) | practice prompts, rounds 33–128 |
|---|---|---|---|
| [PENDING final run of `report_data`] | | | |

Interim (D115): hard prompts after round 32 — H-s101 7.1%, R-s101 10.6%, U-s101 14.0%; practice prompts 0.0–1.6%. A
truncated response scores 0. Arms that write longer answers lose more to the cap, so truncation is part of the outcome, not
a separate error.

## 9. Compute

[PENDING final `qis.analysis.cost_table`] GPU-hours per category, every allocation charged, including failed, cancelled
and superseded attempts. Interim (9 Nov 20:30Z): ≈ 177 GPU-h over 175 allocations.

## 10. Limitations

- Two training seeds. Agreement in sign and size across the two seeds is the main evidence; the intervals do not cover
  training variability. Seed 202 lacks H.
- H_train has 46 problems, 21 from Omni-MATH; the test set is MATH only.
- Practice pools are 86% authored items (equal across R and U).
- The cap changed after round 32 in every run; curves across that point mix two caps.
- vLLM sampling is not bitwise reproducible; replays differ (D99).
- Annotation of techniques was done by the agent and audited, not reviewed by a human.

## 11. Reproduction

Code: commit [PENDING]. Tables and figures: `qis.analysis.report_data`, `qis.analysis.report_tables`,
`qis.analysis.figures`, `qis.analysis.cost_table`; source tables in `reports/tables/`. Decisions: `research/DECISIONS.md`.
