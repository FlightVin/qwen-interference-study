# Dataset construction log

Purpose: the evidence trail for every pool decision (project.md §5, A2). Counts are per item:
successes c out of n samples, frozen model/prompt/decoding, cap 4,096 (D18).

## Stage 1 (16 samples/item; job 11072609, L40 babel-w5-28, 2026-11-07)
Per-item counts: `data/screening/<pool>_s1_counts.jsonl`. No verifier-unknown attempts; no prompt over the 1,024-token cap.

| pool (construction) | candidates | band | shortlisted | at 16/16 | at 0/16 | truncated attempts |
|---|---|---|---|---|---|---|
| hard (L4–5, multi-step) | 154 | 0–5/16 | 56 | 34 | 32 | 3.4% |
| hard_reserve (L3, D19) | 30 | 0–5/16 | 4 | 8 | 0 | 1.3% |
| related (direct practice) | 110 | 4–14/16 | 23 | 56 | 7 | 0.4% |
| unrelated (modular) | 152 | 4–14/16 | 41 | 76 | 2 | 0.6% |

**Reading.** Qwen3-1.7B (non-thinking, T=1) solves most MATH items of every role reliably. The
hard band (2–8/32 at stage 2) can be met only by items at roughly 1–5/16 now (28 incl. reserve)
plus a few of the 32 zero-success items. Projection: H_train ≈ 25–35, below the hard minimum 64.
The auxiliary pools are also short of 64 pairs (A2 options apply).

**Fallback feasibility (model-free upper bound).** Linear-equation word problems (algebra +
prealgebra, levels 4–5, any word-problem cue, an equation in the solution): ≤ ~128 construction
candidates before the skill audit, and easier on average than the factoring hard pool. The
fallback cannot be expected to produce more H_train items than the primary skill.

## Pass 2 (blind) — targets and probe panels (2026-11-07)
`data/annotations/pass2_targets_probes.jsonl`: 291 items (target_dev 63, first 150 of target_test in
seeded order, probe_related_dev 31, probe_unrelated_dev 47), three isolated agents, rubric v1+v1.1,
no access to pass-1 labels. All support quotes verbatim. Agreement with pass 1: 258/291 (88.7%);
disagreements: label 22, direct_practice 7, modular 4 → third read (both rationales) pending.

## Pass 2 (blind) — hard candidates; third read (2026-11-07)
- `data/annotations/pass2_hard.jsonl`: 184 hard + reserve candidates, two isolated agents, rubric
  v1+v1.1+v1.2. Agreement with pass 1 on (label, direct_practice): 158/184; all 158 central multi-step.
- `data/annotations/third_read.jsonl`: 59 disagreements (26 hard, 33 targets/probes). 59 resolved,
  0 unresolved; 58 follow pass 2, 1 follows pass 1 (`intermediate_algebra/test/00211`). Pass 1
  predates the v1.1 edge-case rules, which explains the split. Outcome: absent 26, incidental 21,
  central/direct 11, central/multi-step 1.
- Rubric gaps the third reader resolved by the nearest rule (medium confidence): completing a cube
  or fourth power (→ absent, like completing the square); perfect-square recognition then cancel
  (→ central via v1.3 F2); undetermined-coefficient factorizations (→ absent); rational equations
  count as polynomial equations for `direct_practice`; remainder used only as an inner step →
  `modular = no`.

## Stage 2 on the stage-1 shortlist (32 fresh attempts; job 11072896) and calibration screen (job 11072944)
| pool | items | core 2–8 | moderate 9–12 | aux band 12–24 | unknown |
|---|---|---|---|---|---|
| hard (L4–5) | 56 | 18 | 4 | — | 0 |
| hard_reserve (L3) | 4 | 3 | 0 | — | 0 |
| related | 23 | — | — | 7 | 0 |
| unrelated | 41 | — | — | 18 | 0 |
Calibration (16 samples): calib_train 200 → low 1–5/16: 27, high 6–12/16: 35; calib_val 200 → 20 / 31.
Both splits meet ≥ 16 per band. Full native stage 2 over all 184 hard candidates: job 11073157 (A4.1).

## Auxiliary gap and authored tranche (A2; recorded BEFORE any item is authored) — see D32
