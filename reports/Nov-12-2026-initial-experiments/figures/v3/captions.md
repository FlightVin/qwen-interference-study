# Report v3 figure captions

Purpose: one caption per figure in `reports/figures/v3/`, numbered as in `reports/REPORT.md` (page order). Each caption
names the policies, problems, attempts and uncertainty measure shown, and the source table. Terms follow
`research/DEFINITIONS.md` v1.1. Figures are built by `reports/build/build_v3.py` and read only the source tables in
`reports/tables/v3/`. Colors: H grey-blue, R orange, U teal, base black. Seed 101 is drawn with circles and solid lines;
seed 202 with squares and dashed lines. Multi-panel figures stack their panels top to bottom.

## Figure 1 `f01_endpoints.png` — Sealed test: base and round-128 endpoints
Policies: the base policy (Qwen3-1.7B `70d244cc`) and the six round-128 endpoints (H, R, U × seeds 101, 202). Problems: the
128 sealed-test problems, 32 attempts each, cap 12,288 tokens. Points show pass@1. Bars show 95% problem-bootstrap
intervals (10,000 resamples of the 128 problems). They are wide (±4.9 to ±6.0 points) because they include
problem-to-problem variation. Compare policies with the paired contrasts in Figure 2, not by overlap of these bars. The
intervals cover problem and attempt sampling only, not variation between training runs. H-s202 was evaluated on L40S
(problems of shards k0–k2) and RTX A6000 (k3–k5); R endpoints on L40; base, H-s101 and U endpoints on L40S. Source:
`t01_endpoints`.

## Figure 2 `f02_contrasts_test.png` — Sealed test: paired contrasts per seed
Contrasts R − U, R − H and U − H between round-128 endpoints of the same training seed, in percentage points of sealed-test
pass@1 (128 problems × 32 attempts). Bars: 95% paired problem-bootstrap intervals (same resampled problems for both
policies; audit pass B, 10,000 draws). The vertical line marks zero difference. Two of the four R − H and U − H intervals
include zero. With two seeds per arm, no interval here describes variation across future training runs. Source:
`t02_contrasts_test`.

## Figure 3 `f03_dev_rounds.png` — Dev pass@1 over training rounds
All six runs; top panel seed 101, bottom panel seed 202. Problems: the 53 dev problems, 8 attempts each, evaluated in-run at
rounds 0, 32, 64, 96 and 128. Round 0 ("base") is the same base policy with the same evaluation seeds in every run. Its
spread (40.3% to 42.2%) is the sampling noise between evaluation jobs. The dotted line marks the cap change: training and
evaluation used 8,192 tokens up to round 32 (open markers) and 12,288 from round 33 (filled markers). Any change from round
32 to round 64 therefore mixes training with the larger cap. No intervals are drawn. Per-point 95% problem-bootstrap
intervals (±9.2 to ±10.6 points) are in the source table. H-s202 rounds 64–128 are trajectory a2-r60 (resumed from round
60). Source: `t03_curves`.

## Figure 4 `f04_dev_hard_attempts.png` — Dev pass@1 against hard training attempts
The same evaluations as Figure 3, plotted against the cumulative number of hard training attempts. H trains on 128 hard
attempts per round (16 slots × 8 attempts) and reaches 16,384 at round 128. R and U train on 64 hard attempts per round (the
8 shared hard slots) and reach 8,192. Hard attempts are not the only cost: R and U also generate 64 practice attempts per
round. Markers as in Figure 3; no intervals drawn (sizes in `t03_curves`). Source: `t03_curves`.

## Figure 5 `f05_traj_JH_JR.png` — Training paths in (J_H, J_R)
One path per run through rounds 0 → 32 → 64 → 96 → 128; top panel seed 101, bottom panel seed 202. Arrows point in the
direction of training. Round 128 is the large black-edged marker, labelled with the arm letter. x: J_H, pass@1 on the 32
hard dev problems (the first 32 of dev), 8 attempts each. y: J_R, pass@1 on the 28 factoring probe problems, 8 attempts
each. Every policy is evaluated on every probe, so H and U also have J_R scores. Open markers: cap 8,192 (rounds 0, 32);
filled: cap 12,288. The three round-0 points of a seed ("r0") are the same base policy. No intervals drawn; 95%
problem-bootstrap intervals are ±11.3 to ±14.6 points (J_H) and ±6.0 to ±11.6 points (J_R). Source: `t03_curves`.

## Figure 6 `f06_traj_JH_JU.png` — Training paths in (J_H, J_U)
As Figure 5, with y = J_U: pass@1 on the 32 modular-arithmetic probe problems, 8 attempts each. J_U rises in H and R, which
never practise modular arithmetic; Figure 7 tests whether this rise is specific to the probe. No intervals drawn; J_U
intervals are ±2.5 to ±10.9 points. Source: `t03_curves`.

## Figure 7 `f12_control.png` — Gains on unpractised subjects and on the probes
Six round-128 endpoints. Top: gain over the base policy on the control panel `control_v1`. It holds 64 MATH test problems
(16 each from geometry, counting and probability, precalculus, prealgebra; levels 4–5) that appear in no study set. Each
problem has 8 attempts at cap 12,288, and the panel was pre-registered in `research/audit-a26/transfer/PREREG.md`. Middle
and bottom: the same run's in-run J_U and J_R gain from round 0 (cap 8,192) to round 128 (cap 12,288). Bars: 95% paired
problem-bootstrap intervals (10,000 draws, generator seed 2026). The base scores 50.8% on control and about 81% on J_U and
J_R, so the control panel has more room to gain. Control evaluations ran on L40S (base, H-s101, U-s101, U-s202) and L40
(H-s202, R-s101, R-s202). Source: `t09_control`.

## Figure 8 `f07a_delta_j_r0.png` — One optimizer step from the base policy (round 0)
Each point is one update draw: a single GRPO step on 4 prompts × 8 attempts of one type (hard, related or unrelated; draw
seeds 4101–4104), starting from the base policy with a fresh optimizer. After the step, J_H (32 problems), J_R (28) and
J_U (32) were evaluated with 16 attempts per problem at cap 8,192, one panel per probe. Evaluation seeds are identical
across draws. y = draw minus `none` (no update, evaluated in the same job on L40). Solid line: `none`. Dashed line:
`none_rerun` (no update, separate L40S job; `none_cycled` gives identical values). The gap between the two references
(+3.1 J_H points) is the reference noise. No intervals drawn. Source: `t04_diag`.

## Figure 9 `f07b_delta_j_r64.png` — One optimizer step from R-s101 round 64
As Figure 8, starting from R-s101 round 64. The cap 8,192 is lower than the 12,288 this source trained with: 36.1% of its
diagnostic J_H attempts hit it. `none` ran in the same job on RTX 6000 Ada; `none_rerun` on L40S. The gap between the two
references is +2.5 J_H points. Crossed open markers: the three draws whose gradient was exactly zero (related-4101,
related-4102, unrelated-4103). AdamW still stepped with its stored moments, and the three evaluations are identical. No
intervals drawn. Source: `t04_diag`.

## Figure 10 `f08_length_pass1.png` — pass@1 before and after 4,096 more tokens
Five round-128 endpoints (H-s202 was not extended). Top: sealed test, 128 problems × 32 attempts. Bottom: stress panels,
221 problems × 8 attempts. Open marker: pass@1 at cap 12,288. Filled marker: pass@1 after every capped attempt was
continued by up to 4,096 tokens (16,384 in total) and regraded; finished attempts keep their score. Every checkpoint gains
1.6 to 2.9 points on the test and 2.3 to 3.1 points on stress. 95% paired problem-bootstrap intervals of each gain are in
the source table. Source: `t05_length`.

## Figure 11 `f08b_length_capped.png` — Capped attempts after 4,096 more tokens
For the same five endpoints, the attempts that hit the 12,288-token cap (n in each row label) split into finished correct
(recovered), finished wrong, and still capped at 16,384. Shares are of the capped attempts; exact counts are in the source
table. Top: sealed test; bottom: stress panels. Source: `t05_length`.

## Figure 12 `f09_length_contrasts.png` — Paired contrasts before and after the extension
R − U per seed, and R − H and U − H for seed 101 only (H-s202 was not extended). Top: sealed test (128 × 32); bottom: stress
panels (221 × 8). Open marker: contrast at cap 12,288. Filled marker: contrast after extension to 16,384. Bars: 95% paired
problem-bootstrap intervals (audit pass B). Intervals of the change itself are in the source table. Source:
`t06_length_contrasts`.

## Figure 13 `f10_stress_tiers.png` — Stress panels: pass@1 per tier
Policies: base, the six round-128 endpoints, and the only two intermediate checkpoints evaluated: R-s101 round 64 and
H-s202 round 64 (trajectory a2-r60; the a2-orig round-64 weights were never saved). Filled markers: round 128; open
markers: round 64. Top: Harder (95 problems); bottom: SuperHard (126). Tiers come from one annotation of solution
structure, not from model success. Each problem has 8 attempts at cap 12,288. Bars: 95% problem-bootstrap intervals within
the tier. All cells ran on L40S except H-s202 round 128 (two shards on L40, one on RTX A6000). Source: `t07_stress`.

## Figure 14 `f11_stress_contrasts.png` — Stress panels: paired contrasts per seed
R − U, R − H and U − H between round-128 endpoints of the same seed. Panels from top: all 221 stress problems, Harder (95),
SuperHard (126); 8 attempts per problem. Bars: 95% paired problem-bootstrap intervals within the tier (audit pass B). Under
the any-box grading sensitivity check, the seed-202 R − U and R − H intervals on all problems change their relation to zero
(`research/audit-a26/discrepancies.md` §3.2). Source: `t08_stress_contrasts`.
