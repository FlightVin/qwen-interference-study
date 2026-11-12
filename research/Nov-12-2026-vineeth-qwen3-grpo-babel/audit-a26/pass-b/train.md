# A26 pass B — training-side reconstruction (agent b-train)

Purpose: rebuild trajectories, learning curves, GRPO group statistics, the local update diagnostic, loss and recovery on
training prompts, and costs from raw records with b-train's own code. Main result: every training-side number checked
agrees with the raw records; R − U on the in-run panels changes sign between rounds and seeds; no diagnostic ΔJ separates
branch types beyond evaluation noise.

Method. `tools/b-train/btrain.py` (compute, stdlib + numpy, standby 11122432); `costs.py`, `tables.py` (laptop, stdlib).
Unrounded outputs: `data/train_*.json`. No project code was run; the minibatch plan and seed streams are re-implemented
from `grpo.py:44-64` and `seeds.py`. Bootstrap: own seed 26111111, 10,000 draws over problems, same draw for both policies.
Rounds: **1-based training round**; evaluation rounds = completed rounds. Labels: **V** verified from primary records by
my code; **C** claimed (a document, or a row I cannot recompute); **U** unknown.
Independence: DEFINITIONS v1 states some numbers below (group totals, momentum-only counts, H-s202 r64 dev, diagnostic J_H);
I recomputed each. I also read the `qis-diag-ref.sh` recipe comment (states the r0 J_R shift) and
`resume_audit_a17*.txt` in the H-s202 run directory (checkpoint listings only). I did not open the other agent's `eval_*.json`.

## 1. Canonical trajectory per run (V)

Rule. A metrics row is canonical if it is the **last write** of its key in `metrics.jsonl`: `(round)` for round rows,
`(round, minibatch)` for step rows, `(panel, round)` for evaluation rows. Canonical records are the files left in
`records/`; earlier attempts were moved to `records/superseded-<job>` or `records_a2orig_r60-64`.
Why the rule holds: (1) every resume re-runs all rounds after its resume point and the run then continues to round 128;
(2) for every run, all 128 canonical round rows match the canonical record files on five fields (all-wrong, mixed,
all-correct groups, truncated attempts, mean response length); (3) every superseded round row matches the superseded
record file of its own job on the same five fields; (4) step rows match by `valid_tokens` (section 4).

| run | segment (job, GPU) | resume from | round rows written | canonical round / step / eval rows | how it ended |
|---|---|---|---|---|---|
| H-s101 | 11081726, L40 | 0 | 1–128 | 128 / 512 / 20 | complete |
| R-s101 | 11081724, L40S | 0 | 1–128 | 124 / 496 / 16 | `final_checkpoint_failed` (Errno 512 at rounds 108, 120, 128) |
| R-s101 | 11092945, L40S | `round_0124` | 125–128 | 4 / 16 / 4 | complete |
| U-s101 | 11081725, L40S | 0 | 1–128 | 128 / 512 / 20 | complete |
| H-s202 | 11082192, L40 | 0 | 1–64 | 60 / 240 / 8 | stopped after r64 eval; `round_0064.tmp` left (C: `resume_audit_a17.txt` listing) |
| H-s202 | 11112377, L40S | `round_0060` | 61–128 | 68 / 272 / 12 | complete |
| R-s202 | 11081729, L40 | 0 | 1–19 + 3 step rows of round 20 | 16 / 64 / 4 | checkpoint failures at 4, 12; died in round 20 |
| R-s202 | 11094190, L40S | `round_0016` | 17–128 | 112 / 448 / 16 | complete |
| U-s202 | 11094037, L40S | 0 | 1–80 | 64 / 256 / 12 | job FAILED (recipe comment: Errno 512 writing metrics, C) |
| U-s202 | 11097811, L40S | `round_0064` | 65–81 | 0 / 0 / 0 | job FAILED |
| U-s202 | 11100159, L40S | `round_0064` | 65–128 | 64 / 256 / 8 | complete |

Attempts a1 (s101) are in `$P/runs/superseded/*-a1/` (`attempt_id` a1, no cap schedule, no J_H row; reached rounds 37 H,
55 R, 51 U). Naive round-row counts overcount: 132 (H-s202), 132 (R-s101), 131 (R-s202), 161 (U-s202). Schedule: shared-slot
IDs and request seeds equal in H, R, U in 128/128 rounds per seed; R/U practice slots share the pair index in 1,024/1,024;
each H_train problem has 20–23 shared presentations (R/U), 40–50 in total (H) (`train_schedule_check.json`).

## 2. Learning curves (V)

From the evaluation records (`records/eval_<panel>_rNNNN.jsonl.gz`), n = 8 per problem everywhere, unknown verifier status
0 everywhere. J_H = the first 32 dev IDs (equal to bundle `probe_J_H`). Cap from records: every truncated attempt has
exactly 8,192 tokens at rounds 0 and 32 and 12,288 at rounds 64–128. All 120 canonical metric eval rows equal the records
problem by problem. Trunc = share of attempts that hit the cap.

| run | round | cap | dev | dev trunc | J_H | J_H trunc | J_R | J_R trunc | J_U | J_U trunc |
|---|---|---|---|---|---|---|---|---|---|---|
| H-s101 | 0 | 8192 | 0.417 | 0.000 | 0.426 | 0.000 | 0.808 | 0.000 | 0.805 | 0.004 |
| H-s101 | 32 | 8192 | 0.540 | 0.170 | 0.570 | 0.145 | 0.821 | 0.000 | 0.836 | 0.027 |
| H-s101 | 64 | 12288 | 0.637 | 0.125 | 0.656 | 0.117 | 0.839 | 0.000 | 0.883 | 0.012 |
| H-s101 | 96 | 12288 | 0.639 | 0.113 | 0.672 | 0.098 | 0.844 | 0.000 | 0.883 | 0.012 |
| H-s101 | 128 | 12288 | 0.663 | 0.170 | 0.688 | 0.160 | 0.893 | 0.000 | 0.938 | 0.012 |
| R-s101 | 0 | 8192 | 0.417 | 0.005 | 0.418 | 0.004 | 0.817 | 0.000 | 0.812 | 0.000 |
| R-s101 | 32 | 8192 | 0.542 | 0.111 | 0.527 | 0.137 | 0.839 | 0.004 | 0.871 | 0.012 |
| R-s101 | 64 | 12288 | 0.736 | 0.198 | 0.738 | 0.215 | 0.915 | 0.013 | 0.957 | 0.004 |
| R-s101 | 96 | 12288 | 0.748 | 0.191 | 0.758 | 0.168 | 0.929 | 0.022 | 0.961 | 0.012 |
| R-s101 | 128 | 12288 | 0.729 | 0.184 | 0.766 | 0.152 | 0.911 | 0.031 | 0.961 | 0.008 |
| U-s101 | 0 | 8192 | 0.422 | 0.000 | 0.434 | 0.000 | 0.817 | 0.000 | 0.816 | 0.000 |
| U-s101 | 32 | 8192 | 0.601 | 0.172 | 0.621 | 0.184 | 0.830 | 0.004 | 0.891 | 0.020 |
| U-s101 | 64 | 12288 | 0.710 | 0.149 | 0.734 | 0.164 | 0.875 | 0.004 | 0.922 | 0.027 |
| U-s101 | 96 | 12288 | 0.684 | 0.233 | 0.688 | 0.246 | 0.915 | 0.009 | 0.969 | 0.020 |
| U-s101 | 128 | 12288 | 0.691 | 0.198 | 0.703 | 0.203 | 0.924 | 0.009 | 0.941 | 0.020 |
| H-s202 | 0 | 8192 | 0.403 | 0.000 | 0.410 | 0.000 | 0.808 | 0.000 | 0.805 | 0.000 |
| H-s202 | 32 | 8192 | 0.566 | 0.144 | 0.566 | 0.141 | 0.826 | 0.000 | 0.879 | 0.016 |
| H-s202 | 64 (a2-r60) | 12288 | 0.637 | 0.151 | 0.613 | 0.164 | 0.862 | 0.004 | 0.910 | 0.012 |
| H-s202 | 64 (a2-orig) | 12288 | 0.658 | 0.132 | 0.652 | 0.141 | 0.853 | 0.004 | 0.922 | 0.012 |
| H-s202 | 96 | 12288 | 0.660 | 0.215 | 0.668 | 0.211 | 0.920 | 0.004 | 0.887 | 0.035 |
| H-s202 | 128 | 12288 | 0.670 | 0.210 | 0.680 | 0.219 | 0.911 | 0.018 | 0.902 | 0.031 |
| R-s202 | 0 | 8192 | 0.403 | 0.000 | 0.410 | 0.000 | 0.808 | 0.000 | 0.805 | 0.000 |
| R-s202 | 32 | 8192 | 0.599 | 0.118 | 0.605 | 0.125 | 0.879 | 0.000 | 0.863 | 0.020 |
| R-s202 | 64 | 12288 | 0.613 | 0.116 | 0.617 | 0.117 | 0.821 | 0.009 | 0.906 | 0.016 |
| R-s202 | 96 | 12288 | 0.642 | 0.191 | 0.676 | 0.160 | 0.879 | 0.000 | 0.879 | 0.016 |
| R-s202 | 128 | 12288 | 0.656 | 0.222 | 0.676 | 0.203 | 0.902 | 0.000 | 0.918 | 0.023 |
| U-s202 | 0 | 8192 | 0.422 | 0.000 | 0.434 | 0.000 | 0.817 | 0.000 | 0.816 | 0.000 |
| U-s202 | 32 | 8192 | 0.611 | 0.130 | 0.625 | 0.129 | 0.826 | 0.004 | 0.879 | 0.004 |
| U-s202 | 64 | 12288 | 0.679 | 0.151 | 0.695 | 0.133 | 0.879 | 0.004 | 0.945 | 0.008 |
| U-s202 | 96 | 12288 | 0.679 | 0.186 | 0.680 | 0.188 | 0.906 | 0.000 | 0.961 | 0.008 |
| U-s202 | 128 | 12288 | 0.703 | 0.142 | 0.734 | 0.117 | 0.920 | 0.000 | 0.941 | 0.016 |

- H-s202 r64, a2-orig − a2-r60 (paired): dev +0.0212 [−0.0142, +0.0566]; J_H +0.0391 [−0.0078, +0.0859]; J_R −0.0089
  [−0.0491, +0.0357]; J_U +0.0117 [−0.0078, +0.0352]. a2-orig = metrics line 360 (job 11082192); a2-r60 = line 388.
- R-s101 superseded r128 evaluation (job 11081724): dev 0.710, J_H 0.738, J_R 0.884, J_U 0.934; rounds 125–128 were re-run.
- Round 0 evaluates identical weights with identical seeds, yet dev ranges 0.403–0.422 across jobs (cross-job sampling).

Training reward per role (records; equal to the round rows, max difference 0). Per round, with cumulative hard attempts:
`train_rewards.json`.

| run | rounds 1–32 hard / other | 33–64 | 65–96 | 97–128 | hard trunc 97–128 |
|---|---|---|---|---|---|
| H-s101 | 0.383 / 0.402 | 0.690 / 0.680 | 0.826 / 0.832 | 0.870 / 0.867 | 0.055 |
| R-s101 | 0.397 / 0.784 | 0.704 / 0.932 | 0.789 / 0.969 | 0.825 / 0.983 | 0.104 |
| U-s101 | 0.439 / 0.815 | 0.691 / 0.964 | 0.726 / 0.988 | 0.772 / 0.989 | 0.156 |
| H-s202 | 0.486 / 0.458 | 0.745 / 0.747 | 0.824 / 0.831 | 0.854 / 0.850 | 0.086 |
| R-s202 | 0.437 / 0.785 | 0.645 / 0.924 | 0.755 / 0.957 | 0.792 / 0.961 | 0.127 |
| U-s202 | 0.418 / 0.827 | 0.665 / 0.957 | 0.771 / 0.981 | 0.814 / 0.984 | 0.099 |

"hard" = `hard_common` (the shared slots, same problems in all arms of a seed); "other" = `hard_extra` (H) or practice (R/U).
At equal hard exposure (8,192 hard attempts), mean hard success over the trajectory is H 0.539 / 0.609 (rounds 1–64) vs
R 0.679 / 0.657 and U 0.657 / 0.667 (rounds 1–128). This comparison is confounded: H reaches that exposure in half the
rounds and half the optimizer steps, and before most of the 12,288-cap rounds.

## 3. Earliest observed ordering of R and U (V)

R − U pass@1 per seed, paired over problems. "First" = first trained evaluation round (32–128) with the same sign as at
round 128. "Stays" = first round from which the sign never changes again. Caps: 8,192 at rounds 0 and 32, 12,288 from 64,
so a change between 32 and 64 mixes training and the cap.

| seed | panel | r0 | r32 | r64 | r96 | r128 | first | stays | R − U at "first" [95% interval] |
|---|---|---|---|---|---|---|---|---|---|
| s101 | dev | −0.0047 | −0.0590 | +0.0259 | +0.0637 | +0.0377 | 64 | 64 | +0.0259 [−0.0212, +0.0731] |
| s101 | J_H | −0.0156 | −0.0938 | +0.0039 | +0.0703 | +0.0625 | 64 | 64 | +0.0039 [−0.0625, +0.0781] |
| s101 | J_R | +0.0000 | +0.0089 | +0.0402 | +0.0134 | −0.0134 | 128 | 128 | −0.0134 [−0.0804, +0.0402] |
| s101 | J_U | −0.0039 | −0.0195 | +0.0352 | −0.0078 | +0.0195 | 64 | 128 | +0.0352 [−0.0039, +0.0820] |
| s202 | dev | −0.0189 | −0.0118 | −0.0660 | −0.0377 | −0.0472 | 32 | 32 | −0.0118 [−0.0590, +0.0354] |
| s202 | J_H | −0.0234 | −0.0195 | −0.0781 | −0.0039 | −0.0586 | 32 | 32 | −0.0195 [−0.0781, +0.0391] |
| s202 | J_R | −0.0089 | +0.0536 | −0.0580 | −0.0268 | −0.0179 | 64 | 64 | −0.0580 [−0.1339, +0.0045] |
| s202 | J_U | −0.0117 | −0.0156 | −0.0391 | −0.0820 | −0.0234 | 32 | 32 | −0.0156 [−0.0586, +0.0273] |

- The round-128 order differs between seeds on dev, J_H and J_U (R ahead in s101, U ahead in s202).
- Every interval at the "first" round includes 0. Six of 40 cells exclude 0: s101 dev r96 [+0.0118, +0.1203]; s101 J_H r32
  [−0.1797, −0.0195] (U ahead; reversed later), r96 [+0.0039, +0.1484], r128 [+0.0078, +0.1211]; s202 J_R r32
  [+0.0045, +0.1116] (R ahead; reversed later); s202 J_U r96 [−0.1523, −0.0234].
- In s202 the round-0 (untrained, identical weights) difference already has the round-128 sign on all four panels.
- Stress panels: no seed has both an R and a U intermediate checkpoint (retained: R-s101 r64, H-s202 r64), so no earlier
  stress ordering exists (V: `$P/stress-eval` listing).

## 4. GRPO group statistics and momentum-only steps (V)

Groups per run, rounds 1–128 (2,048 groups; hard = `hard_common`; other = `hard_extra` or practice):

| run | all-wrong | mixed | all-correct | hard: wrong / mixed / correct | other: wrong / mixed / correct | momentum-only steps (of 512) | first (round) | per 32-round quarter |
|---|---|---|---|---|---|---|---|---|
| H-s101 | 100 | 1,289 | 659 | 45 / 657 / 322 | 55 / 632 / 337 | 18 | 70 | 0, 0, 6, 12 |
| R-s101 | 58 | 961 | 1,029 | 58 / 646 / 320 | 0 / 315 / 709 | 66 | 38 | 0, 9, 24, 33 |
| U-s101 | 60 | 930 | 1,058 | 60 / 663 / 301 | 0 / 267 / 757 | 53 | 27 | 1, 7, 16, 29 |
| H-s202 | 79 | 1,287 | 682 | 35 / 641 / 348 | 44 / 646 / 334 | 20 | 73 | 0, 0, 7, 13 |
| R-s202 | 48 | 1,100 | 900 | 48 / 723 / 253 | 0 / 377 / 647 | 25 | 59 | 0, 2, 3, 20 |
| U-s202 | 54 | 947 | 1,047 | 54 / 659 / 311 | 0 / 288 / 736 | 65 | 15 | 1, 8, 23, 33 |

Totals and momentum-only counts agree with DEFINITIONS v1 §3.D (both rows). Per-round counts: `train_groups.json`.

- **Minibatch reconstruction.** Sum of `n_tokens` over each predicted minibatch = step-row `valid_tokens` in 512/512 steps
  per run and in every superseded step row with records (16, 16, 12, 132). Confirms DEFINITIONS §4 item 4 on data.
- **Momentum-only step** = all 4 groups non-mixed. Predicted = observed `grad_norm_preclip` 0.0 in every run (0 mismatches);
  all 32 attempts have `adv` exactly 0.0; lr 2e-6 on all steps.
- In R/U, every momentum-only step contains 2 all-correct practice groups (balanced 2 + 2 minibatches need both practice
  groups non-mixed, and practice groups were never all-wrong: 0 of 4,096). In H they start only at rounds 70/73.
- **Advantages.** Recorded `adv` equals (r − mean)/(population std + 1e-6) within 1.2e-7 for all 16,384 attempts per run;
  0 nonzero advantages in non-mixed groups; 0 zero advantages in mixed groups.
- Steps with pre-clip norm above 1.0 (clipped): H 5/6, R 2/5, U 8/2 (s101/s202). Median nonzero norm 0.31–0.42.
- R-s202 step rows carry `opt_step` 77–524 for rounds 17–128: the resume read 76 from `round_0016`, saved while rounds
  17–19 ran (live-counter defect). Index R-s202 steps by (round, minibatch).

## 5. Local update diagnostic (V unless marked)

Files: `$P/diag-local/{r0,r64}/` (`diag_*.jsonl`, `records/<label>-<branch>-<seed>/`). r0 = base, fresh optimizer; r64 =
R-s101 `round_0064` (sha `6fe89540…`, saved by job 11081724 at 2026-11-08 12:24Z, canonical segment). One GRPO step on
4 prompts × 8 attempts per branch (seeds 4101–4104), then J_H/J_R/J_U at n = 16 with seeds identical across branches (V).
Cap 8,192 at both labels (all truncated attempts have 8,192 tokens). Rows equal my recount in 45/45 cells per label.

| label | branch type | ΔJ_H mean (sd, 4 draws) | ΔJ_R | ΔJ_U | ΔJ_H vs `none_cycled` |
|---|---|---|---|---|---|
| r0 | hard | +0.0142 (0.0075) | −0.0257 (0.0097) | −0.0127 (0.0202) | −0.0171 |
| r0 | related | +0.0269 (0.0111) | −0.0246 (0.0048) | −0.0220 (0.0190) | −0.0044 |
| r0 | unrelated | +0.0049 (0.0168) | −0.0268 (0.0041) | +0.0005 (0.0115) | −0.0264 |
| r0 | reference: `none_rerun` − `none` | +0.0312 | −0.0201 | −0.0137 | — |
| r0 | exact permutation p (range of 3 type means) | 0.074 | 0.926 | 0.217 | — |
| r64 | hard | +0.0010 (0.0079) | −0.0117 (0.0078) | −0.0088 (0.0025) | −0.0244 |
| r64 | related | +0.0103 (0.0051) | −0.0056 (0.0039) | −0.0122 (0.0075) | −0.0151 |
| r64 | unrelated | +0.0034 (0.0179) | −0.0095 (0.0059) | −0.0044 (0.0051) | −0.0220 |
| r64 | reference: `none_rerun` − `none` | +0.0254 | −0.0089 | +0.0117 | — |
| r64 | exact permutation p | 0.540 | 0.377 | 0.173 | — |

ΔJ = branch − `none` (pass@1). Absolute values, per-draw ΔJ with intervals, truncation: `train_diag.json`.

- **No ΔJ separates branch types beyond noise.** The source evaluated twice in separate processes (same weights, seeds)
  differs by up to 0.031. This shift exceeds the range of type means in 5 of 6 cells (sixth: r0 J_U, 0.0225 vs 0.0137,
  p 0.217); permutation p ≥ 0.074 everywhere. Mean ΔJ_H is positive against `none`, negative against `none_cycled`. The r0
  J_R shift of all 12 branches (−0.020 to −0.040) matches `none_rerun` − `none` (−0.020): it comes from the `none` run.
- **Three r64 branches had zero gradient**: `related-4101`, `related-4102`, `unrelated-4103` (all 4 groups all-correct,
  `grad_norm_preclip` 0.0; C, from the diagnostic rows). Their evaluations are identical to each other problem by problem
  on all three panels (V), so their weights are identical: the step moved the weights through the stored Adam moments
  only. Their ΔJ (+0.006, −0.007, −0.008) is within the reference noise. Two of four "related" draws at r64 therefore
  contain no related-practice gradient.
- **Update batches at r64 (C, rows only; update attempts are not saved).** Hard batches had 7–15 of 32 attempts capped at
  8,192 (reward 0), related 0–10, unrelated 0. A hard update at r64 therefore penalised long responses at a cap below the
  12,288 the source was trained with.
- **Same policy, two measurements.** R-s101 r64: in-run J_H 0.738 (n 8, cap 12,288, trunc 0.215) vs diagnostic `none`
  0.605 (n 16, cap 8,192, trunc 0.371). Diagnostic r0 `none` (J_H 0.400, J_R 0.828, J_U 0.807) against the six in-run
  round-0 evaluations of the same base weights (J_H 0.410–0.434, J_R 0.808–0.817, J_U 0.805–0.816): below on J_H, above on
  J_R, inside on J_U.
- **What was connected and what was not.** Same trajectory: the r64 diagnostic source, the R-s101 in-run curve, R-s101
  training records and momentum-only steps (66, of which 24 in rounds 65–96). Never connected: (1) the one-step ΔJ at r64
  and any later R-s101 round (no checkpoint between 64 and 128 was retained); (2) ΔJ and loss/recovery on training prompts
  (different prompts; no shared measurement); (3) any "sustained stall": no record defines or measures one (I report reward
  plateaus only); (4) U and H states: no diagnostic exists for them. r0 equals the start state of all six runs, but no
  run's first step used the diagnostic prompts.

## 6. Loss and recovery on training prompts (V, descriptive)

Per H_train problem, presentations in round order (H: shared + extra; R/U: shared). **Loss** = ≥ 1 of 8 correct, then
all wrong at the next presentation; **recovered** = a later presentation with ≥ 1 correct. Sampling expectation = Σ (1 − p)^8,
p = the problem's rate in the same 32-round block without the scored presentation ("local") or over the run ("pooled").

| run | scope | opportunities | losses (rate) | recovered | not recovered by 128 | median rounds to recover | expected: local / pooled |
|---|---|---|---|---|---|---|---|
| H-s101 | all hard | 1,903 | 38 (0.020) | 37 | 1 | 3.0 | 40.1 / 15.9 |
| H-s101 | shared only | 934 | 14 (0.015) | 13 | 1 | 6.0 | 21.7 / 8.2 |
| R-s101 | shared | 921 | 22 (0.024) | 21 | 1 | 6.0 | 26.3 / 17.4 |
| U-s101 | shared | 921 | 29 (0.031) | 26 | 3 | 8.5 | 30.2 / 19.4 |
| H-s202 | all hard | 1,924 | 29 (0.015) | 28 | 1 | 2.0 | 32.1 / 21.9 |
| H-s202 | shared only | 944 | 12 (0.013) | 11 | 1 | 7.0 | 15.5 / 10.9 |
| R-s202 | shared | 932 | 25 (0.027) | 23 | 2 | 7.0 | 23.5 / 14.2 |
| U-s202 | shared | 925 | 23 (0.025) | 22 | 1 | 5.5 | 28.2 / 16.1 |

- Observed / expected (local) = 0.65–1.07 in every row: the losses are consistent with binomial sampling of 8 attempts at
  these rates; the data show no excess that would indicate a separate forgetting process. (Pooled is lower: early rounds.)
- No arm order holds across seeds: U > R in s101 (0.031 vs 0.024), R > U in s202 (0.027 vs 0.025). H is lowest in both.
- Every H_train problem was solved at least once in every run. Episodes, with rounds and cap-boundary flags:
  `train_loss_recovery.json`.

## 7. Costs (V for sacct; categorisation is my rule)

`bin/ae slurm sacct -j <266 ledger job IDs> -D -X`, 2026-11-11 about 21:15 EST; `-D` keeps requeued instances. GPU-h =
ElapsedRaw × GPUs; CPU-h = ElapsedRaw × CPUs. Category = first matching expid prefix (`costs.py`). 4 jobs were RUNNING
(3 `qis-a26-ckpt-k*`, `qis-control-base` submitted during this audit): partial. Standbys (`qis-dbg-*`) are not in the ledger.

| category | GPU-h | CPU-h | jobs | GPU-h by final row state |
|---|---|---|---|---|
| training, main runs | 173.42 | 1,387.38 | 23 | COMPLETED 105.21, FAILED 39.65, CANCELLED 28.55 |
| training, calibration | 22.50 | 180.01 | 14 | COMPLETED 22.42, CANCELLED 0.08 |
| evaluation | 94.47 | 760.85 | 72 | COMPLETED 74.01, PREEMPTED 16.25, CANCELLED 4.21 |
| diagnostics (incl. A11 smoke rounds) | 18.07 | 153.53 | 31 | COMPLETED 15.70, FAILED 1.46, CANCELLED 0.91 |
| data / screening | 7.07 | 56.97 | 24 | COMPLETED 5.53, CANCELLED 0.86, FAILED 0.68 |
| audits | 1.44 | 11.57 | 9 | RUNNING 1.13, COMPLETED 0.31 |
| infrastructure (harness smoke, tests, env, prune, watchers) | 0.00 | 12.36 | 36 | — |
| analysis / reporting | 0.04 | 4.34 | 56 | COMPLETED 0.04 |
| uncategorised (`qis-control-base`) | 0.03 | 0.22 | 1 | RUNNING |
| **total** | **317.04** | **2,567.23** | 266 (280 allocation rows) | |

Failed / superseded overlay (GPU-h): a1 jobs 13.90; main jobs with no canonical row 3.79 (U-s202 11097811 3.75, H-s202
11112338 0.03, 8 never started); superseded rounds/evals inside canonical jobs ≥ 6.31 (trainer `t_round` + eval `seconds`);
FAILED/CANCELLED rows of other categories 8.21 (80.07 CPU-h, 28 rows; some output reused by resumable jobs, e.g.
`qis-diag-local-r64`). Total ≈ 32.2 GPU-h.

| run | all allocations | a1 | jobs without canonical rows | superseded inside canonical jobs (≥) | canonical training (≤) |
|---|---|---|---|---|---|
| H-s101 | 33.61 | 4.12 | 0.00 | 0.00 | 29.49 |
| R-s101 | 32.95 | 5.08 | 0.00 | 1.87 | 26.00 |
| U-s101 | 27.32 | 4.71 | 0.00 | 0.00 | 22.62 |
| H-s202 | 31.12 | 0.00 | 0.03 | 1.55 | 29.54 |
| R-s202 | 22.79 | 0.00 | 0.00 | 0.52 | 22.27 |
| U-s202 | 25.63 | 0.00 | 3.75 | 2.38 | 19.50 |

## 8. Traced cases (V)

**(a) Resumed run, R-s101 round 125** (`train_r0124`). Line 703 (job 11081724): hard 0.84375, practice 1.0, 7 truncated;
matches `records/superseded-11081724/`. Line 734 (job 11092945, from `round_0124`): hard 0.8125, practice 0.984375,
12 truncated; matches `records/train_r0124.jsonl.gz`; last write, so canonical. Same prompts and seeds, different attempts.
Minibatch 0 norm: 0.1765 superseded, 0.0 canonical. Exception: U-s202 round 65 was reproduced identically (counts, mean
length 4,367.18) by all three jobs resuming from `round_0064`; they diverge from round 66.

**(b) All-correct group.** R-s101 round 2, slot 9, `syn-R3-005`: rewards 8 × 1, `adv` 8 × 0.0; objective min(q·0, clip(q)·0)
= 0, so no gradient, but it counts in the 1/32 divisor. Minibatch 1 with 3 mixed groups (slots 4, 7, 14): opt_step 6,
norm 0.6175, loss 1.009e-4, `valid_tokens` 38,743 (= my sum).

**(c) Momentum-only step (earliest).** U-s202 round 15 (step `round` 14), minibatch 0, opt_step 57: norm 0.0, loss 0.0,
lr 2e-6, `valid_tokens` 69,351. Slots 0, 6 (`intermediate_algebra/train/00379`, `algebra/train/00288`) all wrong; slots 9,
11 (`syn-U2-008`, `syn-U3-017`) all correct; all 32 `adv` 0.0. AdamW still stepped on its stored moments (wd 0).

**(d) Ordinary mixed group.** R-s101 round 1, slot 0, `omni/02316`, rewards [0,1,0,1,0,0,0,0] (c = 2), all finished.
Recorded `adv` +1.7320468 / −0.5773489; formula with ε 1e-6 gives +1.7320468 / −0.5773489 (√3 and −1/√3 without ε).
Minibatch 1: opt_step 2, norm 0.8706, `valid_tokens` 57,072.

## 9. Anomalies and corrections to DEFINITIONS v1

1. DEFINITIONS §3.B omits that job 11081729 (R-s202) logged 3 of 4 step rows of round 20 before it died (superseded).
2. DEFINITIONS §4 item 6: the stale counter is also in R-s202 metrics step rows (opt_step +12 from round 17), not only in
   `state.json`. Severity: numbers, if `opt_step` is used as a step index.
3. `metrics.jsonl` keeps superseded rows in four runs. Any reader that does not apply the last-write rule mixes attempts
   (for example two round-128 evaluations in R-s101, two round-64 evaluations in H-s202).
4. Diagnostic: the `none` reference is an outlier against `none_rerun` (same weights, same seeds, separate process) by up to
   0.031. Any ΔJ reported against `none` alone includes this shift.
5. Diagnostic r64: three of 12 updates were momentum-only and gave identical weights (section 5).
6. Base-model in-run evaluations differ across jobs (dev 0.403–0.422; R − U at round 0 from −0.023 to 0.000).
7. Other DEFINITIONS v1 numbers agree: group totals, momentum-only counts and first rounds, H-s202 r64 dev 0.658/0.637,
   diagnostic J_H 0.606 vs 0.738, diagnostic J_H truncation 34–39% (mine 33.6–38.7%), caps, J_H = dev[:32], schedules.

## 10. Not reconstructed

- Diagnostic update batches (attempts not saved; C). Gradients and weight changes: logged norms only; momentum-only step size U.
- Stress ordering before round 128 (no R/U pair). Exact GPU time of partial rounds and checkpoint writes; standby cost.
- Why job 11082192 (H-s202) stopped after its round-64 evaluation: last event `checkpoint_started` (U).
