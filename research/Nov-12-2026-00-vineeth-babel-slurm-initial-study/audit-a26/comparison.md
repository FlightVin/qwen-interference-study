# A26 reconcile — pass A against pass B

Purpose: compare the two independent audit passes number by number wherever both computed the same quantity, and
resolve every disagreement from primary records (A26 §3). Written 2026-11-11, ~10 PM EST, by agent `reconcile`.
Terms follow `research/DEFINITIONS.md` v1. Labels: **verified** (primary records checked), **claimed**, **unknown**.

**Inputs.** Pass A: `pass-a/data/numbers.json`, `pass-a/data/a-data-schedule_audit.json` and the pass-A markdown files.
Pass B: `pass-b/data/*.json`. Report v2: `reports/archive/REPORT-v2.md`.
**Tools** (`tools/reconcile/`): `compare.py` (laptop, stdlib; output `compare.json`), `costs_reconcile.py` (laptop,
stdlib; output `costs_reconcile.json`), `sens_heldout.py` and `sens_multibox.py` (compute, standby 11122601; outputs
`sens_*.json`, see `discrepancies.md` §3). Agreement rule: counts and point estimates must be equal within 1e-9 (both
passes compute exact ratios from the same records); interval bounds agree when they differ by ≤ 0.002 (the passes used
different bootstrap RNG seeds: pass A `default_rng(20261111)`, pass B 20261111 for evaluations and 26111111 for training).

## 1. Result

| Comparison set | Comparisons | Agree | Largest interval-bound difference |
|---|---|---|---|
| Scripted (`compare.py`, appendix A) | 161 | 157 | 0.0013 (stress R − U s101 Harder, upper bound) |
| Manual (§2, rows M1–M20) | 20 | 20 after resolution | — |

- Every count and every point estimate that both passes computed is identical: 12 sealed-test cells, 16 test contrasts
  and two-seed means, 4 common-GPU reruns with their identity counts, the truncation decomposition, coverage, 9 stress
  cells, 18 stress tier rows, 7 stress contrasts with both decomposition terms, 4 tier contrasts, 10 length rows, 24
  length contrasts, cue cells and gains, audit denominators, 40 in-run R − U values, H-s202 dev, group totals,
  momentum-only steps (count, first round and the exact set of (round, minibatch) pairs in all six runs), and the
  diagnostic means.
- The 4 scripted disagreements are one convention difference, not a data difference (row M1).
- Outside the scripted set, three apparent conflicts were resolved: the tranche-1 interval, pass A against production
  (M2, Monte-Carlo); pass A's two statements about `diag_local.py` revisions (M3, settled by `git diff`); and the
  "33 genuine failures" rule (M4, same rule gives the same count). Pass B's per-run canonical GPU-h differs from report
  v2, not from pass A (M7).
- No disagreement remains. No pass made an arithmetic error that changed a number.

## 2. Comparison and resolution table

Pass A / pass B values are unrounded where they matter; points = percentage points. "Same" means identical to the
displayed precision and, in `compare.json`, to 1e-9.

| # | Quantity | Pass A | Pass B | Agree? | Resolution |
|---|---|---|---|---|---|
| S1 | Six r128 endpoints, base, r116 fallback, 4 cgpu reruns (pass@1, pass@8, solved ≥ 1 of 32, truncated) | base 0.5256 / 0.8352 / 116 / 2; H-s101 0.7695; R-s101 0.8186; U-s101 0.7769; H-s202 0.7239; R-s202 0.7478; U-s202 0.7922; fb 0.7432; cgpu R/U s101 0.8225 / 0.7781, s202 0.7612 / 0.7922 | same | yes | — |
| S2 | Per-seed contrasts (95% paired problem bootstrap) | R−U s101 +0.0417 [+0.0203, +0.0632]; s202 −0.0444 [−0.0723, −0.0178]; R−H +0.0491 / +0.0239 [−0.0042, +0.0532]; U−H +0.0073 [−0.0193, +0.0356] / +0.0684 | same estimates; s202 R−U [−0.0728, −0.0176]; R−H s202 [−0.0049, +0.0532] | yes | Bounds differ ≤ 0.0012: Monte-Carlo error. |
| S3 | Two-seed means (point) | R−U −0.0013, R−H +0.0365, U−H +0.0378 | same | yes | Pass A's interval resamples problems only; pass B adds the df-1 paired t (R−U ± 0.548, R−H ± 0.160, U−H ± 0.388). Both say it carries no seed variance. |
| S4 | Common-GPU reruns: identical token sequences / problems with equal c / rerun − original | 48/62, 133/46, 1,964/79, 4,096/128; R-s202 +0.0134 [+0.0015, +0.0251] | same; R-s202 [+0.0022, +0.0256] | yes | Both intervals exclude 0 for R-s202 (L40 → L40S). |
| S5 | Truncation decomposition R − U | s101 +0.0141 / +0.0277; s202 −0.0431 / −0.0013 | same | yes | — |
| S6 | Coverage, s202 R − U | 122 both-solved problems carry −0.0422; one-arm 3 carry −0.0022 | 122 / −0.0422; A-only +0.0007, B-only −0.0029 | yes | — |
| S7 | Stress cells, tiers, contrasts, terms | 9 cells × 6 values; 18 tier rows; 7 contrasts | same | yes | Bounds differ ≤ 0.0013. |
| S8 | Length extension rows and contrasts | 10 rows (e.g. R-s202 test 538 / 87 / 104 / 347; 0.7478 → 0.7690); 24 contrast cells | same | yes | Bounds differ ≤ 0.0012. |
| S9 | Cue cells and difference-in-differences | s202 cells 0.6198 / 0.5729 / 0.6458 / 0.6927; DiD +0.0208 / −0.0938 | same | yes | Pass A did not re-bootstrap the cue intervals; pass B did ([−0.0625, +0.1146], [−0.1927, −0.0156]). |
| S10 | Audit denominators | 96 / 192 / 35 / 13 | same | yes | See M4 for 33. |
| S11 | In-run R − U, 2 seeds × 4 panels × 5 rounds | 40 values (e.g. s202 dev −0.0189, −0.0118, −0.0660, −0.0377, −0.0472) | same | yes | — |
| S12 | H-s202 (a2-r60) dev r64 / r96 / r128 | 0.6368 / 0.6604 / 0.6698 | same | yes | a2-orig r64 0.6580 in both (pass B line 360; pass A trace row 360). |
| S13 | Group totals (all-wrong / mixed / all-correct) | e.g. H-s101 100 / 1,289 / 659 … U-s202 54 / 947 / 1,047 | same | yes | — |
| S14 | Momentum-only steps (count, first round) | 18 (70), 66 (38), 53 (27), 20 (73), 25 (59), 65 (15) | same, and the same (round, minibatch) set | yes | — |
| S15 | Diagnostic ΔJ_H means vs `none`; `none_rerun` − `none` | r0 +0.0142 / +0.0269 / +0.0049, ref +0.0312; r64 +0.0010 / +0.0103 / +0.0034, ref +0.0254 | same | yes | — |
| M1 | Earliest observed ordering, s202 dev / J_H / J_U (J_R) | round 0 (holds from 0; J_R holds from 64) | round 32 (J_R 64) | **no → resolved** | Same data (S11). Pass A counts round 0; pass B counts trained evaluations only. Round 0 evaluates identical base weights with identical seeds in both arms, so its R − U (−0.023 to 0.000 across panels and seeds) is cross-job sampling, not an ordering of trained policies. Rule adopted: earliest observed ordering = first **trained** evaluation round (32–128) with the round-128 sign; report round 0 beside it as the noise reference. s202: r32 for dev, J_H, J_U; r64 for J_R. s101: r64 for dev, J_H, J_U; r128 for J_R. Every r32 s202 interval includes 0. |
| M2 | Stress tranche-1, s202 R − U at 12,288, interval | [−0.086, +0.047]; production [−0.090, +0.043] | not computed | resolved | Reconcile reran the bootstrap with 8 seeds on `length-diag/*/t1/rows.jsonl.gz` (64 problems × 4 attempts): lower bound −0.0859 or −0.0898, upper +0.0430 or +0.0469. The statistic lives on a 1/256 lattice; both reported intervals are valid draws. Monte-Carlo, not a defect. |
| M3 | `diag_local.py` revisions | `code.md`: identical for r0 (594d408) and r64 (4b47386); `eval-map.md`: "594d408 adds 21 lines" | — | pass A internal conflict → resolved | `git diff 4b47386 594d408 -- src/qis/diag_local.py` is empty. 042cebd (reference job) changes it by 22 lines (`--reference-only`). `code.md` is right; `eval-map.md` misattributes the 042cebd change. |
| M4 | "33 genuine failures" in the audit | 33 (both reads `grader_issue` excluded) | 33 under the both-reads rule; 31 under an any-read rule | yes | Same rule gives the same count. Name the rule wherever 33 is printed. |
| M5 | Multiple-box answers | stress lower bound (`00152` pattern, last four boxes contain 2 and 4): H-s101 4, R-s202 3, R-s101 2, U-s101 2, H-s202 2, R-s101 r64 5, H-s202 r64 1 | sealed test, earlier box = gold string: 5, 8, 0, 1, 1, 1, 1 (17 of 28,672) | different quantities | Reconcile reproduced pass B's 17 exactly and extended both with the project verifier (`discrepancies.md` §3.2). |
| M6 | Costs | not computed | 317.04 GPU-h, 2,567.23 CPU-h, 266 jobs, 280 allocations | resolved against report v2 (315.6 / 2,555.5 / 260 / 274) | §3 below. |
| M7 | Per-run "canonical" training GPU-h | not computed | ≤ 29.49 (H-s101), 26.00 (R-s101), 22.62 (U-s101), 29.54 (H-s202), 22.27 (R-s202), 19.50 (U-s202) | B vs report differ | Report v2 "of which canonical" (29.5, 27.9, 22.6, 31.1, 22.8, 25.6) counts whole jobs of the chain, including superseded rounds and, for U-s202, job 11097811 (3.75 GPU-h) that wrote no canonical row. Pass B's figure is the better definition (upper bound). Discrepancy N6. |
| M8 | H-s202 "r64" stress checkpoint | a2-r60, verified: `round_0064` saved by job 11112377 at 2026-11-11 03:26:38Z, `37963fac…`; a2-orig left only `round_0064.tmp` | same hash and save time verified; trajectory claimed | yes | Pass A's `resume_audit_a17.txt` evidence settles the trajectory: a2-r60. |
| M9 | H-s202 endpoint and stress shard GPUs | test k0–k2 L40S, k3–k5 RTX A6000; stress k0, k2 L40, k1 A6000 | same | yes | DEFINITIONS v1 §4 item 12 is wrong about k2 (v1.1 change C1). |
| M10 | Training GPU per segment | H-s101 L40; R-s101, U-s101, U-s202 L40S; H-s202 L40 r1–60 → L40S r61–128; R-s202 L40 r1–16 → L40S r17–128 | same | yes | New finding (discrepancy I8). |
| M11 | Segments and superseded rows | row totals H-s202 528/132, R-s101 528/132, R-s202 527/131, U-s202 644/161 | same; adds that job 11081729 wrote 3 of 4 step rows of round 20 | yes | 527 = 512 + 12 (rounds 17–19) + 3 (round 20): pass A's count contains pass B's 3 rows. |
| M12 | Schedule checks | shared IDs and request seeds 128/128 per seed; R/U pair index 128/128; shared presentations 20–23; H total 40–48 (s101), 41–50 (s202) | 128/128; 1,024/1,024 practice slots; 20–23; 40–50 | yes | — |
| M13 | Minibatch grouping | H and R share pair grouping in 0/128 (s101), 2/128 (s202); `valid_tokens` 512/512 | `valid_tokens` 512/512 plus superseded step rows | yes | DEFINITIONS §4 item 4 verified on data. |
| M14 | Verifier-unknown | 2, base sealed test (`intermediate_algebra/test/00030`, samples 8, 29) | 2, base | yes | Effect bound in `discrepancies.md` §3.3. |
| M15 | Diagnostic no-update J_H at r64 | 0.60546875 → 60.5 (report 60.6 wrong) | 0.605 | yes | Report v2 and DEFINITIONS v1 (0.606) round it wrongly. |
| M16 | Diagnostic J_H truncation at r64 | 2,771 / 7,680 = 36.1% (all branches pooled) | 33.6–38.7% per branch; `none` 37.1% | yes | Different aggregations of the same records. |
| M17 | Diagnostic reference noise | `none` vs rerun compares different GPUs (r0 L40 vs L40S; r64 RTX 6000 Ada vs L40S) | "same weights, separate process" differs by up to 0.031; ΔJ_H sign flips between `none` and `none_cycled` | yes, complementary | Both true: the reference gap includes a hardware change (`none_cycled` = `none_rerun` exactly). |
| M18 | Common-GPU U reruns | preempted; records written by an earlier allocation (p5-16, p9-20); last allocation 9 s / 37 s | same | yes | — |
| M19 | Loss and recovery on training prompts | production transcription only (report 2.5–4.9% PS → all-wrong) | own definition: loss rate 0.013–0.031; observed / expected (sampling) 0.65–1.07 | not the same quantity | No conflict. Pass B's sampling comparison is new evidence (discrepancy I6). |
| M20 | Training reward per role, equal-exposure comparison | not computed | e.g. hard success rounds 97–128: H 0.870 / 0.854, R 0.825 / 0.792, U 0.772 / 0.814 | pass B only | Unchecked by a second pass; label **single-pass**. |

## 3. Costs: report v2 against pass B

Script `tools/reconcile/costs_reconcile.py` joined `reports/tables/sacct.txt` (report v2's input) with pass B's
`train_costs.json` rows by job ID (output `costs_reconcile.json`).

| Set | Jobs | Allocations | GPU-h | CPU-h |
|---|---|---|---|---|
| Report v2 (`sacct.txt`, read 2026-11-11 23:45Z) | 260 | 274 | 315.578 | 2,555.494 |
| Pass B (`sacct -D`, read ~21:15 EST) | 266 | 280 | 317.043 | 2,567.232 |
| Pass B restricted to the report's 260 jobs | 260 | 274 | 315.578 | 2,555.494 |
| The 6 extra jobs | 6 | 6 | 1.464 | 11.737 |

The 6 extra jobs are all A26 audit work submitted after report v2: `qis-a26-ckpt-list` 11122475 (0 GPU-h, COMPLETED),
`qis-a26-ckpt-k0` 11122477, `-k1` 11122479, `-k2` 11122480, `-k3` 11122481 (0.31–0.38 GPU-h each; three were RUNNING),
and `qis-control-base` 11122684 (0.03 GPU-h, RUNNING). On the 260 common jobs the two passes agree to the second
(every per-job GPU-h, CPU-h and allocation count equal). **Resolution:** the report's totals are right for the study
before A26. Excluding the A26 jobs leaves 315.6 GPU-h and 2,555.5 CPU-h over 260 jobs. Pass B's "audits" category (9
jobs, 1.44 GPU-h) also contains 4 pre-A26 jobs (`qis-a8-cpu` 11078552, `qis-ckpt-inspect` 11092886 and 11092924, `qis-r32-provenance` 11086589), which the
report counts under "CPU jobs" or "other GPU jobs". The report's per-run "canonical" GPU-h is a different matter (M7, N6).

## Appendix A. Scripted comparisons (`tools/reconcile/compare.json`)

| # | Group | Quantity | Pass A | Pass B | Agree? | Max abs diff |
|---|---|---|---|---|---|---|
| 1 | test cell | base: pass@1, pass@8, solved>=1/32, truncated | 0.5256, 0.8352, 116, 2 | 0.5256, 0.8352, 116, 2 | yes | 0 |
| 2 | test cell | H-s101: pass@1, pass@8, solved>=1/32, truncated | 0.7695, 0.9371, 125, 445 | 0.7695, 0.9371, 125, 445 | yes | 0 |
| 3 | test cell | R-s101: pass@1, pass@8, solved>=1/32, truncated | 0.8186, 0.9613, 125, 437 | 0.8186, 0.9613, 125, 437 | yes | 0 |
| 4 | test cell | U-s101: pass@1, pass@8, solved>=1/32, truncated | 0.7769, 0.9446, 125, 501 | 0.7769, 0.9446, 125, 501 | yes | 0 |
| 5 | test cell | H-s202: pass@1, pass@8, solved>=1/32, truncated | 0.7239, 0.9076, 122, 632 | 0.7239, 0.9076, 122, 632 | yes | 0 |
| 6 | test cell | R-s202: pass@1, pass@8, solved>=1/32, truncated | 0.7478, 0.9221, 123, 538 | 0.7478, 0.9221, 123, 538 | yes | 0 |
| 7 | test cell | U-s202: pass@1, pass@8, solved>=1/32, truncated | 0.7922, 0.9485, 124, 333 | 0.7922, 0.9485, 124, 333 | yes | 0 |
| 8 | test cell | H-s202 r116 fallback: pass@1, pass@8, truncation share | 0.7432, 0.9281, 0.1477 | 0.7432, 0.9281, 0.1477 | yes | 0 |
| 9 | test cell | cgpu R-s101: pass@1 | 0.8225 | 0.8225 | yes | 0 |
| 10 | test cell | cgpu U-s101: pass@1 | 0.7781 | 0.7781 | yes | 0 |
| 11 | test cell | cgpu R-s202: pass@1 | 0.7612 | 0.7612 | yes | 0 |
| 12 | test cell | cgpu U-s202: pass@1 | 0.7922 | 0.7922 | yes | 0 |
| 13 | test contrast | R-U s101 [95%] | +0.0417 [+0.0203, +0.0632] | +0.0417 [+0.0203, +0.0632] | yes | 0 |
| 14 | test contrast | R-U s202 [95%] | -0.0444 [-0.0723, -0.0178] | -0.0444 [-0.0728, -0.0176] | yes | 0.0004822 |
| 15 | test contrast | R-U two-seed mean (point) | -0.0013 | -0.0013 | yes | 0 |
| 16 | test contrast | R-H s101 [95%] | +0.0491 [+0.0195, +0.0793] | +0.0491 [+0.0208, +0.0791] | yes | 0.001221 |
| 17 | test contrast | R-H s202 [95%] | +0.0239 [-0.0042, +0.0532] | +0.0239 [-0.0049, +0.0532] | yes | 0.0007324 |
| 18 | test contrast | R-H two-seed mean (point) | 0.0365 | 0.0365 | yes | 0 |
| 19 | test contrast | U-H s101 [95%] | +0.0073 [-0.0193, +0.0356] | +0.0073 [-0.0193, +0.0359] | yes | 0.0002441 |
| 20 | test contrast | U-H s202 [95%] | +0.0684 [+0.0359, +0.1023] | +0.0684 [+0.0356, +0.1023] | yes | 0.0002502 |
| 21 | test contrast | U-H two-seed mean (point) | 0.0378 | 0.0378 | yes | 0 |
| 22 | test contrast | cgpu R-U s101 [95%] | +0.0444 [+0.0242, +0.0647] | +0.0444 [+0.0244, +0.0657] | yes | 0.0009766 |
| 23 | test contrast | cgpu R-U s202 [95%] | -0.0310 [-0.0596, -0.0037] | -0.0310 [-0.0596, -0.0042] | yes | 0.0004883 |
| 24 | test contrast | H-s101 - base (point) | 0.2439 | 0.2439 | yes | 0 |
| 25 | test contrast | R-s101 - base (point) | 0.2930 | 0.2930 | yes | 0 |
| 26 | test contrast | U-s101 - base (point) | 0.2512 | 0.2512 | yes | 0 |
| 27 | test contrast | H-s202 - base (point) | 0.1982 | 0.1982 | yes | 0 |
| 28 | test contrast | R-s202 - base (point) | 0.2222 | 0.2222 | yes | 0 |
| 29 | test contrast | U-s202 - base (point) | 0.2666 | 0.2666 | yes | 0 |
| 30 | cgpu rerun | R-s101: identical token sequences, problems with equal c | 48, 62 | 48, 62 | yes | 0 |
| 31 | cgpu rerun | R-s101: rerun - original [95%] | +0.0039 [-0.0081, +0.0156] | +0.0039 [-0.0081, +0.0161] | yes | 0.0004883 |
| 32 | cgpu rerun | R-s202: identical token sequences, problems with equal c | 133, 46 | 133, 46 | yes | 0 |
| 33 | cgpu rerun | R-s202: rerun - original [95%] | +0.0134 [+0.0015, +0.0251] | +0.0134 [+0.0022, +0.0256] | yes | 0.0007324 |
| 34 | cgpu rerun | U-s101: identical token sequences, problems with equal c | 1,964, 79 | 1,964, 79 | yes | 0 |
| 35 | cgpu rerun | U-s101: rerun - original [95%] | +0.0012 [-0.0093, +0.0120] | +0.0012 [-0.0095, +0.0115] | yes | 0.0004883 |
| 36 | cgpu rerun | U-s202: identical token sequences, problems with equal c | 4,096, 128 | 4,096, 128 | yes | 0 |
| 37 | decomposition | R-U s101: tR, tU, aR, aU, trunc term, acc term | 0.1067, 0.1223, 0.9164, 0.8851, 0.0141, 0.0277 | 0.1067, 0.1223, 0.9164, 0.8851, 0.0141, 0.0277 | yes | 0 |
| 38 | decomposition | R-U s202: tR, tU, aR, aU, trunc term, acc term | 0.1313, 0.0813, 0.8609, 0.8623, -0.0431, -0.0013 | 0.1313, 0.0813, 0.8609, 0.8623, -0.0431, -0.0013 | yes | 0 |
| 39 | coverage | R-U s202: both-solved problems, their contribution | 122, -0.0422 | 122, -0.0422 | yes | 0 |
| 40 | stress cell | base: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.3727, 0.2579, 0.5250, 0.0006, 0.3729, 152 | 0.3727, 0.2579, 0.5250, 0.0006, 0.3729, 152 | yes | 0 |
| 41 | stress tier | base harder: correct, truncated, solved>=1/8, acc finished | 399, 1, 0.8211, 0.5257 | 399, 1.0000, 0.8211, 0.5257 | yes | 0 |
| 42 | stress tier | base superhard: correct, truncated, solved>=1/8, acc finished | 260, 0, 0.5873, 0.2579 | 260, 0.0000, 0.5873, 0.2579 | yes | 0 |
| 43 | stress cell | H-s101-r128: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6414, 0.5258, 0.7947, 0.1770, 0.7794, 183 | 0.6414, 0.5258, 0.7947, 0.1770, 0.7794, 183 | yes | 0 |
| 44 | stress tier | H-s101-r128 harder: correct, truncated, solved>=1/8, acc finished | 604, 69, 0.9263, 0.8741 | 604, 69, 0.9263, 0.8741 | yes | 0 |
| 45 | stress tier | H-s101-r128 superhard: correct, truncated, solved>=1/8, acc finished | 530, 244, 0.7540, 0.6937 | 530, 244, 0.7540, 0.6937 | yes | 0 |
| 46 | stress cell | R-s101-r128: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6951, 0.5843, 0.8421, 0.2048, 0.8741, 184 | 0.6951, 0.5843, 0.8421, 0.2048, 0.8741, 184 | yes | 0 |
| 47 | stress tier | R-s101-r128 harder: correct, truncated, solved>=1/8, acc finished | 640, 75, 0.9474, 0.9343 | 640, 75, 0.9474, 0.9343 | yes | 0 |
| 48 | stress tier | R-s101-r128 superhard: correct, truncated, solved>=1/8, acc finished | 589, 287, 0.7460, 0.8169 | 589, 287, 0.7460, 0.8169 | yes | 0 |
| 49 | stress cell | U-s101-r128: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6533, 0.5417, 0.8013, 0.2296, 0.8480, 184 | 0.6533, 0.5417, 0.8013, 0.2296, 0.8480, 184 | yes | 0 |
| 50 | stress tier | U-s101-r128 harder: correct, truncated, solved>=1/8, acc finished | 609, 87, 0.9474, 0.9049 | 609, 87, 0.9474, 0.9049 | yes | 0 |
| 51 | stress tier | U-s101-r128 superhard: correct, truncated, solved>=1/8, acc finished | 546, 319, 0.7460, 0.7925 | 546, 319, 0.7460, 0.7925 | yes | 0 |
| 52 | stress cell | H-s202-r128: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6012, 0.4940, 0.7434, 0.2551, 0.8071, 177 | 0.6012, 0.4940, 0.7434, 0.2551, 0.8071, 177 | yes | 0 |
| 53 | stress tier | H-s202-r128 harder: correct, truncated, solved>=1/8, acc finished | 565, 115, 0.9263, 0.8760 | 565, 115, 0.9263, 0.8760 | yes | 0 |
| 54 | stress tier | H-s202-r128 superhard: correct, truncated, solved>=1/8, acc finished | 498, 336, 0.7063, 0.7411 | 498, 336, 0.7063, 0.7411 | yes | 0 |
| 55 | stress cell | R-s202-r128: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6318, 0.5327, 0.7632, 0.2302, 0.8207, 180 | 0.6318, 0.5327, 0.7632, 0.2302, 0.8207, 180 | yes | 0 |
| 56 | stress tier | R-s202-r128 harder: correct, truncated, solved>=1/8, acc finished | 580, 102, 0.9368, 0.8815 | 580, 102, 0.9368, 0.8815 | yes | 0 |
| 57 | stress tier | R-s202-r128 superhard: correct, truncated, solved>=1/8, acc finished | 537, 305, 0.7222, 0.7639 | 537, 305, 0.7222, 0.7639 | yes | 0 |
| 58 | stress cell | U-s202-r128: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6584, 0.5556, 0.7947, 0.1516, 0.7760, 188 | 0.6584, 0.5556, 0.7947, 0.1516, 0.7760, 188 | yes | 0 |
| 59 | stress tier | U-s202-r128 harder: correct, truncated, solved>=1/8, acc finished | 604, 70, 0.9579, 0.8754 | 604, 70, 0.9579, 0.8754 | yes | 0 |
| 60 | stress tier | U-s202-r128 superhard: correct, truncated, solved>=1/8, acc finished | 560, 198, 0.7698, 0.6914 | 560, 198, 0.7698, 0.6914 | yes | 0 |
| 61 | stress cell | R-s101-r64: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6425, 0.5347, 0.7855, 0.2613, 0.8698, 183 | 0.6425, 0.5347, 0.7855, 0.2613, 0.8698, 183 | yes | 0 |
| 62 | stress tier | R-s101-r64 harder: correct, truncated, solved>=1/8, acc finished | 597, 111, 0.9368, 0.9199 | 597, 111, 0.9368, 0.9199 | yes | 0 |
| 63 | stress tier | R-s101-r64 superhard: correct, truncated, solved>=1/8, acc finished | 539, 351, 0.7460, 0.8204 | 539, 351, 0.7460, 0.8204 | yes | 0 |
| 64 | stress cell | H-s202-r64: pass@1, SuperHard, Harder, trunc, acc finished, solved/221 | 0.6239, 0.5208, 0.7605, 0.1510, 0.7348, 183 | 0.6239, 0.5208, 0.7605, 0.1510, 0.7348, 183 | yes | 0 |
| 65 | stress tier | H-s202-r64 harder: correct, truncated, solved>=1/8, acc finished | 578, 66, 0.9368, 0.8329 | 578, 66, 0.9368, 0.8329 | yes | 0 |
| 66 | stress tier | H-s202-r64 superhard: correct, truncated, solved>=1/8, acc finished | 525, 201, 0.7460, 0.6506 | 525, 201, 0.7460, 0.6506 | yes | 0 |
| 67 | stress contrast | R-U s101 all [95%] | +0.0419 [+0.0158, +0.0690] | +0.0419 [+0.0153, +0.0679] | yes | 0.001131 |
| 68 | stress contrast | R-U s101 all: trunc term, acc term | 0.0214, 0.0204 | 0.0214, 0.0204 | yes | 0 |
| 69 | stress contrast | R-U s202 all [95%] | -0.0266 [-0.0503, -0.0023] | -0.0266 [-0.0503, -0.0017] | yes | 0.0005656 |
| 70 | stress contrast | R-U s202 all: trunc term, acc term | -0.0628, 0.0362 | -0.0628, 0.0362 | yes | 0 |
| 71 | stress contrast | R-H s101 all [95%] | +0.0537 [+0.0266, +0.0826] | +0.0537 [+0.0260, +0.0814] | yes | 0.001131 |
| 72 | stress contrast | R-H s101 all: trunc term, acc term | -0.0229, 0.0766 | -0.0229, 0.0766 | yes | 0 |
| 73 | stress contrast | U-H s101 all [95%] | +0.0119 [-0.0141, +0.0385] | +0.0119 [-0.0147, +0.0385] | yes | 0.0005798 |
| 74 | stress contrast | U-H s101 all: trunc term, acc term | -0.0428, 0.0547 | -0.0428, 0.0547 | yes | 0 |
| 75 | stress contrast | R-H s202 all [95%] | +0.0305 [-0.0017, +0.0633] | +0.0305 [-0.0006, +0.0628] | yes | 0.001131 |
| 76 | stress contrast | R-H s202 all: trunc term, acc term | 0.0203, 0.0103 | 0.0203, 0.0103 | yes | 0 |
| 77 | stress contrast | U-H s202 all [95%] | +0.0571 [+0.0243, +0.0911] | +0.0571 [+0.0238, +0.0911] | yes | 0.0005656 |
| 78 | stress contrast | U-H s202 all: trunc term, acc term | 0.0819, -0.0248 | 0.0819, -0.0248 | yes | 0 |
| 79 | stress contrast | R-s101-r128 - R-s101-r64 all [95%] | +0.0526 [+0.0300, +0.0758] | +0.0526 [+0.0305, +0.0758] | yes | 0.0005656 |
| 80 | stress contrast | R-s101-r128 - R-s101-r64 all: trunc term, acc term | 0.0493, 0.0033 | 0.0493, 0.0033 | yes | 0 |
| 81 | stress contrast | H-s202 r128 - r64 (point) | -0.0226 | -0.0226 | yes | 3.123e-17 |
| 82 | stress contrast | R-U s101 harder [95%] | +0.0408 [+0.0053, +0.0776] | +0.0408 [+0.0053, +0.0763] | yes | 0.001316 |
| 83 | stress contrast | R-U s101 superhard [95%] | +0.0427 [+0.0040, +0.0794] | +0.0427 [+0.0050, +0.0804] | yes | 0.0009921 |
| 84 | stress contrast | R-U s202 harder [95%] | -0.0316 [-0.0671, +0.0013] | -0.0316 [-0.0671, +0.0026] | yes | 0.001316 |
| 85 | stress contrast | R-U s202 superhard [95%] | -0.0228 [-0.0556, +0.0119] | -0.0228 [-0.0565, +0.0109] | yes | 0.0009921 |
| 86 | length row | test H-s101: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 445, 93, 141, 211, 0.7695, 0.7922 | 445, 93, 141, 211, 0.7695, 0.7922 | yes | 0 |
| 87 | length row | test R-s101: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 437, 94, 50, 293, 0.8186, 0.8416 | 437, 94, 50, 293, 0.8186, 0.8416 | yes | 0 |
| 88 | length row | test U-s101: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 501, 118, 67, 316, 0.7769, 0.8057 | 501, 118, 67, 316, 0.7769, 0.8057 | yes | 0 |
| 89 | length row | test R-s202: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 538, 87, 104, 347, 0.7478, 0.7690 | 538, 87, 104, 347, 0.7478, 0.7690 | yes | 0 |
| 90 | length row | test U-s202: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 333, 66, 65, 202, 0.7922, 0.8083 | 333, 66, 65, 202, 0.7922, 0.8083 | yes | 0 |
| 91 | length row | stress H-s101: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 313, 55, 126, 132, 0.6414, 0.6725 | 313, 55, 126, 132, 0.6414, 0.6725 | yes | 0 |
| 92 | length row | stress R-s101: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 362, 49, 61, 252, 0.6951, 0.7229 | 362, 49, 61, 252, 0.6951, 0.7229 | yes | 0 |
| 93 | length row | stress U-s101: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 406, 51, 64, 291, 0.6533, 0.6821 | 406, 51, 64, 291, 0.6533, 0.6821 | yes | 0 |
| 94 | length row | stress R-s202: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 407, 50, 96, 261, 0.6318, 0.6601 | 407, 50, 96, 261, 0.6318, 0.6601 | yes | 0 |
| 95 | length row | stress U-s202: capped, recovered, finished wrong, still capped, pass@1 12k, 16k | 268, 41, 81, 146, 0.6584, 0.6816 | 268, 41, 81, 146, 0.6584, 0.6816 | yes | 0 |
| 96 | length contrast | test R-s101 - U-s101 at 12,288 [95%] | +0.0417 [+0.0210, +0.0627] | +0.0417 [+0.0203, +0.0632] | yes | 0.0007324 |
| 97 | length contrast | test R-s101 - U-s101 at 16,384 [95%] | +0.0359 [+0.0149, +0.0569] | +0.0359 [+0.0151, +0.0571] | yes | 0.0002441 |
| 98 | length contrast | test R-s101 - U-s101 change [95%] | -0.0059 [-0.0164, +0.0037] | -0.0059 [-0.0159, +0.0037] | yes | 0.0004883 |
| 99 | length contrast | test R-s202 - U-s202 at 12,288 [95%] | -0.0444 [-0.0715, -0.0178] | -0.0444 [-0.0728, -0.0176] | yes | 0.001221 |
| 100 | length contrast | test R-s202 - U-s202 at 16,384 [95%] | -0.0393 [-0.0667, -0.0134] | -0.0393 [-0.0671, -0.0134] | yes | 0.0004883 |
| 101 | length contrast | test R-s202 - U-s202 change [95%] | +0.0051 [-0.0029, +0.0129] | +0.0051 [-0.0029, +0.0129] | yes | 0 |
| 102 | length contrast | test R-s101 - H-s101 at 12,288 [95%] | +0.0491 [+0.0198, +0.0793] | +0.0491 [+0.0208, +0.0791] | yes | 0.0009827 |
| 103 | length contrast | test R-s101 - H-s101 at 16,384 [95%] | +0.0493 [+0.0212, +0.0801] | +0.0493 [+0.0215, +0.0793] | yes | 0.0007324 |
| 104 | length contrast | test R-s101 - H-s101 change [95%] | +0.0002 [-0.0078, +0.0085] | +0.0002 [-0.0078, +0.0085] | yes | 0 |
| 105 | length contrast | test U-s101 - H-s101 at 12,288 [95%] | +0.0073 [-0.0200, +0.0354] | +0.0073 [-0.0193, +0.0359] | yes | 0.0007324 |
| 106 | length contrast | test U-s101 - H-s101 at 16,384 [95%] | +0.0134 [-0.0110, +0.0388] | +0.0134 [-0.0105, +0.0396] | yes | 0.0007385 |
| 107 | length contrast | test U-s101 - H-s101 change [95%] | +0.0061 [-0.0024, +0.0154] | +0.0061 [-0.0022, +0.0151] | yes | 0.0002441 |
| 108 | length contrast | stress R-s101 - U-s101 at 12,288 [95%] | +0.0419 [+0.0158, +0.0679] | +0.0419 [+0.0153, +0.0679] | yes | 0.0005656 |
| 109 | length contrast | stress R-s101 - U-s101 at 16,384 [95%] | +0.0407 [+0.0153, +0.0673] | +0.0407 [+0.0141, +0.0673] | yes | 0.001117 |
| 110 | length contrast | stress R-s101 - U-s101 change [95%] | -0.0011 [-0.0113, +0.0096] | -0.0011 [-0.0113, +0.0096] | yes | 0 |
| 111 | length contrast | stress R-s202 - U-s202 at 12,288 [95%] | -0.0266 [-0.0503, -0.0023] | -0.0266 [-0.0503, -0.0017] | yes | 0.0005656 |
| 112 | length contrast | stress R-s202 - U-s202 at 16,384 [95%] | -0.0215 [-0.0458, +0.0028] | -0.0215 [-0.0458, +0.0034] | yes | 0.0005656 |
| 113 | length contrast | stress R-s202 - U-s202 change [95%] | +0.0051 [-0.0051, +0.0153] | +0.0051 [-0.0051, +0.0153] | yes | 0 |
| 114 | length contrast | stress R-s101 - H-s101 at 12,288 [95%] | +0.0537 [+0.0260, +0.0820] | +0.0537 [+0.0260, +0.0814] | yes | 0.0005656 |
| 115 | length contrast | stress R-s101 - H-s101 at 16,384 [95%] | +0.0503 [+0.0238, +0.0775] | +0.0503 [+0.0232, +0.0769] | yes | 0.0005656 |
| 116 | length contrast | stress R-s101 - H-s101 change [95%] | -0.0034 [-0.0153, +0.0079] | -0.0034 [-0.0153, +0.0079] | yes | 0 |
| 117 | length contrast | stress U-s101 - H-s101 at 12,288 [95%] | +0.0119 [-0.0141, +0.0379] | +0.0119 [-0.0147, +0.0385] | yes | 0.0005798 |
| 118 | length contrast | stress U-s101 - H-s101 at 16,384 [95%] | +0.0096 [-0.0175, +0.0373] | +0.0096 [-0.0187, +0.0368] | yes | 0.001131 |
| 119 | length contrast | stress U-s101 - H-s101 change [95%] | -0.0023 [-0.0141, +0.0090] | -0.0023 [-0.0147, +0.0090] | yes | 0.0005656 |
| 120 | cue | s202 cells: R uncued, R cued, U uncued, U cued | 0.6198, 0.5729, 0.6458, 0.6927 | 0.6198, 0.5729, 0.6458, 0.6927 | yes | 0 |
| 121 | cue | cue gain R-s202, U-s202, DiD s101, DiD s202 (cue gain R - U s101, cue gain R - U s202) | -0.0469, 0.0469, 0.0208, -0.0938 | -0.0469, 0.0469, 0.0208, -0.0938 | yes | 1.492e-16 |
| 122 | audit | responses, reads, reward-0, capped reward-0 | 96, 192, 35, 13 | 96, 192, 35, 13 | yes | 0 |
| 123 | in-run R-U | s101 dev: R-U at rounds 0/32/64/96/128 | -0.0047, -0.0590, 0.0259, 0.0637, 0.0377 | -0.0047, -0.0590, 0.0259, 0.0637, 0.0377 | yes | 9.714e-17 |
| 124 | ordering | s101 dev: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 64, 64 | 64, 64 | yes | 0 |
| 125 | in-run R-U | s101 J_H: R-U at rounds 0/32/64/96/128 | -0.0156, -0.0938, 0.0039, 0.0703, 0.0625 | -0.0156, -0.0938, 0.0039, 0.0703, 0.0625 | yes | 0 |
| 126 | ordering | s101 J_H: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 64, 64 | 64, 64 | yes | 0 |
| 127 | in-run R-U | s101 J_R: R-U at rounds 0/32/64/96/128 | 0.0000, 0.0089, 0.0402, 0.0134, -0.0134 | 0.0000, 0.0089, 0.0402, 0.0134, -0.0134 | yes | 6.418e-17 |
| 128 | ordering | s101 J_R: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 128, 128 | 128, 128 | yes | 0 |
| 129 | in-run R-U | s101 J_U: R-U at rounds 0/32/64/96/128 | -0.0039, -0.0195, 0.0352, -0.0078, 0.0195 | -0.0039, -0.0195, 0.0352, -0.0078, 0.0195 | yes | 0 |
| 130 | ordering | s101 J_U: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 64, 128 | 64, 128 | yes | 0 |
| 131 | in-run R-U | s202 dev: R-U at rounds 0/32/64/96/128 | -0.0189, -0.0118, -0.0660, -0.0377, -0.0472 | -0.0189, -0.0118, -0.0660, -0.0377, -0.0472 | yes | 8.674e-17 |
| 132 | ordering | s202 dev: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 0, 0 | 32, 32 | **no** | 32 |
| 133 | in-run R-U | s202 J_H: R-U at rounds 0/32/64/96/128 | -0.0234, -0.0195, -0.0781, -0.0039, -0.0586 | -0.0234, -0.0195, -0.0781, -0.0039, -0.0586 | yes | 0 |
| 134 | ordering | s202 J_H: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 0, 0 | 32, 32 | **no** | 32 |
| 135 | in-run R-U | s202 J_R: R-U at rounds 0/32/64/96/128 | -0.0089, 0.0536, -0.0580, -0.0268, -0.0179 | -0.0089, 0.0536, -0.0580, -0.0268, -0.0179 | yes | 6.245e-17 |
| 136 | ordering | s202 J_R: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 0, 64 | 64, 64 | **no** | 64 |
| 137 | in-run R-U | s202 J_U: R-U at rounds 0/32/64/96/128 | -0.0117, -0.0156, -0.0391, -0.0820, -0.0234 | -0.0117, -0.0156, -0.0391, -0.0820, -0.0234 | yes | 0 |
| 138 | ordering | s202 J_U: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds | 0, 0 | 32, 32 | **no** | 32 |
| 139 | dev curve | H-s202 (a2-r60) dev r64, r96, r128 | 0.6368, 0.6604, 0.6698 | 0.6368, 0.6604, 0.6698 | yes | 0 |
| 140 | groups | H-s101: all-wrong, mixed, all-correct | 100, 1,289, 659 | 100, 1,289, 659 | yes | 0 |
| 141 | momentum-only | H-s101: zero-gradient steps, first round | 18, 70 | 18, 70 | yes | 0 |
| 142 | momentum-only | H-s101: identical (round, minibatch) set of zero-gradient steps | 1 | 1 | yes | 0 |
| 143 | groups | R-s101: all-wrong, mixed, all-correct | 58, 961, 1,029 | 58, 961, 1,029 | yes | 0 |
| 144 | momentum-only | R-s101: zero-gradient steps, first round | 66, 38 | 66, 38 | yes | 0 |
| 145 | momentum-only | R-s101: identical (round, minibatch) set of zero-gradient steps | 1 | 1 | yes | 0 |
| 146 | groups | U-s101: all-wrong, mixed, all-correct | 60, 930, 1,058 | 60, 930, 1,058 | yes | 0 |
| 147 | momentum-only | U-s101: zero-gradient steps, first round | 53, 27 | 53, 27 | yes | 0 |
| 148 | momentum-only | U-s101: identical (round, minibatch) set of zero-gradient steps | 1 | 1 | yes | 0 |
| 149 | groups | H-s202: all-wrong, mixed, all-correct | 79, 1,287, 682 | 79, 1,287, 682 | yes | 0 |
| 150 | momentum-only | H-s202: zero-gradient steps, first round | 20, 73 | 20, 73 | yes | 0 |
| 151 | momentum-only | H-s202: identical (round, minibatch) set of zero-gradient steps | 1 | 1 | yes | 0 |
| 152 | groups | R-s202: all-wrong, mixed, all-correct | 48, 1,100, 900 | 48, 1,100, 900 | yes | 0 |
| 153 | momentum-only | R-s202: zero-gradient steps, first round | 25, 59 | 25, 59 | yes | 0 |
| 154 | momentum-only | R-s202: identical (round, minibatch) set of zero-gradient steps | 1 | 1 | yes | 0 |
| 155 | groups | U-s202: all-wrong, mixed, all-correct | 54, 947, 1,047 | 54, 947, 1,047 | yes | 0 |
| 156 | momentum-only | U-s202: zero-gradient steps, first round | 65, 15 | 65, 15 | yes | 0 |
| 157 | momentum-only | U-s202: identical (round, minibatch) set of zero-gradient steps | 1 | 1 | yes | 0 |
| 158 | diagnostic | r0: mean ΔJ_H vs none (hard, related, unrelated) | 0.0142, 0.0269, 0.0049 | 0.0142, 0.0269, 0.0049 | yes | 0 |
| 159 | diagnostic | r0: none_rerun - none (J_H) | 0.0312 | 0.0312 | yes | 0 |
| 160 | diagnostic | r64: mean ΔJ_H vs none (hard, related, unrelated) | 0.0010, 0.0103, 0.0034 | 0.0010, 0.0103, 0.0034 | yes | 0 |
| 161 | diagnostic | r64: none_rerun - none (J_H) | 0.0254 | 0.0254 | yes | 0 |
