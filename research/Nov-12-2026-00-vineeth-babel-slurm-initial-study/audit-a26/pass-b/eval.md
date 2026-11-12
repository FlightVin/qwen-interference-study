# A26 pass B — evaluation reconstruction (agent b-eval)

Purpose: rebuild every evaluation number of the QIS study from per-attempt records, with code written for this audit.
Computed 2026-11-11 20:40–21:15 EST on CPU standby 11122432. Unrounded values are in `data/eval_*.json`; tables round to 4 decimals.

## 1. Method and independence
- Inputs: `records.jsonl.gz` (sealed test, reruns, shards, stress, cue) and `rows.jsonl.gz` (length extension) under
  `$P = /data/user_data/vineethb/qis`; panel files `$P/data/main-v1/target_test_sealed.json`, `$P/data/stress/stress_v1.json`
  (sha256 `757393be…`, equal to the repo copy), `data/cue/*.json` (sha256 equal to the recipe values).
  Response audit: `research/evidence/a17/audit/{key.json, ann_0..3.json}` and `panels/audit_sample_v1.json` only.
- Not read: any `summary.json`, `length.json`, `cue_analysis.*`, `audit_merged.json`, `audit.md`, `reports/`, `src/qis/analysis/`.
  The only production numbers seen were those printed in `research/DEFINITIONS.md` v1 (allowed input).
- Code: `tools/b-eval/` (stdlib + numpy; `trace.py` also uses the Qwen3 tokenizer to decode ids). Outputs on compute in
  `$P/audit-a26/b-eval/`, copied to `data/`. Scripts ran from `~/ae-qis/audit-a26/b-eval/` with copies of `data/cue/*.json` and `a17/audit/key.json` (as `audit_key.json`).
- Definitions: pass@1 = mean over problems of c/n (unknown status counts as incorrect). pass@8 (n = 32) = mean of the
  unbiased 1 − C(n−c,8)/C(n,8); for n = 8 it is the observed coverage at 8. Coverage at n = share of problems with c ≥ 1.
  Truncation = `finish == "length"`. Accuracy among finished a = correct / attempts not capped (pooled).
- Paired problem bootstrap: 10,000 draws with numpy seed 20261111 (pass-B seed, not the project's 2027). Each draw
  resamples problem indices with replacement and uses the same indices for both policies. The interval is the 2.5/97.5
  percentile of the mean per-problem difference.
- Decomposition (exact): with t = truncated/attempts, a = correct/finished, p = (1−t)·a,
  `p_A − p_B = −(t_A − t_B)·ā + (1 − t̄)·(a_A − a_B)`, where ā and t̄ are the means of the two policies.
  The first term is the "truncation term" and the second the "accuracy term". The residual is < 1e-16 in every row.
- Coverage vs reliability: split mean(c_A/n − c_B/n) by problem class: solved (c ≥ 1) by both, by A only, by B only,
  by neither. The "both" class carries the reliability part; "A only" + "B only" carry the coverage part.

## 2. Identity of every record set (verified)
The recipe hash equals the ledger `expsha` for every recipe checked (test, stress r64, cue, lentest, len).
The model hash comes from the job log line `model.pt sha256 recorded=… on-disk=…` or from `results/<expid>/ckpt_identity.txt`.
It equals `file_sha256["model.pt"]` in today's `state.json`. The GPU comes from `sacct` NodeList plus `scontrol show node` Gres.

| record set | checkpoint (recipe) | model.pt sha256 | state.json round / attempt | GPU of the generating run |
|---|---|---|---|---|
| test base | base | — | — | L40S (q5-20) |
| test H-s101 | `qis-main-H-s101-sym/round_0128` | `052c8a62…` | 128 / a2 | L40S |
| test R-s101 | `…R-s101-sym/round_0128` | `aedcca6e…` | 128 / a2 | **L40** (w5-24) |
| test U-s101 | `…U-s101-sym/round_0128` | `92aef0d2…` | 128 / a2 | L40S |
| test H-s202 (6 shards) | `…H-s202-sym/round_0128` | `25134562…` (all 6 logs) | 128 / a2 | k0, k1, k2 L40S; **k3, k4, k5 A6000** |
| test R-s202 | `…R-s202-sym/round_0128` | `22fd8364…` | 128 / a2 | **L40** (w5-28) |
| test U-s202 | `…U-s202-sym/round_0128` | `7158b095…` | 128 / a2 | L40S |
| test H-s202 fallback (6 shards) | `…H-s202-sym-fallback/ckpt` (`ROUND.txt` 116) | `7b9ff3a4…` (k0, k5 logs) | 116 / a2 | L40S (all 6 nodes) |
| cgpu R/U × s101/s202 | the four r128 dirs above | same four hashes | — | L40S (all four) |
| stress, 5 r128 cells except H-s202 | the r128 dirs | same hashes (`ckpt_identity.txt`) | — | L40S (`gpu.txt`; `sacct` nodes) |
| stress H-s202 r128 (3 shards) | `…H-s202-sym/round_0128` | `25134562…` | — | **k0, k2 L40; k1 A6000** |
| stress R-s101 r64 | `…R-s101-sym/round_0064` | `6fe89540…` | 64 / a2; model.pt mtime 2026-11-08 07:13 EST | L40S |
| stress H-s202 r64 | `…H-s202-sym/round_0064` | `37963fac…` | 64 / a2; `saved_utc` 2026-11-11T03:26:38Z | L40S |
| cue (5 policies) | base + four r128 dirs | hashes in job log 11104738 | — | L40S (p9-20) |
| length-test / length-diag | the five r128 dirs (no H-s202) | match in every job log | — | L40S (recipe constraint) |

- **verified:** the stress "H-s202 r64" cell is the checkpoint saved 2026-11-11 03:26Z. The stress job started at
  02:05 EST on 2026-11-11, after that save.
- **claimed:** that this checkpoint belongs to trajectory a2-r60 (DEFINITIONS v1 §3.B). I did not read training metrics.
- **verified:** each merged H-s202 file equals the union of its shards attempt by attempt (identical ids, reward, finish,
  seed): test r128 4,096/4,096, fallback 4,096/4,096, stress 1,768/1,768. Shard panels equal `items[k::6]` (stress `[k::3]`).

## 3. Sealed test (task 1)
All 12 cells: 128 problems × 32 attempts = 4,096 attempts, with 0 missing, 0 extra and 0 duplicate attempts.
Each problem has one seed, and the seed is identical across all 12 cells (0 problems differ).
Every capped attempt has exactly 12,288 tokens and reward 0. The only unknown statuses are **2, both in base**;
they count as incorrect.

| cell | pass@1 (correct/4,096) | pass@8 (unbiased) | coverage at 32 | truncation (count) | acc. among finished | missing/dup/unknown |
|---|---|---|---|---|---|---|
| base | 0.5256 (2153) | 0.8352 | 0.9062 (116/128) | 0.0005 (2) | 0.5259 (2153/4094) | 0/0/2 |
| H-s101 | 0.7695 (3152) | 0.9371 | 0.9766 (125) | 0.1086 (445) | 0.8633 (3152/3651) | 0/0/0 |
| R-s101 | 0.8186 (3353) | 0.9613 | 0.9766 (125) | 0.1067 (437) | 0.9164 (3353/3659) | 0/0/0 |
| U-s101 | 0.7769 (3182) | 0.9446 | 0.9766 (125) | 0.1223 (501) | 0.8851 (3182/3595) | 0/0/0 |
| H-s202 | 0.7239 (2965) | 0.9076 | 0.9531 (122) | 0.1543 (632) | 0.8559 (2965/3464) | 0/0/0 |
| R-s202 | 0.7478 (3063) | 0.9221 | 0.9609 (123) | 0.1313 (538) | 0.8609 (3063/3558) | 0/0/0 |
| U-s202 | 0.7922 (3245) | 0.9485 | 0.9688 (124) | 0.0813 (333) | 0.8623 (3245/3763) | 0/0/0 |
| H-s202 r116 fallback | 0.7432 (3044) | 0.9281 | 0.9609 (123) | 0.1477 (605) | 0.8720 (3044/3491) | 0/0/0 |
| cgpu R-s101 | 0.8225 (3369) | 0.9621 | 0.9766 (125) | 0.1094 (448) | 0.9235 (3369/3648) | 0/0/0 |
| cgpu R-s202 | 0.7612 (3118) | 0.9343 | 0.9609 (123) | 0.1235 (506) | 0.8685 (3118/3590) | 0/0/0 |
| cgpu U-s101 | 0.7781 (3187) | 0.9514 | 0.9688 (124) | 0.1233 (505) | 0.8875 (3187/3591) | 0/0/0 |
| cgpu U-s202 | 0.7922 (3245) | 0.9485 | 0.9688 (124) | 0.0813 (333) | 0.8623 (3245/3763) | 0/0/0 |

**Common-GPU reruns against their originals, per attempt** (same seeds: 4,096/4,096 in each pair):

| run | identical responses (token ids) | identical reward | orig 1→rerun 0 / orig 0→rerun 1 | pass@1 orig | pass@1 rerun | rerun − orig [95%] | problems with equal c |
|---|---|---|---|---|---|---|---|
| R-s101 (L40 → L40S) | 48/4096 | 3556/4096 | 262 / 278 | 0.8186 | 0.8225 | +0.0039 [−0.0081, +0.0161] | 62/128 |
| R-s202 (L40 → L40S) | 133/4096 | 3455/4096 | 293 / 348 | 0.7478 | 0.7612 | **+0.0134 [+0.0022, +0.0256]** | 46/128 |
| U-s101 (L40S → L40S) | 1964/4096 | 3691/4096 | 200 / 205 | 0.7769 | 0.7781 | +0.0012 [−0.0095, +0.0115] | 79/128 |
| U-s202 (L40S → L40S) | 4096/4096 | 4096/4096 | 0 / 0 | 0.7922 | 0.7922 | 0 | 128/128 |

The R-s202 rerun differs from its original by more than problem-and-attempt noise explains. Its interval excludes 0.
Both U reruns used L40S nodes. U-s202 reproduced every response; U-s101 reproduced 48%. The cause of that difference
is unknown. The two cgpu-U jobs show final runs of 9 s and 37 s in `sacct`. Each was preempted, and its records were
written by an earlier run (U-s101 on p5-16, U-s202 on p9-20, both L40S); the final run only skipped completed phases.

## 4. Per-seed contrasts (task 2)
Paired bootstrap per seed, 128 problems. "trunc." and "acc." are the two terms of the decomposition in §1.

| contrast | pass@1 A vs B | Δ pass@1 [95%] | Δ pass@8 [95%] | Δ truncation [95%] | trunc. term | acc. term | acc. A vs B |
|---|---|---|---|---|---|---|---|
| R−U s101 | 0.8186 vs 0.7769 | +0.0417 [+0.0203, +0.0632] | +0.0167 [−0.0004, +0.0358] | −0.0156 [−0.0308, −0.0002] | +0.0141 | +0.0277 | 0.9164 vs 0.8851 |
| R−U s202 | 0.7478 vs 0.7922 | −0.0444 [−0.0728, −0.0176] | −0.0264 [−0.0563, +0.0008] | +0.0500 [+0.0315, +0.0708] | −0.0431 | −0.0013 | 0.8609 vs 0.8623 |
| R−H s101 | 0.8186 vs 0.7695 | +0.0491 [+0.0208, +0.0791] | +0.0242 [+0.0055, +0.0458] | −0.0020 [−0.0259, +0.0217] | +0.0017 | +0.0473 | 0.9164 vs 0.8633 |
| R−H s202 | 0.7478 vs 0.7239 | +0.0239 [−0.0049, +0.0532] | +0.0145 [−0.0087, +0.0412] | −0.0229 [−0.0476, +0.0000] | +0.0197 | +0.0042 | 0.8609 vs 0.8559 |
| U−H s101 | 0.7769 vs 0.7695 | +0.0073 [−0.0193, +0.0359] | +0.0076 [−0.0112, +0.0272] | +0.0137 [−0.0081, +0.0364] | −0.0120 | +0.0193 | 0.8851 vs 0.8633 |
| U−H s202 | 0.7922 vs 0.7239 | +0.0684 [+0.0356, +0.1023] | +0.0409 [+0.0142, +0.0721] | −0.0730 [−0.0994, −0.0498] | +0.0627 | +0.0056 | 0.8623 vs 0.8559 |

Sensitivity rows (same method):

| contrast | Δ pass@1 [95%] | trunc. term | acc. term |
|---|---|---|---|
| R − H-fallback-r116, s202 | +0.0046 [−0.0220, +0.0317] | +0.0142 | −0.0095 |
| U − H-fallback-r116, s202 | +0.0491 [+0.0205, +0.0793] | +0.0576 | −0.0085 |
| cgpu R − cgpu U, s101 (both L40S) | +0.0444 [+0.0244, +0.0657] | +0.0126 | +0.0318 |
| cgpu R − cgpu U, s202 (both L40S) | −0.0310 [−0.0596, −0.0042] | −0.0366 | +0.0055 |
| endpoint − base, s101: H / R / U | +0.2439 / +0.2930 / +0.2512 (all intervals > +0.19) | — | — |
| endpoint − base, s202: H / R / U | +0.1982 / +0.2222 / +0.2666 (all intervals > +0.15) | — | — |

**Sign checks (verified):** in every row the truncation term has the sign opposite to Δt, the accuracy term has the
sign of Δa, and the terms sum to Δ pass@1. R−U changes sign between seeds: in s101 both terms favour R; in s202 the
truncation term alone is −0.0431 of −0.0444. With common-GPU reruns R−U keeps both signs (+0.0444, −0.0310).

**Two-seed mean (descriptive only):**

| contrast | s101 | s202 | mean | half-range |
|---|---|---|---|---|
| R−U | +0.0417 | −0.0444 | −0.0013 | 0.0431 |
| R−H | +0.0491 | +0.0239 | +0.0365 | 0.0126 |
| U−H | +0.0073 | +0.0684 | +0.0378 | 0.0305 |

What a two-seed interval means. Each per-seed interval covers only the resampling of problems and attempts for two
fixed trained policies. Training-run variance from two runs has 1 degree of freedom: a paired t interval is
mean ± 12.71 × half-range (±0.548 for R−U, ±0.160 for R−H). A crossed bootstrap over two seeds mostly mixes the two
per-seed values; its width shows that the seeds disagree, not the size of an arm effect. The data therefore show a
seed-dependent R−U of opposite sign (+0.042 and −0.044, each interval excluding 0), not an arm effect.

## 5. Coverage vs reliability (task 3)
Coverage at 32 is near its ceiling. Every contrast comes almost entirely from problems that both policies solve at least once.

| contrast | Δ pass@1 | both: problems / contribution | A only | B only | neither | Δ coverage at 32 [95%] |
|---|---|---|---|---|---|---|
| R−U s101 | +0.0417 | 125 / +0.0417 | 0 / 0 | 0 / 0 | 3 | 0 [0, 0] |
| R−U s202 | −0.0444 | 122 / −0.0422 | 1 / +0.0007 | 2 / −0.0029 | 3 | −0.0078 [−0.0391, +0.0156] |
| R−H s101 | +0.0491 | 125 / +0.0491 | 0 / 0 | 0 / 0 | 3 | 0 [0, 0] |
| R−H s202 | +0.0239 | 121 / +0.0220 | 2 / +0.0024 | 1 / −0.0005 | 4 | +0.0078 [−0.0156, +0.0391] |
| U−H s101 | +0.0073 | 125 / +0.0073 | 0 / 0 | 0 / 0 | 3 | 0 [0, 0] |
| U−H s202 | +0.0684 | 122 / +0.0640 | 2 / +0.0044 | 0 / 0 | 4 | +0.0156 [0, +0.0391] |

In s101 the same 125 problems are solved by H, R and U; the same 3 are never solved. Reliability carries 100% of every s101 contrast and ≥ 92% of every s202 contrast.

## 6. Stress panels (task 4)
The file has 221 items: Harder 95 and SuperHard 126. **Every cell evaluated all 221 items × 8 attempts = 1,768 attempts**,
with 0 missing, 0 duplicate and 0 unknown attempts. Seeds are equal across all 9 cells. Every capped attempt has 12,288 tokens.
Status `no_answer` appears once in base and once in R-s202 r128 (reward 0).

| cell | pass@1 (correct/1,768) | Harder | SuperHard | truncation (count) | trunc. Harder / SuperHard | acc. finished | coverage at 8 (solved/221) | cov. Harder / SuperHard |
|---|---|---|---|---|---|---|---|---|
| base | 0.3727 (659) | 0.5250 | 0.2579 | 0.0006 (1) | 0.0013 / 0 | 0.3729 | 0.6878 (152) | 0.8211 / 0.5873 |
| H-s101 r128 | 0.6414 (1134) | 0.7947 | 0.5258 | 0.1770 (313) | 0.0908 / 0.2421 | 0.7794 | 0.8281 (183) | 0.9263 / 0.7540 |
| R-s101 r128 | 0.6951 (1229) | 0.8421 | 0.5843 | 0.2048 (362) | 0.0987 / 0.2847 | 0.8741 | 0.8326 (184) | 0.9474 / 0.7460 |
| U-s101 r128 | 0.6533 (1155) | 0.8013 | 0.5417 | 0.2296 (406) | 0.1145 / 0.3165 | 0.8480 | 0.8326 (184) | 0.9474 / 0.7460 |
| H-s202 r128 | 0.6012 (1063) | 0.7434 | 0.4940 | 0.2551 (451) | 0.1513 / 0.3333 | 0.8071 | 0.8009 (177) | 0.9263 / 0.7063 |
| R-s202 r128 | 0.6318 (1117) | 0.7632 | 0.5327 | 0.2302 (407) | 0.1342 / 0.3026 | 0.8207 | 0.8145 (180) | 0.9368 / 0.7222 |
| U-s202 r128 | 0.6584 (1164) | 0.7947 | 0.5556 | 0.1516 (268) | 0.0921 / 0.1964 | 0.7760 | 0.8507 (188) | 0.9579 / 0.7698 |
| R-s101 r64 | 0.6425 (1136) | 0.7855 | 0.5347 | 0.2613 (462) | 0.1461 / 0.3482 | 0.8698 | 0.8281 (183) | 0.9368 / 0.7460 |
| H-s202 r64 | 0.6239 (1103) | 0.7605 | 0.5208 | 0.1510 (267) | 0.0868 / 0.1994 | 0.7348 | 0.8281 (183) | 0.9368 / 0.7460 |

Per-seed contrasts (paired over items; "both" = items solved by both, "one" = items solved by one policy only):

| contrast | N | Δ pass@1 [95%] | trunc. term | acc. term | Δ coverage at 8 [95%] | both: n / contrib.; one: contrib. |
|---|---|---|---|---|---|---|
| R−U s101 all | 221 | +0.0419 [+0.0153, +0.0679] | +0.0214 | +0.0204 | 0 [−0.0362, +0.0362] | 176 / +0.0396; +0.0023 |
| R−U s101 Harder | 95 | +0.0408 [+0.0053, +0.0763] | +0.0145 | +0.0263 | 0 [0, 0] | 90 / +0.0408; 0 |
| R−U s101 SuperHard | 126 | +0.0427 [+0.0050, +0.0804] | +0.0255 | +0.0171 | 0 [−0.0635, +0.0635] | 86 / +0.0387; +0.0040 |
| R−U s202 all | 221 | −0.0266 [−0.0503, −0.0017] | −0.0628 | +0.0362 | −0.0362 [−0.0633, −0.0136] | 180 / −0.0141; −0.0124 |
| R−U s202 Harder | 95 | −0.0316 [−0.0671, +0.0026] | −0.0370 | +0.0054 | −0.0211 [−0.0526, 0] | 89 / −0.0276; −0.0039 |
| R−U s202 SuperHard | 126 | −0.0228 [−0.0565, +0.0109] | −0.0772 | +0.0544 | −0.0476 [−0.0873, −0.0159] | 91 / −0.0040; −0.0188 |
| R−H s101 all | 221 | +0.0537 [+0.0260, +0.0814] | −0.0229 | +0.0766 | +0.0045 [−0.0317, +0.0407] | 176 / +0.0498; +0.0040 |
| R−H s202 all | 221 | +0.0305 [−0.0006, +0.0628] | +0.0203 | +0.0103 | +0.0136 [−0.0226, +0.0498] | 170 / +0.0170; +0.0136 |
| U−H s101 all | 221 | +0.0119 [−0.0147, +0.0385] | −0.0428 | +0.0547 | +0.0045 [−0.0317, +0.0407] | 176 / +0.0090; +0.0028 |
| U−H s202 all | 221 | +0.0571 [+0.0238, +0.0911] | +0.0819 | −0.0248 | +0.0498 [+0.0181, +0.0860] | 174 / +0.0311; +0.0260 |
| R-s101 r128 − r64 | 221 | +0.0526 [+0.0305, +0.0758] | +0.0493 | +0.0033 | — | — |
| H-s202 r128 − r64 | 221 | −0.0226 [−0.0526, +0.0062] | −0.0802 | +0.0576 | — | — |

Per-tier R−H and U−H rows are in `data/eval_stress.json`. Stress repeats the test pattern: R−U is + in s101 and − in s202.
In s202 the truncation term is negative and the accuracy term positive. Unlike the test, coverage matters on stress:
in s202, U solves 8 items that R never solves, and R solves none that U misses.
Caveat: the H-s202 r128 stress cell ran on L40 and A6000; all other stress cells ran on L40S.

## 7. Length extension 12,288 → 16,384 (task 5)
Checks against the original records (verified; each count is 0 in all 10 checkpoint × panel sets): capped originals
with n_tokens ≠ 12,288; originals at 12,288 tokens not capped; eligible attempts not extended; extended attempts not
eligible; row fields (n_tokens_1, finish_1, reward_1) differing from the original; unextended attempts whose reward
changed; n_tokens_2 ≠ 12,288 + new tokens; totals above 16,384; still-capped attempts not at 16,384 or with reward 1;
unknown statuses at 16,384.

Rows cover every original attempt exactly once. Stress rows are t1 (256) + rest (1,512), with no overlap.
H-s202 and base were not extended.

| panel | policy | attempts | capped at 12,288 | extended | recovered (correct at 16,384) | finished wrong | still capped | pass@1 12,288 | pass@1 16,384 | change |
|---|---|---|---|---|---|---|---|---|---|---|
| test | H-s101 | 4096 | 445 | 445 | 93 | 141 | 211 | 0.7695 | 0.7922 | +0.0227 |
| test | R-s101 | 4096 | 437 | 437 | 94 | 50 | 293 | 0.8186 | 0.8416 | +0.0229 |
| test | U-s101 | 4096 | 501 | 501 | 118 | 67 | 316 | 0.7769 | 0.8057 | +0.0288 |
| test | R-s202 | 4096 | 538 | 538 | 87 | 104 | 347 | 0.7478 | 0.7690 | +0.0212 |
| test | U-s202 | 4096 | 333 | 333 | 66 | 65 | 202 | 0.7922 | 0.8083 | +0.0161 |
| stress | H-s101 | 1768 | 313 | 313 | 55 | 126 | 132 | 0.6414 | 0.6725 | +0.0311 |
| stress | R-s101 | 1768 | 362 | 362 | 49 | 61 | 252 | 0.6951 | 0.7229 | +0.0277 |
| stress | U-s101 | 1768 | 406 | 406 | 51 | 64 | 291 | 0.6533 | 0.6821 | +0.0288 |
| stress | R-s202 | 1768 | 407 | 407 | 50 | 96 | 261 | 0.6318 | 0.6601 | +0.0283 |
| stress | U-s202 | 1768 | 268 | 268 | 41 | 81 | 146 | 0.6584 | 0.6816 | +0.0232 |

| contrast | at 12,288 [95%] | at 16,384 [95%] | change (16k − 12k) [95%] |
|---|---|---|---|
| test R−U s101 | +0.0417 [+0.0203, +0.0632] | +0.0359 [+0.0151, +0.0571] | −0.0059 [−0.0159, +0.0037] |
| test R−U s202 | −0.0444 [−0.0728, −0.0176] | −0.0393 [−0.0671, −0.0134] | +0.0051 [−0.0029, +0.0129] |
| test R−H s101 | +0.0491 [+0.0208, +0.0791] | +0.0493 [+0.0215, +0.0793] | +0.0002 [−0.0078, +0.0085] |
| test U−H s101 | +0.0073 [−0.0193, +0.0359] | +0.0134 [−0.0105, +0.0396] | +0.0061 [−0.0022, +0.0151] |
| stress R−U s101 | +0.0419 [+0.0153, +0.0679] | +0.0407 [+0.0141, +0.0673] | −0.0011 [−0.0113, +0.0096] |
| stress R−U s202 | −0.0266 [−0.0503, −0.0017] | −0.0215 [−0.0458, +0.0034] | +0.0051 [−0.0051, +0.0153] |
| stress R−H s101 | +0.0537 [+0.0260, +0.0814] | +0.0503 [+0.0232, +0.0769] | −0.0034 [−0.0153, +0.0079] |
| stress U−H s101 | +0.0119 [−0.0147, +0.0385] | +0.0096 [−0.0187, +0.0368] | −0.0023 [−0.0147, +0.0090] |

Between 12% and 24% of capped attempts recover (test 16–24%, stress 12–18%). No contrast changes by more than 0.0061, and every change interval contains 0.
The s202 R−U deficit on the test shrinks from −0.0444 to −0.0393 but stays negative. Its interval still excludes 0.

## 8. Cue study and response audit (task 6)
Cue: 24 items × 8 attempts per policy and form. There are 0 missing, 0 duplicate and 0 unknown attempts.
The item IDs are identical in both forms, and the seeds are equal across forms for 192/192 attempts in every policy.
Model hashes are in job log 11104738.

| policy | uncued pass@1 (c/192) | cued pass@1 (c/192) | cued − uncued [95%] | truncation uncued / cued |
|---|---|---|---|---|
| base | 0.4219 (81) | 0.4115 (79) | −0.0104 [−0.0781, +0.0573] | 0 / 0 |
| R-s101 | 0.6823 (131) | 0.6927 (133) | +0.0104 [−0.0365, +0.0677] | 0.1875 / 0.1771 |
| U-s101 | 0.7240 (139) | 0.7135 (137) | −0.0104 [−0.0729, +0.0417] | 0.1823 / 0.2031 |
| R-s202 | 0.6198 (119) | 0.5729 (110) | −0.0469 [−0.0990, 0.0000] | 0.1875 / 0.2188 |
| U-s202 | 0.6458 (124) | 0.6927 (133) | +0.0469 [−0.0052, +0.1094] | 0.1562 / 0.0781 |

Cue gain R − U (difference in differences): s101 +0.0208 [−0.0625, +0.1146]; s202 −0.0938 [−0.1927, −0.0156].
Uncued R−U: s101 −0.0417 [−0.1198, +0.0260]; s202 −0.0260 [−0.0990, +0.0417]. No cue effect is consistent across seeds.
The s202 cue-gain interval excludes 0 and comes from 24 items; it is exploratory.

Response audit (counted from `key.json` and the four `ann_*.json` read files; `tools/b-eval/audit_reads.py`):

| quantity | count |
|---|---|
| responses (16 items × 6 policies; attempt 0; no H-s202) | 96 (48 SuperHard, 48 Harder) |
| key fields vs stress records (reward, finish, n_tokens, tier) | 96/96 equal (`eval_stress.json`) |
| sample rule recomputed: first 8 per tier by sha256(`audit-v1:`+id) | equal to the file, set and order, both tiers |
| reads per response | 2 for all 96 (192 reads) |
| reward 0 / of these capped | 35 / 13 (base 8/0, H-s101 5/1, R-s101 4/1, U-s101 4/3, R-s202 8/5, U-s202 6/3) |
| reward 1 | 61 (reads: 120 "correct", 2 "unclear" on item `…train/00331`) |
| reads of reward-0 responses called "correct" | 0 |
| reads' `capped` flag equals the record | 192/192 |
| category agreement per response (both reads equal) | all 88/96; reward-0 29/35; reward-0 uncapped 18/22 |
| `factoring_used` agreement | all 84/96; reward-0 30/35 |

Reads of the 35 reward-0 responses (70 reads):

| category | missed_opportunity | factored_then_failed_later | hit_cap | factored_incorrectly | grader_issue | other_method |
|---|---|---|---|---|---|---|
| reads | 26 | 17 | 11 | 9 | 6 | 1 |
| responses with both reads agreeing | 13 | 6 | 5 | 3 | 2 | 0 |

- The 13 capped responses got 11 "hit_cap", 10 "missed_opportunity", 3 "factored_then_failed_later", 2 "factored_incorrectly" reads.
- Grader issue: 2 responses have both reads "grader_issue" (EGQA R-s202, FVWP H-s101; both `math:precalculus/train/00152`).
  2 more have one such read (EYZW, GNYA on `math:algebra/test/00352`).
- "33 after excluding grader-limitation responses" (DEFINITIONS v1) matches the both-reads rule. The any-read rule gives 31.

## 9. Traced cases (task 7)
- (a) Correct attempt. Cell base, `intermediate_algebra/test/00874`, sample 4: seed 1478459057, 926 tokens, finish `stop`.
  The decoded tail ends `\boxed{q(x) = -3(x + 2)(x - 1)}`, and my own last-box extractor returns the same string as the
  record's `extracted`. The gold is `-3(x+2)(x-1)`, the status `correct` and the reward 1.
  The problem has 10/32 correct, so it adds 10/32/128 = 0.00244 to the base pass@1 of 2153/4096 = 0.5256.
- (b) Capped attempt and its extension. R-s101, the same problem, sample 26. The original has 12,288 tokens, finish
  `length`, status `truncated` and reward 0. The row has n_tokens_1 = 12,288, `extended` = true and 1,431 new tokens,
  so n_tokens_2 = 13,719 with finish_2 `stop`. The last box is `-3x^2 - 3x + 6`, graded correct (reward_2 1), so the
  attempt counts as "recovered". The problem goes from 16/32 to 17/32, and R-s101 test pass@1 from 3353 to 3447 correct
  (0.8186 → 0.8416).
- (c) Multiple boxes. Base, `intermediate_algebra/test/00373`, sample 15, finish `stop`. Boxes `2` then `\infty`; the
  record extracted `\infty` (the last box), so it is `wrong` with reward 0, against gold `2`.
  Census, 7 primary test cells (order base, H-s101, R-s101, U-s101, H-s202, R-s202, U-s202): my extractor equals the
  record's `extracted` in 28,672/28,672 responses. Responses with ≥ 2 boxes: 1823, 1874, 2291, 454, 2257, 483, 500; with
  distinct boxes: 454, 301, 187, 65, 175, 88, 64. Finished reward-0 responses where an earlier box equals the gold string
  (whitespace removed): 5, 8, 0, 1, 1, 1, 1 (17 of 28,672); exact string match only.
- (d) H-s202 r128 shard. `intermediate_algebra/test/00758` is panel index 3, so shard k3 (run on A6000). The shard holds
  32 attempts: 32 correct, 0 capped. The merged file has the same 32 attempts (identical ids). The problem adds 1/128 = 0.0078
  to the merged pass@1 of 2965/4096 = 0.7239.

## 10. Anomalies and items not reconstructed
1. **GPU mix in H-s202 cells (verified).** Test r128: 63 of 128 problems (shards k3–k5, 2,016 attempts) ran on A6000, the
   rest on L40S. Stress r128: L40 (k0, k2) and A6000 (k1); every other stress cell ran on L40S. R originals ran on L40.
   The R-s202 L40 → L40S rerun moved pass@1 by +0.0134 [+0.0022, +0.0256], so GPU type matters for H-s202 contrasts.
2. **The R−U sign flips between seeds** on the test and the stress panel. It survives the common-GPU rerun and the
   16,384 extension. On the 24-item cue panel, uncued R−U is negative in both seeds (both intervals contain 0).
3. **Unknown verifier status:** 2 attempts (base, sealed test), counted as incorrect. All other cells have 0.
4. **Same-GPU reproducibility is not uniform:** U-s202 reproduced 4,096/4,096 responses, U-s101 1,964/4,096 (both L40S). Cause unknown.
5. **Panel sizes:** every stress cell evaluated all 221 items (file 221). H-s202 r128 is absent from the response audit,
   the cue study (which has no H policy) and both length extensions.
6. **DEFINITIONS v1:** my recomputed values agree (1,964/4,096; 35/13/33; panel sizes; fallback round 116; r64 hashes).
   §4 item 12 says shards "k2–k5 ran on any 48 GB GPU": k2 ran on L40S, k3–k5 on A6000. The stress H-s202 r128 GPU mix is not in §4.
7. **Not reconstructed:** the trajectory label (a2-r60 vs a2-orig) of H-s202 r64 needs training metrics (claimed);
   whether the 17 "earlier box equals gold" cases are true grader losses needs Math-Verify equivalence (unknown);
   answers split across boxes were not counted.
