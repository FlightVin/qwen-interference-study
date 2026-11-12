# A26 pass A — claim-by-claim verification of report v2 (agent a-evals)

Purpose: recompute every evaluation-based number in report v2 (`reports/archive/REPORT-v2.md`) from raw attempt records, compare it with the report and with the production analysis outputs, and answer the A26 §4 acceptance items. Unrounded values: `data/numbers.json` (`claims`, `recomputed`). Terms follow `research/DEFINITIONS.md` v1.

**Method.** `tools/a-evals/scan_evals.py` (compute, standby 11122432, 2026-11-11 EST) read every record file and exported per-attempt outcomes per problem. `tools/a-evals/recompute_claims.py` (laptop, numpy, no project imports) recomputed each claim. Point estimates are exact. Intervals use an independent paired problem bootstrap (10,000 draws, `default_rng(20261111)`); a reported bound counts as verified when it lies within 0.3 points of mine, and every interval was also compared with the production JSON. In-run dev and probe values come from canonical `metrics.jsonl` rows (`tools/a-evals/traces.py`). Training-derived tables (§4.4, §4.5) were checked as transcriptions of the production outputs only (`tools/a-evals/production_compare.py`). Labels: **verified** (primary records), **production only** (matches the production output; not recomputed from raw records), **unverifiable**, **not checked**.

## 1. Result summary

236 claims were checked: 223 verified, 11 differ, 1 interval differs from my draws by > 0.3 points but equals the production output, 1 production only. In addition, 24 displayed equations of the form `a − b = c` fail at displayed precision, and 4 other displayed values disagree with their own displayed inputs.

| Report section | Claims | Verified | Not verified (claim id in `numbers.json`) |
|---|---|---|---|
| Opening 1 / Q1 | 2 | 0 | `open.range`, `open.gain` |
| Opening 3, 5; Q3 | 7 | 5 | `open5.range`, `open5.sh` |
| 2 (endpoints, fallback, common GPU, reproducibility) | 35 | 35 | — |
| 3.1–3.2 (dev, probes) | 12 | 12 | — |
| 4.1–4.3 (reversal, coverage, truncation) | 35 | 35 | — |
| 5.2–5.4 and length contrasts | 61 | 57 | `5.3.gainrange`, `5.3.H`, `5.3.recrange`; tranche-1 s202 R − U at 12,288 interval (mine [−8.6, +4.7], report = production [−9.0, +4.3]; 64 problems) |
| 6 (local diagnostic) | 5 | 4 | `6.r64` |
| 7.2 (stress) | 73 | 70 | `7.2.gain`, `7.2.gainedlost`, `7.2.p8gap` |
| 7.3 (audit denominators) | 3 | 3 | — |
| 8 (cue) | 3 | 2 | `8.factor` (production only) |

## 2. Numbers that differ or are unverifiable

| # | Claim (report line) | Report | Recomputed (unrounded) | Evidence |
|---|---|---|---|---|
| 1 | "five finished trained policies scored 74.8%–81.9%" (23, 92) | 74.8–81.9 | six r128 endpoints 72.388–81.860; all six finished | `test/H-s202` summary 0.723877 |
| 2 | "a gain of +22 to +29 points" (24, 93) | +22 to +29 | six endpoints +19.82 to +29.30 (without H-s202 +22.22) | base 0.525635 |
| 3 | "the trained checkpoints 62.4–69.5%" on 221 stress problems (66) | 62.4–69.5 | eight trained cells 60.12–69.51; 62.4 is H-s202 **r64** | `stress/H-s202-r128` 0.601244 |
| 4 | "SuperHard pass@1 roughly doubled (25.8% → 52.1–58.4%)" (67, 784) | 52.1–58.4 | 49.40–58.43 (H-s202 r128 498/1,008) | tier recount |
| 5 | "Every trained checkpoint gains 25–32 points over the base model" (783) | 25–32 | 22.85–32.24 (H-s202 r128 +22.85) | stress all-221 |
| 6 | "It loses 2–5 that the base model solved" (785) | 2–5 | 2–8; H-s202 r128 loses 8 (production `stress.json` coverage also 8) | per-problem c > 0 |
| 7 | "pass@8 … differs by at most 3.7 points between arms" (788) | 3.7 | R vs U max 3.62 (s202: 0.851 vs 0.814); including H the max is 4.98 (U − H, s202) | n = 8 coverage |
| 8 | "Every checkpoint gains 2.4–2.9 points of pass@1" (stress, 577) | 2.4–2.9 | 2.32 (U-s202) to 3.11 (H-s101); R/U only 2.32–2.88 | length rows |
| 9 | "It gains 3.2 points (64.1% → 67.3%)" H-s101 stress (584) | 3.2 | 3.11 (0.641403 → 0.672511) | length rows |
| 10 | "55–72% are still capped" (579) | 55–72 | 54.48–71.67 (U-s202 146/268) | length rows |
| 11 | r64 no-update J_H "60.6%" (646) | 60.6 | 0.60546875 → 60.5 | `research/evidence/a15/diag_r64.jsonl` |
| 12 | "+9 to +15 points more mentions of 'factor'" (853) | +9 to +15 | production values +8.85 to +15.10; texts not re-decoded | production only (`cue_analysis.json`) |
| 13 | Ranges in §4.2 "82.5–88.0%", "+21.7 to +28.8", "+20.4 to +25.3" (400–401) | as printed | correct for five policies; with H-s202: 79.6–88.0, +19.3 to +28.8, +17.9 to +25.3 | `coverage.json` includes H-s202 (verified: values equal records) |

Values the report states correctly but labels loosely: "Extra tokens per recovered answer" (§5.2–5.4) is all new tokens on every extended attempt divided by the number recovered (e.g. R-s202 sealed test 1,781,371 / 87 = 20,476); recovered attempts themselves used 1,767 new tokens on average.

## 3. A26 §4 acceptance items

### 3.1 Ranges that omit H-s202
**Finding: verified defect.** Items 1–6 and 13 of section 2 omit H-s202 r128 although the run finished and its cells are in the report's own tables (sealed 72.4%; stress 60.1% overall, 49.4% SuperHard, 74.3% Harder). Explicit sets: six r128 endpoints on the sealed test 72.4–81.9% (base 52.6%, gains +19.8 to +29.3); eight trained stress cells 60.1–69.5% overall and 49.4–58.4% on SuperHard; the six r128 stress cells 60.1–69.5% and 49.4–58.4%. The response audit genuinely covered five trained policies (base, H-s101, R-s101, U-s101, R-s202, U-s202; 16 responses each): verified from `audit_merged.json`.

### 3.2 Earliest observed R/U ordering, per seed and panel
**Finding: the report's "appeared between rounds 32 and 64 in both seeds" holds for seed 101 only.** Values are R − U in points from canonical in-run rows (n = 8 per problem; dev 53, J_H 32, J_R 28, J_U 32 problems; cap 8,192 at rounds 0 and 32, 12,288 from 64). Round 0 is the same base weights with the same seeds in both arms, so a round-0 difference is sampling non-reproducibility (DEFINITIONS §3.B).

| Seed | Panel | r0 | r32 | r64 | r96 | r128 | Endpoint sign first observed | Sign holds from |
|---|---|---|---|---|---|---|---|---|
| 101 | dev | −0.5 | −5.9 | +2.6 | +6.4 | +3.8 | r64 | r64 |
| 101 | J_H | −1.6 | −9.4 | +0.4 | +7.0 | +6.2 | r64 | r64 |
| 101 | J_R | 0.0 | +0.9 | +4.0 | +1.3 | −1.3 | r128 (negative) | r128 |
| 101 | J_U | −0.4 | −2.0 | +3.5 | −0.8 | +2.0 | r64 | r128 |
| 202 | dev | −1.9 | −1.2 | −6.6 | −3.8 | −4.7 | **r0** | **r0** |
| 202 | J_H | −2.3 | −2.0 | −7.8 | −0.4 | −5.9 | **r0** | **r0** |
| 202 | J_R | −0.9 | +5.4 | −5.8 | −2.7 | −1.8 | r0 | r64 |
| 202 | J_U | −1.2 | −1.6 | −3.9 | −8.2 | −2.3 | **r0** | **r0** |

Training success on the shared hard prompts (production `divergence.json`, 8-round bins, R − U): seed 101 negative in bins 9–40, positive from 41–48 except 49–56 (−0.6), so the endpoint sign holds from 57–64; seed 202 positive in bins 1–24, first negative at 25–32 (−0.2), then −6.1 and −4.5 in 33–48, sign changes until 89–96, negative from 97–104. Sealed test and stress: R/U pairs exist only at r128 (no same-seed R/U pair at rounds 32–96; R-s101 r64 has no U partner). The cap changed between the r32 and r64 evaluations, so any change in that window mixes training with the evaluation budget.

### 3.3 Seed-202 sealed-test length recovery
**Finding: verified.** R-s202: 538 capped at 12,288; 87 recovered (16.2%), 104 finished wrong, 347 still capped; pass@1 0.747803 → 0.769043 (+2.12). U-s202: 333 capped; 66 recovered (19.8%), 65 finished wrong, 202 still capped; 0.792236 → 0.808350 (+1.61). R − U −4.44 → −3.93 (change +0.51 [−0.3, +1.3]). Both checkpoints gained absolute success; the ordering did not change. "More room does not rescue those answers" (line 119) contradicts the 87 and 66 measured rescues.

### 3.4 Per-seed R − H and U − H intervals that include zero
**Finding: the report's statement is verified for the sealed test; more intervals include zero on other panels.** Per-seed 95% paired problem bootstrap (report value; my recomputation agrees within 0.1):

| Panel | R − H s101 | R − H s202 | U − H s101 | U − H s202 |
|---|---|---|---|---|
| Sealed test (original) | +4.9 [+2.0, +8.0] | **+2.4 [−0.4, +5.4]** | **+0.7 [−2.0, +3.5]** | +6.8 [+3.5, +10.2] |
| Sealed test (common GPU for R/U) | +5.3 [+2.5, +8.2] | +3.7 [+0.9, +6.6] | **+0.9 [−1.8, +3.5]** | +6.8 [+3.5, +10.2] |
| Sealed test at 16,384 | +4.9 [+2.0, +8.0] | not run | **+1.3 [−1.2, +3.9]** | not run |
| Stress (221) | +5.4 [+2.5, +8.2] | **+3.1 [−0.1, +6.3]** | **+1.2 [−1.4, +3.8]** | +5.7 [+2.5, +9.2] |

Bold intervals include zero. The H-s202 cells mix GPU types (L40S/A6000 on the test, L40/A6000 on stress). The two-seed means (+3.6 [+1.6, +5.8], +3.8 [+1.6, +6.0]) resample problems only and carry no seed variation.

### 3.5 Response-audit denominators
**Finding: verified** from `research/evidence/a17/audit/audit_merged.json`: 96 responses (16 items × 6 policies), 192 reads (2 per response); 35 responses with reward 0, 13 of them capped; 2 grader-limitation responses (both reads `grader_issue`, `math:precalculus/train/00152`, R-s202 and H-s101) leave 33 genuine failures; 70 failure reads = yes 29 / no 27 / unclear 10 / n.a. 4; 66 judged reads; per response both-yes 13, both-no 10. The excerpting of responses over 8,000 characters and the "WRONG" label on capped responses come from `build.py` (not rerun here).

### 3.6 Displayed equations that fail at displayed precision
**Finding: 24 of 107 displayed `a − b = c` equations fail** (`recompute_claims.py` regex over the report; the printed c is the correctly rounded unrounded difference in every case):

| Line | Equation as printed | a − b of displayed values |
|---|---|---|
| 68, 130, 573, 773 | 63.2 − 65.8 = −2.7 | −2.6 |
| 90, 234 | 78.3 − 78.5 = −0.1 | −0.2 |
| 130, 573 | 66.0 − 68.2 = −2.1 | −2.2 |
| 163, 572 | 58.4 − 54.2 = +4.3 | +4.2 |
| 258, 274 | 82.3 − 77.8 = +4.4 | +4.5 |
| 258 | 77.8 − 77.0 = +0.9 | +0.8 |
| 373 | 85.5 − 88.3 = −2.9 | −2.8 |
| 550 | 84.2 − 79.2 = +4.9 | +5.0 |
| 551 | 80.6 − 79.2 = +1.3 | +1.4 |
| 571 | 86.1 − 81.8 = +4.2 | +4.3 |
| 575 | 56.0 − 58.5 = −2.6 | −2.5 |
| 587 | 68.2 − 67.3 = +1.0 | +0.9 |
| 606 | 67.6 − 69.5 = −2.0 | −1.9 |
| 606 | 71.1 − 73.0 = −2.0 | −1.9 |
| 645 | 40.5 − 41.5 = −0.9 | −1.0 |
| 646 | 60.9 − 60.6 = +0.2 | +0.3 |
| 778 | 69.5 − 64.3 = +5.3 | +5.2 |

Non-equation displays with the same problem: §3.2 table seed 202 r96 "64.2% | 67.9% | −3.8" (displayed difference −3.7); §4.3 "−7.0 … (R 68.6% vs U 75.7%)" (−7.1); §7.2 "at most 3.7 points" (3.62); §5.3 "2.4–2.9" and "3.2" (differences of rounded values).

## 4. Verified clusters (one line each; ids in `numbers.json`)

- **§2 endpoints:** base 52.6% / pass@8 83.5%; H/R/U per seed; all six per-seed intervals and the three two-seed means equal production `test-cgpu/endpoints.json` (my bounds within 0.1). Fallback r116 74.3% / 92.8% / 14.8% capped; r128 72.4% / 90.8% / 15.4%; 1.9 points below r116.
- **§2 common GPU:** cells 82.3 / 77.8 / 76.1 / 79.2; nine contrasts and means; reproducibility 48, 133, 1,964, 4,096 identical token sequences and 62, 46, 79, 128 identical c per problem; rerun − original +0.4, +1.3, +0.1, 0.0; max 1.3; U-s101 48%.
- **§3.1–3.2:** H-s202 (a2-r60) dev 63.7 / 66.0 / 67.0; all ten dev rows; base ≈ 81% on J_R/J_U and gains 8–15 points in all six runs.
- **§4.1:** signs 31 / 44 / 53 / 31, all-four 3 and 26; Spearman −0.09 and −0.04; all five band rows; swing −8.6, B1 −5.7; top-3 30%, top-5 45%; `algebra/test/00213` 23/13 → 11/24.
- **§4.2:** coverage table rows for all seven policies (pass@1, pass@8, solved ≥ 1 of 32, 0/32, truncated attempts); 14 unreachable at baseline (base attempts 16–31); seed 202 both-reach 122 carry −4.2, one-arm 3 carry −0.2.
- **§4.3 / Q2:** truncation shares 10.7/12.2 and 13.1/8.1; accuracy among finished 91.6/88.5 and 86.1/86.2; terms +1.4/+2.8 and −4.3/−0.1 with intervals; within 8,192 +5.1 and −7.0 with cells and intervals; Pearson −0.51 / −0.59; 538 truncated.
- **§5:** every row of the sealed-test, full-stress and tranche-1 tables (counts, percentages, extra-token ratios); 14 contrasts × (12,288, 16,384, change) with intervals; ranges 1.6–2.9 and 16–24% (sealed test), 12–15% recovered (stress), 6–11 of 41–56 and 51–70% (tranche 1).
- **§6:** r0 row (40.0 / 43.2 / 41.5 / 42.7 / 40.5 / −2.2 / −0.9); r64 means and differences; zero-gradient draws 61.1 / 89.5 / 93.2; unrelated draws 59.2–63.3. Diagnostic records: n = 16, cap 8,192; at r64, 2,771 of 7,680 J_H attempts (36.1%) hit the cap.
- **§7.2:** all nine all-221 rows, all 18 tier rows (pass@1, correct/attempts, solved ≥ 1 of 8, 0/8, truncated, accuracy among finished), all seven contrast rows (pass@1 with interval, both terms with intervals, pass@8 pair); gained 33–38; H-s202 −2.3 from r64 to r128; every cell 1,768 attempts and 0 verifier-unknown.
- **Q3 tier contrasts:** Harder +4.1 / −3.2 and SuperHard +4.3 / −2.3 with intervals.
- **§8:** cells 62.0 / 57.3 / 64.6 / 69.3; gains −4.7 / +4.7; differences in cue gain +2.1 / −9.4 (intervals equal production; not re-bootstrapped).
- **§4.4–4.5 (production only):** all six rows of each §4.4 table and the §4.5 table equal `signal.json` / `persistence.json`; all-correct practice share after round 32: R 62.1–90.2%, U 74.2–92.2% (report 62–90%, 74–92%). §3.2 training-success bins and answer lengths (4,435 vs 4,755; 4,874 vs 5,590) equal `divergence.json`; §4.3 loops (0/538, median 4.7%) equal `loops.json`.

## 5. Interval consistency across sections

The same contrast is printed with different intervals in different sections because each analysis used its own bootstrap stream: sealed-test seed-101 R − U [+2.1, +6.3] (§2) vs [+2.1, +6.2] (§5.2); seed-202 [−7.2, −1.7] vs [−7.3, −1.8]; stress seed-101 R − U [+1.5, +6.8] (§7.2) vs [+1.6, +6.8] (§5.3); Harder s101 [+0.7, +7.8] (Q3) vs [+0.5, +7.8] (§5.3); SuperHard s101 [+0.5, +8.0] vs [+0.6, +8.0]; Harder s202 [−6.7, +0.3] vs [−6.6, +0.3]; SuperHard s202 [−5.6, +1.0] vs [−5.6, +1.1]; R − H s101 stress [+2.5, +8.2] vs [+2.5, +8.3]; U − H s101 stress [−1.4, +3.8] vs [−1.5, +3.9]. §5.2 notes this for the sealed test only. All are within Monte Carlo error; one stream per contrast would remove the inconsistency.

## 6. Other textual defects found while checking

- §7.2 line 710: "Each checkpoint made 8 attempts … on L40S" is false for H-s202 r128 (L40 × 2, RTX A6000 × 1). Line 711–712 "H-s202 round 128 runs only after that training run ends (section 9)" is stale: the cell is in the table.
- Opening 4 / §6: the diagnostic ran at cap 8,192 although R-s101 r64 trained at 12,288; 36.1% of r64 diagnostic J_H attempts were capped (DEFINITIONS §4 item 1). The report does not state the cap.
- Base sealed test: 2 of 4,096 attempts are verifier-unknown and count as incorrect (not stated in the report; effect ≤ 0.05 points).

## 7. Not checked by a-evals

§9 cost table and GPU-hours (`reports/tables/cost.json`); §7.1 construction counts (2,039 candidates, 207/219, 171/219, obstacle counts); §5.5 continuation reading; §4.6 sampling-null counts (1.7–5.7, 0–5, 6); §7.3 category columns per policy; the sealed-test two-box count (9 attempts, `twobox.py`); opening 2 and §4.4 token and GPU-hour totals beyond the transcription check; checkpoint hashes (recorded values only; another agent hashes the files).
