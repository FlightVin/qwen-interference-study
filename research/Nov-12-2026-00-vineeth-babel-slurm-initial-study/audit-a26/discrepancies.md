# A26 discrepancy log

Purpose: one row per defect found by audit passes A and B and by the reconcile step, with the original claim, the
evidence, the correction, its scientific consequence and where it is fixed (A26 §3). Written 2026-11-11, ~10 PM EST, by
agent `reconcile`. Line numbers refer to `reports/archive/REPORT-v2.md` ("L…"); "DEF" means
`research/DEFINITIONS.md` v1. Points = percentage points. "Fixed where" names the place the fix belongs; nothing was
edited here. Pass comparison: `comparison.md`. Sensitivity checks: §3 below.

Severity order: **I** = changes an interpretation, **N** = changes a reported number only, **W** = wording, provenance or
code hygiene with no effect on a reported number.

## 1. Discrepancy table

### 1.1 Changes interpretation

| # | Original claim | Evidence | Correction | Scientific consequence | Fixed where |
|---|---|---|---|---|---|
| I1 | "more room does not rescue those answers" (L119); capped answers "were not near-misses" (L44, L134); "a bigger budget does not convert them" (L137–138); "A larger inference budget is not a fix" (L145); "Extending the budget did not move R − U … capped answers are mostly stuck searches" (L183) | `pass-a/eval-map.md` §5, `pass-b/eval.md` §7 (equal): sealed test, seed 202: R recovered 87 of 538 capped attempts (16.2%), U 66 of 333 (19.8%); pass@1 R 74.78 → 76.90 (+2.12), U 79.22 → 80.84 (+1.61). All 5 extended checkpoints gain +1.61 to +2.88 (test) and +2.32 to +3.11 (stress). R − U change ≤ 0.59 points; every overall change interval contains 0 (only the seed-202 Harder tier change, +1.6 [+0.1, +3.2], excludes 0). Still capped at 16,384: 47–67% of capped test attempts | Two observations, stated separately: (1) 4,096 more tokens raised absolute success for every checkpoint by 1.6–3.1 points; 12–24% of capped attempts finished correctly. (2) The ordering did not change: R − U moved by ≤ 0.6 points in both seeds on both panels | The budget is not neutral: it helps every checkpoint. What stays true is narrower: the extension does not change R − U, so the budget does not explain the seed-dependent ordering. "Capped = stuck" holds for 76–84% (test) and 82–88% (stress) of capped attempts (still capped or finished wrong), not for all | report (Q2, §5, Q4; A26 §4 item 3) |
| I2 | "On answers that finished, R and U were equally accurate" (L40); "R lost nothing on the answers it finished" (L129); "So R-s202 did not lose arithmetic skill" (L136–137) | Decomposition verified by both passes: s202 accuracy among finished R 86.09% vs U 86.23%, term −0.13 [−2.40, +2.01] (pass A). Accuracy among finished conditions on which attempts finished; R-s202 left 538 attempts unfinished, U-s202 333 (DEF §3.C "accuracy among finished") | Report the split as a descriptive identity. State that finished responses are a selected subset and that the comparison cannot show skill was preserved | The claim "R-s202 lost no skill, only finishing" is withdrawn. Supported claim: the s202 difference appears through truncation; whether R's unfinished attempts would have been as accurate is unknown (the extension recovered a smaller share for R, 16.2% vs 19.8%) | report (§4.3, Q2; A26 §4 item 3) |
| I3 | "In both seeds the endpoint sign of R − U appeared between rounds 32 and 64" (L150, L338); "R and U separated soon after the cap change, in opposite directions" (L285); "Both runs of each pair split in the first window after the cap change … That is what run-to-run path differences look like" (L180–181) | `comparison.md` S11, M1. R − U (points), s202 dev / J_H / J_U: r0 −1.9 / −2.3 / −1.2 (identical base weights), r32 −1.2 / −2.0 / −1.6, r64 −6.6 / −7.8 / −3.9. Every r32 interval contains 0. J_R s202: first r64. s101: first r64 on dev, J_H, J_U (J_U holds from r128), J_R r128 (negative). Cap 8,192 at r0, r32; 12,288 from r64. Training success on shared hard prompts (production `divergence.json`): s202 R − U negative from bin 25–32 (−0.2) | Per seed and panel, report the first trained evaluation with the round-128 sign and the round from which it holds, with round 0 beside it as the noise reference (table in `pass-b/train.md` §3) | Seed 101: the ordering appears at r64 on dev, J_H, J_U (the cap change and training are confounded in that window). Seed 202: U is ahead from the first trained evaluation (r32) on dev, J_H, J_U, before the cap change, but by no more than the round-0 gap between identical policies. The claim that both seeds split in the post-cap window, used as evidence for run-to-run path differences, is not supported; the s202 timing is unresolved at n = 8 × 53 | report (Q3, §3.2, Q4; A26 §4 item 2); DEF v1.1 C10 |
| I4 | "practice of either kind was a better use of the slots than more hard problems" (L33–34); "Practice of either kind: yes" (L81); "both are positive in both seeds" read as replication (L94–95); two-seed intervals R − H +3.6 [+1.6, +5.8], U − H +3.8 [+1.6, +6.0] (L32) | Sealed test per seed: R − H +4.9 [+2.0, +8.0], **+2.4 [−0.4, +5.4]**; U − H **+0.7 [−2.0, +3.5]**, +6.8 [+3.5, +10.2]. Stress: R − H s202 [−0.1, +6.3], U − H s101 [−1.4, +3.8] also contain 0. The two-seed intervals resample problems only (DEF §3.C); paired t with df 1: R − H 3.65 ± 16.0, U − H 3.78 ± 38.8 points. H-s202 cells mixed GPU types (test 63 of 128 problems on A6000; stress L40/A6000); the R-s202 L40 → L40S rerun moved pass@1 by +1.34 [+0.22, +2.56]. R − H s202 against the r116 fallback: +0.46 [−2.15, +3.17] | Describe the observed allocation trade-off: in these two runs per arm the R − H and U − H point estimates were positive, two of four per-seed intervals include 0, H generated more tokens (68.6 and 74.7 M vs 52.8–65.1 M). Present two-seed intervals as problem-resampling only | No general curriculum recommendation follows. An interval that contains 0 is not evidence of equivalence and not of advantage. The direction "practice ≥ more hard problems" is an observation in 2 runs per arm with hardware confounds in the H-s202 cells | report (opening, Q1, Q4; A26 §4 item 4) |
| I5 | "62–92% of R and U practice groups were all-correct and carried no learning signal. The practice slots therefore mostly cost generation time without updating the model" (L36–37); "gave no gradient. The arms therefore differed little in what they trained on" (L184–185); "The practice prompts stopped giving a learning signal" (L434) | Both passes: momentum-only steps (all 4 groups non-mixed, pre-clip gradient exactly 0, AdamW still steps) per run of 512: H-s101 18, R-s101 66, U-s101 53, H-s202 20, R-s202 25, U-s202 65; identical (round, minibatch) sets. Every R/U momentum-only step contains 2 all-correct practice groups (`pass-b/train.md` §4). All-correct groups still count in the 1/32 loss divisor (`run.py:302`) | All-correct practice groups contribute no direct gradient. Every optimizer step still moved the weights (the minibatch's mixed hard groups, or the stored AdamW moments). Retire "learning signal" (DEF §3.D) | "Without updating the model" is false. The practice slots shaped every update: they halve the share of mixed groups in each minibatch (1/32 scaling) and produced 25–66 momentum-only steps in R/U | report (opening 2, Q1, Q4, §4.4); DEF v1.1 C11 |
| I6 | Ray-interference chain: "Link 1 … was not detected. One unrelated update did not lower the hard probe by 3 points or more" (L50–52); Link 2 "observed rarely and briefly" (L52–54); "Link 3 … was not observed" (L55); "The competing explanation that fits the same evidence best is run-to-run variation" (L57) | Diagnostic (`pass-b/train.md` §5, `pass-a/code.md` D10): 4 draws per branch type; permutation p ≥ 0.074 in all 6 cells; reference gap `none_rerun` − `none` J_H +3.1 (r0) and +2.5 (r64) points, larger than the range of type means in 5 of 6 cells; at r64 two of four "related" draws had zero gradient. Link 2: losses on training prompts are 0.65–1.07 × the binomial-sampling expectation (no excess). Link 3: no record defines or measures a stall. Connected on one trajectory: R-s101 r64 diagnostic source, R-s101 in-run curve and training records. Never connected: ΔJ and later R-s101 rounds (no checkpoint between 64 and 128), ΔJ and loss/recovery (different prompts), U and H states | Remove the "3 points" claim. Present three separate kinds of evidence: (a) local one-step ΔJ: not resolvable from reference noise; (b) loss and recovery on training prompts: at the rate sampling alone produces; (c) sustained stalls: not defined, not measured. List what was measured on the same trajectory and what was never linked | "Not detected" stays, but means "the test could not resolve an effect of this size"; it does not show the absence of interference. Run-to-run variation is not the best-fitting explanation by any test; it is the explanation the data cannot exclude, and it now includes a hardware confound (I8) | report (opening 4, §6 table; A26 §4 item 5) |
| I7 | Local diagnostic reported without its cap and with one reference (§6, L633–664; `delta_j.png`); "the direct branch comparisons above do not depend on it" (L651) | Cap 8,192 at both labels (`diag_local.py:121` @4b47386; all truncated attempts have 8,192 tokens); R-s101 r64 trained at 12,288. r64 J_H: 36.1% of 7,680 diagnostic attempts capped; no-update J_H 60.5% (n 16, cap 8,192) vs in-run 73.8% (n 8, cap 12,288). Mean ΔJ_H is positive against `none` and negative against `none_cycled` (= `none_rerun`, same values) at both labels. The references ran on different GPUs (r0: L40 vs L40S; r64: RTX 6000 Ada vs L40S). Three r64 branches (related 4101, 4102; unrelated 4103) were momentum-only and gave identical evaluations | State the cap and the GPU of every reference. Plot each branch draw with both references. Branch-vs-branch differences (unrelated − related −2.2 / −0.7) are reference-free but within the draw spread | ΔJ at r64 measures change under an 8,192-token budget, not the training budget. The sign of every type-mean ΔJ_H depends on the chosen reference, so no "update lowered/raised J_H" statement is supported | report (§6, figure); DEF v1.1 C7 |
| I8 | Training GPU type not stated; "tokens are the hardware-free cost" (L112); run-to-run variation as the explanation (L57, L75, L180) | New, verified by both passes (`metrics.jsonl` start events; `sacct`): H-s101 L40 (all rounds); R-s101, U-s101, U-s202 L40S; H-s202 L40 rounds 1–60 then L40S 61–128; R-s202 L40 rounds 1–16 then L40S 17–128. Evaluation: the R-s202 L40 → L40S rerun moved pass@1 +1.34 [+0.22, +2.56] | State the training GPU of every segment. Within seed 101, H trained on L40 and R/U on L40S; within seed 202, all three switched hardware at different rounds | Arm contrasts are not hardware-matched in training. "Run-to-run variation" includes GPU-kernel differences that cannot be separated from training-path differences with these runs | report (§1/§4 design, appendix); DEF v1.1 C5 |
| I9 | "more factoring practice would plausibly fix about 40% of the failures" (L69–71); "Fixable by factoring practice" column (§7.3) | Audit counts verified (96 responses, 192 reads, 35 reward 0 of which 13 capped, 33 after removing the 2 both-read grader cases; 13 both-yes of 33; 29 yes of 66 judged reads). Packets labelled every reward-0 response "WRONG", capped ones included, and showed responses over 8,000 characters as excerpts (`build.py:39-41`) | Report the reads as annotations about a hypothetical intervention, with the excerpt and "WRONG" caveats | Not an observed training effect; no claim about what more factoring practice would do | report (opening 5, §7.3; A26 §4 item 6) |
| I10 | Stress cell "H-s202 r64" (L724) and "H-s202 −2.3 from r64 to r128" without trajectory | Both passes: the cell used `round_0064` saved 2026-11-11 03:26:38Z by job 11112377 (`37963fac…`), trajectory a2-r60. a2-orig round-64 weights were never saved (`round_0064.tmp` only). a2-orig − a2-r60 at r64: dev +2.1 [−1.4, +5.7], J_H +3.9 [−0.8, +8.6] | Label the cell "H-s202 r64 (a2-r60)". Say that the a2-orig round-64 policy (dev 65.8%) has no weights | The r64 → r128 stress change refers to the a2-r60 trajectory only; DEF §4 item 5 confirmed | report (§7.2); DEF v1.1 C4 |
| I11 | "Within a seed, R/U have identical hard schedules" read as identical hard training | Verified on data (both passes): shared problem IDs and request seeds identical in 128/128 rounds per seed; same-seed round-1 shared attempts token-identical in only 0–11 of 64; H and R/U put shared groups into the same minibatch pairs in 0/128 (s101) and 2/128 (s202) rounds; predicted plan matches `valid_tokens` 512/512 | "Equal hard exposure" = same problems, slot order and request seeds; attempts and minibatch grouping differ | Minor: H vs R/U differences include grouping of the shared groups, not only the second pool | report (§1 design); DEF v1.1 C9 |

### 1.2 Changes numbers

| # | Original claim | Evidence | Correction | Scientific consequence | Fixed where |
|---|---|---|---|---|---|
| N1 | "the five finished trained policies scored 74.8%–81.9%, a gain of +22 to +29 points" (L23–24, L92–93) | All six runs finished; H-s202 r128 = 2,965/4,096 = 72.39% (both passes) | Six r128 endpoints 72.4–81.9%; gains over base (52.6%) +19.8 to +29.3. Name the checkpoint set with every range | Number only; "trained − base is large in every run" holds | report (A26 §4 item 1) |
| N2 | Stress: "trained checkpoints 62.4–69.5%" (L66); "SuperHard … 52.1–58.4%" (L67, L784); "gains 25–32 points" (L783); "loses 2–5" (L785); "pass@8 … differs by at most 3.7 points between arms" (L788). §4.2: "82.5–88.0%", "+21.7 to +28.8", "+20.4 to +25.3" (L399–401) | H-s202 r128 stress 60.12% overall, 49.40% SuperHard, gain +22.85, loses 8; R vs U pass@8 gap max 3.62, with H 4.98. §4.2 with H-s202: 79.6–88.0, +19.3 to +28.8, +17.9 to +25.3 | Eight trained stress cells 60.1–69.5% (six r128 cells the same), SuperHard 49.4–58.4%, gains +22.9 to +32.2, losses 2–8, pass@8 gap R vs U ≤ 3.6 (≤ 5.0 incl. H) | Number only; "SuperHard roughly doubled" holds (×1.9–2.3) | report (A26 §4 item 1) |
| N3 | "Every checkpoint gains 2.4–2.9 points" (L577); "gains 3.2 points" H-s101 (L584); "55–72% are still capped" (L579) | Stress length rows (both passes): gains 2.32 (U-s202) to 3.11 (H-s101); H-s101 0.6414 → 0.6725 = +3.11; still capped 54.48–71.67% | 2.3–3.1 points (R/U 2.3–2.9); H-s101 +3.1; 54–72% | Number only | report |
| N4 | 24 displayed equations `a − b = c` fail at displayed precision (appendix B); 4 more displays disagree with their inputs: §3.2 s202 r96 "64.2% \| 67.9% \| −3.8" (−3.7 displayed), §4.3 "−7.0 … (68.6% vs 75.7%)" (−7.1), "at most 3.7" (3.62), "2.4–2.9" and "3.2" (N3) | `pass-a/numbers.md` §3.6; every printed c is the correctly rounded unrounded difference | Print the two absolute scores and the independently rounded contrast in separate columns with one rounding note | Number display only | report (A26 §4 item 7) |
| N5 | r64 no-update J_H "60.6%" (L643, L646); DEF §4 item 1 "0.606" | `research/evidence/a15/diag_r64.jsonl`: 0.60546875 | 60.5% (0.605) | None | report; DEF v1.1 C7 |
| N6 | Per-run "canonical" GPU-h 29.5 / 27.9 / 22.6 / 22.8 / 25.6 / 31.1 (L102–110, L453–460, L867–877); "R and U … 22.6–27.9 GPU-h on their canonical trajectories" (L35); U-s202 chain includes 11097811 (L904) | `comparison.md` M7, §3: the figures are whole-job allocations of the chain. U-s202 includes job 11097811 (3.75 GPU-h; rounds 65–81, all superseded) and ≥ 2.38 GPU-h of superseded rounds; R-s101 ≥ 1.87; H-s202 ≥ 1.55 (a2-orig rounds 61–64) + 0.03 (job 11112338). Study totals 315.6 GPU-h / 2,555.5 CPU-h / 260 jobs reproduce exactly | Canonical training GPU-h (upper bound, pass B): H-s101 29.5, R-s101 26.0, U-s101 22.6, H-s202 29.5, R-s202 22.3, U-s202 19.5; R/U range 19.5–26.0. Drop 11097811 from the canonical chain | Number only; GPU-h already mixes GPU types (tokens are the cost measure) | report (Q1, §4.4, §9) |
| N7 | R-s202 prompt tokens "1.48 M *" from the run counter (L108–111) | Stale counter (D127): R-s202 round-row counters `gen_tokens`, `prompt_tokens`, `hard_responses` and step-row `opt_step` (77–524) are offset after round 16 (`pass-a/code.md` D5; `pass-b/train.md` §4) | Recompute prompt tokens from the 128 canonical record files (not yet done) | Number only (small) | report; DEF v1.1 C2 |
| N8 | "One grader limitation was confirmed … at most 7 of 1,768 attempts in any stress cell … at most 3 attempts between R and U" (L835–838) | Reconcile §3.2 (project verifier): stress r128 cells have 2–13 finished reward-0 attempts whose earlier box is verifier-equivalent to gold or whose boxes jointly give the gold; R-s202 12 vs U-s202 2. A second limitation: multiple-choice stress items with numeric gold (`math:number_theory/train/00662`, `00656`) where the last box is the option letter (6 of R-s202's 9 earlier-box cases) | Report both limitations with the counts. Keep the frozen last-box rule primary | Primary numbers unchanged. Under the lenient "any box" rule, stress s202 R − U moves −2.66 [−5.09, −0.23] → −2.09 [−4.52, +0.34]; its exclusion of 0 is not robust to the grading rule. Sealed-test R − U is unchanged (≤ 0.00) | report (§7.3) |

### 1.3 Wording, provenance and code

| # | Original claim | Evidence | Correction | Scientific consequence | Fixed where |
|---|---|---|---|---|---|
| W1 | "Each checkpoint made 8 attempts … on L40S" (L710); "H-s202 round 128 runs only after that training run ends (section 9)" (L711–712) | `sacct` 11120596, 11120598, 11121110: H-s202 r128 stress shards k0, k2 L40, k1 RTX A6000; the cell is in the table | All stress cells L40S except H-s202 r128 (L40 × 2, A6000 × 1); delete the stale sentence | None | report |
| W2 | DEF §4 item 12: "Shards k2–k5 ran on any 48 GB GPU" | `sacct` 11120634 (L40S), 11120590/1/5 (RTX A6000); recipe allowed any 48 GB GPU | k0–k2 L40S (65 problems), k3–k5 RTX A6000 (63); add the stress shard GPUs | None (report v2 L245 is right) | DEF v1.1 C1 |
| W3 | DEF §4 item 6: stale counters in `state.json` only | R-s202 metrics step rows carry `opt_step` 77–524 for rounds 17–128 (+12) | Add step rows and round-row counters; index R-s202 steps by (round, minibatch) | None found (no reported number uses the counters except N7) | DEF v1.1 C2 |
| W4 | DEF §3.B R-s202 segment "1–16 + 17–128" | Job 11081729 also wrote rounds 17–19 and 3 of 4 step rows of round 20 before it died (all superseded); U-s202 job 11097811 wrote rounds 65–81, none canonical; H-s202 job 11112338 wrote no row | List each job's written and canonical rows | None | DEF v1.1 C3 |
| W5 | DEF §5 item 7: a2-orig recipe (expsha 97c04ad0) "is not in the workspace" | Preserved as `~/ae-qis/jobs/qis-main-H-s202-sym-7c44fa5-97c04ad04599.sh`; differs from the resumed recipe only by the `resume_audit` phase and `retain_rounds` `[]` → `[64]` | Point to the preserved job script | None | DEF v1.1 C4 |
| W6 | "held-out" dev and sealed test (L285, L294; DEF §3.A "dev … after removing near-duplicates of test") | `pass-a/data-inventory.md` §4: dev #42 `algebra/train/01291` is H_train `algebra/train/01269` with new constants (masked 3-gram Jaccard 1.0); sealed test `intermediate_algebra/test/00383` is a close variant of H_train `intermediate_algebra/train/00301` (0.725); `00891` ↔ `00048` 0.500 (phrasing only). Splits rule (5-shingle ≥ 0.7) sees none | "Held out by ID and by the 5-shingle rule; 2 close variants of training problems (1/53 dev, 1/128 test) remain" | None: removing them moves no conclusion (§3.1: per-seed contrasts ≤ 0.27 points, gains −0.27 to −0.53 points, dev ≤ 0.77 points) | report (data table); DEF v1.1 C6 |
| W7 | Base sealed test 52.6% with no unknown note | 2 of 4,096 base attempts verifier-unknown (`intermediate_algebra/test/00030`, samples 8, 29), counted incorrect | State it with the bound (§3.3) | None (≤ 0.05 points pass@1) | report |
| W8 | "Extra tokens per recovered answer" (§5.2–5.4) | e.g. R-s202: 1,781,371 new tokens on 538 extended attempts / 87 recovered = 20,476; the 87 recovered used 1,767 new tokens each | Rename "new tokens on all extended attempts per recovered answer" or give both | None | report |
| W9 | Same contrast printed with different intervals in different sections (e.g. s101 R − U [+2.1, +6.3] §2 vs [+2.1, +6.2] §5.2; 9 pairs, `pass-a/numbers.md` §5) | Separate bootstrap streams per analysis; all within Monte-Carlo error | One stream per contrast, or one printed interval | None | report |
| W10 | Report stated the tranche-1 s202 R − U interval [−9.0, +4.3] | Pass A [−8.6, +4.7]; reconcile: bounds move by one lattice step (1/256) across 8 bootstrap seeds (`comparison.md` M2) | Note the discreteness for 64 × 4 panels | None | report (optional) |
| W11 | DEF §3.A J_R, J_U: no levels stated; cue study: no label note; stress: "unused problems" (L683) | J_R 14/28 and J_U 16/32 are MATH levels 1–3; 21/24 cue items have a pass-1 label only; stress MATH items come from study splits (44 construction, 8 study_dev, 21 calib, 2 excluded, 54 test) but none within masked-3-gram 0.5 of a study item | Add the facts | None | DEF v1.1 C6 |
| W12 | "33 genuine failures" (L69–70, §7.3) | Both-reads `grader_issue` rule gives 33; any-read rule gives 31 | Name the rule | None | report; DEF v1.1 C8 |
| W13 | DEF §1 / pass A `eval-map.md`: "594d408 adds 21 lines to 4b47386" (`diag_local.py`) | `git diff 4b47386 594d408 -- src/qis/diag_local.py` empty; 042cebd (reference job) adds 22 lines (`--reference-only`) | `diag_local.py` identical for r0 and r64 branches; the reference job used 042cebd | None | DEF v1.1 C7 (audit record corrected here) |
| W14 | Cue: "+9 to +15 points more mentions of 'factor'" (L853) | Production `cue_analysis.json` only (+8.85 to +15.10); texts not re-decoded by either pass | Label as single-source | None known | report |
| W15 | DEF §4 items 2, 7, 8, 10, 11, 13, 15, 16, 17 (two J_H measurements; round conventions; two pass@8 estimators; senses of "hard"; two "minibatch seed" meanings; audit packet labels; π_old; checkpoint cadence; stale DATASET.md/D151) | Confirmed by the passes; no new evidence against them | Keep in v1.1; apply in the report | None beyond what each item states | report |
| W16 | (code) `train_summary.py` averages all step rows and lists duplicate eval rows | `pass-a/code.md` D7; feeds monitoring, not the report | Take the last row per key | None on published numbers | code |
| W17 | (code) `run.py:216-220` skips an evaluation whose records file exists | `pass-a/code.md` D6; both affected resumes were handled by moving files; every canonical eval file written by its retained segment | Key the skip on an `eval` row after the last `resume` event | None now; latent | code |
| W18 | (code) weight-load check sums 4 non-fused tensors (`evaluate_ckpt.py:54-63`) | `pass-a/code.md` D9; file identity is checked by sha256 in every precheck | Check fused tensors or compare logits | None found | code |
| W19 | (provenance) standby analyses outside the ledger; 45 recipe versions exist only in `~/ae-qis/jobs/` (on `/home`) | `pass-a/code.md` D8, D11; `jobs.md` §5 | Copy `~/ae-qis/jobs/` into the workspace; record the commit in each analysis output | None | workspace |

## 2. A26 §4 acceptance items

| A26 §4 item | Verdict | Rows |
|---|---|---|
| 1. Ranges omit H-s202 (sealed 72.4%; stress 60.1%, SuperHard 49.4%); audit covered five trained policies | Defect verified. Corrected ranges use named sets. The response audit did cover base + 5 trained policies (H-s202 absent): verified | N1, N2 |
| 2. Earliest observed ordering per seed and panel | Defect verified for seed 202: U ahead from r32 on dev, J_H, J_U (r0 noise of the same size); seed 101 from r64; no matched intermediate R/U stress checkpoint | I3 |
| 3. "More room does not rescue those answers" vs 16.2% / 19.8% rescues | Defect verified. Absolute success rose for every checkpoint; the ordering stayed. Accuracy among finished cannot establish preserved skill | I1, I2 |
| 4. R − H, U − H intervals including zero | Verified: 2 of 4 test intervals and 2 of 4 stress intervals include 0. Recommendation language removed | I4 |
| 5. Interference mechanism | Defect verified. "3 points" claim unsupported; diagnostic at cap 8,192 with reference-dependent sign; loss/recovery at sampling rate; stalls undefined; links never connected listed | I6, I7 |
| 6. Audit denominators and "fixable" | Denominators verified (96, 35, 2 reads, 33 by the both-reads rule; 31 by any-read); excerpts and capped "WRONG" labels confirmed; "fixable" is an annotation | I9, W12 |
| 7. Contrasts from unrounded data; failing equations | 24 of 107 equations fail; 4 more displays; every printed contrast is correct when computed unrounded | N4 |

## 3. Sensitivity checks (reconcile, compute, records only)

### 3.1 Close variants of training problems (`tools/reconcile/sens_heldout.py` → `sens_heldout.json`)
Removing sealed-test `intermediate_algebra/test/00383` (127 problems). Per-problem counts on 00383: base 3/32, endpoints
21–32/32. Bootstrap: 10,000 draws, seed 20261112, same draws for the full and reduced sets.

| Quantity | 128 problems | 127 (no 00383) | change |
|---|---|---|---|
| base / H-s101 / R-s101 / U-s101 | 52.56 / 76.95 / 81.86 / 77.69 | 52.90 / 76.80 / 81.72 / 77.58 | +0.34 / −0.16 / −0.14 / −0.10 |
| H-s202 / R-s202 / U-s202 | 72.39 / 74.78 / 79.22 | 72.19 / 74.85 / 79.06 | −0.19 / +0.07 / −0.16 |
| R − U s101 / s202 | +4.17 [+2.03, +6.25] / −4.44 [−7.25, −1.73] | +4.13 [+1.97, +6.27] / −4.21 [−6.94, −1.48] | −0.04 / +0.24 |
| R − H s101 / s202 | +4.91 [+2.03, +7.84] / +2.39 [−0.42, +5.35] | +4.92 [+1.94, +8.00] / +2.66 [−0.15, +5.61] | +0.01 / +0.27 |
| U − H s101 / s202 | +0.73 [−2.00, +3.44] / +6.84 [+3.54, +10.21] | +0.79 [−1.97, +3.64] / +6.87 [+3.59, +10.29] | +0.06 / +0.03 |
| cgpu R − U s101 / s202 | +4.44 / −3.10 | +4.43 / −2.95 | −0.01 / +0.15 |
| endpoint − base (six) | +19.82 to +29.30 | +19.29 to +28.81 | −0.27 to −0.53 |

No interval changes its relation to 0. Removing also `00891` (126 problems) gives the same picture (largest contrast
change +0.47 for s202 R − U; R − H s202 [−0.10, +5.70]). Dev without `algebra/train/01291` (52 problems; not in J_H,
which is dev[0:32]; round-0 evaluations solve it 2–3/8, trained evaluations 5–8/8): dev pass@1 changes −0.77 to +0.33 points over all
31 (run, round) cells; R − U dev changes by ≤ 0.50 points and keeps its sign in all 10 (seed, round) cells, so the
earliest-ordering table (I3) is unchanged.

### 3.2 Multiple boxes (`tools/reconcile/sens_multibox.py` → `sens_multibox.json`, `sens_multibox_hits.json`)
Rule: for every finished attempt with reward 0 (unknown and capped stay 0), decode the stored ids, take every
`\boxed{}`/`\fbox{}` content before the last one, and test it against gold with the project verifier
(`qis.verify.Verifier`, unchanged 779c54d → 9b2fd05; 2,701 verifier calls, 0 unknown). "Split" = no single box
matches but the distinct box contents joined by ", " (all, or the last 2–4) verify. "Any box" pass@1 counts both as
correct. **The frozen rule (last box only) stays primary.** The any-box rule is an upper bound on grader loss: it also
credits responses that changed their answer (for example base `00373`: boxes `2` then `\infty`).

| Cell | Finished, reward 0 | with ≥ 2 boxes | earlier box = gold string | earlier box ≡ gold (verifier) | split across boxes | pass@1 last box | pass@1 any box | change |
|---|---|---|---|---|---|---|---|---|
| test base | 1,939 | 1,007 | 5 | 7 | 3 | 52.56 | 52.81 | +0.24 |
| test H-s101 | 499 | 347 | 8 | 13 | 2 | 76.95 | 77.32 | +0.37 |
| test R-s101 | 306 | 226 | 0 | 2 | 0 | 81.86 | 81.91 | +0.05 |
| test U-s101 | 413 | 91 | 1 | 2 | 0 | 77.69 | 77.73 | +0.05 |
| test H-s202 | 499 | 394 | 1 | 1 | 0 | 72.39 | 72.41 | +0.02 |
| test R-s202 | 495 | 103 | 1 | 1 | 0 | 74.78 | 74.80 | +0.02 |
| test U-s202 | 518 | 104 | 1 | 1 | 0 | 79.22 | 79.25 | +0.02 |
| test H-s202 r116 fallback | 447 | 325 | 2 | 2 | 0 | 74.32 | 74.37 | +0.05 |
| test cgpu R-s101 / R-s202 | 279 / 472 | 209 / 111 | 1 / 1 | 3 / 1 | 0 / 0 | 82.25 / 76.12 | 82.32 / 76.15 | +0.07 / +0.02 |
| test cgpu U-s101 / U-s202 | 404 / 518 | 90 / 104 | 2 / 1 | 3 / 1 | 0 / 0 | 77.81 / 79.22 | 77.88 / 79.25 | +0.07 / +0.02 |
| stress base | 1,108 | 535 | 10 | 11 | 0 | 37.27 | 37.90 | +0.62 |
| stress H-s101 r128 | 321 | 201 | 6 | 9 | 4 | 64.14 | 64.88 | +0.74 |
| stress R-s101 r128 | 177 | 107 | 3 | 4 | 2 | 69.51 | 69.85 | +0.34 |
| stress U-s101 r128 | 207 | 43 | 1 | 1 | 5 | 65.33 | 65.67 | +0.34 |
| stress H-s202 r128 | 254 | 199 | 3 | 4 | 1 | 60.12 | 60.41 | +0.28 |
| stress R-s202 r128 | 244 | 65 | 8 | 9 | 3 | 63.18 | 63.86 | +0.68 |
| stress U-s202 r128 | 336 | 58 | 0 | 0 | 2 | 65.84 | 65.95 | +0.11 |
| stress R-s101 r64 | 170 | 107 | 5 | 8 | 8 | 64.25 | 65.16 | +0.90 |
| stress H-s202 r64 | 398 | 248 | 8 | 9 | 2 | 62.39 | 63.01 | +0.62 |

The "earlier box = gold string" column reproduces pass B's 17 sealed-test cases exactly (5, 8, 0, 1, 1, 1, 1). The
verifier finds 27 such cases in the 7 primary test cells (28,672 attempts) and 5 split answers (base 3 on
`intermediate_algebra/test/00025`, H-s101 2). On stress, 6 of R-s202's 9 earlier-box cases are one multiple-choice item
(`math:number_theory/train/00662`, gold `1`, last box `B`); `00656` (gold `16`, last box `C`) adds 2 in R-s202 and 1
in R-s101. Split answers concentrate on `math:precalculus/train/00152` (gold `2,4`) and `omni:omni/03336`.

| Contrast (points) | Last box (frozen) [95%] | Any box [95%] | change |
|---|---|---|---|
| test R − U s101 / s202 | +4.17 [+2.08, +6.27] / −4.44 [−7.23, −1.78] | +4.17 [+2.08, +6.25] / −4.44 [−7.20, −1.78] | 0.00 / 0.00 |
| test R − H s101 / s202 | +4.91 [+2.00, +7.89] / +2.39 [−0.42, +5.37] | +4.59 [+1.71, +7.54] / +2.39 [−0.42, +5.37] | −0.32 / 0.00 |
| test U − H s101 / s202 | +0.73 [−1.90, +3.54] / +6.84 [+3.49, +10.28] | +0.42 [−2.17, +3.15] / +6.84 [+3.52, +10.25] | −0.32 / 0.00 |
| test cgpu R − U s101 / s202 | +4.44 / −3.10 | +4.44 / −3.10 | 0.00 / 0.00 |
| stress R − U s101 / s202 | +4.19 [+1.53, +6.79] / −2.66 [−5.09, −0.23] | +4.19 [+1.58, +6.79] / **−2.09 [−4.52, +0.34]** | 0.00 / +0.57 |
| stress R − H s101 / s202 | +5.37 [+2.60, +8.20] / +3.05 [−0.11, +6.39] | +4.98 [+2.21, +7.75] / **+3.45 [+0.23, +6.84]** | −0.40 / +0.40 |
| stress U − H s101 / s202 | +1.19 [−1.53, +3.90] / +5.71 [+2.38, +9.16] | +0.79 [−1.87, +3.39] / +5.54 [+2.21, +8.99] | −0.40 / −0.17 |

Reading: on the sealed test the box rule moves no R − U contrast and moves s101 R − H and U − H by −0.3 points (H-s101
has the most multi-box answers). On stress it changes two s202 interval verdicts (bold); every sign is unchanged.

### 3.3 Verifier-unknown (`sens_heldout.json` → `unknown_test`)
Only the base sealed-test cell has unknown status: 2 attempts on `intermediate_algebra/test/00030` (that problem is
0/32 as scored). Counting both as correct: pass@1 0.525635 → 0.526123 (+0.049 points), pass@8 0.8352 → 0.8387, solved
≥ 1 of 32 116 → 117. Every endpoint − base gain shrinks by at most 0.049 points. No trained cell, stress cell or cue
cell has an unknown status (both passes). Effect ≤ 0.05 points: verified.

## 4. Counts

| Severity | Rows |
|---|---|
| Changes interpretation | 11 (I1–I11) |
| Changes numbers | 8 (N1–N8) |
| Wording, provenance, code | 19 (W1–W19) |
| **Total** | **38** |

## Appendix B. Displayed equations that fail at displayed precision (report v2)

Source: `pass-a/data/numbers.json` → `recomputed.equations` (107 equations scanned, 24 fail). In every case the printed
contrast c is the correctly rounded unrounded difference; the inputs a, b were rounded separately.

| # | Line | Printed | a − b of the displayed values |
|---|---|---|---|
| 1 | 68 | 63.2 − 65.8 = −2.7 | −2.6 |
| 2 | 90 | 78.3 − 78.5 = −0.1 | −0.2 |
| 3 | 130 | 63.2 − 65.8 = −2.7 | −2.6 |
| 4 | 130 | 66.0 − 68.2 = −2.1 | −2.2 |
| 5 | 163 | 58.4 − 54.2 = +4.3 | +4.2 |
| 6 | 234 | 78.3 − 78.5 = −0.1 | −0.2 |
| 7 | 258 | 82.3 − 77.8 = +4.4 | +4.5 |
| 8 | 258 | 77.8 − 77.0 = +0.9 | +0.8 |
| 9 | 274 | 82.3 − 77.8 = +4.4 | +4.5 |
| 10 | 373 | 85.5 − 88.3 = −2.9 | −2.8 |
| 11 | 550 | 84.2 − 79.2 = +4.9 | +5.0 |
| 12 | 551 | 80.6 − 79.2 = +1.3 | +1.4 |
| 13 | 571 | 86.1 − 81.8 = +4.2 | +4.3 |
| 14 | 572 | 58.4 − 54.2 = +4.3 | +4.2 |
| 15 | 573 | 63.2 − 65.8 = −2.7 | −2.6 |
| 16 | 573 | 66.0 − 68.2 = −2.1 | −2.2 |
| 17 | 575 | 56.0 − 58.5 = −2.6 | −2.5 |
| 18 | 587 | 68.2 − 67.3 = +1.0 | +0.9 |
| 19 | 606 | 67.6 − 69.5 = −2.0 | −1.9 |
| 20 | 606 | 71.1 − 73.0 = −2.0 | −1.9 |
| 21 | 645 | 40.5 − 41.5 = −0.9 | −1.0 |
| 22 | 646 | 60.9 − 60.6 = +0.2 | +0.3 |
| 23 | 773 | 63.2 − 65.8 = −2.7 | −2.6 |
| 24 | 778 | 69.5 − 64.3 = +5.3 | +5.2 |

Other displays with the same problem: §3.2 table seed 202 r96 "64.2% | 67.9% | −3.8" (displayed difference −3.7);
§4.3 "−7.0 … (R 68.6% vs U 75.7%)" (−7.1); §7.2 "at most 3.7 points" (3.62 → 3.6); §5.3 "2.4–2.9" and "3.2" (N3).
