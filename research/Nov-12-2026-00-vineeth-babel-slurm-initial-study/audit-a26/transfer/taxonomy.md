# A26 x-transfer — failure taxonomy of the in-run probe attempts

Purpose: find out what kind of failure disappears when J_U and J_R rise in every arm. Written 2026-11-11 (EST) by agent x-transfer.

**Main result.** The J_U and J_R gains are fewer *wrong final answers*. They are not format repairs, not scoring strictness
and not the cap change. Format failures are absent at every round. Lenient re-scoring rescues at most 5 of 224–256 attempts per
cell. The gains survive almost unchanged when round-128 success is restricted to 8,192 tokens. The same problems improve in H, R and U.
This rules out E1 (format/termination) and E3 (cap) as the main cause, and makes E5 (noise) unlikely. It cannot separate E2
(general carefulness) from E4 (skill overlap); the control panel (`control.md`) does that.

## Method
- Records: `$P/runs/qis-main-{H,R,U}-s{101,202}-sym/records/eval_{probe_J_U,probe_J_R,dev}_r{0000,0032,0064,0096,0128}.jsonl.gz`
  (90 cells, 0 missing). J_H = the 32 `probe_J_H` IDs taken from the dev file. n = 8 per problem. Golds from
  `$P/data/main-v1/bundle_main_v1.json`.
- Text = `tokenizer.decode(ids, skip_special_tokens=True)` with the Qwen3-1.7B tokenizer, as in `run.py:124`.
- Classes (exclusive, in order): correct (reward 1); unknown (reward None); truncated (finish `length`); no box (no `\boxed{`/`\fbox{`);
  unbalanced last box; multi-box wrong (≥ 2 boxes, reward 0); single-box wrong (1 box, reward 0).
- Lenient re-scores (project verifier `qis.verify.Verifier`, compute): any box (not only the last) matches the gold; the last
  number in the text matches the gold. Also "correct ≤ 8,192": correct with `n_tokens` ≤ 8,192.
- Tools: `tools/taxonomy.py` (compute, standby 11122432, 2026-11-11 21:12 EST, 70 s), `tools/transfer_analysis.py taxonomy`
  (laptop, numpy; paired problem bootstrap, 10,000 resamples, seed 2026). Raw: `taxonomy.json`; tables: `taxonomy_summary.json`.
  Multi-box detail: `tools/multibox_sample.py` → `multibox.json` (H-s101 and R-s202, rounds 0 and 128).
- All numbers **verified** (computed from the records above).

## 1. Classes, round 0 → 128 (share of attempts)
In-run caps: 8,192 at rounds 0 and 32; 12,288 at round 128 (DEFINITIONS v1 §3.C response cap).

### J_U (32 problems, 256 attempts per cell)
| run | pass@1 r0 → r32 → r128 | truncated r0 → r128 | no box r0 → r128 | multi-box wrong r0 → r128 | single-box wrong r0 → r128 | lenient rescues r0 / r128 |
|---|---|---|---|---|---|---|
| H-s101 | 0.805 → 0.836 → 0.938 | 0.004 → 0.012 | 0.000 → 0.000 | 0.059 → 0.023 | 0.133 → 0.027 | 0 / 0 |
| H-s202 | 0.805 → 0.879 → 0.902 | 0.000 → 0.031 | 0.000 → 0.000 | 0.086 → 0.043 | 0.109 → 0.023 | 0 / 0 |
| R-s101 | 0.812 → 0.871 → 0.961 | 0.000 → 0.008 | 0.000 → 0.000 | 0.074 → 0.020 | 0.113 → 0.012 | 0 / 0 |
| R-s202 | 0.805 → 0.863 → 0.918 | 0.000 → 0.023 | 0.000 → 0.000 | 0.086 → 0.020 | 0.109 → 0.039 | 0 / 0 |
| U-s101 | 0.816 → 0.891 → 0.941 | 0.000 → 0.020 | 0.000 → 0.000 | 0.062 → 0.012 | 0.121 → 0.027 | 0 / 0 |
| U-s202 | 0.816 → 0.879 → 0.941 | 0.000 → 0.016 | 0.000 → 0.004 | 0.062 → 0.004 | 0.121 → 0.035 | 0 / 0 |

### J_R (28 problems, 224 attempts per cell)
| run | pass@1 r0 → r32 → r128 | truncated r0 → r128 | no box r0 → r128 | multi-box wrong r0 → r128 | single-box wrong r0 → r128 | lenient rescues r0 / r128 |
|---|---|---|---|---|---|---|
| H-s101 | 0.808 → 0.821 → 0.893 | 0.000 → 0.000 | 0.000 → 0.004 | 0.062 → 0.049 | 0.129 → 0.054 | 0 / 0 |
| H-s202 | 0.808 → 0.826 → 0.911 | 0.000 → 0.018 | 0.000 → 0.000 | 0.062 → 0.062 | 0.129 → 0.009 | 0 / 2 |
| R-s101 | 0.817 → 0.839 → 0.911 | 0.000 → 0.031 | 0.000 → 0.000 | 0.080 → 0.040 | 0.103 → 0.018 | 0 / 3 |
| R-s202 | 0.808 → 0.879 → 0.902 | 0.000 → 0.000 | 0.000 → 0.000 | 0.062 → 0.040 | 0.129 → 0.058 | 0 / 2 |
| U-s101 | 0.817 → 0.830 → 0.924 | 0.000 → 0.009 | 0.000 → 0.000 | 0.080 → 0.009 | 0.103 → 0.058 | 0 / 0 |
| U-s202 | 0.817 → 0.826 → 0.920 | 0.000 → 0.000 | 0.000 → 0.000 | 0.080 → 0.027 | 0.103 → 0.054 | 0 / 0 |

### J_H for comparison (32 problems, 256 attempts per cell)
| run | pass@1 r0 → r32 → r128 | truncated r0 → r128 | no box r0 → r128 | multi-box wrong r0 → r128 | single-box wrong r0 → r128 | lenient rescues r0 / r128 |
|---|---|---|---|---|---|---|
| H-s101 | 0.426 → 0.570 → 0.688 | 0.000 → 0.160 | 0.000 → 0.000 | 0.320 → 0.094 | 0.254 → 0.059 | 1 / 2 |
| H-s202 | 0.410 → 0.566 → 0.680 | 0.000 → 0.219 | 0.000 → 0.000 | 0.285 → 0.082 | 0.305 → 0.020 | 1 / 0 |
| R-s101 | 0.418 → 0.527 → 0.766 | 0.004 → 0.152 | 0.000 → 0.000 | 0.293 → 0.066 | 0.285 → 0.016 | 1 / 0 |
| R-s202 | 0.410 → 0.605 → 0.676 | 0.000 → 0.203 | 0.000 → 0.000 | 0.285 → 0.020 | 0.305 → 0.102 | 1 / 0 |
| U-s101 | 0.434 → 0.621 → 0.703 | 0.000 → 0.203 | 0.000 → 0.000 | 0.340 → 0.027 | 0.227 → 0.066 | 1 / 1 |
| U-s202 | 0.434 → 0.625 → 0.734 | 0.000 → 0.117 | 0.000 → 0.000 | 0.340 → 0.016 | 0.227 → 0.133 | 1 / 1 |

Across all 90 cells: no-box plus unbalanced ≤ 1 attempt; unknown 0; lenient "any box" rescues ≤ 5 finished and ≤ 3 truncated
attempts; "last number" rescues ≤ 1 finished and ≤ 2 truncated attempts; no `<think>` text in any attempt.

**Round-0 rows repeat across runs.** Identical class counts appear for H-s202 = R-s202 and for U-s101 = U-s202 (J_U), and for
H-s101 = H-s202 = R-s202 (J_R). This agrees with DEFINITIONS v1 §3.B (round-0 dev attempts of R-s202 and H-s202 token-identical,
424/424). The six round-0 cells are therefore fewer than six independent samples of the base model.

## 2. Gains r0 → r128 with paired problem bootstrap (95%)
| run | ΔJ_U | ΔJ_U, r128 correct ≤ 8,192 | ΔJ_R | ΔJ_R ≤ 8,192 | ΔJ_H | ΔJ_H ≤ 8,192 |
|---|---|---|---|---|---|---|
| H-s101 | +0.133 [+0.051, +0.227] | +0.125 [+0.043, +0.219] | +0.085 [+0.031, +0.152] | +0.076 [+0.022, +0.143] | +0.262 [+0.164, +0.367] | +0.203 [+0.105, +0.309] |
| H-s202 | +0.098 [+0.023, +0.184] | +0.098 [+0.023, +0.184] | +0.103 [+0.022, +0.196] | +0.071 [−0.000, +0.156] | +0.270 [+0.160, +0.383] | +0.184 [+0.086, +0.285] |
| R-s101 | +0.148 [+0.078, +0.230] | +0.137 [+0.070, +0.215] | +0.094 [+0.004, +0.188] | +0.076 [−0.009, +0.165] | +0.348 [+0.238, +0.465] | +0.242 [+0.129, +0.363] |
| R-s202 | +0.113 [+0.039, +0.195] | +0.109 [+0.039, +0.188] | +0.094 [+0.036, +0.161] | +0.062 [+0.000, +0.129] | +0.266 [+0.168, +0.371] | +0.203 [+0.113, +0.301] |
| U-s101 | +0.125 [+0.051, +0.211] | +0.113 [+0.043, +0.195] | +0.107 [+0.036, +0.188] | +0.089 [+0.018, +0.170] | +0.270 [+0.176, +0.371] | +0.188 [+0.102, +0.281] |
| U-s202 | +0.125 [+0.051, +0.207] | +0.125 [+0.051, +0.207] | +0.103 [+0.040, +0.174] | +0.089 [+0.018, +0.170] | +0.301 [+0.203, +0.406] | +0.234 [+0.141, +0.336] |

## 3. What the evidence says about each explanation
| explanation | verdict | evidence |
|---|---|---|
| E1 format/termination | **not supported** on J_U/J_R | No-box failures are 0 at round 0 in every run; the base always boxes. Lenient scoring rescues ≤ 5 attempts per cell. Truncation on J_U/J_R is ≤ 0.004 at round 0 and ≤ 0.031 at round 128, so it moves the wrong way. |
| E3 evaluation cap | **minor** | Restricting round-128 success to ≤ 8,192 tokens lowers ΔJ_U by 0.000–0.012 and ΔJ_R by 0.009–0.032. 23–76% of each run's J_U gain is already present at round 32, where both evaluations used cap 8,192 (+0.031 to +0.075). The cap matters for J_H (gain falls by 0.06–0.11). |
| E5 noise | **unlikely** | All six ΔJ_U intervals exclude 0; ΔJ_R intervals exclude 0 in all six runs at the 12,288 view. Per-problem changes agree across arms (below). |
| E2 general carefulness vs E4 skill overlap | **not separable here** | The removed failures are wrong final answers: single-box wrong falls by 0.07–0.12 and multi-box wrong by 0.04–0.07 on J_U. Both explanations predict this. See `control.md`. |

**Multi-box wrong is a style, not a scoring loss.** In the sampled cells most multi-box wrong attempts repeat one wrong value
(distinct box contents = 1: J_U round 0, H-s101 11 of 15, R-s202 16 of 22; J_H round 0, 48 of 82 and 40 of 73). The base model
restates its final answer ("Final Answer: \boxed{45}" twice); no box holds the gold. The class shrinks because the answer is
right more often and the restating habit fades, not because grading of the last box changed.

## 4. Which J_U problems improve, and in which arms
Pooled base = round-0 correct over 6 runs × 8 = 48 attempts (not 48 independent samples, see §1). Arm change = mean of the two
seeds' round-128 c/8 minus the pooled base.

| problem | pooled base (of 48) | H | R | U |
|---|---|---|---|---|
| `number_theory/train/00372` | 6 | +0.88 | +0.69 | +0.75 |
| `number_theory/train/00802` | 7 | +0.48 | +0.73 | +0.42 |
| `number_theory/train/00258` | 9 | −0.12 | +0.44 | +0.06 |
| `number_theory/train/00140` | 11 | +0.52 | +0.27 | +0.77 |
| `number_theory/train/00248` | 19 | +0.42 | +0.48 | +0.48 |
| `number_theory/train/00863` | 22 | +0.54 | +0.54 | +0.48 |
| `number_theory/train/00357` | 31 | +0.29 | +0.23 | +0.35 |
| `number_theory/train/00282` | 37 | +0.17 | +0.23 | +0.23 |
| `number_theory/train/00456` | 37 | +0.23 | +0.23 | +0.17 |
| `number_theory/train/00160` | 38 | +0.21 | +0.02 | +0.15 |
| `number_theory/train/00306` | 39 | +0.06 | +0.12 | +0.12 |
| `number_theory/train/00807` | 42 | +0.00 | +0.12 | +0.06 |
| `number_theory/train/00435`, `00717` | 45 each | +0.06 | +0.06 | +0.06 |
| `number_theory/train/00169` | 45 | −0.06 | +0.06 | +0.06 |

The other 17 of 32 J_U problems have pooled base ≥ 0.95 and do not move (arm means −0.012 to −0.001). So the J_U gain comes from
about 12 problems, and **the same problems improve in H, R and U**: Pearson correlation of per-problem arm changes H–U 0.94,
H–R 0.81, R–U 0.82. J_R behaves the same (12 of 28 problems below 0.95; correlations H–U 0.88, H–R 0.78, R–U 0.76).
Example: `00372` asks for 1234567890 mod 99. The base sums single digits (answer 45; gold 72); all arms learn to get it right
in most attempts. Practising modular arithmetic (U) gives no larger change on J_U than H does.

## 5. Limits
- In-run evaluations only (n = 8); round-0 cells of different runs share attempts.
- Lenient checks use the project verifier on box contents and on the last number only; a correct answer stated in prose without
  a number is not detected.
- Classes describe the final answer, not the reasoning. "Single-box wrong" mixes careless and conceptual errors.
