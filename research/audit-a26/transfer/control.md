# A26 x-transfer — control-panel evaluation (control_v1)

Purpose: test whether the J_U/J_R gains that appear in every arm are specific to the probes' skills or are a general effect of
the training. Plan and decision rule: `PREREG.md` (committed in 7e08d75 before any control output existed).
Written 2026-11-11 (EST) by agent x-transfer. All numbers are **verified** from the job outputs listed below.

**Main result.** The gains are general. Every round-128 endpoint gains 21–28 points over the base on 64 MATH test problems
from four subjects that no arm practises: geometry, counting and probability, precalculus and prealgebra. The gain is
about twice the J_U gain in every arm (ρ 1.7–2.5). This meets the pre-registered "general transfer" rule in H, R and U. It
removes about half (48–50%) of the base's errors, close to the share removed on J_R (49–57%) and J_U (59–69%). Arms do not differ on control
by more than 6 points in either seed, and the sign of each contrast flips between seeds. So the J_U/J_R rise in H and R does not need hidden overlap with modular arithmetic or
factoring (E4). The training improves final-answer accuracy broadly (E2). Format and termination (E1) do not explain it:
the base always boxes, never truncates on this panel, and lenient scoring rescues 0 of its 252 failures.

## Jobs (all COMPLETED, exit 0; `sacct`, 2026-11-11 EST)
| expid | job | commit | GPU | start → end (EST) | checkpoint `model.pt` sha256 (= `state.json`) |
|---|---|---|---|---|---|
| `qis-control-base` | 11122684 | 7e08d75 | L40S | 21:11 → 21:14 | base (`70d244cc`) |
| `qis-control-H-s101` | 11122732 | 0c7ac68 | L40S | 21:16 → 21:40 | `052c8a62…` |
| `qis-control-H-s202` | 11122733 | 0c7ac68 | L40 | 21:16 → 21:45 | `25134562…` |
| `qis-control-R-s101` | 11122734 | 0c7ac68 | L40 | 21:17 → 21:40 | `aedcca6e…` |
| `qis-control-R-s202` | 11122735 | 0c7ac68 | L40 | 21:17 → 21:40 | `22fd8364…` |
| `qis-control-U-s101` | 11122737 | 0c7ac68 | L40S | 21:18 → 21:40 | `92aef0d2…` |
| `qis-control-U-s202` | 11122739 | 0c7ac68 | L40S | 21:18 → 21:32 | `7158b095…` |

Every endpoint passed the precheck: panel sha256 `fc2ceb70…`, on-disk `model.pt` hash equal to `state.json`, round 128,
attempt a2, GPU empty at start, and the weight read-back check inside `evaluate_ckpt`. The two commits differ only in
`research/` files; `src/` and the panel are identical. n 8, cap 12,288, seed label `control-v1`, panel key `control`, verifier
unknowns 0 in every policy. Outputs: `$P/control-eval/<policy>/{summary.json,records.jsonl.gz}`.

## 1. Per policy (64 problems × 8 attempts)
| policy | pass@1 | truncated | acc. among finished | no box | multi-box wrong | single-box wrong | pass@1 ≤ 8,192 | C&P | geometry | prealgebra | precalculus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.508 | 0.000 | 0.508 | 0.000 | 0.193 | 0.299 | 0.508 | 0.453 | 0.352 | 0.820 | 0.406 |
| H-s101 | 0.734 | 0.033 | 0.760 | 0.006 | 0.098 | 0.129 | 0.719 | 0.789 | 0.703 | 0.867 | 0.578 |
| H-s202 | 0.756 | 0.053 | 0.798 | 0.000 | 0.131 | 0.061 | 0.711 | 0.773 | 0.750 | 0.859 | 0.641 |
| R-s101 | 0.791 | 0.035 | 0.820 | 0.000 | 0.100 | 0.074 | 0.766 | 0.789 | 0.805 | 0.867 | 0.703 |
| R-s202 | 0.721 | 0.035 | 0.747 | 0.002 | 0.045 | 0.197 | 0.695 | 0.742 | 0.609 | 0.875 | 0.656 |
| U-s101 | 0.762 | 0.061 | 0.811 | 0.000 | 0.031 | 0.146 | 0.723 | 0.805 | 0.711 | 0.875 | 0.656 |
| U-s202 | 0.734 | 0.023 | 0.752 | 0.004 | 0.023 | 0.215 | 0.717 | 0.711 | 0.688 | 0.852 | 0.688 |

Shares are of all 512 attempts; subject columns are pass@1 over 16 problems (C&P = counting_and_probability).

## 2. Gains vs base and comparison with the probe gains (paired problem bootstrap 95%, 10,000 resamples, seed 2026)
| run | Δ control | Δ by subject (C&P, geometry, prealgebra, precalculus) | ΔJ_U (r128 − r0, in-run) | ΔJ_U ≤ 8,192 | ΔJ_R | ρ = Δcontrol / ΔJ_U |
|---|---|---|---|---|---|---|
| H-s101 | +0.227 [+0.145, +0.309] | +0.336, +0.352, +0.047, +0.172 | +0.133 [+0.051, +0.227] | +0.125 | +0.085 | 1.71 |
| H-s202 | +0.248 [+0.166, +0.330] | +0.320, +0.398, +0.039, +0.234 | +0.098 [+0.023, +0.184] | +0.098 | +0.103 | 2.54 |
| R-s101 | +0.283 [+0.199, +0.369] | +0.336, +0.453, +0.047, +0.297 | +0.148 [+0.078, +0.230] | +0.137 | +0.094 | 1.91 |
| R-s202 | +0.213 [+0.150, +0.275] | +0.289, +0.258, +0.055, +0.250 | +0.113 [+0.039, +0.195] | +0.109 | +0.094 | 1.88 |
| U-s101 | +0.254 [+0.178, +0.334] | +0.352, +0.359, +0.055, +0.250 | +0.125 [+0.051, +0.211] | +0.113 | +0.107 | 2.03 |
| U-s202 | +0.227 [+0.154, +0.301] | +0.258, +0.336, +0.031, +0.281 | +0.125 [+0.051, +0.207] | +0.125 | +0.103 | 1.81 |

ρ > 1 mostly reflects headroom: base pass@1 is 0.508 on control and 0.81 on J_U/J_R. As a share of base errors removed
(seed means, (end − base)/(1 − base)): control H 0.48, R 0.50, U 0.49; J_R H 0.49, R 0.50, U 0.57; J_U H 0.59, R 0.69, U 0.68.
Prealgebra (base 0.82) moves least, +0.03 to +0.06.

## 3. Arm contrasts on control, per seed (descriptive; two seeds cannot estimate run variance)
| seed | H − R | H − U | R − U |
|---|---|---|---|
| s101 | −0.057 [−0.100, −0.016] | −0.027 [−0.070, +0.014] | +0.029 [−0.002, +0.062] |
| s202 | +0.035 [−0.004, +0.076] | +0.021 [−0.014, +0.057] | −0.014 [−0.055, +0.025] |

Signs flip between seeds. There is no consistent arm ordering on control.

## 4. Pre-registered decision (PREREG §6)
| arm | seed-mean Δcontrol | seed-mean ΔJ_U | ρ | both seeds > 0, ≥ 1 interval excludes 0 | verdict |
|---|---|---|---|---|---|
| H | +0.237 | +0.115 | 2.06 | yes (both exclude 0) | **general transfer** |
| R | +0.248 | +0.131 | 1.90 | yes (both exclude 0) | **general transfer** |
| U | +0.240 | +0.125 | 1.92 | yes (both exclude 0) | **general transfer** |

**E1 vs E2 split (pre-registered rule) and a necessary caveat.** The rule counts the drop in truncated + no-box + multi-box
wrong as "format". That drop is 0.033 of 0.237 in H (14%), 0.085 of 0.248 in R (34%) and 0.122 of 0.240 in U (51%). By the
rule, H and R are E2 (carefulness) and U is at the E1 threshold. The rule's "format" bucket is misleading, however:
- truncation *rises* (base 0.000 → 0.023–0.061) and no-box stays ≤ 0.006, so the bucket's drop is all multi-box wrong;
- lenient re-scoring (`control_lenient.json`) finds the gold in **0 of 99** base multi-box wrong attempts (75 of 99 repeat one
  value) and in 0–1 per endpoint; "any box" rescues 0 of 252 base failures. A multi-box wrong attempt is a wrong answer.
So all three arms' control gains are fewer wrong final answers, not format repairs. I report U as "E1 by the pre-registered
rule; E2 on the evidence", and flag the rule's bucket definition as a pre-registration error. The last-number check matches
6–9 failures per policy, base included; it does not change any comparison.

## 5. Bearing on E1–E5 (with `taxonomy.md`)
| explanation | verdict | reason |
|---|---|---|
| E1 format/termination | rejected as the main cause | Base never omits a box or truncates on control or J_U/J_R; lenient scoring rescues ≈ 0; training raises truncation. |
| E2 general carefulness | **supported** | Gains of 21–28 points on four unpractised subjects in all arms; the share of errors removed matches J_R and is close to J_U. |
| E3 cap artefact | rejected | Control compares base and endpoints at the same cap 12,288 and still shows the full gain; ≤ 8,192 views change gains by ≤ 0.045. |
| E4 hidden skill overlap | not needed | The gain does not require overlap with the hard set: it appears where no arm trains. It cannot be excluded as a small extra term. |
| E5 noise | rejected | All 6 control intervals exclude 0 by ≥ 0.145. |

## 6. Limits
- Base and endpoints ran on L40S or L40 (table above); DEFINITIONS v1 §4 item 12 measured ≤ 1.3 points from GPU type.
- J_U/J_R gains are in-run numbers (round-0 cap 8,192; shared round-0 attempts across runs). Not re-evaluated here.
- One base evaluation (n 8); its sampling noise enters every Δ and is covered only through the problem bootstrap.
- MATH test is public; absence from study sets does not show absence from pretraining data. This affects base and endpoints alike.
- "Carefulness" is a label for "fewer wrong final answers on unpractised subjects"; this study does not identify the mechanism
  (e.g. checking, longer reasoning: endpoints write longer answers and truncate more).

## Files
`control.json` (analysis output, unrounded), `control_extract.json` (per-problem class counts per policy), `control_lenient.json`,
`control_build_report.json`; tools `tools/build_control.py`, `tools/control_extract.py`, `tools/control_lenient.py`,
`tools/transfer_analysis.py control`. Panel `data/control/control_v1.json` + `README.md`. Recipes `slurm-agentic-engineering/exp/qis-control-*.sh`.
