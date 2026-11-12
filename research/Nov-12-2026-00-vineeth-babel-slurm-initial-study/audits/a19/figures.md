# Figure checks against source tables (A19 §5; CHECKLIST 8.8d / 9.7) — 2026-11-11 16:35Z

Method: read each embedded PNG and compare every plotted point with its source table, by eye at the plotted precision
(≈ 0.002 on these axes). Source tables are the files the figure code reads.

| Figure | Source | Points checked | Result |
|---|---|---|---|
| `figures/a17/delta_j.png` | `research/evidence/a17/analysis/train/delta_j_table.md` | r0 and r64: 12 branch rows × (J_H, J_R, J_U) + no-update row × 3 = 78 values | verified: every value at its table position; overlapping equal values (e.g. r0 hard J_R −0.0201 twice) show as one marker |
| `figures/a17/pope_JH_vs_JR.png` | `curves.json` `probe_J_H`, `probe_J_R` | H/R/U × s101, R/U s202: 5 rounds each; H-s202 a2-orig rounds 0/32/64; a2-r60 rounds 0/32 | verified |
| `figures/a17/pope_JH_vs_JU.png` | `curves.json` `probe_J_H`, `probe_J_U` | same 31 points | verified; defect: title line 2 clipped at the left edge (cosmetic) |
| `figures/a17/stress_panels.png` | `research/evidence/a17/analysis/stress/stress.md` (tier tables) | 16 pass@1 bars + 16 truncation bars (8 checkpoints × 2 tiers) | verified 18:25Z |
| `figures/a17/truncation.png` | REPORT section 3.3 table; `research/evidence/a17/analysis/test/` | right panel: 2 seeds × 2 terms with intervals + observed R − U; titles: Pearson −0.510 / −0.587 | verified 18:26Z (per-problem scatter checked for its stated correlations and axis range only); defect: right y-label clipped at top (cosmetic) |
| `figures/a17/learning_curves.png` | `curves.json` (dev, probe_J_*, practised_shared_hard_bins) | dev 5 runs × 5 rounds (+ H-s202 a2-orig, a2-r60); J panels as in the POPE figures; shared-hard bins R/U-s101 (16 each) | verified 18:30Z; defect: the two cap-change labels overlap in the exposure panels (cosmetic); H-s202 a2-r60 shown to round 32 until the final a17_train rerun |
| `figures/a17/reversal_scatter.png` | `test-cgpu/reversal.json` | quadrant counts 19 + 12 same sign, 8 + 36 opposite, ties 53 (31 both); mean star; axes; 128 points by band | verified 21:58Z (subagent read, arm-agnostic); defect: legend hid 3 points (00213, 00221, 00891) → legend moved outside (a17_test), redrawn in the test stage |
| `figures/a17/coverage.png` | `test-cgpu/coverage.md` | solved ≥ 1 of 32 and pass@1 for 6 policies | verified; H-s202 added in the test stage |
| `figures/a17/transitions.png` | `train/persistence.json` | 7 panels × 4 windows (28 stacked bars, labels and stacking) | verified; defect: one count label clipped → y headroom added (a17_train) |
| `figures/a17/sampling_null.png` | `test-cgpu/sampling_null.md` | observed and hypergeometric rates at m = 8 and 16 by bin; curve peak 0.25 at p ≈ 0.08 | verified; REPORT 4.6 wording made explicit (m = 8; losses 0–5, gains up to 6) |
| `figures/a17/endpoints.png` | `research/evidence/a17/analysis/test-cgpu/endpoints.json` | 10 bars + 12 intervals | verified 12:52Z (bars equal section 1 tables; legend fixed d0fb202) |

Open: H-s202 a2-r60 has probe evaluations past round 32 now; the POPE figures show it to round 32 only until a17_train
is rerun after the H-s202 run ends. All 12 embedded figures checked; the redraws after the final reruns are checked by eye at delivery.

**Final redraws (after H-s202 ended).** `endpoints.png` re-checked 23:05Z against `test-cgpu/endpoints.md`: 12 bars
(H-s202 0.724 in both groups) and 18 intervals (R − H s202 +0.024 / +0.037; U − H s202 +0.068; means +0.036 / +0.045 and
+0.038) verified. The legend calls the H-s202 hatched bar "common GPU"; REPORT section 2 footnote states it is the same
mixed-hardware evaluation (D159). `transitions.png` redrawn with headroom and checked 21:59Z. `learning_curves.png` and
POPE figures redrawn with H-s202 a2-r60 to round 128 (curves.json dev 0.637 / 0.660 / 0.670 at rounds 64 / 96 / 128).

