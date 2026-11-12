# Report v3 figures: visual inspection and value check

Purpose: record that every figure in `reports/figures/v3/` was opened at reading width and at a narrow width, what was
fixed, and that every number in `reports/tables/v3/` agrees with the audit passes. Written 2026-11-12 (EST) by agent
`figures` (A26 §6).

## 1. Method
- Build: `reports/build/build_v3.py` on CPU standby 11122601, interpreter `/data/user_data/vineethb/qis/env/venv-report/bin/python`
  (Python 3.12.13, matplotlib 3.9.2, numpy 2.0.2). Step `extract` read the per-attempt records under
  `/data/user_data/vineethb/qis` (read only) into `reports/tables/v3/perproblem_counts.json`. Step `build` wrote tables and
  figures to `$P/audit-a26/figures/` and copied them back with scp.
- Determinism: two consecutive builds gave byte-identical files (32 of 32, md5).
- Previews: `sips --resampleWidth 830` (reading width) and `--resampleWidth 400` (narrow width) into a scratch directory
  outside the repo. Each preview was opened with the Read tool and checked for clipped titles, legible labels, overlap, and a
  legend that covers data.
- Size: every figure is 6.4–6.9 in wide at 200 dpi (1,271–1,376 px), body text 10 pt, titles 12 pt.
- Palette: H `#3F5A87`, R `#E07B12`, U `#2BB39A`, base `#222222`. Simulated colour-vision deficiency (Machado 2009, OKLab
  ΔE ×100): worst pair 14.4 (deutan) against a floor of 8; normal vision 24.1 against a floor of 15. Seed is a second cue
  (circle and solid line = 101; square and dashed line = 202).

## 2. Inspection record (final build)
| Figure | 830 px | 400 px | Fixed during inspection |
|---|---|---|---|
| F1 `f01_endpoints` | pass | pass (small text) | none |
| F2 `f02_contrasts_test` | pass | pass (small text) | none |
| F3 `f03_dev_rounds` | pass | pass (small text) | Long title widened the image to 8.2 in, so the title was shortened. The two cap labels ran together, so they were moved above the data with extra headroom. A round-0 note crossed a dashed line, so it moved into the x tick label ("0 / base"). |
| F4 `f04_dev_hard_attempts` | pass | pass (small text) | The x labels of the two panels overlapped, so they were shortened. Round-0 tick label changed to "0 / base". |
| F5 `f05_traj_JH_JR` | pass | pass; arrowheads hard to see | Per-path "r128" labels overlapped points and arrows, so the round-128 points are now ringed and the legend explains the ring. The "r0" label overlapped the axis, so one "r0" label now sits inside the axes and margins were added. |
| F6 `f06_traj_JH_JU` | pass | pass; arrowheads hard to see | Same fixes as F5. |
| F7 `f07_delta_j` | pass | pass (small text) | The image was 7.6 in wide because of long legend entries, so the legend text was shortened. Literal backticks were removed from the axis labels. |
| F8 `f08_length_pass1` | pass | pass (small text) | Shared-y bug: the second panel blanked the checkpoint labels, and a double inversion reversed the row order, so labels and inversion are now set on the first panel only. The two x labels overlapped and were shortened. |
| F8b `f08b_length_capped` | pass | pass (small text) | Same shared-y bug (labels missing), fixed. |
| F9 `f09_length_contrasts` | pass | pass (small text) | Same shared-y bug (row order reversed), fixed. |
| F10 `f10_stress_tiers` | pass | pass (small text) | Same shared-y bug (row order reversed), fixed. The two x labels nearly touched and were replaced by one shared label. |
| F11 `f11_stress_contrasts` | pass | pass (small text) | Same shared-y bug (row labels missing), fixed. |
| F12 `f12_control` | pass | pass (small text) | Same shared-y bug (row labels missing), fixed. Only two x ticks were shown, so ticks were added at 0, 10, 20 and 30. |

At 400 px, 10-pt text renders about 7 px high. It is readable but small in every figure; nothing is clipped or overlapping.
In F5 and F6 the arrowheads on short path segments are hard to see at 400 px. The ring still marks each path end, and the
legend explains the ring. At 830 px every label is legible without zooming.

## 3. Value check (`reports/build/check_v3.py`, stdlib)
The check compares every number in `reports/tables/v3/t*.json` with pass B (`pass-b/data/*.json`, the build input). Where
pass A computed the same quantity, it also compares with `pass-a/data/numbers.json`, `pass-a/data/eval_map.json` and column A
of `tools/reconcile/compare.json`. Point values must match within 1e-12. Pass-A intervals used other bootstrap streams and
must match within 0.01 (1 point). The build computed some intervals itself: absolute pass@1, curve points, per-checkpoint
length gains and stress tiers. These are recomputed with an independent stdlib bootstrap (`random.Random`, 4,000 draws) and
must agree within 0.01. A negative test showed that the check catches errors. It changed one stress value by 0.001 and one
interval bound by 0.02, and the check reported 3 mismatches.

Output, run 2026-11-12 on the final tables:

```
table | check kind | values checked | mismatches
t01_endpoints | A point | 26 | 0
t01_endpoints | B interval | 12 | 0
t01_endpoints | B point | 54 | 0
t01_endpoints | counts | 12 | 0
t01_endpoints | indep. bootstrap | 24 | 0
t02_contrasts_test | A interval | 16 | 0
t02_contrasts_test | A point | 11 | 0
t02_contrasts_test | B interval | 16 | 0
t02_contrasts_test | B point | 19 | 0
t03_curves | A derived R−U | 40 | 0
t03_curves | A point | 23 | 0
t03_curves | B point | 240 | 0
t03_curves | indep. bootstrap | 240 | 0
t03_curves | protocol | 120 | 0
t04_diag | A derived mean | 6 | 0
t04_diag | A point | 80 | 0
t04_diag | B point | 186 | 0
t04_diag | momentum-only set | 2 | 0
t05_length | A point | 60 | 0
t05_length | B point | 60 | 0
t05_length | counts | 10 | 0
t05_length | indep. bootstrap | 20 | 0
t06_length_contrasts | A interval | 48 | 0
t06_length_contrasts | A point | 24 | 0
t06_length_contrasts | B interval | 48 | 0
t06_length_contrasts | B point | 24 | 0
t07_stress | A point | 63 | 0
t07_stress | B point | 54 | 0
t07_stress | indep. bootstrap | 54 | 0
t08_stress_contrasts | A interval | 20 | 0
t08_stress_contrasts | A point | 10 | 0
t08_stress_contrasts | B interval | 36 | 0
t08_stress_contrasts | B point | 36 | 0
t09_control | B derived gain | 12 | 0
t09_control | control.json | 24 | 0
TOTAL | all | 1730 | 0
```

Values without a pass-A counterpart:
- the control panel (`t09_control`): a new evaluation by agent x-transfer, checked against `control.json`; its J_U and J_R
  gains are checked against the pass-B curves;
- H-run curve points and per-run probe values: pass A computed only the R and U dev values and the R − U differences, and
  both match;
- interval bounds computed by the build: checked by the independent bootstrap above.
