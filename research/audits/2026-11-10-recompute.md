# Independent recomputation of the lead table — 2026-11-10

Purpose: check the A13 lead numbers without project code (CHECKLIST 7.3).

Method: plain Python 3 (`python3 -I`, standard library only) on the laptop. Input: `per_problem` counts (c correct of n = 32) in `reports/tables/summary_qis-test-*.json`. pass@1 = mean of c/n over 128 problems. pass@8 = mean of 1 − C(n−c, 8)/C(n, 8). Paired intervals: 10,000-draw problem bootstrap, `random.Random(7)` (the project uses seed 2027, so the bounds differ by bootstrap noise only).

All five files hold the same 128 problem ids.

| Item | Recomputed | `a9_table.json` | Match |
|---|---|---|---|
| Base pass@1 / pass@8 | 0.5256 / 0.8352 | 0.5256 / 0.8352 | yes |
| H-s101 pass@1 / pass@8 | 0.7695 / 0.9371 | 0.7695 / 0.9371 | yes |
| R-s101 pass@1 / pass@8 | 0.8186 / 0.9613 | 0.8186 / 0.9613 | yes |
| U-s101 pass@1 / pass@8 | 0.7769 / 0.9446 | 0.7769 / 0.9446 | yes |
| R-s202 pass@1 / pass@8 | 0.7478 / 0.9221 | 0.7478 / 0.9221 | yes |
| R−U s101 | +0.0417 [+0.0203, +0.0630] | +0.0417 [+0.0212, +0.0632] | yes |
| R−H s101 | +0.0491 [+0.0198, +0.0796] | +0.0491 [+0.0195, +0.0798] | yes |
| U−H s101 | +0.0073 [−0.0200, +0.0359] | +0.0073 [−0.0195, +0.0349] | yes |
| U-s202 pass@1 | 0.7922 | 0.7922 | yes |
| R−U s202 | −0.0444 [−0.0728, −0.0178] | −0.0444 [−0.0718, −0.0173] | yes |
| R−U mean of seeds | −0.0013 [−0.0181, +0.0154] | −0.0013 [−0.0182, +0.0150] | yes |


Verdict: verified for every lead-table number (12:40Z, report-final 11106560).

## Report audit — 19:50Z
An independent read-only agent checked ≈ 400 numbers in `reports/REPORT.md` against the source tables; 392 matched. Fixed at 19:55Z: seed-202 R − U upper bound −0.018 → −0.017 (the lead table had taken the `per_problem.json` interval; `a9_table.json` is canonical; the row above also corrected); B2 +0.028 → +0.029; r0 hard 4104 +0.022 → +0.021; unrelated ΔJ_U +0.001 → +0.000; r64 original 0.606/0.940 → 0.605/0.939; "0.003 above hard" → 0.002; J_R sample size 448, not 512. Overclaims rewritten: §3 answer now rests on between-branch comparisons (absolute ΔJ depends on the shared reference); §2 pass@8 reading limited; §4 dev panel "arms do not differ" → R and U only; §4 schedule verification per seed; §5 cue answer states the seed-202 interval; Appendix A noise sentence. Four unclear passages rewritten.
