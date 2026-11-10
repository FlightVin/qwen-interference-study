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

Verdict: verified for seed 101 and R-s202. Pending: U-s202 (test running) and the seed means.
