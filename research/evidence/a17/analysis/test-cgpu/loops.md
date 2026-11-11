# Truncated test attempts: degenerate repetition or long reasoning?

Rule: rep_frac = share of window start positions in the last 2048 tokens (positions with a full 32-token window) whose 32-token window already started at an earlier position of the same attempt; loop = rep_frac ≥ 0.5; attempts shorter than 32 tokens get 0.

| Policy | Truncated | Truncated: looping (share) | Truncated: median rep_frac | Finished-wrong: looping (share), median rep_frac | Finished-correct: looping (share), median rep_frac |
|---|---|---|---|---|---|
| base | 2/4096 | 1/2 (0.500) | 0.500 | 2/1941 (0.001), 0.003 | 0/2153 (0.000), 0.000 |
| H-s101 | 445/4096 | 0/445 (0.000) | 0.036 | 0/499 (0.000), 0.016 | 0/3152 (0.000), 0.003 |
| H-s202 | 632/4096 | 0/632 (0.000) | 0.053 | 0/499 (0.000), 0.028 | 0/2965 (0.000), 0.006 |
| R-s101 | 437/4096 | 0/437 (0.000) | 0.007 | 0/306 (0.000), 0.003 | 0/3353 (0.000), 0.000 |
| R-s202 | 538/4096 | 0/538 (0.000) | 0.047 | 0/495 (0.000), 0.020 | 0/3063 (0.000), 0.002 |
| U-s101 | 501/4096 | 1/501 (0.002) | 0.015 | 0/413 (0.000), 0.009 | 0/3182 (0.000), 0.001 |
| U-s202 | 333/4096 | 0/333 (0.000) | 0.009 | 0/518 (0.000), 0.004 | 0/3245 (0.000), 0.000 |

## Paired R vs U per seed (shares of 4096 attempts)

| Seed | Truncated & looping: R − U | Truncated & not looping: R − U | Truncation term −Δt·ā = loop part + non-loop part |
|---|---|---|---|
| s101 | 0/4096 − 1/4096: 0.0000 − 0.0002 = -0.0002 [-0.001, +0.000] | 437/4096 − 500/4096: 0.1067 − 0.1221 = -0.0154 [-0.031, -0.000] | +0.0141 = +0.0002 + +0.0139 (ā = 0.9007) |
| s202 | 0/4096 − 0/4096: 0.0000 − 0.0000 = +0.0000 [+0.000, +0.000] | 538/4096 − 333/4096: 0.1313 − 0.0813 = +0.0500 [+0.031, +0.071] | -0.0431 = -0.0000 + -0.0431 (ā = 0.8616) |

## Truncated and looping attempts per band (of band size × 32)

| Policy | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| base | 0/480 | 0/1088 | 1/832 | 0/1024 | 0/672 |
| H-s101 | 0/480 | 0/1088 | 0/832 | 0/1024 | 0/672 |
| H-s202 | 0/480 | 0/1088 | 0/832 | 0/1024 | 0/672 |
| R-s101 | 0/480 | 0/1088 | 0/832 | 0/1024 | 0/672 |
| R-s202 | 0/480 | 0/1088 | 0/832 | 0/1024 | 0/672 |
| U-s101 | 0/480 | 1/1088 | 0/832 | 0/1024 | 0/672 |
| U-s202 | 0/480 | 0/1088 | 0/832 | 0/1024 | 0/672 |

95% paired problem bootstrap (10,000 draws, seed 2027, streams a9/a17-loops/...) on per-problem count/32 differences.
