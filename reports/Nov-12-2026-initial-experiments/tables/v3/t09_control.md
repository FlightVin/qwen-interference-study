# Control panel: gains on subjects no arm practises

Control panel `control_v1`: 64 MATH test problems (16 each from geometry, counting and probability, precalculus, prealgebra; levels 4–5), none used in any study set; 8 attempts, cap 12,288, base and the six round-128 endpoints (pre-registered in transfer/PREREG.md). Control gain = endpoint − base. J_U and J_R gains = in-run pass@1 at round 128 (cap 12,288) − round 0 (cap 8,192) of the same run. Intervals: paired problem bootstrap, 10,000 draws, numpy default_rng(2026) (transfer agent).

Sources: `research/audit-a26/transfer/control.json`; `research/audit-a26/transfer/control.md`; `research/audit-a26/pass-b/data/train_curves.json (probe gains cross-checked)`.

| Run | Control base (%) | Control endpoint (%) | Control gain (points) | 95% interval | J_U gain (points) | 95% interval | J_R gain (points) | 95% interval | Control GPU |
|---|---|---|---|---|---|---|---|---|---|
| H-s101 | 50.8 | 73.4 | +22.7 | [+14.5, +30.9] | +13.3 | [+5.1, +22.7] | +8.5 | [+3.1, +15.2] | L40S |
| H-s202 | 50.8 | 75.6 | +24.8 | [+16.6, +33.0] | +9.8 | [+2.3, +18.4] | +10.3 | [+2.2, +19.6] | L40 |
| R-s101 | 50.8 | 79.1 | +28.3 | [+19.9, +36.9] | +14.8 | [+7.8, +23.0] | +9.4 | [+0.4, +18.8] | L40 |
| R-s202 | 50.8 | 72.1 | +21.3 | [+15.0, +27.5] | +11.3 | [+3.9, +19.5] | +9.4 | [+3.6, +16.1] | L40 |
| U-s101 | 50.8 | 76.2 | +25.4 | [+17.8, +33.4] | +12.5 | [+5.1, +21.1] | +10.7 | [+3.6, +18.8] | L40S |
| U-s202 | 50.8 | 73.4 | +22.7 | [+15.4, +30.1] | +12.5 | [+5.1, +20.7] | +10.3 | [+4.0, +17.4] | L40S |

Base control pass@1 is 50.8%, base J_U and J_R about 81%, so the control panel has more room to gain.

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
