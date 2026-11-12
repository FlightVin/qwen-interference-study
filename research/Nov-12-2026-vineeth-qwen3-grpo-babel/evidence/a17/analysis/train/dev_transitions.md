# Dev-panel transitions (A17, exploratory)

Verified attempts per problem: n = 8 attempts per dev problem in every scheduled evaluation; the per-sample records show [8]; the REPORT.md 'of 16' is wrong

| Run | From → to | Problems | n per problem (from / to) | Lost | Gained | Zero-correct problems (from → to) |
|---|---|---|---|---|---|---|
| H-s101 | 0 → 32 | 53 | [8] / [8] | 1 | 5 | 18 → 14 |
| H-s101 | 32 → 64 | 53 | [8] / [8] | 2 | 9 | 14 → 7 |
| H-s101 | 64 → 96 | 53 | [8] / [8] | 5 | 2 | 7 → 10 |
| H-s101 | 96 → 128 | 53 | [8] / [8] | 2 | 4 | 10 → 8 |
| H-s202-orig | 0 → 32 | 53 | [8] / [8] | 0 | 10 | 19 → 9 |
| H-s202-orig | 32 → 64 | 53 | [8] / [8] | 1 | 3 | 9 → 7 |
| H-s202-r60 | 0 → 32 | 53 | [8] / [8] | 0 | 10 | 19 → 9 |
| H-s202-r60 | 32 → 64 | 53 | [8] / [8] | 1 | 2 | 9 → 8 |
| H-s202-r60 | 64 → 96 | 53 | [8] / [8] | 2 | 2 | 8 → 8 |
| H-s202-r60 | 96 → 128 | 53 | [8] / [8] | 2 | 2 | 8 → 8 |
| R-s101 | 0 → 32 | 53 | [8] / [8] | 0 | 8 | 17 → 9 |
| R-s101 | 32 → 64 | 53 | [8] / [8] | 1 | 4 | 9 → 6 |
| R-s101 | 64 → 96 | 53 | [8] / [8] | 1 | 2 | 6 → 5 |
| R-s101 | 96 → 128 | 53 | [8] / [8] | 2 | 1 | 5 → 6 |
| R-s202 | 0 → 32 | 53 | [8] / [8] | 0 | 12 | 19 → 7 |
| R-s202 | 32 → 64 | 53 | [8] / [8] | 2 | 2 | 7 → 7 |
| R-s202 | 64 → 96 | 53 | [8] / [8] | 3 | 2 | 7 → 8 |
| R-s202 | 96 → 128 | 53 | [8] / [8] | 1 | 2 | 8 → 7 |
| U-s101 | 0 → 32 | 53 | [8] / [8] | 2 | 8 | 18 → 12 |
| U-s101 | 32 → 64 | 53 | [8] / [8] | 0 | 6 | 12 → 6 |
| U-s101 | 64 → 96 | 53 | [8] / [8] | 2 | 1 | 6 → 7 |
| U-s101 | 96 → 128 | 53 | [8] / [8] | 1 | 2 | 7 → 6 |
| U-s202 | 0 → 32 | 53 | [8] / [8] | 0 | 11 | 18 → 7 |
| U-s202 | 32 → 64 | 53 | [8] / [8] | 2 | 2 | 7 → 7 |
| U-s202 | 64 → 96 | 53 | [8] / [8] | 1 | 2 | 7 → 6 |
| U-s202 | 96 → 128 | 53 | [8] / [8] | 4 | 4 | 6 → 6 |

**Repeated evaluations and no-update references** (same request seeds: differences come from non-reproducible generation, not from independent resampling):

| Source | Panel | Rows compared | n per problem | Lost | Gained | pass@1 (first − second) | Checkpoint |
|---|---|---|---|---|---|---|---|
| qis-main-H-s202-sym r64 | dev | rows 360 / 388 | [8] | 1 | 0 | 0.6580 − 0.6368 = +0.0212 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| qis-main-H-s202-sym r64 | probe_J_H | rows 361 / 389 | [8] | 1 | 0 | 0.6523 − 0.6133 = +0.0391 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| qis-main-H-s202-sym r64 | probe_J_R | rows 362 / 390 | [8] | 1 | 0 | 0.8527 − 0.8616 = -0.0089 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| qis-main-H-s202-sym r64 | probe_J_U | rows 363 / 391 | [8] | 0 | 0 | 0.9219 − 0.9102 = +0.0117 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| qis-main-R-s101-sym r128 | dev | rows 720 / 750 | [8] | 2 | 2 | 0.7099 − 0.7288 = -0.0189 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| qis-main-R-s101-sym r128 | probe_J_H | rows 721 / 751 | [8] | 1 | 2 | 0.7383 − 0.7656 = -0.0273 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| qis-main-R-s101-sym r128 | probe_J_R | rows 722 / 752 | [8] | 0 | 0 | 0.8839 − 0.9107 = -0.0268 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| qis-main-R-s101-sym r128 | probe_J_U | rows 723 / 753 | [8] | 0 | 0 | 0.9336 − 0.9609 = -0.0273 | retrained from an earlier checkpoint between the two evaluations: not a no-update reference |
| diag r0 | probe_J_H | none / none_rerun | [16] | 3 | 2 | 0.4004 − 0.4316 = -0.0312 | no update |
| diag r0 | probe_J_R | none / none_rerun | [16] | 1 | 0 | 0.8281 − 0.8080 = +0.0201 | no update |
| diag r0 | probe_J_U | none / none_rerun | [16] | 1 | 0 | 0.8066 − 0.7930 = +0.0137 | no update |
| diag r0 | probe_J_H | none_rerun / none_cycled | [16] | 0 | 0 | 0.4316 − 0.4316 = +0.0000 | no update |
| diag r0 | probe_J_R | none_rerun / none_cycled | [16] | 0 | 0 | 0.8080 − 0.8080 = +0.0000 | no update |
| diag r0 | probe_J_U | none_rerun / none_cycled | [16] | 0 | 0 | 0.7930 − 0.7930 = +0.0000 | no update |
| diag r64 | probe_J_H | none / none_rerun | [16] | 1 | 1 | 0.6055 − 0.6309 = -0.0254 | no update |
| diag r64 | probe_J_R | none / none_rerun | [16] | 0 | 0 | 0.9018 − 0.8929 = +0.0089 | no update |
| diag r64 | probe_J_U | none / none_rerun | [16] | 0 | 0 | 0.9395 − 0.9512 = -0.0117 | no update |
| diag r64 | probe_J_H | none_rerun / none_cycled | [16] | 0 | 0 | 0.6309 − 0.6309 = +0.0000 | no update |
| diag r64 | probe_J_R | none_rerun / none_cycled | [16] | 0 | 0 | 0.8929 − 0.8929 = +0.0000 | no update |
| diag r64 | probe_J_U | none_rerun / none_cycled | [16] | 0 | 0 | 0.9512 − 0.9512 = +0.0000 | no update |
