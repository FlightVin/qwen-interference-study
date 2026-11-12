| Family | Rows | Expids | Commits | Final state (sacct) | GPU types used (instances) | expsha ≠ local recipe | Job script on Babel |
|---|---|---|---|---|---|---|---|
| training | 23 | 9 | 13 | CANCELLED 14, COMPLETED 6, FAILED 3 | l40 3, l40s 12 | 10 | 23/23 |
| sealed test | 18 | 12 | 9 | CANCELLED 6, COMPLETED 12 | a6000 3, l40 2, l40s 7 | 6 | 18/18 |
| sealed test (fallback r116) | 6 | 6 | 1 | COMPLETED 6 | l40s 6 | 0 | 6/6 |
| cgpu reruns | 4 | 4 | 1 | COMPLETED 4 | l40s 8 | 0 | 4/4 |
| stress | 18 | 11 | 8 | CANCELLED 7, COMPLETED 11 | a6000 1, l40 2, l40s 13 | 7 | 18/18 |
| length | 12 | 10 | 5 | CANCELLED 2, COMPLETED 10 | l40s 12 | 1 | 12/12 |
| cue | 2 | 1 | 2 | CANCELLED 1, COMPLETED 1 | l40s 1 | 1 | 2/2 |
| diag | 9 | 4 | 7 | CANCELLED 4, COMPLETED 4, FAILED 1 | 6000ada 2, l40 2, l40s 1 | 1 | 9/9 |
| calibration | 16 | 16 | 6 | CANCELLED 5, COMPLETED 10, FAILED 1 | l40 2, l40s 8 | 0 | 16/16 |
| data build/screening | 31 | 28 | 27 | CANCELLED 3, COMPLETED 25, FAILED 3 | 6000ada 1, a6000 1, l40 3, l40s 10 | 4 | 31/31 |
| tests | 14 | 11 | 14 | COMPLETED 11, FAILED 3 | CPU only | 1 | 14/14 |
| in-run monitoring/inspection | 63 | 18 | 56 | CANCELLED 7, COMPLETED 54, FAILED 2 | l40 1, l40s 4 | 9 | 63/63 |
| ops/storage/recovery | 21 | 14 | 17 | COMPLETED 17, FAILED 4 | CPU only | 1 | 21/21 |
| report | 7 | 2 | 6 | CANCELLED 1, COMPLETED 5, FAILED 1 | l40 1 | 1 | 7/7 |
| selftest/env | 16 | 10 | 12 | CANCELLED 2, COMPLETED 10, FAILED 4 | 6000ada 6, l40s 5 | 5 | 16/16 |
| audits (A26) | 5 | 5 | 1 | COMPLETED 1, RUNNING 4 | l40 4 | 0 | 5/5 |
| **all** | 265 | 161 | 154 | COMPLETED 187, CANCELLED 52, FAILED 22, RUNNING 4 | | 47 | 265/265 |

Science-bearing jobs (training and evaluation families): job, commit, final state, node GPU per instance.

| Family | Expid | Job | Commit | Final state | GPU (instances) | Seed label / cap |
|---|---|---|---|---|---|---|
| training | `qis-main-R-s101-sym` | 11078086 | b8f36ca | CANCELLED | l40s | — |
| training | `qis-main-U-s101-sym` | 11078164 | efdafdd | CANCELLED | l40s | — |
| training | `qis-main-H-s101-sym` | 11078180 | 0a83a74 | CANCELLED | l40s | — |
| training | `qis-main-R-s202-sym` | 11078181 | 0a83a74 | CANCELLED | never started | — |
| training | `qis-main-U-s202-sym` | 11078184 | 0a83a74 | CANCELLED | never started | — |
| training | `qis-main-H-s202-sym` | 11078185 | 0a83a74 | CANCELLED | never started | — |
| training | `qis-main-R-s303-sym` | 11078186 | 0a83a74 | CANCELLED | never started | — |
| training | `qis-main-U-s303-sym` | 11078187 | 0a83a74 | CANCELLED | never started | — |
| training | `qis-main-H-s303-sym` | 11078412 | b2110c8 | CANCELLED | never started | — |
| training | `qis-main-H-s303-sym` | 11078506 | 4927344 | CANCELLED | never started | — |
| training | `qis-main-R-s101-sym` | 11081724 | 9d6349a | FAILED | l40s | boundary-r64/16384 |
| training | `qis-main-U-s101-sym` | 11081725 | 9d6349a | COMPLETED | l40s | boundary-r64/16384 |
| training | `qis-main-H-s101-sym` | 11081726 | 9d6349a | COMPLETED | l40 | boundary-r64/16384 |
| training | `qis-main-R-s202-sym` | 11081729 | 9d6349a | CANCELLED | l40 | boundary-r64/16384 |
| training | `qis-main-U-s202-sym` | 11081731 | 9d6349a | CANCELLED | never started | boundary-r64/16384 |
| training | `qis-main-H-s202-sym` | 11082192 | 7c44fa5 | CANCELLED | l40 | boundary-r64/16384 |
| training | `qis-main-R-s101-sym` | 11092945 | 0afd6e8 | COMPLETED | l40s | boundary-r64/16384 |
| training | `qis-main-U-s202-sym` | 11094037 | dd71b20 | FAILED | l40s | boundary-r64/16384 |
| training | `qis-main-R-s202-sym` | 11094190 | dd71b20 | COMPLETED | l40s | boundary-r64/16384 |
| training | `qis-main-U-s202-sym` | 11097811 | 6724cce | FAILED | l40s | boundary-r64/16384 |
| training | `qis-main-U-s202-sym` | 11100159 | d90dd8c | COMPLETED | l40s | boundary-r64/16384 |
| training | `qis-main-H-s202-sym` | 11112338 | 5f4b4d1 | CANCELLED | l40s | boundary-r64/16384 |
| training | `qis-main-H-s202-sym` | 11112377 | 09f5080 | COMPLETED | l40s | boundary-r64/16384 |
| sealed test | `qis-test-base` | 11079315 | 03a90b7 | CANCELLED | never started | test-final/8192 |
| sealed test | `qis-test-base` | 11079548 | 60d733d | CANCELLED | never started | test-final/16384 |
| sealed test | `qis-test-base` | 11089196 | 779c54d | COMPLETED | l40s | test-final/12288 |
| sealed test | `qis-test-R-s101-sym` | 11095106 | a96dc7d | COMPLETED | l40 | test-final/12288 |
| sealed test | `qis-test-U-s101-sym` | 11095118 | a96dc7d | COMPLETED | l40s | test-final/12288 |
| sealed test | `qis-test-H-s101-sym` | 11095486 | 95ff406 | COMPLETED | l40s | test-final/12288 |
| sealed test | `qis-test-R-s202-sym` | 11100782 | 73597a9 | COMPLETED | l40 | test-final/12288 |
| sealed test | `qis-test-U-s202-sym` | 11100785 | 73597a9 | COMPLETED | l40s | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k0` | 11119210 | f9b294b | COMPLETED | l40s | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k1` | 11119212 | f9b294b | COMPLETED | l40s | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k2` | 11119213 | f9b294b | CANCELLED | never started | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k3` | 11119215 | f9b294b | CANCELLED | never started | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k4` | 11119216 | f9b294b | CANCELLED | never started | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k5` | 11119217 | f9b294b | CANCELLED | never started | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k3` | 11120590 | b77f066 | COMPLETED | a6000 | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k4` | 11120591 | b77f066 | COMPLETED | a6000 | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k5` | 11120595 | b77f066 | COMPLETED | a6000 | test-final/12288 |
| sealed test | `qis-test-H-s202-sym-k2` | 11120634 | 815a9a6 | COMPLETED | l40s | test-final/12288 |
| sealed test (fallback r116) | `qis-test-H-s202-fb-k0` | 11118845 | 7a8190f | COMPLETED | l40s | test-final/12288 |
| sealed test (fallback r116) | `qis-test-H-s202-fb-k1` | 11118846 | 7a8190f | COMPLETED | l40s | test-final/12288 |
| sealed test (fallback r116) | `qis-test-H-s202-fb-k2` | 11118847 | 7a8190f | COMPLETED | l40s | test-final/12288 |
| sealed test (fallback r116) | `qis-test-H-s202-fb-k3` | 11118849 | 7a8190f | COMPLETED | l40s | test-final/12288 |
| sealed test (fallback r116) | `qis-test-H-s202-fb-k4` | 11118850 | 7a8190f | COMPLETED | l40s | test-final/12288 |
| sealed test (fallback r116) | `qis-test-H-s202-fb-k5` | 11118851 | 7a8190f | COMPLETED | l40s | test-final/12288 |
| cgpu reruns | `qis-cgpu-R-s101` | 11112395 | bcba941 | COMPLETED | l40s | test-final/12288 |
| cgpu reruns | `qis-cgpu-R-s202` | 11112396 | bcba941 | COMPLETED | l40s | test-final/12288 |
| cgpu reruns | `qis-cgpu-U-s101` | 11112397 | bcba941 | COMPLETED | l40s→l40s→l40s | test-final/12288 |
| cgpu reruns | `qis-cgpu-U-s202` | 11112398 | bcba941 | COMPLETED | l40s→l40s→l40s | test-final/12288 |
| stress | `qis-stress-base` | 11112846 | e218011 | CANCELLED | never started | stress-v1/12288 |
| stress | `qis-stress-R-s101-r128` | 11112848 | e218011 | CANCELLED | never started | stress-v1/12288 |
| stress | `qis-stress-U-s101-r128` | 11112849 | e218011 | CANCELLED | never started | stress-v1/12288 |
| stress | `qis-stress-R-s202-r128` | 11112850 | e218011 | CANCELLED | never started | stress-v1/12288 |
| stress | `qis-stress-base` | 11112932 | 577aa9f | COMPLETED | l40s | stress-v1/12288 |
| stress | `qis-stress-R-s101-r128` | 11112937 | 577aa9f | COMPLETED | l40s | stress-v1/12288 |
| stress | `qis-stress-U-s101-r128` | 11112938 | 577aa9f | COMPLETED | l40s | stress-v1/12288 |
| stress | `qis-stress-R-s202-r128` | 11112940 | 577aa9f | COMPLETED | l40s | stress-v1/12288 |
| stress | `qis-stress-H-s101-r128` | 11113928 | bcdc620 | COMPLETED | l40s→l40s→l40s | stress-v1/12288 |
| stress | `qis-stress-U-s202-r128` | 11115052 | e4deb79 | COMPLETED | l40s | stress-v1/12288 |
| stress | `qis-stress-R-s101-r64` | 11115377 | d06c7a5 | COMPLETED | l40s→l40s→l40s→l40s | stress-v1/12288 |
| stress | `qis-stress-H-s202-r64` | 11115378 | d06c7a5 | COMPLETED | l40s | stress-v1/12288 |
| stress | `qis-stress-H-s202-r128-k0` | 11119925 | 5a99bac | CANCELLED | never started | stress-v1/12288 |
| stress | `qis-stress-H-s202-r128-k1` | 11119926 | 5a99bac | CANCELLED | never started | stress-v1/12288 |
| stress | `qis-stress-H-s202-r128-k2` | 11119927 | 5a99bac | CANCELLED | never started | stress-v1/12288 |
| stress | `qis-stress-H-s202-r128-k0` | 11120596 | b77f066 | COMPLETED | l40 | stress-v1/12288 |
| stress | `qis-stress-H-s202-r128-k1` | 11120598 | b77f066 | COMPLETED | a6000 | stress-v1/12288 |
| stress | `qis-stress-H-s202-r128-k2` | 11121110 | bcf3e47 | COMPLETED | l40 | stress-v1/12288 |
| length | `qis-len-R-s202` | 11118157 | 5e70344 | CANCELLED | l40s | — |
| length | `qis-len-U-s202` | 11118245 | c894f2b | COMPLETED | l40s | — |
| length | `qis-len-R-s101` | 11118247 | c894f2b | CANCELLED | l40s | — |
| length | `qis-len-U-s101` | 11118248 | c894f2b | COMPLETED | l40s | — |
| length | `qis-len-R-s202` | 11118444 | a674984 | COMPLETED | l40s | — |
| length | `qis-len-R-s101` | 11118448 | a674984 | COMPLETED | l40s | — |
| length | `qis-lentest-R-s202` | 11118658 | ef68bf8 | COMPLETED | l40s | extend-test-v1 |
| length | `qis-lentest-U-s202` | 11118660 | ef68bf8 | COMPLETED | l40s | extend-test-v1 |
| length | `qis-lentest-R-s101` | 11118661 | ef68bf8 | COMPLETED | l40s | extend-test-v1 |
| length | `qis-lentest-U-s101` | 11118662 | ef68bf8 | COMPLETED | l40s | extend-test-v1 |
| length | `qis-lentest-H-s101` | 11119262 | 0416921 | COMPLETED | l40s | extend-test-v1 |
| length | `qis-len-H-s101` | 11119264 | 0416921 | COMPLETED | l40s | — |
| cue | `qis-cue-eval` | 11102440 | 594d408 | CANCELLED | never started | cue-v1/12288 |
| cue | `qis-cue-eval` | 11104738 | 6d4e33a | COMPLETED | l40s | cue-v1/12288 |
| diag | `qis-diag-local-r0` | 11079411 | 870c8f5 | CANCELLED | never started | — |
| diag | `qis-diag-local-r64` | 11084344 | e265c1c | CANCELLED | never started | — |
| diag | `qis-diag-local-r0` | 11099083 | f3c70b5 | CANCELLED | never started | — |
| diag | `qis-diag-local-r64` | 11099088 | f3c70b5 | FAILED | l40 | — |
| diag | `qis-diag-local-r64` | 11101908 | 4b47386 | COMPLETED | 6000ada | — |
| diag | `qis-diag-local-r0` | 11101910 | 4b47386 | CANCELLED | 6000ada | — |
| diag | `qis-interference` | 11102391 | 9742f50 | COMPLETED | none | — |
| diag | `qis-diag-local-r0` | 11102499 | 594d408 | COMPLETED | l40 | — |
| diag | `qis-diag-ref` | 11105174 | 042cebd | COMPLETED | l40s | — |
