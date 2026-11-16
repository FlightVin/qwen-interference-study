# Runs — study s02 (Nov-15-2026-02)

Generated from the harness ledger (`state/LEDGER.jsonl`) and `sacct` on 2026-11-16 (EST). One row per experiment; the
last submission is shown, with the number of submissions. Commit = project commit the job was pinned to; recipe =
hash of `exp/<expid>.sh` (copies in `research/<ID>/recipes/`). Results = harvested distillates in the git-ignored
harness workspace `slurm-agentic-engineering/results/<expid>/`; full outputs on Babel `/data/user_data/vineethb/qis/s02/runs/`.

| expid | job | commit | recipe | subs | final state | end (EST) | note |
|---|---|---|---|---|---|---|---|
| `qis-s02-env` | 11167443 | `f08a94d` | `2433353964cc` | 1 | CANCELLED | 2026-11-15T23:49:43 | redundant environment job; cancelled |
| `qis-s02-calib` | 11167799 | `9604281` | `21069495da91` | 3 | COMPLETED | 2026-11-16T00:29:07 | two earlier attempts failed (tail syntax; fixed) |
| `qis-s02-calib2` | 11167621 | `627ad13` | `c9e4ec028cf8` | 1 | COMPLETED | 2026-11-16T00:23:37 |  |
| `qis-s02-canary` | 11168205 | `0288b41` | `2fbf14b92c0e` | 2 | COMPLETED | 2026-11-16T01:10:21 | cancelled once to repin to graders fix; final completed |
| `qis-s02-regrade2` | 11168204 | `602b151` | `96a23be72ec1` | 1 | CANCELLED | 2026-11-16T03:36:41 | superseded by in-run r0 evals; cancelled |
| `qis-s02-e1-caesar-s1` | 11168342 | `14be5a9` | `0e1ce603c731` | 1 | COMPLETED | 2026-11-16T02:32:04 |  |
| `qis-s02-e1-add9-s1` | 11168346 | `14be5a9` | `5085f0f88203` | 1 | COMPLETED | 2026-11-16T02:52:34 |  |
| `qis-s02-e1-mbpp-s1` | 11168352 | `14be5a9` | `ff0643d5ff9e` | 1 | COMPLETED | 2026-11-16T02:47:37 |  |
| `qis-s02-e1-terse-s1` | 11168353 | `14be5a9` | `d5bfaeed806f` | 1 | CANCELLED | 2026-11-16T06:42:05 | cancelled while held (C reference no longer needed) |
| `qis-s02-e1-countdown-s1` | 11168357 | `14be5a9` | `dc843947a22e` | 1 | CANCELLED | 2026-11-16T01:13:50 | dropped (D4) |
| `qis-s02-e1-arc-s1` | 11168359 | `14be5a9` | `3709df66c880` | 1 | CANCELLED | 2026-11-16T01:13:52 | dropped (D4) |
| `qis-s02-e1-add10-s1` | 11168361 | `14be5a9` | `0ba0d5cc8555` | 1 | CANCELLED | 2026-11-16T06:42:03 | cancelled while held (C reference no longer needed) |
| `qis-s02-j-caesar-terse-s1` | 11168639 | `8e70f56` | `4ae0f3427864` | 1 | COMPLETED | 2026-11-16T04:50:21 |  |
| `qis-s02-j-add9-add10-s1` | 11168641 | `8e70f56` | `0b8ff6ea1090` | 1 | COMPLETED | 2026-11-16T05:08:28 |  |
| `qis-s02-e3-add9mix-s2` | 11169764 | `102eae6` | `72442dc87e79` | 1 | FAILED | 2026-11-16T06:19:21 | FAILED at final checkpoint (disk full); all evals r0-r32 complete |
| `qis-s02-e3-add9d6-s1` | 11169765 | `102eae6` | `e051d336707f` | 1 | FAILED | 2026-11-16T06:19:21 | FAILED at final checkpoint (disk full); all evals r0-r32 complete |
| `qis-s02-e3-add9d8-s1` | 11171817 | `ea73639` | `07808f363d1e` | 2 | COMPLETED | 2026-11-16T08:59:41 |  |
| `qis-s02-e3-add9d4-s1` | 11171814 | `ea73639` | `a9b6424e9c19` | 2 | COMPLETED | 2026-11-16T07:38:27 | first attempt failed at precheck (disk full); this row is the rerun |
| `qis-s02-e1-caesar-s2` | 11171812 | `ea73639` | `49570b1d02fe` | 2 | COMPLETED | 2026-11-16T07:53:21 | first attempt failed (disk full); this row is the clean rerun |
| `qis-s02-j-caesar-terse-s2` | 11171816 | `ea73639` | `b587267d1c14` | 2 | COMPLETED | 2026-11-16T08:55:11 |  |
| `qis-s02-diag-storage` | 11171686 | `9a6f560` | `5a9a6e9719aa` | 1 | FAILED | 2026-11-16T06:22:38 | diagnosis; failed writing on full disk |
| `qis-s02-diag-du` | 11171696 | `9a6f560` | `b8839d389119` | 1 | FAILED | 2026-11-16T06:30:34 | diagnosis; partial output |
| `qis-s02-prune-1` | 11171791 | `9a6f560` | `1779bf48ef60` | 1 | COMPLETED | 2026-11-16T06:31:42 |  |
| `qis-s02-diag-df` | 11171795 | `ea73639` | `4ff3fe0eb830` | 1 | COMPLETED | 2026-11-16T06:38:48 |  |
| `qis-s02-e3b-r16-mixed` | 11171949 | `baeda29` | `95d7ddc4266a` | 2 | COMPLETED | 2026-11-16T08:30:22 | first attempt failed on bad node babel-l9-32; this row is the rerun |
| `qis-s02-e3b-r16-d6` | 11171953 | `baeda29` | `f15e5fc5e245` | 2 | COMPLETED | 2026-11-16T08:43:56 | first attempt failed on bad node babel-l9-32; this row is the rerun |
| `qis-s02-e3b-r16-d6n32` | 11171994 | `baeda29` | `e49da21ceff7` | 2 | COMPLETED | 2026-11-16T08:22:30 | first attempt failed on bad node babel-l9-32; this row is the rerun |
| `qis-s02-an-strata` | 11172404 | `5b173e6` | `f7db0c767540` | 1 | COMPLETED | 2026-11-16T09:38:21 |  |
| `qis-s02-e3-add9mix-s3` | 11172967 | `417184d` | `cfa037cd3cf7` | 1 | CANCELLED | 2026-11-16T08:46:49 | cancelled after ~10 min; rerun as -s3r with retained rounds |
| `qis-s02-e3-add9mix-s3r` | 11173101 | `8abdf93` | `412c60af3927` | 1 | COMPLETED | 2026-11-16T10:26:31 |  |
| `qis-s02-j-caesar-mbpp-s1` | 11173146 | `29cb12d` | `929d62ba6712` | 1 | COMPLETED | 2026-11-16T10:32:33 |  |
| `qis-s02-j-add9-terse-s1` | 11173148 | `29cb12d` | `b30dfa39260d` | 1 | COMPLETED | 2026-11-16T13:05:05 |  |
| `qis-s02-e3-add9d4d6-s1` | 11173237 | `4eb0bd4` | `56252ea23370` | 1 | COMPLETED | 2026-11-16T11:07:23 |  |
| `qis-s02-e3-add9d4d6-s2` | 11173240 | `4eb0bd4` | `2e38587ad656` | 1 | COMPLETED | 2026-11-16T10:40:17 |  |
| `qis-s02-e1-caesar-lr1e6-s1` | 11174276 | `2399dee` | `a51b82179659` | 1 | COMPLETED | 2026-11-16T12:43:08 |  |
| `qis-s02-e3-add9d4d6-s3` | 11174313 | `305dc56` | `b42c5140b917` | 1 | COMPLETED | 2026-11-16T13:02:51 | preempted twice; resumed; rounds 17-24 duplicated in metrics |
| `qis-s02-e1-caesar-s3` | 11175477 | `d2c5199` | `c45380d13932` | 1 | COMPLETED | 2026-11-16T13:56:38 |  |
| `qis-s02-j-caesar-terse-s3` | 11175478 | `d2c5199` | `f92658a394af` | 1 | COMPLETED | 2026-11-16T14:10:12 |  |
| `qis-s02-e3-add9mix-s4` | 11176691 | `338c994` | `d67bbf292395` | 1 | COMPLETED | 2026-11-16T16:23:32 |  |
| `qis-s02-e3-add9mix-s5` | 11176692 | `338c994` | `68502a494c3b` | 1 | COMPLETED | 2026-11-16T16:13:53 |  |
| `qis-s02-e3-add9d4d6-s4` | 11176693 | `338c994` | `6c4c7bd53ece` | 1 | COMPLETED | 2026-11-16T15:23:32 |  |
| `qis-s02-e3-add9d4d6-s5` | 11176696 | `338c994` | `7526d30f5bba` | 1 | COMPLETED | 2026-11-16T15:58:47 |  |
