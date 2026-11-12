# t10 — groups without a within-group reward contrast, per pool and 32-round quarter

Each cell: all-wrong / all-correct groups of 256 (8 slots × 32 rounds). A group is the 8 attempts of one slot in one round. Source: `research/audit-a26/pass-b/data/train_groups.json` (audit pass B).

| Run | Slots | Pool | r1–32 | r33–64 | r65–96 | r97–128 | Total of 1,024 |
|---|---|---|---|---|---|---|---|
| H-s101 | 0–7 | H_train (shared) | 32 / 8 | 3 / 60 | 5 / 114 | 5 / 140 | 45 / 322 |
| H-s101 | 8–15 | extra H_train | 31 / 13 | 12 / 65 | 4 / 113 | 8 / 146 | 55 / 337 |
| R-s101 | 0–7 | H_train (shared) | 27 / 9 | 9 / 69 | 13 / 114 | 9 / 128 | 58 / 320 |
| R-s101 | 8–15 | related practice | 0 / 76 | 0 / 176 | 0 / 226 | 0 / 231 | 0 / 709 |
| U-s101 | 0–7 | H_train (shared) | 22 / 12 | 8 / 69 | 17 / 99 | 13 / 121 | 60 / 301 |
| U-s101 | 8–15 | unrelated practice | 0 / 88 | 0 / 200 | 0 / 233 | 0 / 236 | 0 / 757 |
| H-s202 | 0–7 | H_train (shared) | 14 / 16 | 4 / 80 | 9 / 116 | 8 / 136 | 35 / 348 |
| H-s202 | 8–15 | extra H_train | 23 / 15 | 6 / 70 | 8 / 121 | 7 / 128 | 44 / 334 |
| R-s202 | 0–7 | H_train (shared) | 27 / 15 | 9 / 48 | 1 / 75 | 11 / 115 | 48 / 253 |
| R-s202 | 8–15 | related practice | 0 / 76 | 0 / 159 | 0 / 200 | 0 / 212 | 0 / 647 |
| U-s202 | 0–7 | H_train (shared) | 28 / 16 | 11 / 62 | 7 / 104 | 8 / 129 | 54 / 311 |
| U-s202 | 8–15 | unrelated practice | 0 / 90 | 0 / 190 | 0 / 229 | 0 / 227 | 0 / 736 |
