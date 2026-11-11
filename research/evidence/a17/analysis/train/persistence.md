# Apparent forgetting on the shared hard prompts (A17, exploratory)

Previously solvable (PS): >= 1 correct answer at an earlier appearance of the prompt in the same trajectory. Denominator: later appearances of PS prompts. Capped: answers cut at the length cap (0 / 1-7 / all 8). Next outcome: the group at the prompt's next appearance in the same trajectory.

| Run | Rounds | Later PS appearances | PS → all wrong | capped 0 / some / all | next: AW / M / AC / none | All-wrong groups (capped 0 / some / all) |
|---|---|---|---|---|---|---|
| H-s101 | r1-32 | 378 | 11 (0.029) | 9 / 2 / 0 | 2 / 9 / 0 / 0 | 32 of 256 (29 / 3 / 0) |
| H-s101 | r33-128 | 768 | 13 (0.017) | 11 / 2 / 0 | 8 / 4 / 0 / 1 | 13 of 768 (11 / 2 / 0) |
| H-s202-orig | r1-32 | 408 | 8 (0.020) | 4 / 4 / 0 | 2 / 6 / 0 / 0 | 14 of 256 (9 / 5 / 0) |
| H-s202-orig | r33-128 | 256 | 5 (0.020) | 3 / 2 / 0 | 0 / 4 / 0 / 1 | 5 of 256 (3 / 2 / 0) |
| H-s202-r60 | r1-32 | 408 | 8 (0.020) | 4 / 4 / 0 | 2 / 6 / 0 / 0 | 14 of 256 (9 / 5 / 0) |
| H-s202-r60 | r33-128 | 224 | 3 (0.013) | 2 / 1 / 0 | 0 / 3 / 0 / 0 | 3 of 224 (2 / 1 / 0) |
| R-s101 | r1-32 | 390 | 11 (0.028) | 6 / 5 / 0 | 0 / 11 / 0 / 0 | 27 of 256 (16 / 11 / 0) |
| R-s101 | r33-128 | 765 | 29 (0.038) | 12 / 13 / 4 | 18 / 10 / 0 / 1 | 31 of 768 (12 / 15 / 4) |
| R-s202 | r1-32 | 396 | 14 (0.035) | 10 / 3 / 1 | 1 / 13 / 0 / 0 | 27 of 256 (22 / 4 / 1) |
| R-s202 | r33-128 | 765 | 19 (0.025) | 13 / 3 / 3 | 7 / 10 / 0 / 2 | 21 of 768 (15 / 3 / 3) |
| U-s101 | r1-32 | 396 | 10 (0.025) | 7 / 3 / 0 | 2 / 8 / 0 / 0 | 22 of 256 (19 / 3 / 0) |
| U-s101 | r33-128 | 768 | 38 (0.049) | 7 / 23 / 8 | 17 / 18 / 0 / 3 | 38 of 768 (7 / 23 / 8) |
| U-s202 | r1-32 | 390 | 13 (0.033) | 10 / 3 / 0 | 3 / 10 / 0 / 0 | 28 of 256 (24 / 4 / 0) |
| U-s202 | r33-128 | 768 | 26 (0.034) | 18 / 6 / 2 | 13 / 11 / 1 / 1 | 26 of 768 (18 / 6 / 2) |

**Matched cells** (same prompt, same round; a = first run, b = second run):

| a vs b | Rounds | Cells | AW in a only | AW in b only | AW in both | PS→AW a / b / both |
|---|---|---|---|---|---|---|
| R-s101 vs U-s101 | r1-32 | 256 | 22 | 17 | 5 | 11 / 10 / 1 |
| R-s101 vs U-s101 | r33-128 | 768 | 17 | 24 | 14 | 29 / 38 / 13 |
| R-s202 vs U-s202 | r1-32 | 256 | 19 | 20 | 8 | 14 / 13 / 2 |
| R-s202 vs U-s202 | r33-128 | 768 | 11 | 16 | 10 | 19 / 26 / 10 |
| H-s202-r60 vs R-s202 | r1-32 | 256 | 7 | 20 | 7 | 8 / 14 / 4 |
| H-s202-r60 vs R-s202 | r33-128 | 224 | 3 | 8 | 0 | 3 / 6 / 0 |
| H-s202-r60 vs U-s202 | r1-32 | 256 | 5 | 19 | 9 | 8 / 13 / 4 |
| H-s202-r60 vs U-s202 | r33-128 | 224 | 1 | 7 | 2 | 3 / 9 / 2 |
| H-s202-orig vs R-s202 | r1-32 | 256 | 7 | 20 | 7 | 8 / 14 / 4 |
| H-s202-orig vs R-s202 | r33-128 | 256 | 4 | 8 | 1 | 5 / 7 / 1 |
| H-s202-orig vs U-s202 | r1-32 | 256 | 5 | 19 | 9 | 8 / 13 / 4 |
| H-s202-orig vs U-s202 | r33-128 | 256 | 2 | 8 | 3 | 5 / 11 / 3 |
