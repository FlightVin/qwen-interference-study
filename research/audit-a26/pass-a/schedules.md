# A26 pass A — training schedules from raw records (agent a-data)

Purpose: verify DEFINITIONS v1 §5 items 1–2 from the training records of the six canonical runs. Result: **every check
passed in all six runs**; the schedule, the R/U pairing and the minibatch grouping are as the code says. All findings are
**verified** (compute, 2026-11-11 EST, standby 11122432) unless marked.

**Inputs.** `$P/runs/qis-main-{H,R,U}-s{101,202}-sym/`: `config.json`, `metrics.jsonl` (config, round, step rows),
`records/train_r0000…r0127.jsonl.gz` (128 files × 128 attempt rows per run). **Tool:**
`tools/a-data/schedule_audit.py`, run with project code at `9b2fd05` (`schedule.round_plan`, `grpo.minibatch_plan`,
`seeds`; these files are unchanged since `9d6349a`, `git diff --stat` empty). Output with per-problem counts:
`data/a-data-schedule_audit.json`.

## 1. Canonical vs superseded segments

Canonical records = the 128 files directly in `records/`. Superseded copies were moved to `records/superseded-<job>/`
(R-s101, R-s202, U-s202) and `records_a2orig_r60-64/` (H-s202). I identified the canonical segment by matching each
file's group counts (all-wrong / mixed / all-correct) to the `round` rows of `metrics.jsonl` in file order.

| Run | Config rows (job → start round) | Rounds with >1 round row | Canonical file = last row | Superseded copies match |
|---|---|---|---|---|
| H-s101 | 11081726 → 0 | none | 128/128 | — |
| H-s202 | 11082192 → 0; 11112377 → 60 | 61–64 (2 rows) | 128/128 | `records_a2orig_r60-64` (4 files): row 0 (a2-orig) for 62–64; 61 ties |
| R-s101 | 11081724 → 0; 11092945 → 124 | 125–128 (2 rows) | 128/128 | `superseded-11081724` (4): row 0 for 126–128; 125 ties |
| R-s202 | 11081729 → 0; 11094190 → 16 | 17–19 (2 rows) | 128/128 | `superseded-11081729` (3): row 0 for 17–19 |
| U-s101 | 11081725 → 0 | none | 128/128 | — |
| U-s202 | 11094037 → 0; 11097811 → 64; 11100159 → 64 | 65–80 (3 rows), 81 (2) | 128/128 | `superseded-11094037` (16 files, 65–80) → row 0; `superseded-11097811` (17, 65–81) → row 1; ties where counts coincide |

"Ties": the superseded and canonical attempts gave identical group counts in that round, so counts alone cannot separate
them; neighbouring rounds do. Every config row carries `bundle_sha` `0e255243…8c51` (= fresh hash of the bundle) and
`cap_schedule [[0, 8192], [32, 12288]]` (11/11 rows). Step and round rows beyond 512/128 are the rerun rounds above
(H-s202 528/132, R-s101 528/132, R-s202 527/131, U-s202 644/161); analyses must take the last row per round, as I did.

## 2. Per-round slot checks (each run: 128 rounds × 16 slots = 2,048 groups)

| Check | Result per run |
|---|---|
| Slot `prompt_id` and `role` = `round_plan(arm, seed, rnd)` mapped through the bundle | 2,048/2,048 in all six |
| Request seed of each slot = `seeds.int_seed(seed, "gen/r{rnd}/p{slot}")` | 2,048/2,048 |
| 8 samples per slot, binary rewards, rows within a slot consistent | 2,048/2,048 |
| Recorded `adv` = recomputed group advantage (population std, ε 1e-6, 0 for all-equal) within 1e-4 | 2,048/2,048 |
| 8 distinct shared problems per round | 128/128 |
| H: 8 distinct extra problems, none equal to that round's shared problems | 128/128 (H-s101, H-s202) |
| R/U: 8 distinct pair indices per round | 128/128 |
| Group totals (all-wrong / mixed / all-correct) | H-s101 100/1,289/659; R-s101 58/961/1,029; U-s101 60/930/1,058; H-s202 79/1,287/682; R-s202 48/1,100/900; U-s202 54/947/1,047 — equal to DEFINITIONS §3.D |

## 3. Cross-arm checks within a seed (128 rounds per seed)

| Check | Seed 101 | Seed 202 |
|---|---|---|
| Shared-slot problem IDs identical in H, R, U, same slot order | 128/128 | 128/128 |
| Request seeds identical across H, R, U (all 16 slots) | 128/128 | 128/128 |
| R and U practice slots use the same pair indices in the same slot order | 128/128 | 128/128 |
| R and U minibatch plans identical | 128/128 | 128/128 |
| H and R put the shared groups into the same pairs within minibatches | 0/128 | 2/128 |

The last row confirms DEFINITIONS §4 item 4 on data: `minibatch_plan` iterates roles in sorted order
(`grpo.py:55`), so R/U (`aux` < `hard_common`) give the shared groups the second permutation of the stream and H
(`hard_common` < `hard_extra`) the first. Round 1, seed 101, slot indices per minibatch:
H `[1,7,10,13] [3,5,8,9] [0,6,14,15] [2,4,11,12]`; R = U `[2,5,9,15] [0,1,11,13] [6,7,8,14] [3,4,10,12]`.

## 4. Minibatch grouping and zero-gradient steps

Records do not store minibatch membership, and step rows do not store group IDs. I regenerated each round's plan with
`grpo.minibatch_plan(roles, 4, seeds.rng(seed, "minibatch/r{rnd}"))` and tested it against the canonical step rows
(last row per (round, minibatch)) in two independent ways:

| Check | Result per run |
|---|---|
| Step `valid_tokens` = sum of `n_tokens` of the 32 attempts in the planned minibatch | 512/512 in all six |
| `grad_norm_preclip == 0` ⇔ every planned group in the minibatch is all-correct or all-wrong | 512/512 in all six |
| Each minibatch has 2 groups of each role | 512/512 |
| Every zero-gradient step has adv = 0 in all 32 of its attempts | 18, 20, 66, 25, 53, 65 of 18, 20, 66, 25, 53, 65 |

Zero-gradient (momentum-only) steps and the first round with one (1-based): H-s101 18 (round 70), H-s202 20 (73),
R-s101 66 (38), R-s202 25 (59), U-s101 53 (27), U-s202 65 (15). These equal DEFINITIONS §3.D. The token-sum match
makes the regenerated grouping a fingerprint match, not only a zero-pattern match.

## 5. Presentations per problem over 128 rounds (canonical records)

Expected values are exact means (fixed totals): shared 8 × 128 / 46 = 22.26; H total 16 × 128 / 46 = 44.52; practice
8 × 128 / 63 = 16.25. The spread comes from the shuffled cycles and the within-round skip rule (`schedule.py:22-42`).

| Run | Shared (hard_common) min–max, mean | H extra min–max | H total min–max, mean | Practice min–max, mean | Never presented |
|---|---|---|---|---|---|
| H-s101 | 20–23, 22.26 (20:1, 21:4, 22:23, 23:18) | 18–26 | 40–48, 44.52 | — | 0/46 |
| R-s101, U-s101 | same as H-s101 (identical schedule) | — | — | 16–17, 16.25 (16:47, 17:16) | 0 |
| H-s202 | 20–23, 22.26 (20:2, 21:4, 22:20, 23:20) | 18–27 | 41–50, 44.52 | — | 0/46 |
| R-s202, U-s202 | same as H-s202 | — | — | 15–17, 16.25 (15:1, 16:45, 17:17) | 0 |

Each per-problem table is in `a-data-schedule_audit.json` → `runs.<run>.presentations.<role>.per_problem`. Totals:
1,024 shared and 1,024 extra or practice presentations per run (2,048 groups).

## 6. Disagreements and limits

- No disagreement with DEFINITIONS v1 §2–§4 on schedules, group totals or zero-gradient counts.
- DEFINITIONS §3.A "shared hard prompt slots" sampled 8 rounds; this check covers all 128. §4 item 4 ("not yet checked
  on data") is now verified.
- Report v2 does not state that H groups its shared groups into minibatches differently from R/U. Because each of
  the 4 steps changes the weights, the grouping changes which groups share an AdamW step and the gradients of later steps.
- Not checked here: generated tokens, log probs, checkpoint contents, evaluation records (other agents).
