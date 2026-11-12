# A26 pass A — stored checkpoints: inventory, integrity, resumable state, lineage (agent a-ckpt)

Purpose: answer A26 §2 "Every stored checkpoint" (`project.md` §40) for every retained project checkpoint. Written
2026-11-11, ~10:50 PM EST. Times are EST with UTC in brackets. Terms follow `research/DEFINITIONS.md`.
Labels: **verified** (primary evidence checked), **claimed** (only a document says so), **unknown**.
Data: `data/checkpoints.json` (one row per directory, unrounded). Replays: `replay.md`. Tools:
`research/audit-a26/tools/ckpt_audit.py` (jobs `qis-a26-ckpt-k0..3`, 11122477–11122481) and
`research/audit-a26/tools/ckpt_replay.py` (restore and replay jobs; `lineage` and `inventory` modes, stdlib, laptop).

## 1. Main results
| Question | Answer | Label |
|---|---|---|
| Retained directories | 11 under `$P/ckpt` (`du`: probe/ and superseded/ hold no checkpoint files) | verified |
| Fresh sha256 = `state.json` | 15/15 recorded files that exist match (9 directories with `state.json`); 3 recorded `optim.pt` files are absent (2 deleted by D117, 1 not copied to the fallback); 2 `.tmp` have no `state.json` | verified |
| Fresh sha256 = contemporaneous records | 50/50 `ckpt_identity.txt` lines, 19 manifest entries, 5 endpoint validators, 13 diagnostic rows: no conflict | verified |
| Load, tensors | 11/11 load; 311 FP32 tensors, 2,031,739,904 values; names and shapes = base Qwen3-1.7B; 0 non-finite; per-tensor bit checksums = `state.json` (9/9) | verified |
| Tied embeddings | `lm_head.weight` = `model.embed_tokens.weight` bitwise in 11/11 checkpoints and in the base model; one tied Parameter after loading | verified |
| Optimizer state | 8 `optim.pt`: 310 states, AdamW step = 4 × round, lr 2e-6, β (0.9, 0.999), ε 1e-8, wd 0, 0 non-finite; RNG keys python/numpy/torch/cuda | verified |
| Model identity | every checkpoint with a logged fingerprint reproduces it bit-exactly on the logging GPU type (8/8) | verified |
| Mismatches | none in hashes or tensors. One bookkeeping defect: R-s202 `counters.opt_step` 524 ≠ AdamW step 512 (D127) | verified |

## 2. Inventory
Model sha256 and optimizer sha256 are fresh (prefix shown; full values in JSON). "Rec." = matching contemporaneous records.
Rel. L2 = ‖θ − θ_base‖ / ‖θ_base‖ over all 311 tensors. Writer = the job whose save window contains `saved_utc`.
| Directory (`$P/ckpt/…`) | Arm, seed, round, trajectory | model.pt / optim.pt sha256 | = state | Rec. | Rel. L2 | Writer job, saved (EST) |
|---|---|---|---|---|---|---|
| qis-main-H-s101-sym/round_0128 | H, 101, 128, single segment | `052c8a62b300` / `ca7f80216b64` | yes / yes | 5 + 2 + 1 | 5.736e-4 | 11081726 (L40), Nov 9 4:47 AM [09:47Z] |
| qis-main-H-s202-sym/round_0064 | H, 202, 64, **a2-r60** | `37963facc1cc` / `7baa9ea2ad8d` | yes / yes | 1 + 1 | 4.318e-4 | 11112377 (L40S), Nov 10 10:26 PM [03:26Z] |
| qis-main-H-s202-sym/round_0128 | H, 202, 128, a2-r60 | `2513456224d4` / `f09a54ef8431` | yes / yes | 10 + 2 | 5.210e-4 | 11112377, Nov 11 4:44 PM [21:44Z] |
| qis-main-H-s202-sym-fallback/ckpt | H, 202, 116, a2-r60 (copy) | `7b9ff3a40080` / — | yes / file absent | 7 + 1 | 5.074e-4 | 11112377, Nov 11 12:35 PM [17:35Z]; copied 1:01 PM |
| qis-main-R-s101-sym/round_0064 | R, 101, 64, first segment | `6fe89540d80d` / `c1bf4d289b0f` | yes / yes | 1 + 1 + 13 diag | 4.086e-4 | 11081724 (L40S), Nov 8 7:15 AM [12:15Z] |
| qis-main-R-s101-sym/round_0128 | R, 101, 128, retry 125–128 | `aedcca6e6a44` / — | yes / deleted (D117) | 7 + 3 + 1 | 4.833e-4 | 11092945 (L40S), Nov 9 2:01 AM [07:01Z] |
| qis-main-R-s202-sym/round_0004.tmp | R, 202, 4 (partial) | `4c3e30d0f189` / `e38913dd2057` | no state.json | 0 | 1.346e-4 | 11081729 (L40); save failed (OSError 512) |
| qis-main-R-s202-sym/round_0012.tmp | R, 202, 12 (partial) | `a95959b83055` / `bc0d1c3095a0` | no state.json | 0 | 2.155e-4 | 11081729; save failed (OSError 512) |
| qis-main-R-s202-sym/round_0128 | R, 202, 128, segment 2 | `22fd8364b310` / `c5bf95c52e79` | yes / yes | 7 + 3 + 1 | 5.130e-4 | 11094190 (L40S), Nov 9 8:30 PM [01:30Z] |
| qis-main-U-s101-sym/round_0128 | U, 101, 128, single segment | `92aef0d2a53a` / — | yes / deleted (D117) | 6 + 3 + 1 | 4.941e-4 | 11081725 (L40S), Nov 8 9:39 PM [02:39Z] |
| qis-main-U-s202-sym/round_0128 | U, 202, 128, segment 3 | `7158b0952ee3` / `f4fab07f4e57` | yes / yes | 6 + 3 + 1 | 4.880e-4 | 11100159 (L40S), Nov 10 4:09 AM [09:09Z] |

Rec. = `ckpt_identity.txt` lines in `~/ae-qis/results/` (each "recorded = on-disk" at job start) + a17 manifest entries
+ endpoint validator / diagnostic rows. Every `state.json` holds attempt `a2`, bundle sha `0e255243…` (= the bundle
hashed now) and `config_sha` equal to the recomputed sha of the writer segment's config row (§4).
`LATEST.json` of all six runs points to `round_0128` (verified).

**Fresh hash, load and tensor checks** (`ckpt_audit.py`, 4 general-GPU jobs, 2026-11-11 8:50–9:25 PM EST): streamed
sha256 of every file; `torch.load(weights_only=True)`; names, shapes, dtypes vs the base safetensors (0 missing, 0 extra,
0 shape mismatches in 11/11); finiteness; per-tensor int64 bit sums vs `state.json` `checksums.model` (0/311 mismatches)
and `checksums.optim` (0 mismatches in 6/6 with both). Base = `Qwen3-1.7B` snapshot `70d244cc…`. All checkpoints are
small moves from the base (rel. L2 1.3e-4 to 5.7e-4), consistent with lr 2e-6 for 16–512 steps.

## 3. Tied embeddings
`Learner` comment: "FP32 (tied to embeddings in Qwen3-1.7B)" (`learner.py:69`); the base `config.json` has
`tie_word_embeddings: true`. Checked by `ckpt_replay.py` (`tied` in every result):
- Base safetensors: `lm_head.weight` equals `model.embed_tokens.weight` bitwise ([151936, 2048]). **verified**
- All 11 `model.pt` files: both keys present and bitwise equal (0 differing elements). **verified**
- After `Learner` init and after loading: `model.lm_head.weight is model.model.embed_tokens.weight` (one Parameter; this is
  why AdamW holds 310 states for 311 tensors). **verified** (13/13 loads)
- Why two copies exist on disk: `Saver.save` copies each state-dict entry with `.to("cpu", copy=True)` (`ckpt.py:56-58`),
  which breaks the storage sharing; file size 8,127,059,673 B ≈ 2,031,739,904 × 4 B. No effect on any result.
- `tests/test_lmhead.py` tests only the chunked LM-head recomputation on random tensors; it does not test tying.
- vLLM ties `lm_head` to the embedding; the evaluation load therefore uses the embedding values, which are identical.

## 4. Resumable state vs run records
Canonical step rows = last occurrence per (round, minibatch) in `metrics.jsonl` (copied 2026-11-11 ~9:30 PM EST).
| Checkpoint | AdamW step (all 310) | Canonical step rows before round | `counters.opt_step` | Config sha = writer config row | RNG keys |
|---|---|---|---|---|---|
| H-s101 r128 | 512 | 512 | 512 | `cc7462b4…` yes | 4 |
| H-s202 r64 | 256 | 256 | 256 | `94f67446…` yes (a2-r60 config, `retain_rounds` [64]; a2-orig config is `267a341d…`) | 4 |
| H-s202 r128 | 512 | 512 | 512 | `94f67446…` yes | 4 |
| R-s101 r64 | 256 | 256 | 256 | `a41b4281…` yes (same config in both segments) | 4 |
| R-s202 r128 | 512 | 512 (labels reach 524) | **524** | `30fdaaa9…` yes (segment 2, `ckpt_every` 16) | 4 |
| U-s202 r128 | 512 | 512 | 512 | `be957458…` yes (same config in all 3 segments) | 4 |
| R-s202 r0004.tmp / r0012.tmp | 16 / 48 | 16 / 48 | — (no state.json) | — | 4 |

- AdamW step always equals 4 × round and the canonical step count: every step row calls `learner.step()`, including
  zero-gradient steps (`run.py:319`). **verified**
- R-s202 `counters` are stale (async save serialised the live dict; fixed in `run.py:368`, D127; a-code D5). AdamW state is
  correct; only the counter fields are wrong. **verified**
- RNG state: present in all 8 `optim.pt`; restored by `ckpt.load` (`set_rng_state`). No training path draws from the global
  RNGs (named streams only; dropout off; `qtiles` subsamples only above 2,000,000 values), so the RNG state does not affect
  the trajectory. **verified (code)**
- Sampler position: the next round's prompts, roles and request seeds are a pure function of (arm, seed, round)
  (`schedule.round_plan`, `seeds.int_seed(seed, "gen/r{R}/p{j}")`); resume restarts at `meta["round"]` (`run.py:183,243`).
  The replays rebuilt the plan of rounds 65, 117 and 128 from the checkpoint round and matched every record (128/128 per
  round, 0 mismatches) and every logged `valid_tokens` (12/12). **verified**
- Restore-and-step for every distinct resume path: `replay.md`. Production loader 6/6, step-1 replay 2/2 within 1.94e-4,
  mechanics after round 128 4/4, `diag_local` restore 2/2, weights-only load 7/7, vLLM BF16 load 1/1.

## 5. Lineage
Segments from `start`/`resume` events; save windows from the round row to the `checkpoint` event of that round.
| Checkpoint | Parent trajectory | Evidence |
|---|---|---|
| H-s101 r128 | job 11081726, rounds 1–128, L40 | one `start`; save window 3:40–4:47 AM Nov 9 [08:40–09:47Z]; fingerprint exact on L40 |
| U-s101 r128 | job 11081725, rounds 1–128, L40S | one `start`; fingerprint exact on L40S |
| R-s101 r64 | job 11081724 (rounds 1–124), L40S | saved inside 11081724's round-64 window; fingerprint exact; round-65 replay step 1 within 7.5e-5 |
| R-s101 r128 | retry 11092945 resumed from round_0124 (pruned) | 11081724's final save failed (OSError 512 at 11:17 PM Nov 8 [04:17Z]); fingerprint equals the retry row exactly, superseded row off by 2.74e-3 |
| R-s202 r128 | 11094190 resumed from round_0016 (pruned) after 11081729 (rounds 1–19, L40) | config sha = segment 2; fingerprint exact |
| R-s202 .tmp | 11081729, failed saves at rounds 4 and 12 | `checkpoint_failed` OSError 512 at 10:39 PM and 12:29 AM [03:39Z, 05:29Z]; `optim.pt` mtimes (10:45 PM, 12:56 AM) are after the failure report (unknown why). Content unverifiable: no checksums |
| U-s202 r128 | 11100159, resumed from round_0064 (pruned) after 11094037 and 11097811 | config sha; fingerprint exact |
| H-s202 r64, r116, r128 | see below | |

**H-s202 verdict (verified).**
1. `round_0064` belongs to **a2-r60** (job 11112377, resumed from `round_0060`, sha `6ed5e01b…` per `resume_audit_a17.txt`).
   Its `saved_utc` (10:26 PM Nov 10 [03:26Z]) lies in 11112377's round-64 save window. Its `config_sha` is the a2-r60
   config (`retain_rounds` [64]). Its fingerprint equals the a2-r60 round-64 row bit-exactly (L40S). It misses the a2-orig
   row by 2.78e-2 on L40, a2-orig's own GPU type. Its round-65 replay matches the a2-r60 rows (step 1 1.94e-4). No a2-orig
   round-65 rows exist: job 11082192 was cancelled at 2:16 PM Nov 10 [19:16Z] during its round-64 save.
2. The **a2-orig round-64 policy was never saved.** Job 11082192 logged `checkpoint_started` round 64 at 2:15:45 PM
   [19:15:45Z] and no `checkpoint` event. `resume_audit_a17.txt` (8:37 PM [01:37Z]) lists `round_0064.tmp` with no
   `state.json`. The a2-r60 save removed it (`ckpt.py:160`, `rmtree(tmp)`); the directory is absent now.
3. The **round-116 fallback** is a copy of a2-r60's `round_0116`. Its `state.json` holds round 116, opt_step 464 and config
   `94f67446…`. It was saved at 12:35 PM Nov 11 [17:35Z] inside 11112377's window. The copy job recorded recorded = copy =
   `7b9ff3a4…`, the fresh hash.
4. `round_0128` was written by a2-r60 (11112377) at 4:44 PM Nov 11 [21:44Z]; its fingerprint is exact (L40S).
5. The **stress cell "H-s202 r64"** used `round_0064` = a2-r60: `qis-stress-H-s202-r64` recorded = on-disk = `37963fac…`,
   the hash verified here.

## 6. Uses of each checkpoint
From `ckpt_identity.txt`, the a17 manifest, `eval-map.md` (cue: summary hashes, claimed there) and `diag_r64.jsonl`.
| Checkpoint | Published evaluations |
|---|---|
| H-s101 r128 | sealed test; stress r128; length stress and test; control panel (D161) |
| H-s202 r128 | sealed test (6 shards); stress r128 (3 shards); control panel |
| H-s202 r64 (a2-r60) | stress "H-s202 r64" |
| H-s202 r116 fallback | sealed test fallback (6 shards; not an endpoint) |
| R-s101 r64 | stress "R-s101 r64"; local update diagnostic r64 and its reference reruns (13 rows, model and optim sha match) |
| R-s101, R-s202, U-s101, U-s202 r128 | sealed test; common-GPU rerun; stress r128; length stress and test; control panel; cue |
| R-s202 .tmp | none |

## 7. Weights-only saves: what can be validated
| Save | Validated | Not validated |
|---|---|---|
| R-s101 r128, U-s101 r128 | fresh sha = `state.json` = 7 / 6 job records; 311 tensor checksums; finite; tied; identity by exact fingerprint on L40S; BF16 vLLM load (U-s101) | optimizer, RNG (deleted, D117); resumption impossible |
| H-s202 r116 fallback | fresh sha = `state.json` = copy record = 6 shard records; checksums; finite; tied; config sha; save window | no logged quantity reproducible (no fingerprint at 116; step 465 has zero gradient); optimizer absent |
| R-s202 r0004.tmp, r0012.tmp | load; finite; names/shapes; tied; AdamW step 16/48 = 4 × round | no `state.json`, no checksums, no hash record; bytes after a failed write; not a valid checkpoint |

## 8. Unavailable (pruned or never saved)
Published results depended on these. Their integrity cannot be checked, and is not inferred from copied hashes.
- `round_0032` and `round_0096` policies of all six runs: in-run evaluations at rounds 32 and 96 (curves, A13 round 32).
- `round_0064` of H-s101, U-s101, R-s202 and U-s202: in-run round-64 evaluations. U-s202's was also the source of both U-s202 resumes.
- H-s202 a2-orig round-64 policy: never saved (§5). A16 appendix values come from in-run records only.
- Resume sources: H-s202 `round_0060`, R-s101 `round_0124`, R-s202 `round_0016`, U-s202 `round_0064`.
- H-s202 original `round_0116` (the fallback copy survives).
- `optim.pt` of R-s101 and U-s101 `round_0128` (D117): neither endpoint can resume training.
- The in-run evaluations were computed from the live policy, not from checkpoint files, so no evaluation read a pruned file.
