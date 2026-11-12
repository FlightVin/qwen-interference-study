# A26 pass A — checkpoint restore-and-step replay (agent a-ckpt)

Purpose: prove that each retained checkpoint is the exact state of the trajectory that logged the training rows, by
restoring it with the production loaders and reproducing logged quantities. Written 2026-11-11, ~10:45 PM EST.
Inventory and lineage: `checkpoints.md`. Machine-readable results: `data/checkpoints.json` (`rows[].replays`).
Labels: **verified** (checked against primary evidence), **claimed**, **unknown**.

## 1. Main results
| Check | Result | Label |
|---|---|---|
| Production loader `ckpt.load` (file sha256 + per-tensor bit checksums) | 6/6 checkpoints with `optim.pt` load; AdamW step = 256 (r64) or 512 (r128) on all 310 parameter states | verified |
| Entropy fingerprint, same GPU type as the logging run | 8/8 bit-identical to the logged `entropy_sample_mean` (difference 0.0) | verified |
| Round-65 replay, step 1 (opt_step 257) | grad norm within 1.94e-4 (H-s202 r64) and 7.5e-5 (R-s101 r64); loss and valid tokens exact | verified |
| H-s202 round_0064 trajectory | a2-r60 (bit-identical fingerprint); a2-orig rejected (fingerprint off by 2.78e-2 on L40, 2.61e-2 on L40S) | verified |
| R-s101 round_0128 trajectory | retry job 11092945 (bit-identical); superseded first attempt rejected (2.74e-3) | verified |
| One disposable step after round 128 (H-s101, H-s202, R-s202, U-s202) | AdamW step 512 → 513, loss and weights finite; no logged reference exists | verified (mechanics only) |
| `diag_local` snapshot/restore | model and optimizer bit checksums equal `state.json` after a 4-step replay and restore (2/2) | verified |
| `evaluate_ckpt` BF16 vLLM load (U-s101 r128) | 226/226 vLLM parameters bitwise equal to the BF16 cast; `lm_head` tied to the embedding | verified |
| H-s202 round-116 fallback (weights only) | load and checksums pass; the only replayable step (opt_step 465) has zero gradient, so it does not discriminate | verified load; identity rests on hashes |

## 2. Design
Tool: `research/audit-a26/tools/ckpt_replay.py` (imports `qis.*` on purpose; same file content at every job commit,
`git diff 5253c0b <sha>` empty for 49196bc, 317ba65, 48901a3, fbd042e). Training code at those commits equals the
training jobs' code except `run.py:368` (counter copy; a-code `code.md` §2). The tool never builds `ckpt.Saver`. All
outputs are under `$P/audit-a26/a-ckpt/` and in the harvested `results/qis-a26-replay-*/`.

1. **Restore.** Build `Learner(cfg["model"], cfg["lr"])`, then `ckpt.load(dir, model, opt)`, as `run.py:171-184`. The
   loader checks file sha256 against `state.json` and all 311 model and 310×3 optimizer tensor bit checksums. Weights-only
   saves: `torch.load(weights_only=True)` on bytes hashed while read, then `load_state_dict(strict=True)`.
2. **Entropy fingerprint.** `run.py:350-352` logs, at every 8th round, the mean token entropy of the round-end learner on
   the first 8 attempts of that round (slot 0). The round-end learner is exactly the saved state. The tool recomputes it
   from the checkpoint on the same records (`train_r{R-1}`), with the production `Learner.token_entropy`. This is a pure
   forward pass, so it is deterministic on one GPU type.
3. **Round replay.** For 0-based round R: plan from `round_plan(arm, seed, R)` (checked against every record's prompt,
   role, slot, sample and request seed); advantages recomputed from rewards (equal to the stored ones, max diff 0.0);
   π_old = round-start learner, no grad; minibatches from `grpo.minibatch_plan(roles, 4, seeds.rng(seed, "minibatch/rR"))`;
   loss −obj/32 per attempt, backward per attempt, clip at 1.0, `learner.step()`. Compared with the last logged `step`
   rows of round R (canonical) and with round row R+1 (clip fractions, ratio quantiles).
4. **Displaced-state control.** After the 4-step replay, recompute π_old and minibatch 0's gradient norm (no step) and the
   fingerprint. This shows how large a mismatch a state one round away produces.
5. **Loaders.** `diag_local.snapshot` before and `restore` after the replay, then bit checksums vs `state.json`.
   `evaluate_ckpt.py:44-63` path: BF16 cast pushed into vLLM, then every vLLM parameter read back and compared bitwise
   (fused `qkv_proj` and `gate_up_proj` rebuilt from the checkpoint).

**Tolerances.** Chosen from a calibration computed before the replays were submitted (~9:40 PM EST), but not written to a
file before the results. U-s202 round 65 was logged three times from one round_0064 save (jobs 11094037, 11097811,
11100159, all L40S). The 128 responses were identical in all three (128/128, `records/superseded-*`). Step-1 grad norm:
restore vs restore 5.3e-6, in-memory vs restore 1.94e-4. Steps 2–4: 0.06%–6.6% spread.
| Quantity | Tolerance | Strength |
|---|---|---|
| valid tokens per step | exact | data identity |
| step-1 (first minibatch) grad norm | relative ≤ 1e-3 | strong: depends only on weights and responses (ratio = 1) |
| steps 2–4 grad norm | relative ≤ 0.10 | weak: the first AdamW step amplifies kernel noise |
| entropy fingerprint | relative ≤ 1e-3 | strong only on the logging GPU type (see §3.1) |

## 3. Results
Jobs (all general partition, COMPLETED; GPU from each job's `gpu.txt`): 11122955 hs202r64 L40S p5-24 (9 min 47 s);
11122982 rs101r64 L40S u5-28; 11122986 r128-h L40S o5-28; 11122988 r128-ru L40S o9-16; 11122990 wonly **L40** w5-28;
11123184 fp-l40s L40S m5-32; 11123185 fp-l40 L40 w5-20.

### 3.1 Entropy fingerprints
Logging GPU: H-s101 L40; H-s202 a2-orig L40; H-s202 a2-r60, R-s101, R-s202 (round ≥ 17), U-s101, U-s202 L40S (`start` events).
| Checkpoint | Reference row (metrics) | Logged | Recomputed | Replay GPU | Relative diff |
|---|---|---|---|---|---|
| H-s202 round_0064 | round 64, a2-r60 (job 11112377) | 0.17636318318545818 | 0.17636318318545818 | L40S | 0.0 |
| H-s202 round_0064 | round 64, a2-orig (job 11082192) | 0.19231146480888128 | 0.18696933984756470 | L40 | 2.78e-2 |
| H-s202 round_0064 | round 64, a2-orig | 0.19231146480888128 | 0.18728488311171532 | L40S | 2.61e-2 |
| R-s101 round_0064 | round 64 (job 11081724) | 0.2137958575040102 | 0.2137958575040102 | L40S | 0.0 |
| H-s101 round_0128 | round 128 (job 11081726) | 0.22209162823855877 | 0.22209162823855877 | L40 | 0.0 |
| H-s202 round_0128 | round 128 (job 11112377) | 0.2616247180849314 | 0.2616247180849314 | L40S | 0.0 |
| R-s101 round_0128 | round 128, retry (job 11092945) | 0.40344787761569023 | 0.40344787761569023 | L40S | 0.0 |
| R-s101 round_0128 | round 128, superseded (job 11081724) | 0.406713105738163 | 0.40559953451156616 | L40S | 2.74e-3 |
| R-s202 round_0128 | round 128 (job 11094190) | 0.26003354229032993 | 0.26003354229032993 | L40S | 0.0 |
| U-s101 round_0128 | round 128 (job 11081725) | 0.22742654383182526 | 0.22742654383182526 | L40S | 0.0 |
| U-s202 round_0128 | round 128 (job 11100159) | 0.2893276959657669 | 0.2893276959657669 | L40S | 0.0 |

Cross-GPU runs (not used for identity decisions): H-s101 r128 on L40S 1.33e-3; R-s101 r128 on L40 1.94e-3; U-s101 r128 on
L40 1.29e-3; H-s202 r64 vs a2-r60 on L40 1.94e-3. This exceeds the 1e-3 tolerance and a one-round displacement (below),
so the first weights-only job (L40) was repeated on the logging GPU type (jobs 11123184, 11123185).
**Discrimination:** a state one round away changes the fingerprint by 3.86e-3 (H-s202) and 8.4e-4 (R-s101) on L40S.
The same-GPU match is exact, so any displaced state would have been detected.

### 3.2 Round-65 replay (0-based 64)
| Checkpoint | Step (opt_step) | Grad norm logged | Grad norm replay | Rel. diff | Loss logged / replay | Valid tokens |
|---|---|---|---|---|---|---|
| H-s202 r64 | 257 | 0.2672129273414612 | 0.2672647535800934 | 1.94e-4 | 1.86e-9 / 1.86e-9 | 182,234 = |
| H-s202 r64 | 258 | 0.4395374059677124 | 0.4552682936191559 | 3.58e-2 | 1.87e-5 / −3.80e-5 | 141,872 = |
| H-s202 r64 | 259 | 0.4669998288154602 | 0.4620002508163452 | 1.07e-2 | 4.09e-5 / −6.45e-6 | 113,460 = |
| H-s202 r64 | 260 | 0.14085298776626587 | 0.15157456696033478 | 7.61e-2 | −7.93e-6 / −1.17e-5 | 135,011 = |
| R-s101 r64 | 257 | 0.24150973558425903 | 0.24152787029743195 | 7.5e-5 | −3.73e-9 / −3.73e-9 | 102,629 = |
| R-s101 r64 | 258 | 0.15646030008792877 | 0.15850235521793365 | 1.31e-2 | −5.59e-6 / 2.59e-6 | 176,189 = |
| R-s101 r64 | 259 | 0.1664227396249771 | 0.15753743052482605 | 5.34e-2 | −1.34e-5 / −1.71e-6 | 186,811 = |
| R-s101 r64 | 260 | 0.0 | 0.0 | 0 (zero-gradient step) | 0 / 0 | 107,138 = |

Step 1 passes the strong tolerance in both runs; steps 2–4 pass the weak one. The step-1 difference of H-s202
(1.94e-4) equals the U-s202 in-memory-vs-restore difference. AdamW step after the replay: 260 on all 310 states; weights finite.
Round row 65 (logged / replay): H-s202 clipfrac_pos 3.13e-4 / 3.35e-4, clipfrac_neg 2.53e-4 / 1.85e-4, ratio q0.99
1.03944 / 1.03915; R-s101 clipfrac_pos 1.23e-4 / 1.03e-4, clipfrac_neg 1.13e-4 / 9.4e-5, ratio q0.99 1.039877 / 1.039889.
**Displaced-state control** (step-1 grad norm one round later vs logged): H-s202 1.34e-2, R-s101 7.33e-2, that is 69× and
977× the replay difference. The step-1 test therefore separates the saved state from a state one round away.

### 3.3 Other paths
- One step after round 128 (minibatch 0 of `train_r0127`, plan from `minibatch/r127`): H-s101 grad norm 0.27907, H-s202
  0.17632, R-s202 0.26234, U-s202 0.56850; AdamW 512 → 513; loss finite; weights finite. R-s202 `counters.opt_step` = 524
  but AdamW step = 512 = canonical step rows (stale counters, D127; a-code D5). No logged reference exists for step 513.
- `diag_local` snapshot/restore (the local-diagnostic path, unchanged since 4b47386 except `--reference-only`): after the
  4-step replay, restore returned model and optimizer bit checksums equal to `state.json` and AdamW step 256 (H-s202 r64,
  R-s101 r64).
- `evaluate_ckpt` load (U-s101 r128, L40): all 226 vLLM parameters equal the BF16 cast bitwise (fused tensors rebuilt);
  the only checkpoint tensor not read back is `lm_head.weight`, which vLLM ties to `embed_tokens`. Its FP32 values equal the
  embedding bitwise in every checkpoint (`checkpoints.md` §3), so the tie loses nothing.
- Fallback r116: minibatch 0 of round 117 (opt_step 465) logged grad norm 0.0 and replays 0.0 with 119,241 valid tokens.
  All 4 groups were non-mixed, so this step does not test the weights. Later steps need the momentum in the pruned `optim.pt`.
  No entropy row exists at round 116 (rounds divisible by 8 only).

## 4. What remains unverifiable
- Fallback r116 identity beyond hashes: the fresh sha256 equals `state.json`, the copy record and 6 sealed-test shard records,
  but no logged quantity can be recomputed from its weights.
- Steps 2–4 can be checked only to ~10% (chaotic after the first AdamW step, also between two restores of one checkpoint).
- Partial R-s202 `.tmp` saves: no `state.json` and no logged fingerprint at rounds 4 or 12.
- All pruned checkpoints (`checkpoints.md` §5).
