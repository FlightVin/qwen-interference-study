# REVIEWS
- 2026-11-07 test trigger received at 2026-11-07T02:30:02Z (one-shot cron 9cba2ef9; scheduler path verified)

## Review 1 — 2026-11-07 03:05Z
1. **Changes.** Done: env (+patch), split manifest, CPU tests 29/29, GPU probe, pass-1 annotation (1,496 items), rubric v1.1. Running: `qis-lenprobe-v1` (11072430), `qis-trainer-smoke-v1` (11072428), both general. Failed/cancelled and recorded: probe attempts 1–2 (recipe/env), lenprobe 11072286 and smoke 11072408 (skipped-precheck bug), tests 11072230 (env).
2. **Question alignment.** Both active jobs are gate work (cap decision; trainer correctness + precision audit). No drift: no result-seeking; no study outcome exists yet.
3. **Mechanics.** Probe: template correct, EOS present, all params trainable, update reaches vLLM. Open: logprob mismatch p99 0.127 (BF16 head in probe; trainer uses FP32 head — re-measure in smoke); checkpoint write 512 s (fixed by background writer, untested until smoke).
4. **Confounds/claims.** Annotation contamination in batches 2/5 caught by the quote check and redone (D13). Rubric v1.1 changed roles before any screening (D16) — documented as count-driven, outcome-blind. Risk: E_related may fall below 64 after screening.
5. **Next jobs / limits.** Stage-1 screening waits for the cap decision. GPU use: 2/2 general, 0/6 preempt. Storage: 163–173 GB used of 500 GB account; project ~10 GB. CPU work possible: pass-2 annotation prep.
6. **Most important uncertainty.** Whether the trainer passes G2 (resume equality, precision audit) on the first try; the background checkpoint path is new code. Action: smoke test running.
7. **Reconciliation.** Slurm ↔ ledger agree (14 ledger rows; 2 live jobs both owned). `registry/runs.jsonl` is empty because no scientific run exists; engineering attempts are in the ledger. STATE/CHECKLIST updated in this review.

## Review 2 — 2026-11-07 04:05Z
1. **Changes.** Done: G2 passed (D24–D26); stage-1 study screening; cap frozen 4,096 (D18); A2/A3 recorded; `ops/`, `registry/` untracked (D21). Running: `qis-screen2-study` 11072896 (general), `qis-screen1-calib` 11072944 (general, moved from preempt per D23). Queued: panels 11072990 (CPU) → six calibration runs (preempt, dependency).
2. **Alignment.** Screening and calibration both serve gates G1/G3. No outcome-driven choice: the H reserve (D19) and rubric v1.1 (D16) were fixed before the corresponding screens.
3. **Mechanics.** Stage 1: 0 verifier unknowns, truncation ≤ 3.4%. Smoke: mismatch p99 ~0.11, clip fractions ~1e-4, 5/16 zero-gradient steps (all-solved minibatches) — expected. Background checkpoint slows concurrent training ~3–5× while writing (D26) — watch on real rounds.
4. **Confounds/claims.** Main risk is construction validity: H_train projected 25–35 (< 64 hard minimum); fallback skill no better (model-free bound). This is a data-construction blocker, not a treatment result. Escalate after stage-2 exact counts.
5. **Next jobs / limits.** GPUs: 2/2 general running; 6/6 preempt slots used by pending calibration jobs (0 running). CPU: 1 pending. Storage: smoke checkpoints (~44 GB) to prune after this review; calibration will add up to 6 × 21.9 GB.
6. **Most important uncertainty.** Whether a defensible H_train ≥ 64 exists. Action: stage-2 counts, then options to the owner (band widening recommended provisionally).
7. **Reconciliation.** Slurm ↔ ledger agree (all 11 owned live/pending jobs in ledger). Registry untracked (D21). STATE/CHECKLIST current as of 03:45Z plus calibration queue entry.

## Review 3 — 2026-11-07 05:05Z
1. **Changes.** Done since review 2: G2 evidence tracked; full native stage 2 (5,888 attempts) → 29 eligible H (18 core, 11 moderate); target/probe panels frozen (D34); calibration panels frozen (D33); authored tranche (400) written, intake (394 kept), blind-annotated; Omni-MATH tranche (200) annotated (73 H-eligible); amendments A4, A5, v1.3 tagging recorded. Running: calibration lr 1e-6/sym (11072995, round 4 of 32), authored screening (11073304). Pending: Omni screening + 5 calibration runs on preempt (no start estimate; D37).
2. **Alignment.** All work builds the frozen dataset or calibrates the optimizer; no treatment run exists. Every selection rule was committed before its outcome data (D28, D31, D32, D38, D39).
3. **Mechanics.** Real calibration rounds 138–178 s (gen ~60 s, train 60–116 s), 11–15/16 mixed groups, truncation 1–3/128, logprob mismatch p99 0.107 (D25 limit 0.20), clip fractions ~1e-4, device memory 36–38.5 GiB. Baseline calib_val pass@1 0.4265.
4. **Confounds.** Authored auxiliary items are matched within provenance (D32/D39). Omni items enter H_train only (provenance recorded; held-out MATH untouched). Watch: if H_train is majority Omni, training distribution shifts from MATH targets — report transfer separately (A4.6).
5. **Next jobs / limits.** GPU 2/2 general used; preempt unusable now. Cost model: main run ≈ 128 × 2.7 min + evals ≈ 6.5 GPU-h (< 24 h threshold → 128 rounds feasible); 15 runs ≈ 100 GPU-h ≈ 50 h wall on 2 GPUs.
6. **Most important.** Omni screening is the critical path for H_train but waits for a GPU slot. Action: it takes the next free general slot (before remaining calibration runs).
7. **Reconciliation.** Slurm ↔ ledger agree (8 live owned jobs). Watcher was found not re-armed after the last event; re-armed now (rule: re-arm in the same step as harvesting).

## Review 4 — 2026-11-07 06:06Z
1. **Changes.** Omni screening done (195 after intake; core 33, moderate 14 before audit filters; 2 test-overlap items excluded at intake). Authored tranche 1 screened (R 9 / U 30 eligible in band); tranche 2 (D40) authored, intake (296 R / 118 U), blindly annotated (218 R / 120 U eligible), screening running on general (11073719). Combined H selection queued on CPU (11073966, partition Priority). Calibration run 1 at ~1 h 35 min.
2. **Alignment.** All work is dataset construction or calibration. Selection/matching rules were committed before their outcome data (D38, D39). Difficulty calibration given to authors used auxiliary screening counts only (no treatment data exists).
3. **Mechanics.** Omni truncation 5.4% of attempts (selection excludes truncation-driven items). Verifier unknowns: 1 native, 1 Omni — excluded and logged, never dropped silently. Calibration rounds healthy (review 3).
4. **Confounds.** Authored items now dominate the auxiliary pools: within-provenance matching (D32/D39) keeps synthetic share equal between R and U. Parameterized authoring (D41) handled by family counting. H_train may be majority external (Omni): MATH-target transfer must be reported separately from training-pool performance (A4.6).
5. **Next jobs / limits.** GPU 2/2 general; preempt still not starting (D37). CPU: 1 job pending (queue of 85 on the CPU partition).
6. **Most important.** Finish H selection + R/U matching → freeze dataset; stage-3 baseline. Calibration throughput (2 GPUs) is the next bottleneck: 5 runs + 2 seed-22 runs ≈ 13 GPU-h.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned live jobs). Watcher alive (re-armed 06:0x after the Omni event).

## Review 5 — 2026-11-07 07:05Z
1. **Changes.** D45: cap 4,096 → 8,192 after calibration showed validation truncation 11.5% (genuine long reasoning, not loops). Cancelled six owned 4,096 jobs (~56 GPU-min). H selection at 4,096 reached 49 (D43, superseded pending re-screen). Running: 8,192 re-screen H+calibration (11074495; 107 H items) and auxiliary (11074499; 302 related items at stage 1, 5.1k tok/s). Queued: panels-k8 → six cal8k runs.
2. **Alignment.** The cap change protects the endpoint from a growing truncation confound; decided before any treatment run and applied to every pool.
3. **Mechanics.** 8,192 generation runs (vLLM max_seq_len 9,216). Learner at 9k tokens untested → first cal8k run is the canary.
4. **Confounds.** Pools must be re-selected from 8,192 counts with the same frozen rules (D38/D39); no rule changes.
5. **Next jobs / limits.** GPU 2/2 general; preempt still not starting. Storage ~120 GB used account-wide.
6. **Most important.** The 8,192 canary (memory, round time) — risk of OOM with 9k-token sequences. Action at first cal8k start: check first-round metrics within ~10 min.
7. **Reconciliation.** Slurm ↔ ledger agree (9 live owned jobs). Watcher alive.

## Review 6 — 2026-11-07 08:05Z
1. **Changes.** Aux re-screen 11074499 failed after stage 1 (shortlist script hard-coded the `s1` stage filename; my bug); fixed and resubmitted as 11074744, resuming at shortlist (stage-1 8k counts reused, 541 items). H+calibration re-screen 11074495 finished H (107 items), now screening calibration candidates.
2. **Alignment.** Unchanged: dataset + calibration at the 8,192 cap; no treatment run.
3. **Mechanics.** 8k generation 3.8–6.5k tok/s; no verifier errors reported in logs.
4. **Confounds.** None new.
5. **Next jobs / limits.** GPU 2/2 general (both re-screens); 7 jobs waiting on dependencies (panels-k8, six cal8k).
6. **Most important.** Calibration remains the long pole (≈6 runs × ~3 h at 8k on 2 GPUs). Action: when the H job ends, move the first cal8k run (canary) to general immediately.
7. **Reconciliation.** Slurm ↔ ledger agree; failed attempt 11074499 kept in ledger (cost ~41 GPU-min, stage-1 output reused). Watcher alive.
