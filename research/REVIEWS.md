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
