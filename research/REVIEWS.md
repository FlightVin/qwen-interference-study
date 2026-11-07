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
