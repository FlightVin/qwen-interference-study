# STATE — study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`

Purpose: the current scientific state of this study. Mandate: `agent-project-instructions-archive/<ID>/project-instructions.md`.
Plan: `CHECKLIST.md`. Last update: 2026-11-12 (study start).

## Question
The first study trained Qwen3-1.7B with GRPO on hard algebra plus factoring drills (R), modular-arithmetic drills (U) or
more hard problems (H). It found no evidence of interference. This study asks why.

## Established (from the first study; source `reports/Nov-12-2026-00-…/experiments-report.md`)
- Every arm gained 20–29 points on the sealed test and 21–28 points on untrained subjects.
- R and U scored at or above H in all four comparisons. R − U flipped sign between the two seeds (+4.2, −4.4 points).
- A one-update test could not resolve any cross-domain effect: reference noise was 2.5–3.1 points.
- The weights moved little: ‖θ − θ_base‖ / ‖θ_base‖ = 4.8e-4 to 5.7e-4 at round 128 (checkpoint audit).
- One model scale only (Qwen3-1.7B). No same-seed R and U checkpoints exist below round 128.

## Uncertain
- Whether per-domain gradients conflict at all (never measured).
- Whether the first-order (linear) description of training holds over the 512 steps.
- Whether a smaller model, stronger domain contrast or longer training would show interference.

## Hypotheses under test
- H1 gradient alignment from pre-training. H2 capacity. H3 small-step (near-linear) regime.
- H4 setup: domains too similar; practice saturated by round 64; evaluation too noisy to detect small interference.
- H5 the coupling that ray interference needs (success-dependent learning signal) was weak in our setup.

## Decisions
- Measurements use stored checkpoints only. No training (mandate).
- Gradients are measured on fresh on-policy samples at each checkpoint, 16 answers per problem, split into two groups
  of 8, so that sampling noise can be separated from true alignment.

## Next actions
1. Finish the literature read; write `00-initial-theory.md` with toy-model figures.
2. Run the measurement jobs (`qis-ui-*`): weight-update geometry, gradient geometry, linearity test.
3. Write `01-understanding.md`; update the README files.
