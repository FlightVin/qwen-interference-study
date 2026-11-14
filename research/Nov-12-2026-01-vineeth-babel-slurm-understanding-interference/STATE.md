# STATE — study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`

Purpose: the current scientific state of this study. Mandate: `agent-project-instructions-archive/<ID>/project-instructions.md`.
Plan: `CHECKLIST.md`. Handover: `HANDOVER.md`. Last update: 2026-11-13, 01:45 EST.

## Question
The first study trained Qwen3-1.7B with GRPO on hard algebra plus factoring drills (R), modular-arithmetic drills (U) or
more hard problems (H). It found no evidence of interference. This study asks why.

## Established (source: `reports/<ID>/01-understanding.md` and its `tables/`)
- Base-model group gradients of different problems are almost orthogonal with a small positive mean. Cross-domain cosine
  0.0003–0.0007; within-domain 0.0003–0.0013; same problem 0.020; conflict rates 48.7–49.8% (`qis-ui-geom-base64`).
- Domains do not separate: cross-domain overlap is 0.45–1.26 of within-domain overlap. Estimated cosine between
  expected domain gradients 0.21–0.56, all positive.
- Each of the six runs raised the GRPO objective of all four domains at base-model samples (24 of 24 positive;
  `qis-ui-lp`). Practice arms moved their own domain about 1.7–1.8 times more.
- Run updates share a cosine of 0.03–0.05 whatever the arm; arm-specific trace about 0.01 (`qis-ui-wdelta`).
- The shared part formed in rounds 1–64 (first halves 0.044, second halves 0.004).
- At trained checkpoints the same-problem cosine is 0.0037 ± 0.0024 pooled; cross-domain cosine stays ≥ 0.
- The averaged update is 2.4–3.3 times more effective than one run at equal norm; one run minus the mean of the
  others loses 75–96% of its effect (log-probability measure).
- Forward pass and sampler use BF16-rounded weights; 20–23% of weights differ from base after rounding. The output
  change at half the update is 8–10% of the full change.
- Precision test (`qis-ui-lpprec`, hard + related): FP32 weights follow alpha^2 (0.26–0.27 at half the update);
  weights rounded to BF16 reproduce the training setup (0.08–0.09). Rounding is the cause. Rounded weights give 41%
  (R-s101) and 15% (mean6) of the FP32 output change.
- Rank-r parts of the update (`qis-ui-lowrank-*`, FP32): rank 8 per matrix (0.6% of the parameters) gives 67–77% of
  the effect; the remainder after rank 64 (45–47% of the squared norm) gives 2–3%.
- Dev pass@1: base 41.4%, R-s101 72.4%, mean6 63.6%, scaled-R101 52.7%, resid-R101 43.3% (`qis-ui-soup`, `-b`).
- Domain probe at the base model (`qis-ui-probe`): 12 further domains, no negative coupling under the correctness
  reward. Products of k-digit integers give pass@1 97.7% (k=2) to 5.5% (k=6), 1.8% (k=7), 0.09% (k=8).
- A brevity reward opposes the correctness reward: 16 of 16 cross-problem cosines negative, 10 intervals below zero
  (`qis-ui-geom-short`). First negative coupling measured in either study.
- Trained R-s101 answers are 3.3 times longer than base answers on dev (5,753 against 1,734 tokens).

## Uncertain
- Dev pass@1 of loo5: 64.3% (mean6 63.6%). mean6x (norm 1.26): 44.8%, 55.2% of answers cut at the limit: overshoot.
- Whether the brevity conflict lowers accuracy in training: not run; proposed (`04`, mixture 7; `01`, E1).
- Gradient geometry on other model sizes at 64 answers. Out of scope (owner, 2026-11-12); future work, M1.
- Why the shared signal fades so early; what the shared direction is (answer length is one candidate).
- Everything about a second model size after training. The capacity hypothesis is untested.

## Verdicts
- Gradient alignment: consistent in the weak form (no conflict), not in the strong form (cosines are near zero).
- Capacity: not needed for the result; untested.
- Domains too similar: consistent. Too short / too small: not consistent as stated. Evaluation insensitive: true for
  the one-update test, not the reason for the main result.
- Ray interference: its preconditions (negative coupling, hard problems near zero success) were absent.

## Decisions
- See `DECISIONS.md` U1–U14.

## Next actions
1. All jobs are done and all deliverables are written. Nothing is running. Babel storage of this project was wiped on
   2026-11-13 on the owner's instruction (HANDOVER, last section): no checkpoint or environment exists there.
2. Owner decides which proposed experiments to run (Section 3 of the document; recommended order at its end).
