# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-07 03:10Z (review 1).

## Phase
Setup → data construction and trainer implementation (in parallel). No GPU training yet.

## Question
Does easier practice on polynomial factorization (E_related) improve GRPO learning on hard MATH
problems that use it, compared with matched modular-arithmetic practice (E_unrelated)?
Primary contrast: seed-mean R − U endpoint pass@1 on target_test. Model Qwen3-1.7B, non-thinking.

## Established
- Harness isolated and self-tested (2026-11-07). Account limits verified (harness `cluster/CARD.md`).
- Pinned data: `EleutherAI/hendrycks_math` @ `21a56338` (7,500 train / 5,000 test).
- Keyword estimate: ~564 algebra/intermediate-algebra train items mention factoring; ~264
  number-theory train items are modular without factoring. Role pools may be tight for H (2–8/32).

- Env `venv-v1` built; split manifest `splits-v1` built; verifier/split tests 11/11 passed.
- Trainer design (D10): custom single-GPU GRPO; verl 0.9.0 deviates from our spec in advantage std, dual-clip, minibatch order, truncation visibility.
- Owner addendum A1 (2026-11-07): precision audit at gate G2; four report figures incl. POPE-style J trajectories on fixed probe panels. No existing run affected.

## Uncertain
- Whether ≥64 H_train items reach 2–8/32 success (Qwen3-1.7B solves many MATH algebra items).
- Whether full-parameter GRPO with colocated vLLM fits one 48 GB GPU.

## Resource facts
- Another project uses 6 general GPUs and ~62/64 account CPUs on `preempt_cpu_qos`.
  Our CPU jobs run at 2 CPUs; general GPU slots (2) start within ~1 min.

## Active / queued owned jobs
See `slurm-agentic-engineering/bin/ae status` (env build `qis-env-v1`, data build `qis-data-v1`).

- Pass-1 annotation done; rubric v1.1 roles give construction pools H 154 / R 110 / U 152 (pre-screening).
- GPU probe passed; checkpoints are 20.6 GB and NFS writes ~40 MB/s → background validated writer; one validated + one in-flight checkpoint per run (storage).

## Next actions
1. Length probe → freeze response cap → stage-1 screening (study pools + calibration subsets).
2. Trainer smoke (G2): precision audit, resume equality, real-round profile.
3. Pass-2 blind annotation on screened/selected items; error audit; matching.
