# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-07 02:40Z.

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

## Uncertain
- Whether ≥64 H_train items reach 2–8/32 success (Qwen3-1.7B solves many MATH algebra items).
- verl 0.9.0 semantics vs our GRPO spec (audit running).
- Whether full-parameter GRPO with colocated vLLM fits one 48 GB GPU.

## Resource facts
- Another project uses 6 general GPUs and ~62/64 account CPUs on `preempt_cpu_qos`.
  Our CPU jobs run at 2 CPUs; general GPU slots (2) start within ~1 min.

## Active / queued owned jobs
See `slurm-agentic-engineering/bin/ae status` (env build `qis-env-v1`, data build `qis-data-v1`).

## Next actions
1. Finish env; run data build + verifier tests.
2. Read verl audit; decide trainer design; write GPU probe (template, generation, optimizer step, memory).
3. Write rubric; annotate candidates (pass 1).
