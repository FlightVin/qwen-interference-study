# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-07 03:45Z (verified against Slurm and saved results).

## Phase
Dataset construction (stage-1 screening) and trainer gate G2, in parallel. No training run yet.

## Question
Does easier practice on polynomial factorization (E_related) improve GRPO learning on hard MATH
problems that use it, compared with matched modular-arithmetic practice (E_unrelated)?
Primary contrast: seed-mean R − U endpoint pass@1 on target_test. Qwen3-1.7B @ `70d244cc`, non-thinking.

## Established (evidence in CHECKLIST)
- Environment `venv-v1`: verl 0.9.0 (reference only), vLLM 0.31.0, torch 2.13.0+cu130, transformers 5.10.4, math-verify 0.9.0. CPU tests 30/30 (job 11072715).
- Data: MATH @ `21a56338`, split manifest sha `6580712d…`; pass-1 annotation of 1,496 items; rubric v1.1 roles (D16).
- Response cap frozen at 4,096 tokens (D18): 2.83% truncation, no reward change at 8,192.
- Trainer (D10, D11): custom single-GPU GRPO, verl-parity of the loss reduction tested.
- Precision audit (A1) passed on job 11072622: restart weights and AdamW moments bit-exact after reload; repeated updates accumulate. Full table pending harvest.
- Checkpoints: 20.6 GB, NFS ~40 MB/s → background validated writer, project-wide write lock.

## Owner amendments in force
A1 precision audit + report figures (§15); A2 aux pool target 64 pairs, supplementation rules (§16);
A3 up to 30 CPUs on `preempt_cpu_qos` (§17); D21 `registry/` and `ops/` untracked (local only).
A4 H_train construction authority (§18): full native stage 2, core 2–8/32 + moderate 9–12/32, proceed at 48–64, external/authored/smaller-pilot paths owned by the agent.
GPU ceilings unchanged: 2 general, 6 preempt.

## Uncertain
- H_train: stage 1 projected ~25–35 core-band items from 154 candidates. Under A4 the moderate band
  (9–12/32) also counts; full native stage 2 over all 184 candidates decides the count.
- E_related: 110 candidates; may fall short of 64 pairs after band + matching (A2 options apply).
- Round time and memory of a real 128-response round (smoke run A in progress).

## Next actions
1. Stage-1 results (job 11072609) → shortlist → stage-2 (32 samples) on GPU.
2. Smoke run A/B (job 11072622) → G2 decision; profile round time; cost estimate.
3. Calibration screening (job 11072610, preempt): move to general if not started by 03:52Z.
4. Pass-2 blind annotation on shortlisted items (CPU-free; agents, isolated directories).
