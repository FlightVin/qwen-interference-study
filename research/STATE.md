# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-07 09:10Z (review 7).

## Phase
Dataset construction (H, auxiliary) and optimizer calibration, in parallel. Gate G2 passed. No treatment run yet.

## Question
Does easier practice on polynomial factorization (E_related) improve GRPO learning on hard MATH
problems that use it, compared with matched modular-arithmetic practice (E_unrelated)?
Primary contrast: seed-mean R − U endpoint pass@1 on target_test. Qwen3-1.7B @ `70d244cc`, non-thinking.

## Established (evidence in CHECKLIST)
- Environment `venv-v1`: verl 0.9.0 (reference only), vLLM 0.31.0, torch 2.13.0+cu130, transformers 5.10.4, math-verify 0.9.0. CPU tests 30/30 (job 11072715).
- Data: MATH @ `21a56338`, split manifest sha `6580712d…`; pass-1 annotation of 1,496 items; rubric v1.1 roles (D16).
- Response cap raised to 8,192 (D45): at 4,096, calibration training pushed validation truncation to 11.5% in 32 rounds (genuine long reasoning, not loops). 4,096 screening/calibration superseded; re-screen and recalibrate at 8,192.
- Trainer (D10, D11): custom single-GPU GRPO, verl-parity of the loss reduction tested.
- Precision audit (A1) passed on job 11072622: restart weights and AdamW moments bit-exact after reload; repeated updates accumulate. Full table pending harvest.
- Checkpoints: 20.6 GB, NFS ~40 MB/s → background validated writer, project-wide write lock.

## Owner amendments in force
A1 precision audit + report figures (§15); A2 aux pool target 64 pairs, supplementation rules (§16);
A3 up to 30 CPUs on `preempt_cpu_qos` (§17); D21 `registry/` and `ops/` untracked (local only).
A4 H_train construction authority (§18): full native stage 2, core 2–8/32 + moderate 9–12/32, proceed at 48–64, external/authored/smaller-pilot paths owned by the agent.
A5 technique coverage (§19): techniques ≥ 10% of H_train need ≥ ½ that share in E_related (primary tags F1–F8).
GPU ceilings unchanged: 2 general, 6 preempt.

## Dataset status — G1 passed (cap 8,192)
- H_train 46 (31 core / 15 moderate; native 25 + Omni 21; F1 11, F2 7, F3 11, F4 10, F8 7) — D47.
- E_related / E_unrelated 63 matched pairs (9 natural + 54 authored each; |SMD| ≤ 0.19; A5 coverage met) — D48.
- Targets: dev 53, test 128 (sealed file), probes J_H 32 / J_R 28 / J_U 32 — D34. Bundle `data/main_v1_manifest.json`.
- Stage-3 independent baseline queued (11075046).

## Calibration
- 4,096 run lr1e-6/sym superseded (truncation 11.5%). 8,192 panels frozen (D46); canary passed (D49); lr1e-6/sym and lr5e-7/sym running on general; four runs waiting for slots. Cross-job eval noise ≈ 0.01 (D49).

## Uncertain
- Final H_train size and Omni share; whether R/U reach 64 matched pairs with A5 coverage.
- Preempt availability (D37): main matrix may run on 2 general GPUs (~50 h wall).

## Next actions
1. 8,192 re-screen results → H selection, R/U matching + A5 coverage → stage-3 baseline → freeze dataset.
2. Calibration at 8,192: canary first round check, then all six, then seed-22 finalists → select → freeze PROTOCOL.
3. Cost estimate at 8,192 → endpoint (128 vs 64 rounds) → launch main matrix.
