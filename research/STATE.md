# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-07 19:30Z (calibration selected; protocol frozen; A6 recorded).

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
A6 thinking-mode baseline diagnostic (§20; D56).
A7 GPU ceilings 3 general + 6 preempt (§21; D60).
A1 precision audit + report figures (§15); A2 aux pool target 64 pairs, supplementation rules (§16);
A3 up to 30 CPUs on `preempt_cpu_qos` (§17); D21 `registry/` and `ops/` untracked (local only).
A4 H_train construction authority (§18): full native stage 2, core 2–8/32 + moderate 9–12/32, proceed at 48–64, external/authored/smaller-pilot paths owned by the agent.
A5 technique coverage (§19): techniques ≥ 10% of H_train need ≥ ½ that share in E_related (primary tags F1–F8).
GPU ceilings (A7, §21, supersedes 2 general): **3 general, 6 preempt**.

## Dataset status — G1 passed (cap 8,192)
- H_train 46 (31 core / 15 moderate; native 25 + Omni 21; F1 11, F2 7, F3 11, F4 10, F8 7) — D47.
- E_related / E_unrelated 63 matched pairs (9 natural + 54 authored each; |SMD| ≤ 0.19; A5 coverage met) — D48.
- Targets: dev 53, test 128 (sealed file), probes J_H 32 / J_R 28 / J_U 32 — D34. Bundle `data/main_v1_manifest.json`.
- Stage-3 baseline (D52): H mean p 0.202, mixed-group fraction 0.77; R 0.589 / U 0.569.

## Calibration and freeze (G3)
- Selected **lr 2e-6, clip (0.2, 0.2)** by the frozen §7 rule (D57): mean gain 0.159 vs 0.106 (lr 1e-6/sym). lr2e-6/asym rejected (truncation 7.14%, D54).
- Endpoint T=128 (≤ 21 GPU-h per run). PROTOCOL frozen. Risk: lr 2e-6 lengthens responses; truncation per arm monitored.
- Eval noise: sd 0.020 per within-run gain; cross-job reproduction partial (D50, D51).

## A6 thinking-mode diagnostic (D56)
- Same-cap (8,192) result (D62): thinking raises pass@1 by +0.23–0.25 in every pool, but truncates 51.6% of H and 46% of dev responses; auxiliary practice becomes near-trivial (0.97–0.98; mixed groups 13–25%); R/U stay matched; 5.5–8.1× GPU time. 16,384 profile queued (11078421). Failure inspection in progress.

## Uncertain
- Final H_train size and Omni share; whether R/U reach 64 matched pairs with A5 coverage.
- Preempt availability (D37): main matrix may run on 2 general GPUs (~50 h wall).

## Next actions
1. 8,192 re-screen results → H selection, R/U matching + A5 coverage → stage-3 baseline → freeze dataset.
2. Calibration at 8,192: canary first round check, then all six, then seed-22 finalists → select → freeze PROTOCOL.
3. Cost estimate at 8,192 → endpoint (128 vs 64 rounds) → launch main matrix.
