# STATE

Purpose: where the study stands now. Scientific authority; the harness `state/STATE.md` only points here.
Updated 2026-11-07 05:10Z (review 3).

## Phase
Dataset construction (H, auxiliary) and optimizer calibration, in parallel. Gate G2 passed. No treatment run yet.

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
A5 technique coverage (§19): techniques ≥ 10% of H_train need ≥ ½ that share in E_related (primary tags F1–F8).
GPU ceilings unchanged: 2 general, 6 preempt.

## Dataset status (pre-freeze)
- H_train: native 29 eligible (18 core / 11 moderate); Omni-MATH tranche 73 annotation-eligible, screening pending (job 11073332). Combined rule fixed (D38).
- Auxiliary: natural R 7 / U 18 in band; authored tranche (198 R / 196 U after intake; 114 R / 196 U annotation-eligible) screening (job 11073304). Matching + A5 coverage code fixed (D39).
- Targets frozen (D34): target_dev 53, target_test 128, probes J_H 32 / J_R 28 / J_U 32.
- Calibration panels frozen (D33); calibration run 1 of 6 at round 4 (baseline calib_val 0.4265).

## Uncertain
- Final H_train size and Omni share; whether R/U reach 64 matched pairs with A5 coverage.
- Preempt availability (D37): main matrix may run on 2 general GPUs (~50 h wall).

## Next actions
1. Omni screening → combined H selection → independent baseline (stage 3) → mixed-group fraction.
2. Authored screening → R/U matching, balance, coverage → freeze dataset + PROTOCOL.
3. Calibration: remaining 5 runs on general as slots free; then seed-22 finalists; select config.
