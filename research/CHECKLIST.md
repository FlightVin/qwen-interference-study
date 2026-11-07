# CHECKLIST — execution plan

Purpose: the live plan from setup to the final report. Status: **done** / **active** / **blocked** /
**pending**. Evidence links follow each done item. Gates (G1–G6) must pass before later stages start.

## 0. Owner instructions (NON-NEGOTIABLE)
- 0.1 Ceilings: ≤2 general GPUs, ≤6 preempt GPUs, all project activity. One GPU per job. Enforced by `ops/qis-submit`.
- 0.2 Control only ledger-owned jobs by exact job ID (`ae slurm`). Never touch other projects' jobs/files/envs/W&B.
- 0.3 No GPU standbys reserved speculatively; no plain `ae pool`.
- 0.4 No treatment change after freeze because of interim curves; corrections apply to all affected conditions.
- 0.5 Record consequential decisions (evidence, action, runs, interpretation) in `research/DECISIONS.md`.
- 0.6 Keep failed/superseded attempts in the ledger and registry; charge their compute.
- 0.8 Addendum A2 (2026-11-07, `project.md` §16): 64 aux pairs is a target; modest shortfall allowed with documented coverage/diversity/balance; synthetic supplementation only under a recorded gap + finite budget, same audit/screen/matching, provenance balanced R vs U, template families tracked, eval sets free of training-template variants; choose smaller natural / supplemented / fallback and freeze before any treatment outcome.
- 0.7 Addendum A1 (2026-11-07, `project.md` §15): FP32 trainable weights + FP32 AdamW moments, BF16 rollout and autocast compute; one precision audit per trainer configuration; report figures (main, learning/exposure, POPE-style J trajectories, local update effects).

## 1. Setup and isolation — done except 1.6
- 1.1 done — Harness workspace, unique names, `config.sh` (DECISIONS D1–D7).
- 1.2 done — Self-tests pass: `qis-smoke` pin+CPU QOS, `qis-smoke-dep` chain, `qis-smoke-gen` general GPU (RTX 6000 Ada) + `/data/user_data` (harness `results/qis-smoke*/`).
- 1.3 done — Limits verified (harness `cluster/CARD.md`, 2026-11-07).
- 1.4 done — Hourly review: session cron at :47 local, `ops/qis-review` record/lock/overdue detection; one-shot test trigger received 2026-11-07T02:30:02Z (`research/REVIEWS.md`).
- 1.5 done — Event watcher: `ae-watch` background task, 120 s poll.
- 1.6 done — Environment `venv-v1`: verl 0.9.0, vLLM 0.31.0, torch 2.13.0+cu130, transformers 5.10.4, math-verify 0.9.0; model `Qwen/Qwen3-1.7B` @ `70d244cc` (`qis-env-v1`, job 11072074; harness `results/qis-env-v1/`).
- 1.7 pending — W&B credentials check on Babel (no secret printed); fallback: local JSONL + offline W&B.

## 2. Reading — done except 2.3
- 2.1 done — `research/READING.md` (history PDF + primary sources).
- 2.2 done — verl 0.9.0 audit → `research/VERL-AUDIT.md`; decision D10 (lean custom trainer, verl as reference).
- 2.3 pending — Pin Math-Verify commit and model revision in PROTOCOL.

## 3. Data (gate G1: dataset minima)
- 3.1 done — Split manifest `splits-v1` (sha256 `6580712d…`): 12,500 items, 344 duplicate pairs, 87 train items excluded as test overlaps; construction 4,458 / study_dev 1,478 / calib_train 750 / calib_val 727 / test 5,000. Gold: 12,495 eligible, 2 no box, 3 verifier-unknown. 11/11 tests passed (`qis-data-v1`, job 11072095).
- 3.2 done — Rubric v1 `research/RUBRIC.md`.
- 3.3 done — Annotation pass 1: 1,496 items → `data/annotations/pass1.jsonl` (batches 2, 5 redone after contamination, D13). Rubric v1.1 roles (D16): construction H 154, E_related 110, E_unrelated 152; target_dev 63; target_test 209.
- 3.4 active — Length probe `qis-lenprobe-v1` (11072430): 64 hard-like construction items × 16 at 4096 and 8192.
- 3.5 pending — Screening stage 1 (16 samples) → shortlist; stage 2 (32 samples) → selection; stage 3 (16 fresh, report only).
- 3.6 pending — Blind annotation pass 2 on all selected items; resolve disagreements; exclude unresolved.
- 3.7 pending — Error audit: ≤64 incorrect screening attempts on H candidates.
- 3.8 pending — Match E_related/E_unrelated 1:1; balance table (|SMD| ≤ 0.25); mixed-group probability.
- 3.9 pending — target_dev (64) / target_test (128) by rubric only; calibration panels (≥16 per band per split).
- 3.10 pending — (A1) Fixed probe panels, IDs frozen before any probe result: J_H = target_dev subset; J_R = study_dev E_related-rubric items; J_U = study_dev modular items. Size chosen from cost before evaluation.
- 3.11 pending — (A2) Pool decision after stage-2 screening: count matched pairs, coverage (factoring forms), diversity (template families), balance (SMD). If short: record gap + finite synthetic budget, OR retain smaller natural set, OR fallback skill — choose and record before freeze.
- **G1** — ≥64 H_train, ≥32 dev, ≥64 test, ≥64 per calib split (hard minima). Aux pools: 64 matched pairs target (A2); a modest shortfall needs documented coverage/diversity/balance. Else supplementation or the linear-equation fallback, before any treatment run.

## 4. Trainer (gate G2: correctness tests of project.md §8)
- 4.1 done — Custom single-GPU trainer, verl as reference (D10, D11). Code: `src/qis/train/`.
- 4.2 done — GPU probe 11072268 (RTX 6000 Ada): template fixture ok; 1,721 tok/s at 32 seqs; step 13.3 s/32 responses; logprob |Δ| p99 0.127 (BF16 head); ckpt 20.6 GB, write 512 s → background writer added.
- 4.3 done (CPU part) — 29/29 (`qis-env-v1b`): advantages, clip cases, accumulation, verl parity, minibatch balance, schedules, verifier, splits. GPU part in 4.4–4.7. Unit tests: template/mask/EOS, group boundaries, zero advantages, population std, loss reduction + accumulation, all-param trainability, 4 steps/round, balanced minibatches, signed objective.
- 4.4 pending — Rollout vs teacher-forced logprob agreement; update changes the rollout model.
- 4.5 active — (`qis-trainer-smoke-v1` 11072428, with 4.4 and 4.7) Interruption + resume test vs uninterrupted short run; repeated-seed variation.
- 4.6 pending — Profile a real round; checkpoint size/write time; telemetry overhead <3%.
- 4.7 pending — (A1) Precision audit after the first optimizer step: checkpoint/rollout/compute/param/grad/moment dtypes per parameter; nonzero grad norm on a mixed batch; sampled-weight deltas; repeated updates accumulate; sync at BF16; restart restores FP32 weights and moments bit-exactly; peak memory; table + code revision.
- **G2** — all tests pass on the pinned backend.

## 5. Calibration and freeze (gate G3)
- 5.1 pending — 6 configs (lr × clip), seed 11, 32 rounds; validation at 0/16/32.
- 5.2 pending — Top 2 with seed 22; select by rule; extension to 64 rounds only if the gain gate fails.
- 5.3 pending — Cost estimate → 128 or 64 rounds. Write and commit `research/PROTOCOL.md` (freeze).

## 6. Main runs (after G3)
- 6.1 pending — H/R/U × seeds 101, 202 on 6 preempt slots; seed 303 on general or released slots.
- 6.2 pending — Clipping sensitivity: R/U × 3 seeds, other clipping.
- 6.3 pending — Local update diagnostic (R seed 101, rounds 0 and halfway; 4 branches × 4 seeds).

## 7. Evaluation, audit, report
- 7.1 pending — Dev curves; test at baseline and endpoint (32 samples/problem).
- 7.2 pending — Analysis: primary R−U, t interval, crossed bootstrap (seed 2027), pass@8.
- 7.3 pending — Independent recomputation entry point; verifier sample audit.
- 7.4 pending — Cost reconciliation; `reports/REPORT.md` with figures in `reports/figures/` (A1.2), plotting script and source tables committed, verified from a clean checkout.
