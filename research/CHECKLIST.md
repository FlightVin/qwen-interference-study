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

## 1. Setup and isolation — done except 1.6
- 1.1 done — Harness workspace, unique names, `config.sh` (DECISIONS D1–D7).
- 1.2 done — Self-tests pass: `qis-smoke` pin+CPU QOS, `qis-smoke-dep` chain, `qis-smoke-gen` general GPU (RTX 6000 Ada) + `/data/user_data` (harness `results/qis-smoke*/`).
- 1.3 done — Limits verified (harness `cluster/CARD.md`, 2026-11-07).
- 1.4 done — Hourly review: session cron at :47 local, `ops/qis-review` record/lock/overdue detection; one-shot test trigger received 2026-11-07T02:30:02Z (`research/REVIEWS.md`).
- 1.5 done — Event watcher: `ae-watch` background task, 120 s poll.
- 1.6 active — Environment `venv-v1` (verl 0.9.0, vLLM, math-verify 0.9.0), model download (`qis-env-v1`, job 11072074).
- 1.7 pending — W&B credentials check on Babel (no secret printed); fallback: local JSONL + offline W&B.

## 2. Reading — done except 2.3
- 2.1 done — `research/READING.md` (history PDF + primary sources).
- 2.2 active — verl 0.9.0 semantics audit → `research/VERL-AUDIT.md`.
- 2.3 pending — Pin Math-Verify commit and model revision in PROTOCOL.

## 3. Data (gate G1: dataset minima)
- 3.1 active — Split manifest + gold eligibility + annotation exports (`qis-data-v1`, after 1.6). Tests: `tests/test_verify.py`, `tests/test_splits.py`.
- 3.2 pending — Skill rubric v1 (`research/RUBRIC.md`): central / incidental / absent; role pools fixed BEFORE screening: H/target = central + level 4–5; E_related = central + direct practice + level 1–3; E_unrelated = number-theory modular arithmetic, factorization absent.
- 3.3 pending — Annotation pass 1 (agent) over candidate exports; store all fields of project.md §5.
- 3.4 pending — Length probe (needs 4.2): cap-hit rate on construction items; 8k test if >5%.
- 3.5 pending — Screening stage 1 (16 samples) → shortlist; stage 2 (32 samples) → selection; stage 3 (16 fresh, report only).
- 3.6 pending — Blind annotation pass 2 on all selected items; resolve disagreements; exclude unresolved.
- 3.7 pending — Error audit: ≤64 incorrect screening attempts on H candidates.
- 3.8 pending — Match E_related/E_unrelated 1:1; balance table (|SMD| ≤ 0.25); mixed-group probability.
- 3.9 pending — target_dev (64) / target_test (128) by rubric only; calibration panels (≥16 per band per split).
- **G1** — ≥64 H_train, ≥64 per aux pool, ≥32 dev, ≥64 test, ≥64 per calib split. Else the linear-equation fallback (before any treatment run).

## 4. Trainer (gate G2: correctness tests of project.md §8)
- 4.1 pending — Decide verl configuration vs custom pieces from VERL-AUDIT (advantage with population std, loss reduction, minibatch composition 2H+2A, rollout-logprob check, sampler).
- 4.2 pending — GPU probe on general: non-thinking template fixture, generation, one full optimizer step, peak memory.
- 4.3 pending — Unit tests: template/mask/EOS, group boundaries, zero advantages, population std, loss reduction + accumulation, all-param trainability, 4 steps/round, balanced minibatches, signed objective.
- 4.4 pending — Rollout vs teacher-forced logprob agreement; update changes the rollout model.
- 4.5 pending — Interruption + resume test vs uninterrupted short run; repeated-seed variation.
- 4.6 pending — Profile a real round; checkpoint size/write time; telemetry overhead <3%.
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
- 7.4 pending — Cost reconciliation; `reports/REPORT.md`.
