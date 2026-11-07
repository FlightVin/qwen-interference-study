# CHECKLIST — execution plan

Purpose: the live plan from setup to the final report. Status: **done** / **active** / **blocked** /
**pending**. Evidence links follow each done item. Gates (G1–G6) must pass before later stages start.

## 0. Owner instructions (NON-NEGOTIABLE)
- 0.1 Ceilings: ≤3 general GPUs (A7, 2026-11-07, `project.md` §21; supersedes ≤2), ≤6 preempt GPUs, all project activity. One GPU per job. Enforced by `ops/qis-submit`.
- 0.2 Control only ledger-owned jobs by exact job ID (`ae slurm`). Never touch other projects' jobs/files/envs/W&B.
- 0.3 No GPU standbys reserved speculatively; no plain `ae pool`.
- 0.4 No treatment change after freeze because of interim curves; corrections apply to all affected conditions.
- 0.5 Record consequential decisions (evidence, action, runs, interpretation) in `research/DECISIONS.md`.
- 0.6 Keep failed/superseded attempts in the ledger and registry; charge their compute.
- 0.12 Amendment A6 (2026-11-07, `project.md` §20): thinking-mode baseline diagnostic on an 80-item seeded panel (32 H stratified by source × technique, 16 R + 16 matched U, 16 dev; sealed test untouched), 8 attempts per item per mode; same cap 8,192 first, only `enable_thinking` changes; reasoning + answer count against the cap; grade only the final answer after reasoning (tests); report per pool; 16,384 profile only if thinking truncation > 5%; separate thinking study only after re-screen, matching, calibration and a 32-round truncation check; distinct manifests/run IDs; main study and hourly review continue.
- 0.13 Amendment A7 (2026-11-07, `project.md` §21): 3 concurrent general + 6 preempt GPUs; reconcile with the ledger; fill the third general GPU with the highest-priority ready task; move an eligible pending job if it starts earlier; verify single execution; keep the slot filled while useful work is ready.
- 0.11 Amendment A5 (2026-11-07, `project.md` §19): technique coverage — primary technique per H_train/E_related item (F1–F8, frozen); each technique ≥ 10% of H_train needs E_related share ≥ ½ of it (ceil to whole items); check after R/U matching; repair via authorized E_related selection/augmentation only, re-match; H_train fixed; residual shortfall recorded; table in report.
- 0.10 Amendment A4 (2026-11-07, `project.md` §18): H_train via full native stage 2 (184 candidates, ≤5,888 attempts), core 2–8/32 + moderate 9–12/32, target 64 (≥½ core), proceed at 48–64; else Omni-MATH tranche (≤200), smaller pilot with uniform endpoint shortening, or one authored tranche; one H_train manifest; decisions before treatment outcomes.
- 0.9 Amendment A3 (2026-11-07, `project.md` §17): ≤30 CPUs total on `preempt_cpu_qos`; enforced by `ops/qis-submit`.
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
- 3.4 done — Length probe 11072430: truncation 2.83% at 4096, 0% at 8192, success identical (0.6406) → cap 4096 frozen (D18). H reserve tranche pre-registered (D19).
- 3.5 done (superseded at 8,192 by 3.16) — stage 1 running: study pools 11072609 (general), calibration 11072610 (preempt). Screening stage 1 (16 samples) → shortlist; stage 2 (32 samples) → selection; stage 3 (16 fresh, report only).
- 3.6 done (D47/D48; pass 2 + third read) — Blind annotation pass 2 on all selected items; resolve disagreements; exclude unresolved.
- 3.7 pending — Error audit: ≤64 incorrect screening attempts on H candidates.
- 3.8 done (D48: 63 pairs, |SMD| ≤ 0.19; mixed-group fraction from stage-3 baseline 11075046) — Match E_related/E_unrelated 1:1; balance table (|SMD| ≤ 0.25); mixed-group probability.
- 3.9 done (D34, D46) — calibration panels done (D33). target_dev/target_test selection by rubric pending (pass 2 + third read done).
- 3.10 done (D34: J_H 32, J_R 28, J_U 32) — (A1) Fixed probe panels, IDs frozen before any probe result: J_H = target_dev subset; J_R = study_dev E_related-rubric items; J_U = study_dev modular items. Size chosen from cost before evaluation.
- 3.12 done (D47) — (A4.1) Stage 2 over all 184 hard + reserve candidates (32 attempts each, reuse s2 from 11072896); report counts by band (core/moderate) and source; record exclusions.
- 3.13 done (D47 selection; D52 stage-3 baseline: H mixed-group fraction 0.77, mean p 0.202) — (A4.2) Structure-family labels for hard candidates (blind); selection: 64 target, ≥½ core, diversity, seeded ties; independent baseline (stage 3) after selection; measured mixed-group fraction.
- 3.14 done (D47: Omni tranche, n=46 under A4.4) — (A4.3–5) Only if native < 48 or missing structures: Omni-MATH tranche (≤200, pinned) / smaller pilot (record families, core share, 16T/N and 8T/N repetitions) / authored tranche / fallback.
- 3.16 done (D46–D48) — (D45) Re-screen at 8,192: H candidates with 4k c ≤ 12/32 (native + Omni, audit-eligible), auxiliary candidates with 4k c ≤ 24/32 or stage-1 ≤ 14/16, calibration panels (400 × 16); then re-run selection, matching, coverage.
- 3.15 done (D48: coverage met, no repair) — (A5) Primary + secondary technique tags for every H_train and E_related item (blind annotators; F1–F8); coverage check after R/U matching; repair loop; coverage table → PROTOCOL and report.
- 3.11 done (D48: authored tranches, 63 pairs) — (A2) Pool decision after stage-2 screening: count matched pairs, coverage (factoring forms), diversity (template families), balance (SMD). If short: record gap + finite synthetic budget, OR retain smaller natural set, OR fallback skill — choose and record before freeze.
- **G1 — PASSED 2026-11-07 under A2/A4 (D47, D48):** H_train 46 (A4.4 record), 63 matched aux pairs with A5 coverage, target_dev 53, target_test 128, calibration panels 59/56 at the 8,192 cap. Bundle `data/main_v1_manifest.json` (bundle sha256 `0e255243…`, sealed test `2ffe70dd…`). Original text: ≥64 H_train, ≥32 dev, ≥64 test, ≥64 per calib split (hard minima). Aux pools: 64 matched pairs target (A2); a modest shortfall needs documented coverage/diversity/balance. Else supplementation or the linear-equation fallback, before any treatment run.

## 4. Trainer (gate G2: correctness tests of project.md §8)
- 4.1 done — Custom single-GPU trainer, verl as reference (D10, D11). Code: `src/qis/train/`.
- 4.2 done — GPU probe 11072268 (RTX 6000 Ada): template fixture ok; 1,721 tok/s at 32 seqs; step 13.3 s/32 responses; logprob |Δ| p99 0.127 (BF16 head); ckpt 20.6 GB, write 512 s → background writer added.
- 4.3 done (CPU part) — 30/30 (job 11072715, incl. exact checkpoint checksum): advantages, clip cases, accumulation, verl parity, minibatch balance, schedules, verifier, splits. GPU part in 4.4–4.7. Unit tests: template/mask/EOS, group boundaries, zero advantages, population std, loss reduction + accumulation, all-param trainability, 4 steps/round, balanced minibatches, signed objective.
- 4.4 done — mismatch median ~1e-7, p99 0.10–0.11, p99.9 0.21–0.23; alarm thresholds D25; update reaches rollout (probe: mean |Δlp| 0.0026 after one step).
- 4.5 done — resume test (11072622): no duplicated/missing rounds, counters equal; A/B diverge from round 2 (pre-interruption) = kernel nondeterminism tolerance (D24). Attempts 11072408 (cancelled), 11072428, 11072527 (failed, fixed).
- 4.6 partial — smoke rounds 42–51 s (short items), 36/44.4 GiB device memory, ckpt 21.9 GB in 197–245 s background (D26). Real-length round profile and telemetry overhead: first calibration run.
- 4.7 done — `research/evidence/g2/precision_audit.md` (D24). (A1) Precision audit after the first optimizer step: checkpoint/rollout/compute/param/grad/moment dtypes per parameter; nonzero grad norm on a mixed batch; sampled-weight deltas; repeated updates accumulate; sync at BF16; restart restores FP32 weights and moments bit-exactly; peak memory; table + code revision.
- **G2** — PASSED 2026-11-07 (D24).

## 5. Calibration and freeze (gate G3)
- 5.0 done — 4,096-cap calibration run lr1e-6/sym (11072995): gain +0.081 but validation truncation 11.5% → cap raised to 8,192 (D45); 4,096 runs superseded/cancelled.
- 5.1 active (8,192 cap; D46, D49) — 6 configs (lr × clip), seed 11, 32 rounds; calib_val eval (16/problem) at 0/16/32. Done: 11074527 (5e-7/sym, +0.031), 11074520 (1e-6/sym, +0.122), 11074526 (5e-7/asym, −0.003), 11074519 (1e-6/asym, +0.066). 11074522 (2e-6/sym, +0.146, trunc 4.69%). 11074521 (2e-6/asym, +0.196 but trunc 7.14% → rejected, D54). 4,096 jobs 11072991–11072999 superseded (D45).
- 5.2 done (D57: lr 2e-6 / sym selected; job 11078053) — selection by `qis-calib-select` CPU job (D55). Seed 22: 1e-6/sym running (11075978, D51); 2e-6/sym (11076551, D53) moved to general. Finalists fixed (D54): 2e-6/sym, 1e-6/sym. Top 2 with seed 22; select by rule; extension to 64 rounds only if the gain gate fails.
- 5.3 done (D57: T=128; PROTOCOL frozen) — cost estimate done (T=128 under both outcomes; recipe wall 40 h). PROTOCOL draft committed (all but calibration result, lr/clip, endpoint T, freeze line). Recipe generator `qis.make_main_exps`, evaluator `qis.evaluate_ckpt` ready. Remaining: cost estimate → 128 or 64 rounds; sign freeze.

## 6A. A6 thinking-mode diagnostic (owner amendment, project.md §20; D56)
- 6A.1 active — Panel code + grading tests committed before any output (`src/qis/mode_diag.py`, `tests/test_mode_diag.py`).
- 6A.2 pending — Same-cap run (8,192): both modes, 8 attempts, template check; report per pool: paired Δpass@1, mixed-group frequency, truncation, lengths, GPU cost; R vs U match; H occasional successes.
- 6A.3 pending — Fixed failure sample (≤ 16 per mode): is the focal factoring operation still implicated?
- 6A.4 conditional — If thinking truncation > 5% (H or either aux pool): 16,384 profile on the same panel; longer only if context/memory permit; reported separately.
- 6A.5 pending — Judgment: is a separate thinking-mode study warranted (would need its own re-screen, matching, calibration, 32-round truncation check, distinct manifests).

## 6. Main runs (after G3)
- 6.1 active (canary 11078086 running; D58) — H/R/U × seeds 101, 202 on 6 preempt slots; seed 303 on general or released slots.
- 6.2 pending — Clipping sensitivity: R/U × 3 seeds, other clipping.
- 6.3 pending — Local update diagnostic (R seed 101, rounds 0 and halfway; 4 branches × 4 seeds).

## 7. Evaluation, audit, report
- 7.1 pending — Dev curves; test at baseline and endpoint (32 samples/problem).
- 7.2 partial — `src/qis/analysis/stats.py` (t interval, crossed bootstrap seed 2027, per-problem change), tests 33/33 passed in job 11075445 (11075419 failed: scipy absent from venv; replaced by a verified t table, ~7 CPU-min charged); figures and pass@8 tables pending. Analysis: primary R−U, t interval, crossed bootstrap (seed 2027), pass@8.
- 7.3 pending — Independent recomputation entry point; verifier sample audit.
- 7.4 pending — Cost reconciliation; `reports/REPORT.md` with figures in `reports/figures/` (A1.2), plotting script and source tables committed, verified from a clean checkout.
