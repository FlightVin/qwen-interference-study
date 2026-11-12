# CHECKLIST — A26 audit and report rewrite

Purpose: the live plan for owner task A26 (`project.md` §40). Status: **done** / **active** / **blocked** / **pending**.
Previous checklist (study execution, all closed): git 045c528. Deliverables live in `research/audit-a26/` unless named.

## 0. Owner instructions (NON-NEGOTIABLE)
- 0.1 A26: ≤ 8 general GPUs, ≤ 6 preempt GPUs, ≤ 30 CPUs; one GPU per job; guard `ops/qis-submit` (set to 8).
- 0.2 No new training study. Focused re-evaluation and evaluations that explain curious results are allowed.
- 0.3 Control only ledger-owned jobs by exact ID via `ae slurm`. Leave other projects' jobs and `ae-dco-*` alone.
- 0.4 Preserve history: report v2 archived (done); mark superseded definitions; never rewrite old records.
- 0.5 Record consequential corrections in `research/DECISIONS.md`; state how each changes a conclusion.
- 0.6 Pass B is independent: separate context, own code (no production aggregation/bootstrap/plot imports), results
  written before it sees pass A numbers.
- 0.7 No project code on the laptop or login node. Times in EST. Owner writing standard. No shareable web page.
- 0.8 Remove the opening synopsis, "What we learned" and "Four practical answers"; follow the A26 §5 order.

## 1. Definitions — `research/DEFINITIONS.md` (v1)                                   [active]
- 1.1 Arms and pools: H, H_train (46), 8 shared hard slots, extra slots, R/U, 63 matched pairs, dev, J_H/J_R/J_U.
- 1.2 Seeds: training, minibatch, evaluation sampling; checkpoint round; run attempt; shared vs different across arms.
- 1.3 Metrics: pass@1, pass@8 (estimator from 32), observed coverage at n, accuracy on finished, truncation.
- 1.4 GRPO units: group, mixed group, nonzero advantage, optimizer update, momentum-only step.
- 1.5 Each term: intended vs executed (code path + data), differences, superseded definitions marked.

## 2. Pass A — trace inputs to outputs                                                 [pending]
- 2.1 Code-path inventory: every job → commit, recipe hash, env, inputs (ledger 260 rows).
- 2.2 Data inventory: every set (train, calib, dev, probes, sealed test, cue, stress, audit): source, split, counts,
  families, natural/authored, rules, labels, screening, pairing, exclusions, intended and unintended overlaps, hashes in jobs.
- 2.3 Schedules from logs: shared hard prompts by seed and round, H extra prompts, R/U pairing.
- 2.4 Training/eval code: template, sampling, reward, truncation, advantages, clip, loss, optimizer, precision, masks,
  seeds, checkpoint cadence, cap switch at round 33, resumes, repeated rounds, counters, superseded attempts.
- 2.5 Eval shards/merges: problem and attempt IDs; length extension (prefix, eligibility, terminal, regrade, reproducibility).
- 2.6 Checkpoint inventory: all retained saves; fresh hash vs records; load; tensors; identity; lineage; resumable state;
  restore-and-step per resume path; H-s202 r64 (original vs resumed), r116 fallback, r128 endpoint.
- 2.7 Published evaluation → checkpoint, panel, hardware, attempt set.
- 2.8 Tests and regression checks; traced cases (ordinary, capped, multi-box, resumed, all-correct group, momentum-only
  step, sharded endpoint).

## 3. Pass B — independent reproduction                                                [pending]
- 3.1 Brief: definitions, raw records, manifests, checkpoints, protocol; no report numbers, no pass A outputs.
- 3.2 Reconstruct endpoints, base, common-GPU reruns, curves, probes, branches, coverage, stress tiers, length, costs.
- 3.3 Check pairing, missing attempts, weighting, interval units, signs, bootstrap pairing; same traced cases.

## 4. Discrepancies                                                                    [pending]
- 4.1 Compare A vs B after both are written; resolve from primary records; fix at source; rerun downstream.
- 4.2 `research/audit-a26/discrepancies.md`; DECISIONS entries for consequential corrections.
- 4.3 The seven acceptance tests (STATE "Known report defects") resolved with evidence.

## 5. Curious results — explanatory evaluations                                         [pending]
- 5.1 Why J_U (and J_R) rise in every arm; design the smallest evaluation that discriminates explanations.

## 6. Report rewrite (A26 §5 order)                                                    [pending]
- 6.1 Task → data and measurements → questions → procedure → results → interpretation → appendices.

## 7. Figures and render (A26 §6)                                                      [pending]
- 7.1 Regenerate from verified tables; consistent H/R/U colours, seed line styles.
- 7.2 Render from a clean checkout; inspect every figure at display width and at a narrow viewport.

## 8. Delivery                                                                         [pending]
- 8.1 Report, figures, tables, definitions, inventories, both audit records, discrepancy log; short handoff.
