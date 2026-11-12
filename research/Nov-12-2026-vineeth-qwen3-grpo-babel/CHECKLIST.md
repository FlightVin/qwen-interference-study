> **FROZEN on 2026-11-12 (owner instruction A30). Not updated after this date. Historical record only.**
> Paths below are pre-move: `reports/` → `reports/Nov-12-2026-initial-experiments/`, `data/` →
> `data/Nov-12-2026-qwen3-grpo-math-practice/`, `research/` → `research/Nov-12-2026-vineeth-qwen3-grpo-babel/`.

# CHECKLIST — A26 audit and report rewrite

Purpose: the live plan for owner task A26 (`project.md` §40). Status: **done** / **active** / **blocked** / **pending**.
Previous checklist (study execution, all closed): git 045c528. Deliverables live in `research/audit-a26/` unless named.

## 0. Owner instructions (NON-NEGOTIABLE)
- 0.9 A27 (`project.md` §41): deliver report v4 by Thu 12 Nov 12:00 PM EST; ≤ 8 general GPUs concurrently (guard
  `ops/qis-submit` = 8, annotated); no new training; title names auxiliary math practice during GRPO of Qwen3-1.7B.
- 0.1 A26: ≤ 8 general GPUs, ≤ 6 preempt GPUs, ≤ 30 CPUs; one GPU per job; guard `ops/qis-submit` (set to 8).
- 0.2 No new training study. Focused re-evaluation and evaluations that explain curious results are allowed.
- 0.3 Control only ledger-owned jobs by exact ID via `ae slurm`. Leave other projects' jobs and `ae-dco-*` alone.
- 0.4 Preserve history: report v2 archived (done); mark superseded definitions; never rewrite old records.
- 0.5 Record consequential corrections in `research/DECISIONS.md`; state how each changes a conclusion.
- 0.6 Pass B is independent: separate context, own code (no production aggregation/bootstrap/plot imports), results
  written before it sees pass A numbers.
- 0.7 No project code on the laptop or login node. Times in EST. Owner writing standard. No shareable web page.
- 0.8 Remove the opening synopsis, "What we learned" and "Four practical answers"; follow the A26 §5 order.

## 1. Definitions — `research/DEFINITIONS.md` (v1)                                   [done 26cd48b]
- 1.0 v1 frozen 10:00 PM EST after orchestrator spot checks; 17 differences listed in its §4; 9 open compute items in §5.
- 1.1 Arms and pools: H, H_train (46), 8 shared hard slots, extra slots, R/U, 63 matched pairs, dev, J_H/J_R/J_U.
- 1.2 Seeds: training, minibatch, evaluation sampling; checkpoint round; run attempt; shared vs different across arms.
- 1.3 Metrics: pass@1, pass@8 (estimator from 32), observed coverage at n, accuracy on finished, truncation.
- 1.4 GRPO units: group, mixed group, nonzero advantage, optimizer update, momentum-only step.
- 1.5 Each term: intended vs executed (code path + data), differences, superseded definitions marked.

## 2. Pass A — trace inputs to outputs                                                 [done]
- Done 11:30 PM EST (db8c2ed): a-data (schedules pass in 6 runs; 2 held-out close variants), a-code (54/54 tests;
  265 jobs traced; no defect changes a count), a-evals (236 claims: 223 verified, 11 differ, 24 failing equations).
- 2.6 a-ckpt: inventory from `qis-a26-ckpt-k0..3` (all 4 done) + round-65 replay → done 6282ba0 (`pass-a/checkpoints.md`,
  `replay.md`); the "active" mark was stale (found in A27).
- Agents (10:10 PM EST): a-data (2.2, 2.3) → `pass-a/data-inventory.md`, `schedules.md`; a-code (2.1, 2.4, 2.5 code, 2.8 tests)
  → `pass-a/jobs.md`, `code.md`; a-evals (2.5 shards, 2.7, 2.8 traces, report-v2 numbers) → `eval-map.md`, `traces.md`,
  `numbers.md`; a-ckpt (2.6) starts after `qis-a26-ckpt-k0..3` finish.
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

## 3. Pass B — independent reproduction                                                [done db8c2ed]
- Agents (10:15 PM EST): b-eval (endpoints, contrasts, coverage, stress, length, cue, audit counts) and b-train (curves,
  probes, ordering, group statistics, ΔJ branches, loss/recovery, costs). Forbidden: reports/, evidence, audits, pass-a,
  STATE/CHECKLIST/DECISIONS/REVIEWS, project.md, src/qis/analysis, aggregate files. Own code in `pass-b/tools/`.
- 3.1 Brief: definitions, raw records, manifests, checkpoints, protocol; no report numbers, no pass A outputs.
- 3.2 Reconstruct endpoints, base, common-GPU reruns, curves, probes, branches, coverage, stress tiers, length, costs.
- 3.3 Check pairing, missing attempts, weighting, interval units, signs, bootstrap pairing; same traced cases.

## 4. Discrepancies                                                                    [done 899b4a9]
- 161 scripted comparisons, 157 agree, 4 resolved (`comparison.md`); 38 defects: 11 interpretation, 8 number, 19 wording
  (`discrepancies.md`); D162 lists the changed conclusions. DEFINITIONS v1.1 being applied (agent).
- 4.0 Agent reconcile: `comparison.md` (A vs B), sensitivity (test #30 variant, multi-box, unknowns),
  `discrepancies.md`, `definitions-v1.1-changes.md`.
- 4.1 Compare A vs B after both are written; resolve from primary records; fix at source; rerun downstream.
- 4.2 `research/audit-a26/discrepancies.md`; DECISIONS entries for consequential corrections.
- 4.3 The seven acceptance tests (STATE "Known report defects") resolved with evidence.

## 5. Curious results — explanatory evaluations                                         [done 49196bc, D163]
- 5.0 Agent x-transfer: failure taxonomy of J_U/J_R/J_H attempts (r0 vs r128); control panel `control_v1` (64 MATH test
  items, geometry / counting / precalculus / prealgebra, L4–5, hash-ordered, unused) × base + 6 endpoints × 8 attempts;
  pre-registered in `research/audit-a26/transfer/PREREG.md` before results (D161).
- 5.1 Why J_U (and J_R) rise in every arm; design the smallest evaluation that discriminates explanations.

## 6. Report rewrite (A26 §5 order)                                                    [done 5b211c7]
- 6.1 Reviewed by the orchestrator against A26 §4 (7 acceptance tests, each resolved in §5.1, §5.2, §5.5, §5.4.4–5,
  §5.7, rounding note) and §5 order; self-check: 866/866 numbers traced, 0 equations, 13/13 figures, 0 v2 error phrases.
  Filled TODOs: R-s202 tokens from records (54.1 M); A26 job cost (19 jobs, 5.20 GPU-h, 41.65 CPU-h).
- 6.0 Outline `research/audit-a26/report-v3-outline.md` (b8a0a99); agent writer drafting `reports/Nov-12-2026-initial-experiments/Nov-12-2026-experiments-report.md` v3 with a
  scripted self-check; orchestrator review against A26 §4–§5 next.
- 6.1 Task → data and measurements → questions → procedure → results → interpretation → appendices.

## 7. Figures and render (A26 §6)                                                      [done 1e75db1]
- 7.3 Render 1 (5b211c7): all pass at 1024 px; 420 px text 7.5–8 px → redesign (6 in, 13 pt, stacked, f07 split,
  page-order numbering). Render 2 (clean clone 1e75db1): 14/14 pass at 1024 and 420 px (text 10.4–12.2 px at 420),
  47/47 value spot checks, cross-references correct (`reports/figures/v3/render-inspection.md`).
- 7.2 Agent render: clean clone → GitHub markdown API → headless Chrome at 1024 px and 420 px per figure →
  `reports/figures/v3/render-inspection.md`; fixes then rebuild and re-render.
- 7.0 Agent figures: `reports/build/build_v3.py` → `reports/tables/v3/`, `reports/figures/v3/` (F1–F12), captions,
  previews at 830 px and 400 px, value check against pass A/B.
- 7.1 Regenerate from verified tables; consistent H/R/U colours, seed line styles.
- 7.2 Render from a clean checkout; inspect every figure at display width and at a narrow viewport.

## 8. Delivery                                                                         [done]
- Handoff sent 2026-11-12. Standbys cancelled; compute worktree of a-code removed.
- 8.1 Report, figures, tables, definitions, inventories, both audit records, discrepancy log; short handoff.

## 9. A27 report v4                                                        [done; final]
- 9.1 Inherited state verified: v3 audit, figures and checkpoint inventory complete; v3 report and figures archived.
- 9.2 Live usage checked (0 jobs). D166: H-s202 r128 sealed test on L40S, 8 shards → 8/8 completed; not analysed (owner: report final, stop jobs).
- 9.3 New analysis: per-pool groups without reward contrast by quarter (`tables/v4/t10_signal`; totals = v3 table, 6/6).
- 9.4 Report v4 edits (Appendix J lists them); ray interference cited from Schaul et al. 2019 and POPE §3.2 / Fig. 5.
- 9.5 Figures 1 and 9 inspected at 830 px and 372 px (`reports/figures/v4/render-inspection.md`); others unchanged from v3.
- 9.6 Not done in A27 (deadline): full clean-clone re-render of every figure; a fresh independent pass B of v4 text.
