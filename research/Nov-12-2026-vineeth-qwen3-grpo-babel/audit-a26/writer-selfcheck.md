# Writer self-check for reports/REPORT.md (report v3)

Purpose: record the result of the writer's automatic checks on `reports/REPORT.md` (A26 writer task). Script:
`research/audit-a26/tools/writer/check_report.py` (stdlib only). Run: `python3 research/audit-a26/tools/writer/check_report.py . research/audit-a26/writer-selfcheck.md` from the repository root.

## 1. Summary

| Check | Result |
|---|---|
| Numbers with a decimal point or % traced to a source | 866 of 866 matched; 0 unmatched (83 section references such as 5.4.4 skipped) |
| `=` between two numbers | 0 found |
| Figure links that exist | 13 of 13 |
| PNG files in `reports/figures/v3/` embedded exactly once | 13 of 13 |
| Relative links whose target is missing | 0 |
| Report-v2 error phrases still present | 0 |
| Main text (sections 1–6) lines | 729 (target about 450–600) |
| Appendix lines / total lines | 292 / 1030 |

Limit of check 1: a match means the number occurs in, or rounds from, some value in the named sources; it does not
prove that the number belongs to the quantity the sentence names. Short numbers such as 4.2 match by coincidence
easily. The writer took every number from the source cited beside it; this check catches typos and invented
values, not misattribution.

## 2. Unmatched numbers

None.

## 3. Equations, figures, links, phrases

- `=` between numbers: none.
- Missing figure files: none.
- Figure use count: `f01_endpoints.png` 1, `f02_contrasts_test.png` 1, `f03_dev_rounds.png` 1, `f04_dev_hard_attempts.png` 1, `f05_traj_JH_JR.png` 1, `f06_traj_JH_JU.png` 1, `f07_delta_j.png` 1, `f08_length_pass1.png` 1, `f08b_length_capped.png` 1, `f09_length_contrasts.png` 1, `f10_stress_tiers.png` 1, `f11_stress_contrasts.png` 1, `f12_control.png` 1.
- Missing link targets: none.
- Phrases checked: "five finished trained policies", "between rounds 32 and 64", "does not rescue", "learning signal", "without updating the model", "What we learned", "Four practical answers", "3 points" (not preceded by a digit or decimal point). Hits: none.

## 4. Lines per section

| Section | Lines |
|---|---|
| 1. Task | 43 |
| 2. Data and measurements | 82 |
| 3. Research questions | 18 |
| 4. Experimental procedure | 76 |
| 5. Results | 475 |
| 6. Interpretation, limitations and next decision | 35 |
| Appendix A. Complete tables | 68 |
| Appendix B. Checkpoint inventory and lineage | 81 |
| Appendix C. Job and cost ledger | 23 |
| Appendix D. Hardware per evaluation | 17 |
| Appendix E. Hashes and commands | 40 |
| Appendix F. Audit record | 17 |
| Appendix G. Unavailable cells and what could not be verified | 22 |
| Appendix H. Changes from report version 2 | 24 |

## 5. Sources searched for numbers

69 files: `reports/tables/v3/t01_endpoints.md`, `reports/tables/v3/t02_contrasts_test.md`, `reports/tables/v3/t03_curves.md`, `reports/tables/v3/t04_diag.md`, `reports/tables/v3/t05_length.md`, `reports/tables/v3/t06_length_contrasts.md`, `reports/tables/v3/t07_stress.md`, `reports/tables/v3/t08_stress_contrasts.md`, `reports/tables/v3/t09_control.md`, `reports/tables/v3/t01_endpoints.json`, `reports/tables/v3/t02_contrasts_test.json`, `reports/tables/v3/t03_curves.json`, `reports/tables/v3/t04_diag.json`, `reports/tables/v3/t05_length.json`, `reports/tables/v3/t06_length_contrasts.json`, `reports/tables/v3/t07_stress.json`, `reports/tables/v3/t08_stress_contrasts.json`, `reports/tables/v3/t09_control.json`, `reports/figures/v3/captions.md`, `reports/figures/v3/inspection.md`, `research/DEFINITIONS.md`, `research/PROTOCOL.md`, `research/audit-a26/discrepancies.md`, `research/audit-a26/comparison.md`, `research/audit-a26/pass-a/checkpoints.md`, `research/audit-a26/pass-a/code.md`, `research/audit-a26/pass-a/data-inventory.md`, `research/audit-a26/pass-a/eval-map.md`, `research/audit-a26/pass-a/jobs.md`, `research/audit-a26/pass-a/numbers.md`, `research/audit-a26/pass-a/replay.md`, `research/audit-a26/pass-a/schedules.md`, `research/audit-a26/pass-a/traces.md`, `research/audit-a26/pass-a/data/numbers.json`, `research/audit-a26/pass-a/data/checkpoints.json`, `research/audit-a26/pass-b/eval.md`, `research/audit-a26/pass-b/train.md`, `research/audit-a26/pass-b/data/eval_audit.json`, `research/audit-a26/pass-b/data/eval_cue.json`, `research/audit-a26/pass-b/data/eval_length.json`, `research/audit-a26/pass-b/data/eval_stress.json`, `research/audit-a26/pass-b/data/eval_test.json`, `research/audit-a26/pass-b/data/eval_trace.json`, `research/audit-a26/pass-b/data/train_costs.json`, `research/audit-a26/pass-b/data/train_curves.json`, `research/audit-a26/pass-b/data/train_diag.json`, `research/audit-a26/pass-b/data/train_groups.json`, `research/audit-a26/pass-b/data/train_loss_recovery.json`, `research/audit-a26/pass-b/data/train_ordering.json`, `research/audit-a26/pass-b/data/train_rewards.json`, `research/audit-a26/pass-b/data/train_schedule_check.json`, `research/audit-a26/pass-b/data/train_segments.json`, `research/audit-a26/pass-b/data/train_steps.json`, `research/audit-a26/pass-b/data/train_superseded_time.json`, `research/audit-a26/pass-b/data/train_traces.json`, `research/audit-a26/transfer/PREREG.md`, `research/audit-a26/transfer/control.md`, `research/audit-a26/transfer/taxonomy.md`, `research/audit-a26/transfer/control.json`, `research/audit-a26/tools/reconcile/compare.json`, `research/audit-a26/tools/reconcile/costs_reconcile.json`, `research/audit-a26/tools/reconcile/sens_heldout.json`, `research/audit-a26/tools/reconcile/sens_multibox.json`, `research/audit-a26/tools/reconcile/sens_multibox_hits.json`, `slurm-agentic-engineering/results/qis-a26-ckpt-k0/ckpt_k0.jsonl`, `slurm-agentic-engineering/results/qis-a26-ckpt-k1/ckpt_k1.jsonl`, `slurm-agentic-engineering/results/qis-a26-ckpt-k2/ckpt_k2.jsonl`, `slurm-agentic-engineering/results/qis-a26-ckpt-k3/ckpt_k3.jsonl`, `data/control/README.md`.

