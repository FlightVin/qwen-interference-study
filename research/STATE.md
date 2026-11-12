# STATE

Purpose: where the audit and report stand. Scientific authority; the harness `state/STATE.md` points here.
Updated 2026-11-12 10:56 AM EST [15:56Z]. Owner task: A27 (`project.md` §41), on top of A26 (§40). Previous STATE: git c8a81ac.

## Reports (moved 12 Nov): `reports/Nov-12-2026-initial-experiments/` — experiments report, understanding guide, further things

## A27 (Thu 12 Nov, deadline 12:00 PM EST)
Report v4 delivered: `reports/Nov-12-2026-initial-experiments/Nov-12-2026-experiments-report.md` (v3 archived as `reports/archive/REPORT-v3-2026-11-12.md`, figures in
`reports/archive/figures-v3-2026-11-12/`). No v3 number changed. Added: title naming the subject, design diagram (F1),
worked GRPO group, groups without reward contrast per pool and quarter (F9, `reports/tables/v4/t10_signal`), ray
interference from primary sources mapped to measurements, discussion of what slots 8–15 bought, raw examples (App. I).
New finding: from round 65, 78–92% of R and U practice groups were all correct (200–236 of 256 per quarter) against
44–57% of H's extra hard groups → the R/U contrast was tested mainly in rounds 1–64 (D164).
GPUs: 0 project jobs at start (squeue --me, 10:47 AM EST); 8 L40S shards submitted for D166; no re-evaluation could resolve the main
uncertainty (run-to-run spread) inside the deadline. Checkpoint inventory (pass A 2.6) was already complete (6282ba0).
Report v4 is FINAL (owner, 12 Nov ~12:15 PM EST). D166 rerun: 8/8 shards completed on Babel; not harvested or
analysed by owner instruction (outputs remain in `test-eval/qis-cgpu-H-s202`). No jobs running.

## Task
Audit the finished QIS study (Qwen3-1.7B, full-parameter GRPO; arms H, R, U × training seeds 101, 202; all six runs
complete) and rewrite `reports/Nov-12-2026-initial-experiments/Nov-12-2026-experiments-report.md` so a new reader can trace every claim to data, checkpoints and code.
Order (A26): definitions → audit pass A (trace) and pass B (independent) → discrepancy log → rewrite → figures and render.
No new training. Evaluations that explain curious results (e.g. J_U rising in R and H) are allowed.

## Results under audit (from report v2; claims until pass A and pass B verify them)
Sealed test pass@1 (128 problems × 32 attempts): base 52.6%; s101 H 77.0, R 81.9, U 77.7; s202 H 72.4, R 74.8, U 79.2.
R − U changes sign between seeds (+4.2, −4.4). R − H and U − H are positive in both seeds; two of four per-seed
intervals include zero. Report v2 archived: `reports/archive/REPORT-v2-2026-11-11.md`.

## Audit findings so far (pass A; verified unless marked)
- a-data: all schedule checks pass in six runs (shared slots, extra slots, pairs, minibatch plans, advantages, zero-gradient
  steps 18/20/66/25/53/65). New: sealed-test #30 `intermediate_algebra/test/00383` and dev #42 `algebra/train/01291` are
  close variants of H_train items (masked 3-gram Jaccard 0.725 and 1.0) → sensitivity check without them (pending).
  J_R/J_U include Level 1–3 items (14/28, 16/32). `research/audit-a26/pass-a/data-inventory.md`, `schedules.md`.
- Pass B b-eval (independent code, raw records only): sealed test 12 cells × 4,096 attempts complete (0 missing/duplicate);
  pass@1 equals report v2 for base and all six endpoints. Per-seed intervals: R−H s202 [−0.5, +5.3] and U−H s101
  [−1.9, +3.6] include 0. H-s202 test shards k0–k2 L40S, k3–k5 A6000; its stress shards L40 + A6000 (others L40S).
  17 finished reward-0 sealed-test responses have an earlier box equal to the gold string (grader check pending).
  `research/audit-a26/pass-b/eval.md`.
- a-code: 54/54 tests pass; provenance complete for 265 ledger jobs; schedule, grading, advantages, minibatch plans,
  resume handling and all 46 evaluation + 15 extension outputs verified by re-implementation. Training GPU differs:
  H-s101 all L40; H-s202 L40 r1–60 then L40S; R-s202 L40 r1–16 then L40S; others L40S. 2 unknowns in base test (≤0.05 pt).
  `research/audit-a26/pass-a/code.md`, `jobs.md`.
- Pass B b-train: curves, group totals, momentum-only steps agree with records and DEFINITIONS. Earliest R/U ordering as at
  r128: s101 dev/J_H from r64 (U ahead at r32: dev .601 vs .542); s202 dev/J_H from r32; all first-round intervals include 0.
  Diagnostic: two no-update evaluations of the same weights differ by up to 0.031, more than the branch-type spread in 5/6
  cells; ΔJ_H sign depends on the reference → ΔJ separates nothing. Costs 317.0 GPU-h / 2,567 CPU-h (incl. A26 so far).
  `research/audit-a26/pass-b/train.md`.
- Checkpoints (k1, k3 of 4): 5 dirs pass fresh hash, load, finiteness, per-tensor checksums; H-s202 r128 optim step 512.

## Known report defects to fix (A26 §4 acceptance tests)
1. Ranges that omit H-s202 (sealed 72.4%; stress 60.1%, SuperHard 49.4%).
2. "Sign appeared between rounds 32 and 64" contradicts seed-202 dev at round 32.
3. "More room does not rescue" contradicts 16.2% / 19.8% recovery (s202 sealed test).
4. R−H, U−H wording overstated; per-seed intervals that include zero.
5. Interference claims beyond the measured links; "three-point effect" claim.
6. Audit denominators (96 / 35 / 2 reads / 33) and "fixable" wording.
7. Contrasts must come from unrounded data; no visibly failing equations.

## Done
DEFINITIONS v1 (26cd48b) → v1.1 (2d43d2e); pass A and pass B (db8c2ed); comparison + discrepancy log, 38 rows (899b4a9;
D162); control panel: probe rise is general improvement, +21 to +28 points on untrained subjects (D163); figures F1–F12 and
tables t01–t09, value check 0 mismatches (8bd3f33).

## Status
A26 delivered: report v3 `reports/Nov-12-2026-initial-experiments/Nov-12-2026-experiments-report.md` (5b211c7, figures 1e75db1); DEFINITIONS v1.1; pass A, pass B, comparison,
discrepancy log (38 rows), D162/D163; v2 archived. Waiting for the owner.
