# STATE

Purpose: where the A26 audit and report rewrite stand. Scientific authority; the harness `state/STATE.md` points here.
Updated 2026-11-12 1:30 AM EST [06:30Z]. Owner task: A26 (`project.md` §40). Previous STATE: git 045c528.

## Task
Audit the finished QIS study (Qwen3-1.7B, full-parameter GRPO; arms H, R, U × training seeds 101, 202; all six runs
complete) and rewrite `reports/REPORT.md` so a new reader can trace every claim to data, checkpoints and code.
Order (A26): definitions → audit pass A (trace) and pass B (independent) → discrepancy log → rewrite → figures and render.
No new training. Evaluations that explain curious results (e.g. J_U rising in R and H) are allowed.

## GPUs and jobs (A26 ceilings: 8 general, 6 preempt, 30 CPUs; Thu 12 Nov ~1:30 AM EST)
| State | Pool | Job | Work |
|---|---|---|---|
| running | general ×≤5 | 11122955, 11122982, 11122986, 11122988, 11122990 | `qis-a26-replay-*`: restore-and-step replays |
| running | CPU ×3 | 11122432, 11122600, 11122601 | `qis-dbg-cpu` standbys for audit reads |
| done | general | 11122477–81, 11122684, 11122732–39 | checkpoint integrity (11 dirs pass); control panel (7 policies) |

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

## Next actions
1. Replay results → `pass-a/checkpoints.md`, `replay.md`; fill report Appendix B.
2. Review the v3 draft against A26 §4 (7 acceptance tests) and §5; fix.
3. Render from a clean clone; inspect every figure at 830 px and a narrow viewport; deliver the handoff.
