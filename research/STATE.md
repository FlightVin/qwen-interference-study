# STATE

Purpose: where the A26 audit and report rewrite stand. Scientific authority; the harness `state/STATE.md` points here.
Updated 2026-11-11 8:50 PM EST [01:50Z Thu]. Owner task: A26 (`project.md` §40). Previous STATE: git 045c528.

## Task
Audit the finished QIS study (Qwen3-1.7B, full-parameter GRPO; arms H, R, U × training seeds 101, 202; all six runs
complete) and rewrite `reports/REPORT.md` so a new reader can trace every claim to data, checkpoints and code.
Order (A26): definitions → audit pass A (trace) and pass B (independent) → discrepancy log → rewrite → figures and render.
No new training. Evaluations that explain curious results (e.g. J_U rising in R and H) are allowed.

## GPUs and jobs (A26 ceilings: 8 general, 6 preempt, 30 CPUs; 9:25 PM EST)
| State | Pool | Job | Work |
|---|---|---|---|
| running | CPU | 11122432 | `qis-dbg-cpu` standby (4 CPUs, 12 h) for read-only audit work on compute |
| running | general ×4 | 11122477, -79, -80, -81 | `qis-a26-ckpt-k0..3`: hash + load all 11 retained checkpoint dirs (D160) |
| done | CPU | 11122475 | `qis-a26-ckpt-list`: 11 checkpoint dirs, 2 orphan R-s202 `.tmp` saves (rounds 4, 12) |
GPUs in use: 4 general. Retained: round 128 of all six runs (R-s101, U-s101 weights only), R-s101 r64, H-s202 r64,
H-s202 fallback r116 (weights only). All other rounds were pruned and are unavailable.

## Results under audit (from report v2; claims until pass A and pass B verify them)
Sealed test pass@1 (128 problems × 32 attempts): base 52.6%; s101 H 77.0, R 81.9, U 77.7; s202 H 72.4, R 74.8, U 79.2.
R − U changes sign between seeds (+4.2, −4.4). R − H and U − H are positive in both seeds; two of four per-seed
intervals include zero. Report v2 archived: `reports/archive/REPORT-v2-2026-11-11.md`.

## Known report defects to fix (A26 §4 acceptance tests)
1. Ranges that omit H-s202 (sealed 72.4%; stress 60.1%, SuperHard 49.4%).
2. "Sign appeared between rounds 32 and 64" contradicts seed-202 dev at round 32.
3. "More room does not rescue" contradicts 16.2% / 19.8% recovery (s202 sealed test).
4. R−H, U−H wording overstated; per-seed intervals that include zero.
5. Interference claims beyond the measured links; "three-point effect" claim.
6. Audit denominators (96 / 35 / 2 reads / 33) and "fixable" wording.
7. Contrasts must come from unrounded data; no visibly failing equations.

## Next actions
1. Draft `research/DEFINITIONS.md` v1 from protocol, amendments, manifests and executed code; verify each term.
2. Start the checkpoint inventory (enumerate, hash, load) on compute while definitions are drafted.
3. Pass A, then pass B (independent context and code), then the discrepancy log.
