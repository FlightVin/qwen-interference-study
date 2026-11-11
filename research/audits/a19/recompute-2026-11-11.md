# Independent recomputation of headline numbers (A19 §5; CHECKLIST 8.8a) — 2026-11-11 16:30Z

Purpose: check REPORT headline numbers against the raw attempt records with code that shares nothing with the project
(`recompute.py`, Python standard library only). Run on compute (srun step in job 11112377, read-only) over
`$P/{test-eval,test-eval-cgpu,stress-eval}/*/records.jsonl.gz`. Definitions: pass@1 = mean over problems of correct /
attempts; truncated = share of attempts with finish "length"; accuracy on finished = correct / finished attempts.

| Cell | Problems × attempts | pass@1 | Truncated | Acc. finished | REPORT value | Verdict |
|---|---|---|---|---|---|---|
| test base | 128 × 32 | 0.5256 | 0.05% | 52.6% | 52.6% | verified |
| test H-s101 | 128 × 32 | 0.7695 | 10.9% | 86.3% | 77.0% | verified |
| test R-s101 | 128 × 32 | 0.8186 | 10.7% | 91.6% | 81.9%, trunc 10.7% | verified |
| test U-s101 | 128 × 32 | 0.7769 | 12.2% | 88.5% | 77.7%, trunc 12.2% | verified |
| test R-s202 | 128 × 32 | 0.7478 | 13.1% | 86.1% | 74.8%, trunc 13.1%, acc 86.1% | verified |
| test U-s202 | 128 × 32 | 0.7922 | 8.1% | 86.2% | 79.2%, trunc 8.1%, acc 86.2% | verified |
| common GPU R-s101 | 128 × 32 | 0.8225 | 10.9% | 92.4% | 82.3% | verified |
| common GPU R-s202 | 128 × 32 | 0.7612 | 12.4% | 86.9% | 76.1% | verified |
| common GPU U-s101 | 128 × 32 | 0.7781 | 12.3% | 88.8% | 77.8% | verified |
| common GPU U-s202 | 128 × 32 | 0.7922 | 8.1% | 86.2% | 79.2% | verified |
| stress base | 221 × 8 | 0.3727 | 0.06% | 37.3% | 37.3% | verified |
| stress H-s101 r128 | 221 × 8 | 0.6414 | 17.7% | 77.9% | 64.1% | verified |
| stress R-s101 r128 | 221 × 8 | 0.6951 | 20.5% | 87.4% | 69.5%, trunc 20.5% | verified |
| stress U-s101 r128 | 221 × 8 | 0.6533 | 23.0% | 84.8% | 65.3% | verified |
| stress R-s202 r128 | 221 × 8 | 0.6318 | 23.0% | 82.1% | 63.2%, trunc 23.0%, acc 82.1% | verified |
| stress U-s202 r128 | 221 × 8 | 0.6584 | 15.2% | 77.6% | 65.8%, trunc 15.2%, acc 77.6% | verified |
| stress R-s101 r64 | 221 × 8 | 0.6425 | 26.1% | 87.0% | 64.3%, trunc 26.1% | verified |
| stress H-s202 r64 | 221 × 8 | 0.6239 | 15.1% | 73.5% | 62.4% | verified |

| Contrast | Recomputed | REPORT | Verdict |
|---|---|---|---|
| test R − U s101 | 0.8186 − 0.7769 = +0.0417 | +4.2 | verified |
| test R − U s202 | 0.7478 − 0.7922 = −0.0444 | −4.4 | verified |
| test R − H s101 | 0.8186 − 0.7695 = +0.0491 | +4.9 | verified |
| common GPU R − U s101 | 0.8225 − 0.7781 = +0.0444 | +4.4 | verified |
| common GPU R − U s202 | 0.7612 − 0.7922 = −0.0310 | −3.1 | verified |
| common GPU R − H s101 | 0.8225 − 0.7695 = +0.0530 | +5.3 | verified |
| stress R − U s101 | 0.6951 − 0.6533 = +0.0419 | +4.2 | verified |
| stress R − U s202 | 0.6318 − 0.6584 = −0.0266 | −2.7 | verified |
| stress R − H s101 | 0.6951 − 0.6414 = +0.0537 | +5.4 | verified |

Not covered here: bootstrap intervals (project code; spot-check pending), H-s202 cells (not yet evaluated), length
diagnostic (running). Counted 18 cells and 9 contrasts; all verified.
