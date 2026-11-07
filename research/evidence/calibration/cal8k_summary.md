# Calibration at the 8,192 cap — results table (§7; D46)

Purpose: one row per finished seed-11 calibration run; the §7 selection reads this table.
Panel calib_val (56 problems × 16 samples). Gain = pass@1(round 32) − pass@1(round 0). Cross-job eval noise ≈ 0.01 (D49).

| config | job | p@1 r0 | r16 | r32 | gain | max val trunc | mean round s | wall h | device mem max GiB | status |
|---|---|---|---|---|---|---|---|---|---|---|
| lr5e-7 / sym | 11074527 | 0.4487 | 0.4754 | 0.4799 | +0.031 | 0.45% | 176 | 2.14 | 41.2 | valid |
| lr1e-6 / sym | 11074520 | 0.4386 | 0.4665 | 0.5603 | +0.122 | 1.12% | 237 | 2.70 | 41.3 | valid |
| lr5e-7 / asym | 11074526 | 0.4866 | 0.4955 | 0.4833 | −0.003 | 0.67% | 145 | 1.77 | 40.0 | valid |
| lr1e-6 / asym | 11074519 | 0.4866 | 0.4665 | 0.5525 | +0.066 | 1.45% | 164 | 2.04 | 42.2 | valid |

Note: `mem_peak_gib` in metrics is the torch allocator peak including the vLLM sleep pool (47.5); the
device-level reading is `gpu_used_gib_train`/`gpu_used_gib_gen`.

Round-0 baselines of the identical base model differ across jobs (0.4386, 0.4487, 0.4866): cross-job
evaluations are independent samples (D50). Sampling sd 0.014 per evaluation, 0.020 per within-run gain.
Selection uses the frozen within-run gain; a pooled-baseline comparison (added when all six runs finish) is a disclosed robustness check only.
