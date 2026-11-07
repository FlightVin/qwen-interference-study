# Calibration at the 8,192 cap — results table (§7; D46)

Purpose: one row per finished seed-11 calibration run; the §7 selection reads this table.
Panel calib_val (56 problems × 16 samples). Gain = pass@1(round 32) − pass@1(round 0). Cross-job eval noise ≈ 0.01 (D49).

| config | job | p@1 r0 | r16 | r32 | gain | max val trunc | mean round s | wall h | device mem max GiB | status |
|---|---|---|---|---|---|---|---|---|---|---|
| lr5e-7 / sym | 11074527 | 0.4487 | 0.4754 | 0.4799 | +0.031 | 0.45% | 176 | 2.14 | 41.2 | valid |

Note: `mem_peak_gib` in metrics is the torch allocator peak including the vLLM sleep pool (47.5); the
device-level reading is `gpu_used_gib_train`/`gpu_used_gib_gen`.
