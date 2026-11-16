# DECISIONS — open judgement calls and dated decisions (study s02)

- **2026-11-15 D1 — pool for short jobs: `general`.** Calibration and canary on `general` (started < 1 min);
  preempt had ~11 pending jobs of the other project. Re-measure before the screen.
- **2026-11-15 D2 — code workflow.** The orchestrator wrote the s02 runner extensions on the laptop (no project code
  run there), an independent reviewer agent read the diff (5 defects, fixed), and compute-side unit tests gate any
  training job. No tinker was used for this first change because no CPU standby existed and the calibration needed
  the code at once; later code changes go through tinkers.
- **2026-11-15 D3 — reference arm.** R(T) uses the joint arm's loss divisor (32), so raw per-response T gradients match.
  Under AdamW the divisor barely changes the update size; the remaining J-vs-R differences (Adam second moment,
  grad-norm clip, minibatch composition) are counted as competition channels, not removed.
- **2026-11-16 01:15 D4 — E1 trimmed; joint runs queued.** Cancelled the pending single-task runs countdown (11168357)
  and ARC (11168359) before start: with ~2 usable GPUs they are the least informative (both panels are evaluated in
  every run, so their off-target effects are still measured; countdown answers are long and 17-19% truncated at 4
  numbers). Queued joint runs J(caesar, terse) 11168639 and J(add9, add10) 11168641; J(caesar, mbpp) and J(add9, terse)
  are specified (specs/e1b.json) and wait for a slot. Order for free general GPUs: J(caesar,terse), J(add9,add10),
  R(terse), R(add10), R(mbpp), J(caesar,mbpp), J(add9,terse).
