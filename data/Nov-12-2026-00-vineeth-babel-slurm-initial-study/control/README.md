# control_v1 — A26 control panel (MATH test, unpractised subjects)

Purpose: measure whether training gains appear on subjects no arm practises (A26 x-transfer). Evaluation-only; never used
for training or selection. Rule and analysis plan: `research/audit-a26/transfer/PREREG.md` (written before the panel was built).

- File `control_v1.json`, pool `control`, 64 problems. **sha256 `fc2ceb70ef920ab3c5a33baa78b7221f9a561db734a04942eba20ba48126e112`**.
- Source: MATH `EleutherAI/hendrycks_math@21a5633`, official `test` split, Levels 4–5, from `$P/data/splits-v1/manifest.jsonl`.
- Composition: geometry 16, counting_and_probability 16, precalculus 16, prealgebra 16; Level 5 36, Level 4 28.
- Order: `sha256("a26-control/" + id)` ascending within subject; first 16 eligible per subject. No model output was read.
- Exclusions (of 1,124 candidates): 95 IDs found in a study file (85 of them only via the A17 stress candidate list or stress
  panels), 18 close variants (masked word-3-gram Jaccard ≥ 0.5 with a study text), 0 exact/normalised-text matches, 0 golds
  failing `qis.verify.gold_eligible`. Scan: 156 JSON/JSONL files, 16,776 texts, 2,794 MATH IDs under `$P/data/*` (except
  `splits-v1`, `raw-21a5633`) and repo `data/` at 9b2fd05.
- Built by `research/audit-a26/transfer/tools/build_control.py` on compute (standby 11122432, 2026-11-11 EST); full report
  `research/audit-a26/transfer/control_build_report.json`.
