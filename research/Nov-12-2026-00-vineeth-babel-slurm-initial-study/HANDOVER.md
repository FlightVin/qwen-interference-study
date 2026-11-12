# Handover — Nov 12, 2026 study (Qwen3-1.7B, GRPO, auxiliary math practice)

Purpose: let a new agent take over without reading the chat history. Written 2026-11-12.

## Status

- **Finished.** All six training runs (arms H, R, U × seeds 101, 202) and all evaluations are complete. The final
  report is version 4.
- **No jobs are running.** No standbys are held.
- **Records are frozen.** STATE and CHECKLIST are no longer tracked.

## Where things are

| What | Where |
|---|---|
| Results (read first) | `reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/`: understanding guide (5–10 min) → further things → full experiments report |
| Instructions the study followed | `agent-project-instructions-archive/Nov-12-2026-project-instructions.md` (mandate and owner amendments, §1–§44, in order) |
| Datasets | `data/Nov-12-2026-00-vineeth-babel-slurm-initial-study/` |
| Definitions, protocol, decisions, audits | this folder (`README.md` lists the files) |
| Code | `src/qis/` (train, evaluate, verify, analysis); tests in `tests/` |
| Checkpoints, raw answers, environment | Babel only, under `/data/user_data/<user>/qis/`. They are not in git. Checkpoints are listed in `audit-a26/pass-a/checkpoints.md` |
| Job records | Babel harness `slurm-agentic-engineering/` (ledger, recipes in `exp/`, harvested results); not in git |

## Main conclusions

Training raised every arm's sealed-test score by 20–29 points over the base model's 52.6%. R − U was +4.2 in seed 101 and
−4.4 in seed 202. Practice saturated by about round 64. We found no evidence of interference. Gains were broad, including
+21–28 points on untrained subjects. Details and caveats: the understanding guide's synopsis.

## Known loose ends

1. A single-GPU-type (L40S) rerun of the H-s202 round-128 sealed test (D166) completed but was never analysed. Its
   outputs are on Babel at `test-eval/qis-cgpu-H-s202`; the analysis script is
   `reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/build/cgpu_h_s202.py`.
2. Some older scripts open `data/…` and `research/…` at their pre-move paths (for example
   `src/qis/data/bundle_main.py`, `src/qis/analysis/reconcile_screening.py`). Jobs ran from pinned commits, so the
   results are unaffected.
3. `ops/` (the Babel submission guard) and `registry/` are git-ignored. `registry/` was emptied on 2026-11-12.

## For a next study

- Start a new dated folder in each of `research/`, `data/` and `reports/`. Archive the new instructions in
  `agent-project-instructions-archive/`.
- The open questions, ranked, are in `Nov-12-2026-further-things.md`. The first is more seeds of the same recipe on
  one GPU type.
- Install Python packages from `requirements.txt` (root `README.md`). Babel settings and the exact package lock are in
  `envs/Nov-12-2026-00-vineeth-babel-slurm-initial-study.*`. On Babel, use `slurm-agentic-engineering/`. Read its
  `agent.md` and keep environments and checkpoints under `/data/user_data`, never `/home`.
