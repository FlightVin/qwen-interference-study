# HANDOVER — study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`

Purpose: what a new agent or reader needs to continue this study. Written 2026-11-12, 21:30 EST. `<ID>` is the study ID.

## 1. What exists

| Item | Location |
|---|---|
| Theory and literature | `reports/<ID>/00-initial-theory.md` |
| Measurements, explanation, proposed experiments | `reports/<ID>/01-understanding.md` |
| Figure and table scripts | `reports/<ID>/build/` (`theory_figs.py`, `analyze_wdelta.py`, `analyze_halves.py`, `analyze_geom.py`, `analyze_geom_extra.py`, `analyze_lp.py`) |
| Tables and figures | `reports/<ID>/tables/`, `reports/<ID>/figures/` |
| Measurement code | `src/qis/interf/` |
| Job recipes | `research/<ID>/recipes/` (copies of the harness `exp/qis-ui-*.sh`) |
| Mandate and owner amendments | `agent-project-instructions-archive/<ID>/project-instructions.md` |
| Compute outputs on Babel | `/data/user_data/$USER/qis/ui/` (`samples/`, `grad/`, `wdelta/`, `lp/`, `lp-prec/`, `soup/`, `soup-eval/`, `models/`) |
| Harvested distillates | harness `results/qis-ui-*/` (laptop only; the harness is git-ignored) |

## 2. Jobs still open on 2026-11-12, 21:30 EST

| Experiment | Job | State | When it finishes, do this |
|---|---|---|---|
| `qis-ui-soup` | 11130401 | running | It evaluates base, R-s101-r128, mean6, mean6x, then loo5, resid-R101, scaled-R101 until its 8-hour limit. Harvest; read `summary_<model>.json`. |
| `qis-ui-soup-b` | 11131755 | pending on `preempt` | Evaluates resid-R101, scaled-R101, loo5, mean6x and skips models already done. Its export phase copies all summaries. |
| `qis-ui-lpprec` | 11132332 | running | Harvest; compare `fp32/`, `fp32-wround/`, `train/` `lp.npz` (same format as `qis-ui-lp`). |
| `qis-ui-geom-q4x64` | 11131734 | pending on `general` | Harvest; if `small.npz` exceeds the harvest limit, move it aside as in DECISIONS U9. Rerun `analyze_geom.py`. |
| `qis-ui-geom-q06x64` | 11131757 | pending on `preempt` | as above |

Both soup jobs write to `$PROJ_DATA/ui/soup-eval/<model>/`. If both reach the same model at the same time, cancel the
older job (`ae slurm cancel 11130401`) after it has finished mean6x.

Then fill the three places marked **pending** in `01-understanding.md` (Sections 1.5, 1.6, 1.7), the status line at its
top, and the verdict table if a result changes a verdict.

## 3. How to regenerate the tables and figures

From `reports/<ID>/`, with a Python that has numpy and matplotlib, and `RES` = the harness `results/` folder:

    python build/theory_figs.py
    python build/analyze_wdelta.py $RES && python build/analyze_halves.py $RES
    python build/analyze_geom.py $RES && python build/analyze_geom_extra.py $RES
    python build/analyze_lp.py $RES

## 4. Rules that still apply
- No training runs. Propose them (mandate). Section 3 of `01-understanding.md` holds the proposals.
- GPU ceilings: 8 `general` (owner amendment of 2026-11-12, until told otherwise) and 6 `preempt`. Guard: `ops/qis-submit`.
- Records of the first study are frozen. Its checkpoints under `$PROJ_DATA/ckpt` are read-only.
- The Babel account is shared with another project. Control only this project's jobs, by ledger job ID.

## 5. Known limits of the evidence
- One model size was trained. Capacity claims are untested.
- Trained-checkpoint gradient runs use 16 answers per problem and cannot resolve cosines below about 0.001.
- The transfer measure S_D is a log-probability measure at base-model samples, not an accuracy.
- Answer length was measured for one trained model (R-s101, round 128).
- The first-order alignment of base gradients with the run updates is below the count-sketch noise.
- A bare `ae harvest` early in this study pulled about 20 old first-study experiments into the harness `results/` and
  appended lines to the harness `state/FINDINGS.md`. No tracked first-study record changed.
