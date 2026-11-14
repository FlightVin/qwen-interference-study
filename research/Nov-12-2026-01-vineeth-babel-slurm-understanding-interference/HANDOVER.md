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
| `qis-ui-soup` | 11130401 | cancelled 2026-11-13, 01:41 EST | Nothing. It evaluated base, R-s101-r128 and mean6, hit its time limit in mean6x, requeued, and would have written to the folder that `qis-ui-soup-b` uses. |
| `qis-ui-soup-b` | 11131755 | completed 2026-11-13, 07:19 EST, harvested | Done: all seven accuracy rows in `01` Section 1.5. |
| `qis-ui-lowrank-mean6`, `qis-ui-lowrank-R101` | 11133405, 11133408 | completed, harvested | Done: `tables/lowrank.json`, Section 1.7. |
| `qis-ui-probe`, `qis-ui-probe2` | 11133430, 11134233 | completed, harvested | Done: `04-domains-to-explore.md`, Section 2. |
| `qis-ui-geom-short` | 11133402 | completed, harvested | Done: brevity coupling, `04-domains-to-explore.md`, Section 4. |
| `qis-ui-lpprec` | 11132332 | completed 22:45 EST, harvested | Done: `build/analyze_lpprec.py`, `tables/lpprec.json`, Section 1.6. |
| `qis-ui-geom-q4x64`, `qis-ui-geom-q06x64` | 11131734, 11131757 | cancelled 2026-11-12, 22:44 EST, before start | Nothing. The owner removed them from the scope. They are future work (`01-understanding.md`, M1). |

Both soup jobs write to `$PROJ_DATA/ui/soup-eval/<model>/`. If both reach the same model at the same time, cancel the
older job (`ae slurm cancel 11130401`) after it has finished mean6x.

Then fill the place marked **pending** in `01-understanding.md` (Section 1.5), the status line at its
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


## Babel storage wiped on 2026-11-13 (owner instruction)
- Deleted on the owner's instruction: `/data/user_data/$USER/qis` (all checkpoints, environment, model cache, study
  outputs), `/data/user_data/$USER/BC2RL`, `~/ae-qis`, `~/projects/qwen-interference-study`, `~/ae-bc2rl`, `~/.cache/vllm`.
- Consequence: no result of either study can be recomputed without retraining. Reports, tables, figures and harvested
  summaries (laptop, git) remain. A new study starts from a clean state: clone the repo on Babel, rebuild the
  environment, and run the harness "First run" checks (`ae setup`, self-tests) before any submission.
