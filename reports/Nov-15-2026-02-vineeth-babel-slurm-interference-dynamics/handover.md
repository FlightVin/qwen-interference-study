# Handover — study s02 (Nov-15-2026-02-vineeth-babel-slurm-interference-dynamics)

This file tells the next person or agent how to continue study s02. Read `report.md` first for the results.

## 1. Where things are

| What | Where |
|---|---|
| Report, figures, run table | `reports/<ID>/report.md`, `figures/`, `runs.md` |
| Figure and analysis scripts | `reports/<ID>/build/figs.py`, `build/analyze.py` (read harvested `metrics.jsonl` only) |
| Experiment specs and recipes | `research/<ID>/specs/*.json` → `python -m qis.s02.make_exps --spec ... --expdir slurm-agentic-engineering/exp` |
| Copies of every submitted recipe | `research/<ID>/recipes/` |
| Snapshot of the working records | `research/<ID>/records-2026-11-16/` (STATE, CHECKLIST, FINDINGS, DECISIONS, direction, implementation log) |
| Data bundle identity | `data/<ID>/BUNDLES.md` (bundle `s02-v1`, SHA-256 `a761b981...`) |
| Environment | `envs/<ID>/babel-slurm.env`, `envs/<ID>/pip-freeze.txt` (vllm 0.31.0, torch 2.13.0+cu130, transformers 5.10.4) |
| Owner mandate (verbatim) | `agent-project-instructions-archive/<ID>/project-instructions.md` |
| Harvested distillates | `slurm-agentic-engineering/results/<expid>/` (git-ignored harness workspace on the laptop) |
| Full run outputs, records, checkpoints | Babel `/data/user_data/vineethb/qis/s02/{runs,ckpt,data}` (compute nodes only) |

Code: task generators `src/qis/s02/tasks.py`; graders `src/qis/verify.py`; trainer `src/qis/train/run.py` (s02 keys:
`mix`, `n_per_prompt`, `loss_norm_responses`, `init_ckpt`/`init_round`, `no_ckpt`, `allow_uneven_minibatch`); tests
`tests/test_s02.py` (run as the `tests` phase of every recipe).

## 2. How to run one more experiment

1. Add a spec entry (copy an existing one in `research/<ID>/specs/`). Keep `exclude: babel-l9-32`.
2. Generate the recipe with `make_exps`, commit the spec and recipe copy, then submit with `ops/qis-submit <expid>`
   (it enforces the ceilings: 2 general GPUs, 6 preempt GPUs, 20 CPU jobs; expids must start `qis-s02-`).
3. Watch with `bin/ae-watch` as a background task; harvest with `bin/ae harvest <expid>`.
4. Analyse: `python reports/<ID>/build/analyze.py --res slurm-agentic-engineering/results --out <dir> --runs A,B
   --pairs J:R` (one pair per call), then rerun `figs.py`.

## 3. Rules learned in this study (cost time or compute)

- **Checkpoints are 20-21 GB each.** Study checkpoints filled the shared 500 GB disk once (06:19 EST). Recipes now
  refuse to start below 60 GB free and delete non-retained checkpoints at the end. Keep at most 4 checkpointing runs.
- **NFS writes can slow to ~6 MB/s.** A checkpoint then takes an hour. For non-preemptible runs that need no
  checkpoint, set `no_ckpt: true`.
- **Reading a running job:** `srun --jobid=<id> --overlap -n1 -c1 --mem=50M <cmd>`. Without `--mem` it hangs.
- **A requeued run replays rounds since its last checkpoint;** `metrics.jsonl` then holds duplicate round records.
  Use the last record per (kind, panel, round).
- **vLLM sampling is not reproducible across jobs;** same-model evaluations differ by up to 0.04.
- **A zero-gradient round still moves the weights under AdamW** (momentum). A "null" branch is not null.
- **Adam step sharing:** in a joint run with an active competitor, each task gets a smaller share of each step. A
  joint-run lag of the size seen here can come from that alone (half-lr control, F17).

## 4. Open questions, in priority order

1. **What triggers the add9 d6 collapse?** Seen in 2/5 mixed runs, 0/5 without d8, 0/2 with a competitor (no
   difference significant; p = 0.44 for d8). More seeds alone are costly at this rate; better: log
   per-minibatch composition and gradient norms by stratum; a random-reward stratum in place of d8 (reviewer's
   suggestion) to separate "sparse noisy updates" from d8 content.
2. **Is the Caesar delay under terse caused by brevity spill-over?** The delay appeared in 2 of 3 seeds, exactly
   those where Caesar answers got shorter (F19). Test by intervention: a terse variant with a looser length limit,
   or a length penalty on Caesar answers in the reference run.
3. **Recovery dynamics with more than one collapse state.** F10 rests on one collapse state and one branch per
   condition; separate "more samples" from "larger step" (normalise the loss per group size).
4. **Why d4-only training lowers terse while d6-only raises it** (format drift, unexplained).

## 5. State at hand-over

All 42 study jobs have ended; `runs.md` lists each final state from `sacct`. No study job was running at hand-over.
Compute storage under `/data/user_data/vineethb/qis/s02` holds run directories (metrics, training and evaluation
records) and bundles. Measured at 16:37 EST (job 11177886, `qis-s02-diag-final`): no `round_*` checkpoints remain; study s02 uses about
256 MB (runs 241 MB, data 5.1 MB, ckpt 161 KB, other 9.4 MB); the shared filesystem had 200 GB free (301 of 500 GB
used, mostly other projects).
