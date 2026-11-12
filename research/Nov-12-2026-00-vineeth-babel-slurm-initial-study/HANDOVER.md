# Handover: study Nov-12-2026-00-vineeth-babel-slurm-initial-study

Purpose: let a new agent understand this study and continue from it without the chat history. Written 2026-11-12.
Study ID format and repository layout: root `README.md`. Below, `<ID>` means `Nov-12-2026-00-vineeth-babel-slurm-initial-study`.

## 1. Status

- **Finished.** Six training runs (arms H, R, U × training seeds 101, 202; 128 rounds each) and all evaluations are
  complete. The final report is version 4.
- **Nothing is running.** There are no Slurm jobs and no standby allocations.
- **Records are frozen.** `STATE.md` and `CHECKLIST.md` in this folder are historical. Do not update them.
- **No task is open.** The owner's last instructions (A29–A32) reorganised the repository. They asked for no further
  experiments.

## 2. Read in this order (about 30 minutes)

1. `reports/<ID>/understanding-experiments.md`: the question, datasets, results and synopsis (10 min).
2. `reports/<ID>/further-things.md`: one-row summary plot and ranked open questions (2 min).
3. This file, sections 3–7.
4. `DEFINITIONS.md` (this folder): exact meaning of every term.
5. Only if needed: `reports/<ID>/experiments-report.md` (full report, appendices with job IDs, hashes, checkpoints) and
   `agent-project-instructions-archive/<ID>/project-instructions.md` (the mandate and owner amendments A1–A32).

## 3. The study in one paragraph

Qwen3-1.7B (non-thinking mode) was trained with full-parameter GRPO for 128 rounds. Each round had 16 problem slots ×
8 sampled answers, and a binary reward from Math-Verify on the last `\boxed{}`. Slots 0–7 always held hard problems
from `H_train` (46 algebra problems where factoring is needed but not asked for). Slots 8–15 held either factoring
drills (R), modular-arithmetic drills (U), or more hard problems (H). Evaluation used the sealed test (128 MATH test
problems × 32 answers), in-run probes (dev, J_H, J_R, J_U), stress sets (Harder, SuperHard), a control set of
unpractised subjects, a hint ("cue") study, and a response audit.

## 4. Glossary

| Term | Meaning |
|---|---|
| H, R, U | arms: slots 8–15 hold more hard problems, factoring drills, or modular-arithmetic drills |
| H_train | the 46 hard training problems (all arms) |
| slot, group, round | position for one problem; its 8 sampled answers; one sample-grade-update cycle (4 optimizer steps) |
| dev, J_H | 53 held-out problems like H_train, scored during training; J_H is its first 32 |
| J_R, J_U | probes of direct factoring (28) and modular arithmetic (32) |
| sealed test | 128 MATH test problems scored only at round 128 (the primary endpoint) |
| pass@1 | share of answers correct; an answer cut off at the token limit scores 0 |
| cap | answer token limit: 8,192 in rounds 1–32, 12,288 after; 16,384 in the length-extension test |
| seed 101 / 202 | training seed: sets the problem schedule and sampling seeds; the same within a seed across arms |

## 5. Task history (owner instructions; full text in the instructions archive)

| When (2026) | Instructions | What happened |
|---|---|---|
| Nov 7 | mandate §1–14, A1–A11 | data built and screened; calibration; protocol frozen; training started, then paused (A11) to set the cap change after round 32 |
| Nov 8–11 | A12–A16 | training resumed; interim analyses; the last run (H-s202) finished Nov 11 |
| Nov 10–11 | A17–A23 | study completed: stress, cue, audit, coverage and truncation analyses, length extension |
| Nov 11–12 | A24–A26 | two independent audit passes; discrepancy log; report v3 |
| Nov 12 | A27–A28 | report v4 (study-design figure, practice-saturation analysis, interference mapping); marked final |
| Nov 12 | A29–A32 | short guide and further-things documents; repository reorganised into `<ID>` folders |

## 6. Where everything is

| What | Where |
|---|---|
| Reports, figures, tables | `reports/<ID>/` (`build/` holds the figure scripts) |
| Datasets | `data/<ID>/` (its `README.md` lists every file) |
| Definitions, protocol, decisions D83–D167, audits | this folder (`README.md` lists the files) |
| Exact job recipes (185) and the job ledger (287 submissions: commit SHA, recipe hash, Slurm job ID) | `recipes/`, `jobs-ledger.jsonl` (this folder) |
| Code | `src/qis/`: `train/run.py` (training), `evaluate_ckpt.py`, `eval_merge.py`, `eval_extend.py` (length extension), `verify.py` (grader), `analysis/` |
| Python packages | `requirements.txt`; exact lock `envs/<ID>/pip-freeze.txt`; Babel settings `envs/<ID>/babel-slurm.env` |
| Checkpoints, raw answers, run logs | Babel only, `/data/user_data/<user>/qis/` (inventory: `audit-a26/pass-a/checkpoints.md`) |

The recipes are Bash scripts for the Babel harness `slurm-agentic-engineering/`. Each `phase name "done-test" 'commands'`
block is one restartable step. Off Babel, read them as exact command lines. For example, training runs
`python -m qis.train.run --config config.json`, and evaluation runs `python -m qis.evaluate_ckpt --ckpt … --panel … --n 32`.

## 7. Rules the owner set (keep them unless told otherwise)

- Report times in US Eastern time (EST).
- Follow the owner's writing standard for every document and message. The owner's words:

  > I have been consistently frustrated by your previous answers. Your writing often feels poorly
  > organized, overly technical, excessively detailed yet lacking in substance, and unnecessarily
  > difficult to read or follow. Correct this. Write with clarity, substance, precision, and logical
  > flow. Follow approximately 70% of the style principles of ASD-STE100 (Simplified Technical
  > English): use familiar words, short and direct sentences, active voice, consistent terminology,
  > and clear sentence structure. Keep enough flexibility for natural, fluent explanations and
  > necessary technical depth. Make each point build logically on the previous one. Explain what
  > matters and why, without burying the main ideas under excessive detail. Preserve technical
  > rigor, completeness, and nuance, but eliminate jargon, repetition, vague claims, and unnecessary
  > complexity. After drafting your response, critically reread it from the perspective of a human
  > reader. Identify exactly where the writing becomes confusing, disjointed, verbose, or hard to
  > follow, and rewrite it to fix those problems without losing important information. Do not
  > describe this review process. Output only the revised answer. No fluff.

- No new training without explicit authorization. The last compute limits were ≤ 8 general GPUs, ≤ 6 preempt GPUs
  and ≤ 30 CPUs, with one GPU per job. Act only on this project's jobs and files.
- Keep historical records. Archive a report before rewriting it, and mark superseded content instead of deleting it.
- Record each new owner instruction, dated, at the end of that study's `project-instructions.md`.

## 8. Known loose ends

1. A rerun of the H-s202 round-128 sealed test on L40S GPUs only (D166) finished but was never analysed. The outputs
   are on Babel at `test-eval/qis-cgpu-H-s202`. Analyse them with `reports/<ID>/build/cgpu_h_s202.py --root .`.
2. Records in this folder cite paths from before the reorganisation. The mapping: `reports/…` → `reports/<ID>/…`
   (`REPORT.md` → `experiments-report.md`), `data/…` → `data/<ID>/…`, `research/…` → `research/<ID>/…`.
3. Some older scripts open `data/…` directly (for example `src/qis/data/bundle_main.py`). Run them with `data/<ID>` as
   `data/`. Results are unaffected, because jobs ran from pinned commits.
4. `ops/` (the Babel submission guard) and `registry/` are git-ignored; `registry/` is empty.
5. **The code changed after the study.** This study trained with a response cap of 8,192 tokens in rounds 1–32 and
   12,288 after. Current code has one cap for every round: `MAX_TOKENS = 12288` in `src/qis/limits.py`, which is the
   default of `Rollout`, `qis.train.run` (config `max_tokens`), `qis.evaluate_ckpt` and `qis.make_main_exps`. A
   config that still contains `cap_schedule` is rejected. To reproduce this study exactly, check out the commit that
   each job ran (`jobs-ledger.jsonl`, field `sha`).

## 9. Starting a new study

1. Choose a new ID, for example `<date>-01-<owner>-<compute>-<label>`.
2. Create `reports/<new ID>/`, `data/<new ID>/`, `research/<new ID>/`, `envs/<new ID>/` and
   `agent-project-instructions-archive/<new ID>/project-instructions.md`.
3. Begin with the open questions in `reports/<ID>/further-things.md`. The first is more seeds of the same recipe on
   one GPU type.
4. On Babel, start the harness (`slurm-agentic-engineering/agent.md`, "Session bootstrap"). Keep environments and
   checkpoints under `/data/user_data`, never `/home`.
