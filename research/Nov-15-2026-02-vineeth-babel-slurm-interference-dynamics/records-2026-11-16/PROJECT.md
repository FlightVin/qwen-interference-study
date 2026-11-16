# PROJECT — study `Nov-15-2026-02-vineeth-babel-slurm-interference-dynamics`

Purpose: the durable mandate of this study, in operational form. The owner's text is archived verbatim in
`agent-project-instructions-archive/Nov-15-2026-02-vineeth-babel-slurm-interference-dynamics/project-instructions.md`
(received 2026-11-15, 23:10 EST). Where this file and the archive differ, the archive wins. Short name: **s02**.

## 1. Central question
Under what conditions do updates from one skill suppress another's success, weaken its subsequent learning signal,
and delay its acquisition? Answer RQ1 and RQ2 together. The goal is a broad, precise, carefully bounded account,
not a new algorithm, a full benchmark or a model-size sweep.

- **RQ1.** When does learning skill C impede skill T, and what predicts it? Classify tested pairs as beneficial
  transfer, approximately independent, transiently harmful (repaired), or persistently harmful / delayed. Test
  whether shared component skills, different solution procedures, or competing behavioural demands explain the
  outcome. Do not assume that semantic similarity predicts gradient alignment or transfer. Ask whether the
  interaction changes during training.
- **RQ2.** Does on-policy feedback amplify an initial disadvantage? The proposed chain: adverse cross-skill updates →
  lower T success → fewer mixed T groups (less corrective reward contrast) → slower recovery or delayed acquisition.
  Vary initial T success and competing update strength as independently as practicable. Sub-questions: persistence
  when T successes are scarce; growth of small imbalances; task order; suppression after C's signal weakens;
  alternatives (intrinsic difficulty, saturation, optimizer effects, truncation, evaluation resolution).

## 2. Operational definitions (used in every record of this study)
| Term | Meaning |
|---|---|
| task / skill | a procedural problem family with a fixed prompt format (`src/qis/s02/tasks.py`) |
| T, C | target task; competing (or compatible) task |
| round | sample 8 responses for each scheduled prompt, grade, then `n_minibatch` (4) AdamW steps; π_old fixed per round |
| group | the 8 responses to one prompt. **All-wrong (A0)**, **mixed** (1–7 correct), **all-correct (A1)**. Only mixed groups give a GRPO gradient |
| joint run J(T,C) | each round 8 T groups + 8 C groups; loss divisor 32 responses per step |
| reference R(T) | T alone: the same T prompts and sampling seeds per round, 8 T groups, 4 steps, divisor 32. Matches T exposure and the raw per-response gradient weight. It does NOT match the update size: Adam's second moment and the global grad-norm clip see C's gradients only in J. Those are channels of competition, reported as such |
| p_T(θ) | pass@1 of T on a held-out panel (n samples per problem), overall and per stratum (difficulty level) |
| cumulative deficit | D(t) = p_T(R, t) − p_T(J, t) at matched round t, same seed. Positive = harm |
| acquisition delay | rounds for J minus rounds for R to reach a stated p_T threshold |
| local update effect | from a saved state (model + Adam), the change in T (pass@1 and/or log-prob of fixed T responses) after one round of updates on C only, minus the change after a null round (zero gradient; momentum drift only) |
| transient vs persistent | transient: D(t) returns inside its uncertainty band by the end of the horizon; persistent: it does not, while R is still learning |

## 3. Competing explanations to keep apart
1. Direct adverse update (C's step lowers p_T) vs reduced T exposure / gradient weight / compute share (controlled by R).
2. Optimizer-mediated coupling (Adam second moment, grad clip, momentum) vs gradient direction conflict.
3. Feedback amplification (fewer mixed T groups → weaker corrective signal) vs intrinsic difficulty (T fails alone too),
   saturation (A1 groups), truncation / format collapse, grading artefacts, and evaluation noise.
4. Natural interference (marked, distinct tasks) vs a controlled objective conflict (e.g. a brevity demand). Contradictory
   rewards on identical unmarked prompts are never the main evidence of failed joint acquisition.

## 4. Strategy (adaptive)
Cheap calibration and grader checks → short update-effect probes → short paired trajectories (J vs R) with early,
frequent evaluation → independent confirmation (new seed, `conf` split) of promising or surprising effects → longer
runs only for a named remaining uncertainty. Each experiment family records in `research/<ID>/FAMILIES.md`: question,
competing explanations, comparison and what is held fixed, predicted observations, smallest horizon and evaluation
budget, expansion / stopping / abandonment criteria, and what it cannot establish.

## 5. Success criteria (deliverable)
- A defensible classification of the tested task relationships (helpful / negligible / transient / persistent), each
  with the comparison, effect size, uncertainty (seed- and problem-level) and limitation.
- A bounded answer to whether scarce corrective feedback amplifies harm, with the discriminating test used.
- The alternative explanations that remain; where short runs suffice and where longer evidence is needed.
- A concise report in `reports/<ID>/` (figures, experiment-to-claim evidence table, exploratory vs confirmed) and a
  handover in `research/<ID>/HANDOVER.md` with reproduction paths and a prioritised continuation plan.
- **Soft deadline: 9:00 PM EST, Monday 2026-11-16** for a coherent initial synthesis of both questions. Assess its
  feasibility after two hours of real work (about 01:15 EST 2026-11-16) from measured throughput and queue delays.
  It is a planning target, not a reason to weaken validation; continue after it if evidence is unresolved.

## 6. Constraints
- **Model:** `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, non-thinking chat template
  (`enable_thinking=False`; the prompt ends with an empty think block; fixture in `qis.prompts.template_fixture`).
  Every run starts from this base model or from a checkpoint trained in this study. No model-size sweep.
- **Resource ceilings:** ≤ 2 GPUs on `general`, ≤ 6 GPUs on `preempt`, ≤ 20 concurrent CPU-only jobs on
  `preempt_cpu_qos` (account limits: 64 CPUs and 20 running jobs per user, 32 CPUs per job, shared with another
  project on the same username). Standbys count. Guard: `ops/qis-submit`.
- **Data use:** `train` items for training; `sel` panels for calibration, screening and selection decisions; `conf`
  panels for confirmation and reported effects of screened comparisons. Splits are disjoint by operand pair.
- **Speed changes must be disclosed:** response cap, samples per prompt, batch size and precision are recorded per run;
  a change that may alter the phenomenon is validated before use.
- **Old studies are frozen.** Their findings are hypotheses only. Their checkpoints no longer exist (wiped 2026-11-13).
- Act only on this study's jobs (ledger job IDs, `qis-dbg*` standbys).

## 7. Validity and provenance
Before expanding a run family: validate the generated config and data path; check grading, unknown-verifier handling,
reward normalisation, token masking, task weighting and split integrity; run a canary through rollout, optimisation,
checkpointing, resume and evaluation, and read its metrics before releasing dependants. Keep seed-level results;
treat the problem (not the response) as the sampling unit. Record length, truncation and decoding with correctness.
Every result names: code commit, recipe hash, effective config, model revision, bundle SHA-256 and split, seeds,
precision, optimizer settings, training and evaluation budget. A result that changes direction gets a second
critical review (bug, exposure, grading, precision, noise, effective update size). A null result states what effect
size it could detect. Never write "absent" for "unresolved".

## 8. Records and monitoring
- Canonical, git-ignored workspace: this file, `state/STATE.md` (≤ 60 lines), `state/CHECKLIST.md`,
  `state/direction.md` (living direction + hourly review log), `state/DECISIONS.md`, `state/IMPLEMENTATION-LOG.md`,
  `state/FINDINGS.md`, `log/2026-11.md`.
- Durable, in git: `research/<ID>/` (dated snapshots of the records above, `FAMILIES.md`, recipes, job ledger copy,
  findings), `reports/<ID>/`, `data/<ID>/` (bundle specs and hashes), `envs/<ID>/`.
- Hourly scientific and operational review (session cron at :17; session-only, see `state/direction.md`), plus a
  review after every important completion, failure or surprise. Event monitoring by `bin/ae-watch` only.

## 9. How a run executes
- Environment: `set -a; source envs/<ID>/babel-slurm.env; set +a` on a compute node. venv `$PROJ_DATA/env/venv-s02`
  (study-00 lock `envs/<ID>/pip-freeze.txt`, installed with `--no-deps`). Model cache `$PROJ_DATA/hf`.
- Bundle: `python -m qis.s02.tasks --out B.json --spec family:stratum[:n_train],...`.
- Training: `python -m qis.train.run --config config.json` with `mix`, `loss_norm_responses`, `n_per_prompt`,
  `n_minibatch`, `init_ckpt` + `init_round` (branches). Exit 143 = preemption (recipe requeues).
- Evaluation: in-run `eval_panels` at `eval_rounds`; standalone `qis.evaluate_ckpt`; calibration `qis.s02.calib`.

## 10. Data and paths (compute storage)
- `PROJ_DATA=/data/user_data/$USER/qis`; this study under `$PROJ_DATA/s02/{data,runs,ckpt,eval}`.
- Checkpoints: ~20 GB each (FP32 model + AdamW). Retain only those a planned branch needs. Quota 500 GB.

## 11. What a finished experiment must emit (to `$AE_RESULTS`, ≤ 2 MB)
`metrics.jsonl` (round, step and eval records incl. per-role A0/mixed/A1 counts and per-stratum eval), `config.json`,
`gpu.txt`, and for evaluations a `summary.json` with per-problem counts.

## 12. Deferred questions (do not sweep now; small controls are allowed)
Interventions that restore learning; the role of pre-training; capacity; optimizer and training-procedure effects.

## Amendments
### A1 (owner, 2026-11-15 23:30 EST) — broaden candidate tasks beyond mathematics
Verbatim text in the archive (`project-instructions.md`, Amendment A1). Summary: tasks may come from code generation,
logical puzzles, symbolic reasoning, instruction following and factual tasks with reliable grading; cross-domain pairs
need not include math. Choose a small, diverse set for cheap screening, then deepen the most informative comparisons.
Select for reliable verification, measurable starting success, learnability and informative interactions. Do not
assume that different domains conflict or similar ones cooperate. Distinguish competing behavioural objectives from
distinct skills. Arithmetic stays as one controlled setting, not the scope.
