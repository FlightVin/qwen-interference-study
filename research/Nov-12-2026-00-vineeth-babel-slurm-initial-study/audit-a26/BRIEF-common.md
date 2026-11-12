# A26 audit — common brief for every audit agent

Purpose: shared facts and rules for the agents of audit pass A and pass B (owner instruction A26). Read it fully.

## The study in one paragraph
QIS trained Qwen3-1.7B (non-thinking) with full-parameter GRPO for 128 rounds. Three arms × two training seeds (101, 202)
give six runs, `qis-main-{H,R,U}-s{101,202}-sym`. Every round has 16 prompt slots × 8 attempts. Slots 0–7 are shared hard
problems from H_train (46 problems); slots 8–15 are more hard problems (H), direct factoring practice (R) or direct
modular-arithmetic practice (U), R and U drawn as 63 matched pairs. Policies were evaluated in-run on dev (53 problems; J_H =
its first 32) and probes J_R (28), J_U (32); at round 128 on the sealed test (128 problems × 32 attempts); and on stress
panels, a cue study, a length extension (12,288 → 16,384 tokens) and a local one-step update diagnostic.
**Use the terms exactly as defined in `research/DEFINITIONS.md` v1.** If you find a definition wrong, say so with evidence.

## Locations
- Repo (laptop): `/Users/vineethbhat/Desktop/Courses/deep-rl/project/qwen-interference-study`. Code `src/qis/`, tests
  `tests/`, data manifests `data/`, protocol `research/PROTOCOL.md`, definitions `research/DEFINITIONS.md`.
- Job ledger: `slurm-agentic-engineering/state/LEDGER.jsonl` (expid, jobid, sha, expsha, submitted). Recipes:
  `slurm-agentic-engineering/exp/<expid>.sh` (some old recipes are in `slurm-agentic-engineering/exp-deferred/` or absent).
- Compute data root `$P = /data/user_data/vineethb/qis` (mounted on compute nodes only). Main places:
  `runs/<expid>/` (metrics.jsonl, records/train_rNNNN.jsonl.gz, eval records, superseded segments), `ckpt/<expid>/`,
  `test-eval/`, `test-eval-cgpu/`, `stress-eval/`, `length-diag/`, `length-test/`, `cue-eval/`, `diag*/`, `data/`.
  Discover with `ls`; do not assume.
- Slurm logs: `slurm-agentic-engineering/bin/ae logs <expid>` (never cat a whole log).

## How to run things on compute (read-only on project data)
1. Find a RUNNING CPU standby: `ssh babel 'squeue -u vineethb -n qis-dbg-cpu -h -o "%i %T"'`.
2. Copy your script: `scp myscript.py babel:ae-qis/audit-a26/<your-agent-name>/` (home is shared with compute nodes).
3. Run it: `ssh babel "srun --jobid=<ID> --overlap bash -lc '/data/user_data/vineethb/qis/env/venv-v1/bin/python ~/ae-qis/audit-a26/<your-agent-name>/myscript.py ...'"`.
   The venv has numpy, torch, transformers. Standbys have 4 CPUs and limited RAM: stream `.jsonl.gz` files line by line.
4. Write compute-side outputs only under `$P/audit-a26/<your-agent-name>/` (create it). Never modify anything else in `$P`.
5. Keep local copies of every script you run under the repo path your brief names, so the audit is reproducible.

## Rules
- Never run project code (`src/qis`, tests) on the laptop or the login node. Stdlib-only scripts that read repo data files on
  the laptop are fine. Running project code on compute is allowed only where your brief says so.
- Do not submit, cancel, hold or update Slurm jobs unless your brief explicitly allows it. Do not commit or push.
- Edit only the files your brief names. Never edit `research/DEFINITIONS.md`, the report, the archive or other agents' files.
- Label every finding **verified** (you checked primary evidence), **claimed** (only a document says so) or **unknown**.
  Give the evidence path, command and count for each. Write "unavailable" for pruned or missing files; never infer
  integrity from a copied hash.
- Count with scripts, not by hand. Give numbers with denominators. Keep unrounded values in JSON next to each markdown table.
- Times in EST (UTC in brackets is fine). Writing: short direct sentences, tables for data, no filler.
