# Research records

One folder per study, named by date, owner and compute environment. Each folder holds the working records of the agents
that ran the study: protocol, definitions, decisions, audits and raw evidence. Final results are in `reports/`.

| Folder | Study | Status |
|---|---|---|
| [`Nov-12-2026-00-vineeth-babel-slurm-initial-study/`](Nov-12-2026-00-vineeth-babel-slurm-initial-study/) | GRPO with auxiliary math practice, Qwen3-1.7B | finished; records frozen; read its [`HANDOVER.md`](Nov-12-2026-00-vineeth-babel-slurm-initial-study/HANDOVER.md) |
| [`Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/`](Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/) | why the first study saw no interference: theory, measurements on its checkpoints, capacity, and candidate domain mixtures (no training) | six documents written (`00`–`05`); read its [`HANDOVER.md`](Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/HANDOVER.md) for the state of the last jobs |

## Harness workspace archive

`harness-workspace-archive-2026-11-15.tgz` holds the laptop-only harness workspace as it stood when the owner reset
it on 2026-11-15: `PROJECT.md`, `state/` (STATE, CHECKLIST, DECISIONS, IMPLEMENTATION-LOG, FINDINGS), all `exp/`
recipes, the cluster card and `config.sh`. Harvested `results/` are not included; their numbers are in `reports/`.
Babel storage of the project was deleted on 2026-11-13. New studies start from a clean state and keep their records
here, under `research/<study-ID>/`.
