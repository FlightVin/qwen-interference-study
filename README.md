# Studying interference in post-training

Experiments on how the mix of training problems in RL post-training of language models affects learning, including
whether progress on one kind of problem interferes with progress on another.

Results are in [`reports/`](reports/), one folder per study. Instructions given to the agents that ran each study are
archived in [`agent-project-instructions-archive/`](agent-project-instructions-archive/).

## Studies and their IDs

Each study has one ID. The same ID names its folder in `reports/`, `data/`, `research/` and its files in `envs/`.
The ID has the form `<date>-<number>-<owner>-<compute>-<label>`:

| ID | Meaning |
|---|---|
| `Nov-12-2026-00-vineeth-babel-slurm-initial-study` | the first study (`00`), finished on Nov 12, 2026, run by Vineeth on the CMU Babel Slurm cluster |

## Layout

| Path | Contents |
|---|---|
| `src/`, `tests/` | training, evaluation and analysis code; unit tests |
| `data/<ID>/` | datasets of a study ([`data/README.md`](data/README.md)) |
| `reports/<ID>/` | reports, figures and tables ([`reports/README.md`](reports/README.md)) |
| `research/<ID>/` | working records and handover notes ([`research/README.md`](research/README.md)) |
| `envs/` | per-study runtime settings and exact package locks ([`envs/README.md`](envs/README.md)) |
| `requirements.txt` | Python packages |

## Python installation

Use Python 3.12 on Linux with a CUDA GPU for training and evaluation. Report figures need only a CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest tests
```

Keep model weights, checkpoints and run outputs on large compute storage, not in the repository.

---

P.S. If you work in Vineeth's environment (CMU Babel cluster), run jobs through the Slurm harness in
`slurm-agentic-engineering/`. It is a separate repository and git ignores it here. Load the study settings with
`set -a; source envs/<ID>.env; set +a` on a compute node.
