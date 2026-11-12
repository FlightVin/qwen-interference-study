# Studying interference in post-training

Experiments on how the mix of training problems in RL post-training of language models affects learning, including
whether progress on one kind of problem interferes with progress on another.

Results are in [`reports/`](reports/), one dated folder per study. Instructions given to the agents that ran each
study are archived in [`agent-project-instructions-archive/`](agent-project-instructions-archive/).

## Layout

| Path | Contents |
|---|---|
| `src/`, `tests/` | training, evaluation and analysis code; unit tests |
| `data/` | datasets, one dated folder per study ([`data/README.md`](data/README.md)) |
| `reports/` | reports, figures and tables ([`reports/README.md`](reports/README.md)) |
| `research/` | working records of each study ([`research/README.md`](research/README.md)) |

## Python environment

`.env` lists the Python version, the pinned packages and the environment variables.

```bash
set -a; source .env; set +a
python$PYTHON_VERSION -m venv "$VENV_DIR" && source "$VENV_DIR/bin/activate"
pip install $PIP_PACKAGES             # training and evaluation (Linux + CUDA GPU)
pip install $PIP_PACKAGES_REPORTS     # report figures only (CPU, any OS)
python -m pytest tests                # unit tests
```

Training and evaluation need one 48 GB GPU per job. Keep model weights, checkpoints and run outputs on large compute
storage, not in the repository.

---

P.S. If you work in Vineeth's environment (CMU Babel cluster), run jobs through the Slurm harness in
`slurm-agentic-engineering/`. It is a separate repository and git ignores it here.
