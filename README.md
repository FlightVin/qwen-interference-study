# Studying interference in post-training

Experiments on how the mix of training problems in RL post-training of language models affects learning, including
whether progress on one kind of problem interferes with progress on another.

Agents: start with [`AGENTS.md`](AGENTS.md). Results are in [`reports/`](reports/), one folder per study. Instructions given to the agents that ran each study are
archived in [`agent-project-instructions-archive/`](agent-project-instructions-archive/).

## Studies and their IDs

Each study has one ID. The same ID names its folder in `reports/`, `data/`, `research/`, `envs/` and `agent-project-instructions-archive/`.
File names inside those folders do not repeat the ID or the date.
The ID has the form `<date>-<number>-<owner>-<compute>-<label>`:

| ID | Meaning |
|---|---|
| `Nov-12-2026-00-vineeth-babel-slurm-initial-study` | the first study (`00`), finished on Nov 12, 2026, run by Vineeth on the CMU Babel Slurm cluster |
| `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference` | the second study (`01`), Nov 12, 2026: why the first study saw no interference. Theory and measurements on stored checkpoints; no training |

## Main conclusions so far

| Study | Conclusion | Read |
|---|---|---|
| `00` initial study | GRPO training raised Qwen3-1.7B by 20–29 points on the sealed test in every arm. Which practice filled the other slots mattered little, and its effect flipped with the seed. No evidence of interference. | [guide](reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/understanding-experiments.md) |
| `01` understanding interference | The domains did not interfere because their training signals did not oppose each other. Gradients of different problems are almost orthogonal with a small positive mean (cosine 0.0003–0.0007 across domains, conflict rate 49–50%). Every run moved every domain in the helpful direction. About 96% of each run's weight update is run-specific noise; the shared 4% carries most of the effect and formed in rounds 1–64. Capacity is not needed to explain the result and is untested (one model size). The preconditions of ray interference were absent. Some results are still pending; see the document. | [theory](reports/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/00-initial-theory.md) → [explanation and proposed experiments](reports/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/01-understanding.md) |

## Layout

| Path | Contents |
|---|---|
| `src/`, `tests/` | training, evaluation and analysis code; unit tests. `src/qis/interf/` holds the gradient and weight-update measurements of study `01` |
| `data/<ID>/` | datasets of a study ([`data/README.md`](data/README.md)) |
| `reports/<ID>/` | reports, figures and tables ([`reports/README.md`](reports/README.md)) |
| `research/<ID>/` | working records and handover notes ([`research/README.md`](research/README.md)) |
| `envs/<ID>/` | runtime settings and exact package lock of a study ([`envs/README.md`](envs/README.md)) |
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
`set -a; source envs/<ID>/babel-slurm.env; set +a` on a compute node.
