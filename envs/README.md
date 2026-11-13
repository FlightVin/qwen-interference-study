# Environments

One folder per study. The general package list is `requirements.txt` in the repository root.

| File in `envs/<ID>/` | Contents |
|---|---|
| `babel-slurm.env` | Babel paths, pinned model revision, vLLM settings (`set -a; source <file>; set +a`) |
| `pip-freeze.txt` | full `pip freeze` of that study's environment (after its omegaconf patch) |

| Study | Folder |
|---|---|
| first study | [`Nov-12-2026-00-vineeth-babel-slurm-initial-study/`](Nov-12-2026-00-vineeth-babel-slurm-initial-study/) |
| second study (understanding interference) | no folder of its own: it uses the first study's environment (`venv-v1`) and settings unchanged. Its job recipes set the paths: `research/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/recipes/` |
