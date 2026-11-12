# Environments

One folder per study. The general package list is `requirements.txt` in the repository root.

| File in `envs/<ID>/` | Contents |
|---|---|
| `babel-slurm.env` | Babel paths, pinned model revision, vLLM settings (`set -a; source <file>; set +a`) |
| `pip-freeze.txt` | full `pip freeze` of that study's environment (after its omegaconf patch) |

| Study | Folder |
|---|---|
| first study | [`Nov-12-2026-00-vineeth-babel-slurm-initial-study/`](Nov-12-2026-00-vineeth-babel-slurm-initial-study/) |
