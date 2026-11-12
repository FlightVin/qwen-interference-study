# Environments

Per-study runtime settings and exact package locks. The general package list is `requirements.txt` in the repository root.

| File | Contents |
|---|---|
| `Nov-12-2026-00-vineeth-babel-slurm-initial-study.env` | Babel paths, pinned model revision, vLLM flags (`set -a; source <file>; set +a`) |
| `Nov-12-2026-00-vineeth-babel-slurm-initial-study.freeze.txt` | full `pip freeze` of that study's environment (after its omegaconf patch) |
