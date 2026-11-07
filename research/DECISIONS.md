# Decisions

Dated decisions with their reasons. Newest last. Times are UTC.

## 2026-11-07 — setup

| # | Decision | Reason / evidence |
|---|---|---|
| D1 | Move `project.md` and `context/` from the harness directory to the project root. | `project.md` §2 names root `project.md` as canonical. On this case-insensitive disk, the harness file `PROJECT.md` and `project.md` are the same path, so they cannot coexist. |
| D2 | `context/` (project-history PDF) stays local and untracked (`.git/info/exclude`). | It is background, not protocol, and the GitHub remote may be public. |
| D3 | Authoritative branch `AE_BRANCH=main`. | The project repo had no commits and no other work to protect. |
| D4 | Isolation names: slug `qis`; expid prefix `qis-`; `AE_REMOTE_ROOT='$HOME/ae-qis'`; clone `~/projects/qwen-interference-study`; standbys `qis-dbg*`; tinker branches `qis-inv/*`; `PROJ_DATA=/data/user_data/$USER/qis`. | Another project (`dco`, `~/ae-dc-ogpo`) and older harness roots (`~/ae`, `~/ae-bc2rl`) share the account. `ae submit` detects duplicates by job name `ae-<expid>`, so a shared expid could collide. |
| D5 | No GPU standbys (`AE_DBG_POOL=0`, `AE_DBG_GEN_POOL=0`); at most 1 CPU standby. | `project.md` §3 forbids speculative GPU standbys. |
| D6 | All submissions go through `ops/qis-submit`: a lock plus a slot check (2 general, 6 preempt, 2 CPU jobs) over ledger-owned nonterminal jobs read from `squeue`. It fails closed when Slurm cannot be read. | `project.md` §3 "Scheduling" requires one nonterminal job per slot and a project-local lock. `ae submit` alone counts account-wide jobs. |
| D7 | Never use `ae submit --replace` or `ae cancel <expid>`; control jobs only with `ae slurm <op> <jobid>`. | Both name-based paths could match a same-named job of another project. `ae slurm` checks the ledger. |
