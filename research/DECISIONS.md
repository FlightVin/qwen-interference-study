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
| D8 | Run CPU jobs at 2 CPUs while the account CPU QOS is saturated; reduced pending jobs 11072074 and 11072095 from 4 to 2 CPUs (`ae slurm update NumCPUs=2`). Recipes still say `cpus=4`. | `squeue`: another project holds 62/64 CPUs on `preempt_cpu_qos`; our job pended on `QOSMaxCpuPerUserLimit`. Moving to a general GPU slot would idle a GPU. No scientific impact. |
| D9 | Algebra source = MATH `algebra` + `intermediate_algebra`. Role pools fixed by rubric before screening: H_train/targets = factorization central, level 4–5; E_related = factorization central and the direct task, level 1–3; E_unrelated = number-theory modular arithmetic, factorization absent. Keyword prefilters choose what is annotated. | `project.md` §5 says "start with algebra problems"; intermediate algebra holds most polynomial-factoring work. Fixing roles by MATH level (author difficulty, not model success) keeps target selection independent of measured success. |

## 2026-11-07 — trainer design and addendum A1

| # | Decision | Reason / evidence |
|---|---|---|
| D10 | Build a lean single-GPU GRPO trainer (`src/qis/train/`): vLLM 0.31.0 in-process rollout (BF16, sleep mode), HF transformers learner with FP32 parameters + BF16 autocast, torch AdamW. Use verl 0.9.0 only as a reference: unit tests compare our loss reduction with verl's `agg_loss` (`seq-mean-token-mean`). | `research/VERL-AUDIT.md`: on verl 0.9.0 our spec needs a custom advantage (built-in uses Bessel std, 0.935× at n=8), custom minibatch ordering (TransferQueue order unknown), a custom policy loss (dual-clip always on; `pg_clipfrac_lower` is the dual-clip fraction), a custom reward manager (truncation not visible; no verifier timeout), custom logprob quantiles and a resume sidecar. With most of the algorithm custom, Ray/TransferQueue add risk without adding verified semantics. `project.md` §4 states a preference for verl, not a requirement. |
| D11 | pi_rollout in the ratio = teacher-forced log probs of the round-start learner (FP32 weights, BF16 autocast), computed once per round and fixed for its 4 steps. vLLM sampling log probs are stored per round and compared with them (quantiles logged); large mismatches are investigated, not clipped away. | The rollout policy and the round-start learner have identical parameters; their only difference is numerics (BF16 serving vs learner path). The first step's ratio is then exactly 1, so clipping measures real policy change. Same choice as verl's default `old_log_prob` recompute. |
| D12 | Addendum A1 (precision audit, report figures) applied 2026-11-07. Affected stages: trainer gate G2 (adds the precision audit), dataset/evaluation planning (adds fixed J_R/J_U probe panels, frozen before any probe result is seen), local diagnostic (adds ΔJ on common panels), report (four figures). Affected runs: none so far. Completed jobs (`qis-smoke*`, `qis-env-v1`) ran no optimizer; pending `qis-probe-gpu-v1` (11072118) keeps FP32 parameters + BF16 autocast and FP32 AdamW state, which conforms, so it continues. No attempt is superseded. | Owner message 2026-11-07; `project.md` §15. |
