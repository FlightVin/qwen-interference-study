# CHECKLIST — study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`

Purpose: the live plan. Status: **done** / **active** / **blocked** / **pending**. `<ID>` is the study ID above.

## 0. Owner instructions (NON-NEGOTIABLE)
- 0.1 No training runs. Propose them instead. Sampling, gradients and forward passes on stored checkpoints are allowed.
- 0.2 Ceilings: 8 general GPUs (owner amendment 2026-11-12, until told otherwise; supersedes the 2 of the mandate),
      6 preempt GPUs, one GPU per job; guard `ops/qis-submit`.
- 0.3 All outputs in `reports/<ID>/`. Deliverables: `00-initial-theory.md`, `01-understanding.md`, updated READMEs.
- 0.4 Do not invent results, citations or numbers. Mark what is unverified.
- 0.5 Records of the first study stay frozen. Control only this project's jobs. Times in EST. Owner writing standard.

## 1. Setup
- 1.1 Read both reports, HANDOVER, the training code, the checkpoint inventory. [done]
- 1.2 Create `<ID>` folders; write project instructions, STATE, CHECKLIST, DECISIONS. [done]
- 1.3 Point the harness `PROJECT.md`, `state/STATE.md`, `state/CHECKLIST.md` at this study; set the guard to 2/6. [active]

## 2. Phase 1 — theory (`reports/<ID>/00-initial-theory.md`)
- 2.1 Literature digests: ray interference and RL; multi-task; continual learning; LLM fine-tuning. [active]
- 2.2 Verify every citation and formula used against the source. [pending, needs 2.1]
- 2.3 Figures by code: ray-interference toy model; near-orthogonality in high dimension; second-order step analysis;
      linear-model forgetting versus similarity and width. [pending]
- 2.4 Write the document; end with "What we know about interference" (proven / supported / conjecture). [pending]

## 3. Phase 2 — measurements (code `src/qis/interf/`, outputs `$PROJ_DATA/ui/`)
- 3.1 Weight-update geometry: cosine between run updates, per-layer and per-module norms, rank, sign agreement. [pending]
- 3.2 On-policy samples at base, R-s101 r64, and round-128 checkpoints: pools hard, related, unrelated, control. [pending]
- 3.3 Gradient geometry: per-group GRPO gradients, within- and cross-domain cosines, conflict rates, noise-corrected
      domain cosines, Adam-preconditioned version where optimizer state exists. [pending, needs 3.2]
- 3.4 Linearity test and surrogate transfer matrix: change in log-probability of base samples under each trained model
      versus the first-order prediction. [pending, needs 3.2]
- 3.5 Scale: the same gradient geometry on other Qwen3 sizes at initialisation, if the weights are obtainable. [pending]
- 3.6 Canary first for each new job family; verify outputs; recompute a sample of numbers independently. [pending]

## 4. Phase 2 — writing (`reports/<ID>/01-understanding.md`)
- 4.1 Empirical section with exact numbers, method and code location. [pending, needs 3]
- 4.2 Theory section; each claim marked derived / supported / speculation. [pending]
- 4.3 Proposed experiments: domain contrast, scale, coupling; predictions per hypothesis; compute cost. [pending]

## 5. Phase 3 — documentation and close
- 5.1 Update root `README.md`, `reports/README.md`, `research/README.md`, archive README, `envs/README.md`. [pending]
- 5.2 Audit: every number in both documents against its table; every citation against its source. [pending]
- 5.3 HANDOVER for this study; final STATE; commit. [pending]
