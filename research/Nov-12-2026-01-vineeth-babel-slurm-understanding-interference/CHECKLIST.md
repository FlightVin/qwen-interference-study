# CHECKLIST — study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`

Purpose: the live plan. Status: **done** / **active** / **blocked** / **pending**. `<ID>` is the study ID above.

## 0. Owner instructions (NON-NEGOTIABLE)
- 0.1 No training runs. Propose them instead. Sampling, gradients and forward passes on stored checkpoints are allowed.
- 0.2 Ceilings: 8 general GPUs (owner amendment 2026-11-12, until told otherwise; supersedes the 2 of the mandate),
      6 preempt GPUs, one GPU per job; guard `ops/qis-submit`.
- 0.3 All outputs in `reports/<ID>/`. Deliverables: `00-initial-theory.md`, `01-understanding.md`, updated READMEs.
- 0.4 Do not invent results, citations or numbers. Mark what is unverified.
- 0.5 Records of the first study stay frozen. Control only this project's jobs. Times in EST. Owner writing standard.
- 0.6 Gradient geometry on Qwen3-0.6B and Qwen3-4B is out of scope and is future work (owner amendment 2026-11-12).
- 0.7 The finished 16-answer Qwen3-0.6B run is removed from the report (owner amendment 2026-11-12, 23:00 EST).
- 0.8 New deliverables (same amendment): capacity work (literature first; run only experiments that discriminate, or
      propose) included in `01` before it is final; `02` and `03` = 10–15 minute synopses of `00` and `01` with figures;
      `04-domains-to-explore.md` (domain mixtures for ray interference on Qwen3-1.7B); `05-summary.md` (10 minutes).

## 1. Setup
- 1.1 Read both reports, HANDOVER, the training code, the checkpoint inventory. [done]
- 1.2 Create `<ID>` folders; write project instructions, STATE, CHECKLIST, DECISIONS. [done]
- 1.3 Point the harness `PROJECT.md`, `state/STATE.md`, `state/CHECKLIST.md` at this study; guard at 8/6. [done]

## 2. Phase 1 — theory (`reports/<ID>/00-initial-theory.md`)
- 2.1 Literature digests: ray interference and RL; multi-task; continual learning; LLM fine-tuning. [done]
- 2.2 Verify citations and formulas against the sources; label each reference R / A / NV. [done]
- 2.3 Figures by code (`build/theory_figs.py`, T1–T9). [done]
- 2.4 Write the document; end with "What we know about interference". [done]

## 3. Phase 2 — measurements (code `src/qis/interf/`, outputs `$PROJ_DATA/ui/`)
- 3.1 Weight-update geometry (`qis-ui-wdelta`, job 11130053). [done]
- 3.2 On-policy samples and gradient geometry: base (16 and 64 answers), R-s101 r64, H-s202 r64, R-s202 r128,
      U-s202 r128, H-s101 r128; Adam-scaled where optimizer state exists. [done]
- 3.3 Surrogate transfer matrix and response along the training direction (`qis-ui-lp`, job 11130858). [done]
- 3.4 Averaged checkpoints: log-probability effect [done]; dev accuracy (`qis-ui-soup` 11130401 running,
      `qis-ui-soup-b` 11131755 pending). [active]
- 3.5 Precision test: FP32 against BF16-rounded weights (`qis-ui-lpprec`, job 11132332); `tables/lpprec.json`. [done]
- 3.6 Scale: Qwen3-0.6B at 16 answers [done]. The 64-answer runs on 0.6B and 4B (jobs 11131757, 11131734) were
      cancelled while pending: out of scope by owner amendment 0.6, listed as future work in M1. [closed]
- 3.7 Canary for the job family (`qis-ui-canary`) [done]; sketch norm check 0.998–1.002 [done]; estimator test on
      synthetic data [done].

## 4. Phase 2 — writing (`reports/<ID>/01-understanding.md`)
- 4.1 Empirical section with exact numbers, method and code location; mechanistic studies. [done, one place pending 3.4]
- 4.2 Theory section; each claim marked derived / supported / speculation; verdict per hypothesis. [done]
- 4.3 Proposed experiments E1–E5 and M1 with predictions per hypothesis and compute cost. [done]
- 4.4 Section 1.6 filled with the precision result [done]. Fill Section 1.5 when 3.4 finishes. [pending, needs 3.4]

## 5. Phase 3 — documentation and close
- 5.1 Update root `README.md`, `reports/README.md`, `research/README.md`, archive README, `envs/README.md`. [done]
- 5.2 Audit: numbers in `01-understanding.md` against the tables (one pass done on 2026-11-12; corrections applied).
      Repeat after 4.4. [active]
- 5.3 HANDOVER, STATE, recipes copied to `research/<ID>/recipes/`; commit. [done; repeat after 4.4]

## 6. Capacity (owner amendment 0.8)
- 6.1 Literature review: intrinsic dimension, rank and sparsity of updates, task vectors, superposition, scale. [active]
- 6.2 Decide which measurements on stored checkpoints discriminate; run those, propose the rest. [pending, needs 6.1]
- 6.3 Write the capacity section of `01-understanding.md`; remove the 0.6B run (0.7). [pending, needs 6.2]

## 7. Domain mixtures for ray interference (`reports/<ID>/04-domains-to-explore.md`)
- 7.1 Literature and dataset survey: reported negative transfer, candidate datasets, published small-model scores. [active]
- 7.2 Measure what is cheap and decisive on Qwen3-1.7B base (pass rates; gradient coupling of a shortlist). [pending, needs 7.1]
- 7.3 Write `04-domains-to-explore.md`. [pending]

## 8. Synopses and summary
- 8.1 `02` synopsis of `00-initial-theory.md`. [pending]
- 8.2 `03` synopsis of `01-understanding.md`. [pending, needs 4.4 and 6.3]
- 8.3 `05-summary.md`. [pending, needs 7.3, 8.1, 8.2]
- 8.4 README updates, number audit, HANDOVER, commit. [pending]
