# DECISIONS — study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`

Purpose: dated decisions with their reasons. Times are US Eastern.

| # | Date | Decision | Reason |
|---|---|---|---|
| U1 | 2026-11-12 | The study's `project.md` is `agent-project-instructions-archive/<ID>/project-instructions.md`; STATE and CHECKLIST are in `research/<ID>/`. The harness files `PROJECT.md`, `state/STATE.md` and `state/CHECKLIST.md` point to them. | This is the layout the first study left (root `README.md`, HANDOVER §9). |
| U2 | 2026-11-12 | Measurement jobs run on `general` first (2 slots), `preempt` only for overflow. | `general` started within 2 minutes in the first study; `preempt` did not start within 1 hour (cluster card, 2026-11-07). |
| U3 | 2026-11-12 | Gradients are measured on fresh on-policy samples (16 answers per problem, two groups of 8), not on the stored training records. | Stored records give one group per problem and exist only for the training pools. Two groups per problem separate sampling noise from true alignment. Fresh samples also cover the control subjects and every checkpoint. |
| U4 | 2026-11-12 | Toy-model simulations and figure scripts for the reports run on the laptop. Model code runs only on Babel. | They are small numerical scripts with no model, data or GPU. The harness rule forbids project training and evaluation code on the laptop. |
| U5 | 2026-11-12 | Gradient samples cover four pools: hard, related, unrelated (training pools of `bundle_main_v1.json`) and the 64-problem control panel. | The training pools are what the runs saw. The control panel tests whether untrained subjects share the gradient direction. |
| U6 | 2026-11-12 | The main base-model run uses 64 answers per problem (8 groups). Trained checkpoints use 16. | 16 answers gave intervals too wide to compare domain pairs. Trained checkpoints generate about 9 times slower, so 64 answers there would take about 10 GPU-hours each; proposed, not run. |
| U7 | 2026-11-12 | Averaged checkpoints are evaluated on the first study's dev panel (53 problems, 32 answers), not the sealed test. | The dev panel was already used during training of the first study. The sealed test stays reserved. |
| U8 | 2026-11-12 | Scale is probed only by gradient measurements at initialisation on Qwen3-0.6B and Qwen3-4B (downloaded from the Hugging Face hub). | The mandate forbids training. The weights are public. The limits of this probe are stated in `01-understanding.md` §1.7. |
| U9 | 2026-11-12 | The cosine matrix of `qis-ui-geom-base64` (4.3 MB) was moved to `~/ae-qis/results-large/qis-ui-geom-base64/` on Babel and not harvested. | It exceeds the 2 MB harvest limit. The limit was not raised (harness rule: owner's instruction only). The summaries were harvested. |
| U10 | 2026-11-12 | `qis-ui-soup-b` and `qis-ui-geom-q06x64` run on `preempt`. | `general` was full at submission; the mandate allows 6 preempt GPUs; both recipes resume after a requeue. |
