# DECISIONS — study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`

Purpose: dated decisions with their reasons. Times are US Eastern.

| # | Date | Decision | Reason |
|---|---|---|---|
| U1 | 2026-11-12 | The study's `project.md` is `agent-project-instructions-archive/<ID>/project-instructions.md`; STATE and CHECKLIST are in `research/<ID>/`. The harness files `PROJECT.md`, `state/STATE.md` and `state/CHECKLIST.md` point to them. | This is the layout the first study left (root `README.md`, HANDOVER §9). |
| U2 | 2026-11-12 | Measurement jobs run on `general` first (2 slots), `preempt` only for overflow. | `general` started within 2 minutes in the first study; `preempt` did not start within 1 hour (cluster card, 2026-11-07). |
| U3 | 2026-11-12 | Gradients are measured on fresh on-policy samples (16 answers per problem, two groups of 8), not on the stored training records. | Stored records give one group per problem and exist only for the training pools. Two groups per problem separate sampling noise from true alignment. Fresh samples also cover the control subjects and every checkpoint. |
| U4 | 2026-11-12 | Toy-model simulations and figure scripts for the reports run on the laptop. Model code runs only on Babel. | They are small numerical scripts with no model, data or GPU. The harness rule forbids project training and evaluation code on the laptop. |
