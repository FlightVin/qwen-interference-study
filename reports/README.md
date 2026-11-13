# Reports

| Date | Folder | Start here |
|---|---|---|
| Nov 12, 2026 | [`Nov-12-2026-00-vineeth-babel-slurm-initial-study/`](Nov-12-2026-00-vineeth-babel-slurm-initial-study/) | [Understanding the experiments](Nov-12-2026-00-vineeth-babel-slurm-initial-study/understanding-experiments.md) (5–10 min) → [Further things](Nov-12-2026-00-vineeth-babel-slurm-initial-study/further-things.md) → [Full experiments report](Nov-12-2026-00-vineeth-babel-slurm-initial-study/experiments-report.md) |
| Nov 12, 2026 | [`Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/`](Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/) | [Why no interference: measurements, theory, proposed experiments](Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/01-understanding.md) (start with its Summary) ← background: [Interference: definitions, theory and evidence](Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/00-initial-theory.md) |

**Conclusions of the second study.** The three math domains did not interfere because their GRPO gradients do not
oppose each other: the cross-domain gradient cosine is small and positive, and each trained run raised the objective
of every domain. Most of each weight update is run-specific sampling noise; a shared 4% carries most of the effect.
The study proposes experiments with opposing domains, smaller and larger models, and unsolved hard problems.

Older records that cite `reports/REPORT.md`, `reports/figures/` or `reports/tables/` refer to files now under
`Nov-12-2026-00-vineeth-babel-slurm-initial-study/` (`REPORT.md` is now `experiments-report.md`). Moved on Nov 12, 2026.
