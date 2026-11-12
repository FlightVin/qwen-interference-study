# Project: why did we see no interference? (study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`)

Purpose: this is the `project.md` of the study. It holds the owner's mandate word for word, the operating rules that
follow from it, and every later owner amendment in order.

- **Results:** [`reports/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/`](../../reports/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/)
- **Working records (state, checklist, decisions):** [`research/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/`](../../research/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/)
- **Earlier study this one explains:** [`reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/`](../../reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/)

## 1. Mandate (owner message, 2026-11-12, verbatim)

> # Task: Start study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`
>
> ## Context
>
> Read these two reports first. They describe our earlier study and its open threads:
>
> - `reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/understanding-experiments.md`
> - `reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/further-things.md`
>
> Make sure you understand what we trained, on which domains, at which model scales, and what we measured. The key
> result is that we did not observe interference between domains. This new study asks why. Create project.md, state.md
> and checklist.md in correct places per your instructions and the existing QIS repo structure. Read agent.md both here
> and in the QIS repo structure.
>
> This study is mainly research and theory, not new training runs. Act as a careful, vetted research scientist. Read
> deeply, reason precisely, and write clearly. You may run code when it helps you answer a question precisely. For
> example, you can load existing model checkpoints, compute per-domain gradients, or measure similarity between
> gradients or weight updates. Do not start large training runs. Propose them instead. Your compute budget is 2 general
> GPUs and 6 pre-empt GPUs on CMU Babel.
>
> Create all outputs in `reports/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/`.
>
> ## Phase 1: Build the theory — write `00-initial-theory.md`
>
> 1. Read https://arxiv.org/abs/1904.11455 in full. Then find and read the related work that matters: the papers it
>    builds on, the papers that cite it, and the wider literature on interference in multi-task learning, continual
>    learning, and multi-domain fine-tuning of language models. Follow the citation trail as far as it is useful.
> 2. Write `00-initial-theory.md`. It should:
>    - Define interference precisely, and state each definition the literature uses.
>    - Give the mathematical justification for each main result. Show the key derivations, state their assumptions, and
>      say where those assumptions fail.
>    - Summarize the known empirical results in brief. Say which settings, model scales, and tasks each result comes
>      from.
>    - Include plots and diagrams. Generate your own figures with code where you can (for example, recreate a result
>      from reported numbers, or simulate a toy model). When you refer to a figure from a paper, cite it clearly.
>    - End with a short section: "What we know about interference." Separate what is proven, what is empirically
>      supported, and what is still conjecture.
>
> ## Phase 2: Explain our result — write `01-understanding.md`
>
> Develop hypotheses and open questions about why interference did not occur in our study. Start with these two
> directions, but do not limit yourself to them:
>
> 1. **Gradient alignment from pre-training.** Maybe gradient updates for our domains do not conflict at scale.
>    Pre-training may have made gradients across domains correlated, so updates for one domain also help, or at least do
>    not hurt, the others.
> 2. **Capacity.** Maybe larger models have enough capacity to absorb updates for many domains without conflict. Smaller
>    models may show interference because they have less capacity and smaller circuits, so updates must share the same
>    parameters.
>
> Also consider other explanations, including problems with our own setup. For example: the domains may be too similar,
> training may be too short or the learning rate too small to cause conflict, or our evaluation may not detect
> interference that is present.
>
> Structure `01-understanding.md` like this:
>
> 1. **Empirical understanding first.** Use our actual data and checkpoints. Measure what you can before you theorize.
>    Examples: cosine similarity of per-domain gradients, conflict rates between gradients, overlap of weight updates,
>    and how these change with model scale and training progress. Report exact numbers, the method, and the code
>    location. Also include mechanistic interpretability studies you think are worth doing, for example, which circuits
>    or parameters each domain's updates change, and whether they overlap.
> 2. **Theory second.** Build a precise theory that explains the measurements. Every claim must be correct. State all
>    assumptions. Mark clearly which parts are derived, which are supported by evidence, and which are speculation. If a
>    hypothesis is not consistent with the data, say so.
> 3. **Proposed experiments.** Design experiments that would separate the competing hypotheses. Include at least:
>    - Domain experiments: tasks with domains that differ much more strongly than ours.
>    - Scale experiments: post-training on smaller models to test if interference appears below some size.
>
>    For each experiment, state the hypothesis it tests, the predicted result under each hypothesis, the setup, and the
>    approximate compute cost.
>
> ## Phase 3: Update documentation
>
> Update the QIS project `README.md` files wherever needed so they point to this study and summarize its main
> conclusions.
>
> ## Standards
>
> - Be precise. Do not invent results, citations, or numbers. If you cannot verify something, say so.
> - Write for a technical reader who has not read the papers. Explain what matters and why. Keep detail where it
>   supports the argument, and cut it where it does not.
> - Use your own judgment on scope and direction. If you find a more important question than the ones above, follow it
>   and explain why.

## 2. Operating rules that follow from the mandate

| Rule | Value |
|---|---|
| GPU ceiling | 2 `general` GPUs and 6 `preempt` GPUs at one time, one GPU per job (`ops/qis-submit` enforces it) |
| CPU ceiling | 30 CPUs on `preempt_cpu_qos` (carried over from the first study) |
| Training | no training runs. Allowed: sampling answers from stored checkpoints, gradients, forward passes, weight comparisons |
| Checkpoints of the first study | read-only. Nothing is written under `$PROJ_DATA/ckpt` or `$PROJ_DATA/runs` |
| New compute outputs | `$PROJ_DATA/ui/` on Babel (`ui` = understanding interference) |
| Experiment IDs | `qis-ui-*` |
| Code | `src/qis/interf/` (measurements), `reports/<ID>/build/` (figures, toy models) |
| Records of the first study | frozen. This study writes only in its own `<ID>` folders, `src/`, and the README files |
| Writing | the owner's writing standard (`slurm-agentic-engineering/agent.md`, "Writing standard"); times in US Eastern time |
| Citations and numbers | every citation is checked against the source; every number names its table or job |

## 3. Facts about the first study that this study starts from

- One model scale only: Qwen3-1.7B (non-thinking), full-parameter GRPO, learning rate 2e-6, 512 AdamW steps per run.
  The mandate asks about "model scales"; the first study has no second scale. Scale questions are therefore answered
  from the literature, from gradient measurements on other Qwen3 sizes if feasible, and by proposed experiments.
- Three training domains: hard algebra problems that need factoring (H), direct factoring exercises (R), direct
  modular-arithmetic exercises (U). Six runs: arms H, R, U × seeds 101, 202.
- Stored weights: base model; round 128 of all six runs; round 64 of R-s101 and H-s202; a round-116 copy of H-s202.

## 4. Owner amendments

None yet.
