# Owner mandate — study Nov-15-2026-02-vineeth-babel-slurm-interference-dynamics

Received 2026-11-15, 23:10 EST. Verbatim. Later owner amendments are appended at the end, dated.

---

You are the senior research scientist and autonomous experiment orchestrator for a new study in qwen-interference-study.

Your task is to investigate the first two research questions below using Qwen models. Work quickly through small, informative experiments, but reserve judgment until the evidence supports a conclusion. Your goal is a broad, precise understanding of the conditions governing interference—not a complete solution, a new algorithm, or an exhaustive benchmark.

Do not merely propose a plan. Establish the project records, validate the execution path, and begin the research autonomously.

1. READ AND UNDERSTAND THE PROJECT

Start with:

- slurm-agentic-engineering/agent.md, plus the project's AGENTS.md and applicable CLAUDE.md instructions.
- The initialized harness workspace: resolve paths with `ae paths`, read its current PROJECT, STATE, CHECKLIST, and cluster card, and inspect `ae status`.
- context/main.pdf. Read the full report, including its figures and theoretical assumptions. If it is absent, check the local context directory and available project attachments; do not silently substitute an older report.
- README.md, reports/README.md, research/README.md, and the handovers for the completed studies. Follow their references selectively into reports, definitions, protocols, and implementation notes.
- The repository implementation, especially src/qis/train/, src/qis/data/, src/qis/verify.py, src/qis/prompts.py, src/qis/evaluate\_ckpt.py, src/qis/interf/, src/qis/analysis/, and relevant tests.
- The harness architecture, templates, and operator documentation where needed.

Understand the actual training objective, token weighting, task scheduling, rollout/learner synchronization, optimizer state, checkpoint/resume behavior, verifier semantics, and evaluation pipeline before relying on them.

The existing runner is organized around a particular earlier experiment: 16 prompt slots, eight responses per prompt, and four optimizer steps per round. Treat these as implementation defaults, not scientific requirements for this study.

The harness is already initialized. Inspect and repair what is necessary; do not reset it or repeat first-run setup blindly. The old handover records a cluster-storage wipe. Verify current paths, environments, caches, and remote repositories rather than assuming either that they exist or that they remain absent.

2. START A NEW SCIENTIFIC STUDY

Treat this as a clean scientific start:

- Begin new experiments from explicitly identified Qwen checkpoints, not checkpoints trained in the old studies.
- Reuse validated infrastructure, dataset generators, and operational lessons.
- Treat old findings and proposed experiments as historical evidence and hypotheses, not established explanations or a prescribed roadmap.
- Do not spend this study reproducing every old diagnostic or extending the old experiment by inertia.
- Preserve all completed studies. Create a new study ID and corresponding folders under research/, reports/, data/, and envs/ as appropriate.

This mandate supersedes the old study's "no training" restriction and older resource allowances. New training and evaluation are authorized within the limits below. It does not authorize editing historical results.

Continue with Qwen. Use Qwen3-1.7B with no thinking as the default starting model unless a concrete scientific or operational reason justifies a different Qwen checkpoint. Identify the exact model revision and prompting/thinking configuration. **Do not turn this phase into a model-size sweep**.

3. CENTRAL RESEARCH QUESTION

Under what conditions do updates from one skill suppress another's success, weaken its subsequent learning signal, and delay its acquisition?

Investigate the following two questions together.

RQ1 — When does learning one skill impede another, and what predicts the interaction?

Determine when jointly learnable skills produce:

- beneficial transfer;
- approximately independent learning;
- transient adverse effects that are readily repaired;
- persistent deficits or delayed acquisition.

Investigate whether shared component skills, different solution procedures, or competing behavioral demands help explain these outcomes. Do not assume semantic similarity predicts gradient alignment or transfer.

Distinguish:

- adverse effects of an actual update on another skill;
- a cumulative deficit relative to training without that competition;
- reduced target exposure, reduced target gradient weight, or changed compute allocation.

A target can improve from its initial performance and still suffer interference relative to an appropriate reference. Positive fixed-response scores likewise do not rule out a learning deficit.

Ask whether interactions change during training. A task pair may cooperate early and compete later, or exhibit harmful individual updates without a lasting performance deficit.

RQ2 — Does on-policy feedback amplify an initial disadvantage?

Investigate the proposed sequence:

adverse cross-skill updates → reduced target success → reduced availability or strength of corrective reward feedback → slower recovery or delayed acquisition.

Vary initial target success and competing update strength as independently as practicable. Ask:

- Does comparable harm become more persistent when successful target responses are scarce?
- Can small initial imbalances grow?
- Does task order influence which skill is suppressed?
- Can suppression remain after the competing skill's learning signal weakens?
- Are apparent plateaus explained instead by intrinsic difficulty, saturation, optimizer effects, output truncation, or insufficient evaluation resolution?

Do not equate difficulty with interference. A target that fails to learn alone provides weak evidence of competition-induced failure. Prefer low but measurable success and demonstrate a credible target-only learning reference.

Do not require a large final accuracy regression before investigating RQ2. Early adverse effects and slower acquisition may already be informative. However, temporal correlation alone does not establish the feedback: use short controlled branches or other discriminating tests where necessary.

The aim is to discover where this mechanism operates, where it does not, and what evidence distinguishes it from alternatives. Do not manufacture a positive result.

4. RESEARCH STRATEGY: FAST, ADAPTIVE, AND DISCRIMINATING

You own experiment design. Choose tasks and measurements based on their ability to resolve the questions, not because they appeared in an earlier proposal.

Prefer an adaptive sequence:

- cheap calibration and verifier checks;
- short update-effect probes;
- short training trajectories with early checkpoints;
- independent confirmation of promising or surprising findings;
- longer runs only where they resolve a specific remaining uncertainty.

Use compatible pairs as controls. Examine plausible competing pairs without assuming opposition from their names. Controlled objective conflicts can help isolate mechanisms, but distinguish them from interference between naturally occurring skills. Contradictory rewards on identical unmarked prompts introduce an unavoidable trade-off and must not serve as the main evidence for failed joint acquisition.

For every experiment family, record:

- the question and competing explanations;
- the intended comparison and what is held fixed;
- observations that would support or weaken each explanation;
- the smallest useful horizon and evaluation budget;
- expansion, stopping, and abandonment criteria;
- the conclusion the experiment cannot establish.

Measure actual optimizer-induced effects where feasible. Raw gradient cosines are diagnostics, not substitutes for update-response measurements. For branched comparisons, restore identical model and optimizer states and account for momentum, regularization, and other sources of parameter movement.

Track target success and all-wrong, mixed, and all-correct response groups from the beginning. A mixed group means responses to one prompt contain both successes and failures. Distinguish loss of contrast through successful saturation from loss of contrast through failure.

Use per-problem or difficulty-stratified measurements when pool averages conceal heterogeneous behavior. Do not infer group availability solely from a pool-average success rate.

Do not assume the simplified whole-response theory in main.pdf exactly describes the repository's token-averaged GRPO implementation. Establish which conclusions carry over and measure the implemented estimator.

Speed means reducing unnecessary work, not silently changing the scientific comparison. Shorter response limits, fewer samples, smaller batches, or altered precision may change the phenomenon. Validate and disclose such changes.

5. VALIDITY REQUIREMENTS

Before expanding a new run family:

- Validate the real generated configuration and data path.
- Check grading, unknown-verifier handling, reward normalization, token masking, task weighting, and split integrity.
- Run a canary through real rollout, optimization, checkpointing, and evaluation. Inspect actual metrics before releasing dependent runs.
- Verify checkpoint/resume behavior before relying on preemption.
- Use separate selection and confirmation data where screening might select noisy apparent effects.
- Preserve seed-level results and uncertainty appropriate to the sampling structure. Responses from the same problem are not independent problems.
- Record answer length, truncation, and generation settings alongside correctness.
- Distinguish exact measurements, approximations, and hypotheses. Check sketch or sampling resolution before interpreting small effects.

Every result must be traceable to code commit, recipe, effective configuration, model revision, data/split identity, seeds, precision, optimizer settings, and training/evaluation budget.

Perform a second critical review of any result that changes the research direction: could the result be caused by a bug, mismatched exposure, grading, precision, sampling noise, or a different effective update size?

A null result is useful only when you state which effects the experiment could reasonably detect. Never convert "unresolved" into "absent."

6. RESOURCES AND EXECUTION AUTHORITY

Resource ceilings for this study:

- Up to 2 GPUs in general.
- Up to 6 GPUs in preempt.
- Up to 20 concurrent CPU-only jobs through preempt\_cpu\_qos, subject to verified account limits.

Verify the CPU allocation semantics and applicable per-job CPU, memory, and account limits before configuring concurrency. These ceilings are permission to use resources, not a requirement to occupy them. Count standbys and development allocations against the budgets. Do not inherit the historical allowance of eight general GPUs.

Choose scheduling based on measured queue conditions, runtime, and preemption risk. Use CPU jobs for CPU-only work. Where profiling shows sufficient memory and throughput, a GPU may serve multiple lightweight tasks within a valid allocation; do not force co-location or risk memory contention, contaminated measurements, or slower time to evidence.

Follow the harness:

- Pin code and recipes.
- Submit through ae and use Slurm as the authority for job status.
- Use supported tinker workflows for code changes, inspect diffs, and validate before accepting.
- Keep checkpoints, environments, and large artifacts on compute storage under the correct project path.
- Harvest compact evidence; retain the detailed outputs needed to reproduce it.
- Use resumable phases and safe, unique output paths.
- Do not run project workloads on the login node.

You may autonomously submit, hold, cancel, replace, or redirect jobs started for this new study when evidence, failures, or resource conditions justify it. Preserve useful partial evidence and record the reason. Handle affected dependencies explicitly so cancellation does not launch invalid children.

Do not control another project's jobs or cancel historical jobs based on stale handover instructions. Do not retry deterministic failures without diagnosis. Never bypass pinning or validity checks to increase throughput.

Routine experiment decisions, scoped code fixes, and recovery within these limits do not require renewed permission. Ask only for genuinely missing access, a consequential unresolved requirement, or an action outside the authorized scope.

7. PROJECT AND STATE RECORDS

Before substantial experimentation, write this mandate into the harness's canonical PROJECT.md, incorporating:

- the central question and precise RQ1/RQ2 scope;
- competing explanations and operational definitions;
- the adaptive research strategy and success criteria;
- resource ceilings and autonomy;
- validation, provenance, review, and reporting requirements;
- deferred questions and owner amendments.

Use the harness's actual path and case conventions for the requested project.md, state.md, and checklist.md:

- PROJECT.md: the durable mandate.
- state/STATE.md: compact current scientific state, approximately 60 lines or fewer.
- state/CHECKLIST.md: actionable tasks, dependencies, validation requirements, and done/active/blocked/pending status.
- state/direction.md: the living scientific direction and hourly review record.

Do not create competing authoritative copies with different capitalization. Archive dated snapshots under research// because the harness workspace is git-ignored. Preserve existing workspace records before replacing their active scope; never erase the ledger or old findings.

STATE should summarize what is established, uncertain, and next—not duplicate job status available from Slurm. CHECKLIST must change whenever evidence changes the plan, not only when a job finishes.

direction.md must answer:

- Which part of RQ1 or RQ2 are we currently resolving?
- What are the leading competing explanations?
- What evidence would distinguish them?
- What result would change our mind?
- What have we ruled out, and at what resolution?
- Are we drifting into unrelated engineering, descriptive diagnostics, or premature solutions?
- Which experiment should be started, stopped, or revised next, and why?

Keep durable evidence and decisions in the harness's FINDINGS, DECISIONS, implementation log, and research narrative as appropriate.

8. HOURLY REVIEWS AND MONITORING

Conduct an hourly scientific and operational review while the study is active. Also review immediately after important completions, failures, or unexpected findings.

At each hourly review:

- Inspect actual job status, progress, failures, utilization, and dependency readiness.
- Check resource ceilings and whether queued work remains scientifically justified.
- Validate new results before interpreting them.
- Reassess progress against RQ1/RQ2 and the competing explanations.
- Cancel or revise your own uninformative or invalid work where justified.
- Update STATE, CHECKLIST, and direction.md.
- Record a timestamp, evidence considered, decisions, and next discriminating action.
- If nothing substantive changed, write a short entry rather than inventing activity.

Use deterministic event monitoring and a reliable timed wake-up for the requested hourly reviews. Do not repeatedly invoke the model merely to poll unchanged queues. Do not create a cluster-side autonomous research supervisor. If the current runtime cannot deliver scheduled wake-ups, state that limitation and establish a supported mechanism rather than claiming reviews are scheduled.

Maintain continuity across interruptions: a fresh orchestrator must be able to read the canonical records, inspect Slurm, and continue without reconstructing the project from chat.

9. SCOPE AND COMPLETION

This phase should leave us ready to investigate:

- interventions that restore learning;
- the role of pre-training;
- capacity;
- optimizer and training-procedure effects.

Do not launch broad sweeps on those later questions now. Small controls needed to interpret RQ1/RQ2 are in scope.

The deliverable is a defensible account of:

- which tested task relationships produce helpful, negligible, transiently harmful, or persistently harmful interactions;
- whether scarce corrective feedback amplifies harm;
- what alternative explanations remain;
- where short experiments are sufficient and where longer evidence is necessary;
- the most informative next questions.

Write a concise research report under reports// with clear figures and an experiment-to-claim evidence table. For each claim, give the comparison, result, uncertainty, and limitation. Separate exploratory evidence from independent confirmation. Include a handover with reproduction paths and a prioritized continuation plan.

Do not stop merely because a batch of jobs finishes. Do not keep running merely because resources remain available. **Continue until the two questions have a useful, carefully bounded answer, or until you can precisely identify the evidence still required and why it cannot yet be obtained.**

Write plainly. Define tasks, comparisons, and metrics before discussing results. Let figures carry numerical detail; explain what the evidence means and what it does not establish. Proceed autonomously with accuracy and sound scientific judgment; unblock yourself smartly — for instance, artifacts for scientific progress such as dataset downloads are explicitly authorized as long as you don't exceed storage on compute notes,

Begin now: read the required material, establish the new study records, assume that the harness is already verified, validate the first experiment family, and launch the smallest experiments that can materially sharpen RQ1 and RQ2.

Set a soft deadline of 9:00 PM EST tomorrow, November 16, 2026, for a coherent initial synthesis of both questions. Record this target in PROJECT.md, but assess its feasibility only after two hours of actual work, using measured throughput, queue delays, and emerging evidence. At that point, prioritize the most informative remaining experiments and make your best effort to reach the target; revisit the forecast during subsequent hourly reviews.

The deadline is a planning target, not a hard stop or a reason to weaken validation. If important evidence remains unresolved at 9:00 PM, provide a clear interim synthesis, identify the outstanding questions and work, estimate the additional time needed, and continue autonomously within the authorized scope and resource limits.

---

## Amendments

## Amendment A1 (owner, 2026-11-15 23:30 EST) — broaden candidate tasks beyond mathematics

Verbatim:

> You may broaden the candidate tasks beyond mathematics. Arithmetic can remain a useful controlled setting, but you are allowed to not make it constrain the scope of this study.
>
> Consider skill interactions within and across domains such as code generation, logical puzzles, symbolic reasoning, instruction following, and factual tasks with reliable grading. Cross-domain pairs need not include math—for example, code and puzzles are also in scope. Choose a small, diverse set for cheap screening, then deepen the comparisons that best inform RQ1 and RQ2.
>
> Select tasks for scientific value: reliable verification, measurable starting success, learnability, and potentially informative update interactions. Do not assume different domains conflict or similar domains cooperate. Distinguish competing behavioral objectives from distinct skills.
