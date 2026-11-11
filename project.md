# Project: Does targeted easier practice improve RL on hard problems?

## 1. Mandate and scope

Execute this preliminary study autonomously. Implement, validate, calibrate, train, evaluate, audit, and report the experiments below. Move between stages when their gates pass; do not stop after writing a plan, completing a smoke test, or finishing the first training batch. Fix ordinary engineering failures and continue. Do not seek a positive result by changing the question after observing the comparisons.

The immediate question is:

> Does easier practice on a specified mathematical skill improve learning on harder problems whose audited solutions use that skill, compared with matched easier practice that does not use it?

Use **Qwen/Qwen3-1.7B in non-thinking mode, full-parameter GRPO, one independent job per GPU**. Compare hard-only training, hard plus related easier practice, and hard plus unrelated easier practice. The primary contrast is related minus unrelated on held-out target problems. Hard-only measures whether allocating training effort to either auxiliary set is worthwhile.

This is **post-training transfer within an existing mathematical domain**, not acquisition of knowledge demonstrably absent from pretraining. The checkpoint is already post-trained. Disabling thinking at inference does not remove its training history. No additional SFT is part of this study.

Our longer-term question is whether feedback-grounded reflection can select useful auxiliary practice better than similarity or difficulty alone at the same total cost. **Do not implement that selector yet.** First establish whether our specified auxiliary practice produces useful transfer. Neither a positive result nor a plateau proves ray interference or saddle-point avoidance.

Authorized work includes all stages in this document: setup, dataset construction, calibration, nine primary pilot runs, six clipping-sensitivity runs, bounded diagnostic branches, evaluation, and independent recomputation. This is a finite programme, not permission for indefinite sweeps. Concurrency limits are ceilings, not utilization targets that justify unnecessary experiments. Complete the programme even if the primary result is null, unless a validity or resource gate fails.

**First session, in parallel:** On CPU, read the history, draft the skill rubric, and test the verifier. On the first available general GPU, test non-thinking generation, a full optimizer step, memory, and resume; use the second for a development-only length and difficulty probe when the renderer and verifier pass. Start the hourly review before waiting for either job. The critical path is a credible dataset and working trainer, not filling eight slots.

## 2. Authority, orientation, and reading

Read `slurm-agentic-engineering/agent.md`, the harness README, the active cluster card, and existing project state before acting. Follow the harness for execution and job ownership; this document defines the scientific scope and explicit resource authorization. Later user instructions supersede this document. Do not overwrite existing user work.

Read the project-history PDF(s) in `context/` briefly for orientation: synopsis, iteration transitions, conclusions, and limitations. Start with text extraction; inspect pages where figures or extraction errors matter. Spend no more than about 45 minutes on the first historical pass. Record filenames and a short summary in `research/READING.md`. Do not treat the PDF as the current execution protocol or reopen the MiniGrid programme. If `context/` is absent, record the missing background and continue from this document; do not pretend to have read it.

Read these primary sources before freezing the protocol. Focus on the listed sections, not an exhaustive literature review:

| Source | Read for | What not to infer |
|---|---|---|
| Schaul et al., *Ray Interference* (2019), https://arxiv.org/abs/1904.11455 — Sections 2.1–2.6 and 4 | Conflicting component updates, performance-dependent learning, exact toy dynamics, diagnostic controls | A flat curve alone does not establish this mechanism in an LLM. |
| Qu et al., *POPE* (2026), https://arxiv.org/abs/2601.18779 — Section 3.2/Figure 5, Section 6, Appendix E | Two-problem comparisons, hard-problem exposure, reward scarcity, model and optimizer setup | Do not assume related + hard must beat hard-only or copy the asynchronous recipe unchanged. |
| *Qwen3 Technical Report*, https://arxiv.org/abs/2505.09388 — Sections 4.2 and 4.5; model card https://huggingface.co/Qwen/Qwen3-1.7B | Learnability, rollout sampling, checkpoint history, non-thinking template | The report is not a validated numerical optimizer recipe for our setting. |
| *DAPO*, https://arxiv.org/abs/2503.14476 — Sections 3.1 and 4.1 | Asymmetric clipping and a learning-rate starting point | We are not adopting dynamic sampling, token-weighted loss, or length shaping. |
| Pinned verl GRPO/configuration documentation and code; https://github.com/huggingface/Math-Verify | Actual loss, masks, minibatch semantics, synchronization, answer verification | Configuration names alone do not establish implementation behaviour. |

Write at most two pages of reading notes: what each source establishes, what remains uncertain, and consequences for this protocol. Record versions/commits and exact configuration mappings. Resolve source ambiguities before relying on them. Historical results motivate the study; they are not results of this study.

Keep root `project.md` as the tracked canonical instruction. The harness expects `slurm-agentic-engineering/PROJECT.md`: put a short pointer and the operational paths there, without a competing protocol. Maintain tracked research records because the harness workspace is normally git-ignored.

## 3. Resource contract and isolation

### GPU ownership

- Maximum **2 allocated GPUs in `general`/`normal`** and **6 allocated GPUs in `preempt`/`preempt_qos`** for this project, counting training, evaluation, calibration, debugging, and standby allocations together.
- These are project ceilings, not claims about account-wide entitlement. Other projects run under the same username. Read account occupancy to determine available quota, but never cancel, hold, requeue, modify, attach to, or clean up their jobs or files.
- Initially request one GPU per experiment. Do not combine six jobs into one six-GPU allocation or silently switch to distributed training, LoRA, or a different model.
- Identify owned jobs through this project's ledger, experiment ID, pinned commit/recipe, and recorded Slurm job ID. A username or similar job name is not sufficient proof of ownership. Before controlling a process, also verify its PID start time and project-specific working directory or metadata.
- Do not use broad `pkill`, `scancel -u`, account-wide W&B synchronization, global cache cleanup, or edits to shared environments. Only modify this project's paths and environments. Never expose credentials in logs or config files.
- Existing valid jobs belonging to this project count against the limits. Adopt and monitor them instead of creating duplicates. An ownership ambiguity is a reason to leave the resource untouched.

### Scheduling

The owner's current experience is that `general` starts within minutes and `preempt` can take 12–24 hours. Verify live conditions; neither is a guarantee. Use `general` for short setup, calibration, and urgent evaluations (and run jobs on general and setup and calibrate through general or CPUs if `preempt` would take more than 30 minutes to start). Queue validated production jobs on `preempt` promptly after protocol freeze. Submit independent one-GPU recipes.

Use two logical general slots and six preempt slots. Use one slot per GPU requested, including self-tests and debugger allocations. Keep at most one submitted nonterminal one-GPU job per slot, including pending, held, running, and requeued jobs. This prevents pending jobs from exceeding the ceiling when they start together. A slot is reusable only after Slurm confirms its previous job terminal. Protect submissions with a project-local lock. Do not race duplicate copies across partitions. To move work, cancel/retire the owned pending job and verify its state before submitting its replacement, preserving the run lineage.

Initial production wave: three conditions for seed 101 and three for seed 202 on the six preempt slots. Run seed 303's three conditions on available general slots or released preempt slots, whichever produces results sooner. Schedule clipping sensitivity and diagnostics after their prerequisites are ready. Never leave an allocation idle waiting for an agent to inspect it: exit or continue a predeclared phase. Do not reserve GPU standbys speculatively or invoke plain `ae pool` with its default GPU holdings.

Verify partition/QOS, allowed GPU types, RAM, walltime, and account limits through the harness. Start with 1 GPU, 8 CPU cores, and 64 GiB host RAM per GPU job, adjusting only from measurements. Prefer one 48 GB GPU type across conditions if compatible with the observed queue. Otherwise balance hardware assignment across conditions and record GPU type. Hardware from unrelated benchmark notes is not a throughput estimate for this workload.

Use `exp/<id>.sh` from the installed template. For example, a short engineering job can start with `#EXP gpus=1 cpus=8 mem=64G time=02:00:00 part=general constraint=VRAM_48GB`; verify that feature on the account. Use `part=preempt` and a measured walltime for production. Submit with `ae submit <id>`; use `ae status`, `ae-watch`, and `ae harvest <id>`. Check upstream success manifests in dependent jobs; termination alone need not mean success. Reserve scarce GPU slots for work that needs a GPU.

Use CPU allocations for parsing, annotation tooling, verification audits, statistics, and disk checks. Do not run substantive computation on login nodes. Use a project-specific environment or immutable container, with pinned versions and a tested CUDA/attention stack.

## 4. First actions and execution gates

Perform these actions in order; continue independent CPU work while GPU jobs queue.

1. Inventory project files, active owned jobs, existing credentials/configuration, environment, available quota, and storage. Read the harness and `context/`. Preserve existing work and resolve branch/path conventions from the repository.
2. Before even the smoke jobs, set a unique project slug, a separate harness workspace/ledger, Slurm experiment-ID prefix, compute checkout and remote harness root (where the installed harness supports it). Verify resolved paths with `ae paths`; never reuse another project's default smoke-result path or standby. Establish `PROJ_DATA` under `/data/user_data/$USER/<unique-project-id>` and a unique W&B project/group prefix. Initialize tracked state and a run registry. Use existing authentication; ask only for missing access that cannot be resolved locally.
3. Run `ae doctor` and the relevant harness self-tests within the slot limits. Validate GPU visibility, writable shared storage, and job identity on compute nodes. However, do not take too much time running tests (more than a few minutes) since another project is already using the harness and CMU babel effectively.
4. Implement the data manifest, verifier, seeded sampler, full-parameter trainer, evaluator, checkpoint/resume path, logging, and analysis. Prefer existing verl facilities over a bespoke RL framework; inspect the pinned code to confirm all semantics below.
5. Pass the correctness and interruption tests in Section 8. Profile a real training round. Do not submit production training before these tests pass.
6. Build and freeze datasets; run the optimizer calibration. Freeze the selected configuration and final round budget in `research/PROTOCOL.md` before starting main comparisons.
7. Submit and finish the primary and clipping-sensitivity matrix. Run the prespecified local diagnostics, evaluate, independently recompute, and report. Update state after each transition rather than waiting until the end.

If a gate fails, diagnose and repair it. A failed gate does not authorize bypassing it. Missing historical context or W&B network access does not block otherwise valid local work. Missing Slurm/model access, an unusable dataset, or persistent trainer invalidity blocks the dependent stage only.

## 5. Dataset construction

### Source, skill, and splits

Use a pinned version of MATH, initially from `EleutherAI/hendrycks_math`, with original train/test provenance, stable IDs, problem text, reference solutions, and answer hashes. Start with algebra problems whose reference solutions centrally use **polynomial factorization in solving equations or deriving a requested quantity**. Related auxiliary questions practise that operation more directly. Unrelated auxiliary questions omit that focal operation; use direct modular-arithmetic problems as the initial candidate pool, excluding those whose solution centrally uses polynomial factorization.

Do not assert that the reference method is necessary or that unrelated questions share no useful representations. Record alternative solution routes. This is a study of a specific practice relationship, not a universal test of semantic similarity.

Split the official training data deterministically, before screening: 60% construction/training candidates, 20% study development candidates, 20% calibration candidates, stratified by source subject/level where feasible. Split the calibration pool again into training and validation halves. Use the official test split only for final evaluation. Group identified duplicate/near-duplicate problems before assigning internal splits; remove training-side counterparts of test overlaps. Record all exclusions and their reasons.

Desired final sizes: H_train=128, E_related=128, E_unrelated=128, target_dev=64, target_test=128. Each calibration split should contain at least 64 eligible problems, disjoint from all study sets. These are targets, not assumed dataset facts. If only smaller valid pools exist, use the same auxiliary size for both conditions and record actual counts. Minimum for launching the main study: 64 H_train, 64 per auxiliary pool, 32 target_dev, 64 target_test, and 64 per calibration split. Do not duplicate questions to meet a minimum.

If the first skill fails these minima, one alternative is authorized **before any main treatment run**: single-variable linear-equation manipulation, with harder word problems that require formulating and solving such equations, and matched easier equation-solving questions. Rebuild all affected labels and splits; keep the same experimental design. Do not switch based on a treatment effect. If neither construction is viable, finish the data audit and report the specific blocker instead of launching a weakly defined study.

### Agent annotation and audit

The research agent is authorized to annotate and review the data. Call the result **agent-annotated and audited**, not human-reviewed. Store for each item: source ID, split, focal skill, central/incidental/absent label, supporting reference-solution step, alternative route, confidence, and verifier eligibility.

Conduct a second review in a fresh context without the first labels, confidence, or treatment results. Preserve both passes, disagreements, and resolutions. Exclude unresolved cases. Review all selected items; do not use observed training gains to relabel relatedness. A second agent pass is an audit, not independent ground truth. Audit a seeded sample of up to 64 incorrect screening attempts on hard training candidates, at most one per problem. Mark whether the first identifiable error concerns the focal skill, another skill, or cannot be determined; also flag valid alternative solutions. Report counts including unknowns. This is a diagnostic, not a licence to select the family with the most promising errors. Use the fallback only for the stated construction/minimum-size failure.

### Difficulty screening and matching

Use the exact frozen model, prompt, decoding, and response cap intended for training. Start with 16 samples per candidate. In the fixed candidate order, shortlist H candidates with 0–5 successes/16 and auxiliary candidates with 4–14/16; take up to twice the target size per pool per tranche. The broad first screen deliberately retains some zero-success hard candidates. Then obtain 32 independent samples per shortlisted item. Retain counts from both stages but select by the second stage: H_train candidates have 2–8 successes/32; auxiliary candidates have 12–24/32. These are noisy screening rules, not exact population probabilities. Re-estimate the selected pools with 16 new baseline samples for reporting; do not reselect based on this third sample.

Before screening, rank eligible candidates by a fixed seeded permutation, not by preferred apparent outcomes. Limit initial screening to 1,200 candidates for the primary skill; one additional 1,200-candidate tranche is authorized if needed to meet minima. The alternative skill, if used, has the same limits. Record screening costs and reasons for stopping.

Match E_related and E_unrelated one-to-one using second-stage success counts, prompt length, mean generated response length, and answer type. Prefer exact success-count matches, then four-count bins. Within bins minimize total standardized length distance, with source ID as the deterministic tie-breaker. Save the matched-pair table and matching code. Exclude unmatched items rather than hiding imbalance. Report histograms, standardized differences, and expected mixed-reward-group probability `1-p^8-(1-p)^8`. Equal mean accuracy alone is insufficient. Treat absolute standardized differences above 0.25 in success or response length as a mismatch to repair before freezing; if repair violates minima, apply the dataset gate.

Select development and test target problems by the frozen skill/task rubric, **not by measured model success**. Test outcomes must remain inaccessible to configuration selection. Automated baseline test evaluation may run after freeze and be sealed until endpoint evaluation. Do not expose test solutions to the training prompts, annotation-driven selector, or calibration.

## 6. Model, reward, and GRPO specification

### Fixed model and input

- Model: pinned revision of `Qwen/Qwen3-1.7B`; all parameters trainable; no LoRA, SFT, quantized optimizer, or teacher hints.
- Apply the official chat template with `enable_thinking=False`. Save the rendered template and a fixture proving the flag is honored. This is non-thinking operation of a hybrid-trained checkpoint.
- User text: `Solve the following problem. Show your working and put your final answer in \boxed{}.` followed by the problem. No references or demonstrations in context.
- Prompt cap 1,024 tokens, including chat-template overhead; never silently truncate problems. Exclude overlong prompts consistently or increase the cap once before calibration and freeze it.
- Response cap 4,096 tokens initially. Temperature 1.0, top_p=1.0, top_k disabled according to the pinned backend, repetition penalty 1.0, no other sampling warpers. Apply the same decoding to screening, training, and evaluation.
- Run an initial length probe on construction items before large-scale screening. During construction, if more than 5% of target attempts hit the cap, inspect a fixed sample and test 8,192 tokens. Use 8k if it materially resolves incomplete useful attempts and fits. If substantial truncation remains, report the regime as unsuitable; do not train as though failures were solely mathematical. If the cap changes, supersede affected screening estimates and repeat screening under the final cap before freezing membership. Freeze the cap before calibration and main runs.

### Reward and failures

Use pinned Math-Verify with audited canonical gold answers and a fixed parser configuration. Extract only the designated final answer, not any matching number somewhere in the reasoning. Reward verified correct answers 1, incorrect/missing/malformed final answers 0. Require a non-truncated completion for reward 1; capped incomplete responses receive 0. Report this as a budgeted-completion objective. No format bonus, partial credit, length shaping, or learned reward model.

Validate every gold answer. For capped responses apply the stated completion rule even if the last visible fragment happens to match the answer; log truncation separately. Use a verifier timeout of 5 seconds per response initially, implemented with proper process cleanup. Persist the prediction and diagnostic on timeout. A timeout/crash is **unknown**, not a wrong answer: retry once in an isolated process with 15 seconds. If unresolved, hold the affected rollout round, repair verification or mark the run invalid; never silently drop and replace difficult responses. Fix parser bugs using training/development fixtures, document their scope, and recompute affected evaluations. No arbitrary execution of generated code.

### Algorithm

Use synchronous GRPO, fresh rollouts each round, no critic, no replay buffer, no reference-policy KL term, no entropy bonus. Keep zero-variance groups in the loss denominator with zero numerator contribution; do not filter, resample, or adaptively prioritize them.

| Quantity | Setting |
|---|---|
| Prompts per round | 16 |
| Responses per prompt | 8; total 128 responses/round |
| Optimizer minibatch | 4 complete prompt groups = 32 responses |
| Optimizer steps per round | 4, one pass through the round |
| Mixed-condition minibatch | 2 hard groups + 2 auxiliary groups |
| Microbatch | 1 response initially; accumulate exactly to 32 responses |
| Precision | BF16 computation; standard FP32 optimizer states/master weights where supported |
| Optimizer | AdamW, betas=(0.9,0.999), epsilon=1e-8, weight_decay=0 |
| Learning rate | Selected in Section 7; constant, no warmup |
| Gradient norm limit | 1.0 |
| Clipping | Selected in Section 7 |
| Loss reduction | Mean over valid response tokens per response, then mean over all responses |

For each prompt group, use `A_i=(r_i-mean(r))/(population_std(r)+1e-6)`. An all-equal reward group has exactly zero advantage. Expand the response advantage over its valid response tokens, including the EOS token when generated. Mask prompt and padding tokens. For response-token probability ratio `q=pi_current/pi_rollout`, maximize `min(q*A, clip(q,1-eps_low,1+eps_high)*A)` with the reduction above. Store the rollout-policy log probabilities once and keep them fixed for all four optimizer steps. Synchronize the rollout model after the round.

Inspect and test the pinned framework: its minibatch fields may count prompts rather than expanded responses; its default standard deviation or loss reduction may differ. The mathematical specification here is authoritative. Record resolved settings and actual counts. Accumulation must not accidentally give short microbatches different weights. Keep minibatch composition and optimizer ordering balanced across mixed conditions. Zero direct gradient does not imply zero Adam momentum update.

Use checkpointing and efficient attention to fit one GPU. Alternate generation and training using a supported memory-management path; account for duplicate model copies and caches. If OOM occurs, first reduce generation concurrency and transient token batching, then enable validated offloading. Preserve effective batch size, objective, model, and cap. If these cannot fit, report the resource blocker rather than silently changing the scientific specification.

## 7. Calibration, freezing, and experiment matrix

### Optimizer calibration

Use only the disjoint calibration train/validation sets. Exclude near-duplicate templates of study items. Before optimizer runs, use 16 independent baseline samples per item to select a control panel with both lower (1–5/16) and higher (6–12/16) observed success; include at least 16 items in each band in each split. Freeze both panels and use independent baseline evaluation draws. Do not use treatment, study-development, or test outcomes for optimizer selection. Run six configurations: learning rates `{5e-7, 1e-6, 2e-6}` crossed with clipping epsilons `{(0.2,0.2), (0.2,0.28)}`. These correspond to ratio intervals `[0.8,1.2]` and `[0.8,1.28]`, not KL bounds. Every run uses seed 11, 32 rollout rounds, and the same sampler schedule. Evaluate calibration validation with 16 responses/problem initially and at the endpoint. Save round 16 measurements for diagnosing trends.

Reject configurations with non-finite updates, broken verification, or more than 5% capped validation responses. Rank the remaining configurations by endpoint calibration-validation pass@1 gain. Repeat the top two with seed 22 for 32 rounds. Select using their mean gain across seeds. Treat differences of at most 0.01 as ties; among tied choices prefer lr=1e-6, then 5e-7, then 2e-6, and symmetric before asymmetric clipping. This is a pragmatic selection rule, not a significance test.

Require positive mean validation gain with neither seed worse than baseline by more than 0.02. If fewer than two first-stage configurations are valid, diagnose the common failure, fix it, and rerun affected configurations once; do not call an invalid configuration a finalist. If the validation-gain gate fails, diagnose first. After correctness is reconfirmed, extend only the two finalist runs for both seeds to 64 rounds and reapply the rule. If still unsuccessful, report inadequate calibration; do not expand to an open-ended optimizer search or launch uninterpretable main comparisons. All correctness repairs get new attempt IDs and recorded lineage.

Measure entropy, objective-active clip fractions by advantage sign, probability-ratio quantiles, gradient norms, validation accuracy, and lengths. If clipping is rarely active, say the tested settings have similar realized behaviour. Do not add epochs just to make the clipping ablation look consequential.

### Freeze

Commit `research/PROTOCOL.md`, dataset manifests, resolved configs, seed mappings, evaluator, and analysis plan before main runs. Record paper/software revisions, calibration selection, exact skill, pool sizes, prompt/cap, hardware, and intended costs. Freeze the annotation manifest and its hash. Start all main runs from the original checkpoint, not the calibrated model.

Default endpoint: 128 rounds. Before main results exist, estimate total cost using measured throughput, including all 15 main/sensitivity runs and evaluation. If a production run is projected to exceed 24 active GPU-hours, use 64 rounds for **all** conditions and sensitivity runs and label it a shorter-budget pilot. If even 64 rounds exceeds this planning threshold, investigate throughput and present the measured blocker. The threshold is a conservative project planning default, not an account quota. No extension based on promising curves.

### Primary matrix

| Arm | Prompts/round | Seeds |
|---|---|---|
| H | 16 hard | 101, 202, 303 |
| R | 8 hard + 8 related auxiliary | 101, 202, 303 |
| U | 8 hard + 8 unrelated auxiliary | 101, 202, 303 |

Within each seed block, R/U have identical hard schedules and paired auxiliary selection schedules. Sample via shuffled cycles; avoid duplicate prompts within a round. H uses the common eight hard prompts plus eight additional hard prompts per round. Do not pretend H has matched hard exposure: it has twice as much. All arms have equal prompt/response counts, not necessarily equal tokens or compute.

### Clipping sensitivity

After freeze, run R and U for all three seeds with the **other clipping setting**, keeping the selected learning rate and every other setting fixed: six runs. This sensitivity is authorized regardless of the primary result; it is not another configuration-selection stage. Do not pool it with the primary results or promote whichever setting looks best. Report the R-minus-U contrast under both settings.

### Local update diagnostic

Reserve the halfway full restart checkpoint for this assay before retention cleanup. Use the selected-clipping R trajectory for seed 101, at rounds 0 and halfway to the endpoint, chosen now rather than after results. For each source checkpoint, clone model **and optimizer** state into four branches: no update, one minibatch update using hard problems, related auxiliary problems, or unrelated auxiliary problems. Use four prompts × eight responses per update. Repeat with four prespecified minibatch seeds (4101–4104). Keep source checkpoints unchanged.

Evaluate before/after on a fixed 32-problem development subset with 16 responses/problem and paired evaluation seeds. These are short local diagnostics, not new full training runs. They estimate the effect of actual Adam/GRPO updates; historical momentum is part of that effect. Do not describe them as a pure gradient-cosine measurement or proof of saddle-point avoidance. Run them regardless of whether the primary estimate is positive; do not chase additional checkpoints.

## 8. Validation and reproducibility

Before calibration, pass meaningful tests for: template/masking/EOS handling; group boundaries and zero advantages; population standard deviation; loss reduction and accumulation; all-parameter trainability; four optimizer steps/round; balanced minibatches; verifier correctness; and deterministic dataset splits. Test that a known rewarded response receives the expected signed objective contribution. An artificial fixture is a unit test, not scientific evidence of learning.

Before each rollout round's first update, check that stored rollout log probabilities and teacher-forced training log probabilities on the same tokens agree within a tolerance established on the pinned backend. Record mismatch quantiles; investigate substantial mismatches rather than masking them with wide clipping. Confirm a training update changes the rollout model after synchronization. Keep dropout disabled during both rollout and policy-loss computation.

Seed Python, NumPy, PyTorch CPU/CUDA, dataset construction, prompt order, minibatch order, generation, evaluation, bootstrap, and diagnostic branches. Derive separate named streams from the master seed with a stable hash or SeedSequence, never Python's randomized `hash()`. Use construction seed 1701 and bootstrap seed 2027. Store per-request generation seed mappings where supported; otherwise save engine RNG state and record the limitation. Evaluation streams are independent of training and selection streams.

Enable practical deterministic settings without claiming bitwise reproducibility across GPUs or kernels. Record versions, GPU model, driver, attention kernels, and nondeterministic operations. A repeated-seed smoke test establishes expected numerical variation; it is not an independent scientific replicate.

Force an interruption and resume during the general-pool smoke test. Test restoration of optimizer, scheduler if any, RNG, sampler position, counters, and logging identity. Confirm no skipped or duplicated committed rounds. Mid-round work may be discarded and replayed from the last complete round; log and charge it. Compare with an uninterrupted short run, using documented tolerances rather than demanding impossible cross-backend bit identity.

## 9. Evaluation and statistical interpretation

For a 128-round run, development evaluation occurs at 0,32,64,96,128, with 8 responses/problem; for a 64-round run use 0,16,32,48,64. Final test evaluation occurs only at baseline and endpoint, with 32 responses/problem. Use the same prompt/decoding/cap as training. Baseline results can be reused because the starting model is identical, but record the evaluation seed panel and preserve pairing.

Primary estimand: seed-mean difference R minus U in endpoint **held-out target pass@1**, with equal weight per target problem. Secondary: R minus H, U minus H, development learning curves, training-target success, per-problem changes, and pass@8. Estimate pass@8 from 32 attempts using `1 - C(32-c,8)/C(32,8)`, with numerator zero when `32-c < 8`. Report success distributions and counts of regressions without treating noisy single-problem changes as established effects.

Show all three seed contrasts and a two-sided 95% paired t interval over seeds as a descriptive uncertainty summary. Also provide a paired crossed bootstrap resampling seeds and problem IDs while preserving arm pairing, 10,000 draws, reporting the 2.5th and 97.5th percentiles. Explicitly state that three seeds give weak training-variation estimates and that a large number of questions does not replace training replications. Do not select a claim by whichever interval is favourable. No equivalence or definitive no-effect claim from a nonsignificant pilot.

Predeclare 0.03 absolute success improvement as a practical planning effect, not a significance threshold or a gate for running authorized work. A consistent positive R-U result merits later replication; practical usefulness also requires comparison with H and cost. A negative/null result remains valid evidence within this skill, model, budget, and optimizer. Do not prescribe a selector from a single skill family.

Report actual generated tokens, prompt tokens, optimizer steps, active GPU-hours, elapsed calendar time, and discarded/replayed work separately. Give performance against rounds, hard responses, tokens, and active GPU time at observed checkpoints; do not invent exact matched-compute endpoints through unlabelled interpolation. Different GPU types require separate throughput interpretation. The initial primary experiment controls exposure; it does not yet establish equal-cost selector superiority.

Independently recompute final rewards and metrics from saved attempts with a separate audit entry point. It may reuse the pinned verifier but must not simply read the main summary tables. Compare counts, identities, checkpoint hashes, success totals, seed contrasts, and resource ledgers. Audit a fixed seeded sample of verifier decisions and trajectories qualitatively. Resolve discrepancies before calling the programme complete. Give the primary contrast first, with a table of H/R/U by seed; plot development pass@1 against rounds and tokens, marking the final test measurement separately. A single endpoint difference supports an endpoint claim; to describe faster learning, use the prescheduled development curves, not the most favourable checkpoint.

## 10. Configuration, logs, and W&B

Keep versioned base configs and small explicit experiment overrides. Save the fully resolved config, model/tokenizer revision, dependency lock/container digest, code commit, recipe hash, data hash, seed streams, host/GPU identity, and command for **every** run, including failures. Never rely on a mutable default. Separate scientific run ID from attempt ID; a retry belongs to the same scientific seed, not a new replicate.

Use W&B for concise searchable telemetry, with a unique project name and phase tags for calibration, primary, sensitivity, and diagnostics. Use one W&B group per scientific run within its phase (or equivalent phase/run tags). Resume the same W&B run ID on a true checkpoint restart; give a superseding retry from scratch a new W&B ID linked to its predecessor. Assign every metric row a logical round/step and attempt ID. On recovery, preserve discarded rows but mark them superseded; publish committed metrics once and audit their uniqueness. Keep an append-only local JSONL log as the source of truth; W&B unavailability must not block training. Sync only this project's offline runs when access returns. Never upload secrets, full checkpoints, or the entire dataset by default.

| Cadence | Log |
|---|---|
| Each optimizer step | Total loss, policy loss, pre-clipping gradient norm, whether gradient clipping applied, learning rate, valid tokens/responses, numerical-failure indicators |
| Each rollout round | Success by H/auxiliary, all-zero/all-one/mixed groups, lengths and truncation, malformed answers/verifier errors, policy-ratio quantiles, objective-active clip fractions split by advantage sign, elapsed generation/training/verification time, cumulative tokens and optimizer steps |
| Every 8 rounds | Token-weighted entropy if efficient, fixed small trajectory sample, representative memory/throughput summaries |
| Each scheduled evaluation | Pass@1/pass@8, per-problem count artifact, seed identity, checkpoint hash, evaluation time/tokens |
| Event-driven | Submission, start, checkpoint commit, interruption, resume, OOM, software change, exclusion, failed verification, terminal status |

KL and entropy losses are disabled: record their coefficients as zero and their losses as disabled, not fabricated measurements. If approximate behaviour-policy KL is logged, label it as a diagnostic, distinct from a reference-policy KL penalty. Do not compute full reference KL solely for logging. Use on-device aggregates; avoid CPU transfers or `.item()` calls per token. Target telemetry overhead below 3% of round time and measure it; reduce costly diagnostics, not essential counters, if exceeded.

Persist compact attempt records: run/checkpoint/round/prompt IDs, split, generation seed, token count, termination reason, extracted answer, reward/verifier status, and response token IDs or compressed response text. Keep all evaluation attempts and training responses needed for audits. Store full-vocabulary logits or per-parameter gradient histories **never** by default; transient rollout log probabilities need not be archived permanently. W&B receives scalar summaries and small sampled tables, not every response. Do not change training based on live W&B impressions.

## 11. Storage and preemption

The harness documents 100 GB home and 500 GB user-data quotas, shared with other projects; verify current quotas. Never put checkpoints, datasets, model caches, or large traces in `/home` or `$AE_RUNDIR`. Use project-scoped subdirectories under `PROJ_DATA`; use existing shared read-only caches when appropriate and never evict them.

Before a wave, measure checkpoint size and reserve for concurrent writes, two valid restart copies per active run, planned model-only snapshots, raw records, and temporary downloads. Treat all other projects' space as unavailable. Keep at least **50 GiB of free account quota** after projected peak writes; if current free space is already below that floor, pause new large writes rather than claiming remaining space. Project soft warning: 250 GiB; hard ceiling: 350 GiB, or less if account headroom requires it. These project defaults do not override a smaller real quota. Serialize checkpoint writes if shared storage cannot sustain them, and reduce active job count if necessary; do not borrow another project's space.

Write checkpoints atomically to a temporary path, validate completeness/checksums, then update a manifest pointer. Keep the newest two full restart checkpoints for active runs. Once a run and its evaluation/audit are complete, keep the endpoint model-only export and compact evidence. Retire optimizer state once the evaluator and an independent reload have validated the export and no planned branch needs it. For calibration, keep finalist resume checkpoints through the extension decision. For the diagnostic source, retain optimizer state until all branches pass audit; prune superseded project-owned copies after verifying references and recording deletions. Do not delete evidence just to make a failed result disappear. Avoid redundant copies of base weights and W&B artifacts.

Initially save complete-round restart checkpoints every four rounds and on a warning if safe. Measure write time and adjust cadence before production to keep expected lost work below approximately 15 minutes where feasible. The harness supplies `$AE_PREEMPT_FLAG` and requeue support, but the trainer must honor it. Use the tested harness phase wrapper to requeue after a controlled save/exit; do not assume a process that exits will automatically restart. If the warning is too short, rely on the last verified checkpoint. Restore all scientific state, not just weights. Count failed/replayed work in actual compute cost.

## 12. Monitoring and hourly self-review

Create and start a **project-owned hourly review loop** as part of execution. Use the research agent runtime's supported scheduling/background mechanism, not an unrelated ChatGPT reminder. Make it persistent across normal agent yields, with an ownership record, lock, last-run timestamp, and missed-review detection. Verify one short test trigger reaches the agent, then set the interval to 60 minutes. Do not claim a review is scheduled if it is only written in a checklist.

Prefer a runtime-managed background watcher whose exit actually wakes the orchestrator; a detached process that only writes a file is insufficient. If the scheduler expires, renew this project's recurring task before expiry. If there is no persistent timer, use a managed hour-long background wakeup and re-arm it after each review. On orchestrator restart, reconcile jobs and run one catch-up review if overdue. Record last successful review and next due time in UTC.

Use `ae-watch` or equivalent project-local deterministic scripts for job transitions. Poll owned jobs cheaply, approximately every 2 minutes, and wake the agent on completion, failure, stalled progress, dependency readiness, disk pressure, or review due. Scripts may detect events; they must not invent scientific changes or autonomously launch unapproved variants. Re-arm the watcher after handling an event. Batch concurrent events and suppress duplicate notifications.

At each hourly review, inspect actual Slurm state and fresh artifacts, then answer in `research/REVIEWS.md`:

1. What completed, what is running/queued, and what changed since the last review?
2. Does every active job still answer the stated question? Are we drifting into debugging a single example or seeking a favourable result?
3. Are reward verification, sampling, masks, mixed-group rates, optimizer updates, and truncation behaving as intended?
4. Are there confounds, leakage, protocol deviations, or claims stronger than the evidence?
5. Are the next jobs ready, scientifically necessary, and within GPU/storage limits? Can useful CPU work proceed while GPUs queue?
6. What is the single most important next uncertainty or blocker, and what authorized action resolves it?
7. Do STATE, CHECKLIST, the run registry, W&B, and the Slurm ledger agree? Correct discrepancies from evidence.

Write concise conclusions and decisions, not an unbounded transcript of reasoning. Most reviews should take 5–10 minutes; spend longer only on a concrete issue. Continue healthy training during review. Do not change a frozen treatment because an interim curve is weak. Commit meaningful state updates. If all work is pending, keep event monitoring and the hourly review alive; do not terminate the programme simply because the queue is long.

At programme completion, verify the terminal state of all owned jobs, sync outstanding logs, stop only project-owned monitors/timers, and record their shutdown. If authorization or access blocks all remaining work, preserve checkpoints and monitoring only where it can still usefully detect progress; report precisely what is needed.

## 13. Autonomous recovery and change control

Authorized without another permission request: project-local code fixes, tests, dataset annotation, the declared pre-training fallback, calibration, all listed runs, checkpoint resume, migration of owned queued work between authorized pools, safe retention cleanup, local/offline W&B logging, documentation updates, and commits/pushes to the configured project branch under the harness workflow. No force-pushes, destructive resets, changes to unrelated projects, or account-wide actions.

For a failure: save diagnostics, classify it as infrastructure/implementation/scientific validity, reproduce minimally, fix the cause, test, record the change, and resume or replace the run with correct lineage. Permit at most two unchanged automatic retries for a transient infrastructure failure. A third recurrence requires diagnosis and a changed remedy, not another blind retry. Preemption with a valid checkpoint is normal continuation, not an excluded seed.

Set progress alarms from measured round time: no progress for the larger of 20 minutes or three expected round durations triggers inspection, not immediate cancellation. Identify whether generation, verification, checkpoint I/O, or GPU health is responsible. Do not kill a healthy long round based on a generic timeout.

Before protocol freeze, resolve routine choices within this document and record them. After freeze, document all fixes before resubmission. Performance-neutral fixes may continue after validation. Any fix changing rewards, masks, loss, sampling, dataset membership, or optimizer semantics requires a protocol amendment and consistent reruns of all affected conditions from the original checkpoint. Preserve superseded runs; never mix incompatible runs in a table. The declared clipping sensitivity remains a separate analysis.

Ask the user only when missing credentials/permissions prevent execution, neither authorized task construction works, the specified model cannot run validly on one GPU, calibration remains invalid after the declared extension, a scientific change exceeds this scope, or resources beyond the ceilings are required. State the exact blocker, evidence, attempted remedies, recommendation, and what independent work continues. Do not ask the user to approve a routine next stage already authorized here.

## 14. Records, outputs, and completion

Maintain these tracked records (adapt existing equivalent filenames rather than duplicating them):

- `research/STATE.md`: current phase, active/queued owned jobs, blockers, next actions, latest evidence timestamps.
- `research/CHECKLIST.md`: gates and remaining work, with evidence links for completed items.
- `research/PROTOCOL.md`: frozen design, manifests, calibration rule/result, endpoints, amendments.
- `research/DECISIONS.md`: dated decisions and rationale, including failed alternatives and deviations.
- `research/REVIEWS.md` and `research/READING.md`.
- `configs/`, dataset manifests/annotation records, and the project run registry.
- `reports/REPORT.md`, machine-readable result tables, plotting scripts, and audit outputs.

Reconcile short operational summaries into the harness state files; the tracked research records hold scientific authority. Keep large raw records on compute storage, referenced by manifest and checksum. Do not commit model weights, private credentials, or huge outputs to git.

Declare the programme complete only after validating the data and trainer, finishing calibration and all 15 valid treatment runs, completing the scheduled diagnostics and evaluations, independently recomputing the results, and reconciling costs. If a prerequisite remains blocked or any cell is missing or invalid, deliver a partial report naming the affected comparisons. Never call an incomplete matrix fully executed. A failed gate can produce a legitimate blocked-stage report, but cannot be relabelled a negative treatment result.

The report must explain the question, dataset relationship and ambiguity, model/optimizer, per-seed findings, uncertainty, costs, exclusions/retries, limitations, and what follows. Distinguish beneficial transfer, harmful interference, absence of transfer, and reduced direct hard practice. A weak pilot is not a proof of no effect; a positive pilot is not a dependable selector. Propose fresh-seed/second-skill replication if warranted, but do not launch an unbounded new programme.

Begin now: inspect the project and harness, orient from `context/`, establish the owned resource ledger and hourly review, then execute the gates. Continue until the authorized programme is complete or a concrete blocker requires the user.

## 15. Addendum A1 (owner, 2026-11-07): precision audit and report figures

Recorded verbatim in substance from the owner's message of 2026-11-07. It applies from now on.

### A1.1 Numerical precision of full-parameter GRPO
- BF16 for rollout inference and the learner's supported mixed-precision forward/backward.
  FP32 for the authoritative trainable weights updated by AdamW and for both moment buffers.
  FP32 parameters with BF16 autocast satisfies the weight requirement; do not allocate another
  FP32 copy. Prefer FP32 gradient accumulation across microbatches.
- Compute selected-token log-probability differences, importance ratios, advantage normalization,
  loss reductions and gradient norms in FP32 where supported. Use fused or chunked operations;
  never materialize full-vocabulary FP32 logits for every token.
- Inspect the pinned checkpoint and the actual learner after its first optimizer step. Record
  separately: checkpoint tensor dtype, rollout dtype, learner compute dtype, optimizer-owned
  parameter dtype, gradient accumulation dtype, moment-buffer dtypes, across all parameter
  groups/shards. A config field is not evidence. For fused kernels, separate documented behaviour
  from measurement.
- On a disposable smoke run (or a calibration checkpoint), on a mixed-reward minibatch: record a
  nonzero gradient norm; measure the change in a fixed sample of optimizer-owned weights; verify
  that repeated small updates accumulate; that rollout sync receives the updated weights at the
  expected serving precision; and that a restart restores FP32 weights and moments without a BF16
  round trip. Save one small audit table, the code revision and peak GPU memory. One check per
  trainer configuration; no per-step tensor dumps.
- Existing jobs: determine what they used. Conforming jobs continue. Confirmed BF16 optimizer
  updates, reduced-precision moments or lossy restarts: preserve evidence, find the earliest
  affected stage, correct it consistently across comparable runs, recalibrate if calibration used
  the affected path, record superseded attempts and their cost. If memory is tight, first reduce
  rollout concurrency or use validated optimizer-state offload; keep one GPU per job.

### A1.2 Final report `reports/REPORT.md` with embedded figures
- Figures in `reports/figures/`, embedded by relative Markdown paths; commit the plotting script
  and compact source tables; verify the report renders from a clean checkout. Each caption names
  the question set, metric, number of training seeds and evaluation attempts, and the meaning of
  any interval.
- **Main result:** held-out hard-problem pass@1 for H, R, U; paired R−U per seed and mean;
  clipping sensitivity shown separately.
- **Learning and exposure:** hard-dev pass@1 at scheduled checkpoints vs rounds and vs cumulative
  hard-problem rollouts, individual seeds; generated tokens and active GPU-hours in a companion
  panel/table.
- **POPE-style trajectories:** J_H = mean sampled success on a fixed hard-dev panel; J_R, J_U on
  fixed related / unrelated auxiliary probe panels; every arm evaluated on the same panels. Plots
  (J_R, J_H) and (J_U, J_H), common axes, marked start, arrows between successive checkpoints.
  Use fixed seeded dev probes where available; if probes come from auxiliary training pools, label
  the axis "training-pool performance". Same axis definition for all arms. Freeze probe IDs before
  inspecting probe results. Disclose missing checkpoints.
- **Local update effects:** from the prespecified branches (identical model + optimizer
  snapshots), the observed ΔJ_H after hard, related, unrelated and no-update branches, with
  variation across minibatch seeds; optionally paired arrows (ΔJ_R, ΔJ_H), (ΔJ_U, ΔJ_H). Label
  as finite changes after an AdamW/GRPO update. "∇J" or "gradient alignment" only for an actual
  parameter-gradient measurement. No vector field or saddle-point inference.
- Reuse scheduled evaluations and retained checkpoints. If extra probe evaluations cost
  materially, choose a smaller fixed panel before looking at results and record the cost.

## 16. Addendum A2 (owner, 2026-11-07): auxiliary pool size, supplementation, fallback choice

Recorded faithfully from the owner's message of 2026-11-07, before any treatment run.

- **64 matched pairs is a planning target**, not a hard minimum, for E_related/E_unrelated. A
  modest shortfall is acceptable if the retained pairs give adequate skill coverage, diversity,
  and balance in baseline success and response length. Document the tradeoff. A shortage says
  nothing about whether related practice helps.
- **Authored or augmented training problems** are allowed in either auxiliary pool to fill
  specific gaps found during screening. First record the gap and a finite generation budget.
  Apply the same skill audit, answer verification, difficulty screen and matching rules to every
  candidate. Report how many were generated and retained per arm. Balance synthetic provenance
  across arms so it cannot explain the contrast. Track source and template families; count
  near-duplicate variants together when judging diversity. Keep held-out evaluation problems
  independent and free of training-template variants.
- **Choice:** use judgment to pick the strongest defensible option — a slightly smaller natural
  set, a supplemented set, or the agreed fallback skill. Freeze the dataset, rationale and
  manifests before examining treatment outcomes.
- Supersedes, for the auxiliary pools only, the §5 sentence "Minimum for launching the main
  study: … 64 per auxiliary pool"; the other minima stand.

## 17. Amendment A3 (owner, 2026-11-07): CPU allowance

"You are allowed to use 30 CPUs from the CPU-only queue since my other project has relinquished
some of its capacity." This project may hold up to 30 CPUs in total on `preempt_cpu_qos`
(all owned nonterminal CPU jobs together). GPU ceilings are unchanged.

## 18. Amendment A4 (owner, 2026-11-07): resolving the H_train shortage

Recorded faithfully from the owner's message of 2026-11-07. The agent owns the choice among the
constructions below and escalates only if all credible paths are exhausted or the remedy exceeds
authorized scope or resources. All dataset and endpoint decisions precede any treatment outcome.
Optimizer calibration and trainer validation continue in parallel.

1. **Full native stage 2.** Revisit all 154 original hard candidates and all 30 level-3 reserves.
   For every audited, verifier-eligible item obtain 32 stage-2 attempts independent of stage 1
   (same checkpoint, prompt, decoding, cap, verifier). Reuse valid stage-2 attempts already
   collected. Maximum 5,888 attempts across the 184 candidates, less valid stage-2 work done.
   Record exclusions; report counts by band and source before screening another dataset.
2. **Bands and target.** Focal skill stays factorization. Core hard = 2–8/32; moderately hard =
   9–12/32. Target 64 distinct problems, at least half in the core band. Prefer varied solution
   structures; break ties by a recorded seeded order. With 48–64 credible native items, proceed.
   After selection collect an independent baseline and report it without reselecting. Measure the
   actual fraction of eight-response groups containing both a success and a failure.
3. **External tranche (if native is too small or lacks important structures).** One tranche of up
   to 200 candidates from a pinned Omni-MATH revision: verifiable, multi-step, short-answer,
   audited solutions centrally using factorization. Same annotation, bands, verifier, response
   budget. Exclude proofs, guessable yes/no, and items whose difficulty comes mainly from
   truncation. Check contest identity, wording and solution templates for overlap with every split.
   Preserve the held-out MATH panel; record provenance and screening yield.
4. **Fewer than 48 suitable items.** A smaller factorization pilot may launch without asking.
   First record the number of distinct solution families, the core-band share, and expected
   repetitions per item per arm (H-only 16T/N presentations per item; mixtures 8T/N). With
   adequate diversity but excessive repetition, shorten the endpoint uniformly across arms and
   freeze it before launch. If the pool is credible, proceed directly.
5. **Authored tranche (if missing structures make the small pilot uninformative).** One bounded
   tranche of authored hard factorization problems: first record missing structures, candidate
   budget, stopping rule; same independent annotation, difficulty screen, verifier audit,
   deduplication; track synthetic provenance; minor template variants count as one family. If this
   fails, assess the linear-equation fallback on measured eligibility and proceed if credible.
6. **Freeze and launch.** One H_train manifest for all arms. R and U get identical hard-prompt
   schedules. Balance source and difficulty strata in H-only's additional hard prompts. Keep the
   primary R−U comparison; report training performance separately from held-out MATH transfer.
   Document pool, endpoint, provenance, limitations; launch authorized runs when gates pass.
A count below 64 or 48 does not require escalation.

## 19. Amendment A5 (owner, 2026-11-07): technique coverage of E_related relative to H_train

Recorded faithfully from the owner's message of 2026-11-07. It replaces "exact proportions are
unnecessary" with a measurable coverage criterion.

1. Before treatment, freeze a short list of factoring techniques with definitions and examples.
   Assign each H_train and E_related item one **primary** technique (the factoring operation that
   does the main work in its reference solution) and record secondary techniques for description.
   Only primary tags enter the coverage calculation, so each item counts once.
2. For each technique compute its share of H_train and of E_related from the final item counts.
   If a technique is ≥ 10% of H_train, its E_related share must be ≥ half its H_train share,
   rounded up to a whole item (example: 16/64 hard items = 25% → with 64 related items, ≥ 8
   related items, 12.5%). Techniques below 10% of H_train have no individual quota.
3. Apply the check after matching related and unrelated items. Repair shortfalls only through the
   already authorized selection or augmentation of E_related, then redo difficulty and length
   matching and recompute coverage. H_train stays fixed during repair. Existing screening,
   deduplication and augmentation limits stand.
4. Thresholds are pilot design choices. If a shortfall remains after authorized repair, record
   technique, required count, achieved count and limitation, then proceed under the dataset
   viability rules. The coverage table goes in the final report. If treatment has started, audit
   the frozen pools without changing membership or restarting runs.

## 20. Amendment A6 (owner, 2026-11-07): thinking-mode baseline diagnostic

Recorded faithfully from the owner's message of 2026-11-07 (~19:05Z). It adds a diagnostic; it does
not change the validated non-thinking study, which continues unchanged with its hourly review.

Reason. Qwen3-1.7B was distilled in both thinking and non-thinking modes (Qwen3 Technical Report,
§4.5). The aim is to test whether mode choice changes task difficulty, failure modes, and the scope
of the transfer result.

1. **Panel.** Run a small baseline comparison on available project GPU capacity. Before seeing
   results, select a fixed, seeded panel of 80 distinct problems: 32 from H_train (stratified by
   source and factoring technique), 16 related auxiliary, 16 matched unrelated auxiliary, and 16 hard
   development items. Leave the sealed test set untouched. Use eight attempts per item per mode.
   Reuse non-thinking baseline attempts only if panel, checkpoint, decoding, cap and
   attempt-selection rule are compatible.
2. **Same-cap comparison first.** Compare modes at the same 8,192-token response cap. Change only
   `enable_thinking`; keep the same problem prompt, sampling parameters and verifier. Verify the
   rendered templates. In thinking mode, reasoning and final-answer tokens both count against the
   cap. Grade only the final answer after the reasoning segment. Test that a boxed candidate inside
   the reasoning cannot earn reward and that an incomplete response receives the existing
   truncation treatment.
3. **Report**, by pool: paired item-level changes in pass@1, mixed-reward group frequency,
   truncation, response lengths and GPU cost. Show whether related and unrelated practice remain
   difficulty-matched and whether the original hard pool still gives occasional successes. Inspect a
   fixed sample of failures for whether the focal factoring operation is still implicated. Eight
   attempts per item suffice for this aggregate screen; do not use them to reclassify individual
   items into the original 32-attempt difficulty bands.
4. **Longer cap, only if needed.** If thinking-mode truncation exceeds 5% on the hard panel or in
   either auxiliary pool, profile a 16,384-token cap on the same panel. If it still binds, profile a
   longer cap only if the pinned model context and measured single-GPU memory permit. Configure the
   serving context for prompt plus response. Report the same-cap comparison separately from any
   longer-budget result, since the latter changes both mode and available compute.
5. **Decision use.** Use the results to judge whether a separate thinking-mode transfer study is
   warranted. Before launching one: re-screen and match the pools under its chosen mode and cap,
   repeat optimizer calibration, and demonstrate acceptable truncation after 32 RL rounds; monitor
   length and truncation by arm thereafter. Keep distinct manifests, run IDs and results for the two
   modes. Continue the existing study and its hourly review while this diagnostic runs.

## 21. Amendment A7 (owner, 2026-11-07 ~19:20Z): GPU ceilings raised

Recorded faithfully from the owner's message. **Supersedes the "two general GPUs" ceiling** in the
execution mandate: this project may use **3 concurrent general GPUs and 6 preempt GPUs** across all
its activities. Reconcile running and pending jobs against the project ledger, use the third general
GPU for the highest-priority ready task, move an eligible pending job when that gives an earlier
start, verify each job runs only once, and keep the slot filled while useful work is ready.

## 22. Amendment A8 (owner, 2026-11-07 ~20:15Z): close the thinking-mode diagnostic; non-thinking priorities

Recorded faithfully from the owner's message. It closes A6 (§20) early and sets priorities for the
frozen non-thinking study.

1. **Close A6.** Preserve D62 and D63 as completed evidence. Withdraw the remaining A6 work,
   including the 16,384-token profile: cancel job 11078421 if pending or running (after ownership
   check) and record its cost; if finished, keep its artifacts and spend no more compute on it.
   Reassign the slot to the highest-priority ready main run, checking H-s303 first to avoid a
   duplicate. Stay within 3 general and 6 preempt GPUs.
2. **Narrow D63.** Report its counts as observations from 16 sampled failures per mode: in 9
   non-thinking cases the visible text showed no factoring attempt and 2 showed an incorrect
   factorization; 10 thinking cases were classified as truncated after factoring. A response tail
   cannot establish what happened earlier, and an alternative method can be mathematically valid.
   The hypothesis under test is whether direct practice helps the learner recognize and use
   factoring in harder problems.
3. **Baseline error audit (checklist §3.7).** Complete it on saved non-thinking screening attempts.
   Give the reviewer the problem statement and the full response where available. Distinguish:
   failure to recognize an opportunity to factor; incorrect execution; a valid alternative approach
   followed by a different error; final-answer extraction failure; insufficient evidence. Record the
   first identifiable error and keep unknowns. Check for mathematically complete answers split
   across boxes. Keep the frozen primary grading rule; report verified extraction misses separately.
4. **Cap and truncation reporting.** Keep the 8,192-token cap and the frozen completion rule for all
   main and sensitivity runs. Report truncation with pass@1 at scheduled evaluations. During
   training, log capped counts with denominators separately for hard and auxiliary prompts by arm,
   and compare R and U on their shared hard schedule. The 5% threshold governed calibration
   selection only; a later crossing triggers inspection and reporting, with no automatic change to
   the training cap.
5. **Records reconciliation at the next hourly review.** Correct stale STATE text. Document why the
   calibration panels hold 59/56 items against the original 64-per-split minimum. Reconcile
   checklist §§3.12 and 3.16 with the count of native candidates (of 184) that received 32 fresh
   attempts at 8,192 and the count excluded using 4,096-token scores. Keep frozen cohorts intact.
   Refresh the compute forecast from observed production round times and check peak storage before
   filling more slots.
Continue healthy main runs while doing the CPU-side tasks. Keep sealed test outcomes out of training
decisions and proceed autonomously through the prespecified analysis and report.

## 23. Amendment A9 (owner, received 2026-11-07 ~22:00Z): delivery deadline and reduced scope

Recorded faithfully from the owner's message. **Time-budget decision.** Deadline: Tuesday 2026-11-10 09:30
America/New_York (14:30 UTC). **Endpoint results viewed at amendment time: none.** No run had reached an
endpoint and no target_test outcome existed; only round-0 dev/probe evaluations (identical base model), training
metrics through rounds 13–26 of the three seed-101 runs, and truncation counts had been seen.

1. Run exactly six primary runs: H, R, U at seeds 101 and 202, symmetric clip (0.2, 0.2), frozen 128-round
   endpoint. Remove all seed-303 runs and all alternate-clipping sensitivity runs from the current scope; disable
   their recipes and pending dependencies; reconcile Slurm with the ledger; cancel verified project-owned jobs for
   them, preserving artifacts, status and costs; prevent resubmission.
2. Prioritize completing the six runs and their evaluations. Keep the R-s101 round-64 model and optimizer
   checkpoint and the planned local-update diagnostic from rounds 0 and 64. Use spare GPU capacity for the common
   base-model evaluation, completed endpoints and the diagnostic. Keep test outcomes out of training decisions. Do
   not fill freed slots with new experiments.
3. Start report preparation and independent recomputation as soon as possible. As each endpoint finishes, validate
   attempt counts, verifier outputs, checkpoint identity and the cost ledger. Prepare figures incrementally. Keep the
   8,192 cap and frozen grading rule. Report truncation by arm and by hard vs auxiliary prompts.
4. Lead the report with a two-row table (one row per seed): H, R, U held-out pass@1, paired R−U, paired R−H; then
   the means. Show whether direction and magnitude agree across seeds. Label any interval from resampled problems as
   conditional on these trained policies. Do not present a two-seed interval as strong evidence about training
   variability. Report local branch results separately; identify clipping sensitivity as deferred.
5. Target a complete, audited Markdown report with figures by Monday 2026-11-09 20:00 Eastern (2026-11-10 01:00
   UTC), leaving Tuesday morning for recovery. Update the forecast at each hourly review from actual throughput. If
   work will miss the deadline, deliver the verified results available by 09:30 Eastern with missing cells or
   diagnostics clearly identified. Preserve the primary comparison and its verification ahead of secondary figures.
Continue autonomously within 3 general and 6 preempt GPUs; no further approval needed for this reduced plan.

## 24. Amendment A10 (owner, received 2026-11-07 ~22:22Z): 8k-to-16k response budget

Recorded faithfully from the owner's message. Raise the response budget of the six primary non-thinking runs to
16,384 generated tokens (prompt cap 1,024 unchanged; six-run scope, GPU ceilings and the 2026-11-10 09:30 Eastern
deadline unchanged). Apply it at one common rollout-round boundary across H, R, U for seeds 101 and 202, chosen as the
earliest practical shared boundary that discards no valid work, and recorded before further comparative results are
inspected; runs below it use 8,192 until they reach it. Preserve weights, FP32 optimizer state, sampler position and
counters; keep the frozen dataset and hyperparameters. Describe the study as using an 8k-to-16k training schedule
whose questions were screened at 8k. Before release, smoke-test a complete rollout and optimizer step on one 48 GB
GPU with a genuinely long response; serving context ≥ 17,408 tokens; verify generation and trainer limits and that
nothing clips at 8,192; reduce concurrency or micro-batching if needed while preserving the effective GRPO batch.
If 16k cannot fit or measured throughput threatens the deadline, choose the largest feasible common cap between 8k
and 16k before any arm trains under the amendment, and document it. Continue the truncation diagnosis on saved
attempts (by arm and prompt type: share crossing 8,192, share of those finishing correctly by the new cap, share
exhausting it; a fixed sample inspected for productive reasoning, repetition and answer-format failures), keeping
the frozen correctness rule at the applicable cap. Use the amended evaluation cap for the base model and every
endpoint, and also report success when completion is required within 8,192 tokens from the same traces. Mark the
cap transition on learning curves; report generated tokens and GPU-hours. Keep the six primary comparisons and
their verification ahead of secondary plots. No further cap increases.

## 25. Amendment A11 (owner, received 2026-11-08 ~00:10Z): pause; cap transition moved to the end of round 32

Recorded faithfully from the owner's message. Supersedes the round-64 boundary of A10 (§24); everything else in A10
stands. Rounds 1–32 use the 8,192 cap; **round 33 is the first training round under the new cap** for every arm and
seed (H, R, U × 101, 202), through round 128.
1. Pause all six main runs now: freeze automatic submissions and requeues for the main matrix, reconcile Slurm with
   the ledger, hold pending main jobs, stop active main jobs after preserving their latest valid checkpoints, and
   verify that no main job advances. Keep the truncation study, CPU analysis and the hourly review active. Record each
   run's last completed round and checkpoint.
2. Finish the non-thinking truncation study before resuming: on the fixed development panel distinguish productive
   long solutions, repetition and final-answer extraction failures; report how often answers cross 8,192, how often
   those finish correctly with the longer allowance, and the additional GPU cost. Smoke-test 16,384 through generation,
   grading and a full optimizer step. Use 16,384 if feasible within memory and the deadline, otherwise one feasible
   intermediate cap on the same evidence. Freeze the exact cap before resuming any arm. Keep the sealed test closed.
3. Recovery: runs past round 32 restore the complete end-of-round-32 state (FP32 weights, optimizer moments, sampler
   and RNG states, counters); if unavailable, restore the nearest earlier verified full checkpoint and replay through
   round 32 at 8,192. Runs below 32 finish that phase at 8,192. Mark old round-33-and-later attempts superseded, keep
   provenance, charge compute. Analysis and W&B summaries select only the resumed trajectory.
4. Record the amendment (truncation evidence, interim outcomes viewed). Resume all six runs through round 128. Verify
   from each run's logs that round 33 uses the new cap in rollout, training and grading. Mark the transition on
   learning curves and report truncation by arm and prompt type. Update the delivery forecast after measuring
   longer-cap throughput and replay. Within 3 general / 6 preempt GPUs.

## 26. Amendment A12 (owner, received 2026-11-08 ~15:00Z = 10:00 AM EST): deadline moved
Owner's words: "You may move the deadline to 11AM on Tuesday."
Effect: the delivery deadline of A9 moves from Tuesday 2026-11-10 09:30 ET (14:30Z) to **Tuesday 2026-11-10 11:00 AM
EST (16:00Z)**. Scope, report rules and GPU limits of A9 are unchanged. Supersedes the deadline in §23 only.
Also (same session): report all times and timelines to the owner in EST.

## 27. Amendment A13 (owner, received 2026-11-08 ~16:20Z = 11:20 AM EST): round-32 provenance and report lead
Owner's request (faithful summary of the pasted instruction):
1. Record the revised report deadline, Tuesday 2026-11-10 11:00 a.m. Eastern (16:00 UTC), in project.md, STATE and
   CHECKLIST (same deadline as A12). Keep all six authorized runs progressing.
2. Reconcile the seed-101 round-32 development results: the earlier report gave R = 0.507, U = 0.547; the latest gives
   R = 0.542, U = 0.601. Trace each value to run and attempt ID, checkpoint hash, evaluation panel and sampling seeds,
   response cap, verifier version and raw attempt records. Recompute pass@1 from correct-answer counts and
   denominators. Check whether the round-32 replay, a repeated evaluation or a reporting error explains it.
3. Select the canonical series by provenance: the prespecified evaluation of the checkpoint on the retained training
   trajectory. Record the explanation and evidence in DECISIONS, preserve both evaluations, use the canonical values
   consistently in tables and figures. If unresolved, flag it and qualify any round-32-to-64 comparison.
4. Lead the final report with sealed-test pass@1 for H, R, U and the paired R−U and R−H differences for each of seeds
   101 and 202, then their means. Keep test outcomes out of training decisions. Deliver the audited Markdown report by
   the revised deadline.

## 28. Amendment A14 (owner, received 2026-11-09 ~23:25Z = 6:25 PM EST): deadline extended

Verbatim: "You may extend deadline to 2PM on Tuesday."

Given in reply to the orchestrator's note that U-s202's test would end only ≈ 2.5 h before the A13 deadline and that a
third U-s202 failure would leave its seed-202 row missing. The report deadline is now **Tuesday 2026-11-10 2:00 p.m.
Eastern (19:00 UTC)**. Supersedes the deadline of A12/A13; every other A13 requirement stands.

## 29. Amendment A15 (owner, received 2026-11-10 ~01:35Z = Mon 9 Nov 8:35 PM EST): deadline 19:30 EST; exploratory analyses

Took effect 2026-11-10 01:45Z, when recorded. Supersedes the A14 deadline. Every analysis added here is **exploratory**;
the primary estimand, protocol and analysis plan are unchanged.

**Endpoint results visible when A15 took effect (D122):** no sealed-test outcome (pass@1, pass@8, per-problem counts) of
any model had been read. Visible from the sealed test: structural fields of the completed test evaluations (base,
R-s101, U-s101, H-s101: completions 4,096, cap 12,288, token totals, throughput, and the truncated share — base 0.05%,
R-s101 10.7%, U-s101 12.2%, H-s101 10.9%). Visible non-sealed: dev and probe curves of all runs to round 128 (s101), 96
(R-s202) and 64 (U-s202) (D115). Test R-s202 was running; U-s202 still training (round ≈ 85).

Verbatim:

> The revised deadline is Tuesday, November 10, 2026, at 19:30 America/New_York (Wednesday, November 11 at 00:30 UTC).
> Update project.md, PROTOCOL, DECISIONS, STATE, and CHECKLIST. Record when this amendment took effect and which endpoint
> results were already visible. Label the added analyses exploratory.
>
> Complete the six authorized H/R/U runs for seeds 101 and 202 under the frozen protocol. First reconcile Slurm, the run
> ledger, retained checkpoints, and measured evaluation throughput. Reserve enough time for all six endpoints, the original
> local-update diagnostic, independent recomputation, and the Markdown report. Use the extra time in the following priority
> order.
>
> 1. Determine which problems improve and how reliably. Use the existing 32 sealed-test attempts per problem to report
> pass@1 and estimated pass@8 for each arm and seed. Show paired R−U and R−H differences for both metrics. Plot per-problem
> changes from the base model so we can see whether gains are broad or concentrated in a few questions. For a breakdown by
> initial difficulty, assign bands using a fixed 16 of the 32 base attempts, then use the other 16 to estimate baseline
> accuracy within those bands. Use the same bands for every arm and report their sizes. Describe zero observed successes as
> 0/32. This analysis requires no new generations.
>
> 2. Trace interference on the hard problems shared by R and U. Within each seed, align observations by problem ID and
> training round using the matched hard-prompt schedule. In fixed round windows, report hard success and the fractions of
> eight-answer groups that are all wrong, mixed, or all correct. A decline in mixed groups can mean either mastery or loss of
> reward signal; show which occurred. Report truncation separately, distinguishing all-wrong groups in which every answer
> finished from those containing capped answers. Mark the round-33 response-cap change. Compare these training trajectories
> with the fixed development panel to see which patterns extend beyond repeated questions.
>
> 3. Finish the prespecified local-update diagnostic at the base and round-64 R-s101 checkpoints. Report the change in hard
> development success after hard, related, and unrelated updates relative to the corresponding no-update control. Show
> individual minibatch repeats. If the round-64 U-s101 checkpoint and its optimizer state are already available, repeat the
> diagnostic there using the same panel, minibatches, seeds, update size, response cap, and controls. This tests whether the
> effect of an auxiliary update depends on the preceding training mixture. Skip the U extension if it requires replaying a
> substantial training trajectory or puts primary delivery at risk.
>
> 4. If time permits, run one paired strategy-cue diagnostic on harder factoring problems. Its question is whether related
> practice helps the model use factorization when prompted and whether that benefit also appears when the model must
> recognize the opportunity itself.
>
> Select 24 previously unused, verifier-compatible MATH problems whose worked solutions centrally use factoring and require
> another decision, such as constructing an equation, choosing a substitution, or converting factors into a count. Prefer
> unused official MATH test items. Exclude overlaps and close variants of every training, calibration, development, probe,
> and sealed-test item. Freeze the IDs and rubric before evaluating trained policies. Select by solution structure and report
> the measured baseline difficulty afterward.
>
> Evaluate each question as written and with one fixed cue appended: "Consider whether a factorization could simplify the
> problem." Keep the decoding settings, answer format, verifier, and response cap identical. Evaluate the base model and the
> four R/U endpoints across the two seeds, with eight attempts per item per form: 24 × 2 × 5 × 8 = 1,920 completions.
> Forecast the cost from measured throughput before launching.
>
> Report uncued pass@1, cued pass@1, and the cue gain for each policy. For each seed, report (R cued − R uncued) − (U cued −
> U uncued), preserving item pairing when estimating uncertainty. Include truncation and response lengths, and inspect a
> fixed sample to see whether the cue changed the solution method. A larger cue gain for R would suggest that naming the
> strategy helps R express what it learned. Treat that interpretation as tentative on this 24-item panel.
>
> Lead reports/REPORT.md with the sealed-test H, R, and U scores and paired R−U and R−H differences for each seed, followed
> by their means. Then address three questions separately: Did the choice of auxiliary practice affect held-out success? Did
> an unrelated update reduce hard success at a measured checkpoint? Did previously solvable hard problems subsequently
> produce more all-wrong groups? State which links the evidence supports and how uncertain they are. Keep the strategy-cue
> diagnostic separate from the evidence for ray interference.
>
> Continue autonomously within three general and six preempt GPUs. Use existing records before spending GPU time. Keep
> training decisions independent of exploratory outcomes. If the schedule tightens, drop the strategy-cue panel first, then
> the optional U-checkpoint branches. Deliver the verified primary results, figures, and cost reconciliation by the
> deadline, with any omitted diagnostic identified.

## 30. Amendment A16 (owner, received 2026-11-10 ~04:45Z = Mon 9 Nov 11:45 PM EST): partial H-s202

In reply to the orchestrator's note that H-s202 cannot finish by the deadline and the offer of a partial run. Verbatim:

> I'd prefer the partial run, and an appendix explaining results on the partial run at similar rounds for R-202 and
> U-202. This is not very important, and should not take precedence over suggestions and nice-to-haves as elaborated on
> earlier.

Implementation (D128): H-s202 runs under its frozen a2 recipe at the **lowest priority**, only on a general GPU no A15
item needs (after the round-64 diagnostic), and is stopped in time for the report. It is not an endpoint: no sealed-test
evaluation; seed-202 H cells of the primary table stay missing. A report appendix compares it with R-s202 and U-s202 at
matching rounds (dev evaluations and shared-schedule training windows).

## 31. Amendment A17 (owner, received 2026-11-11 ~01:25Z = Tue 10 Nov 8:25 PM EST): complete the GRPO mixture study

Supersedes the A15 deadline and the A16 "partial H-s202, no test" scope. New deadline: **Wed 11 Nov 2026 7:30 PM EST =
Thu 12 Nov 00:30Z**. Verbatim:

> # Complete the GRPO mixture study
>
> You have until **Wednesday, November 11, 2026 at 7:30 PM EST** to complete the experiments, verify the results, and deliver `reports/REPORT.md` with its figures. Begin immediately. Exercise the judgment of a senior research scientist. You have authority to schedule, resume, cancel, and repair this project’s jobs, resolve routine blockers, and reallocate compute as evidence and throughput warrant.
>
> Read `project.md`, the frozen protocol and amendments, `research/STATE.md`, `research/CHECKLIST.md`, `research/DECISIONS.md`, and the current report. Record this instruction as an amendment. **Launch H-s202 as soon as its checkpoint is verified.** Start the common-GPU evaluations, panel construction, and CPU analyses in parallel wherever resources permit.
>
> Respect the project limit of **3 general GPUs and 6 preempt GPUs**, account-wide availability, and the shared storage quota. Verify job ownership against the ledger before acting. Resume the **hourly review and monitoring loop** now. At each review, record what changed, which competing explanation remains plausible, which next task would distinguish it, and whether the current schedule still delivers a verified report on time. Update state, checklist, and ETAs from observed throughput.
>
> ## 1. Keep the comparisons and their purpose clear
>
> The scientific question is whether the choice of easier practice changes how a model learns harder problems during on-policy RL. R and U receive the same eight hard prompts each round. R spends its other eight slots on factoring exercises, U on matched modular-arithmetic exercises, and H on eight additional hard problems.
>
> - **R versus U** asks whether the choice of auxiliary skill matters at equal hard-prompt exposure.
> - **R versus H and U versus H** ask whether either use of those eight slots was more valuable than additional hard practice.
> - **Base versus trained checkpoints** measures the overall gain during training. It provides context for the smaller differences between mixtures.
> - **The branch experiments** ask what one update did at particular checkpoints. **The training records** ask whether a loss of success persisted and reduced opportunities to learn. **Held-out tests** ask whether any effect transferred to different problems.
>
> Keep the frozen 128-problem test primary. Identify the common-GPU evaluation as a reproducibility check and the new panels and mechanism analyses as exploratory. Publish the original results alongside subsequent measurements. Once trained outputs are visible, use them for analysis only. Freeze panel selection, scoring, and comparison rules before examining those outputs.
>
> ## 2. Finish the missing control and verify the endpoint comparison
>
> **Complete H-s202.** The report says the run reached round 64, while the latest update identifies round 60 as the resumable checkpoint. Verify the latest durable model, optimizer, RNG state, and prompt schedule. Check storage headroom. If round 60 is the latest valid save, resume there, retain the original round-64 records, and identify the trajectory used for the endpoint. Complete round 128 and run the frozen sealed evaluation. If the deadline intervenes, report the last verified round on the development curves; only a round-128 test belongs in the endpoint comparison.
>
> **Evaluate the four R/U endpoints on one GPU type.** First inventory checkpoint hashes, hardware, inference settings, and evaluation seeds for the original evaluations. Establish whether each original R/U pair already shared a GPU type. Freeze one common hardware and evaluation recipe before viewing rerun scores. Use the original 128 problems, 32 attempts, verifier, and 12,288-token cap. Include both H endpoints on that recipe as feasible, avoiding duplicate evaluations where one job satisfies both requirements.
>
> Report absolute H, R, and U scores and paired R−U, R−H, and U−H differences for each completed seed. Does the R/U reversal survive the common setup? The earlier 2.5–3.1-point discrepancies came from a smaller hard probe. Measure endpoint reproducibility directly. A rerun that changes hardware and sampled responses cannot isolate a hardware effect. Intervals over problems describe the evaluated checkpoints; two training seeds give limited information about the next training run.
>
> ## 3. Explain why the two seeds gave different answers
>
> Use existing records for these analyses. They are high priority and can proceed while GPUs run.
>
> 1. **Identify the problems behind the reversal.** For each original test problem, plot its R−U difference in seed 101 against its R−U difference in seed 202. Show direction agreement, ties, and sampling uncertainty. Compute each existing difficulty band’s contribution to the overall reversal, weighting by its share of the 128 problems. Examine the 34-problem B1 band in particular. Did a few items dominate, or did the ranking change across much of the band? Give the absolute R and U scores in both seeds.
> 2. **Separate broader coverage from more reliable answers.** Report per-problem success rates, pass@1, pass@8 where available, and the number of distinct problems solved at least once at a fixed attempt count. Retain the independent baseline split used to define difficulty bands. Describe a problem with zero observed successes as `0/n`. Explain whether training made more problems reachable, improved reliability on already-reachable ones, or did both.
> 3. **Measure how much GRPO signal each arm received.** By seed, training window, and prompt role, show all-wrong, mixed, and all-correct groups. Separate the eight shared hard prompts, H’s extra hard prompts, and each auxiliary pool. Alongside these fractions show cumulative mixed-group counts, generated tokens, and GPU-hours. Determine whether an auxiliary pool became so easy that its groups frequently contained no reward contrast. A mixed group establishes reward variation; its count does not determine gradient size or direction. Use existing gradient, loss, and clipping logs only at the granularity actually recorded.
> 4. **Test whether apparent forgetting persisted.** On the shared hard schedule, match R and U by problem and round. Report their discordant all-wrong groups, which previously solvable items later became all-wrong, and what happened at the next appearance of those items. Give eligible denominators and count items with no later observation. Mark the cap change after round 32. For all-wrong groups, distinguish zero, some, and all eight answers hitting the cap. One capped answer in a group does not establish that a larger allowance would have rescued it.
> 5. **Estimate how often sampling alone creates an apparent loss.** Resolve the report’s discrepancy over whether development transitions used eight or sixteen attempts. Inspect raw records and regenerate affected results. Split repeated attempts at an unchanged checkpoint into two disjoint groups of the appropriate size and measure how often a problem appears to go from some success to zero, or vice versa. Prefer the same development items where records permit. Otherwise use sealed-test attempts and document the panel difference. Repeated splits reuse data. Use this as a reference for interpreting observed misses, rather than counting each split as an independent experiment.
>
> ## 4. Audit the local evidence for interference
>
> At the base and R-s101 round-64 snapshots, compare the no-update, hard-only, related-only, and unrelated-only branches. Give absolute pre- and post-update values of the hard, factoring, and modular probes (`J_H`, `J_R`, `J_U`), the corresponding changes, the four minibatch draws, and the mixed-group count. Show direct unrelated-versus-related and unrelated-versus-hard comparisons because the branches share a noisy no-update reference.
>
> Distinguish an unchanged checkpoint from an AdamW step with zero current reward gradient but nonzero stored moments. Document the unavailable U checkpoint. Use observed variation to qualify the diagnostic’s resolution; avoid claiming a detection threshold without a supporting analysis.
>
> Assess the proposed mechanism one link at a time: **Did an auxiliary update lower hard success? Did affected problems subsequently produce fewer rewarding attempts? Did their learning then stall?** Specify which links have direct evidence, which are observational, and which remain unresolved. A held-out endpoint difference alone cannot establish this chain.
>
> ## 5. Build two independent factorization stress panels
>
> Create **Harder-128** and **SuperHard-128**, each with 128 distinct, verifier-compatible problems from varied, traceable sources. Their purpose is to test whether training gains extend to harder problems and to identify what prevents further gains.
>
> - Write the selection rubric first. **Harder** problems should require recognizing a substantive factoring step inside a longer solution without an explicit cue. **SuperHard** problems should add a documented obstacle, such as a hidden substitution, a consequential choice of approach, or a case analysis after factoring. Record the obstacle for each item. Long statements and large coefficients alone do not establish difficulty.
> - Seek at least three identifiable source collections per panel. Record original provenance, problem ID, answer, factoring technique, answer type, and required reasoning. Check exact text, normalized expressions, near-duplicates, and solution templates against both panels and every training, calibration, practice, dev, probe, cue, and sealed-test set. Keep close variants out of opposite panels. Disjoint panels do not establish that the base model never encountered these problems in pretraining.
> - Preserve enough overlap in sources and techniques to make the tier comparison interpretable. Publish the actual composition. Verify gold answers and equivalent forms under the frozen grader. Accept valid alternative solution methods. Identify agent annotations as such.
> - Freeze the manifests, family IDs, answers, rubric, and hashes before evaluating trained checkpoints. Measure actual difficulty with fresh base-model attempts. If the tiers overlap or SuperHard is near the floor, retain and report that result. If fewer than 128 items can be validated by delivery, state exactly how many qualified and why.
>
> ## 6. Evaluate the stress panels at baseline, during training, and at the endpoint
>
> Inventory retained checkpoints. Evaluate both panels uncued on the common base, at least one comparable saved intermediate round across available arms, and all six final checkpoints as they become available. Prefer round 64 for the intermediate comparison if coverage permits. Publish a checkpoint-by-panel coverage table. Use retained saves and record missing comparisons.
>
> Keep the verifier, sampling settings, attempt schedule, and 12,288-token cap common across models. Benchmark throughput on a small representative batch. Target **eight attempts per problem**. If necessary, complete a balanced four-attempt pass across both panels and comparable checkpoints before deepening any cell. Compute comparisons at a common attempt count and report pass@8 only where it is supported.
>
> For each tier and seed, give absolute pass@1, coverage at the stated attempt count, truncation, correct/total attempts, and distinct problems solved. Distinguish several successes on one problem from first successes across several problems.
>
> For a small fixed sample, inspect full responses alongside the prompt, worked solution, and verifier outcome. Preselect item IDs and attempt indices, compare the same items across policies, and hide arm identities during annotation where practical. Record whether the model missed a useful factoring opportunity, factored incorrectly, factored correctly and failed later, solved by another method, hit the cap, or exposed a grader issue. Allow “unclear.” Give denominators and a few traceable examples. Use this audit to answer **whether more direct factoring practice would plausibly address the failures we observed**.
>
> ## 7. Deliver a report the reader can interpret from the figures and absolute scores
>
> Embed the actual figures in `reports/REPORT.md` with relative image links. A filename in prose is insufficient. Place each figure beside the claim it informs. Inspect the rendered Markdown from a clean checkout.
>
> Required figures:
>
> 1. **Original and common-GPU endpoints:** absolute H/R/U scores by seed, with paired contrasts and intervals alongside them.
> 2. **Learning curves:** absolute dev and probe scores against training round and cumulative hard exposure. Mark the cap change and distinguish practised problems from held-out evaluations.
> 3. **POPE-style J plots:** plot `J_H` against `J_R`, and separately against `J_U`. Define each probe, connect observed checkpoints in time order, label rounds, and use comparable axes.
> 4. **Measured ΔJ plots:** show individual one-step changes by branch and checkpoint, including the no-update reference and the corresponding absolute scores in the figure or an adjacent table. Draw only changes the experiment measured.
> 5. **Explanatory figures:** include the per-problem seed-reversal scatterplot, hard-group transitions and subsequent recovery, and base/intermediate/final results on both stress panels. Combine panels where this improves readability.
>
> **Never report a delta alone.** Give the metric, both absolute values, subtraction direction, and panel or denominator in the same sentence, figure, or adjacent table. For example: “On the 128-problem sealed test, seed-101 R scored 81.9% and U scored 77.7%, so R−U was +4.2 percentage points.” Apply this rule to the executive summary, branch diagnostics, cue comparisons, and stress panels. Absolute scores show the remaining headroom.
>
> Open the report with a concise **What we learned** section that answers six questions:
>
> 1. How much did training improve hard-math success over base, and how much additional difference did the choice of practice make?
> 2. Given eight discretionary prompt slots, what do the two seeds say about choosing factoring, modular arithmetic, or more hard practice? How do their token and GPU costs affect that judgment?
> 3. Which problems produced the seed reversal, and did the differences reflect coverage, reliability, or both?
> 4. Which links in the proposed ray-interference mechanism did we observe? What competing explanation best fits the same evidence?
> 5. How far did gains transfer across the original, Harder, and SuperHard panels? Where could we distinguish missed strategy recognition, faulty execution, later reasoning errors, and truncation?
> 6. Which **one** larger experiment is most justified by the dominant uncertainty, and what result would change that recommendation?
>
> ## 8. Protect the deadline and verify the deliverable
>
> Prioritize H-s202, the four common-GPU R/U evaluations, the CPU analyses, and a balanced stress-panel comparison. If throughput falls short, reduce repeated attempts and optional intermediate rounds before sacrificing balanced coverage or verification. Avoid launching new training seeds, new arms, or another domain. Make routine tradeoffs autonomously and record their consequences.
>
> Independently recompute headline tables from attempt records, reconcile costs against the ledger, verify checkpoint and manifest hashes, and check every figure against its source table. Deliver the **Markdown report with embedded figures, both versioned panel manifests, machine-readable results, reproduction commands, and a clear inventory of incomplete cells** by the deadline. The report should leave the reader able to make a training decision, understand why that decision is uncertain, and identify the next experiment that could change it.

## 32. Owner instruction A18 (received 2026-11-11 ~15:55Z = Wed 10:55 AM EST): state and checklist current

> Please keep your state and checklist current. Include the timeline and GPUs in use in your state.

Applied: `research/STATE.md` carries a "Timeline" table (EST and UTC, from observed throughput) and a "GPUs in use" table,
updated at every hourly review and at every submission or job end. This overrides the harness rule that STATE holds no job
states, for these two sections only. Recorded as CHECKLIST 0.25.

## 33. Amendment A19 (owner, received 2026-11-11 ~16:03Z = Wed 11:03 AM EST): use the remaining GPU allocation to finish the study and explain the length effect

Verbatim:

> # Use the remaining GPU allocation to finish the study and explain the length effect
>
> Continue autonomously through **Wednesday, November 11 at 7:30 PM EST**. Record this amendment in `project.md`, `research/STATE.md`, `research/CHECKLIST.md`, and `research/DECISIONS.md`. Keep the hourly scientific reviews running. Start by reconciling live Slurm state, the project ledger, retained checkpoints, and free storage. The 10:58 AM snapshot is the starting point; schedule from what is true now.
>
> ## 1. Protect the final H-s202 result and shorten its critical path
>
> Keep H-s202 training uninterrupted. Prepare the six sealed-test shards, the stress evaluation, and the merge and verification jobs now. Check what actually happens between the final optimizer update, the durable round-128 save, and `RUN_COMPLETE`. If the final model can be verified before the marker, release evaluation as soon as the immutable weights, round number, final-step provenance, and checksum all pass. Verify afterward that the completed run identifies the same weights. If the marker provides an integrity guarantee you cannot independently establish, retain the existing gate.
>
> Apply the authorized intermediate fallback rule. Keep its scores clearly labeled with the actual round and outside the primary endpoint table. Once the verified endpoint is available, prioritize its complete 128-problem sealed evaluation, followed by its frozen stress-panel cell. Use independent shards with unique attempt IDs and verify the full set before publishing a score. Do not delay the final endpoint to finish optional diagnostics.
>
> ## 2. Use spare GPUs now to test whether longer answers would actually succeed
>
> This is the highest-value additional experiment. The original seed-202 test scores were R 74.8% and U 79.2%. Their difference was largely associated with truncation. Accuracy among responses that finished cannot tell us what the capped responses would have done with more room.
>
> - Run an **exploratory evaluation-only length diagnostic** on the four final R/U checkpoints, covering both seeds. Use the frozen stress panels, where the reversal also appeared. Before generating anything, draw a fixed sample stratified by panel and source, independent of arm outcomes. A practical first tranche is **64 problems, 32 per tier, four attempts per problem and checkpoint**. Use the same problems and attempt schedule for all four models. Preserve the original sealed-test and stress scores.
> - First inspect existing raw records. If exact continuation from a saved capped prefix is supported and validated, extend eligible attempts from **12,288 to 16,384 response tokens**. Preserve the original tokens, remaining context capacity, and grading rule. Otherwise, generate fresh trajectories to 16,384 and score **each trajectory twice**: once as it would have scored under the frozen 12,288 rule, and once at 16,384. This gives a within-trajectory comparison even if a fresh rollout differs from the original. Test the cutoff implementation on boundary cases before launching the full batch.
> - Report, per arm and seed, **success at both caps; attempts reaching the first cap; additional correct answers; answers that eventually finished incorrectly; answers still capped; and extra generated tokens per additional success**. Show whether R−U changes direction or magnitude at the longer cap. Keep denominators visible. Use paired problem-level uncertainty and report sparse counts plainly.
> - Inspect a small, preselected sample of extended responses. Did extra tokens complete sound reasoning, repair an error, continue a loop, or only delay an incorrect answer? If few answers recover, that is informative. If many recover, quantify whether recovery favors one arm. Either outcome concerns **inference allowance for these checkpoints**. It does not establish what training with a longer cap would have produced.
>
> Benchmark a small shard first. Give the four checkpoints equal coverage before adding attempts. Use one GPU type where feasible, and record hardware and sampling settings.
>
> ## 3. Fill the most informative checkpoint gaps
>
> Inventory actual retained weights before submitting jobs. The current stress table includes **R-s101 at round 64 and H-s202 at round 64**. Those cells come from different arms and seeds, so they cannot show when R and U began to diverge within a seed.
>
> If suitable checkpoints exist, evaluate the missing partner for a **same-seed, same-round R/U comparison** on both stress panels. Prefer a round with both partners available. Show absolute success, truncation, and accuracy among finished answers for both arms at that round and at the endpoint. If no matched intermediate pair survives, use the fixed development evaluations and training logs to locate when lengths and success began to diverge. State the limitation and avoid retraining to recreate missing weights.
>
> Keep the stress panels frozen at their validated sizes, **Harder-95 and SuperHard-126**. Report each tier separately, including baseline success and truncation, so the reader can judge whether SuperHard was empirically harder. Do not append questions after seeing the results.
>
> ## 4. Keep useful work queued and the report audit moving
>
> Use up to **3 general GPUs and 6 preempt GPUs**, subject to account availability, with at most 30 CPUs under the existing guard. Submit ready, restartable diagnostic shards to preempt now. Backfill free general slots with shards that can finish or yield before the endpoint needs them. Preserve completed attempt records across preemption. Use exact ledger-owned job IDs, prevent duplicate execution, and leave other projects' jobs and files alone. Do not reserve speculative standbys.
>
> At each job completion or preemption, reconcile the queue and refill eligible slots promptly. In `STATE.md`, distinguish **running, queued, and genuinely blocked** work. For unused capacity, give the concrete reason and next eligible job. Keep the CPU verification, figure checks, and clean-checkout render moving in parallel. Once outputs are verified, idle capacity is preferable to another redundant rerun.
>
> ## 5. Deliver the scientific answer
>
> The report must let us answer four practical questions below. Specifically, update your checklist to necessitate these analyses in the final report in addition to existing item:
>
> 1. **Was related practice worth the training slots?** Give base, H, R, and U held-out scores for each seed, plus prompt, token, and GPU costs. State which comparisons replicate across seeds.
> 2. **Did the policy lose mathematical accuracy, or did it become less likely to finish within budget?** Present the existing descriptive decomposition beside the new within-trajectory length experiment. If the answers differ by seed, explain each seed separately.
> 3. **When did the difference emerge, and did it extend to harder problems?** Use comparable checkpoints, show absolute scores for both stress tiers, and distinguish measured trajectories from gaps in checkpoint coverage.
> 4. **What should the next experiment change?** Decide whether the strongest evidence calls for more training seeds, better control of response length, or a different choice of auxiliary practice. Name the observation that most influenced that judgment.
>
> Show both absolute values beside every delta. Independently recompute the headline results, check the embedded J and ΔJ figures against their source tables, and reconcile the final costs. One specific audit item: `CHECKLIST.md` reports 29 "yes," 27 "no," and 10 "unclear" judgments out of 70 failure reads. Those counts sum to 66. Resolve the remaining four from raw records and distinguish **35 failed responses** from **70 annotator reads** before interpreting that audit.
>
> Deliver the verified Markdown report, figures, tables, and a precise inventory of any incomplete cells by **7:30 PM EST**. If time tightens, preserve the final H-s202 sealed test, a balanced four-checkpoint length diagnostic, and independent verification first.

Reconciliation at receipt (16:05Z): H-s202 11112377 running (general, round 109 at 15:40Z, LATEST round 108); CPU standby
11112647 pending (Priority). Retained weights: H-s101 r128; H-s202 r64, r108; R-s101 r64, r128; R-s202 r128 (+ two `.tmp`
partial saves r4, r12, not usable); U-s101 r128; U-s202 r128. No same-seed R/U pair exists at any intermediate round.
Storage: 359/500 GB used, 142 GB free (project 188 GB).
