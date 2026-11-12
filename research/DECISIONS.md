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
| D13 | Annotation pass-1 batches 2 and 5 discarded and redone in isolated directories (`annot/iso2`, `annot/iso5`). Contaminated outputs kept as `annot/superseded/*.contaminated.jsonl` (laptop scratch; summary here). Batch-4 has 27 paraphrased (non-verbatim) `support` quotes; those items get pass-2 review like all selected items. | Six parallel annotators shared one scratch directory and overwrote each other's helper scripts. Check: `support` must be a substring of the item's own solution. Batches 0, 1, 3: 0 failures. Batch 2: 123 failures (93 quoted from batch 3); batch 5: 124 (118 from batch 3). No downstream use had occurred. Rule: every annotator gets its own directory, and every pass is validated by the substring check before merge. |
| D14 | vLLM runs with the PyTorch-native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) in every job (screening, training, evaluation). | Probe attempt 2 (job 11072208) failed in vLLM warm-up: FlashInfer's top-k/top-p sampler JIT needs `ninja`. With top_p=1 and top_k disabled the native sampler draws from the full softmax at T=1, with no JIT kernel to audit. Same sampler everywhere, so no comparison mixes samplers. |
| D15 | Patch `venv-v1` in place (`qis-env-v1b`): `omegaconf>=2.3,<2.5` and `ninja`. | uv resolved omegaconf 1.4.1; verl 0.9.0 then fails to import (`MISSING`), which blocked only the verl parity test. No training or screening has used the venv yet. Freeze diff saved in harness `results/qis-env-v1b/`. |
| D16 | Rubric v1.1: role pools defined by `direct_practice` (H/targets = central, direct_practice=no, level 4–5; E_related = central, direct_practice=yes, any level), plus edge-case clarifications. Supersedes the level-based rule of D9 for roles. | Pass-1 counts (`data/annotations/pass1.jsonl`, sha256 prefix `76d6d675`): v1 rule gives E_related 69 construction candidates, below what the 12–24/32 band and matching need for 64. v1.1 gives H 154, E_related 110, E_unrelated 152 (construction); target_dev 63, target_test 209. Decided before any screening sample; independent of model outcomes. The v1.1 rule matches `project.md` §5 ("practise that operation more directly") more closely than a level cut. Risk recorded: E_related may still fail the 64 minimum after screening; then the authorized fallback applies. |
| D17 | Addendum A2 recorded (`project.md` §16). Affects stage 3 (dataset construction) only: auxiliary pools target 64 matched pairs, modest shortfall allowed with documented coverage/diversity/balance; synthetic supplementation allowed under a pre-recorded gap and budget, identical audit/screen/matching, provenance balanced across R and U, template families tracked, no training-template variants in evaluation sets. Choice among smaller natural set / supplemented set / fallback skill is made after screening and frozen before any treatment run. No run affected (none exists). | Owner message 2026-11-07, received while stage-1 screening was not yet started. |
| D18 | Freeze the response cap at **4,096 tokens**. | Length probe `qis-lenprobe-v1` (11072430, L40): 64 construction hard-like items × 16. At 4,096: 2.83% truncated (29/1,024), 0 truncated responses would have matched, success 0.6406, mean length 1,219. At 8,192: 0% truncated, success 0.6406 (identical), mean 1,226. Below the 5% rule; 8,192 resolves nothing that changes reward. |
| D19 | Pre-register an **H reserve tranche** before any screening outcome: construction items labelled central, `direct_practice=no`, MATH level 3 (30 items). Screened in stage 1 with the main pools; added to H_train candidates only if the primary H pool (level 4–5) yields < 64 items after stage 2, in its fixed seeded order. target_dev/target_test stay level 4–5. | Length-probe success distribution on hard-like items (20/64 at 16/16; 19/64 in 0–5/16) projects ~30 H_train items in the 2–8/32 band from 154 candidates, below the hard minimum 64 (not relaxed by A2). Level is MATH author difficulty, not model success; the stage-2 band still enforces hardness. Recorded before any study-candidate screening. |
| D20 | Amendment A3: CPU ceiling 30 CPUs on `preempt_cpu_qos`. `ops/qis-submit` now sums `%C` over owned nonterminal CPU jobs and refuses a submission that would exceed 30 (replaces the 2-CPU-job slot rule of D6/D8). | Owner message 2026-11-07. |
| D21 | `registry/` and `ops/` (Slurm submission guard, review bookkeeping, run registry) removed from git tracking and git-ignored; files stay on the laptop and keep working. Earlier commits are not rewritten. | Owner instruction 2026-11-07. Supersedes the `project.md` §14 listing of the run registry as a tracked record; run identity remains in the harness ledger and in per-run records on compute storage. |
| D22 | Correction of the record: trainer-smoke attempts are 11072408 (cancelled before start), 11072428 (failed, checksum), 11072527 (failed, audit OOM), 11072622 (current). An earlier status message cited 11072527/11072528 wrongly; 11072528 is another project's job and was never touched (all control goes through `ae slurm`, which refuses non-ledger IDs). | Ledger rows vs `sacct`, 2026-11-07 03:45Z. |
| D23 | Move calibration screening `qis-screen1-calib` from preempt to general. Preempt job 11072610 stays queued until a general slot frees (both held by 11072896 screen2 and 11072622 smoke); then cancel 11072610, confirm terminal, resubmit on general with the same recipe content and stream names (`calib16`). | 11072610 pending (Priority) 35 min at 03:54Z; `project.md` §3: calibrate through general when preempt takes > 30 min to start. No scientific change (same items, seeds, decoding). |
| D24 | **Gate G2 passed** (job 11072622, RTX 6000 Ada, code `d1b38a0`). Evidence `research/evidence/g2/`. Precision (A1): optimizer-owned params, grads, AdamW moments FP32 (310/310 tensors); rollout BF16; checkpoint-on-disk BF16; sync == bf16(learner) exactly; restart bit-exact weights + moments; first-step grad norm 1.14; updates accumulate. Resume: rounds 1–4 each committed once, step counters equal. Uninterrupted vs resumed runs diverge from round 2, i.e. BEFORE the interruption: this is run-to-run kernel nondeterminism, the documented tolerance; resume correctness rests on the bit-exact restore. | `project.md` §8 and §15. |
| D25 | Log-prob mismatch alarm (vLLM BF16 sampling vs FP32 learner, same tokens): investigate if a round's |Δ| p99 > 0.20 or p99.9 > 0.50 (smoke baseline: p99 0.10–0.11, p99.9 0.21–0.23, median ~1e-7). | Tolerance established on the pinned backend (§8). |
| D26 | Checkpoint every 4 rounds with the background writer (21.9 GB, 197–245 s write, validated). While it writes, training steps slow (round train time 23 s → 80–123 s); accepted for now, to be re-measured on real rounds; option: write from a subprocess to avoid contention. | Smoke metrics; keeps expected lost work ≈ 2 rounds (< 15 min at the projected 2–3 min/round). |
| D27 | Amendment A4 recorded (`project.md` §18, `research/PROTOCOL.md`). Affects stage 3 (H_train construction, endpoint) only; calibration and trainer validation continue unchanged. Immediate actions: stage 2 (32 attempts, stream `s2`) on all 184 hard + reserve candidates, reusing valid `s2` attempts of job 11072896; add solution-structure family labels to the blind pass-2 audit of hard candidates; selection rule (core 2–8, moderate 9–12, ≥ half core, structure diversity, seeded ties) fixed before the full stage-2 counts are seen. | Owner message 2026-11-07, before any treatment run. |
| D28 | H_train selection rule fixed in code (`src/qis/data/select_hard.py`) and rubric v1.2 families (F1–F8) committed BEFORE any full stage-2 count was viewed: eligibility = pass-1 and blind pass-2 agree (central, multi-step), 32 fresh attempts, no unknowns, band core 2–8 or moderate 9–12, truncated attempts < 50% of failures; composition = all core up to 64, then moderate up to min(64−C, C); family round-robin with seeded order (`select_hard/order`). L3 reserve items enter on equal terms with source recorded (A4.1). | A4.2: ties by recorded seeded order, structural diversity, ≥ half core. Committing the rule first prevents outcome-driven selection. |
| D29 | Amendment A5 recorded (`project.md` §19, PROTOCOL, CHECKLIST 0.11/3.15). Technique list = rubric v1.2 families F1–F8 with definitions and examples, frozen now. Primary tags for H_train come from blind pass 2 (running); E_related items and secondary tags get the same blind tagging before matching. Coverage check runs after R/U matching, before freeze. | Owner message 2026-11-07; no treatment run exists, so membership may still change only through authorized E_related repair. |
| D30 | Owner tagging rule (2026-11-07) adopted as rubric v1.3: primary tag = specific factoring operation (F1 trinomial, F2 special product, F3 grouping/SFFT, F4 factor theorem/known-root extraction, F8 other); substitution, rational simplification, sign analysis etc. are separate step tags. Re-tag all 184 hard candidates and all 110 E_related candidates blind under v1.3 (tags only; eligibility labels unchanged). Third read resolves the 26 hard and 33 target/probe pass-1/pass-2 disagreements, before any full stage-2 count is viewed. | Owner message 2026-11-07 (A5 audit instruction). The audit table will show F4 and F5 definitions with one worked step each. |
| D31 | `select_hard.py` updated before any full stage-2 count was viewed: eligibility uses pass-1/pass-2 agreement or a third-read resolution to central multi-step; diversity family = v1.3 primary operation (D30). | Consistency with rubric v1.3 and the rubric's third-read rule. |
| D32 | **Auxiliary gap and authored tranche (A2, A5).** Measured gap: natural E_related in band 12–24/32 = 7 (target 64 pairs); natural E_unrelated = 18. Natural related items are mostly solved 32/32 (direct factoring is easy for the model). Remedy: ONE authored tranche per arm, budget **200 related + 200 unrelated candidates**, stopping rule: author once, screen once (16 → shortlist 4–14/16 → 32 attempts → band 12–24), no second tranche; then match. Design: (a) related items = direct practice of one primary operation, stratified 40 each over F1, F2, F3, F4, F8 (A5 coverage), surrounding steps allowed; (b) unrelated items = direct modular computation, 40 each over: remainder of a power, linear congruence, modular inverse, last digit(s), simultaneous congruences (CRT); (c) numeric or short closed-form answers only — no "factor the expression" answer formats, because Math-Verify accepts an expanded polynomial as equal to its factorization; (d) every authored item carries a template-family id; minor variants count as one family; (e) same blind annotation (pass 1 + pass 2), independent answer check by computer algebra on a CPU job, Math-Verify gold self-check, MinHash dedupe against all MATH splits (no near-duplicates of dev/test), same screening and matching; (f) provenance balance: R and U are matched within provenance (natural with natural, authored with authored), so both arms have identical synthetic shares. Authoring is done by isolated agents that never see any model output or screening result. | A2 authorizes supplementation of either auxiliary pool after recording gap and budget; A5 coverage needs F3/F4 related practice that natural items barely supply (related candidates: F3 8, F4 7). |
| D33 | Calibration panels frozen (job 11072990): calib_train 62 items (low 1–5/16: 27, high 6–12/16: 35), calib_val 51 items (20 / 31); bundle sha256 `f7f34a1d…`. Each calibration run draws 16 of 62 training prompts per round (≈ 8 presentations per item over 32 rounds). Calibration run 11072995 (lr 1e-6, symmetric clip) moved preempt → general with `ae slurm update` (same job ID) because a general slot was free and preempt jobs pend on Priority. | `project.md` §7 (≥ 16 items per band per split met); §3 move rule. |
| D34 | Held-out target and probe panels frozen by rubric only (job 11073178; `data/panels/panels.json`, file hashes `panels.sha256`): target_dev 53 (desired 64; minimum 32 met — study_dev had 63 rubric candidates, 10 excluded by the audit), target_test 128, probe J_H 32 (first 32 of target_dev), J_R 28 (study_dev direct-practice factoring), J_U 32 (study_dev direct modular). No model outcome was used. Probe IDs are frozen before any probe evaluation (A1.2). | `project.md` §5 (targets by rubric, not success), A1.2. |
| D35 | Authored intake (job 11073226): 400/400 gold answers verifier-eligible; 6 excluded as near-duplicates of MATH items (MinHash/Jaccard ≥ 0.5, stricter than the 0.7 split rule; one resembled test item `number_theory/test/00176`); kept 198 related (82 template families) and 196 unrelated (45 families); max Jaccard to any MATH item 0.6. Authored screening (`qis-screen-authored`) waits for a free general slot. | D32 dedupe and held-out independence (A2: evaluation sets free of training-template variants). |
| D36 | **External tranche (A4.3).** Native H pool cannot reach 48 (full native stage 2, 5,888 attempts: core 22, moderate 12 before audit filters; job 11073157). Source: Omni-MATH `KbsdJames/Omni-MATH` @ `40ba231d` (test.jsonl sha256 `7c87be8e…`, 4,428 items). Prefilter (model-free): domain contains Algebra; solution mentions factoring; no proof/"show that"/functional-equation tasks; difficulty ≤ 5.5 (harder olympiad items would score 0/32); short answer (≤ 40 chars, not yes/no/true/false); no figure. 273 pass; the tranche is the first **200** in a fixed hash order (`sha256("qis-omni-1701/"+id)`) → `data/external/omni_tranche_v1.jsonl`. Pipeline: blind pass 1 + pass 2 + third read (same rubric; H role = central, direct_practice=no), v1.3 tags, intake (verifier gold check, prompt cap, MinHash vs all MATH splits at J ≥ 0.5, contest-identity check), then **32 attempts directly** (no 16-sample stage: the A4 bands are defined on 32 fresh attempts and stage-1 shortlisting undercounted natives), same model/prompt/decoding/cap/verifier; same selection rule and truncation exclusion. Provenance recorded per item; held-out MATH panels untouched. | `project.md` §18 (A4.3). Decided before any Omni-MATH item was screened. |
| D37 | Treat preempt as unavailable until it starts jobs: run GPU work on the 2 general slots in priority order (1) Omni screening 11073332, (2) remaining calibration runs 11072991, 11072996–11072999; move each pending preempt job to general with `ae slurm update` (same job ID) when a general slot frees. Keep them queued on preempt meanwhile (they cost nothing pending and may start). | 2026-11-07 04:48Z: all six owned preempt jobs pending on Priority with no start estimate (`squeue --start` N/A) after ~1 h; other preempt jobs on this account pending since 2026-11-05. `project.md` §3: use general when preempt takes > 30 min. Planning impact: main matrix may run largely on 2 general GPUs (wall time ≈ total GPU-hours / 2). |
| D38 | Combined H selection (`select_hard_combined.py`) committed before any Omni-MATH screening count was viewed: native eligible list (29) ∪ Omni items passing the same audit (plus proof_like=no) and intake; identical bands, truncation exclusion and composition; pooled family round-robin in seeded order; source recorded. Omni audit: 194/200 pass agreement, 73 H-eligible by annotation; third read on 6 → none central. | A4.2–3; one frozen rule for both sources. |
| D39 | R/U matching + A5 coverage code (`src/qis/data/match_aux.py`) committed before authored screening counts were viewed: aux band 12–24/32; within-provenance one-to-one matching ranked by exact count > 4-count bin, same answer type, standardized prompt/response-length distance, ids; target 64 pairs; balance repair drops worst pairs until |SMD| ≤ 0.25 for success and response length; coverage per A5 (techniques ≥ 10% of H_train need R count ≥ ceil(0.5·share_H·|R|)); repair = re-match with deficient techniques offered first (≤ 3 attempts); residual deficits recorded. Reports mixed-group probability 1−p⁸−(1−p)⁸. | `project.md` §5, §16 (A2), §19 (A5). |
| D40 | **Second and final authored tranche** (revises the single-tranche stopping rule of D32; no treatment outcome exists). Measured gap after tranche 1 (job 11073304): eligible in-band (12–24/32) related = 16 (natural 7 pending pass 2, authored 9), unrelated = 48 (18 + 30); 116/198 authored related items scored 16/16 at stage 1 — direct practice is too easy for this model. A 16-item E_related is not a "modest shortfall" (A2); the fallback skill would be easier still. Budget: **300 related** (60 each F1, F2, F3, F4, F8; direct practice made computationally demanding: non-monic/large coefficients, degree ≥ 3, several factoring steps, answers needing all roots) + **120 unrelated** (authored partners for within-provenance matching). Authors receive difficulty calibration from tranche 1 only (example auxiliary items with their screening counts; no treatment information). Same intake, blind pass 1/pass 2/third read, screening (16 → 4–14 → 32), matching and coverage. Stopping rule: final — afterwards proceed with the pairs obtained under A2, documenting the shortfall. | A2 authorizes augmentation of either auxiliary pool to fill gaps found during screening, under a recorded gap and finite budget. |
| D41 | Authored tranche 2 received (420 items: R3 100 [44 families], R4 100 [33], R5 100 [31], U3 120 [27]); all answers independently verified by the authors' scripts. Provenance note: R4, R5 and U3 were assembled by parameterized scripts from hand-chosen designs (deviation from the "no template loop" brief); diversity is therefore judged by `template_family` counts (minor variants = one family, A2), not item counts. Same pipeline: intake (dedupe vs MATH and tranche 1), blind pass 1/pass 2 with opaque ids, third read, screening, matching. | D40 budget; A2 family-counting rule. |
| D42 | Tranche-2 intake (job 11073694): 420/420 gold answers verifier-eligible; 6 excluded as near-duplicates of tranche-1 items (J 0.55–0.83); kept 296 related (108 families) and 118 unrelated (27 families). Screening job 11073719 (preempt; moves to general ahead of calibration when a slot frees). | D40 pipeline. |
| D43 | **H_train selected: 49 items** by the frozen combined rule (D38; job 11073966, `data/screening/hsel_combined_v1.json`): 31 core + 18 moderate (core 63% ≥ ½); native 29 (28 L4–5, 1 L3) + Omni-MATH 20; primary operations F1 10, F2 9, F3 12, F4 9, F8 9. 49 lies in A4's 48–64 "proceed" range, so no smaller-pilot or authored-H path is needed. Expected presentations per item at T rounds: H-only 16T/49, mixtures 8T/49 (T=128: 41.8 and 20.9). Next: independent stage-3 baseline (16 fresh samples, stream s3) and measured mixed-group fraction; reported, not used to reselect. | A4.2, A4.4. |
| D44 | Calibration run lr 1e-6 / clip (0.2, 0.2) / seed 11 (job 11072995) completed 32 rounds: calib_val pass@1 0.4265 → 0.4706 (r16) → 0.5074 (r32); validation truncation 2.2% → 6.2% → **11.5%**, above the §7 rejection threshold (5% capped validation responses) → this configuration is invalid under the rule. Training: mean length 1,312–1,964 tokens, up to 13/128 truncated per round; logprob mismatch p99 ≤ 0.113; clip fractions ≤ 2e-4; median round 140 s. No decision on the rule's consequence until all six configurations finish; remaining runs ordered lr 5e-7 next (lower drift expected). If fewer than two configurations are valid, apply §7 (diagnose the common failure, fix, rerun affected configurations once). | `project.md` §7 rejection rule. |
| D45 | **Response cap raised from 4,096 to 8,192 tokens** (supersedes D18), before protocol freeze and before any treatment run. Evidence: calibration lr 1e-6/sym (11072995) raised calib_val truncation 2.2% → 6.2% → 11.5% over 32 rounds while pass@1 rose 0.427 → 0.507; diagnosis (11074473, `research/evidence/calibration/diag_trunc_lr1e-6-sym.json`): truncated responses are genuine long reasoning, not loops (mean 20-token-window loop score 0.24 → 0.03; 1/94 loop-like at r32), concentrated on hard items, and non-truncated length also grows (1,390 → 1,622). At 128 rounds a 4,096 cap would make truncation a growing, possibly arm-dependent confound. §6 names 8,192 as the remedy when > 5% of attempts are capped; the length probe showed 8,192 removes baseline truncation with identical success (0.6406) and fits (vLLM max_model_len 9,216). Consequences, applied consistently: (1) all 4,096 screening estimates are superseded and kept; membership is re-screened at 8,192 with fresh streams; items whose 4,096 success already exceeds a band's upper limit are not re-screened (success at a larger cap is ≥ success at the smaller cap path-wise: the first 4,096 tokens are identical, and capped responses score 0); (2) calibration restarts at 8,192 (all six configurations, seed 11); the 4,096 run 11072995 is kept as superseded evidence; cancelled 4,096 calibration jobs 11072991 (4 min 53 s GPU), 11072996–11072999 (never started); (3) held-out target/probe panels are rubric-selected and unaffected; (4) cost: rounds lengthen (estimate 1.3–1.5×); main runs remain < 24 GPU-h. | `project.md` §6 (cap remedy; "if the cap changes, supersede affected screening estimates and repeat screening under the final cap before freezing membership"), §7 rejection rule. |
| D46 | Calibration panels at 8,192 frozen (job 11074507; supersedes D33): calib_train 59 (low 22 / high 37), calib_val 56 (low 21 / high 35); bundle sha256 `a90a0d71…`. 8k screening truncation 0.13–0.28% of attempts. Canary calibration run 11074520 (lr 1e-6, sym) moved to general. | §7 minimum 16 per band per split met at the final cap. |
| D47 | **H_train at the final cap: 46 items** (job 11074898, `data/screening/hsel_k8.json`; supersedes D43): 31 core + 15 moderate (core 67%); native 25 (24 L4–5, 1 L3) + Omni-MATH 21; primary operations F1 11, F2 7, F3 11, F4 10, F8 7; excluded 60 outside bands, 1 verifier unknown. A4.4 record (46 < 48): 5 distinct operation families, each ≥ 7 items, 2 sources; core share 67%; presentations per item at T rounds = 16T/46 (H-only), 8T/46 (mixtures) → T=128: 44.5 / 22.3; T=64: 22.3 / 11.1. Judgment: diversity adequate and repetition comparable to the 42.7 implied at the 48-item threshold the owner accepted, so no extra uniform shortening; the endpoint is set by the §7 cost rule. No authored H tranche (no missing structure). | A4.4 decision rule. |
| D48 | **Auxiliary pools at the final cap: 63 matched R/U pairs** (job 11075024, `data/screening/match_k8.json`): eligible in band R 77 (natural 13, authored 64), U 82 (17, 65); within-provenance matching → 9 natural + 54 authored pairs (identical synthetic share 85.7% in both arms); 53/63 exact success-count matches; |SMD| success 0.025, response length 0.19 (≤ 0.25); 1 pair dropped in balance repair; expected mixed-group probability 0.963 (R), 0.964 (U). Diversity: R 9 natural + 54 authored in 40 template families (max 3 per family); U 9 + 54 in 35 families (max 5). **A5 coverage met with no repair** (techniques ≥ 10% of H: F1 23.9%, F2 15.2%, F3 23.9%, F4 21.7%, F8 15.2%; required R counts 8/5/8/7/5; achieved 17/6/11/19/10). 63 vs target 64 is a modest shortfall under A2 (coverage, diversity and balance documented). No natural item lacking pass 2 reached the band at 8k. | A2, A5, `project.md` §5 matching. |
| D49 | **8,192 canary passed** (cal8k lr 1e-6/sym, job 11074520, 7 rounds at check): device memory 35.8–40.0 GiB of 44.4; rounds 146–208 s (one 471 s round during a background checkpoint write); training truncation 0–1/128; logprob mismatch p99 0.105–0.112 (within D25). Remaining cal8k runs released as general slots free. **Measured evaluation noise:** two runs evaluating the identical base model on calib_val with identical eval seeds at round 0 gave pass@1 0.4386 vs 0.4487 (Δ 0.010; 56 problems × 16): vLLM seeded sampling is not reproducible across jobs/nodes (batch-dependent numerics). Consequences: (1) calibration gains are computed within each run (own baseline), as specified; (2) the 0.01 tie rule is at the noise floor, so ties will be common — the prespecified tie-break order then decides; (3) for the main study, baseline evaluations are recorded per run and the shared baseline is averaged across runs where reused; "paired seeds" give common seeds, not identical samples — reported as a limitation. | `project.md` §7 tie rule, §9 pairing. |
| D50 | **Evaluation noise revised (supersedes the 0.010 figure in D49).** Three round-0 evaluations of the identical base model on calib_val with identical eval seeds (jobs 11074520, 11074527, 11074526) gave pass@1 0.4386, 0.4487, **0.4866**. Only 8–13 of 56 problems got identical counts between jobs, so cross-job evaluations behave as independent samples, not paired ones. Expected sampling sd (pooled per-problem p, 16 samples): 0.014 per evaluation, **0.020 for a within-run gain** (two evaluations). The 0.048 gap is 2.4 sd of a difference, consistent with chance among three pairs. Consequences: (1) the §7 rule stays as frozen (within-run gain; tie threshold 0.01; tie-break order) — no rule change after seeing results; (2) gains below ~0.04 are not distinguishable from zero at one seed, which the seed-22 repeat of the top two partly addresses; (3) as a disclosed robustness check, the calibration table also reports each run's round-32 pass@1 against the pooled round-0 mean (not used for selection); (4) main study: R, U, H endpoints are compared on the same sealed test with 32 samples × 128 problems (sd ≈ 0.007 per evaluation) and seed-level variation enters the t interval and bootstrap, so the design already covers this noise. | `project.md` §7 (rule frozen), §9 (uncertainty from seeds and problems). |
| D51 | **Correction to D50 and early seed-22 queueing.** (1) Round-0 evaluations of 11074519 (L40S, babel-q9-24) and 11074526 (L40S, babel-o9-32) agree on 56/56 per-problem counts, while 11074527 (L40S, babel-q9-32) agrees with them on only 8/56. Cross-job reproduction is therefore unreliable, not absent: some jobs reproduce exactly, others diverge. The D50 noise bound (sd 0.020 per gain when samples diverge) stands as the planning figure; the selection rule is unchanged. (2) Seed-11 standings with 4 of 6 configs done: 1e-6/sym +0.122, 1e-6/asym +0.066, 5e-7/sym +0.031, 5e-7/asym −0.003. 1e-6/sym leaves the top 2 only if both 2e-6 runs exceed +0.122. Its seed-22 repeat (`qis-cal8k-lr1e-6-sym-s22`) is queued now on preempt (idle budget, no general slot taken) to shorten the critical path; it is cancelled and charged if 1e-6/sym falls out of the top 2. Selection still applies the §7 rule after all six seed-11 runs. | `project.md` §7; owner authority to submit/cancel owned jobs. |
| D52 | **Stage-3 independent baseline at 8,192** (job 11075046, first preempt start of the study; 16 fresh samples per item, stream s3; `research/evidence/stage3/`). H_train 46: mean pass@1 0.202, 4 items 0/16, 0 items 16/16, measured mixed-group fraction 71/92 = 0.77 (groups of 8). E_related 63: 0.589, mixed 0.944. E_unrelated 63: 0.569, mixed 0.937 (expected from stage 2: 0.963/0.964, D48). R − U baseline success gap 0.020 (matched within noise). Reported only (A4.2); no reselection. The 4 H items at 0/16 are consistent with regression from 2–3/32 at stage 2 and stay in H_train under the frozen rule. | A4.2 (independent baseline after selection, measured mixed-group fraction). |
| D53 | **Calibration lr 2e-6/sym (job 11074522): gain +0.146** (0.4866 → 0.5603 → 0.6328); validation truncation 0.56% → 2.46% → **4.69%**, below the §7 rejection threshold (> 5%), so the configuration is valid under the frozen rule. Training truncation up to 9/128 per round; mean length 1,437 → 1,627; logprob mismatch p99 ≤ 0.124. It is a certain finalist (only 2e-6/asym is outstanding, and at most one config can rank above it), so its seed-22 repeat (`qis-cal8k-lr2e-6-sym-s22`) is queued now. The rule is applied as written; no truncation-trend criterion is added after seeing results. Recorded risk: if 2e-6 is selected, truncation in 128-round main runs may exceed 5%; main-run truncation per arm is monitored each round and reported, and §6 (cap) would apply only through a recorded deviation, never silently. | `project.md` §7 rejection and ranking rule (frozen). |
| D54 | **First-stage calibration complete; finalists fixed.** lr 2e-6/asym (job 11074521): pass@1 0.4866 → 0.5960 → 0.6830 (gain +0.196) but validation truncation **7.14%** at round 32 (> 5%) → **rejected** under §7, though it had the largest gain. Valid ranking: lr2e-6/sym +0.146, lr1e-6/sym +0.122, lr1e-6/asym +0.066, lr5e-7/sym +0.031, lr5e-7/asym −0.003. Finalists: **lr2e-6/sym and lr1e-6/sym**; their seed-22 repeats (11076551, 11075978) were already queued (D51, D53), so nothing is cancelled. Selection: mean gain over seeds 11 and 22; ties ≤ 0.01 → lr 1e-6; gate: positive mean gain, neither seed below its baseline by > 0.02; a seed-22 run with > 5% capped validation responses makes that finalist invalid. Observation: higher lr lengthens responses fast (2e-6/asym mean length up to ~2,500 tokens); the asymmetric clip amplifies it. | `project.md` §7. |
| D55 | **Calibration selection is computed by script in a CPU job.** `src/qis/analysis/calib_select.py` encodes the frozen §7 rule (rejection criteria at every evaluation, seed-11 ranking, seed-22 validity, mean gain, 0.01 tie with the prespecified order, positive-gain and −0.02 per-seed gate); recipe `qis-calib-select` runs it on the committed metrics and records input checksums. Deviation recorded: one read-only preview run of this script on the laptop at 18:08Z (small committed JSON; output matched the hand table: ranking 2e-6/sym, 1e-6/sym, 1e-6/asym, 5e-7/sym, 5e-7/asym; 2e-6/asym rejected). This, and earlier laptop one-line summaries of harvested metrics, break the "no project code on the laptop" rule; from now on every reported statistic is computed in a CPU job. | `project.md` §3 (CPU allocations for statistics); agent.md. |
| D56 | **A6 thinking-mode diagnostic design** (owner amendment A6, project.md §20; non-thinking study continues unchanged). Code `src/qis/mode_diag.py`, tests `tests/test_mode_diag.py`, committed before any thinking-mode output. Panel (seeded, PANEL_SEED 2611, fixed before results): 32 H_train by proportional allocation over source × primary-technique cells (largest remainder, seeded tie-break), 16 seeded related/unrelated matched pairs (32 items), 16 seeded target_dev items; 80 distinct; sealed test untouched. 8 attempts per item per mode, request seeds shared across modes (stream 2612). Only `enable_thinking` changes; template check asserts the two renders differ only by the empty think stub and that `</think>` is a single token. Thinking grading: reasoning + answer tokens count against the cap; only text after the first `</think>` is graded; no `</think>` → capped: truncation treatment (0), uncapped: 0 (`no_final`); tests prove a boxed answer inside reasoning earns nothing. **Non-thinking attempts are generated fresh in the same job, not reused** from stage 3: stage 3 used 16 attempts from another stream, and same-node timing is needed for the GPU-cost comparison; cross-job sampling does not reproduce (D50, D51). Failure sample fixed by rule: H items in panel order, first failed attempt per item, ≤ 16 per mode. Same-cap (8,192) comparison first; a 16,384 profile only if thinking truncation > 5% on H or either auxiliary pool, reported separately. Distinct run IDs (`qis-a6-*`) and output directory (`$PROJ_DATA/a6/`). | project.md §20 (A6). |
| D57 | **Calibration selection and protocol freeze.** CPU job 11078053 (`qis-calib-select-b`; tests 4/4; `research/evidence/calibration/selection.json`, input checksums alongside) applied the frozen §7 rule: finalists lr2e-6/sym (seed 11 +0.146, seed 22 +0.172, mean **+0.159**; seed-22 truncation ≤ 2.1%) and lr1e-6/sym (+0.122, +0.090, mean +0.106); difference 0.053 > 0.01, no tie; both pass the gain gate. **Selected: lr 2e-6, clip (0.2, 0.2).** Sensitivity arm uses (0.2, 0.28) at lr 2e-6 by §7 ("the other clipping setting"), although that pair exceeded 5% validation truncation in calibration (D54): it runs as authorized and its truncation is reported per arm. **Endpoint T = 128** (cost estimate ≤ 21 GPU-h per run for every projection; `research/evidence/cost/main_cost_estimate.md`). Earlier failed selection attempt 11078035 (parsing bug, 1 s CPU) kept in the ledger. | project.md §7 selection and freeze. |
| D58 | **Freeze commit `b8f36ca`; launch order.** Main recipes generated from the frozen protocol (`qis.make_main_exps --lr 2e-6 --clip-selected sym --rounds 128`; a recipe-writing helper, run as orchestration). Canary (agent.md run-family rule): `qis-main-R-s101-sym` (job 11078086, general; it also retains round 64 for the local-update diagnostic). The other 14 runs are released after the canary's first real training round passes checks. A6 same-cap diagnostic `qis-a6-modes-k8` (job 11078087) on the second general GPU. GPU use: 2/2 general. | agent.md canary rule; project.md §7, §20. |
| D59 | **A6 job 11078087 stopped at its pre-run grading tests** (correct gate behaviour; no generation ran, ~3 GPU-min charged). Cause: a verifier worker imports math_verify on its first job inside the 5 s limit; on a cold node the import exceeded 5 s and the 15 s retry (fresh, equally cold process) → status unknown (reward None) where the test expected 0. Not a grading-rule defect: unknown is never scored as wrong, and main/calibration runs show 0 unknowns (vLLM already loads sympy). The frozen `verify.py` is not changed during the main study. Fix in the diagnostic only: `mode_diag.warm()` scores a trivial item on every worker until all succeed, before tests and before any timed scoring. Rerun as `qis-a6-modes-k8-b`. Recorded as a known limitation for the main-study verifier: a cold first verification can show as unknown and hold a round (safe, possibly slow). | project.md §6 (unknown handling), §20. |
| D60 | **A7 applied: 3 general + 6 preempt GPUs** (`ops/qis-submit` SLOTS_general 2 → 3; `normal` QOS allows 8 GPUs per user, shared with the owner's other projects). Reconciliation at 19:20Z: Slurm ↔ ledger agree — canary 11078086 (general, running), A6 11078115 (general, running); nothing pending. Highest-priority ready work = the 14 frozen main runs, gated by the canary rule (first real training round not yet written; canary still in its round-0 evaluation). Third general slot: `qis-main-U-s101-sym` (job 11078164; the R−U partner of the canary seed) submitted, moved to general and **held** until the canary passes, so it holds queue position and cannot start early. Single execution verified: one ledger row, one Slurm job. Code check: commits b8f36ca (canary) and efdafdd (this run) differ only in `src/qis/mode_diag.py`, its test and research docs; training/verifier/prompt/data code identical. | project.md §21 (A7); agent.md canary rule. |
| D61 | **Canary passed; main matrix released.** `qis-main-R-s101-sym` (11078086) round 0: dev 0.422, J_R 0.817, J_U 0.816 (truncation 0, unknown 0); round 1: hard 0.203 (stage 3: 0.202), aux 0.625, mixed groups 12/16, truncation 1/128, mean length 1,743 (estimate 1,706), 175 s/round, train memory 38.6 GiB, mismatch p99 0.103, clip fractions ≤ 2.3e-4. Released: U-s101 (11078164, general, running). Queued on preempt (slot cap 6): H-s101 11078180, R-s202 11078181, U-s202 11078184, H-s202 11078185, R-s303 11078186, U-s303 11078187. Waiting for slots (not yet submitted): H-s303 and the six asymmetric sensitivity runs. Rule: whenever a general slot frees, move the oldest pending primary run to general. Training code identical across commits b8f36ca, efdafdd, 0a83a74. | agent.md canary rule; project.md §7, §21. |
| D62 | **A6 same-cap result (8,192; job 11078115; `research/evidence/a6/`).** Grading tests 3/3; template check ok (`</think>` = id 151668; renders differ only by the think stub). Panel: 32 H over 10 source × technique cells, 16 R/U pairs, 16 dev. Per pool, non-thinking → thinking (8 attempts per item; computed in the job): **H** pass@1 0.188 → 0.418 (paired Δ +0.230, 95% bootstrap CI [0.125, 0.336]), mixed groups 78% → 69%, any success 78% → 75%, **truncation 0.4% → 51.6%**, mean length 2,224 → 7,196 tokens (6,669 reasoning); **dev** 0.258 → 0.508 (Δ +0.250 [0.102, 0.422]), truncation 0% → 46.1%; **related** 0.727 → 0.969 (Δ +0.242), mixed groups 88% → 25%, truncation 1.6%; **unrelated** 0.734 → 0.984 (Δ +0.250), mixed groups 100% → 13%, truncation 1.6%. R − U over matched pairs: −0.008 [−0.094, 0.086] (off), −0.016 [−0.047, 0.016] (on): still difficulty-matched. GPU generation time thinking/non-thinking: H 6.9×, related 8.1×, unrelated 5.5×, dev 6.7× (tokens 3.2–4.1×). Unknown verifier results 0; no `no_final` responses. Interpretation (aggregate screen, not item reclassification): thinking mode makes the auxiliary practice near-trivial at this cap (most groups all-correct → little GRPO signal) and solves far more hard items despite truncating half of them. **A6.4 trigger met** (thinking truncation on H 51.6% > 5%) → 16,384 profile, thinking only, same panel (`qis-a6-think-k16`), reported separately. Slot: H-s303 (11078412, never started, 0 compute) cancelled to stay within 6 preempt GPUs; it is resubmitted when a preempt slot frees. Failure-sample inspection pending. | project.md §20 (A6.3, A6.4). |
| D63 | **A6 failure-sample inspection** (32 fixed records, agent-annotated, `research/evidence/a6/failure_inspection.{jsonl,md}`; counts recomputed with `jq`; 2 records spot-checked against the text). Focal factoring operation implicated — non-thinking: yes 11, no 2, unclear 3 (dominant locus `factoring_not_attempted` 9/16: value testing, coefficient systems, guessing; wrong factorization 2/16). Thinking: yes 3, no 11, unclear 2 (dominant locus `truncated_after_focal_step` 10/16: factoring done, then verification or looping until the 8,192 cap; several had reached the gold answer). One thinking "wrong" record gave both correct values in two boxes and the frozen last-box rule extracted only one (applies identically to both modes). Interpretation: in non-thinking mode the focal operation remains the main failure point, which supports the main study's premise; in thinking mode at 8,192 the binding constraint is the length budget, not the factoring step. Limitations: 16 per mode, tails only (problem texts were not in the files given to the annotator; judged from model text, gold and tag quote), agent labels not human-reviewed. | project.md §20 (A6.3). |
| D64 | **A6 closed (owner amendment A8, project.md §22).** D62 and D63 kept as completed evidence. Job 11078421 (`qis-a6-think-k16`, owned: one ledger row) was pending and never started → cancelled at 20:17Z; elapsed 0 s, cost 0. Its slot went to `qis-main-H-s303-sym`: status checked first (only ledger row 11078412, cancelled, never ran; no live job) → resubmitted as job 11078506 (preempt); one live H-s303 job verified. Training code identical to the freeze commit (b8f36ca vs 4927344). Ceilings unchanged: 3 general, 6 preempt. | project.md §22.1. |
| D65 | **D63 interpretation narrowed (A8.2).** The D63 counts are observations from 16 sampled failures per mode, read from response tails only: in 9 non-thinking cases the visible text showed no factoring attempt and in 2 it showed an incorrect factorization; 10 thinking cases were classified as truncated after factoring. A tail cannot show what happened earlier, and an alternative method can be mathematically valid, so these counts do not show that factoring "is the failure point" in either mode. Hypothesis under test (unchanged): whether direct practice helps the learner recognize and use factoring in harder problems. The D63 sentence "supports the main study's premise" is withdrawn. | project.md §22.2. |
| D66 | **Truncation reporting during training (A8.4).** The cap (8,192) and completion rule are unchanged for all main and sensitivity runs. A CPU-side monitor (`src/qis/analysis/trunc_monitor.py`) reads the per-response training records (`runs/<run>/records/train_r*.jsonl.gz`: role, finish) and reports per run and round: capped / total for hard and auxiliary prompts, and R vs U on their shared hard schedule (same rounds, same `hard_common` prompts); scheduled evaluations report truncation with pass@1. A > 5% crossing flags inspection and reporting only. No frozen trainer code changes. | project.md §22.4. |
| D67 | **Records reconciliation, storage plan and forecast (A8.5; CPU jobs 11078552, 11078608; `research/evidence/a8/`).** (a) **Native H candidates (§3.12/§3.16):** 184 native items had 32 attempts at 4,096; 117 were excluded by their 4,096 score (c > 12/32; success at a larger cap is ≥ success at the smaller cap path-wise, D45); 67 had c ≤ 12, of which **55 received 32 fresh attempts at 8,192** and 12 were not re-screened (annotation-ineligible); 0 items with c > 12 were re-screened; 25 natives entered H_train; plus 52 Omni-MATH items re-screened at 8,192. Frozen cohorts unchanged. (b) **Calibration panels 59/56:** of 200 candidates screened per split at 8,192 (16 samples), exactly 22+37 = 59 (train) and 21+35 = 56 (val) fell in the low (1–5/16) and high (6–12/16) bands; the panels contain every in-band item. §7 (≥ 16 per band per split) is met. §3 asks for "at least 64 eligible problems" per calibration split: the splits hold 750/727 eligible problems, but if "eligible" is read as panel-eligible (in band), the panels are 5 and 8 items short. This was not flagged before calibration; calibration is complete and frozen and is not rerun (A8: keep frozen cohorts intact). Consequence: a smaller validation panel adds noise to calibration gains (documented in D50); reported as a deviation under the strict reading. (c) **Storage (peak check before filling slots):** `/data/user_data` 359–374 GB of 500 used (shared with the owner's other projects, ~110 GB); project checkpoints 240 GB, of which ~182 GB were calibration restart checkpoints (8 cal8k + 2 superseded cal-4k dirs). Retention rule (project.md line 240: calibration checkpoints kept through the extension decision, made in D57) → pruned by job 11078641 (exact named dirs, sizes/file lists logged). Active main runs keep two 20 GB restart checkpoints (40–60 GB transiently) → **concurrent main runs capped at 6** (3 general + ≤ 3 preempt) until completed runs free space; seed-303 jobs 11078186, 11078187, 11078506 held. (d) **Forecast from production rounds** (R-s101 ~255 s/round incl. checkpoints over 16 rounds; U-s101 ~245 s over 11): R/U ≈ 9.6 h at constant length, ~11 h with growth; H ≈ 13 h; 15 runs ≈ 170 GPU-h; wall ≈ 57 h on 3 general GPUs alone (all done ~10 Nov 05:00Z), ~30–35 h if 3 preempt GPUs run in parallel. (e) Truncation monitor (D66) at 20:21Z: R-s101 hard 7/1,024, aux 0/2,048; U-s101 0/704, 0/1,408; shared hard schedule R 3/704 vs U 0/704 (11 rounds); no flags; round-0 evaluations identical across the three runs (dev 0.4222). | project.md §3, §7, line 240, §22.5. |
| D68 | **Calibration checkpoints pruned** (job 11078641; log `research/evidence/a8/prune_cal_deleted.txt` with sizes and file lists): 10 project-owned directories, 204,209 MiB in total (8 × `qis-cal8k-*` ≈ 20.4 GB each; `qis-cal-lr1e-6-sym-s11` 20.4 GB, superseded 4,096 run; `qis-cal-lr1e-6-asym-s11` 1 MiB). No main-run, probe or diagnostic checkpoint touched. `df` immediately afterwards still read 359 GB used (NFS may release space late) → re-check before raising the 6-run cap (D67). Storage-scan job 11078608 ended FAILED in its per-subdirectory loop after printing the per-run sizes used in D67 (read-only; no effect). | project.md line 240 (retention), line 268 (safe cleanup authorized). |
| D69 | **§3.7 baseline error audit complete** (A8.3; 64 incorrect non-thinking 8,192-cap screening attempts, one per H candidate, seeded; full problem + response; agent reviewer; `research/evidence/a8/audit37_{labels.jsonl,report.md}`; counts recomputed with `jq`). First identifiable (uncorrected) error: missed factoring opportunity **30**; other error after correct factoring **20**; incorrect execution of factoring **6**; valid alternative approach then a different error **6**; final-answer extraction failure **2**; insufficient evidence **0**. 8 labels are borderline (5 setup errors before factoring, labelled missed opportunity; 3 correct factoring of an already wrong expression, labelled other error after factoring) → counts can shift by up to 8. **Verified extraction misses (reported separately, never regraded):** 2/64 — `intermediate_algebra/train/00062` (`x \geq 8` vs gold `[8,\infty)`; equivalent under the problem's domain x > 4, not matched by the verifier) and `…/00074` (`(f(0), f(1)) = (-1, 1)`: correct pair behind a label). **Split-box complete answers: 0** (the reviewer's `split_box_complete=true` on these two records is a misapplied flag: neither answer is split). Reading (observational, one attempt per problem): among sampled failures the most common first error is not using the tagged factoring step (30/64), and a further 6/64 show it executed wrongly; 20/64 fail after correct factoring and 6/64 use a valid alternative. Limitations: one attempt per problem, single agent annotator, no human review, no reference solutions. Frozen grading unchanged. | project.md §3 audit rule, §22.3. |
| D70 | **Storage re-check and run cap raised to 7** (monitor job 11078827, 21:06Z): `/data/user_data` 195 GB used / 306 GB free (the 204 GB pruned in D68 is now released); project checkpoints 60 GB. Project headroom ≈ 366 GB; at ~40 GB per active run (two restart checkpoints; +20 GB transiently while writing) plus the retained R-s101 round-64 checkpoint and a 50 GB margin → cap **7 concurrent main runs** (3 general + ≤ 4 preempt). Released R-s303 (11078186); U-s303 and H-s303 stay held. Production rounds have slowed as responses lengthen (mean 2,500–2,600 tokens by rounds 13–26 vs 1,743 at round 1; 259–306 s per round) → forecast refreshed: R/U ≈ 11–12 h, H ≈ 13–14 h per run. Truncation (D66): R-s101 hard 28/1,664, aux 10/3,328; U-s101 hard 31/1,344, aux 4/2,688; H-s101 3/1,664; shared hard schedule (21 rounds) R 14/1,344 (1.0%) vs U 31/1,344 (2.3%); no flags. | project.md line 240; §22.4–5. |
| D71 | **A9 executed (deadline scope; project.md §23).** Amendment received 2026-11-07 ~22:00Z; **endpoint results viewed: none** (no endpoint reached; target_test unopened). Reconciliation 22:02Z: Slurm ↔ ledger agree. Cancelled verified project-owned deferred jobs, none ever started (cost 0): R-s303 11078186 (pending), U-s303 11078187 (held), H-s303 11078506 (held); earlier H-s303 attempt 11078412 already cancelled (D64). Sensitivity runs were never submitted. Recipes for all 9 deferred runs moved to `slurm-agentic-engineering/exp-deferred/`; `ops/qis-submit` refuses `*-s303-*` and `qis-sens-*` (exit 5; verified). In scope: R/U/H-s101 running (general, 11078086/11078164/11078180); R/U/H-s202 queued (preempt, 11078181/11078184/11078185). Slot plan: each s101 completion → move the matching s202 job to general immediately (critical path); base-model test evaluation, s101 endpoint evaluations and the local-update diagnostic queue on preempt and move to general when an s202 run ends, if still pending; no new experiments in freed slots. Analysis plan replaced for the two-seed scope (PROTOCOL A9 section). Forecast: s101 ends ~8 Nov 07:00–10:00Z; s202 ends ~8 Nov 19:00–23:00Z; evaluations + diagnostic ~9 Nov 02:00Z; report target 10 Nov 01:00Z (margin ≈ 20 h before the target, ≈ 33 h before the hard deadline). | project.md §23. |
| D72 | **A9 tooling ready (before any endpoint).** Local-update diagnostic `src/qis/diag_local.py` (one GRPO step per branch from an identical model+optimizer snapshot; related/unrelated share matched pair indices; eval seeds shared across branches and sources) with CPU tests; full suite 41/41 (job 11079320). Sealed test recipes from `qis.make_test_exps` (base + six endpoints; precheck: run complete, round_0128 model.pt sha256 = state.json, sealed-file sha256). Endpoint validator `src/qis/analysis/validate_endpoint.py`. A9 table code `src/qis/analysis/report_tables.py` (per-seed rows, paired R−U/R−H, conditional problem bootstrap over 128 problems, seed agreement, two-seed t interval only with a warning). Queued on preempt: base test eval 11079315, diagnostic r0 11079411 (spare capacity only; no new experiments). Note: the checkpoint writer keeps the latest round plus retained rounds (not two full checkpoints): ~20 GB per active run, ~40 GB while writing. | project.md §23.2–4. |
| D73 | **A10 boundary = round 64, recorded before any further comparative result was inspected** (entry committed 2026-11-07 22:26Z). Amendment received ~22:22Z. **Motivating truncation evidence** (monitor job 11079420, 22:20Z): dev truncation at round 32 = 11.6% (R-s101) and 16.7% (U-s101) vs 0% at round 0; cumulative capped training responses on hard prompts R-s101 83/2,304 (3.6%), U-s101 105/2,048 (5.1%), H-s101 34/3,200 (1.1%); shared hard schedule (32 rounds) R 2.7% vs U 5.1%; mean response length 2,529–3,089 tokens (1,743 at round 1); round time 294–380 s. **Outcomes viewed at amendment time:** no endpoint; no target_test outcome. Viewed interim dev/probe results at round 32: R-s101 dev 0.507, J_R 0.857, J_U 0.859; U-s101 dev 0.547, J_R 0.871 (U J_U r32 not yet logged); round-0 values (identical across runs); per-round training success; truncation counts. The `qis-trunc-inspect-r32` output (job 11079455) had arrived but was **not read** before this entry. **Progress at choice:** R-s101 round 36, U-s101 round 32, H-s101 round 25 (22:20Z); seed-202 runs not started. **Why 64:** the code change plus long-response smoke test need ~2–3 h, and R-s101 reaches 64 in ~3 h, so an earlier boundary would idle GPUs without saving work; 64 is a scheduled checkpoint (every 4) and the protocol's halfway round, and R-s101 already retains round_0064 (local-update source, now a pure 8k-trained state). Rounds 1–64 use cap 8,192; rounds 65–128 use the amended cap for every run. **Transition procedure:** each run continues at 8,192 to round 64 (scheduled dev evaluation at 64 runs at 8,192 inside the old job, before checkpoint 64); after its round_0064 checkpoint is validated, the old job is stopped during round 65 (signal → exit without saving; the next-round partial work is discarded, no completed round is lost), its requeue is cancelled, and a replacement job at the new commit resumes from round_0064 with model, FP32 AdamW state, RNG state and counters restored by `ckpt.load`; the schedule position is the round index, so the sampler continues exactly. Seed-202 runs train with the same schedule from round 0. | project.md §24. |
| D74 | **A10 implementation and release plan.** Code (commit 29fe565 + 60d733d): per-round `cap_schedule` in the trainer (rounds 0-based < 64 → 8,192; ≥ 64 → 16,384; scheduled evaluations use the cap of the last trained round, so the round-64 dev evaluation stays at 8,192 in every run; each round logs `max_tokens`); vLLM engine context = 1,024 + largest scheduled cap (17,408), per-call `max_tokens` asserted ≤ engine context; nothing else in the trainer or verifier is fixed at 8,192 (grep). Recipe generator: `--cap-schedule 0:8192,64:16384 --attempt a2 --seeds 101,202 --no-sensitivity`; config rewritten at every start. Tests `tests/test_capsched.py` (job 11079542). **Smoke test** `qis-smoke-k16` (11079543): one full round (16×8, 4 optimizer steps of 4 groups, micro-batch 1) at cap 16,384 with two prompts forced to ≥ 15,384 tokens; checks engine context ≥ 17,408, a response > 8,192, one teacher-forced log prob per response token in both passes, finite gradients, peak device memory (nvidia-smi every 5 s) and phase times. If it fails or throughput threatens the deadline, the largest feasible common cap in (8,192, 16,384) is chosen before any arm trains past round 64. If preempt has not started the smoke test when R-s101 reaches 64, it runs on R-s101's general GPU first. **Stopping old jobs at 64:** watchers `qis-boundary-watch-{R,U,H}-s101` (CPU) wait for a complete round_0064, hard-link it to `boundary_0064` (protection from rolling deletion), and exit; then the orchestrator cancels the old job (`ae slurm cancel`; trainer killed without saving; only partial round 65 lost), confirms the allocation ended, and submits the replacement (same run directory; resumes from round_0064). Seed-202 jobs queued under the old recipes behave identically below round 64 and are replaced with schedule-aware recipes after the smoke test passes. **Evaluations:** sealed test evaluations at cap 16,384 for the base model and every endpoint, also reporting success with completion within 8,192 tokens from the same traces (exact: identical first 8,192 tokens at equal seeds); old base job 11079315 (8,192, never started) cancelled and resubmitted as 11079548. The local-update diagnostic keeps an 8,192 evaluation cap for both sources (both are 8k-trained states; internal consistency); recorded as a limitation. **Truncation diagnosis:** pre-boundary evidence (job 11079455): truncated responses are long reasoning, not loops (dev r32 loop score mean 0.034, 0/71 above 0.5; 1 loop-like response in 8 training rounds); the A10 breakdown (share crossing 8,192, share then correct by 16,384, share exhausting 16,384, by arm and prompt type) comes from rounds 65+ training records and the 16,384 evaluations, plus a fixed inspected sample. | project.md §24. |
| D75 | **Owner instructions of 2026-11-07 ~23:15Z (continuation of A10).** (1) At round 64: keep the pre-switch checkpoint (`boundary_0064` hard link for every s101 run; R-s101 also retains round_0064); the smoke test runs and must pass before any update under the new cap. (2) Fixed dev panel at the amended cap from the pre-switch checkpoints: every s101 replacement first evaluates `boundary_0064` on target_dev (53 problems × 32 samples, cap 16,384, seed label `boundary-r64`), reporting pass@1 at 16,384, pass@1 with completion within 8,192 (same traces), the share crossing 8,192, and the share crossing 8,192 that then finish correctly (`over_8192_then_correct`). The sealed test is not used. (3) The same evaluation benchmarks endpoint-evaluation throughput (tokens/s, seconds per 1,696 completions) → revised forecast for the 28,672 final completions (7 evaluations × 128 × 32) plus grading and audit. (4) Learning curves (`src/qis/analysis/curves.py`, run hourly in `qis-monitor`): all-wrong / mixed / all-correct groups per round separately for hard and auxiliary prompts; R vs U at matched rounds on the shared hard schedule; every round kept; snapshots saved under `research/evidence/curves/`. First snapshot (23:20Z, interim, seed 101, 42 matched rounds, cap 8,192): shared hard schedule cumulative success R 0.398 vs U 0.451, capped R 129/2,688 vs U 223/2,688; auxiliary groups mostly all-correct from round ~24 (R 4–5/8, U 5–8/8 all-correct per round); hard groups mostly mixed; entropy 0.15–0.31 per block, no collapse. Interim single-seed curves; no decision rests on them. | A10 continuation (owner message). |
| D76 | **W&B view fixed (logging only; no training change).** Defects found in the live mirror: evaluation records were logged without the panel in the key (dev, J_R, J_U overwrote one `eval/pass1` series), nested dicts (per-role `success`, ratio and mismatch quantiles) were dropped, and J_H (32-problem dev subset) was never logged. Local `metrics.jsonl` is complete and remains the analysis source. Fix: (1) trainer `Log.write` keys evaluations as `eval/<panel>/<metric>`, flattens nested dicts one level, and logs `eval/probe_J_H/*` derived from each dev evaluation (effective in the A10 replacement jobs from round 65 and in the seed-202 runs); (2) companion runs `<run>-curves` and `qis-RvU-s<seed>-sym-curves` (job_type `derived`) backfill complete curves from local records each hour via `qis-monitor` (`src/qis/analysis/wandb_companion.py`, incremental by round; first push job 11080162 verified online). | Owner question; project.md §10 (W&B mirror, local JSONL authoritative). |
| D77 | **A11 pause executed** (amendment received ~00:10Z 2026-11-08). Slot guard refuses `qis-main-*` unless `QIS_A11_RESUME=1` (verified, exit 6). Reconciled Slurm ↔ ledger (all 10 affected jobs owned). Held: R/U/H-s202 11078181/11078184/11078185 (never started) and base test eval 11079548 (sealed test stays closed). Cancelled: boundary watchers 11079553/5/6 (obsolete); active main jobs at 00:11:55Z — R-s101 11078086 (5 h 05 min), U-s101 11078164 (4 h 42 min), H-s101 11078180 (4 h 07 min); total **13.9 GPU-h, all superseded beyond the replayed trajectory**. No main job advancing (squeue/sacct verified). **Last completed round / latest valid checkpoint** (scan job 11080461): R-s101 round 55 / round_0052; U-s101 51 / round_0048; H-s101 37 / round_0036; each holds only that checkpoint (rolling deletion), so **no end-of-round-32 state exists** → nearest verified full state = round 0 (base model, fresh AdamW) → all three s101 runs replay rounds 1–32 at 8,192 (A11.3). **Truncation evidence at amendment time:** dev at round 32 truncated R 11.6%, H 15.6%, U 16.7% (round 0: 0%); cumulative capped hard training responses R 5.1%, U 7.7%, H 2.5% (by ~round 40–45); shared hard schedule R 4.4% vs U 7.7% (40 rounds); truncated responses long reasoning, not loops (D74). **Interim outcomes viewed:** round-0 and round-32 dev/probe evaluations of all three s101 runs (dev p@1 R 0.507, U 0.547; J_H p@1 H 0.555, R 0.523, U 0.555; J_R R 0.857, U 0.871, H 0.862; J_U R 0.859, U 0.902, H 0.902), training curves to rounds 55/51/37 (curves snapshot 23:20Z), entropy, optimizer diagnostics; no endpoint; no test outcome. Truncation study started: dev at 16,384 (53×32) from the latest superseded checkpoints R r52 (11080483), U r48 (11080484), H r36 (queued), smoke test 11079543 moved to general. | project.md §25. |
| D78 | **Smoke test 11079543 failed: CUDA out of memory at 16,384** (backward of a forced ~15k-token response; 44.1 of 44.4 GiB in use; job then hung in GPU teardown after a cumem allocator error and was cancelled, ~55 GPU-min charged). Cause: the chunked FP32 LM head kept every chunk's full-vocabulary logits for backward (512 × 151,936 FP32 ≈ 311 MB per chunk; ~10 GB at 16k tokens, ~5 GB at 8k). Fix (`learner.lm_head_logprobs`): recompute each chunk's logits during backward (`torch.utils.checkpoint`, non-reentrant) — identical arithmetic, effective GRPO batch and micro-batch (1 response) unchanged; test `tests/test_lmhead.py` (FP32 outputs and gradients equal to 1e-6). Because all six runs restart under A11, every arm uses this code for its whole trajectory. Rerun: tests 11080736, smoke 11080758 (general). | project.md §25.2; A10 (reduce micro-batching/concurrency if needed, preserving the effective batch). |
| D79 | **Smoke test passed at 16,384** (job 11080758, RTX 6000 Ada 44.4 GiB; `slurm-agentic-engineering/results/qis-smoke-k16b/`): engine context 17,408; 16 responses > 8,192 (max 16,384, forced); teacher-forced log probs one per token in both passes; 4/4 finite steps (grad norm 0.46–0.61); device peak 38,754 MiB (nvidia-smi, 5 s) — 6.6 GiB headroom; worst-case round (two groups at 16k) = 523 s generation + 46 s old log probs + 159 s training + 1 s sync ≈ 12.2 min. **Memory is feasible at 16,384; throughput decides the cap** (frozen after the dev truncation study). Evaluator throughput fix: checkpoint weights now load on the CPU and are pushed into vLLM, which gets 0.85 of the GPU instead of 0.30 (the learner is no longer instantiated); validated by re-evaluating R-s101 round 52 on the same seeds (`qis-a11-dev16k-R-check`) once 11080483 ends. H r36 dev evaluation (11080946) uses the new evaluator. | project.md §25.2. |
| D80 | **Old-evaluator dev runs cancelled for the deadline** (02:07Z): 11080483 (R r52) and 11080484 (U r48) had run 1 h 48 min each at vLLM memory share 0.30 with no completion in sight (est. 3+ h); cancelled, **3.6 GPU-h charged**, no outputs used. Resubmitted with the CPU-load evaluator (vLLM share 0.85) as 11081333 / 11081334 (output dirs `*-v2`). Evaluator validation changed from a same-seed old-vs-new comparison (would need the slow old run) to a direct **weight-load read-back check**: after pushing BF16 weights into vLLM, the sums of four non-fused tensors (embeddings, final norm, layer-0 input norm, layer-27 down-projection) read from the vLLM model must equal the checkpoint's BF16 sums (rel. tol 1e-3) or the job stops; result stored as `weight_check` in each summary. H r36 (11080946) started before this check was added and uses the same loading code. | Deadline (A9/A11); evaluator correctness. |
| D81 | **Cap frozen at 12,288 (A11.2), 2026-11-08 ~03:00Z, before any arm resumes.** Evidence: (a) memory — smoke test passed at 16,384 (peak 37.8 of 44.4 GiB, D79), so 12,288 is also feasible; (b) dev panel at 16,384 from H-s101 round 36, the state nearest round 32 (job 11080946; `research/evidence/a11/`): pass@1 0.606 at 16,384 vs 0.581 within 8,192; 13.7% of responses cross 8,192 and 2.5% cross then finish correctly (18.5% of crossers); correct by length band: ≤ 8,192 58.1%, 8,193–12,288 **2.24%**, 12,289–16,384 **0.29%** → 12,288 keeps 88% of the gain from extending beyond 8,192; exceedance: > 10,240 6.6%, > 12,288 3.1%, > 14,336 1.7%, capped at 16,384 0.94%; crossers show no repetition (loop score ≈ 0.03 in every outcome; 0 above 0.5); extra tokens beyond 8,192 = 9.7% of all tokens; (c) deadline — a round lasts as long as its longest response; with 0.94% of responses reaching 16,384, ~70% of 128-response rounds contain one (generation ≈ 520 s; smoke worst case 12.2 min/round), whereas at 12,288 generation is bounded near ≈ 390 s; ≈ 130 s × 96 rounds ≈ 3.5 h per run, ≈ 7 h on the critical path (seed blocks run sequentially on 3 general GPUs; preempt idle); endpoint test evaluations ≈ 4 h each at the measured 1,393 tokens/s. Projected report time ≈ 10 Nov 14:00–15:00Z at 16,384 (at or past the hard deadline) vs ≈ 07:00–08:00Z at 12,288. Hence **16,384 is feasible in memory but not within the deadline; frozen common cap = 12,288 for rounds 33–128 in all six runs; evaluations at the final cap 12,288 with an 8,192 view; no further changes.** R r52 and U r48 dev 16,384 results (reruns 11081333/4) are reported when they land; the freeze does not depend on them. | project.md §25.2. |
| D82 | **A11 resume (2026-11-08 ~03:10Z).** Superseding (job 11081697): attempt-a1 run and checkpoint directories of R/U/H-s101 moved to `runs/superseded/<run>-a1` and `ckpt/superseded/<run>-a1` (nothing deleted); W&B tag `superseded-A11` requested for the a1 live runs and the old `-curves` companions; companion state reset (new companions are keyed `<run>-a2-curves`). New recipes (commit 9d6349a): attempt a2, cap schedule [[0, 8192], [32, 12288]] (round 33 first at 12,288), engine context 13,312, R-s101 retains round 64 (local-update source, now a mixed 8k/12k state), precheck refuses a run directory or checkpoint tree from another attempt. Old held s202 jobs 11078181/4/5 (old recipes, never started) cancelled; base test eval 11079548 (16,384, never started) cancelled (test recipes now 12,288). Submitted: R-s101 11081724 (moved to general), U-s101 11081725, H-s101 11081726, R-s202 11081729, U-s202 11081731 (preempt; s101 moved to general as the dev evaluations end); H-s202 waits for a preempt slot (cap 6 incl. diagnostic r0). All six start at round 0 (s101 replay rounds 1–32 at 8,192). | project.md §25.3–4. |

## D83 — 2026-11-08 04:05Z — dev16k-R harvested; U-s101 moved to general
- `qis-a11-dev16k-R` (11081333) COMPLETED. R-s101 round 36 (superseded a1 trajectory), dev panel, cap 16,384, n=32.
  pass@1 0.592 at 16,384; 0.572 within 8,192. Crossing 8,192: 19.6% (H: 13.7%). Crossing then correct: 2.0% of all.
  Capped at 16,384: 2.4%. Throughput 1,216 tok/s. Weight read-back check: all 4 tensors match (rel. diff < 1e-5).
- Evidence: `research/evidence/a11/dev16k_R-s101-r36_summary.json`. These are dev outcomes, not test outcomes.
- Moved `qis-main-U-s101-sym` (11081725, ledger row verified) from preempt to general with `ae slurm update`.
  This project now has 2 running + 1 pending on general (limit 3). H-s101 moves when dev16k-U (11081334) ends.

## D84 — 2026-11-08 04:15Z — correction to D83; H-s202 submitted; monitor refreshed
- **Correction.** `qis-a11-dev16k-R` evaluated checkpoint `round_0052` (summary `ckpt` field), not round 36. D83's numbers
  are right; its round label was wrong. Evidence renamed to `dev16k_R-s101-r52_summary.json`.
  Consequence: R (round 52) and H (round 36) differ in training length, so the higher crossing rate of R (19.6% vs 13.7%)
  does not show an arm difference. Compare arms only at matched rounds (U is round 48).
- Submitted `qis-main-H-s202-sym` (11082192, preempt, attempt a2) through `ops/qis-submit` after U-s101 left preempt.
  Its pinned commit is `7c44fa5`; the other five a2 runs use `9d6349a`. The only change between them is
  `wandb_companion.py` (skip run dirs without metrics), which the trainer does not import.
- Resubmitted `qis-monitor` (11082191): the last monitor output predates the A11 pause.

## D85 — 2026-11-08 04:20Z — dev16k-U harvested; H-s101 on general; truncation study rerun
- `qis-a11-dev16k-U` (11081334) COMPLETED. U-s101 round 48 (superseded a1), dev, cap 16,384, n=32: pass@1 0.591 at 16,384,
  0.558 within 8,192; crossing 8,192 19.5%; crossing then correct 3.3% of all; capped 2.3%; 1,134 tok/s; weight check passed.
  Evidence `research/evidence/a11/dev16k_U-s101-r48_summary.json`. Dev outcomes only; rounds differ by arm (H 36, R 52, U 48).
- Moved `qis-main-H-s101-sym` (11081726) to general; it started 04:18Z on babel-w5-24. General: 3/3 = R-s101, U-s101, H-s101.
- Resubmitted `qis-a11-truncstudy` (11082251) over H r36, R r52, U r48.
- R-s101 a2 reached round ~14 by 04:07Z (≈4.7 min/round at 8,192); round 33 expected ≈05:35Z.

## D86 — 2026-11-08 04:25Z — A11 truncation study, all three arms (dev, superseded a1 checkpoints)
Source `qis-a11-truncstudy` (11082251); evidence `research/evidence/a11/trunc_study_HRU.json`. Dev panel, n=32, 1,696
responses per arm, cap 16,384. Rounds differ (H 36, R 52, U 48), so this is a cap study, not an arm comparison.

| | H r36 | R r52 | U r48 |
|---|---|---|---|
| Cross 8,192 | 13.7% | 19.6% | 19.5% |
| Crossers: correct / wrong / capped 16,384 | 43 / 173 / 16 | 34 / 257 / 41 | 56 / 236 / 39 |
| Correct with 8,193–12,288 tokens (share of all) | 2.24% | 1.47% | 2.71% |
| Correct with 12,289–16,384 tokens | 0.29% | 0.53% | 0.59% |
| Exceed 12,288 | 3.1% | 7.2% | 7.4% |
| Loop score > 0.5 among crossers | 0 | 0 | 0 |
| Generated tokens, cap 12,288 vs 8,192 | +8.6% | +13.1% | +13.6% |

Reading: crossing responses do not loop (mean loop score 0.03–0.05). Most crossers end wrong; the 12,288 cap
recovers 1.5–2.7 points of pass@1 on dev and 12,288–16,384 adds 0.3–0.6 more. The cap decision (D81, 12,288) stands;
no test data was used. The fixed 24-response inspection sample is being labelled productive / repetition / extraction.

## D87 — 2026-11-08 04:35Z — inspection sample labelled (24 crossing responses)
An independent subagent labelled the fixed sample (`inspection_sample_HRU.jsonl`, seed 1701 "a11/sample", 8 per arm) from
the text alone; labels in `research/evidence/a11/inspection_labels.json`. Counts verified by script; I checked 3 cases
(00508 s30: boxed 12377 vs gold 12378; 00347 s11/s18: 2/3 chosen from a numerical estimate) against the excerpts.

| | productive | repetition | extraction miss |
|---|---|---|---|
| H r36 | 3 | 5 | 0 |
| R r52 | 5 | 3 | 0 |
| U r48 | 2 | 6 | 0 |

- No extraction misses: every extracted answer equals the final box; the grader's reward is right in all 24.
- "Repetition" here means no progress (restating or re-verifying), not token loops: loop score stays ≤ 0.06; 4 cases
  copy an earlier block verbatim. 22/24 end with an answer; the 2 capped ones do not.
- 5 of 9 correct crossers reach the answer by a guess or contradictory reasoning (e.g. picking 2/3 from a numerical
  estimate). Extra length past 8,192 buys few, and partly unreliable, correct answers. n = 8 per arm: descriptive only.

## D88 — 2026-11-08 05:25Z — capcheck recipe fix
`qis-a11-capcheck` 11082575 FAILED at 0 s: the recipe did not define `PROJ_DATA` (my error). Added the standard
`PROJ_DATA` line; resubmitted as 11082618 (running, waiting for R-s101 round 34). Compute charged: 0 s CPU.

## D89 — 2026-11-08 06:40Z — A11.6 verified for R-s101: round 33 runs at 12,288
Source `qis-a11-capcheck` 11082618; evidence `research/evidence/a11/capcheck_R-s101.json` (attempt a2).

| Round | Logged cap | Max response tokens | Responses > 8,192 | Capped (all at the cap, all graded truncated, reward 0) |
|---|---|---|---|---|
| 31 | 8,192 | 8,192 | 0 | 4 at 8,192 |
| 32 | 8,192 | 8,192 | 0 | 7 at 8,192 |
| 33 | 12,288 | 12,288 | 11 | 1 at 12,288 |
| 34 | 12,288 | 12,154 | 9 | 0 |

- Rollout, training and grading switch together at round 33, as A11 requires. 128 records per round.
- Evaluations at round 32 used 8,192 (`eval_cap(32)` = cap of round 32), as designed. The first evaluation at 12,288 is at
  round 40. `qis-a11-capcheck2` (11083320) checks it and U-s101, H-s101 once all three have logged round 41.
- Note for the report: R-s101 a2 dev truncation at round 32 was 11.1% at 8,192 (dev outcome, not used for any decision).

## D90 — 2026-11-08 07:10Z — forecast from measured round times; seed-202 start order
Measured (monitor 11083516, last ≤6 rounds per run; `t_round` includes evaluation rounds):

| Run | Last round | Mean round time | Cap of those rounds | Capped at last round |
|---|---|---|---|---|
| R-s101 | 39 | 422 s | 12,288 | 2/128 |
| U-s101 | 33 | 294 s (8k), 720 s (one 12k round) | mixed | 6/128 |
| H-s101 | 23 | 569 s | 8,192 | 13/128 |

H-s101 responses are longer (13/128 capped at 8,192 at round 23). Under A8 a training truncation rate above 5% is
inspect-and-report only; it is not a cancellation criterion and no change follows.

Forecast (general only, 3 GPUs): R-s101 ends ≈ 18:00–20:00Z today; U-s101 ≈ 20:00–24:00Z; H-s101 ≈ 9 Nov 02:00–08:00Z
(12k round time for H unknown; assumed 10–15 min). If seed 202 waits for general, the run that starts last finishes
≈ 19–25 h later.
Decision: start the seed-202 runs on general in order H, U, R (longest first). The first general slot to free goes to
H-s202, so the slowest run is not last. This changes only the order of identical, pre-specified runs; no outcome data
was used beyond run duration. Preempt submissions stay queued in case preempt capacity appears first.
Risk: if H-s202 starts ≈ 19:00Z today and takes 25 h, it ends ≈ 9 Nov 20:00Z. U-s202 and R-s202 start ≈ 9 Nov 00:00–08:00Z
and end ≈ 9 Nov 19:00Z – 10 Nov 03:00Z. Test evaluations (~4 h each, 3 GPUs) then end ≈ 10 Nov 07:00Z, before the
14:30Z deadline. The margin is ≈ 7 h; the 10 Nov 01:00Z report target will be missed.

## D91 — 2026-11-08 09:08Z — correction to D89 (evaluation rounds); capcheck2 replaced
- **Correction.** `eval_rounds` is [0, 32, 64, 96, 128] (main recipes). D89 said the first evaluation at 12,288 is at
  round 40; it is at round 64 (`eval_cap(64)` = cap of round 64 = 12,288). Checked against the recipe config.
- `qis-a11-capcheck2` 11083320 waited for round 41, which served no purpose and risked its 4 h limit. Cancelled after
  2:33 h of a waiting CPU job (charged: 2.55 CPU-job-hours, 2 CPUs). Replacement 11084106 waits for round 36 in all three
  s101 runs (transition rounds 31-35). The round-64 evaluation cap is checked after round 64.
- Progress at 09:05Z (monitor 11084097): R-s101 round 52, U-s101 round 48, H-s101 round 32. 12k round times: R 285–965 s,
  U 294–568 s. H at 8k: 521–626 s, capped 14–23 of 128 per round (report-only, A8).
- Dev at round 32 (a2; viewed, not used for decisions): H 0.540, R 0.542, U 0.601.

## D92 — 2026-11-08 09:55Z — A11.6 verified for all three s101 runs
Source `qis-a11-capcheck2` 11084106; evidence `research/evidence/a11/capcheck_s101.json` (all attempt a2).
In R-, U- and H-s101, rounds 31–32 log `max_tokens` 8192 and rounds 33–35 log 12288. Every round has 128 records;
every capped response ends exactly at its round's cap and is graded truncated with reward 0.

| Capped / responses > 8,192 (of 128) | Round 32 | Round 33 | Round 34 | Round 35 |
|---|---|---|---|---|
| H | 23 / 0 | 12 / 21 | 3 / 10 | 11 / 28 |
| R | 7 / 0 | 1 / 11 | 0 / 9 | 1 / 8 |
| U | 12 / 0 | 6 / 8 | 5 / 15 | 2 / 5 |

H still caps 2–9% of responses at 12,288 (report-only under A8). Remaining: the seed-202 runs (same check after
their round 35) and the round-64 evaluation cap.

## D93 — 2026-11-08 11:08Z — deadline risk: seed-202 order changed to R, U first; H-s202 only if capacity allows
Measured round times at 12,288 (monitor 11084490, rounds shown in peek; include no evaluation round):
R-s101 615–919 s (mean ≈ 743 s), U-s101 377–637 s (≈ 515 s), H-s101 802–1,109 s (≈ 950 s). H caps 7–16% at 12,288.
Progress at ≈11:05Z: R round 63, U round 62, H round 40. Preempt: no start estimate for any queued job (825 pending).

Projected ends of seed 101 (incl. ≈1 h of evaluations): U ≈ 21:40Z today; R ≈ 9 Nov 01:30Z; H ≈ 9 Nov 11:30Z.
Projected seed-202 durations from the same profiles: U ≈ 18 h, R ≈ 25 h, H ≈ 32 h.
Sealed-test evaluations: 7 × ≈ 4 h GPU (128 problems × 32 at ≈ 1,200 tok/s, measured on dev).

D90's order (H first) gives last training end ≈ 10 Nov 05:40Z; the 7 test evaluations then need ≈ 10 h on 3 GPUs
→ ≈ 15:40Z, after the 14:30Z deadline. Under A9 (deliver verified partial results if the deadline is at risk) and the
A10 continuation (prioritize paired R−U), the plan becomes:
1. First general slot (U-s101 end) → R-s202 (≈ 25 h → ≈ 9 Nov 22:40Z).
2. Second slot (R-s101 end) → U-s202 (≈ 18 h → ≈ 9 Nov 19:30Z).
3. Third slot (H-s101 end) → sealed-test evaluations: base, then H/R/U-s101; R/U-s202 evaluations as they finish.
   All evaluations end ≈ 10 Nov 03:30Z; report ≈ 08:00Z.
4. H-s202 stays queued on preempt (11082192). It moves to general only if a re-forecast shows it and its evaluation
   finish by ≈ 10 Nov 10:00Z. Otherwise the report gives R−H for seed 101 only, marked partial.
This uses only run durations, never outcomes. Supersedes the order in D90. Re-forecast hourly; the decision binds at
the first slot (≈ 21:40Z).

## D94 — 2026-11-08 12:08Z — round-64 evaluations run at 12,288 (A11.6 grading cap verified); forecast refresh
- Monitor 11084694: R-s101 and U-s101 logged their round-64 evaluations with `max_tokens` 12288 (R: dev; U: dev,
  probe_J_R, probe_J_U; R's probes were still running). Training and evaluation caps now verified for the transition.
- Viewed, not used for decisions (dev, round 64, cap 12,288): R 0.736 (truncated 19.8%), U 0.710 (14.9%); derived
  J_H R 0.738, U 0.734. Dev truncation at 12,288 is higher than in the a1 dev-16k study (7.2–7.4% exceed 12,288 at
  rounds 48–52): responses lengthen with training. Report-only (A8).
- A dev evaluation at 12,288 takes 2,378–2,826 s (≈ 40–47 min).
- Forecast refresh (round times last 6 rounds: R ≈ 735 s, U ≈ 496 s, H ≈ 926 s): U-s101 ends ≈ 22:20Z, R-s101 ≈
  9 Nov 02:20Z, H-s101 ≈ 9 Nov 11:30Z. D93 plan unchanged. R-s101 `round_0064` now exists, so `qis-diag-local-r64`
  can pass its precheck when it starts.

## D95 — 2026-11-08 14:15Z — forecast refresh (rounds lengthen)
Monitor 11085378 (≈14:10Z): U-s101 round 76 (mean last-6 round 615 s), R-s101 72 (808 s), H-s101 52 (965 s).
Capped per round (of 128): U 3–12, R 6–16, H 10–31. Round times grow as responses lengthen.
Projected ends (incl. evaluations at 96 and 128, ≈ 0.8 h each): U-s101 ≈ 9 Nov 00:40Z; R-s101 ≈ 04:20Z; H-s101 ≈ 13:00Z.
D93 plan: R-s202 (≈ 27 h) from ≈ 00:40Z → ≈ 10 Nov 03:40Z; U-s202 (≈ 20 h) from ≈ 04:20Z → ≈ 10 Nov 00:20Z; test
evaluations from ≈ 9 Nov 13:00Z, all done ≈ 10 Nov 08:00Z if each takes 4–5 h. Margin to 14:30Z ≈ 6 h. H-s202 does
not fit on general under these projections; it remains queued on preempt only.

## D96 — 2026-11-08 15:05Z — A12: deadline 10 Nov 16:00Z (11:00 AM EST); plan D93 kept
The extra 1.5 h does not make H-s202 fit. With H-s202 on the first general slot (≈ 9 Nov 00:40Z, ≈ 34 h), R-s202 and
U-s202 move to the later slots, and no general GPU is free for the sealed-test evaluations until ≈ 10 Nov 07:00Z;
7 evaluations × 4–5 h on 3 GPUs then end ≈ 17:00–18:00Z, after 16:00Z. D93 (R-s202, U-s202, then evaluations; H-s202
on preempt only) keeps all evaluations ≈ 10 Nov 08:00Z; margin to the deadline ≈ 8 h. Re-checked at every slot.

## D97 — 2026-11-08 15:15Z — sealed-test evaluation time limit 5 h → 9 h
Expected run time per test evaluation: 128 × 32 = 4,096 completions; at the measured `evaluate_ckpt` throughput
(1,134–1,393 tok/s, D83/D85) and ≈ 4,700–5,000 mean tokens (later policies are longer) this is ≈ 4.0–5.0 h, too close
to the old 5 h limit; the eval phase is not resumable, so a timeout loses all work. Raised to 9 h in
`make_test_exps.py` and the 7 `exp/qis-test-*` recipes. No change to n, cap, seeds or problems.

## D98 — 2026-11-08 16:15Z — forecast refresh (U slower, H faster); plan D93 kept
Monitor 11086128 (≈16:10Z): U-s101 round 85 (mean last-8 round 744 s), R-s101 81 (845 s), H-s101 61 (806 s; was 965 s).
Projected ends (incl. evaluations at 96 and 128, ≈ 0.8 h each): U-s101 ≈ 9 Nov 02:40Z (9:40 PM EST Sun); R-s101 ≈ 04:45Z
(11:45 PM EST); H-s101 ≈ 09:30Z (4:30 AM EST Mon). D93 order: R-s202 ≈ 27 h → ≈ 10 Nov 05:40Z; U-s202 ≈ 20 h → ≈ 10 Nov
00:45Z; test evaluations from ≈ 9 Nov 09:30Z on the third slot, then on the U-s202 and R-s202 slots; last (R-s202) ends
≈ 10 Nov 10:40Z (5:40 AM EST). Margin to 16:00Z ≈ 5 h. H-s202 (≈ 27 h + 5 h evaluation from 09:30Z → ≈ 17:30Z) still
does not fit; it stays on preempt only.

## D99 — 2026-11-08 17:10Z — A13: round-32 dev discrepancy explained by the A11 replay; canonical = a2
Source: job 11086589 (`qis.analysis.r32_provenance`, commit ade7e1e), output `research/evidence/a11/r32_provenance.json`.

| Run | Attempt | Job | Code sha | Logged pass@1 | Correct / denominator (raw records) | Capped at 8,192 | Records sha256 (prefix) |
|---|---|---|---|---|---|---|---|
| R-s101 | a1 (superseded) | 11078086 | b8f36ca | 0.507 | 215 / 424 = 0.5071 | 49 | 15426dcf67e78896 |
| R-s101 | **a2 (retained)** | 11081724 | 9d6349a | 0.542 | 230 / 424 = 0.5425 | 47 | b45d98758453a3ba |
| U-s101 | a1 (superseded) | 11078164 | efdafdd | 0.547 | 232 / 424 = 0.5472 | 71 | 51faa5dadb112ab7 |
| U-s101 | **a2 (retained)** | 11081725 | 9d6349a | 0.601 | 255 / 424 = 0.6014 | 73 | afd87b4eec5d9239 |
| H-s101 | a1 (superseded) | 11078180 | 0a83a74 | 0.552 | 234 / 424 = 0.5519 | 66 | 86cdb1c12fe21b80 |
| H-s101 | **a2 (retained)** | 11081726 | 9d6349a | 0.540 | 229 / 424 = 0.5401 | 72 | de338f0a7405b126 |

Common to all six: panel dev (53 problems × 8 samples = 424); bundle sha 0e255243; sampling seeds identical per
(problem, sample) across attempts (EVAL_SEED 4242, key `eval/dev/<id>/r32/n8`); cap 8,192 (all capped responses end at
8,192); verifier, prompts and seed code byte-identical between the a1 commits and 9d6349a (`git diff`); exactly one dev
round-32 row per attempt (no repeated evaluation); logged per-problem counts equal the raw-record recomputation in all
six (no reporting error); macro and micro pass@1 agree (n = 8 for every problem).
Cause: the A11 replay. The a2 runs retrained rounds 1–32 from the base model with the same prompts and seeds, but the
rollouts are not bitwise reproducible: already in training round 1 (base weights) only 36/128 (R), 126/128 (U) and
3/128 (H) responses are token-identical between attempts (engine context 9,216 → 13,312 and rollout code changed in
A11; vLLM sampling is batch-dependent). The trajectories then diverge, so a2 round 32 is a different policy from a1
round 32. None of the 424 dev responses is token-identical across attempts; rewards agree on 319/333/327 of 424.
Checkpoint hash: unavailable for both attempts; neither retains round_0032 (rolling deletion; a1 holds 52/48/36 only,
a2 holds 64+/88/60). The policy is identified by run, attempt, round and the records sha256 above.
Decision: canonical round-32 series = a2 (prespecified round-32 evaluation on the retained trajectory, A11.4). a1 values
stay in D73/D77 and in `runs/superseded/<run>-a1` as superseded evidence; they appear in no table or figure. Status:
resolved. Qualification that remains: round 32 (cap 8,192) vs round 64 (cap 12,288) mixes training progress with the
cap change; any 32→64 comparison is labelled accordingly. Size check: a1→a2 differences (+0.035, +0.054, −0.012) are
of the order of evaluation sampling sd (≈ 0.02 per value) plus trajectory variation; R−U at round 32 is −0.040 (a1)
and −0.059 (a2); dev is a monitoring panel only.

## D100 — 2026-11-08 18:20Z — forecast refresh; plan D93 kept
Monitor 11087148 (≈18:15Z): U-s101 round 95 (mean last-8 723 s), R-s101 90 (792 s), H-s101 68 (682 s). H dev round 64
logged (12,288 cap; monitoring only). Projected ends: U-s101 ≈ 9 Nov 02:20Z (9:20 PM EST Sun), R-s101 ≈ 04:05Z (11:05 PM
EST), H-s101 ≈ 07:05Z (2:05 AM EST Mon). R-s202 (≈ 27 h) → ≈ 10 Nov 05:20Z; U-s202 (≈ 20 h) → ≈ 10 Nov 00:05Z; test
evaluations from ≈ 9 Nov 07:05Z; last (R-s202) ≈ 10 Nov 10:20Z (5:20 AM EST); margin ≈ 5.5 h. H-s202 from the H-s101 slot
(≈ 27.7 h + ≈ 5 h test evaluation) would end ≈ 10 Nov 15:45Z and would block four test evaluations: still does not fit.

## D101 — 2026-11-08 21:10Z — base-model test evaluation queued early on preempt
`qis-test-base` (job 11089196, commit 779c54d, recipe 21506d37a4f3; 128 × 32, cap 12,288, limit 9 h) submitted to the
free sixth preempt slot (preempt now 6/6). It depends on no training output and informs no training decision; its
result stays unread until the report stage. If it starts and finishes, the D93 test queue on general shrinks by one
evaluation (≈ 5 h). If it is still pending when H-s101 ends, it moves to general (`ae slurm update ... Partition=general
QOS=normal`). If preempted mid-run it restarts from scratch (not resumable; compute charged).

## D102 — 2026-11-09 02:45Z — first general slot: U-s101 done → R-s202 on general (D93)
U-s101 (11081725) COMPLETED 0:0 at 02:40:59Z (elapsed 22:37:01; allocation ended per sacct). R-s202 (11081729, ledger row
qis-main-R-s202-sym, sha 9d6349a) moved preempt → general/normal at ≈02:42Z and started immediately on babel-w5-24.
Endpoint validation `qis-validate-endpoints` (11092019, commit c430002) submitted for every run with round_0128.
Preempt now 5/6 owned (diag r0, U-s202, H-s202, diag r64, test-base). R-s202 projected ≈ 25–27 h → ≈ 10 Nov 04:00–06:00Z.

## D103 — 2026-11-09 04:30Z — R-s101 final checkpoint failed (transient /data error); one recorded retry from round 124
Evidence: job 11081724 FAILED 5:0 at 04:19:06Z (elapsed 1-01:17:31 = 25.3 GPU-h, charged). Exit 5 = `final_checkpoint_failed`
(run.py:378–380). `round_0128.tmp` holds full-size model.pt and optim.pt but no state.json (listing job 11092924); the
blocking save ran ≈ 03:26–04:19Z. Same pattern as the Errno 512 failures at rounds 108 and 120 and R-s202 round 4 on another
node (Reviews 48, 50) → transient /data infrastructure error, not code or data. Last valid checkpoint: round_0124
(LATEST.json; sha-verified at write). Rounds 1–128, all scheduled evaluations up to round 128 and the training curve were
complete; only the final checkpoint is missing, and the test evaluation needs it.
Action (one recorded retry, agent.md): (1) job 11092933 moved the failed trajectory's records train_r0124–r0127 and
eval_{dev,probe_J_R,probe_J_U}_r0128 into `runs/qis-main-R-s101-sym/records/superseded-11081724/` (nothing deleted),
because the trainer skips an evaluation whose records file exists and would otherwise pair stale round-128 evaluations
with the replayed policy; (2) resubmitted qis-main-R-s101-sym as 11092945 (commit 0afd6e8: no diff to 9d6349a in
src/qis/train, verify, prompts, seeds; recipe hash 71f6f8b7ef45 = original; attempt a2) after confirming 11081724's
allocation ended; moved to general; running since 04:27Z. It resumes from round_0124 and replays rounds 125–128
(≈ 1 h), reruns the round-128 evaluations (≈ 1 h) and saves round_0128.
Provenance: canonical rounds 125–128 and round-128 evaluations = the retry's (retained trajectory). metrics.jsonl holds
both; analysis keeps the rows after the retry's `resume` event (last occurrence per round/panel; wandb_companion and
curves already take the last occurrence / the records files). The superseded records and the round-128 dev 0.710 of the
failed trajectory are kept as evidence only. If this retry's final save fails too, stop and investigate /data (no
further automatic retries).
Slot effect: this general slot runs the R-s101 retry (≈ 2–2.5 h) before U-s202 (D93 order otherwise unchanged). U-s202
then ends ≈ Mon 10 PM EST; its test evaluation ≈ Tue 3 AM EST, still before R-s202's (≈ 5:30 AM EST).

## D104 — 2026-11-09 05:20Z — /data write slowdown threatens R-s202; checkpoint-cadence option prepared, trigger set
Evidence (monitor 11093383, ≈05:10Z): R-s101 retry 11092945 resumed at round 124 (`resume` event 04:51Z; checkpoint load
24 min) and completed round 125. Checkpoint writes (22 GB each) now take 55–75 min (≈ 6 MB/s) vs 2–5 min before:
H-s101 round 112 04:01→04:57Z; R-s202 round 8 03:44Z → still busy when round 12 finished (Saver serialises writes, so
round 12's save waited ≈ 57 min while the GPU idled; rounds 9–12 took 1,017 s of compute). Errno 512 failures on 4 saves
in 3 runs on 3 nodes. Projection without stalls: R-s202 ends ≈ 10 Nov 07:10Z, test evaluation ≈ 12:10Z (margin ≈ 4 h).
With a ≈ 1 h stall every 4 rounds it would not finish before the deadline.
Option (operational, not scientific; PROTOCOL does not fix `ckpt_every`): raise `ckpt_every` 4 → 16 for seed-202 runs
on general (non-preemptible), cutting writes 4×; cost: restart R-s202 from its latest committed checkpoint (≤ 4 rounds +
≈ 25 min load) and a larger loss window on a node failure. U-s202 (not started) could take it at no restart cost.
Trigger: `qis-ckpt-watch` 11093453 reports when R-s202's round-12 checkpoint commits. If that write takes > 20 min,
implement the change through `make_main_exps.py` (no hand edits), validate the generated config, and apply it to U-s202
before it starts and to R-s202 by restart. If writes are back to ≤ 10 min, no change.

## D105 — 2026-11-09 06:25Z — D104 trigger met: seed-202 runs on general move to `ckpt_every` 16
Evidence (`qis-ckpt-watch` 11093453): R-s202's round-12 save wrote 05:05–05:56Z and then failed (round_0012.tmp left, no
state.json); its round-16 save committed at 06:16:54Z (≈ 14 min write after waiting for round 12). Trigger (> 20 min) met.
R-s202 is in the 8k phase (≈ 280 s/round, 4 rounds ≈ 19 min), so 15–50 min writes stall the GPU at every save until
round 32; fewer writes also cut contention on the shared checkpoint lock that R-s101's and H-s101's final saves need.
Actions:
1. Recipes regenerated by `make_main_exps.py --ckpt-every 16` (commit dd71b20; default 4 reproduces all six current recipes
   byte-identically; the 16 variant differs only in `ckpt_every`). Installed for R-s202 and U-s202 only. H-s202 keeps 4
   (preempt-only: preemption recovery needs short intervals). Cadence is operational; PROTOCOL does not fix it; the
   training code path is unchanged.
2. U-s202 11081731 (never started, old recipe) cancelled; resubmitted as 11094037 (recipe d01e221ecd03), queued on preempt,
   moves to general when the R-s101 retry ends (D103).
3. R-s202 11081729 cancelled at 06:18:01Z (≈ 3.6 h, charged; allocation ended per sacct/squeue). Records of its rounds
   17–19 (train_r0016–r0018) preserved in `runs/qis-main-R-s202-sym/records/superseded-11081729/` (job 11094187).
   Resubmitted as 11094190 (commit dd71b20, recipe 51ccb108474c, attempt a2), moved to general, running on babel-p9-24
   from 06:20Z; resumes from round_0016. Lost work ≈ 3 rounds + checkpoint load.
Provenance: canonical R-s202 rounds 17+ = the resumed job's; metrics rows after its `resume` event win (last occurrence).
Rounds 1–16 at cadence 4 and 17–128 at 16 is a logging/recovery difference only. Risk accepted: a node failure on general
loses ≤ 16 rounds.

## D106 — 2026-11-09 08:30Z — forecast after D103/D105
Monitor 11094805 (≈08:25Z): H-s101 round 127 (mean last-6 921 s); R-s202 33 distinct rounds (resumed at 16 at 06:24Z; round 33
is its first 12,288 round; checkpoint 32 writing); U-s202 19 (started 07:06Z; checkpoint 16 committed in 15 min).
H-s101 also had an Errno 512 failure at round 120 (124 committed). Projections: H-s101 ends ≈ 10:00–10:30Z (round-128
evaluations + blocking save); R-s202 ≈ 95 rounds × 780 s + 3 evaluations → ≈ 10 Nov 07:30Z; U-s202 ≈ 13 × 300 s + 96 × 720 s
+ 4 evaluations → ≈ 10 Nov 07:30Z. Test evaluations: s101 runs and base on the H-s101 slot from ≈ 10:30Z (≈ 5 h each);
s202 runs on their own slots from ≈ 07:30Z → last ends ≈ 10 Nov 12:30–13:00Z (7:30–8:00 AM EST). Margin to 16:00Z ≈ 3 h,
down from 5.5 h (D100) because of the failed final save and the restarts. Report drafting proceeds in parallel on the
s101 results so that only the seed-202 rows remain at the end.

## D107 — 2026-11-09 09:15Z — test-evaluation order on the H-s101 general slot
Primary contrast first: `qis-test-R-s101-sym` (11095106) and `qis-test-U-s101-sym` (11095118) queued on the two free preempt
slots (preempt 6/6 owned; they start there only if preempt frees). When H-s101 ends, move to general in order: R-s101 →
U-s101 → base (11089196) → H-s101 (submitted after its validation). Each ≈ 5 h (limit 9 h). Seed-202 test evaluations run
on the seed-202 slots when those runs end (≈ 10 Nov 07:30Z). Test outcomes are not read until all training has ended and
inform no training decision (A13).

## D108 — 2026-11-09 12:15Z — forecast refresh (seed 202 faster than D106)
Monitor 11095848 (≈12:10Z): R-s202 completed round 62 (last-6 mean 467 s/round; checkpoint 48 committed in 19 min);
U-s202 completed round 51 (last-6 mean 548 s; checkpoint 48 in 22 min). D106 assumed 720–780 s/round. Remaining:
R-s202 65 rounds ≈ 8.5 h + 3 evaluations → ends ≈ 21:30–23:00Z today; U-s202 76 rounds ≈ 11.6 h + 3 evaluations →
ends ≈ 10 Nov 00:30–02:30Z. Ranges allow round times to rise to D106's rate in the remaining rounds. Seed-202 test evaluations
(≈ 5 h each) then end ≈ 10 Nov 03:00–07:30Z. Margin to 16:00Z ≈ 8–13 h (D106: ≈ 3 h). No action changes: the D107 order
stands; H-s202 (≈ 29.5 h on 12k, by H-s101) still cannot finish by the deadline and stays pending on preempt only.

## D109 — 2026-11-09 14:16Z — test U-s101 moved to general (D107 step)
Test R-s101 11095106 COMPLETED 14:14Z (general, 4:23 elapsed; allocation ended, verified with sacct and squeue). Test
U-s101 11095118 (ledger-owned) moved preempt → general/normal at 14:16Z, the next step of the D107 order. Test outcomes
remain unread until all training ends (A13).

## D110 — 2026-11-09 16:40Z — R-s202 round-80 save failed (Errno 512); continue, no action
Monitor 11097078: R-s202 (11094190) `checkpoint_failed` round 80 at 15:25Z, `OSError: [Errno 512]`, 8 min after start;
the run continued and completed round 85 (last-6 mean 505 s/round). Same /data fault as D103/D104; the H-s101 precedent
(round 120 failed, round 124 committed) shows a failed asynchronous save does not stop training. Last valid R-s202
checkpoint: round 64 (13:13Z, 815 s write). U-s202 committed round 64 (no failure). A failed save is not a cancellation
criterion (owner rule). Exposure: a node failure before round 96 commits would lose ≤ 32 rounds (≈ 4.5 h); a failed
final save at 128 exits 5 and needs one recorded retry from round 112 (≈ 16 rounds + load ≈ 2.5–3 h, D103 precedent).
D108 margin (≈ 8–13 h) absorbs either. A `round_0080.tmp` directory (≤ 22 GB) is left; storage cleanup only under the
owner's rule. Next check: round-96 save in the next monitor.

## D111 — 2026-11-09 17:45Z — U-s202 failed (Errno 512 on metrics write); one recorded retry from round 64
Evidence: job 11094037 (general, babel-s5-32) FAILED 1:0 at 17:40:24Z after 10:36:29 (10.6 GPU-h, charged). Traceback:
`run.py:312 log.write("step", ...)` → `OSError: [Errno 512]` on `metrics.jsonl` flush; the following CUDA allocator error
is shutdown noise. Same /data fault as D103/D104/D110. Last train record r0079 (17:02Z); last metrics row
`checkpoint_started` round 80 (17:02Z); round 80 never committed (`round_0080.tmp` left). LATEST = round 64 (14:47Z,
state.json present). Classified as a transient infrastructure error → the one recorded retry allowed by agent.md/§owner
rules. U-s202 has had no earlier failure retry (11081731 was replaced before starting, D105).
Actions:
1. `qis-u202-recover-prep` 11097803: moved train_r0064–r0079 (16 files) to `records/superseded-11094037/`; no eval file
   after round 64 existed; metrics.jsonl last line valid (no repair). Nothing deleted.
2. Allocation of 11094037 ended (sacct End 17:40:24Z; not in squeue). Retry 11097811 submitted via `ops/qis-submit`
   (QIS_A11_RESUME=1; same recipe d01e221ecd03, ckpt_every 16; commit 6724cce, no code change) and moved to general/normal
   at 17:45Z. It resumes from round 64 (attempt a2; duplicate metrics rows after resume: last occurrence canonical).
Cost: 16 rounds of lost progress (≈ 2.4 h) + ≈ 25 min checkpoint load. Forecast: 64 rounds × ≈ 540 s + 2 evaluations +
final save → ends ≈ 10 Nov 03:30–05:00Z; its test evaluation then ends ≈ 08:00–10:00Z. Margin to 16:00Z ≈ 6–8 h.
A second infrastructure failure of U-s202 gets no automatic retry: it goes to the owner with the then-current forecast.
The base test waits for test U-s101's slot (D107); the freed slot goes to the critical-path retry.

## D112 — 2026-11-09 18:43Z — test-base moved to general (D107 step)
Test U-s101 11095118 COMPLETED 18:40:36Z (general, 4:24; allocation ended, not in squeue). Test-base 11089196
(ledger-owned) moved preempt → general/normal at 18:43Z. Remaining on this slot after base: test H-s101 11095486
(≈ 4.5 h each → slot free ≈ 10 Nov 04:00Z). Test outcomes unread (A13).

## D113 — 2026-11-09 19:15Z — test-base done in 30 min (verified complete); test H-s101 moved to general
Test-base 11089196 COMPLETED 19:12Z after 29:59. Short runtime checked before use (outcome fields not read):
128 problems × n 32 = 4,096 completions, max_tokens 12,288, seed_label test-final, 6.27 M tokens (≈ 1,530 per
completion), truncated 0.05%, 3,566 tok/s. Trained s101 tests (same structure, 4,096 completions, cap 12,288,
test-final): R 18.7 M tokens, truncated 10.7%, 4:04 h of generation; U 19.1 M tokens, truncated 12.2%, 4:15 h. The base
model writes ≈ 3× fewer tokens, and long sequences lower throughput (≈ 1,260 tok/s), which explains the gap.
Test H-s101 11095486 (ledger-owned) moved preempt → general/normal at 19:14Z (last step of D107 on this slot); ends
≈ 23:45Z. This slot is then free ≈ 4 h earlier than D112 assumed; its use (diagnostics 11079411/11084344 or idle) is
decided then, without reference to test outcomes.

## D114 — 2026-11-09 20:45Z — local-update diagnostic made resumable; slot plan for the remaining GPU work
The local-update diagnostic (A9.2, A1.2 "Local update effects"; jobs 11079411 r0, 11084344 r64) has never started on
preempt. Size: 1 source + 3 branches × 4 minibatch seeds = 13 evaluations × 92 problems × 16 = 19,136 completions at
cap 8,192. Estimates from the s101 test throughput (D113): r0 (base source, ≈ 1,500 tokens/answer, ≈ 3,500 tok/s)
≈ 2.5–3 h; r64 (trained source, ≈ 3,000 tokens/answer, ≈ 1,300–1,500 tok/s) ≈ 10–12 h, above its 8 h limit. The old
code ran branch-major (all hard seeds, then related, then unrelated) and could not resume, so a timeout would lose the
unrelated branch entirely.
Change (commit 020b08b; operational, per-branch results unchanged: each branch restores the same snapshot and draws
seeds fixed by (branch, seed)): seed-major order (hard, related, unrelated for 4101, then 4102, …) and resume that skips
(branch, seed) rows already in `diag_<label>.jsonl` (partial last line ignored). Test added; full suite job 11099032.
r64 recipe: time 8 h → 14 h; precheck now asserts the source `state.json` has attempt_id a2 (D99).
Slot plan (general, 3 GPUs; from ≈ 23:45Z): slot of test H-s101 → diagnostic r64 (longest first); R-s202 slot →
R-s202 endpoint validation + test, then diagnostic r0; U-s202 slot → U-s202 validation + test. Expected: all GPU work
done ≈ 10 Nov 10:00–12:00Z; a partial r64 (complete seed sets only) is reported as partial (A9). Old diagnostic jobs
(pending, never started, cost 0) are cancelled and resubmitted at commit 020b08b after the suite passes.
D114 execution (21:20Z): suite job 11099032 46/46 passed (includes the new resume test). Old diagnostic jobs 11079411 (r0)
and 11084344 (r64) cancelled (ledger-owned; pending since submission, never started; cost 0). Resubmitted at commit
f3c70b5: r0 11099083 (8 h), r64 11099088 (14 h), both queued on preempt; r64 moves to general when test H-s101 ends.
Report pipeline: training venv has no matplotlib; figures use a separate venv `$PROJ_DATA/env/venv-report`
(matplotlib 3.9.2, built by `qis-report-dry`; training venv untouched). First dry run 11099043 failed on the import
(no output; ≈ 0 CPU-min); retry 11099061.

## D115 — 2026-11-09 20:35Z — report pipeline dry run; interim dev curves viewed
`qis-report-dry` 11099061 (commit f3c70b5): `report_data` over the six runs: attempt a2 on every row of every run
(attempt_ok True); evaluations one canonical row per (panel, round) — R-s101 dev round 128 was logged twice (the failed
final-save job and the D103 retry each evaluated it; records of the first were moved aside), the last is canonical.
Truncation of hard training responses, rounds 33-128 so far: H-s101 7.1%, R-s101 10.6%, U-s101 14.0%, R-s202 7.9%
(to round 104), U-s202 9.2% (to round 80); auxiliary 0.0-1.6%. Active training hours on the canonical trajectory: H-s101
26.3, R-s101 21.0, U-s101 19.3 (failed and superseded segments are charged separately from sacct). Figures drawn:
learning vs rounds and vs cumulative hard responses, POPE (J_R, J_H) and (J_U, J_H).
**Viewed (orchestrator):** dev pass@1 curves of all runs to their latest scheduled evaluation (s101 to 128, R-s202 to 96,
U-s202 to 64) and the probe trajectories. Not test data. No decision remains that these could influence: training
configuration, schedules and the analysis are fixed; the remaining actions (test evaluations, diagnostics) are fixed in
D107/D114. Note for the report: round-0 dev values differ across runs (0.403-0.422) although model and seeds are
identical — vLLM generation is not bitwise reproducible (D99); this is a floor on evaluation noise.

## D116 — 2026-11-09 21:20Z — ROOT CAUSE of the Errno 512 failures: account quota full (495/500 GB); prune under §11
Evidence: `qis-storage-df` 11099700 (21:11Z): `nas5:/srv/nfs/data/user_data/vineethb  500G  495G  5.4G  99%`. The
quota is shared with the owner's other projects. A 22 GB checkpoint cannot fit; small metrics writes also fail when the
quota is exhausted, which explains U-s202's crash (D111) and the save failures D103/D104/D110 and R-s202 round 96
(`checkpoint_failed` 19:44Z after 76 min). Errno 512 is the kernel's ERESTARTSYS surfacing from the NFS client, not a
data-corruption signal; committed checkpoints (state.json + sha256) remain valid. Our earlier diagnosis "slow /data"
(D104) was incomplete: the slowness and failures track the quota filling, partly with our own failed `.tmp` writes.
Rule (project.md §11): keep ≥ 50 GiB free; below the floor, pause new large writes; prune superseded project-owned
copies after verifying references and recording deletions; never use other projects' space.
Action: `qis-prune-tmp` 11099764 deletes only (1) failed partial writes `round_*.tmp` of the completed s101 runs and
R-s202's dead round_0080.tmp / round_0096.tmp (U-s202 skipped: a save is running) and (2) the superseded attempt-a1
checkpoint trees under `ckpt/superseded/` (A11/D82; references checked: no active or pending recipe uses them; their
evidence use — truncation study D86/D87, provenance D99 — is complete and recorded). Each deletion is logged with its
size (`deleted.txt`, copied to research/evidence/storage/). Not touched: logs, metrics, records, valid checkpoints,
R-s101 round_0064 (diagnostic source), s101 round_0128 (endpoints).
Next, if still below the floor: retire optimizer state of s101 endpoints whose test evaluation completed (R, U; H after
its test ends), keeping model.pt + state.json (§11 retirement rule; the test evaluator reloaded and verified model.pt).
Contingency fixed now, before any further result: if R-s202's final save fails (exit 5), it gets its one recorded
retry from its latest committed checkpoint (112 if that commits, else 64). From 64 the retry cannot finish with a test
before 16:00Z; R-s202's test row is then reported as missing (A9.5).

## D117 — 2026-11-09 21:55Z — prune result; U-s202 failed again; resumed after the root cause was fixed
Prune `qis-prune-tmp` 11099764 (21:24–21:28Z) deleted 164 GB, every path logged (`research/evidence/storage/
prune_11099764_deleted.txt`): `.tmp` failed writes H-s101 r120 8.1, R-s101 r108 19.6 and r120 18.8, R-s202 r80 21.9 and
r96 8.1 GB; superseded a1 trees H/R/U-s101 21.9 GB each. Account quota 495 → 425 GB used (76 GB free; floor 50 GiB).
U-s202 retry 11097811 FAILED at 21:31:06Z (3:45 h): `InterruptedError: [Errno 4]` closing `train_r0080.jsonl.gz`
(run.py:355). Errno 4 (EINTR) and Errno 512 (ERESTARTSYS) are the same mechanism: a signal interrupts a slow NFS call
and Python does not retry close(). The NFS calls were slow because the account quota was full (D116); this failure
came 3 min after the prune ended, while the in-flight close was still affected. LATEST = round 64 (round-80 save
started 20:38Z, never committed); `qis-u202-recover-prep2` 11100079 moved train_r0064–r0080 (17 files) to
`records/superseded-11097811/`; metrics last line valid. 3.75 GPU-h charged.
Decision: resume U-s202 a second time (its D111 retry is used; D111 said a second failure goes to the owner). Reasons:
the owner authorised resuming project jobs without asking and asked that all six runs keep progressing (A13); the
one-retry rule guards against repeating an undiagnosed failure, and this cause is now diagnosed and removed (quota 99% →
85% → ≈ 80% after the optimizer retirement below). The owner is told at once and may cancel. A third failure stops the
run; U-s202 is then reported as missing (A9.5).
Storage for peak writes: `qis-prune-optim` 11100135 retires optimizer state of R-s101 and U-s101 round_0128 (their test
evaluations reloaded model.pt with matching sha256; endpoint validation passed; no planned branch uses them; model.pt
and state.json kept) and removes U-s202's dead round_0080.tmp. Forecast: U-s202 from round 64 ≈ 25 min load + 64 rounds
× ≈ 600 s + 2 evaluations ≈ 11.5 h → ends ≈ 10 Nov 09:30Z; test ≈ 4.5 h → ≈ 14:00Z; report must then finish in ≈ 2 h.
D117 execution (21:57Z): `qis-prune-optim` 11100135 deleted R-s101 and U-s101 round_0128/optim.pt (13.8 GB each) and
U-s202 round_0080.tmp (21.9 GB); model.pt (8,127,059,673 bytes) and state.json kept in both. Quota now 366/500 GB used
(≥ 135 GB free). Deletion logs: `research/evidence/storage/qis-prune-tmp_deleted.txt`, `qis-prune-optim_deleted.txt`.
U-s202 resume 11100159 submitted at d90dd8c (recipe d01e221ecd03, unchanged) and moved to general/normal.

## D118 — 2026-11-09 23:10Z — seed-202 tests queued behind their runs on general
Test R-s202 11100782 (afterany R-s202 11094190) and test U-s202 11100785 (afterany U-s202 11100159) submitted at 73597a9
and moved to general/normal, so each GPU passes from training to its test without an orchestrator wake-up overnight.
Their prechecks require RUN_COMPLETE and round_0128 model.pt sha256 = state.json, so a failed run makes its test stop in
seconds. Endpoint validation (`qis-validate-endpoints`, CPU) runs after each run ends, in parallel with the test; a
validation failure would void the test row (recorded, not hidden). Test outcomes stay unread until all training ends.

## D119 — 2026-11-09 23:30Z — A14: deadline 19:00Z; U-s202 third-failure rule revised
A14 (project.md §28): report due Tue 10 Nov 2:00 p.m. EST (19:00Z), 3 h later. Plan unchanged: U-s202 ends ≈ 09:00Z,
its test ≈ 13:30Z; margin to 19:00Z ≈ 5.5 h. Revised contingency (replaces D117's "a third failure stops the run",
decided before any further result): on a third U-s202 failure, resume from its latest committed checkpoint only if the
projected test end is ≤ 17:00Z (2 h for final tables, audit and delivery); otherwise stop it and report the U-s202 row,
and therefore seed-202 R − U, as missing (A9.5). The cause must first be checked: a recurrence of the storage fault with
the quota healthy would be a new problem and is reported to the owner. R-s202's contingency (D116) uses the same 17:00Z
test-end limit. Report finalization, audit and delivery move by 3 h: audit ≈ 13:30–16:30Z, delivery by 19:00Z.

## D120 — 2026-11-09 23:46Z — test H-s101 done; diagnostic r64 moved to general (D114 slot plan)
Test H-s101 11095486 COMPLETED 23:42Z (4:27; general). Structure checked, outcome fields unread: test panel, 128 × 32 =
4,096 completions, cap 12,288, test-final, 20.1 M tokens, truncated 10.9%. All four s101-slot tests are done (base, R,
U, H). Diagnostic r64 11099088 (ledger-owned; pending since submission) moved preempt → general/normal at 23:45Z
(allocation of 11095486 ended).

## D121 — 2026-11-10 01:05Z — diagnostic r64 OOM (code defect, fixed); both diagnostics resubmitted
Job 11099088 (general, babel-x5-28) FAILED 00:58Z after 1:09:50 (1.16 GPU-h charged): `torch.OutOfMemoryError` in
`opt.load_state_dict` inside `restore()` (44.38 of 44.42 GiB in use). Cause: the no-update branch's evaluation leaves the
vLLM engine awake, and restoring the 13.8 GB optimizer snapshot to the GPU then exceeds memory. Training always sleeps the
engine before learner work; the diagnostic loop did not. The defect was in the original code too: the diagnostic had
never run on a GPU (CPU unit tests only, D72), so it was not caught. Deterministic code defect, not infrastructure.
Fix (commit 4b47386): `roll.sleep()` before each `restore()`. Results unchanged in definition. The no-update ("none")
row for r64 was written and flushed before the crash; the resume (D114) skips it. Resubmitted r64 11101908 (general) and
replaced pending r0 11099083 (old commit with the same defect; never started; cost 0) by 11101910 (preempt; moves to the
R-s202 slot after its test). r64 now has ≈ 12 branch evaluations left (≈ 9–11 h) → ends ≈ 10:00–12:00Z; a partial run
reports complete seed sets only (A9.5).

## D122 — 2026-11-10 01:50Z — A15 recorded; reconciliation; exploratory analysis rules frozen before any test outcome
A15 (project.md §29) took effect 01:45Z. Deadline: Tue 10 Nov 19:30 EST = Wed 11 Nov 00:30Z. Added analyses are
exploratory. Endpoint results visible at that moment: listed in project.md §29 (no test outcome read; test structural
fields incl. truncated share for base/R/U/H-s101; dev curves per D115).

**Reconciliation (01:47Z).** Slurm ↔ ledger (184 rows): owned live jobs U-s202 11100159 (general, running, round ≈ 85),
test R-s202 11100782 (general, running since 01:31Z), diag r64 11101908 (general, running since 01:01Z), test U-s202
11100785 (general, dependency), H-s202 11082192, diag r0 11101910, validation 11102275 (preempt, pending). Agree.
Retained checkpoints: H-s101 r128 (model + optim); R-s101 r64 (model + optim; diagnostic source) and r128 (model);
U-s101 r128 (model); R-s202 r128 (written 01:30Z; validation pending); U-s202 r64 (latest). **U-s101 round 64 was never
retained** (`retain_rounds: []`): the optional U-checkpoint diagnostic would need a 64-round replay (≈ 7–8 GPU-h, and not
bitwise reproducible, D99) → **skipped** per A15.3. Measured throughput: trained-model test 4,096 completions in 4.1–4.5 h
(≈ 1,250–1,280 tok/s, 4,600–4,900 tokens/answer at cap 12,288); base test 30 min (3,566 tok/s, 1,530 tokens/answer).

**Six runs: H-s202 cannot finish by the A15 deadline.** H-s101 took 26.3 h of active training (29.5 h wall) for 128
rounds; 22.7 h remain to 00:30Z, all 3 general GPUs are busy until ≈ 06:00Z, and the preempt pool has not started any
main job in this study (D37). Even an immediate start would end ≈ 28 h later, after the deadline and before its test.
H-s202 stays queued; its seed-202 H cells are reported as missing (A9.5). Told to the owner.

**Slot plan.** General A: test R-s202 → diag r0 (≈ 3 h) → strategy-cue evaluations if launched. B: U-s202 → its test.
C: diag r64 (≈ 9–11 h). Projected GPU work ends ≈ 14:30Z; report finalization, recomputation and audit 14:30–23:30Z.

**Frozen exploratory rules (before any test outcome is read):**
1. *Per-problem reliability (A15.1).* pass@1 = c/32; pass@8 = 1 − C(32−c, 8)/C(32, 8) per problem, averaged with equal
   weight; paired R−U and R−H per seed for both metrics, problem bootstrap as the primary table. Per-problem change =
   arm pass@1 − base pass@1, plotted sorted per arm and seed. Difficulty bands from base test samples with index 0–15
   (k of 16 correct): B0 k = 0; B1 1–5; B2 6–10; B3 11–15; B4 16. Baseline accuracy within a band is estimated from base
   samples 16–31 only. Same bands for every arm; band sizes reported; zero successes written "0/32".
2. *Interference trace (A15.2).* Shared hard prompts = role `hard_common` (identical prompt at identical round in R and
   U of one seed). Windows of 16 rounds: 1–16, 17–32 | cap change | 33–48, …, 113–128. Per window and arm: hard success;
   share of 8-answer groups all-wrong / mixed / all-correct; all-wrong groups split into "all 8 finished" and "≥ 1
   capped". Mastery vs signal loss: groups that leave "mixed" are classed by destination (all-correct vs all-wrong).
   "Previously solvable": a shared hard problem with ≥ 1 correct answer in an earlier appearance in that run; outcome =
   share of its later appearances that are all-wrong (split finished/capped), R vs U per seed. Dev comparison: per dev
   problem across scheduled evaluations (0/32/64/96/128), the share of problems with c > 0 at one evaluation and 0/8 at
   the next. Training-set results describe repeated questions; dev results describe held-out ones.
3. *Local update (A15.3).* As PROTOCOL §8; ΔJ_H per branch and minibatch seed vs the matching source's no-update row.
4. *Strategy cue (A15.4).* Dropped first if time tightens. Selection rule and IDs frozen in a later decision before any
   trained-policy evaluation.

## D123 — 2026-11-10 02:20Z — strategy-cue panel frozen (A15.4, EXPLORATORY); cost forecast; diag r0 slot
**Panel.** 24 unused official MATH test items (intermediate_algebra 14, algebra 10; L5 19, L4 5), selected by a delegated
read-only agent from local annotations and the pinned MATH parquet (rev 21a5633) — no model output or difficulty number
used. Rule (verbatim in `research/evidence/cue/panel_selection.json`): eligible = MATH test, final label central and not
direct practice, confidence not low, no figure, unused in manifest/panels/calibration/a11 sample and never screened, no
close text variant of any used item (word 3-gram Jaccard ≥ 0.5 or template match), single verifiable answer; order by
sha256("cue-v1:"+id); manual structure check (factoring central AND another decision: construct an equation, choose a
substitution, convert factors to a count, other named); first 24 accepted (rank 34 of 51 eligible). Orchestrator check:
24 unique IDs, none in the manifest or panels by exact ID (one substring hit was `intermediate_algebra/test/00778`, a
different item). Caveats: ≥ 18 items rest on annotation pass 1 only; three items share a construction template
(00474, 00091, 00715) and two the sum-of-cubes idea (00819, 00561); two interval-answer items excluded untested.
Frozen files: `data/cue/cue_v1_uncued.json` (sha256 c3399a42…), `data/cue/cue_v1_cued.json` (1b3aa7bf…; problem +
"\n\nConsider whether a factorization could simplify the problem."). Same pool key and seed label → identical sampling
seeds per (item, sample) across forms (paired). Frozen before any policy was evaluated on it.
**Cost forecast** (measured throughput, D122): base 2 × 192 completions ≈ 5 min; each trained endpoint 2 × 192
completions ≈ 10–15 min plus ≈ 3 min load → total ≈ 2–2.5 GPU-h; recipe `qis-cue-eval` (time 5 h), queued afterany
U-s202 on general so all five policies run in one job (≈ 10:00–12:30Z). Dropped first if the schedule tightens.
**Diag r0** 11101910 moved to general now; it takes the first free general GPU (test R-s202 ends ≈ 06:00Z).

## D124 — 2026-11-10 02:10Z — general GPU ceiling exceeded for 99 s; corrected; slot chain made structural
At 02:04Z diag r0 11101910, moved to general at ≈ 02:00Z (D123), started as a FOURTH owned general job (with
U-s202, test R-s202, diag r64), above the owner's ceiling of 3 (A7). Slurm's normal QOS did not refuse it; my move did
not check the running count. Cancelled at 02:06Z after 1:39 (≈ 0.03 GPU-h, charged); squeue confirmed 3 owned general
jobs. Resubmitted as 11102499 afterany test R-s202 (11100782) on general; cue evaluation 11102440 now afterany 11102499
AND U-s202 11100159. Each general slot is now a dependency chain — A: test R-s202 → diag r0 → cue; B: U-s202 → test
U-s202; C: diag r64 — so at most 3 can run at once.
Rule: before moving a job to general, count owned RUNNING + dependency-free PENDING general jobs; move only if < 3, or
chain it behind a general job with afterany (recorded in the harness log).

## D125 — 2026-11-10 02:25Z — interference trace, interim (EXPLORATORY; U-s202 only to round ≈ 88)
Job 11102391 (suite 49/49; trace over training records only). Evidence `research/evidence/a15/interference_interim.json`.
Interim reading, to be rerun after U-s202 ends: (1) groups leaving "mixed" mostly became all-correct (mastery), not
all-wrong: H-s101 124 vs 14, R-s101 121 vs 22, U-s101 106 vs 29, R-s202 94 vs 25, U-s202 84 vs 21 (partial). (2) Previously
solvable shared hard problems that later gave an all-wrong group: H-s101 24/957 (2.5%), R-s101 40/960 (4.2%; 22 with ≥ 1
capped answer), U-s101 48/966 (5.0%; 34 capped), R-s202 33/963 (3.4%; 10 capped), U-s202 30/675 (4.4%; 9 capped,
partial). (3) Paired R vs U cells (same problem, same round) where only one arm is all-wrong: small counts per 16-round
window (0–15). (4) Dev panel: problems lost (c > 0 → 0/8) between evaluations 0–5 per interval per run. No training
decision depends on this.

## D126 — 2026-11-10 04:00Z — post-training analysis queued; unsealing gate in code
`qis-report-final` 11103165 (commit bea479c) queued afterany U-s202 (11100159). Its precheck refuses unless R/U/H-s101
and R/U-s202 have RUN_COMPLETE, so no test outcome is read before all training has ended (A13). It runs report_data,
interference (final), report_tables (A9/A13 table on the test summaries present), per_problem (A15.1) and all figures
(adds per-problem change and interference-window plots). The U-s202 test (≈ 14:00Z) will be missing in this first pass;
the job is rerun after it ends. Storage at 03:48Z: 292/500 GB (209 GB free).

## D127 — 2026-11-10 04:35Z — R-s202 validation: optimizer-step label offset after resume (bookkeeping defect)
Validation 11102674 (validator 1125a1e): H-s101, R-s101, U-s101 all_pass (R/U-s101 optim.pt listed as retired, D117).
R-s202: every check passes except "optimizer": 524 distinct `opt_step` labels vs 512 expected (rounds 128 distinct,
attempts, evaluations, verifier, checkpoint identity, attempt a2 all pass). Cause: `run.py` passed the live `counters`
dict to the asynchronous `Saver`, which serialises the metadata when the write finishes, so a checkpoint carries the
counters of a few rounds later than its round. R-s202 resumed from round 16 (D105) after the cancelled job had run to
round ≈ 19 → labels offset by 12 (3 rounds × 4 steps). **Not a training defect:** model, optimizer state (AdamW's own
step count, used for bias correction), and RNG state are copied to host at `save()`; the data schedule is indexed by
round; `opt_step` is only a log label. Affected bookkeeping in resumed runs (R-s202, R-s101 retry, U-s202): opt_step
labels, `gen_tokens`, `prompt_tokens`, `hard_responses` counters in metrics.
Fixes (0151edc): validator counts (round, minibatch) pairs and reports label count separately; `report_data` takes
generated tokens from the training records; trainer snapshots `dict(counters)` at save (affects no running or queued
job — all are pinned). Validation rerun queued. Report notes the counter defect.

## D128 — 2026-11-10 04:50Z — A16: partial H-s202 at lowest priority
H-s202 11082192 (ledger-owned; pending since 8 Nov; pinned 7c44fa5, recipe 97c04ad04599 = frozen a2 protocol, ckpt_every 4;
the only training-code change since is the D127 counter copy, bookkeeping only) chained afterany diag r64 (11101908) and
moved to general/normal. It starts on slot C only when the diagnostic ends (≈ 08:30–10:00Z); slots A (test R-s202 →
diag r0 → cue) and B (U-s202 → test U-s202) are unaffected, so it takes no GPU from any A15 item and the 3-GPU ceiling
holds. **Stop rule (fixed now):** cancel H-s202 at 19:00Z (2 PM EST) or earlier if the report needs the slot or storage;
expected progress ≈ rounds 45–60 (H-s101 pace: ≈ 5 min/round to 32 at 8k, ≈ 13–16 min/round after). Scheduled dev
evaluations at rounds 0 and 32 (64 if reached) give the comparison points. Not an endpoint; no test evaluation. Appendix
(exploratory): H-s202 vs R-s202 vs U-s202 dev pass@1 at matching rounds and shared-hard-schedule windows (H's
hard_common role sees the same hard stream position as R/U only if the schedules coincide — checked before comparing).
D127 result (05:10Z): validation 11103416 (validator 0151edc) — H-s101, R-s101, U-s101, R-s202 all_pass (R-s202 512
(round, minibatch) steps; 524 opt_step labels reported separately). Evidence `research/evidence/endpoints/`.

## D129 — 2026-11-10 07:15Z — cue analysis added to the cue job; pending job replaced
The A15.4 analysis needs per-response text (method marker and the fixed inspection sample), and response text exists
only on compute. New module `qis.analysis.cue` (commit 6d4e33a; synthetic test added; suite job 11104737). It reports
uncued and cued pass@1, cue gain per policy, the per-seed difference in differences (R cue gain − U cue gain) with an item
bootstrap, truncation, mean and median tokens, the share of responses that contain "factor", and a fixed sample (items at
panel positions 0, 6, 12, 18; sample 0; every policy and form). The sample positions were fixed before any cue output
exists. Recipe `qis-cue-eval` gained phase `analyse`; generation settings are unchanged. Pending job 11102440 (never
started; 0 GPU-h) was cancelled and replaced by 11104738 with the same dependencies (afterany diag r0 11102499 and
U-s202 11100159), moved to general/normal as in D123. The 3-GPU ceiling holds: it starts only after slot A frees.

## D130 — 2026-11-10 08:10Z — general chains swapped so no GPU idles
At 08:05Z diag r0 (11102499) had one branch left (≈ 10 min per branch; ends ≈ 08:20Z). The cue job 11104738 also waits
for U-s202 (ends ≈ 09:10Z), so its slot would idle ≈ 50 min. Diag r64 (11101908) has seed 4104 left (≈ 47 min per
branch; ends ≈ 10:30–11:00Z). Change: H-s202 11082192 now afterany diag r0; cue 11104738 now afterany diag r64 AND
U-s202. Chains: A diag r0 → H-s202 (partial; stop rule 19:00Z unchanged, D128); B U-s202 → test U-s202; C diag r64 →
cue. Each chain is serial, so the 3-GPU ceiling holds. Cost: the cue panel starts ≈ 1.5 h later (ends ≈ 13:30Z), still
far before the deadline; H-s202 gains ≈ 2 h. Interim diag progress read from job logs (J values only; not analysed).

## D131 — 2026-11-10 08:25Z — diag r0 complete; reference check queued (EXPLORATORY)
Diag r0 11102499 COMPLETED 08:15Z: 13 rows (none + 3 branches × seeds 4101–4104), `research/evidence/a15/diag_r0.jsonl`.
H-s202 11082192 started 08:15Z on that GPU (D130).
**Observation.** Against "none", every update row lowers J_R: 12 of 12 rows, by 0.020–0.040 (means hard −0.026,
related −0.025, unrelated −0.027). ΔJ_H means: hard +0.014, related +0.027, unrelated +0.005. A shift of the same size in
every branch, including the related branch, points to the reference row as a possible cause. "none" is evaluated first
in a fresh process; every update branch is evaluated after a sleep → restore → reload cycle. A second candidate is real:
at r0 the optimizer is fresh, so Adam's first step moves every weight by ≈ lr regardless of gradient size.
**Check.** `qis.diag_local --reference-only` (commit 5d95d03) evaluates the source twice with no update: "none_rerun"
(fresh process, as "none") and "none_cycled" (after the cycle). Recipe `qis-diag-ref` (r0 then r64; ≈ 1.5–2 h), job
11105174 on general, afterany cue 11104738 (chain C, ceiling holds). Reading rule fixed now: if none_cycled − none_rerun
is within ±0.01 on J_R, the cycle does not explain the shift; differences between branches are unaffected in either case
because every branch shares the same reference. The report will state the reference result next to each ΔJ.

## D132 — 2026-11-10 09:40Z — U-s202 complete and validated; report gated on validation
U-s202 11100159 COMPLETED 09:10Z (11:16 elapsed). Test U-s202 11100785 started 09:10Z (chain B). Report-final 11103418
became runnable at that moment, before U-s202's endpoint was validated; I held it, submitted `qis-validate-endpoints`
11105454 (sha 9b9da6d), set the report to afterok 11105454, and released it. Validation COMPLETED 09:35Z: all five
endpoints (H/R/U-s101, R/U-s202) all_pass; evidence `research/evidence/endpoints/`. Report-final now queued (CPU). It
unseals the five completed tests; it reruns after test U-s202 ends (≈ 13:40Z).

## D133 — 2026-11-10 09:45Z — sealed test outcomes read (five of six tests); interim, seed 101 complete only
Unsealing gate met: all five planned complete runs have RUN_COMPLETE and passed endpoint validation (D132); H-s202 is a
partial run without a test (A16). Report-final 11103418 (pinned 1e03098) ran 09:37–09:38Z; tables and figures copied to
`reports/tables/`, `reports/figures/`. First read of sealed outcomes: 09:45Z. Test U-s202 is still running, so seed 202
has R only.
**Primary (seed 101).** pass@1 H 0.770, R 0.819, U 0.777 (base 0.526). R − U +0.042 [+0.021, +0.063]; R − H +0.049
[+0.020, +0.080]; U − H +0.007 [−0.020, +0.035]. Intervals: 128-problem paired bootstrap, conditional on these policies.
R-s202 0.748, which is 0.071 below R-s101: the seed-to-seed spread of one arm exceeds the seed-101 R − U effect, so the
seed-202 R − U pair decides how far the result generalises. No change to any plan or job follows from these numbers.
**Exploratory (A15.1).** pass@8 R − U +0.017 [−0.000, +0.036], R − H +0.024 [+0.005, +0.046] (seed 101). By base band
(held-out baseline): R-s101 exceeds U-s101 mostly in B1 (base 1–5 of 16; 0.755 vs 0.658) and B3 (0.959 vs 0.935); B0
(0/16 at base; 15 problems) is equal (0.306 vs 0.298). Problems improved / worse vs base: H 94/13, R 102/6, U 101/9.
Note: `diag_r64.jsonl` in that report is a partial copy taken while the diagnostic ran; it is replaced at the rerun.
D133 addendum (09:50Z): figure fix fb4f344 — runs without RUN_COMPLETE (H-s202, A16) are excluded from every main
figure and drawn only in `a16_partial.png` (exact 8-round bins the partial run completed). Suite 11105646 51/51; rerun
report-final 11105647 (fb4f344) outputs copied to `reports/`. H vs R hard_common prompt sets identical in 128/128 rounds
(s101) and 15/15 rounds so far (s202): the A16 comparison uses the same shared prompts. Dev at round 128 (non-sealed):
U-s202 0.703 vs R-s202 0.656, the reverse of seed 101 (R 0.729 vs U 0.691); it agrees with R-s202's lower test score.

## D134 — 2026-11-10 10:35Z — diag r64 complete; zero-gradient rows give a momentum-only reference (EXPLORATORY)
Diag r64 11101908 COMPLETED 10:21Z: 13 rows, `research/evidence/a15/diag_r64.jsonl` (source R-s101 round_0064, its
saved AdamW state). Cue 11104738 started 10:22Z on that GPU (chain C).
**Zero-gradient rows.** Related 4101 and 4102 (all 8 answers correct in every group) and unrelated 4103 had no mixed
group, so the advantage and the gradient are 0 (grad norm 0.0). AdamW still steps from its stored moments, so these
rows are a "step without signal". All three give identical J_H, J_R and J_U (0.6113, 0.8951, 0.9316): evaluation after
the cycle is exactly repeatable for identical weights. Against "none" this momentum-only step gives ΔJ_H +0.006,
ΔJ_R −0.007, ΔJ_U −0.008 (part may be the cycle; reference check 11105174 separates it).
**Informative rows (ΔJ_H vs none).** hard +0.001 mean (−0.006, +0.002, +0.012, −0.004); related (4103, 4104) +0.016,
+0.014; unrelated (4101, 4102, 4104) −0.006, +0.027, −0.014. Against the momentum-only row: related +0.010, +0.008;
unrelated −0.012, +0.021, −0.020. No branch moves J_H consistently at r64; the spread across seeds (≈ 0.04) exceeds
every mean. Reported as finite changes after one update, per seed, without pooling the zero-gradient rows as
independent evidence.

## D135 — 2026-11-10 12:25Z — U-s202 test done; reference check started early
- Test U-s202 11100785 COMPLETED 12:21Z (exit 0): pass@1 0.792, pass@8 0.949. Seed 202 reverses seed 101: R-s202 0.748, so R−U ≈ −0.044 at seed 202 (interval from report-final). Reported as observed (A13); no change to plans.
- Report-final 11106560 submitted (938edaa) to build the full lead table, seed means and agreement.
- A general GPU slot freed. Diag-ref 11105174 depended on the cue job only to respect the 3-GPU ceiling. Dependency cleared at 12:24Z; general use stays at 3 (cue, H-s202, diag-ref).

## D136 — 2026-11-10 12:40Z — Primary result: seeds disagree
- Report-final 11106560 (938edaa) COMPLETED; outputs fetched into `reports/tables` and `reports/figures`. All six test summaries present.
- R − U pass@1: seed 101 +0.042 [+0.021, +0.063]; seed 202 −0.044 [−0.072, −0.017] (a9_table; corrected 19:55Z from the per_problem interval); mean −0.001 [−0.018, +0.015]. pass@8 R − U: +0.017 (s101), −0.026 (s202).
- Recomputed with stdlib Python from per-problem counts; all match (`research/audits/2026-11-10-recompute.md`).
- Interpretation written into REPORT §1–2: the seeds disagree in sign, and the seed-to-seed spread in R (0.071) exceeds either effect. The study does not show that practice choice changes held-out success; it cannot exclude an effect below ~4 points. The intervals are conditional on the trained models. No further runs: seed 303 was deferred (A9) and the deadline is 00:30Z.

## D137 — 2026-11-10 12:50Z — Cue panel result (A15.4, exploratory)
- Cue-eval 11104738 COMPLETED 12:36Z (exit 0); `cue_analysis.json`/`.txt` in `research/evidence/a15/`.
- Cue raises "factor" mentions in all five policies (+9 to +15 points); no cue gain excludes zero. DiD (R gain − U gain): s101 +0.021 [−0.063, +0.120]; s202 −0.094 [−0.193, −0.016]. Opposite signs → no evidence the cue helps R more than U. Written to REPORT §5, separate from interference.
- Fixed-sample inspection (40 responses): grader consistent on problems 0 and 12 (correct answer 10 for problem 0 checked by hand). Counts toward the CHECKLIST 7.3 verifier audit.

## D138 — 2026-11-10 14:20Z — Reference check result; REPORT §3 answer (EXPLORATORY)
- r0 (L40S, babel-o5-28): none_rerun = none_cycled exactly (J_H 0.4316, J_R 0.808, J_U 0.793). D131 rule: the cycle does not explain the J_R shift.
- The original r0 "none" (L40, babel-w5-24) was J_H 0.4004, J_R 0.8281. The update rows' mean J_R (0.802) matches the rerun. Reading: the uniform J_R drop came from one high reference draw shared by all Δs, not from the update.
- r64 none_rerun (L40S) J_H 0.6309, J_R 0.8929, J_U 0.9512 vs original (6000 Ada) 0.6055, 0.9018, 0.9395. none_cycled identical (diag-ref 11105174 COMPLETED 14:33Z; `research/evidence/a15/diag_r*_ref.jsonl`).
- Noise floor: two evaluations of the same model differ by 0.025–0.031 in J_H. Answer written to REPORT §3: the unrelated update did not measurably lower J_H at r0 or r64; the design detects only drops ≥ ≈ 0.03.

## D139 — 2026-11-10 16:15Z — H-s202 stop rule: wait for the round-64 dev evaluation, bounded at 20:00Z
- Monitor 11107872: H-s202 at round 52 (checkpoint started 16:07Z); 11–16 min per round at the 12,288 cap. Storage 180 GB free.
- Round 64 has a scheduled dev evaluation, which D128 named as a comparison point "if reached". Forecast: round 64 ≈ 18:55–19:10Z, plus its evaluations.
- Rule change: cancel 11082192 when the round-64 dev evaluation is recorded, or at 20:00Z (3 PM EST), whichever comes first. Earlier if the report needs the slot or storage (unchanged).
- Why it is safe: no other GPU work is waiting (general 1/3); the remaining report work is CPU-side and takes < 1 h; 20:00Z leaves 4.5 h before the 00:30Z deadline. A16 stays lowest priority: it does not delay any other item.
- Not a training decision based on test data: H-s202 has no test evaluation.

## D140 — 2026-11-10 19:20Z — H-s202 stopped after the round-64 evaluations
- Round-64 evaluations recorded: dev 19:10Z (pass@1 0.658), probes J_H, J_R, J_U by 19:16Z. Waited 6 min past the dev line for the probe set (inside the D139 bound).
- 11082192 checked against the ledger, cancelled by ID at 19:16Z; sacct CANCELLED, allocation ended. Last checkpoint round 60 retained; logs and metrics kept. Rounds 65+ not run.
- Next: report-final (interference + tables + A16 figure) and the final cost table.
- 19:18Z: the CPU queue had 148 pending jobs and monitor 11108988 had waited > 1 h. Cancelled 11108988 by ID (superseded: H-s202 stopped). Moved report-final 11109597 to general/normal with 1 GPU (GPU QOS needs ≥ 1 GPU; general use 0/3 after the stop). It started at once. Same treatment planned for the cost job. The GPU-hours are charged in the cost table.

## D141 — 2026-11-11 01:55Z — A17 accepted; H-s202 resumed; common-GPU recipe frozen before any rerun score
- **Amendment A17** recorded (`project.md` §31). Deadline Thu 12 Nov 00:30Z (Wed 7:30 PM EST). Supersedes A15's deadline and A16's "no H-s202 test".
- **H-s202 checkpoint.** `LATEST.json` → round 60 (`round_0060`, saved 10 Nov 17:44Z). `round_0064.tmp` is an incomplete save from the run cancelled at 19:16Z (D140); the saver deletes it at the next round-64 save. The stale write lock of job 11082192 is > 1 h old, so the saver breaks it (`LOCK_STALE_S` 3600). Storage 343/500 GB used, 158 GB free (01:37Z). The trainer's loader re-checks file SHA-256, tensor checksums, optimizer state and RNG state; the prompt schedule is a pure function of seed and round.
- **Resume.** First submission 11112338 cancelled by ID at 01:50Z during its hash check, to fix two recipe faults before training: the audit's done-predicate could pass on a partial file, and no intermediate checkpoint was retained. Resubmitted as 11112377 (general, `retain_rounds` [64], new phase `resume_audit` that hashes model.pt and optim.pt against `state.json` and copies the original `metrics.jsonl` to `metrics.pre_resume_a17.jsonl`).
- **Trajectory naming.** Original a2 trajectory to round 64 (dev 0.658 at round 64) = **H-s202 a2-orig**, kept for the A16 appendix. Resumed trajectory from round 60 = **H-s202 a2-r60**; only it can reach round 128 and supply the endpoint. Rounds 61–64 are re-run; analyses select rows after the round-60 `resume` event.
- **Original test hardware (sacct + sinfo, read 01:52Z).** R-s101 11095106 babel-w5-24 **L40**; U-s101 11095118 babel-q5-20 **L40S**; H-s101 11095486 babel-q5-20 **L40S**; R-s202 11100782 babel-w5-28 **L40**; U-s202 11100785 babel-o9-24 **L40S**; base 11089196 babel-q5-20 **L40S**. Neither R/U pair shared a GPU type; in both seeds R ran on L40 and U on L40S.
- **Frozen common recipe (before any rerun score is seen).** GPU type L40S (Slurm constraint + in-job `nvidia-smi` name check). Code and environment unchanged (`git diff` of evaluate_ckpt, verify, prompts, seeds, rollout since the original commits: empty; venv-v1). Same sealed panel (sha 2ffe70dd…), n = 32, cap 12,288 with the 8,192 view, frozen verifier, and the **original seed label `test-final`**. Same seeds isolate the hardware effect: a U rerun on L40S is a direct reproducibility test (prediction: identical attempts); an R rerun on L40S changes only the GPU type. Outputs go to `test-eval-cgpu/`; the originals are kept and published beside them.
- **Coverage.** Four reruns: R-s101, R-s202 (general), U-s101, U-s202 (preempt; moved to general when a slot frees). H-s101 and base already ran on L40S with this recipe, so their originals are the common cells (no duplicate run). The H-s202 endpoint test runs on L40S under this recipe, so one job fills both roles.
- **Primary analysis unchanged.** The original sealed-test table stays primary; the common-GPU table is a reproducibility check.

## D142 — 2026-11-11 ~02:25Z — A17 §3 first results: the seed-202 reversal is a truncation difference (EXPLORATORY)
Source: `research/evidence/a17/analysis/test/` (`qis.analysis.a17_test` at 353e489, original sealed-test records; run inside
allocation 11112395 with `--gres=gpu:0`). The original endpoint table reproduces `a9_table.json` exactly (values and intervals).
- **Truncation decomposition** (exact identity R − U = −Δt·ā + (1 − t̄)·Δa; 95% problem bootstrap). Seed 202: R 0.748 vs U 0.792,
  R − U −0.044 = truncation term −0.043 [−0.061, −0.027] + accuracy-on-finished term −0.001 [−0.024, +0.020]; R truncated
  538/4096 (13.1%) vs U 333/4096 (8.1%); accuracy on finished answers R 0.861 vs U 0.862. Seed 101: R 0.819 vs U 0.777,
  R − U +0.042 = truncation +0.014 [+0.000, +0.028] + accuracy +0.028 [+0.009, +0.047]; R truncated 10.7% vs U 12.2%.
  Per problem, R − U success vs R − U truncation share: Pearson −0.51 (s101), −0.59 (s202).
- **Within 8,192 tokens** (same traces): R − U +0.051 [+0.031, +0.073] (s101), −0.070 [−0.100, −0.042] (s202): the reversal persists at the
  shorter budget, so it is not created by the 12,288 cap; it follows which arm writes longer answers.
- **Reversal location:** band B1 (34/128) carries −0.057 of the −0.086 reversal; top 5 B1 items carry 45% of that; per-problem R − U is
  uncorrelated across seeds (Spearman −0.09 overall, −0.04 in B1). Coverage vs reliability: R − U is reliability on problems both
  reach (s202 −0.042 of −0.044); training gains over base are mostly reliability on base-reachable problems (+0.20 to +0.25).
- **Sampling null:** at an unchanged checkpoint, two disjoint groups of 8 attempts show 1.7–5.7 of 128 problems "lost" by chance alone
  (hypergeometric expectation). Dev evaluations used n = 8 (all 28 dev rows), so the report's "≥ 1 correct of 16" is wrong → fix.
- **Interpretation (competing explanation now leading):** the R/U difference behaves like a seed-dependent difference in response length
  (truncation) rather than a difference in solving skill. Next distinguishing task: training-record length and truncation by arm and
  seed (`a17_train` signal/persistence) — does R-s202's length grow earlier/faster on the shared hard prompts, and on aux prompts?

## D143 — 2026-11-11 ~02:40Z — Stress panels frozen: SuperHard-128 and Harder-98 (A17 §5; EXPLORATORY); H-s202 records backed up
**Panels (frozen before any model is evaluated on them).** Rubric `research/evidence/a17/panels/RUBRIC-stress-v1.md` (committed before
candidate export). Pool `data/external/a17/candidates_v1.jsonl`: 2,039 candidates from MATH (hendrycks_math@21a5633, L4–5, 1,104),
Omni-MATH (@40ba231, 769), OlympiadBench OE_TO_maths_en_COMP (@91184b5, 166); IDs already used or screened anywhere were removed
before export. All 2,039 annotated in 16 hash-ordered batches by delegated agents (`annotator: agent:claude-sonnet`), one pass each.
Selection `qis.data.stress_panels` at 4b8b88e, run inside allocation 11112396 (`--gres=gpu:0`). Excluded: not eligible 1,731; low
confidence 19; gold not confirmed 3; units/text in gold 4; exact text of a study item 41 (screened candidate lists on compute);
close variant (3-gram Jaccard ≥ 0.5) 11. Used-item scan: 8,521 texts in 66 files under every `$PROJ_DATA/data/*` directory except
`splits-v1` and `raw-21a5633` (the full MATH split manifest and raw download; not study sets — a first run that included them
excluded every MATH item and was discarded), plus repo `data/`.
- **SuperHard-128**: 128 of 130 eligible; MATH 60, Omni-MATH 54, OlympiadBench 14; obstacles case_analysis 44, nonobvious_grouping 29,
  hidden_substitution 28, approach_choice 14, multi_factorization 13; 127 families.
- **Harder-98**: 98 of 100 eligible (2 closed by the one-panel-per-family rule); MATH 70, Omni-MATH 17, OlympiadBench 11; 97 families.
  **Fewer than 128 qualified** because the whole candidate pool was annotated and only 100 Harder items met the rubric; the MATH cap of
  64 was exceeded (70) only after the other sources were exhausted (rubric §4/§5). Not extended: a second candidate tranche would
  cost ≥ 1.5 h of annotation and delay every stress evaluation on a 22-hour deadline (A17 §8 priority: balanced coverage first).
- Cross-panel: 0 duplicate IDs, 0 shared families, max masked 3-gram Jaccard 0.431. Verifier: every gold passes `gold_eligible`.
- SHA-256: `superhard_v1.json` 928cc06f…, `harder_v1.json` d6bd4dc0…, `stress_v1.json` (evaluation file, pool "stress", 226 items)
  586edc31…; identical on compute (`$PROJ_DATA/data/stress/`) and in the repo (`data/stress/`).
- Caveats to report: one annotation pass per item (no second blind pass before freeze); annotator strictness varied by batch (8–20%
  qualifying); absence from study sets does not show absence from pretraining data.
**Evaluation recipe (fixed now):** `qis-stress-<policy>`: L40S, n = 8, seed label `stress-v1`, cap 12,288, frozen verifier, uncued;
policies base, six round-128 endpoints, R-s101 round 64, H-s202 a2-r60 round 64. One generation pass over all 226 items per policy.
**H-s202 records (02:02Z).** The resumed run would overwrite `records/train_r0060–0063` and skip its own round-64 evaluation because the
original eval files existed (run.py:218–220; found by the a17_train author). Before round 61 finished, the originals were copied
(train) and moved (eval round 64) to `runs/qis-main-H-s202-sym/records_a2orig_r60-64/` with SHA256SUMS; the original metric rows stay
in `metrics.jsonl` and `metrics.pre_resume_a17.jsonl`. The resumed trajectory therefore gets its own round-64 dev/probe evaluation.

## D144 — 2026-11-11 02:12Z — Stress-panel response-audit sample fixed before any stress output exists (A17 §6)
`research/evidence/a17/panels/audit_sample_v1.json`: 8 SuperHard + 8 Harder items, first by sha256("audit-v1:"+id); attempt
index 0 for every policy (base, H-s101, R-s101, U-s101, R-s202, U-s202, H-s202 r128 if evaluated) → 96–112 responses. Responses
are shown to annotating agents arm-blind (opaque codes; mapping kept separately) with the prompt, worked solution, gold and verifier
outcome. Categories: missed_opportunity, factored_incorrectly, factored_then_failed_later, other_method, hit_cap, grader_issue,
correct, unclear. The question it answers: would more direct factoring practice plausibly fix the observed failures?
Also started (02:10Z): blind second read of all 226 frozen panel items by 4 new agents (problem + solution + gold only) to measure
annotation agreement; report-only, panels unchanged.

## D145 — 2026-11-11 ~02:45Z — Stress panels re-frozen before any evaluation: SuperHard-126, Harder-95 (supersedes the D143 freeze)
The blind second read (4 agents, 219 of the D143 items) found a cross-source duplicate in Harder (`omni:omni/01670` =
`math:intermediate_algebra/train/00357`; the masked 3-gram check missed it because `x^{4}` and `x^4` tokenised differently) and a
prose gold (`omni:omni/00423`). No stress evaluation had started: the four submitted jobs (11112846/8/9/50) were held, then
cancelled by ID; none ran. Selection rules fixed in `qis.data.stress_panels` (1ecd8f5): braces/`$`/LaTeX spacing ignored in text
normalisation and n-grams; prose golds excluded (rubric 1.1); two items of one family with Jaccard ≥ 0.8 or the same normalised gold
count as one problem. The brace fix also caught 4 more close variants of study items (11 → 15). Annotations unchanged.
- **SuperHard-126** (pool exhausted at 126 eligible): MATH 60, Omni-MATH 52, OlympiadBench 14; obstacles case_analysis 45,
  hidden_substitution 29, nonobvious_grouping 26, approach_choice 14, multi_factorization 12; 126 families.
- **Harder-95** (pool exhausted): MATH 69, Omni-MATH 15, OlympiadBench 11; 95 families. Cross-panel: 0 shared IDs/families, max
  Jaccard 0.411. Every family now has one item.
- SHA-256: `stress_v1.json` 757393be…, `superhard_v1.json` 0cf1cf93…, `harder_v1.json` 0fb34905…; the D143 files are kept as
  `data/stress/superseded-d143/` and `$PROJ_DATA/data/stress.superseded-d143/` (never evaluated).
- **Second-read agreement** (first read vs blind second read, 219 of 221 final items read twice): eligible 207/219 (94.5%); gold
  agrees 219/219; tier SuperHard→SuperHard 105/124, Harder→Harder 66/95 (overall exact tier 171/219 = 78%); obstacle code agrees
  81/105. Second read calls 12 items ineligible (mostly "short route avoids factoring"). Report these; panels stay as frozen.
- Audit sample (D144 rule) recomputed on the final panels: identical 16 IDs.

## D146 — 2026-11-11 02:20Z — GPU plan for A17: three general GPUs only; U-s101 rerun and intermediate stress cells lowest priority
**Measurements.** Preempt: 1,017 jobs pending; this account's preempt starts in 72 h: 5, median wait 41 h (`ae-capacity`, 02:18Z) — treat
preempt as unavailable. General: median wait 0.01 h; project ceiling 3 (A7), all 3 in use (H-s202, R-s101 rerun, R-s202 rerun).
H-s202 a2-r60: round 61 800 s (incl. warm-up), round 62 630 s at cap 12,288 → planning 11–13 min/round for 66 rounds + evaluations at
64/96/128 → round 128 ≈ 14:30–17:00Z; sealed test ≈ 4.4 h on L40S → ≈ 19:00–21:30Z.
**Order on general** (each freed slot takes the next item): (1) U-s202 common-GPU rerun (the direct reproducibility check; its original
already ran on L40S with the same seeds, prediction: identical attempts); (2) stress base (throughput benchmark) and stress R/U endpoints
s101/s202 and H-s101; (3) H-s202 sealed test at RUN_COMPLETE (slot A); (4) stress H-s202 r128; (5) stress R-s101 r64 and H-s202 r64
(optional intermediates); (6) U-s101 common-GPU rerun (optional: its original is already an L40S cell under the D141 recipe, so the
common-GPU table is complete without it; it adds a second reproducibility measurement).
**Fallbacks** (decided now, applied if the ETA slips past 21:30Z for the H-s202 test): split the H-s202 sealed test into two half-panel
jobs on two GPUs (per-item seeds do not depend on batch composition: `eval/<key>/<id>/<label>/n<n>`); drop optional cells in reverse
order (6, 5); stress attempts stay at n = 8 unless the measured throughput says the balanced set cannot finish by 19:00Z, then n = 4
for every cell.

## D147 — 2026-11-11 03:00Z — Training-time length does not predict the test truncation gap (EXPLORATORY)
Mean generated tokens per response on the shared hard prompts (`signal.json`, records; rounds 97–128): R-s101 5,837 vs U-s101 6,282;
R-s202 5,285 vs U-s202 5,414; H-s101 4,857. R wrote shorter shared-hard answers than U in both seeds and every window after round 32,
yet R-s202 truncated more than U-s202 on the sealed test (13.1% vs 8.1%). Practice-pool answers: R longer than U in both seeds
(factoring answers are longer). So the seed-202 test truncation gap is not a general length trend visible in training; it is specific
to the final policies on the test problems. Next distinguishing check: are the truncated test answers degenerate loops or long
non-repeating reasoning (a17_test section 7, requested 02:58Z)? Loops would point to a policy-specific failure mode (instability
of the endpoint checkpoint) rather than a skill difference.

## D148 — 2026-11-11 ~03:10Z — Truncated test answers are long unconverged searches, not loops (EXPLORATORY)
`a17_test` section 7 at 434b439 (`research/evidence/a17/analysis/test/loops.md`): an attempt "loops" if ≥ 50% of the 32-token windows
in its last 2,048 tokens already occurred earlier in the attempt. Truncated attempts that loop: R-s101 0/437, U-s101 1/501, R-s202
0/538, U-s202 0/333, H-s101 0/445; median repeated share 0.007–0.047. So the R − U truncation term is entirely non-looping: s202
R 538/4096 vs U 333/4096 truncated-not-looping, R − U +5.0 points [+3.1, +7.1] of attempts. Two decoded R-s202 truncated answers
(sample 0): `intermediate_algebra/test/00220` (a telescoping product that factors via n⁴ + ¼) spends 12,288 tokens trying numeric
patterns with 35 "wait"s and never factors; `intermediate_algebra/test/00482` cycles through Vieta substitutions without converging.
Reading: the truncation difference is a difference in search behaviour on hard test problems (long unconverged exploration), which
is plausibly where missed strategy recognition shows up; exact-window matching can miss paraphrased cycling.

## D149 — 2026-11-11 04:10Z — H-s202 sealed test runs as six shards; the D146 ETA was too early
- **ETA correction.** D146 and Review 91 assumed 12 min per round. H-s101 rounds grew from ~540 s (round 66) to ~1,000 s (rounds
  114–128), and the round-96 evaluation added ~45 min. H-s101 needed 54,499 s (15.1 h) from round 66 to round 128. H-s202 reached
  round 66 at 03:49Z; its rounds 64→66 took 4,151 s against 3,586 s for H-s101 (×1.16). Projection: round 128 at 19:00–21:20Z;
  RUN_COMPLETE (round-128 evaluations + checkpoint) ≈ 20:00–22:20Z.
- **Problem.** One L40S test job takes 4.1–4.5 h (original tests: 1,205–1,448 tokens/s). It would end 00:10–02:50Z, at or after the deadline.
- **Decision.** Split the sealed test into 6 interleaved shards (`qis.eval_merge --split 6`: items i mod 6), recipes
  `qis-test-H-s202-sym-k0..k5` (k0–k2 general, k3–k5 preempt; all L40S, D141 recipe, seed label `test-final`). Per-problem seeds
  are unchanged because the seed key holds only panel key, problem id, label and n. Shards differ from one job only in vLLM batch
  composition, which D51 already shows is not bitwise reproducible across jobs. The last shard to finish merges all six into
  `test-eval/qis-test-H-s202-sym` under a file lock. The unsharded recipe `qis-test-H-s202-sym` is not submitted.
- **Validation.** `qis.eval_merge` (22aa5e2) re-merged the original U-s202 test split into 6 shards: every summary field and the
  per-problem list equal the original; the merged records file is byte-identical.
- **Submission.** At RUN_COMPLETE (watcher event), not now: both slot budgets are full and a pending job counts as a slot
  (`ops/qis-submit`). General slots must be free of stress work by ≈ 19:30Z.
- **Cut-off.** The report is delivered by 00:30Z regardless. H-s202 test cells are filled if the merge exists by ≈ 23:45Z;
  otherwise they are listed as incomplete with the H-s202 round-128 dev and probe values.

## D150 — 2026-11-11 09:00Z — H-s202 endpoint test likely misses the deadline; fallback test of the latest checkpoint
- **Evidence.** H-s202 rounds 84–86 took 3,765 s (H-s101: 1,953 s). Cause: longer responses, not node contention. H-s202 mean
  response length in rounds 83–86 was 5,992–6,847 tokens against 3,466–4,531 for H-s101 (`len_mean` in metrics.jsonl); node load
  4.9 on 8 CPUs. Rounds 66→86: 17,575 s against 14,505 s (×1.21); rounds 83–86 alone ×1.4–1.9.
- **Projection.** H-s101 needed 39,994 s for rounds 86→128. At ×1.21: round 128 ≈ 22:10Z, RUN_COMPLETE ≈ 23:10Z. At ×1.4:
  RUN_COMPLETE ≈ 01:15Z. The six-shard test needs ≈ 1.7 h more. Only a return to ×1.0 (RUN_COMPLETE ≈ 21:00Z) meets 00:30Z.
- **Decision.** (1) Training continues to round 128 (A17); the endpoint test runs as six shards at RUN_COMPLETE (D149) even if
  after the deadline, and results go into a dated addendum. (2) Fallback: at ≈ 18:00Z, if projected RUN_COMPLETE is later than
  22:15Z, copy the latest saved checkpoint (`ckpt_every` 4; expected round ≈ 108–112) to `ckpt/qis-main-H-s202-sym-fallback/`,
  verify its model.pt hash against state.json, and run the six-shard sealed test on it (general GPUs, D141 recipe). (3) The report
  shows that value only as "H-s202 at round N (incomplete run)", outside the seed-mean and primary tables, because a shorter run
  is not comparable with 128-round R and U endpoints. (4) The owner is told now; a deadline extension of ≈ 3 h would let the
  endpoint test land.

## D151 — 2026-11-11 09:15Z — Common-GPU check: L40S reproduces bit for bit; hardware does not explain either R − U sign
- **Reproducibility.** The U-s202 rerun on L40S (11112398) reproduces the original U-s202 evaluation (also L40S) exactly:
  4,096/4,096 token sequences, rewards and request seeds identical. Same GPU type + same seeds = identical attempts.
- **Hardware effect on R.** R reruns on L40S share the request seeds with the original L40 runs but only 48/4,096 (s101) and
  133/4,096 (s202) token sequences are identical. pass@1 L40S − L40: R-s101 0.823 − 0.819 = +0.004 [−0.008, +0.016];
  R-s202 0.761 − 0.748 = +0.013 [+0.002, +0.025].
- **Contrast on matched hardware.** Seed 202: R − U = 0.761 − 0.792 = −0.031 [−0.060, −0.004] (original −0.044 [−0.072, −0.017]).
  Seed 101: R-s101 L40S 0.823 against U-s101 0.777 (original, L40S; the U-s101 rerun is still running, and by the U-s202 result it
  is expected to be identical). The seed reversal survives on a common GPU type; hardware moved R by at most 1.3 points.
- Outputs: `research/evidence/a17/analysis/test-cgpu/` (a17_test at 434b439, run in worktree d3fa9d5 on CPU standby 11112647).
- **Correction (11:55Z, Review 100).** "L40S reproduces bit for bit" does not hold in general. The U-s101 rerun on L40S
  (11112397) matched the original (L40S) on 1,964/4,096 token sequences only; pass@1 0.778 vs 0.777. The cause is unknown
  (different node). The conclusion stands: s101 R − U on L40S = 0.823 − 0.778 = +0.044 [+0.024, +0.065].

## D152 — 2026-11-11 09:25Z — Stress panels repeat the seed pattern; seed 202's deficit there is all truncation (EXPLORATORY)
- Cells: base, R/U × s101/s202 at round 128 (n = 8, 221 problems; SuperHard 126, Harder 95). All 1,768 attempts per cell, 0 unknown.
- **Seed 101:** R − U = 0.695 − 0.653 = +0.042 [+0.015, +0.068]; truncation term +0.021 [+0.004, +0.039], accuracy term +0.020
  [−0.001, +0.041]; pass@8 equal (0.833).
- **Seed 202:** R − U = 0.632 − 0.658 = −0.027 [−0.051, −0.002]; truncation term −0.063 [−0.082, −0.043] (R truncates 23.0% of
  attempts, U 15.2%); accuracy term **+0.036 [+0.012, +0.061]** (accuracy on finished attempts R 0.821, U 0.776). pass@8
  0.814 − 0.851 = −0.036 [−0.063, −0.014].
- **Reading.** On harder, held-out problems the sign of R − U again follows the seed, and in seed 202 R is *more* accurate on the
  answers it finishes; it loses only because more of its attempts hit the 12,288 cap. This supports D142: the seed difference is a
  length/termination difference, not a difference in factoring skill. U-s202 is the shortest-answering trained policy (mean 5,239
  tokens against 5,929–6,058).
- Outputs: `research/evidence/a17/analysis/stress/` (a17_stress d3fa9d5); figure `reports/figures/a17/stress_panels.png`.

## D153 — 2026-11-11 11:05Z — Response audit run (D144) on six policies; two blind reads per response
- **Sample.** The D144 sample unchanged: 16 items (8 SuperHard, 8 Harder), attempt 0, for base, H-s101, R-s101, U-s101, R-s202
  and U-s202 at round 128 → 96 responses. H-s202 r128 is not evaluated yet; it is an incomplete cell of the audit.
- **Blinding.** `research/evidence/a17/audit/build.py` (run on the CPU standby) decodes each response, gives it a random
  4-letter code, and shuffles the six responses within each item. The code → policy key stayed on compute
  (`~/ae-qis/audit/key.json`) until all annotations were written. Packets carry the problem, gold, the panel annotation's
  reference route, key factoring step and obstacle, and the verifier result. Responses over 8,000 characters show the first
  5,000 and last 2,500 characters.
- **Reads.** Four annotating agents (Claude Sonnet). The 16 items form 4 groups of 4; reader a reads groups a and a+1 (mod
  4), so every response is read twice by different agents. Agreement is reported per category.
- **Use.** Report-only (section 5). It answers "would more direct factoring practice plausibly fix these failures?" It
  changes no panel, score or decision.

## D154 — 2026-11-11 16:10Z — A19 recorded; length diagnostic by exact continuation; H-s202 gate kept (EXPLORATORY design, fixed before any output)
- **A19** (owner, ~16:03Z; `project.md` §33) recorded verbatim. Deadline unchanged: Thu 12 Nov 00:30Z (Wed 7:30 PM EST).
- **Reconciliation (16:05Z).** H-s202 11112377 running (general); standby 11112647 pending. Retained weights: H-s101 r128;
  H-s202 r64, r108 (moving); R-s101 r64, r128; R-s202 r128; U-s101 r128; U-s202 r128 (R-s202 `round_0004.tmp`,
  `round_0012.tmp` are partial saves, not used). Storage 359/500 GB.
- **H-s202 critical path (A19 §1).** `qis.train.run` order at round 128: round record → dev evaluation (`eval_rounds`,
  ≈ 60–75 min) → blocking save (`.tmp`, hash, fsync, re-read hash, rename, LATEST.json) → `saver.wait` → `complete` event →
  `RUN_COMPLETE`. The round-128 weights reach disk only after the dev evaluation; `RUN_COMPLETE` follows the verified save
  within seconds and adds the `saver.last.ok` guarantee. Releasing on the checkpoint before the marker would save seconds,
  so the existing gate stays. To remove reaction delay, the endpoint shards will be submitted with `--after` the training job as soon as slots allow
  (pending jobs count as slots in `ops/qis-submit`; expected after the fallback test ends).
- **No matched intermediate R/U pair (A19 §3).** Only R-s101 and H-s202 keep round 64. Within-seed divergence will be located
  from the fixed dev evaluations and training logs; no retraining.
- **Length diagnostic (A19 §2): exact continuation.** Supported: `Rollout.generate` takes token ids; the engine context is
  PROMPT_CAP + 16,384. Each capped stress-v1 attempt (finish `length`, exactly 12,288 tokens) is continued from prompt ids +
  its own 12,288 response ids with 4,096 more tokens (T 1.0, top_p 1.0, seed `extend/stress/<id>/extend-v1/s<j>`, the same
  for all checkpoints). Finished attempts are not regenerated. Score at 12,288 = the stored reward (every selected attempt is
  re-graded and must match); score at 16,384 = same verifier on the full trajectory (still capped → 0). Code
  `qis.eval_extend`; boundary tests `tests/test_eval_extend.py` run in each job's precheck.
- **Sample (fixed now).** Tranche 1: 64 problems, 32 per tier, sources by largest remainder (Harder MATH 23 / Omni 5 / Olympiad
  4; SuperHard 15 / 13 / 4), items in sha256(`a19-length-v1/<id>`) order, samples 0–3 → 256 attempts per checkpoint.
  Tranche 2 ("rest") = all other attempts of the 221 × 8 panel, run only after tranche 1 in the same job.
- **Checkpoints and hardware.** R/U × s101/s202 round 128; L40S only. Canary R-s202 first; the other three after its
  tranche-1 summary is checked.
- **Inspection sample (preselected).** Per checkpoint, the 3 tranche-1 attempts capped at 12,288 with the smallest
  sha256(`a19-inspect/<id>/<sample>`); read after extension and classified: completed sound reasoning / repaired an error /
  continued a loop / delayed an incorrect answer.
- **Scope of the claim.** Inference allowance for these checkpoints only; not what training with a longer cap would give.

## D155 — 2026-11-11 17:35Z — A20: use the extra general GPUs for the length diagnostic on the sealed test (EXPLORATORY; fixed before output)
- **A20** (owner, ~17:15Z; `project.md` §34): 8 general GPUs until 21:00Z, then 3. `ops/qis-submit` enforces both.
- **Use of the extra slots, in A19 keep-first order.** (1) H-s202 training keeps its GPU. (2) The four stress full-panel
  length jobs move to general (all running by 17:30Z). (3) New: the same exact-continuation diagnostic (D154) on the
  **original sealed-test records** of R/U × s101/s202 (128 × 32 attempts each; `--select all`; seed label
  `extend-test-v1`; L40S; outputs `$P/length-test/<label>/all`). Reason: A19 §2 asks about the sealed-test gap itself
  (R-s202 74.8% vs U-s202 79.2%); the stress panels only repeat it. The original scores stay unchanged; this is
  exploratory. Estimated 1–1.5 h per checkpoint (≈ 330–540 capped attempts each), done well before 21:00Z.
  (4) The D150 fallback at 18:00Z: shards go to whichever pool has free slots; all six fit before 21:00Z.
- **After 21:00Z** general returns to 3: H-s202 training plus the endpoint shards at RUN_COMPLETE.
- Analysis: `qis.analysis.a19_length --panel-key test --sets all=all`, R − U per seed at 12,288 and 16,384 with the paired
  problem bootstrap (128 problems).

## D156 — 2026-11-11 18:58Z — H-s202 stress cell as three shards; endpoint shards pre-submitted after the fallback
- **Why.** RUN_COMPLETE ≈ 21:10–22:10Z; endpoint ≈ 22:10–23:10Z. A single stress job (1,768 attempts, ≈ 2 h) after the
  endpoint would end ≈ 00:10–01:10Z, after the deadline. Three shards (`qis-stress-H-s202-r128-k0..k2`, panel split by
  index mod 3 with `qis.eval_merge`, as D149 did for the test) take ≈ 45 min. Per-problem seeds are keyed by item, so the
  shards reproduce the single-job attempt schedule (D149 validated the merge byte-identically). The single-job recipe
  `qis-stress-H-s202-r128` is superseded and will not be submitted.
- **Gate (A19 §1).** Each shard checks `RUN_COMPLETE`, the model.pt hash against state.json, provenance (round 128,
  opt_step 512, attempt a2) and the stress-panel hash. The endpoint shards carry the same checks.
- **Order.** Endpoint shards first (A19: sealed test, then stress). After the fallback shards end (≈ 19:45Z) and before
  21:00Z: submit endpoint k0, k1 (general) and k2–k5 (preempt) `--after qis-main-H-s202-sym`; general then holds 3
  (training + k0 + k1), within the post-21:00Z ceiling. Stress shards are submitted `--after` the endpoint shards and moved
  to general as endpoint shards end.

## D157 — 2026-11-11 19:12Z — A21: 8 general GPUs until the final report; all six endpoint shards on general; H-s101 length cell
- **A21** (owner, ~19:08Z; `project.md` §35) extends A20 until the final report. `ops/qis-submit` allows 8 general until
  00:30Z.
- **Idle GPUs at 19:05Z.** The owner asked whether all general GPUs were in use: they were not (5 of 8 running; the
  endpoint shards k0, k1 held 2 guard slots while waiting on the training dependency). Fixes:
  (1) New: the length diagnostic for H-s101 r128 (`qis-lentest-H-s101` 11119262, sealed test; `qis-len-H-s101` 11119264,
  stress t1 + rest; same D154/D155 method). Reason: it tests whether R-s101's +4.9-point lead over H survives more room;
  it is not a rerun.
  (2) All six endpoint shards moved to general (`ae slurm update`; k0, k1 had been moved to preempt by a command the owner
  interrupted). They can start only when training ends (≥ 21:08Z); at that time general holds at most the two H-s101 jobs
  + 6 shards = 8 GPUs. With six shards in parallel the endpoint takes ≈ 45–50 min after RUN_COMPLETE.
- **Stress H-s202 r128** (D156): 3 shards submitted when the endpoint shards start; general GPUs free up as the H-s101
  jobs end.
- **A22 (~19:11Z): general ≤ 7.** Endpoint k5 (11119217) back to preempt; peak general = 7 (before RUN_COMPLETE: training +
  3 fallback shards + R-s101 continuation + 2 H-s101 jobs; at RUN_COMPLETE: 2 H-s101 jobs + endpoint k0–k4). k5 moves to
  general only when a general job has ended. Stress shards (D156) are submitted only into free slots under 7.

## D158 — 2026-11-11 21:05Z — A23: 8 general GPUs until the report ends; stress shards share the endpoint start
- **A23** (owner, ~21:00Z; `project.md` §37) restores 8 general GPUs until delivery. `ops/qis-submit` set to 8.
- **Placement.** At RUN_COMPLETE the six endpoint shards (k0–k5, general) and stress shards k0, k1 (moved to general)
  start together: 8 GPUs. Stress k2 (11119927) waits on preempt and moves to general when the first endpoint shard ends.
  Endpoint ≈ 45–50 min after RUN_COMPLETE; stress ≈ 45 min for k0, k1 and ≈ 45 min after the first endpoint shard for k2.
  The sealed test keeps priority (A19 §1): the stress shards use only GPUs the endpoint does not need.


## D159 — 2026-11-11 21:55Z — H-s202 endpoint shards k3–k5 and stress shards on any 48 GB GPU (deadline; L40S queue)
- **Evidence.** RUN_COMPLETE 21:45:07Z (round 128, opt_step 512, a2; model.pt `2513456224d4…`). All eight general shards
  pending. Slurm start estimates (squeue --start, 21:50Z): k0, k1 21:49Z; k2 22:04Z; k3 23:40Z; k4 00:16Z; k5 00:26Z;
  stress k0, k1 00:50Z, 01:00Z. 95 jobs wait for L40S on general. With L40S only, the endpoint lands ≈ 01:10Z, after
  the 00:30Z deadline.
- **Decision.** Keep k0–k2 on L40S (they start now). Cancel and resubmit k3–k5 and stress k0–k2 with `constraint=VRAM_48GB`
  (L40S, L40, A6000 or 6000 Ada); the precheck records the GPU name instead of refusing a non-L40S GPU. Everything else
  is unchanged (seeds per item, n, cap, verifier, merge).
- **Consequence for the report.** The H-s202 endpoint is no longer a pure single-GPU-type cell: each shard's GPU is reported.
  The common-GPU check (section 2) measured the effect of a GPU type change at ≤ 1.3 points of pass@1 per checkpoint; the
  same caution applies to H-s202's R − H and U − H. A17 asked for both H endpoints on the common recipe "as feasible"; it
  is not feasible by the deadline.
- **21:57Z extension.** Endpoint k2 (11119213, L40S) was still pending while k0, k1, k3–k5 ran; it was cancelled and
  resubmitted on any 48 GB GPU, so the endpoint does not wait on the L40S queue. k0 and k1 run on L40S.

## D160 — 2026-11-12 ~02:05Z (Wed 9:05 PM EST) — A26 audit: checkpoint integrity checks on general GPUs
**Evidence.** A26 (`project.md` §40) asks for a fresh hash, load and tensor check of every retained checkpoint. Each
checkpoint is up to ~20.6 GB (model.pt ≈ 6.9 GB fp32 + optim.pt ≈ 13.8 GB). The CPU QOS had 111 pending jobs at 01:55Z and
the CPU standby 11122432 was pending. A26 authorizes all 8 general GPUs for verification.
**Action.** `qis-a26-ckpt-list` (CPU, listing only) and `qis-a26-ckpt-k0..3` (general, 1 GPU each, 96 GB RAM; directories
split by index mod 4) run the independent tool `research/audit-a26/tools/ckpt_audit.py` at 9b2fd05. The GPU itself is idle
in these jobs; the slot buys a prompt start and enough memory. Read-only on project data.

## D161 — 2026-11-12 ~03:15Z (Wed 10:15 PM EST) — A26: explain the cross-arm rise of J_U and J_R
**Evidence.** J_U and J_R rise 8–15 points in every arm, including arms that never practise the skill. The shared hard
problems are factoring-central and none is annotated modular = yes. A26 authorizes evaluations that explain such results.
**Action.** (1) Classify existing in-run probe attempts at rounds 0–128 by failure type (truncated, no box, multiple boxes,
wrong box) and re-score leniently. (2) Evaluate base and the six round-128 endpoints on a new control panel `control_v1`:
64 unused MATH test problems (geometry, counting and probability, precalculus, prealgebra; levels 4–5; 16 per subject,
hash-ordered, chosen without model output), 8 attempts, cap 12,288, seed label `control-v1`. The hypotheses and the
comparison are pre-registered in `research/audit-a26/transfer/PREREG.md` before any result. Exploratory; not a new
training study.
**Interpretation rule.** If all arms gain similarly on subjects none of them trained on, the J_U rise is read as general
improvement from GRPO training, not skill transfer from the shared hard problems.

## D162 — 2026-11-12 ~05:00Z (Thu 12:00 AM EST) — A26 audit: corrections that change conclusions of report v2
**Evidence.** Audit pass A (`research/audit-a26/pass-a/`) and the independent pass B (`pass-b/`) agree on every shared count
and point estimate (157 of 161 scripted comparisons; the 4 others resolved in `research/audit-a26/comparison.md`). No defect
changes a training update, a grade, an endpoint score or a primary contrast. The discrepancy log
(`research/audit-a26/discrepancies.md`) lists 38 defects: 11 change an interpretation, 8 change a number, 19 are wording.
**Corrections that change a conclusion (report v3 adopts each):**
1. Length extension (I1, I2). Old: "more room does not rescue those answers"; R-s202 "did not lose arithmetic skill". New:
   4,096 more tokens raised every checkpoint's pass@1 by 1.6–3.1 points and recovered 12–24% of capped attempts; only the
   R − U ordering stayed (change ≤ 0.6 points). Accuracy among finished compares selected responses; the skill claim is withdrawn.
2. Timing (I3). Old: the R − U sign appeared between rounds 32 and 64 in both seeds. New: seed 101 at round 64; seed 202 U is
   ahead from round 32 (before the cap change) by no more than the round-0 gap between identical policies.
3. Allocation (I4). Old: practice "was a better use of the slots". New: R − H and U − H point estimates are positive in both
   seeds, but 2 of 4 per-seed intervals include 0, and the H-s202 cells mixed GPU types. No curriculum recommendation.
4. Updates (I5). Old: practice slots "cost generation time without updating the model". New: 18–66 of 512 steps per run moved
   the weights with zero gradient (stored AdamW moments); every step updated the weights.
5. Interference (I6, I7). Old: link 1 "not detected" at 3 points. New: the local diagnostic ran at cap 8,192, its reference
   evaluations differ by up to 3.1 J_H points across GPUs, and the sign of mean ΔJ_H depends on the reference; the test
   cannot resolve the effect. Loss on training prompts occurs at the sampling rate. Stalls were never defined or measured.
6. Hardware (I8). New: training GPU type differs between runs (H-s101 L40; H-s202 and R-s202 switched L40 → L40S). Arm
   contrasts are not hardware-matched; "run-to-run variation" includes this confound.
7. Response audit (I9). "Fixable by factoring practice" is a reader annotation about a hypothetical intervention, not an effect.
**Action.** DEFINITIONS v1 → v1.1 (`research/audit-a26/definitions-v1.1-changes.md`); report v3 is written from the audited
tables; report v2 stays archived unchanged.
