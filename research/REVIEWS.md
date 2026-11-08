# REVIEWS
- 2026-11-07 test trigger received at 2026-11-07T02:30:02Z (one-shot cron 9cba2ef9; scheduler path verified)

## Review 1 — 2026-11-07 03:05Z
1. **Changes.** Done: env (+patch), split manifest, CPU tests 29/29, GPU probe, pass-1 annotation (1,496 items), rubric v1.1. Running: `qis-lenprobe-v1` (11072430), `qis-trainer-smoke-v1` (11072428), both general. Failed/cancelled and recorded: probe attempts 1–2 (recipe/env), lenprobe 11072286 and smoke 11072408 (skipped-precheck bug), tests 11072230 (env).
2. **Question alignment.** Both active jobs are gate work (cap decision; trainer correctness + precision audit). No drift: no result-seeking; no study outcome exists yet.
3. **Mechanics.** Probe: template correct, EOS present, all params trainable, update reaches vLLM. Open: logprob mismatch p99 0.127 (BF16 head in probe; trainer uses FP32 head — re-measure in smoke); checkpoint write 512 s (fixed by background writer, untested until smoke).
4. **Confounds/claims.** Annotation contamination in batches 2/5 caught by the quote check and redone (D13). Rubric v1.1 changed roles before any screening (D16) — documented as count-driven, outcome-blind. Risk: E_related may fall below 64 after screening.
5. **Next jobs / limits.** Stage-1 screening waits for the cap decision. GPU use: 2/2 general, 0/6 preempt. Storage: 163–173 GB used of 500 GB account; project ~10 GB. CPU work possible: pass-2 annotation prep.
6. **Most important uncertainty.** Whether the trainer passes G2 (resume equality, precision audit) on the first try; the background checkpoint path is new code. Action: smoke test running.
7. **Reconciliation.** Slurm ↔ ledger agree (14 ledger rows; 2 live jobs both owned). `registry/runs.jsonl` is empty because no scientific run exists; engineering attempts are in the ledger. STATE/CHECKLIST updated in this review.

## Review 2 — 2026-11-07 04:05Z
1. **Changes.** Done: G2 passed (D24–D26); stage-1 study screening; cap frozen 4,096 (D18); A2/A3 recorded; `ops/`, `registry/` untracked (D21). Running: `qis-screen2-study` 11072896 (general), `qis-screen1-calib` 11072944 (general, moved from preempt per D23). Queued: panels 11072990 (CPU) → six calibration runs (preempt, dependency).
2. **Alignment.** Screening and calibration both serve gates G1/G3. No outcome-driven choice: the H reserve (D19) and rubric v1.1 (D16) were fixed before the corresponding screens.
3. **Mechanics.** Stage 1: 0 verifier unknowns, truncation ≤ 3.4%. Smoke: mismatch p99 ~0.11, clip fractions ~1e-4, 5/16 zero-gradient steps (all-solved minibatches) — expected. Background checkpoint slows concurrent training ~3–5× while writing (D26) — watch on real rounds.
4. **Confounds/claims.** Main risk is construction validity: H_train projected 25–35 (< 64 hard minimum); fallback skill no better (model-free bound). This is a data-construction blocker, not a treatment result. Escalate after stage-2 exact counts.
5. **Next jobs / limits.** GPUs: 2/2 general running; 6/6 preempt slots used by pending calibration jobs (0 running). CPU: 1 pending. Storage: smoke checkpoints (~44 GB) to prune after this review; calibration will add up to 6 × 21.9 GB.
6. **Most important uncertainty.** Whether a defensible H_train ≥ 64 exists. Action: stage-2 counts, then options to the owner (band widening recommended provisionally).
7. **Reconciliation.** Slurm ↔ ledger agree (all 11 owned live/pending jobs in ledger). Registry untracked (D21). STATE/CHECKLIST current as of 03:45Z plus calibration queue entry.

## Review 3 — 2026-11-07 05:05Z
1. **Changes.** Done since review 2: G2 evidence tracked; full native stage 2 (5,888 attempts) → 29 eligible H (18 core, 11 moderate); target/probe panels frozen (D34); calibration panels frozen (D33); authored tranche (400) written, intake (394 kept), blind-annotated; Omni-MATH tranche (200) annotated (73 H-eligible); amendments A4, A5, v1.3 tagging recorded. Running: calibration lr 1e-6/sym (11072995, round 4 of 32), authored screening (11073304). Pending: Omni screening + 5 calibration runs on preempt (no start estimate; D37).
2. **Alignment.** All work builds the frozen dataset or calibrates the optimizer; no treatment run exists. Every selection rule was committed before its outcome data (D28, D31, D32, D38, D39).
3. **Mechanics.** Real calibration rounds 138–178 s (gen ~60 s, train 60–116 s), 11–15/16 mixed groups, truncation 1–3/128, logprob mismatch p99 0.107 (D25 limit 0.20), clip fractions ~1e-4, device memory 36–38.5 GiB. Baseline calib_val pass@1 0.4265.
4. **Confounds.** Authored auxiliary items are matched within provenance (D32/D39). Omni items enter H_train only (provenance recorded; held-out MATH untouched). Watch: if H_train is majority Omni, training distribution shifts from MATH targets — report transfer separately (A4.6).
5. **Next jobs / limits.** GPU 2/2 general used; preempt unusable now. Cost model: main run ≈ 128 × 2.7 min + evals ≈ 6.5 GPU-h (< 24 h threshold → 128 rounds feasible); 15 runs ≈ 100 GPU-h ≈ 50 h wall on 2 GPUs.
6. **Most important.** Omni screening is the critical path for H_train but waits for a GPU slot. Action: it takes the next free general slot (before remaining calibration runs).
7. **Reconciliation.** Slurm ↔ ledger agree (8 live owned jobs). Watcher was found not re-armed after the last event; re-armed now (rule: re-arm in the same step as harvesting).

## Review 4 — 2026-11-07 06:06Z
1. **Changes.** Omni screening done (195 after intake; core 33, moderate 14 before audit filters; 2 test-overlap items excluded at intake). Authored tranche 1 screened (R 9 / U 30 eligible in band); tranche 2 (D40) authored, intake (296 R / 118 U), blindly annotated (218 R / 120 U eligible), screening running on general (11073719). Combined H selection queued on CPU (11073966, partition Priority). Calibration run 1 at ~1 h 35 min.
2. **Alignment.** All work is dataset construction or calibration. Selection/matching rules were committed before their outcome data (D38, D39). Difficulty calibration given to authors used auxiliary screening counts only (no treatment data exists).
3. **Mechanics.** Omni truncation 5.4% of attempts (selection excludes truncation-driven items). Verifier unknowns: 1 native, 1 Omni — excluded and logged, never dropped silently. Calibration rounds healthy (review 3).
4. **Confounds.** Authored items now dominate the auxiliary pools: within-provenance matching (D32/D39) keeps synthetic share equal between R and U. Parameterized authoring (D41) handled by family counting. H_train may be majority external (Omni): MATH-target transfer must be reported separately from training-pool performance (A4.6).
5. **Next jobs / limits.** GPU 2/2 general; preempt still not starting (D37). CPU: 1 job pending (queue of 85 on the CPU partition).
6. **Most important.** Finish H selection + R/U matching → freeze dataset; stage-3 baseline. Calibration throughput (2 GPUs) is the next bottleneck: 5 runs + 2 seed-22 runs ≈ 13 GPU-h.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned live jobs). Watcher alive (re-armed 06:0x after the Omni event).

## Review 5 — 2026-11-07 07:05Z
1. **Changes.** D45: cap 4,096 → 8,192 after calibration showed validation truncation 11.5% (genuine long reasoning, not loops). Cancelled six owned 4,096 jobs (~56 GPU-min). H selection at 4,096 reached 49 (D43, superseded pending re-screen). Running: 8,192 re-screen H+calibration (11074495; 107 H items) and auxiliary (11074499; 302 related items at stage 1, 5.1k tok/s). Queued: panels-k8 → six cal8k runs.
2. **Alignment.** The cap change protects the endpoint from a growing truncation confound; decided before any treatment run and applied to every pool.
3. **Mechanics.** 8,192 generation runs (vLLM max_seq_len 9,216). Learner at 9k tokens untested → first cal8k run is the canary.
4. **Confounds.** Pools must be re-selected from 8,192 counts with the same frozen rules (D38/D39); no rule changes.
5. **Next jobs / limits.** GPU 2/2 general; preempt still not starting. Storage ~120 GB used account-wide.
6. **Most important.** The 8,192 canary (memory, round time) — risk of OOM with 9k-token sequences. Action at first cal8k start: check first-round metrics within ~10 min.
7. **Reconciliation.** Slurm ↔ ledger agree (9 live owned jobs). Watcher alive.

## Review 6 — 2026-11-07 08:05Z
1. **Changes.** Aux re-screen 11074499 failed after stage 1 (shortlist script hard-coded the `s1` stage filename; my bug); fixed and resubmitted as 11074744, resuming at shortlist (stage-1 8k counts reused, 541 items). H+calibration re-screen 11074495 finished H (107 items), now screening calibration candidates.
2. **Alignment.** Unchanged: dataset + calibration at the 8,192 cap; no treatment run.
3. **Mechanics.** 8k generation 3.8–6.5k tok/s; no verifier errors reported in logs.
4. **Confounds.** None new.
5. **Next jobs / limits.** GPU 2/2 general (both re-screens); 7 jobs waiting on dependencies (panels-k8, six cal8k).
6. **Most important.** Calibration remains the long pole (≈6 runs × ~3 h at 8k on 2 GPUs). Action: when the H job ends, move the first cal8k run (canary) to general immediately.
7. **Reconciliation.** Slurm ↔ ledger agree; failed attempt 11074499 kept in ledger (cost ~41 GPU-min, stage-1 output reused). Watcher alive.

## Review 7 — 2026-11-07 09:05Z
1. **Changes.** Gate G1 passed at the 8,192 cap: H_train 46 (D47), 63 matched aux pairs with A5 coverage and balance (D48), panels unchanged (D34), calibration panels (D46), main bundle built (sha `0e255243…`; sealed test `2ffe70dd…`). Canary cal8k healthy (D49). Running: cal8k lr1e-6/sym (52 min), lr5e-7/sym (27 min). Queued: 4 cal8k, stage-3 baseline.
2. **Alignment.** All selections followed rules committed before the data (D38/D39/D47 rule application unchanged at 8k). No treatment run.
3. **Mechanics.** 8k rounds ~150–210 s; memory peak 40.0 / 44.4 GiB (4.4 GiB headroom — watch for OOM on long batches); truncation ≤ 1/128.
4. **Confounds.** Cross-job eval noise 0.010 at identical seeds (D49) → calibration ties likely; main-study baselines must be averaged across runs. Authored share of aux pools 86% (equal across arms by construction).
5. **Next jobs / limits.** GPU 2/2 general; preempt idle (no starts). Calibration ETA: 6 runs × ~1.8 h / 2 GPUs ≈ 5.5 h, plus 2 seed-22 runs ≈ 1.8 h.
6. **Most important.** Prepare main-study configs, cost estimate and PROTOCOL freeze text now so the matrix can launch the moment calibration selects; draft the analysis plan.
7. **Reconciliation.** Slurm ↔ ledger agree (7 live owned jobs). Watcher alive.

## Review 8 — 2026-11-07 10:05Z
1. **Changes.** No job ended this hour. Running on general: cal8k lr1e-6/sym (11074520, ~1.9 h in; round 7 + baseline eval at the 09:10Z peek) and lr5e-7/sym (11074527). Pending on preempt: four cal8k runs and stage-3 baseline 11075046. New code: `src/qis/analysis/stats.py` (paired t interval, crossed bootstrap seed 2027, per-problem change) with `tests/test_analysis.py`; test job `qis-tests-b` 11075419 submitted. PROTOCOL draft, main recipe generator and checkpoint evaluator committed last hour.
2. **Alignment.** Analysis code is written before any endpoint data, as §9 requires. No treatment run.
3. **Mechanics.** Calibration rounds at 8k ~150–210 s plus ~8 min per validation pass; no fault in logs at the last peek.
4. **Confounds.** None new. Cross-job eval noise 0.010 (D49) stays the main caution for calibration ranking.
5. **Next jobs / limits.** GPU 2/2 general; preempt 0 starts (D37). CPU 4/30.
6. **Most important.** Keep both general GPUs busy: when a cal8k run ends, move the next pending one (order: 5e-7/asym, 1e-6/asym, 2e-6/sym, 2e-6/asym; stage-3 baseline in between if a CPU-light gap appears) to general at once.
7. **Reconciliation.** Slurm ↔ ledger agree (7 live owned GPU jobs + 1 CPU test job). Watcher alive (btjiyyann).

## Review 9 — 2026-11-07 11:05Z
1. **Changes.** Two calibration runs done (table `research/evidence/calibration/cal8k_summary.md`): lr5e-7/sym gain +0.031, lr1e-6/sym gain +0.122; both valid (validation truncation ≤ 1.12%, 0 unknown). Moved 11074526 (5e-7/asym) and 11074519 (1e-6/asym) to general; both running. Analysis tests 33/33 (11075445) after 11075419 failed on missing scipy (replaced by a verified t table).
2. **Alignment.** Calibration follows the frozen §7 rule; no selection yet (4 configs outstanding, then seed-22 repeats).
3. **Mechanics.** Rounds 176 s (5e-7) and 237 s (1e-6; longer responses as the policy learns). Device memory max 41.3 GiB of ~44.4. The logged `mem_peak_gib` 47.5 is the allocator peak incl. the vLLM pool, not device use.
4. **Confounds.** Round-0 baselines differ 0.010 across jobs at identical seeds (D49); gains are within-run.
5. **Next jobs / limits.** GPU 2/2 general; preempt still 0 starts. Provisional cost (measured 237 s/round, 5 dev evals): 9.0 GPU-h per 128-round run (13.2 h with a 1.5× margin for longer hard-item responses) — below the 24 GPU-h rule, so T=128 is expected. 15 runs ≈ 135–200 GPU-h ≈ 68–99 h wall on 2 GPUs.
6. **Most important.** Calibration remains the critical path: ETA for the last seed-11 runs ~16:00Z, seed-22 repeats ~18:30Z. Stage-3 baseline (3 h) waits behind calibration unless preempt starts.
7. **Reconciliation.** Slurm ↔ ledger agree (4 live owned GPU jobs; old cancelled/failed rows kept). Watcher alive (b87y26pwf).

## Review 10 — 2026-11-07 12:05Z
1. **Changes.** No job ended. Peek 11075838 (read-only): cal8k 5e-7/asym at round ≥17 and 1e-6/asym at round ≥15 of 32, ~1 h in.
2. **Alignment.** Unchanged; calibration under the frozen §7 rule.
3. **Mechanics.** Both runs healthy: 0 truncated training responses in recent rounds, mixed groups 13–15/16, logprob mismatch p99 0.105–0.111 (alarm thresholds D25 not hit), ratio q0.001–q0.999 within 0.91–1.10, clip fractions ≤ 1.2e-4, device memory ≤ 38.8 GiB. Rounds 114–216 s.
4. **Confounds.** None new.
5. **Next jobs / limits.** GPU 2/2 general; preempt 0 starts; `/data/user_data` 163/500 GB used (account-wide). Harness quirk: `ae harvest` waits while `squeue` still shows COMPLETING after `sacct` reports COMPLETED; read the distillate directly or harvest later.
6. **Most important.** Same critical path: both asym runs end ~13:30Z → move the 2e-6 pair to general at once.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (b87y26pwf).

## Review 11 — 2026-11-07 13:05Z
1. **Changes.** Two more calibration runs done: 5e-7/asym −0.003 (11074526), 1e-6/asym +0.066 (11074519); all four finished runs valid. 2e-6/sym (11074522) running on general; 2e-6/asym (11074521) moved to general. Seed-22 repeat of 1e-6/sym queued on preempt (11075978, D51).
2. **Alignment.** Selection waits for all six seed-11 runs, per §7. The early seed-22 queueing uses idle preempt budget and changes no rule.
3. **Mechanics.** Healthy runs: device memory ≤ 42.2 GiB, validation truncation ≤ 1.45%, 0 unknown verifier results.
4. **Confounds.** Eval reproduction across jobs is partial (56/56 identical in one pair, 8/56 in another; D51). Noise planning figure 0.020 per gain (D50).
5. **Next jobs / limits.** GPU 2/2 general + 1 pending preempt (budget 6). Seed-11 calibration ends ~15:45Z; then the top-2 repeats.
6. **Most important.** Shorten the critical path to the freeze: done by queueing the likely seed-22 repeat now. Next: on the 2e-6 results, apply §7 and queue the second repeat.
7. **Reconciliation.** Slurm ↔ ledger agree (3 live owned GPU jobs + stage-3 pending). Our watcher re-armed; another project's ae-watch process (PID 17928) left untouched.

## Review 12 — 2026-11-07 14:05Z
1. **Changes.** First preempt start of the study: stage-3 baseline 11075046 started 13:47Z (preempt_qos). 2e-6/sym (11074522, since 12:50Z) and 2e-6/asym (11074521, since 13:07Z) running on general. Seed-22 repeat 11075978 pending on preempt (Slurm start estimate 11-11, unreliable).
2. **Alignment.** Unchanged. Stage-3 counts are reported only, never used to reselect H (A4.2).
3. **Mechanics.** No new failures. Watcher alive (bvz1fzoxo).
4. **Confounds.** None new.
5. **Next jobs / limits.** GPU 2/2 general, 1/6 preempt running + 1 pending. Seed-11 calibration ends ~15:00–15:45Z.
6. **Most important.** When a 2e-6 run ends: apply §7 with all six, queue the second seed-22 repeat, and move both repeats to general as slots free (preempt start remains unpredictable, D37).
7. **Reconciliation.** Slurm ↔ ledger agree (4 live owned jobs, all `qis-` rows).

## Review 13 — 2026-11-07 15:05Z
1. **Changes.** Stage-3 baseline done (D52): H mixed-group fraction 0.77, mean p 0.202; R 0.589 / U 0.569 (mixed 0.94 each). Calibration 2e-6/sym (2 h 16 min) and 2e-6/asym (1 h 58 min) still running on general.
2. **Alignment.** Unchanged; selection after both 2e-6 runs, per §7.
3. **Mechanics.** W&B online logging confirmed for all finished calibration runs (no `wandb_failed` events); local JSONL is the analysis source.
4. **Confounds.** None new.
5. **Next jobs / limits.** GPU 2/2 general; preempt 0 running, 1 pending (11075978).
6. **Most important.** On the 2e-6 results: apply §7, queue the second seed-22 repeat, move both repeats to general.
7. **Reconciliation.** Slurm ↔ ledger agree (3 live owned jobs). Watcher alive (b8wlrdna2).

## Review 14 — 2026-11-07 16:05Z
1. **Changes.** lr2e-6/sym done: gain +0.146, validation truncation 4.69% (valid; D53); its seed-22 repeat queued on preempt (11076551). lr1e-6/sym seed 22 (11075978) moved to general, at round 8. Peek 11076768: lr2e-6/asym (11074521) finished round 32 and is in its final evaluation (round-16 pass@1 0.596 vs 0.487 at round 0).
2. **Alignment.** §7 applied as written; finalist set fixed once 2e-6/asym reports.
3. **Mechanics.** lr2e-6/asym is slower (2 h 58 min so far) because responses grew: mean length 1,800–2,500 in rounds 29–32, training truncation 3–8/128, round-16 evaluation 873 s vs ~450 s at round 0. Not stalled. Logprob mismatch p99 ≤ 0.118.
4. **Confounds.** lr 2e-6 lengthens responses fast; truncation risk for 128-round runs if 2e-6 is selected (D53).
5. **Next jobs / limits.** GPU 2/2 general, 1 pending preempt. CPU 1/30 (peek).
6. **Most important.** On 2e-6/asym: fix the finalists (cancel 1e-6/sym s22 only if 2e-6/asym > +0.122), move 11076551 to general.
7. **Reconciliation.** Slurm ↔ ledger agree (3 live owned GPU jobs). Watcher alive (biluz30jp).

## Review 15 — 2026-11-07 17:05Z
1. **Changes.** First-stage calibration complete (D54): lr2e-6/asym +0.196 rejected (validation truncation 7.14%); finalists lr2e-6/sym (+0.146) and lr1e-6/sym (+0.122). Both seed-22 repeats running on general (11075978 since 15:24Z; 11076551 since 16:14Z). Cost estimate prepared for both outcomes (`research/evidence/cost/main_cost_estimate.md`).
2. **Alignment.** §7 applied as written; the rejection of the highest-gain config follows the frozen truncation threshold.
3. **Mechanics.** Main recipe wall limit raised 24 h → 40 h: the worst projected run (H arm, lr 2e-6, slow L40 node) is ~17.5–21 GPU-h, too close to 24 h.
4. **Confounds.** H_train base responses are 1.5× longer than calibration items (2,199 vs ~1,450 tokens); H arm costs ~25% more per round than R/U (reported; equal prompt/response counts, not equal tokens, as §7 states).
5. **Next jobs / limits.** GPU 2/2 general. Projected main study 189–264 GPU-h; T=128 under every projection (≤ 21 GPU-h per run).
6. **Most important.** At ~18:30–19:00Z: apply §7 to the seed-22 results, fill PROTOCOL, freeze, generate recipes, launch the canary main run.
7. **Reconciliation.** Slurm ↔ ledger agree (2 live owned jobs). Watcher alive (bke26mee7).

## Review 16 — 2026-11-07 18:05Z
1. **Changes.** Seed-22 repeats still running on general: lr1e-6/sym (11075978, 2 h 41 min), lr2e-6/sym (11076551, 1 h 52 min). New: `src/qis/analysis/calib_select.py` (frozen §7 rule as code) and recipe `qis-calib-select` (CPU job) for the official selection.
2. **Alignment.** Selection becomes mechanical and recomputable from committed metrics with input checksums.
3. **Mechanics.** No failures.
4. **Confounds.** None new.
5. **Next jobs / limits.** GPU 2/2 general; CPU 0/30.
6. **Most important.** Process issue found and fixed: I ran a read-only preview of the selection script, and earlier one-line metric summaries, on the laptop, which breaks the "no project code on the laptop" rule (D55). From now on every reported statistic is computed in a CPU job.
7. **Reconciliation.** Slurm ↔ ledger agree (2 live owned jobs). Watcher alive (bke26mee7).

## Review 17 — 2026-11-07 19:08Z
1. **Changes.** Calibration selected lr 2e-6 / clip (0.2, 0.2) (D57; CPU job 11078053, tests 4/4); protocol frozen at `b8f36ca`, T=128 (D58). Owner amendment A6 recorded (project.md §20, D56); thinking-mode diagnostic code and grading tests committed before output. Canary `qis-main-R-s101-sym` (11078086) running on general since 19:07Z; A6 job `qis-a6-modes-k8` (11078087) pending on general.
2. **Alignment.** Main study runs the frozen design; A6 is a separate diagnostic with distinct IDs and outputs.
3. **Mechanics.** First main-arm (R) run: new code path (mixed minibatches, dev/probe evals, retained round-64 checkpoint) → canary rule; first-round check due ~19:30Z. Selection attempt 11078035 failed (parsing bug, fixed with a test).
4. **Confounds.** lr 2e-6 lengthens responses; asymmetric sensitivity arm exceeded 5% truncation in calibration. Both monitored per arm.
5. **Next jobs / limits.** GPU 2/2 general (1 running, 1 pending); preempt 0/6. 14 main runs wait for the canary check.
6. **Most important.** Canary first-round check, then release 14 runs (preempt first, general as slots free).
7. **Reconciliation.** Slurm ↔ ledger agree (2 live owned jobs). Watcher alive (b8h8tyfqi).

## Review 18 — 2026-11-07 20:07Z
1. **Changes.** A7 applied (3 general GPUs, D60). Canary passed; main matrix released (D61): R-s101, U-s101, H-s101 running on general; R-s202, U-s202, H-s202, R-s303, U-s303 queued on preempt. A6 same-cap result (D62): thinking raises pass@1 +0.23–0.25 in every pool but truncates 51.6% of H responses; auxiliary practice near-trivial (mixed groups 13–25%); R/U still matched; 5.5–8.1× GPU time. 16,384 profile queued (11078421); H-s303 (never started) cancelled to stay within 6 preempt GPUs.
2. **Alignment.** Main study runs the frozen design; A6 is separate (own IDs, outputs, evidence directory).
3. **Mechanics.** Canary round 1 healthy (175 s, 38.6 GiB, mismatch p99 0.103, truncation 1/128).
4. **Confounds.** None new for the main study. A6: auxiliary pools lose GRPO signal under thinking mode — a design problem for any thinking-mode transfer study.
5. **Next jobs / limits.** GPU 3/3 general running; preempt 6/6 queued (0 started). H-s303 and 6 sensitivity runs wait for slots.
6. **Most important.** Keep all three general GPUs busy (move the oldest queued primary run on each free slot); finish A6 failure inspection (agent running) and the 16,384 profile.
7. **Reconciliation.** Slurm ↔ ledger agree (9 live owned jobs; cancelled 11078412 kept in the ledger). Watcher alive (b5j64v5my).

## Review 19 — 2026-11-07 21:05Z
1. **Changes.** A8 executed: A6 closed (11078421 cancelled, cost 0; D64), D63 narrowed (D65), truncation monitor (D66), reconciliation + calibration-panel deviation note + forecast (D67), calibration checkpoints pruned (204 GB; D68), §3.7 audit complete (D69: missed opportunity 30/64, execution 6, after-correct-factoring 20, valid alternative 6, extraction 2, split-box 0). Storage released → cap 7, R-s303 released (D70).
2. **Alignment.** Main study unchanged; sealed test untouched; A6 closed.
3. **Mechanics.** Main runs healthy: rounds 13–26, mismatch p99 ≈ 0.11, train memory ≤ 41.6 GiB, round time 259–306 s (lengthening responses).
4. **Confounds.** Shared hard schedule truncation R 1.0% vs U 2.3% (21 rounds): small, monitored; no 5% crossing.
5. **Next jobs / limits.** GPU 3/3 general running; preempt 4 queued (R-s202, U-s202, H-s202, R-s303), 2 held (U-s303, H-s303). Storage 195/500 GB.
6. **Most important.** Keep general slots filled as runs finish (~06:00Z on 8 Nov); local-update diagnostic after R-s101 round 64; test evaluations at endpoints.
7. **Reconciliation.** Slurm ↔ ledger agree (9 live owned jobs). Watcher alive (b69kcksnz).

## Review 20 — 2026-11-07 22:19Z (extended by amendments A9 and A10)
1. **Changes.** A9: scope cut to six primary runs; seed-303 jobs cancelled (cost 0); sensitivity deferred (D71). Tooling for diagnostic, test evals, endpoint validation and the report table (D72). A10: 8k-to-16k schedule at common boundary round 64 (D73), code + smoke test + watchers (D74).
2. **Alignment.** Frozen data and hyperparameters kept; boundary recorded before further comparative results; test set unopened.
3. **Mechanics.** s101 runs healthy; responses 2.5–3.1k tokens, rounds 294–380 s.
4. **Confounds.** Dev truncation at round 32: R 11.6%, U 16.7%; training hard capped U 5.1% (shared schedule R 2.7% vs U 5.1%). Truncated responses are long reasoning, not loops. This motivated A10.
5. **Next jobs / limits.** GPU 3/3 general (s101); preempt 6/6 queued (3 s202, base test eval, diagnostic r0, smoke). CPU: 3 watchers + tests. Storage 200/500 GB.
6. **Most important.** Pass the smoke test before round 64 (~01:30Z on 8 Nov) and execute the transition without losing completed rounds.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher re-armed.

## Review 21 — 2026-11-07 23:05Z
1. **Changes.** A10 code tests 44/44 (11079542). Boundary watchers running. Smoke test, base test eval, diagnostic r0 and s202 runs queued on preempt (no preempt start).
2. **Alignment.** Rounds 1–64 at 8,192 in all runs; nothing has trained under the amendment yet.
3. **Mechanics.** Healthy: mismatch p99 0.115–0.128, train memory ≤ 41.2 GiB; rounds 316–449 s; mean length 2,808–3,617.
4. **Confounds.** Capped hard training responses (cumulative): R-s101 5.1%, U-s101 7.7%, H-s101 2.5%; shared hard schedule (40 rounds) R 4.4% vs U 7.7%. Under the 8,192 phase these are capped → reward 0; A10 addresses rounds 65–128. Reported per A8.4.
5. **Next jobs / limits.** GPU 3/3 general; preempt 6/6 queued; CPU 3 watchers. Storage 201/500 GB. Round 64 forecast: R ~01:00Z, U ~01:30Z, H ~03:00Z (8 Nov).
6. **Most important.** The smoke test gates the transition; if still pending at R-s101 round 64, run it on R's GPU before releasing any 16k round.
7. **Reconciliation.** Slurm ↔ ledger agree (12 live owned jobs). Watcher alive (bfer77hh1).

## Review 22 — 2026-11-08 00:07Z
1. **Changes.** Owner continuation of A10 recorded (D75): boundary dev eval at 16,384 from `boundary_0064` (both budgets, crossing-then-correct, throughput benchmark), learning curves by role and R vs U shared schedule. W&B mirror defects fixed and companion runs backfilled (D76). Probe table at round 32 (J_H derived from dev): J_H p@1 H 0.555, R 0.523, U 0.555 (base 0.434). Tests 44/44 (11080163).
2. **Alignment.** No training under the new cap yet; boundary 64 unchanged; sealed test unopened.
3. **Mechanics.** s101 runs healthy; R-s101 ~round 55, U ~50, H ~40 (estimates from round rates).
4. **Confounds.** Truncation at 8,192 differs by arm (dev r32: R 11.6%, H 15.6%, U 16.7%) and lowers p@1; the 16k boundary evaluation quantifies it.
5. **Next jobs / limits.** GPU 3/3 general; preempt 6/6 queued (none started); CPU: 3 watchers. Storage ~200/500 GB.
6. **Most important.** The R-s101 boundary (~01:00Z): watcher event → cancel old job → smoke on R's GPU → replacement with boundary dev eval. Process fix: ae-watch was not running after the probe-table failure; re-armed now (rule: re-arm after every event, including failures I caused).
7. **Reconciliation.** Slurm ↔ ledger agree (12 live owned jobs + probe-table finishing).

## Review 23 — 2026-11-08 01:06Z
1. **Changes.** A11 pause executed (D77): s101 jobs stopped at rounds 55/51/37 (13.9 GPU-h superseded), s202 and base test eval held; no round-32 state → replay from round 0. Truncation study: dev 16,384 evals of R r52 / U r48 running. Smoke test OOM at 16k (D78) → LM-head recompute fix; smoke rerun queued.
2. **Alignment.** No main training; sealed test closed; cap not yet frozen.
3. **Mechanics.** The OOM is a real constraint found before any 16k training — the smoke test did its job.
4. **Confounds.** None new.
5. **Next jobs / limits.** General 3/3 (2 dev evals + smoke rerun pending); preempt: diag r0 queued, 4 held.
6. **Most important.** Pass the smoke rerun, finish the truncation study, freeze the cap, resume.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (beb24mu29).

## Review 24 — 2026-11-08 02:05Z
1. **Changes.** Smoke test passed at 16,384 (D79: peak 37.8 GiB, worst-case round 12.2 min). Evaluator throughput fix and weight-load read-back check (D79, D80); slow old dev evals cancelled (3.6 GPU-h) and rerun.
2. **Alignment.** Main matrix paused; sealed test closed; cap not frozen.
3. **Mechanics.** Memory feasible at 16k; throughput is the binding constraint.
4. **Confounds.** None new.
5. **Next jobs / limits.** General 3/3: dev16k H (new evaluator, 40 min in), R and U reruns. Preempt: diag r0 queued; s202 and base test eval held.
6. **Most important.** Freeze the cap as soon as the dev truncation study lands (~03:00–03:15Z), then resume all six runs. Pause has cost ~2 h of training time so far.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (b0nchn972).

## Review 25 — 2026-11-08 03:05Z
1. **Changes.** Cap frozen at 12,288 (D81) from H r36 dev at 16,384 (12,288 keeps 88% of the beyond-8k gain; 16,384 infeasible for the deadline). A11 resume (D82): a1 dirs superseded; six runs restarted at round 0 as attempt a2; R-s101 11081724 training on general since 03:01Z (precheck incl. stale guard ok; boundary_eval skipped as designed).
2. **Alignment.** Rounds 1–32 at 8,192, 33–128 at 12,288 for every arm; sealed test closed until endpoints.
3. **Mechanics.** Training restarted cleanly; first-round health check pending.
4. **Confounds.** The R-s101 round-64 diagnostic source is now a mixed 8k/12k state (rounds 33–64 at 12,288); reported as such.
5. **Next jobs / limits.** General 3/3 (R-s101 + 2 dev 16k evals ending soon); preempt 5/6 queued (U/H-s101, R/U-s202, diag r0); H-s202 waits for a slot.
6. **Most important.** Move U-s101 and H-s101 onto the general GPUs as the dev evals end; verify round 33 cap at ~05:30Z (R-s101).
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (b1hmgiolw).

## Review 26 — 2026-11-08 04:05–04:20Z
1. **Changes.** dev16k-R (11081333) completed: R-s101 round 52, pass@1 0.592 at 16,384, 0.572 within 8,192; 19.6% cross 8,192; weight check passed (D83). D83 mislabelled the round as 36; corrected in D84. U-s101 moved to general and started 04:03Z. H-s202 submitted (11082192). Monitor resubmitted (11082191).
2. **Valid?** Yes so far. All a2 prechecks passed (stale-run guard included); `boundary_eval` skipped by design. No round metrics inspected yet this hour; the monitor will show them.
3. **Surprises.** My own labelling error (round 52 vs 36). The H/R crossing-rate gap is confounded by round, not evidence of an arm effect.
4. **Risks.** U-s101 waited ~1 h in the general queue. Report target 10 Nov 01:00Z unlikely; deadline 14:30Z holds on current forecast.
5. **Next jobs / limits.** General 3/3 (R-s101, U-s101, dev16k-U). Preempt 5/6 queued (H-s101, R/U/H-s202, diag r0).
6. **Most important.** Verify round 33 uses 12,288 in R-s101 (expected ~05:30Z); move H-s101 to general when dev16k-U ends.
7. **Reconciliation.** Slurm ↔ ledger agree: 8 owned live jobs, all with ledger rows. `registry/runs.jsonl` is untracked and unused (D21); the ledger is the run record. Watcher alive (blc1tl7lp).

## Review 27 — 2026-11-08 05:05–05:19Z
1. **Changes.** Truncation study done for H/R/U (D86) and the inspection sample labelled (D87: no extraction misses; long responses mostly no-progress or wrong). All three s101 a2 runs train on general. At 05:06Z (monitor 11082493): R-s101 round 23, U-s101 round 15, H-s101 round 8. W&B companions `*-a2-curves` and `qis-RvU-s101-sym-a2-curves` populate.
2. **Valid?** Yes. All rounds so far are ≤ 32, so they run at 8,192 by design. Round times: R 388 s, U/H 230 s.
3. **Surprises.** `peek` did not export `max_tokens`, and `curves` defaults a missing value to 8,192, so neither could prove the transition. Fixed `peek` (d12cf9c); added `qis.analysis.capcheck` (59d3927).
4. **Risks.** Preempt has 866 pending jobs; the seed-202 runs may not start there, so they run on general after seed 101. Forecast unchanged: report ≈ 10 Nov 07:00–08:00Z; deadline 14:30Z holds. Storage 266/500 GB.
5. **Next jobs / limits.** General 3/3. Preempt 4 GPU jobs queued (R/U/H-s202, diag r0). CPU: `qis-a11-capcheck` 11082575 waits for R-s101 round 34 (≈06:20Z).
6. **Most important.** The A11.6 check: round 33 at 12,288 in rollout, training and grading (capcheck, R-s101 first; U and H later).
7. **Reconciliation.** Slurm ↔ ledger agree (7 owned GPU jobs + 1 CPU job, all ledger rows). Harness note: `ae harvest` skipped the new monitor output (stale `.harvested`); I copied the distillates directly. Watcher alive (b87d137fw).

## Review 28 — 2026-11-08 06:05Z
1. **Changes.** `qis-a11-capcheck` first attempt failed at 0 s (missing `PROJ_DATA`, D88); 11082618 is running and waits for R-s101 round 34. No other job changed state.
2. **Valid?** Yes. The three s101 runs run continuously on general (R 3:04, U 2:02, H 1:47 elapsed); no restarts.
3. **Surprises.** None new.
4. **Risks.** Seed-202 runs still queued on preempt (Priority). Forecast unchanged (report ≈ 10 Nov 07:00–08:00Z; deadline 14:30Z).
5. **Next jobs / limits.** General 3/3; preempt 4 GPU jobs queued; 1 CPU job running.
6. **Most important.** Read the capcheck result for R-s101 (round 33 at 12,288 in rollout, training, grading) when it fires (≈06:20–06:40Z).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (bmi015fu4).

## Review 29 — 2026-11-08 07:05–07:12Z
1. **Changes.** R-s101 round-33 transition verified (D89). Monitor 11083516: R-s101 round 39, U-s101 round 33, H-s101 round 23. `qis-a11-capcheck2` 11083320 waits for round 41 in all three.
2. **Valid?** Yes. No restarts; round metrics carry `max_tokens` as expected.
3. **Surprises.** H-s101 rounds are slow (569 s at 8,192) with 13/128 capped at round 23 — longer responses than R or U at this stage. Report-only under A8.
4. **Risks.** H is now the critical path; margin to the deadline ≈ 7 h (D90). Seed-202 runs still pending on preempt.
5. **Next jobs / limits.** General 3/3. Seed-202 start order on general: H, U, R (D90).
6. **Most important.** When the first s101 run ends, move H-s202 (11082192) to general first.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (bibndpl0v).

## Review 30 — 2026-11-08 08:05Z
1. **Changes.** None in job state. All three s101 runs keep running on general (R 5:04, U 4:02, H 3:47 elapsed).
2. **Valid?** Yes; no restarts.
3. **Surprises.** None new.
4. **Risks.** `qis-a11-capcheck2` (started 06:36Z, 4 h limit) waits for H-s101 round 41; at ≈10 min per round H reaches it ≈10:15–10:45Z, so the check may time out. If it does, resubmit it with a longer limit (no study impact).
5. **Next jobs / limits.** General 3/3; preempt 4 GPU jobs queued; seed-202 order on general H, U, R (D90).
6. **Most important.** Keep the general GPUs filled when the first s101 run ends (≈18:00–20:00Z).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (bibndpl0v).

## Review 31 — 2026-11-08 09:05–09:09Z
1. **Changes.** Monitor 11084097: R-s101 round 52, U-s101 round 48, H-s101 round 32. capcheck2 replaced (D91).
2. **Valid?** Yes. All rounds ≥ 33 in R and U log `max_tokens` 12288; H round 32 logs 8192. Evaluations at round 32 used 8,192.
3. **Surprises.** My error: D89 named round 40 as the first 12k evaluation; the schedule is every 32 rounds, so it is round 64 (D91).
4. **Risks.** H truncation at 8k is 11–18% per round (report-only). Forecast unchanged (D90).
5. **Next jobs / limits.** General 3/3; preempt 4 GPU jobs queued; CPU capcheck2 11084106.
6. **Most important.** capcheck2 result for U and H; then the round-64 eval-cap check.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (bibndpl0v).

## Review 32 — 2026-11-08 10:05Z
1. **Changes.** Round-33 cap transition verified in all three s101 runs (D92). Submitted `qis-diag-local-r64` (11084344, preempt) to gain queue age; its precheck refuses to run until R-s101 `round_0064` (retained) has `optim.pt` and `state.json`.
2. **Valid?** Yes; s101 runs continuous on general (R 7:04, U 6:01, H 5:47); no restarts.
3. **Surprises.** None.
4. **Risks.** Preempt queue still blocks seed 202 and both diagnostics. If r64 starts before R-s101 reaches round 64 (≈11:00Z), its precheck fails and it needs one resubmission. Forecast unchanged (D90).
5. **Next jobs / limits.** General 3/3; preempt 5/6 (R/U/H-s202, diag r0, diag r64).
6. **Most important.** Keep the general GPUs filled when the first s101 run ends; check the round-64 evaluation cap.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (b9sm9buty).

## Review 33 — 2026-11-08 11:05–11:08Z
1. **Changes.** Monitor 11084490: R-s101 round 63, U-s101 round 62, H-s101 round 40. All rounds ≥ 33 log 12,288.
2. **Valid?** Yes; no restarts. Storage 265/500 GB.
3. **Surprises.** H-s101 rounds at 12,288 take ≈ 950 s (7–16% capped), much slower than R (≈ 743 s) and U (≈ 515 s).
4. **Risks.** All six runs plus the 7 sealed-test evaluations no longer fit before 10 Nov 14:30Z on 3 general GPUs; preempt gives no start estimate. Plan changed (D93): R−U for both seeds first; H-s202 conditional; partial delivery of R−H for seed 202 if needed (A9).
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued. Next slot ≈ 21:40Z (U-s101 end) → R-s202.
6. **Most important.** Re-forecast hourly from round times; act at the first free slot per D93.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (b9sm9buty).

## Review 34 — 2026-11-08 12:05Z
1. **Changes.** Monitor 11084694: R-s101 64, U-s101 64, H-s101 44. Round-64 evaluations ran at 12,288 (D94).
2. **Valid?** Yes; caps verified in rollout, training, grading and evaluation for s101 runs.
3. **Surprises.** Dev truncation at 12,288 at round 64 is 15–20%, higher than the a1 study suggested (report-only).
4. **Risks.** Deadline plan D93 stands; projections unchanged within ±1 h.
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued. First slot ≈ 22:20Z → R-s202 (D93).
6. **Most important.** Act at the first free general slot per D93; keep re-forecasting.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (b9sm9buty).

## Review 35 — 2026-11-08 13:05Z
1. **Changes.** None in job state. s101 runs continue on general (R 10:04, U 9:02, H 8:47 elapsed).
2. **Valid?** Yes; no restarts.
3. **Surprises.** None.
4. **Risks.** Unchanged (D93/D94): first general slot ≈ 22:20Z; H-s202 conditional.
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued.
6. **Most important.** On the U-s101 completion event: validate its endpoint, move R-s202 (11081729) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (b9sm9buty); main-run completions are not suppressed.

## Review 36 — 2026-11-08 14:05Z
1. **Changes.** Monitor 11085378: U-s101 76, R-s101 72, H-s101 52.
2. **Valid?** Yes; no restarts.
3. **Surprises.** Round times keep growing (U 615 s, R 808 s, H 965 s mean of last 6).
4. **Risks.** First slot slips to ≈ 9 Nov 00:40Z; margin to deadline ≈ 6 h under D93 (D95). H-s202 does not fit on general.
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued.
6. **Most important.** On U-s101 completion: validate endpoint, move R-s202 to general.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (b9sm9buty).

## Review 37 — 2026-11-08 15:05Z (10:05 AM EST)
1. **Changes.** A12: deadline Tue 10 Nov 11:00 AM EST (16:00Z) (D96; plan D93 kept). Test-evaluation time limit raised to 9 h (D97).
2. **Valid?** Yes; s101 runs continuous on general (R 12:04, U 11:02, H 10:47 elapsed).
3. **Surprises.** None.
4. **Risks.** Test evaluations may take ≈ 5 h each; all done ≈ Tue 4:30 AM EST; margin ≈ 6 h.
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued.
6. **Most important.** On U-s101 completion (≈ 7:40 PM EST): validate endpoint, move R-s202 to general.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (b9sm9buty).

## Review 38 — 2026-11-08 16:05Z (11:05 AM EST)
1. **Changes.** Monitor 11086128: U-s101 85, R-s101 81, H-s101 61; checkpoints at 80/84/60 written ok.
2. **Valid?** Yes; caps 12,288 in rounds; no restarts; capped per round U 13, R 16, H 6 (report-only, A8).
3. **Surprises.** H rounds got faster (806 s vs 965 s); U rounds slower (744 s vs 615 s).
4. **Risks.** U-s101 end slips to ≈ 9:40 PM EST; last test evaluation ≈ Tue 5:40 AM EST; margin ≈ 5 h (D98).
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued (no starts).
6. **Most important.** On U-s101 completion: validate endpoint, move R-s202 (11081729) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs + monitor 11086128 done). Watcher alive (b9sm9buty).

## Review 39 — 2026-11-08 17:05Z (12:05 PM EST)
1. **Changes.** A13 recorded (`project.md` §27). Round-32 dev provenance resolved (D99, job 11086589): earlier values were the superseded a1 trajectory; canonical = a2 (R 0.542, U 0.601, H 0.540).
2. **Valid?** Yes; s101 runs continuous on general (R 14:04, U 13:02, H 12:47 elapsed); no restarts.
3. **Surprises.** The A11 replay is not bitwise reproducible from the base model (D99); no effect on validity, recorded for the report.
4. **Risks.** Unchanged from D98: margin ≈ 5 h to Tue 11:00 AM EST.
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued (no starts). Next monitor ≈ 18:05Z.
6. **Most important.** On U-s101 completion (≈ 9:40 PM EST): validate endpoint, move R-s202 (11081729) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (bn30h9a02). Review cron created 7 Nov 02:13Z; expiry ≈ 14 Nov (no action).
