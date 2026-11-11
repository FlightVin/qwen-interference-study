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

## Review 40 — 2026-11-08 18:05Z (1:05 PM EST)
1. **Changes.** Monitor 11087148: U-s101 95, R-s101 90, H-s101 68; H dev at round 64 logged (12,288).
2. **Valid?** Yes; caps 12,288; checkpoints ok; no restarts. Capped per round U 3–17, R 8–16, H 6–12 of 128 (report-only).
3. **Surprises.** None; H rounds faster again (682 s).
4. **Risks.** Margin ≈ 5.5 h (D100); H-s202 still does not fit on general.
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued (no starts).
6. **Most important.** On U-s101 completion (≈ 9:20 PM EST): validate endpoint, move R-s202 (11081729) to general.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs + monitor done). Watcher alive (bn30h9a02).

## Review 41 — 2026-11-08 19:05Z (2:05 PM EST)
1. **Changes.** None in job state (R 16:04, U 15:02, H 14:47 elapsed).
2. **Valid?** Yes; no restarts.
3. **Surprises.** None.
4. **Risks.** Unchanged (D100, margin ≈ 5.5 h).
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued. Next monitor ≈ 20:05Z.
6. **Most important.** Endpoint readiness: `validate_endpoint` now also checks attempt id a2 in metrics and the round_0128 checkpoint (D99); new recipe `qis-validate-endpoints` (all runs with round_0128) for the U-s101 completion event.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs). Watcher alive (bn30h9a02).

## Review 42 — 2026-11-08 20:05Z (3:05 PM EST)
1. **Changes.** Monitor 11088493 (≈20:15Z): U-s101 101, R-s101 96, H-s101 79. Dev round 96 logged for U and R (12,288; monitoring only).
2. **Valid?** Yes; caps 12,288; checkpoints ok; no restarts. Capped per round U 4–18, R 2–17, H 5–10 of 128 (report-only).
3. **Surprises.** None. U and R rounds faster (674 s, 688 s); H slower (758 s).
4. **Risks.** Projected ends: U ≈ 9:10 PM EST, R ≈ 10:10 PM EST, H ≈ 3:10 AM EST Mon; D100 margin (≈ 5.5 h) holds within ±1 h.
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued (no starts).
6. **Most important.** On U-s101 completion: submit `qis-validate-endpoints`, move R-s202 (11081729) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned jobs + monitor done). Watcher alive (bn30h9a02).

## Review 43 — 2026-11-08 21:05Z (4:05 PM EST)
1. **Changes.** None in training jobs (R 18:04, U 17:02, H 16:47 elapsed). `qis-test-base` 11089196 queued on preempt (D101).
2. **Valid?** Yes; no restarts.
3. **Surprises.** None.
4. **Risks.** Unchanged (margin ≈ 5.5 h); D101 can only shorten the test queue.
5. **Next jobs / limits.** General 3/3; preempt 6/6 queued (no starts).
6. **Most important.** On U-s101 completion (≈ 9:10 PM EST): submit `qis-validate-endpoints`, move R-s202 to general.
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned jobs). Watcher alive (bn30h9a02); test completions are not suppressed.

## Review 44 — 2026-11-08 22:05Z (5:05 PM EST)
1. **Changes.** Monitor 11089798 (≈22:13Z): U-s101 111, R-s101 105, H-s101 88.
2. **Valid?** Yes; caps 12,288; checkpoints ok; no restarts. Capped per round U 6–15, R 1–14, H 3–13 of 128 (report-only).
3. **Surprises.** None.
4. **Risks.** Projected ends: U ≈ 9:20 PM EST, R ≈ 10:50 PM EST, H ≈ 2:45 AM EST Mon; D100 margin (≈ 5.5 h) holds.
5. **Next jobs / limits.** General 3/3; preempt 6/6 queued (no starts, incl. `qis-test-base` 11089196).
6. **Most important.** On U-s101 completion: submit `qis-validate-endpoints`, move R-s202 (11081729) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned jobs + monitor done). Watcher alive (bn30h9a02).

## Review 45 — 2026-11-08 23:05Z (6:05 PM EST)
1. **Changes.** None in job state (R 20:04, U 19:02, H 18:47 elapsed).
2. **Valid?** Yes; no restarts.
3. **Surprises.** None.
4. **Risks.** Unchanged (Review 44 projections; margin ≈ 5.5 h).
5. **Next jobs / limits.** General 3/3; preempt 6/6 queued (no starts). Next monitor ≈ 00:05Z.
6. **Most important.** On U-s101 completion (≈ 9:20 PM EST): submit `qis-validate-endpoints`, move R-s202 (11081729) to general.
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned jobs). Watcher alive (bn30h9a02).

## Review 46 — 2026-11-09 00:05Z (7:05 PM EST Sun)
1. **Changes.** Monitor 11091008 (≈00:10Z): U-s101 121, R-s101 116, H-s101 96 (H dev round 96 logged; monitoring only).
2. **Valid?** Yes; caps 12,288; checkpoints ok; no restarts. Capped per round U 3–17, R 2–17, H 3–11 of 128 (report-only).
3. **Surprises.** None.
4. **Risks.** Projected ends: U ≈ 9:25 PM EST, R ≈ 10:15 PM EST, H ≈ 1:55 AM EST Mon; margin ≥ 5.5 h.
5. **Next jobs / limits.** General 3/3; preempt 6/6 queued (no starts).
6. **Most important.** On U-s101 completion: submit `qis-validate-endpoints`, move R-s202 (11081729) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned jobs + monitor done). Watcher alive (bn30h9a02).

## Review 47 — 2026-11-09 01:05Z (8:05 PM EST Sun)
1. **Changes.** None in job state (R 22:04, U 21:02, H 20:47 elapsed).
2. **Valid?** Yes; no restarts.
3. **Surprises.** None.
4. **Risks.** Unchanged (Review 46 projections). R-s202 time limit 40 h fits general's 2-day wall.
5. **Next jobs / limits.** General 3/3; preempt 6/6 queued (no starts).
6. **Most important.** On U-s101 completion (≈ 9:25 PM EST): submit `qis-validate-endpoints`, move R-s202 (11081729) to general.
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned jobs). Watcher alive (bn30h9a02).

## Review 48 — 2026-11-09 02:05Z (9:05 PM EST Sun)
1. **Changes.** Monitor 11091807 (≈02:10Z): U-s101 finished round 128 (final evaluation + blocking checkpoint in progress); R-s101 126; H-s101 104.
2. **Valid?** Yes. R-s101 logged `checkpoint_failed` at rounds 108 (22:42Z) and 120 (01:04Z): `OSError: [Errno 512]` on the /data write (transient filesystem error); 112, 116 and 124 then wrote ok (sha-verified). Training is unaffected; only fewer restart points existed for those 4-round windows.
3. **Surprises.** Errno 512 on /data checkpoint writes (first seen in this study).
4. **Risks.** A failed blocking round-128 write makes the run exit 5 with `final_checkpoint_failed`, never `complete` (run.py:378–380); recovery = one recorded transient retry resuming from round 124 (≈ 1 h). Endpoint validation checks round_0128 sha.
5. **Next jobs / limits.** General 3/3; preempt 6/6 queued (no starts). H-s101 projected ≈ 3:10 AM EST Mon.
6. **Most important.** On U-s101 completion: submit `qis-validate-endpoints`, move R-s202 (11081729) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned jobs + monitor done). Watcher alive (bn30h9a02).

## Review 49 — 2026-11-09 03:05Z (10:05 PM EST Sun)
1. **Changes.** U-s101 COMPLETED 02:41Z; endpoint validation 11092019 all_pass (attempt a2, round_0128 sha; hard truncation over 128 rounds 984/8,192 = 12.0%, aux 21/8,192 = 0.3%). R-s202 moved to general and running since 02:42Z (D102).
2. **Valid?** Yes; U-s101 endpoint validated; R-s101 in its round-128 evaluation; H-s101 running.
3. **Surprises.** None new (R-s101 Errno 512 noted in Review 48).
4. **Risks.** R-s101 final blocking checkpoint pending (a failure exits 5, recover from round 124).
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued (no starts).
6. **Most important.** On R-s101 completion: validate endpoint; move U-s202 (11081731) to general (D93).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned nonterminal jobs). Watcher alive (b68d3juv3).

## Review 50 — 2026-11-09 04:05Z (11:05 PM EST Sun)
1. **Changes.** R-s101 finished round 128 and its round-128 evaluations (dev 2,835 s); its blocking final save is slow, not hung: `round_0128.tmp` model.pt complete 03:51Z, optim.pt full size (13.76 GB) 04:12Z (job 11092886 listing); sha read-back and rename pending. Monitor 11092858: H-s101 112, R-s202 12.
2. **Valid?** Yes so far. /data writes are slow tonight: R-s202 round-4 save failed with Errno 512 after 36 min (no R-s202 checkpoint yet; round 8 save in progress). Failed saves left partial dirs: R-s101 round_0108.tmp (19.5 GB), round_0120.tmp (18.8 GB), R-s202 round_0004.tmp (21.9 GB) ≈ 60 GB; not deleted (no cleanup rule triggered; space was 195/500 GB).
3. **Surprises.** My `srun --overlap` diagnostic into 11081724 printed nothing in 90 s and was killed by `timeout` (its own step .0 cancelled; batch step unaffected, sacct verified). Rule: inspect /data from a separate CPU job, not by srun into a training job.
4. **Risks.** If R-s101's final save fails, the job exits 5; recovery resumes from round_0124 (≈ 1 h + evals). Slow /data also lengthens H-s101 and R-s202 saves (asynchronous for non-final rounds).
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued (no starts).
6. **Most important.** On R-s101 completion: validate endpoint; move U-s202 (11081731) to general.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned nonterminal jobs; monitor and inspect jobs done). Watcher alive (b68d3juv3).

## Review 51 — 2026-11-09 05:05Z (12:05 AM EST Mon)
1. **Changes.** R-s101 11081724 FAILED at the final save (Errno 512); records preserved; retry 11092945 resumed at round 124 on general (D103). Monitor 11093383: retry at 125, H-s101 115, R-s202 12.
2. **Valid?** Yes; the retry continues the retained a2 trajectory from a sha-verified checkpoint; superseded round-125–128 records kept.
3. **Surprises.** /data writes ≈ 6 MB/s (22 GB in 55–75 min); GPU idles while a save waits for the previous one (R-s202 ≈ 57 min).
4. **Risks.** R-s202 cannot meet the deadline if stalls persist (D104).
5. **Next jobs / limits.** General 3/3; preempt 5/6 queued (no starts).
6. **Most important.** Decide the checkpoint-cadence change on the `qis-ckpt-watch` 11093453 result (D104).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned nonterminal jobs + ckpt-watch). Watcher alive (bm5hr0tck).

## Review 52 — 2026-11-09 06:05Z (1:05 AM EST Mon)
1. **Changes.** D104 trigger met → D105: R-s202 restarted from round 16 with `ckpt_every` 16 (11094190, general); U-s202 resubmitted with 16 (11094037, preempt queue). R-s101 retry 11092945 replaying rounds 125–128.
2. **Valid?** Yes; both seed-202 recipes generated (not hand-edited), default regeneration verified identical; superseded records preserved.
3. **Surprises.** R-s202 round-12 save failed after ≈ 51 min (Errno 512 pattern); round 16 committed in ≈ 14 min (/data slightly better).
4. **Risks.** R-s101 retry's final save under slow /data (no further automatic retry, D103). R-s202 restart cost ≈ 3 rounds + load.
5. **Next jobs / limits.** General 3/3 (H-s101, R-s101 retry, R-s202); preempt 5/6 queued.
6. **Most important.** On R-s101 retry completion: validate; move U-s202 11094037 to general.
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned nonterminal jobs). Watcher alive (bjphtsetd).

## Review 53 — 2026-11-09 07:14Z (2:14 AM EST Mon)
1. **Changes.** R-s101 retry 11092945 COMPLETED 07:01:52Z (0:0, 2:34:37; final save succeeded). U-s202 11094037 moved to general, running since ≈07:04Z (ckpt_every 16). Endpoint validation 11094459 running (re-hashes 22 GB checkpoints on slow /data).
2. **Valid?** R-s101 pending validation; U-s101 validated.
3. **Surprises.** None new.
4. **Risks.** H-s101 final save under slow /data (blocking; failure → exit 5, retry from round 124/128-4 per D103 rule: one retry only for R-s101; H-s101 would get its own one recorded retry).
5. **Next jobs / limits.** General 3/3 (H-s101, R-s202, U-s202); preempt 5/6 queued. On H-s101 end: test evaluations on its general slot (`qis-test-base` first unless it started on preempt).
6. **Most important.** Confirm R-s101 endpoint validation (attempt a2, round_0128 sha, rounds 125–128 from the retry).
7. **Reconciliation.** Slurm ↔ ledger agree (8 owned nonterminal jobs). Watcher alive (bvtqg1k0t).
- 07:22Z addendum: R-s101 endpoint validation 11094459 all_pass (attempt a2 in metrics and round_0128; sha match; 132 round rows = 128 distinct; hard truncation 777/8,192 = 9.5%, aux 126/8,192 = 1.5%). Evidence `research/evidence/endpoints/endpoint_qis-main-R-s101-sym.json`.

## Review 54 — 2026-11-09 08:05Z (3:05 AM EST Mon)
1. **Changes.** Monitor 11094805: H-s101 127; R-s202 33 (12,288 from round 33); U-s202 19.
2. **Valid?** Yes; R-s101 and U-s101 endpoints validated; R-s202 resume at 16 confirmed (`resume` event).
3. **Surprises.** H-s101 round-120 save failed (Errno 512); round 124 committed.
4. **Risks.** Margin ≈ 3 h (D106). H-s101 final save under slow /data.
5. **Next jobs / limits.** General 3/3; preempt 4/6 queued (no starts).
6. **Most important.** On H-s101 completion: validate; start test evaluations on its slot. Capcheck for seed-202 after both pass round 35.
7. **Reconciliation.** Slurm ↔ ledger agree (7 owned nonterminal jobs). Watcher alive (bvtqg1k0t).

## Review 55 — 2026-11-09 09:05Z (4:05 AM EST Mon)
1. **Changes.** No training state change. Test evaluations R-s101 11095106 and U-s101 11095118 queued on preempt (D107).
2. **Valid?** Yes.
3. **Surprises.** None.
4. **Risks.** Margin ≈ 3 h (D106).
5. **Next jobs / limits.** General 3/3; preempt 6/6 owned (no starts).
6. **Most important.** On H-s101 completion: validate; move test R-s101 to general (D107).
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned nonterminal jobs). Watcher alive (bvtqg1k0t).

## Review 56 — 2026-11-09 10:05Z (5:05 AM EST Mon)
1. **Changes.** H-s101 11081726 COMPLETED 09:48:02Z (0:0, 1-05:29:32). Test R-s101 11095106 moved to general (D107); precheck ok (sealed test hash OK; model.pt sha recorded = on-disk aedcca6e…); eval running since 09:53Z on babel-w5-24. H-s101 endpoint validation 11095270 running.
2. **Valid?** Yes; all three s101 runs completed; R and U validated.
3. **Surprises.** None.
4. **Risks.** Margin ≈ 3 h (D106).
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202, test R-s101); preempt 6/6 owned.
6. **Most important.** On test R-s101 end (≈ 15:00Z): move test U-s101 to general.
7. **Reconciliation.** Slurm ↔ ledger agree (9 owned nonterminal jobs). Watcher alive (b2b9jx0s7).

## Review 57 — 2026-11-09 11:05Z (6:05 AM EST Mon)
1. **Changes.** H-s101 endpoint validation 11095270 all_pass (128 rounds, 1 start; hard truncation 1,140/16,384 = 7.0%). Test H-s101 queued 11095486 (preempt; last in D107 order). Capcheck for seed 202 submitted (`qis-a11-capcheck-s202` 11095700; waits for round 36 in both).
2. **Valid?** Yes; all s101 endpoints validated; test R-s101 running (1:14 elapsed).
3. **Surprises.** None.
4. **Risks.** Margin ≈ 3 h (D106).
5. **Next jobs / limits.** General 3/3; preempt 6/6 owned.
6. **Most important.** On test R-s101 end (≈ 10 AM EST): move test U-s101 to general.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (b2b9jx0s7).

## Review 58 — 2026-11-09 12:15Z (7:15 AM EST Mon)
1. **Changes.** Monitor 11095848: R-s202 at round 62, U-s202 at round 51; both on cap 12,288; checkpoint 48 committed in both (19 and 22 min). Test R-s101 running (2:24 elapsed).
2. **Valid?** Yes. Last-6 rounds: no failures; truncation R 26/384, U 34/384 hard-and-practice samples; clip fractions ≤ 3e-4; ratio quantiles within [0.88, 1.13].
3. **Surprises.** Round times 467 s (R) and 548 s (U), faster than D106's 720–780 s.
4. **Risks.** Margin now ≈ 8–13 h (D108). Remaining risk: /data save failures (D104); each run has 5 more saves.
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202, test R-s101); preempt 6/6 owned, all pending.
6. **Most important.** On test R-s101 end (≈ 10 AM EST): move test U-s101 to general (D107).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (ber64mxk3).

## Review 59 — 2026-11-09 13:05Z (8:05 AM EST Mon)
1. **Changes.** None since Review 58. R-s202 and U-s202 training on general; test R-s101 running (3:14 elapsed of a 9 h limit).
2. **Valid?** Yes; no new failures (the FAILED rows in `ae status` are old, already recorded jobs).
3. **Surprises.** None.
4. **Risks.** Unchanged (D108 margin ≈ 8–13 h; /data save risk, D104).
5. **Next jobs / limits.** General 3/3; preempt 6/6 owned, all pending.
6. **Most important.** On test R-s101 end (≈ 10 AM EST): move test U-s101 to general (D107). Next monitor ≈ 14:15Z.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (ber64mxk3, no events).

## Review 60 — 2026-11-09 14:10Z (9:10 AM EST Mon)
1. **Changes.** Monitor 11096244: R-s202 completed round 71 (round-64 evaluations logged; checkpoint 64 committed); U-s202 completed round 64 (its round-64 evaluations are next). Test R-s101 running (4:19 elapsed).
2. **Valid?** Yes. Last 6 rounds: mean 477 s (R) and 508 s (U); truncation R 31/384, U 42/384; clip fraction ≤ 2.4e-4; ratio quantiles within [0.87, 1.13]; no checkpoint stall.
3. **Surprises.** None. Round-64 dev values not read for any decision (logged only).
4. **Risks.** On track inside D108: R-s202 ≈ 56 rounds left → ends ≈ 22:00Z; U-s202 ≈ 63 left → ends ≈ 00:00–01:00Z Tue. /data save risk remains (D104).
5. **Next jobs / limits.** General 3/3; preempt 6/6 owned, all pending.
6. **Most important.** On test R-s101 end (≈ 10 AM EST): move test U-s101 to general (D107).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (ber64mxk3, no events).

## Review 61 — 2026-11-09 15:05Z (10:05 AM EST Mon)
1. **Changes.** Test R-s101 11095106 COMPLETED 14:14Z (4:23). Test U-s101 11095118 moved to general (D109); started 14:16Z on babel-q5-20; precheck ok (model sha 92aef0d2… recorded = on-disk); eval phase running.
2. **Valid?** Yes. Test outcomes not read (A13).
3. **Surprises.** None.
4. **Risks.** Unchanged (D108 margin ≈ 8–13 h; /data save risk, D104). Test U-s101 ends ≈ 18:45Z; then base, then H-s101 (≈ 4.5 h each) → last s101-slot test ≈ 10 Nov 04:00Z.
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202, test U-s101); preempt 5/6 owned, all pending.
6. **Most important.** On test U-s101 end (≈ 1:45 PM EST): move test-base 11089196 to general (D107). Next monitor ≈ 16:10Z.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (b93cczsj4, no events).

## Review 62 — 2026-11-09 16:40Z (11:40 AM EST Mon; delayed ≈ 30 min by a laptop network outage)
1. **Changes.** Monitor 11097078: R-s202 completed round 85, U-s202 round 74; both logged round-64 evaluations. R-s202 round-80 save failed (Errno 512, D110); U-s202 checkpoint 64 committed. Test U-s101 running (2:24 elapsed).
2. **Valid?** Yes. Last 6 rounds: mean 505 s (R), 543 s (U); truncation R 40/384, U 48/384; clip fraction ≤ 4.9e-4; ratio quantiles within [0.86, 1.15].
3. **Surprises.** Errno 512 recurred (first since D105).
4. **Risks.** R-s202 last valid checkpoint is round 64; final-save failure would cost ≈ 3 h (D110). Forecast: R ≈ 42 rounds left → ends ≈ 23:00Z; U ≈ 53 left → ends ≈ 01:00Z Tue. Inside D108.
5. **Next jobs / limits.** General 3/3; preempt 5/6 owned, all pending.
6. **Most important.** Watch R-s202 round-96 save (next monitor ≈ 18:15Z). On test U-s101 end (≈ 1:45 PM EST): move test-base to general (D107).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (b93cczsj4, no events).

## Review 63 — 2026-11-09 17:05Z (12:05 PM EST Mon)
1. **Changes.** None in Slurm since Review 62. STATE "Next actions" rewritten (stale items: s202 capcheck done; s101 runs done; D108–D110 state).
2. **Valid?** Yes. General 3/3 in use (R-s202, U-s202, test U-s101; checked 16:46Z).
3. **Surprises.** None.
4. **Risks.** R-s202 last valid checkpoint is round 64 (D110); round-96 save pending.
5. **Next jobs / limits.** General 3/3; preempt 5/6 owned, all pending.
6. **Most important.** Monitor ≈ 18:15Z to confirm R-s202 round-96 save. On test U-s101 end (≈ 18:45Z): move test-base to general (D107).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (b93cczsj4, PID 77279; DNS errors during the 16:00Z outage, recovered).

## Review 64 — 2026-11-09 18:10Z (1:10 PM EST Mon)
1. **Changes.** U-s202 11094037 FAILED 17:40Z (Errno 512 on metrics write); recover-prep 11097803 done; retry 11097811 started 17:45Z on general (D111). Monitor 11098014 (≈18:08Z): R-s202 completed round 96; its round-96 save and evaluations not yet logged. U-s202 retry: no `start`/`resume` row yet (checkpoint load ≈ 25 min expected; latest rows are the failed attempt's).
2. **Valid?** Yes. R-s202 last 6 rounds: mean 575 s; truncation 50/384; clip ≤ 4.3e-4; ratio quantiles within [0.85, 1.14].
3. **Surprises.** Errno 512 now also hits metrics writes (not only checkpoint writes): a write fault can kill a run at any step.
4. **Risks.** R-s202: last valid checkpoint round 64; round-96 save in progress. Forecast R-s202 ≈ 32 rounds × 575 s + 2 evaluations → ends ≈ 00:00–01:00Z Tue; U-s202 ≈ 03:30–05:00Z Tue (D111). U-s202 has used its one retry.
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202 retry, test U-s101); preempt 5/6 owned, all pending.
6. **Most important.** Confirm U-s202 resume at round 64 and R-s202 round-96 commit (monitor ≈ 19:10Z). On test U-s101 end (≈ 18:45Z): move test-base to general (D107).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bx36np6gr).

## Review 65 — 2026-11-09 19:10Z (2:10 PM EST Mon)
1. **Changes.** Test U-s101 11095118 COMPLETED 18:40Z; test-base 11089196 moved to general (D112), started 18:42Z, precheck ok. Monitor 11098508 (≈19:08Z): U-s202 retry 11097811 `start` + `resume` round 64 at 18:10Z (load ≈ 24 min); now at round 70. R-s202 at round 97; round-96 evaluations logged; round-96 save started 18:28Z and not committed after ≈ 40 min.
2. **Valid?** Yes. Last 6 rounds: R mean 665 s (includes round-96 evaluations), truncation 47/384; U mean 557 s, truncation 54/384; clip ≤ 4.3e-4; ratio quantiles within [0.85, 1.14].
3. **Surprises.** None new; /data writes slow again (round-96 save ≥ 40 min).
4. **Risks.** R-s202 last valid checkpoint still round 64 until round 96 commits. Forecast: R-s202 ≈ 31 rounds → ends ≈ 00:45–01:45Z Tue; U-s202 ≈ 58 rounds → ends ≈ 03:30–05:00Z Tue (D111). Last test ≈ 08:00–10:00Z. Margin ≈ 6–8 h.
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202, test-base); preempt 4/6 owned, all pending.
6. **Most important.** Confirm R-s202 round-96 commit at next monitor (≈ 21:10Z). On test-base end (≈ 23:15Z): move test H-s101 to general (D107).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (brhzz0nec).

## Review 66 — 2026-11-09 20:05Z (3:05 PM EST Mon)
1. **Changes.** Test-base 11089196 COMPLETED 19:12Z (30 min; structure verified, D113). Test H-s101 11095486 moved to general, started 19:15Z, precheck ok (model sha 052c8a62… recorded = on-disk). All four s101-slot tests now done or running.
2. **Valid?** Yes. Test outcome fields unread (A13); only structural fields (completions 4,096, cap 12,288, tokens, truncation) checked.
3. **Surprises.** Base test ran 30 min vs ≈ 4.4 h for trained models (explained in D113: ≈ 3× fewer tokens, higher throughput).
4. **Risks.** R-s202 round-96 save status unknown since 19:08Z (next monitor ≈ 21:10Z). U-s202 retry used.
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202, test H-s101); preempt 3/6 owned, all pending (H-s202, diag r0, diag r64).
6. **Most important.** Report pipeline (CHECKLIST 7.2–7.4) is the critical path once training is stable: build tables, figures and cost reconciliation from non-sealed inputs now, so only the test rows remain at unsealing.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (b8stf9lw7, no events).

## Review 67 — 2026-11-09 21:05Z–22:00Z (4:05–5:00 PM EST Mon)
1. **Changes.** Monitor 11099614: R-s202 round 108; its round-96 save failed (Errno 512, 19:44Z). Storage check found the account quota at 495/500 GB — root cause of all Errno 512 failures (D116). Prunes freed ≥ 191 GB (D116, D117); quota now ≤ 366 GB. U-s202 retry 11097811 FAILED 21:31Z (EINTR, same mechanism); resumed from round 64 as 11100159 (D117). Diagnostics resubmitted resumable (D114). Report pipeline dry run passed (D115).
2. **Valid?** Yes for committed checkpoints and records (sha256-checked); failed writes left only `.tmp` dirs (deleted, logged).
3. **Surprises.** Quota exhaustion was not detected earlier because monitors checked project usage, not account free space (rule added to the harness log).
4. **Risks.** U-s202 now ends ≈ 09:30Z and its test ≈ 14:00Z (2 h before the deadline). R-s202 last valid checkpoint still round 64 until round 112 commits.
5. **Next jobs / limits.** General 3/3 once U-s202 starts (R-s202, test H-s101, U-s202); preempt: H-s202, diag r0, diag r64 pending.
6. **Most important.** Confirm U-s202 resumes and R-s202's round-112 save commits; move diag r64 to general when test H-s101 ends (≈ 23:45Z).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bgybgcize).

## Review 68 — 2026-11-09 22:10Z (5:10 PM EST Mon)
1. **Changes.** Monitor 11100269: R-s202 round-112 checkpoint committed 21:56Z in 276 s (vs 815 s at round 64 and failures at 80/96) — the quota fix (D116/D117) restored write speed. R-s202 now at round 113; last valid checkpoint 112. U-s202 11100159 `start` + `resume` round 64 at 21:55Z (checkpoint load < 1 min vs ≈ 24 min before).
2. **Valid?** Yes. R-s202 last 6 rounds: mean 643 s, truncation 62/384, clip ≤ 3.5e-4, ratio within [0.87, 1.14]. U-s202 rows shown are the failed attempt's until new rounds log.
3. **Surprises.** None; the faster load and save confirm the storage diagnosis.
4. **Risks.** U-s202 is the critical path: 64 rounds × ≈ 600 s + 2 evaluations → ends ≈ 09:00Z Tue; test ≈ 13:30Z. R-s202 ends ≈ 01:00–01:30Z.
5. **Next jobs / limits.** General 3/3; preempt: H-s202, diag r0, diag r64 pending. Quota ≤ 366/500 GB.
6. **Most important.** Move diag r64 to general when test H-s101 ends (≈ 23:45Z); validate R-s202's endpoint and start its test when it ends.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bgybgcize; one ssh timeout logged, recovered).

## Review 69 — 2026-11-09 23:05Z (6:05 PM EST Mon)
1. **Changes.** No job state change since Review 68. Seed-202 tests queued behind their runs on general (D118). Suite 47/47 (11100305); REPORT.md draft committed.
2. **Valid?** Yes.
3. **Surprises.** None.
4. **Risks.** U-s202 critical path (ends ≈ 09:00Z, test ≈ 13:30Z). Quota ≤ 366/500 GB.
5. **Next jobs / limits.** General: R-s202, U-s202, test H-s101 running; tests R/U-s202 pending on dependencies. Preempt: H-s202, diag r0, diag r64 pending.
6. **Most important.** Move diag r64 to general when test H-s101 ends (≈ 23:45Z).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bgybgcize, PID 3025).

## Review 70 — 2026-11-10 00:05Z (7:05 PM EST Mon)
1. **Changes.** A14: deadline 19:00Z (D119). Test H-s101 done 23:42Z (D120); diag r64 11099088 running on general since 23:48Z (precheck: source attempt a2; tests 4/4). Monitor 11101332: R-s202 at round 125 (checkpoint 112 valid); U-s202 at round 77 after resuming at 64.
2. **Valid?** Yes. Last 6 rounds: R mean 629 s, truncation 54/384; U mean 641 s, truncation 76/384; clip ≤ 4.2e-4; ratio within [0.87, 1.14].
3. **Surprises.** None.
4. **Risks.** U-s202: 51 rounds left × ≈ 640 s + 2 evaluations → ends ≈ 09:45Z; test ≈ 14:15Z (D119 limit 17:00Z). R-s202 ends ≈ 00:45–01:15Z; its test then starts automatically (D118).
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202, diag r64); tests R/U-s202 pending on dependencies; preempt: H-s202, diag r0.
6. **Most important.** When R-s202 ends: run endpoint validation (CPU) alongside its test; then put diag r0 on that slot after the test.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bpnb6ehp9).

## Review 71 — 2026-11-10 01:05Z (8:05 PM EST Mon)
1. **Changes.** Diag r64 11099088 FAILED 00:58Z (OOM in snapshot restore: code defect, fixed in 4b47386, D121); resubmitted 11101908 (running on general since 01:01Z, skips the stored no-update row); pending r0 replaced by 11101910 (same fix).
2. **Valid?** Yes. The r64 no-update row from the failed job is complete and is reused; nothing else from that job is used.
3. **Surprises.** A latent defect in a never-GPU-tested path; caught by its first GPU run.
4. **Risks.** Diag r64 now ends ≈ 10:00–12:00Z (partial seed sets acceptable, A9.5). R-s202 in its final round/evaluation/save (12 rounds since 00:10Z monitor → ends ≈ 01:15–01:45Z). U-s202 ends ≈ 09:45Z.
5. **Next jobs / limits.** General 3/3 (R-s202, U-s202, diag r64); test R-s202 starts on R-s202's GPU (dependency). Preempt: H-s202, diag r0.
6. **Most important.** Confirm diag r64 passes its first branch (background check); on R-s202 end, validate its endpoint (CPU) while its test runs.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bbha8sifn).

## Review 72 — 2026-11-10 02:05Z (9:05 PM EST Mon)
1. **Changes.** A15 recorded (D122). R-s202 COMPLETED 01:30Z (19:11 h); test R-s202 running since 01:31Z. Diag r64 11101908 passed its first branch (≈ 40 min per branch). Cue panel frozen (D123). Exploratory code committed (per_problem, interference). Diag r0 briefly ran as a 4th general job (99 s; D124) — cancelled and chained.
2. **Valid?** Yes. R-s202 endpoint validation 11102275 and the interference job 11102391 are waiting in the CPU queue (86 pending CPU jobs on preempt_cpu_qos).
3. **Surprises.** The normal QOS did not enforce the owner's 3-GPU ceiling (D124).
4. **Risks.** CPU queue delay for analysis jobs. U-s202 ends ≈ 09:45Z; GPU chain ends ≈ 14:00Z; deadline 00:30Z Wed.
5. **Next jobs / limits.** General 3/3; chains: A test R-s202 → diag r0 → cue; B U-s202 → test U-s202; C diag r64.
6. **Most important.** Get R-s202 validated (CPU queue); review interference output when it lands.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree after D124. Watcher alive (b4exdi1r0).

## Review 73 — 2026-11-10 03:05–03:50Z (10:05 PM EST Mon)
1. **Changes.** Monitor 11102871 (03:48Z): U-s202 at round 96; checkpoint 80 committed 00:31Z in 127 s; checkpoint 96 writing. Account quota 292/500 GB (209 GB free; floor 50 GiB). Interim interference trace recorded (D125). Validation rerun 11102674 (validator fixed for retired files, 1125a1e) waiting in the CPU queue.
2. **Valid?** Yes. U-s202 last 6 rounds: mean 620 s, truncation 60/384, clip ≤ 3.8e-4, ratio within [0.89, 1.12].
3. **Surprises.** The monitor had written `df` to `storage.txt` all along; the wait script did not fetch it. Now it prints it every pass.
4. **Risks.** U-s202 ends ≈ 09:30Z (32 rounds × 620 s + evaluation + final save); test ≈ 14:00Z. CPU-queue delays (≈ 1 h) for analysis jobs.
5. **Next jobs / limits.** General 3/3 with chains (D124). Preempt: H-s202, validation.
6. **Most important.** R-s202 endpoint validation result; prepare the post-training analysis job (report_tables, per_problem, report_data, figures, cost) so it runs as soon as U-s202 ends.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (b2lujg5q2).

## Review 74 — 2026-11-10 04:05Z (11:05 PM EST Mon)
1. **Changes.** Post-training analysis 11103165 queued with the unsealing gate (D126); figures for A15.1/A15.2 added (bea479c). Validation rerun 11102674 running (21 min).
2. **Valid?** Yes; no new failures.
3. **Surprises.** None.
4. **Risks.** Unchanged: U-s202 ends ≈ 09:30Z; test ≈ 14:00Z; deadline 00:30Z Wed.
5. **Next jobs / limits.** General 3/3 (U-s202, test R-s202, diag r64) + chains; preempt: H-s202, report-final (dependency).
6. **Most important.** Read R-s202 endpoint validation when it lands.
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (b2lujg5q2).

## Review 75 — 2026-11-10 05:05Z (12:05 AM EST Tue)
1. **Changes.** D127 (counter bookkeeping offset after resume; fixes 0151edc); report-final replaced at HEAD (11103418, afterany U-s202). A16: partial H-s202 chained after diag r64 on general (D128). Validation rerun 11103416 running.
2. **Valid?** Yes; training state unaffected by D127.
3. **Surprises.** None new.
4. **Risks.** Unchanged: U-s202 ends ≈ 09:30Z; test ≈ 14:00Z; deadline 00:30Z Wed.
5. **Next jobs / limits.** General 3/3 running (U-s202, test R-s202, diag r64); chains: A → diag r0 → cue; B → test U-s202; C → H-s202 (partial, stop by 19:00Z).
6. **Most important.** R-s202 validation result; then test R-s202 end (≈ 06:00Z).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bw72lypja).

## Review 76 — 2026-11-10 06:05Z (1:05 AM EST Tue)
1. **Changes.** R-s202 endpoint all_pass (11103416; D127 result). Test R-s202 11100782 COMPLETED 06:04Z (4:33): structure only read — 128 × 32 = 4,096 completions, cap 12,288, test-final, 19.4 M tokens, truncated 13.1%. Diag r0 11102499 started 06:05Z on that GPU (chain A); precheck ok.
2. **Valid?** Yes. Outcomes unread (A13).
3. **Surprises.** None.
4. **Risks.** U-s202 ends ≈ 09:30Z; test ≈ 14:00Z. Diag r64 at 5 h running (expected end ≈ 09:00–10:00Z).
5. **Next jobs / limits.** General 3/3 (U-s202, diag r64, diag r0); chains: A → cue; B → test U-s202; C → H-s202 partial.
6. **Most important.** U-s202 end → validation + analysis job (queued).
7. **Reconciliation.** Slurm ↔ ledger ↔ registry agree. Watcher alive (bw72lypja).

## Review 77 — 2026-11-10 07:10Z (2:10 AM EST Tue)
1. **Changes.** U-s202 finished round 114 of 128 at 07:08Z (checkpoint round 112 committed in 115 s); ends ≈ 09:10Z. Diag r64 at 6 h, diag r0 at 1 h, both running. Storage 207 GB free. Cue analysis module written (6d4e33a; suite 11104737 50/50); pending cue job 11102440 replaced by 11104738 with the same dependencies (D129).
2. **Valid?** Yes. No outcome read.
3. **Surprises.** None.
4. **Risks.** Unchanged: test U-s202 ends ≈ 13:40Z; deadline 00:30Z Wed.
5. **Next jobs / limits.** General 3/3 (U-s202, diag r64, diag r0); chains: A → cue 11104738; B → test U-s202 11100785; C → H-s202 partial. Preempt CPU: report-final 11103418.
6. **Most important.** U-s202 end → endpoint validation and report-final.
7. **Reconciliation.** Slurm ↔ ledger agree (11102440 CANCELLED by me, in ledger, 0 GPU-h). Watcher re-armed (bw7mm9jmc).

## Review 78 — 2026-11-10 08:05Z (3:05 AM EST Tue)
1. **Changes.** Diag r0: 11 of 12 seed-branch rows done (last ends ≈ 08:20Z). Diag r64: seeds 4101–4103 done (9 rows); seed 4104 left, ends ≈ 10:30–11:00Z. U-s202 on schedule (ends ≈ 09:10Z).
2. **Valid?** Yes. J values seen in logs only; analysis waits for the full JSONL.
3. **Surprises.** None.
4. **Risks.** One general GPU would have idled ≈ 50 min after diag r0. Fixed (D130).
5. **Next jobs / limits.** Chains now: A diag r0 → H-s202 partial; B U-s202 → test U-s202; C diag r64 → cue. General 3/3.
6. **Most important.** Keep GPUs busy: chain swap done (D130). Next: U-s202 end → validation, report-final.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (bw7mm9jmc).
- 08:25Z addendum: diag r0 complete (13 rows). All update rows lower J_R by 0.020–0.040 vs "none"; reference check 11105174 queued (D131). H-s202 partial started 08:15Z.

## Review 79 — 2026-11-10 09:05Z (4:05 AM EST Tue)
1. **Changes.** Diag r0 done (D131). H-s202 partial running since 08:15Z. Diag r64: 10 of 13 rows (seed 4104 hard done at 08:51Z); ends ≈ 10:30Z. Suite 11105173 50/50 on 042cebd (reference mode).
2. **Valid?** Yes. Diag r0 J_R shift under check (D131).
3. **Surprises.** Uniform J_R drop in all r0 update rows (D131).
4. **Risks.** U-s202 round 128 + final evaluations: ends ≈ 09:30–09:45Z, slightly later than the 09:10Z forecast (final dev and probe evaluations not counted).
5. **Next jobs / limits.** General 3/3 (U-s202, diag r64, H-s202). Chains: B → test U-s202; C → cue → reference check. Preempt CPU: report-final.
6. **Most important.** U-s202 end → endpoint validation; report-final unseals.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (bzq5mh36x).

## Review 80 — 2026-11-10 10:05Z (5:05 AM EST Tue)
1. **Changes.** U-s202 COMPLETED 09:10Z; all five endpoints all_pass (11105454; D132). Sealed outcomes read 09:45Z for five tests (D133). Figure fix fb4f344 (partial runs out of main figures; A16 figure); suite 51/51. Diag r64 12 of 13 rows.
2. **Valid?** Yes. Report gated on validation (afterok).
3. **Surprises.** R-s202 test 0.071 below R-s101; seed-202 dev at r128 favours U (0.703 vs 0.656). Recorded, no action.
4. **Risks.** Seed 202 may reverse R − U; the report must lead with per-seed values (A13). Test U-s202 ends ≈ 13:40Z.
5. **Next jobs / limits.** General 3/3 (test U-s202, H-s202, diag r64); chain C → cue → reference check.
6. **Most important.** Draft REPORT.md structure now so the final numbers drop in after 13:40Z.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (bgpfn8u2p).

## Review 81 — 2026-11-10 11:05Z (6:05 AM EST Tue)
1. **Changes.** Diag r64 done (D134). Cue 11104738 started 10:22Z: base and R-s101 forms done; U-s101 checkpoint hash verified. REPORT section 3 r64 table added.
2. **Valid?** Yes. Checkpoint identity checked by sha256 in the cue job.
3. **Surprises.** Three r64 update draws had zero gradient (all practice groups all-correct); used as momentum-only reference (D134).
4. **Risks.** None new. Test U-s202 ends ≈ 13:40Z; reference check ≈ 14:30Z; H-s202 stop 19:00Z; deadline 00:30Z.
5. **Next jobs / limits.** General 3/3 (cue, test U-s202, H-s202); chain C → reference check.
6. **Most important.** Wait for test U-s202 → final report-final; meanwhile cost table prep.
7. **Reconciliation.** Slurm ↔ ledger agree. Watcher alive (buy5w20ki).

## Review 82 — 2026-11-10 12:05Z (7:05 AM EST Tue)
1. **Changes.** Interim cost 220.8 GPU-h over 205 allocations (11106097). Cue 11104738 loaded its 6th of 10 policy-forms at 11:51Z (≈ 16 min each). Monitor 11106458 submitted for an H-s202 progress snapshot.
2. **Valid?** Yes. Each cue policy load passed its weight check.
3. **Surprises.** None.
4. **Risks.** Cue ends ≈ 13:00Z, not 12:35Z (16 min per form). Reference check then ends ≈ 15:00Z. Still well before the 00:30Z deadline.
5. **Next jobs / limits.** General 3/3 (cue, test U-s202, H-s202); chain cue → diag-ref. Preempt CPU: monitor.
6. **Most important.** Test U-s202 end (≈ 13:40Z) → report-final → seed-202 rows of the lead table.
7. **Reconciliation.** Slurm ↔ ledger agree (4 live jobs, all project-owned). Watcher alive (bthhtqk89).
- Addendum 12:25Z (monitor 11106458): H-s202 has 32 rounds (hard_common 0.516 at r32; ≈ 8–9 min per round at the 8,192 cap). Checkpoints every 4 rounds. Forecast at the 19:00Z stop: about round 70–75 (rounds slow after the 12,288 cap). Storage 184 GB free (was 207 GB at 07:08Z); projected ≥ 150 GB at 19:00Z, above the 50 GiB floor.

## Review 83 — 2026-11-10 13:05Z (8:05 AM EST Tue)
1. **Changes.** Test U-s202 done 12:21Z; report-final 11106560 → lead table final, audited (D136: R − U s101 +0.042, s202 −0.044, mean −0.001). Cue panel done (D137). Diag-ref started 12:24Z (D135); r0 reference done 12:43Z.
2. **Valid?** Yes. Lead numbers recomputed independently; grader consistent on 40 inspected cue responses.
3. **Surprises.** Seeds disagree in sign on the primary contrast. r0 reference: none_cycled = none_rerun exactly (cycle has no effect, D131 rule met), but the fresh rerun on an L40S differs from the original "none" (L40, babel-w5-24) by J_H +0.031, J_R −0.020, J_U −0.014.
4. **Risks.** ΔJ values below ≈ 0.03 are near the evaluation noise floor; REPORT §3 must say so.
5. **Next jobs / limits.** General 2/3 (H-s202, diag-ref). No runnable GPU work waiting.
6. **Most important.** Finish diag-ref r64 (≈ 13:30Z) → write REPORT §3 answer with the reference result.
7. **Reconciliation.** Slurm ↔ ledger agree (2 live jobs, both project-owned). Watcher alive (ba3kivn6h).

## Review 84 — 2026-11-10 14:05Z (9:05 AM EST Tue)
1. **Changes.** Diag-ref r64 none_rerun done 13:41Z; none_cycled running (≈ 14:30Z). REPORT §3 answer written (D138).
2. **Valid?** Yes. Same-node evaluation is exactly repeatable; cross-hardware reruns differ by up to 0.031.
3. **Surprises.** The original r0 "none" was a lucky draw that made all 12 ΔJ_R negative (D138).
4. **Risks.** None new. H-s202 stop 19:00Z; deadline 00:30Z.
5. **Next jobs / limits.** General 2/3 (H-s202, diag-ref). No runnable GPU work waiting.
6. **Most important.** Fill r64 none_cycled; then prepare the 19:00Z steps (stop, A16 appendix, final report, cost).
7. **Reconciliation.** Slurm ↔ ledger agree (2 live jobs). Watcher alive (ba3kivn6h).

## Review 85 — 2026-11-10 15:05Z (10:05 AM EST Tue)
1. **Changes.** Diag-ref COMPLETED 14:33Z (r64 cycle: no effect). REPORT §1–5 final; §4 dev-panel table and evidence-links paragraph added (6cce3f8).
2. **Valid?** Yes.
3. **Surprises.** None.
4. **Risks.** None new. Remaining work waits for the 19:00Z H-s202 stop; deadline 00:30Z leaves ≈ 5 h after it.
5. **Next jobs / limits.** General 1/3 (H-s202). No study GPU work waiting.
6. **Most important.** 19:00Z: cancel 11082192 → interference + report-final → A16 appendix; regenerate and run the cost table.
7. **Reconciliation.** Slurm ↔ ledger agree (1 live job). Watcher alive (bb1hd5lk6).

## Review 86 — 2026-11-10 16:05Z (11:05 AM EST Tue)
1. **Changes.** None since Review 85. Monitor 11107872 submitted for an H-s202 progress check.
2. **Valid?** Yes.
3. **Surprises.** None.
4. **Risks.** None new. 19:00Z stop; deadline 00:30Z.
5. **Next jobs / limits.** General 1/3 (H-s202). Preempt CPU: monitor.
6. **Most important.** Confirm H-s202 is still advancing (stall check); then the 19:00Z steps.
7. **Reconciliation.** Slurm ↔ ledger agree (1 live job + monitor). Watcher alive (bb1hd5lk6).

## Review 87 — 2026-11-10 17:05Z (12:05 PM EST Tue)
1. **Changes.** None in Slurm. Appendix inputs gathered: R-s202 / U-s202 dev pass@1 at rounds 0, 32, 64 = 0.403 / 0.422, 0.599 / 0.611, 0.613 / 0.679.
2. **Valid?** Yes.
3. **Surprises.** Round-0 dev of the same base model: 0.403 (H, R) vs 0.422 (U). Same size as the D138 evaluation noise; the appendix will state it.
4. **Risks.** None new.
5. **Next jobs / limits.** General 1/3 (H-s202).
6. **Most important.** 18:05Z review: check H-s202 progress toward round 64 (D139).
7. **Reconciliation.** Slurm ↔ ledger agree (1 live job). Watcher alive (bb1hd5lk6).

## Review 88 — 2026-11-10 18:05Z (1:05 PM EST Tue)
Opened 18:05Z; closed late at 19:05Z without an entry. Only action: monitor 11108988 submitted (still PENDING in the CPU queue at 19:05Z). No other change.

## Review 89 — 2026-11-10 19:05Z (2:05 PM EST Tue)
1. **Changes.** H-s202 checkpoints at rounds 56 (17:09Z) and 60 (17:57Z); round 64 trained by 18:33Z. Round-64 evaluations running (GPU 100%, read by `srun --overlap` inside 11082192; monitor 11108988 still queued).
2. **Valid?** Yes. Not stalled: GPU busy; evaluation length matches other runs (dev ≈ 34–37 min).
3. **Surprises.** CPU queue wait > 1 h for the monitor.
4. **Risks.** None for the deadline. Dev-64 expected ≈ 19:10Z; D139 bound 20:00Z.
5. **Next jobs / limits.** General 1/3 (H-s202). Wait script `wait_dev64.sh` (2-min poll) exits on dev-64 or at 20:00Z.
6. **Most important.** On dev-64: cancel 11082192 by ID, confirm the allocation ended, then interference + report-final + cost.
7. **Reconciliation.** Slurm ↔ ledger agree (H-s202 running, monitor pending). Watcher alive (bb1hd5lk6).

## Review 90 — 2026-11-10 20:05Z (3:05 PM EST Tue)
1. **Changes.** Report final and audited (0c5fe04); delivered 19:58Z; commits pushed to origin/main (dc2081d) at the owner's request.
2. **Valid?** Yes. Independent audit applied (research/audits/2026-11-10-recompute.md).
3. **Surprises.** None.
4. **Risks.** None. No live project jobs.
5. **Next jobs / limits.** None planned. 0 GPUs in use.
6. **Most important.** Nothing open; wait for owner feedback on the report.
7. **Reconciliation.** Slurm: 0 live project jobs; ledger has no nonterminal rows to watch. Watcher not re-armed (nothing to watch).
- 20:10Z (3:10 PM EST): hourly review cron cancelled at the owner's request; study closed.

## Review 91 — 2026-11-11 02:50Z (9:50 PM EST Tue) — A17 restart
1. **Changes since resume (01:25Z).** A17 recorded (§31). H-s202 resumed from round 60 (job 11112377; hashes verified); rounds 61–64 at
   630–800 s (round 64 ended 02:40Z; round-64 evals running). Original r60–64 records backed up (D143). Common-GPU reruns (D141): R-s101
   11112395, R-s202 11112396 running on general (L40S, 1:08 h); U-s101 11112397 started on preempt; U-s202 11112398 pending. A17 CPU
   analyses done: a17_test (D142) and a17_train (signal, persistence, dev n = 8, branches). Stress panels frozen as SuperHard-126 /
   Harder-95 (D145, after a second-read fix; no evaluation had started). Draft REPORT committed with pending cells.
2. **Competing explanation still plausible.** Length drift: R − U follows which arm truncates more (s202: −4.3 of −4.4 points from
   truncation; accuracy on finished answers equal). Not yet excluded: hardware (R on L40, U on L40S in both seeds).
3. **Distinguishing task.** Common-GPU R reruns (same seeds, L40S): if R − U on L40S keeps both signs and the truncation split,
   hardware is excluded. Training-record length by arm/seed (tokens per response on shared hard prompts) tests whether length drift
   started in training.
4. **Schedule.** H-s202 ≈ 12 min/round → round 128 ≈ 15:30–16:30Z (+ evals) → L40S sealed test ≈ 4.4 h → ≈ 20:30–21:00Z; split test
   on two GPUs if later (D146). R reruns end ≈ 06:00Z; then U-s202 rerun + stress base/endpoints on the two freed general GPUs
   (≈ 2–2.5 GPU-h each, benchmark pending). Report verification ≈ 21:30–00:00Z. Verdict: on time, little slack; optional cells
   (U-s101 rerun now running on preempt, stress r64) drop first.
5. **GPUs.** General 3/3 (H-s202, two R reruns); preempt 1/6 (U-s101 rerun). Within limits.
6. **Most important.** Keep general slots full when R reruns end (move U-s202 rerun and stress base to general).
7. **Reconciliation.** All live jobs are ledger rows (exact IDs). 11112647 is this project's CPU standby (pending). Watcher alive
   (bnb00222j).

## Review 92 — 2026-11-11 03:50Z (10:50 PM EST Tue)
1. **Changes.** H-s202 (11112377) finished round 66 at 03:49Z; the round-64 evaluations and checkpoint ended 03:36Z. Stress base
   11112932 started on preempt at 03:49Z (L40S constraint). R reruns 11112395/6 at 2:08 h of ~4.3 h; U-s101 rerun 11112397 at 1:03 h.
   New: `qis.eval_merge` (22aa5e2) and six shard recipes for the H-s202 test (D149).
2. **Competing explanation still plausible.** Hardware (R on L40, U on L40S originally) against length drift/truncation (D142, D148).
3. **Distinguishing task.** Common-GPU reruns (first two finish ≈ 06:00Z): R − U on L40S with the original seeds.
4. **Schedule.** Review 91's ETA was too early (D149). H-s101 rounds grew to ~1,000 s and round 96 added an evaluation; H-s202 runs
   ×1.0–1.16 of H-s101. RUN_COMPLETE ≈ 20:00–22:20Z (3:00–5:20 PM EST Wed). Six-shard test ≈ 1.3–2.5 h → ≈ 21:30Z–00:50Z.
   Verdict: report on time; the H-s202 test cell is at risk in the late case (cut-off ≈ 23:45Z, D149). Everything else prepares
   the report so the H-s202 cell drops in by script.
5. **GPUs.** General 3/3 running (H-s202, two R reruns). Preempt 2 running + 4 pending = 6/6 slots. Within limits.
6. **Most important.** Shard the H-s202 test (done, validated). Next: when R reruns end, move U-s202 rerun and stress endpoints to
   general; finish all stress work on general by ≈ 19:30Z.
7. **Reconciliation.** All live jobs are ledger rows (exact IDs): 11112377, 11112395–8, 11112932/37/38/40; standby 11112647.
   Other `ae-dco-*` jobs are not this project's. Watcher bnb00222j alive.

## Review 93 — 2026-11-11 04:50Z (11:50 PM EST Tue)
1. **Changes.** Stress base done (11112932: 1,768/1,768 attempts, pass@1 0.373, pass@8 0.688, 16 min). Stress R-s101-r128 11112937
   and U-s101-r128 11112938 started on preempt (48 and 42 min). H-s101-r128 submitted (11113928, pending). H-s202 at round 71 (04:49Z).
2. **Competing explanation still plausible.** Hardware (original R on L40, U on L40S) against length drift/truncation (D142, D148).
3. **Distinguishing task.** Common-GPU R reruns: R-s101 ≈ 05:50Z, R-s202 ≈ 06:15Z (original durations 4.07 h and 4.48 h).
4. **Schedule.** H-s202 rounds 67–71 averaged 717 s; rounds 66→71 took 3,584 s against 3,487 s for H-s101 (×1.03). Projection: round
   128 ≈ 19:25Z, RUN_COMPLETE ≈ 20:25Z (range 20:00–21:30Z) → six-shard test ≈ 21:45–23:00Z → report verified by 00:30Z.
   Verdict: on time, slack ≈ 1.5 h in the central case.
5. **GPUs.** General 3/3 running. Preempt 3 running + 3 pending = 6/6 slots. Within limits.
6. **Most important.** No action needed now; at R-rerun completion move U-s202 rerun and pending stress cells to general.
7. **Reconciliation.** Live jobs 11112377, 11112395–8, 11112937/38/40, 11113928 and standby 11112647 are ledger rows. Watcher
   b4frpa46w alive.

## Review 94 — 2026-11-11 05:50Z (12:50 AM EST Wed)
1. **Changes.** Common-GPU R-s101 rerun done (L40S pass@1 0.823 vs original L40 0.819; truncated 10.9% vs 10.7%). U-s202 rerun
   moved to general (11112398, running since 05:47Z). U-s101 rerun 11112397 preempted at 05:44Z after 2.95 h (no partial output).
   Stress U-s202-r128 submitted (11115052, preempt). H-s202 at round 75 (05:44Z).
2. **Competing explanation still plausible.** For seed 101, hardware is excluded: R − U with R on L40S is +0.045 (0.823 − 0.777)
   against +0.042 originally. For seed 202, hardware against truncation is still open.
3. **Distinguishing task.** R-s202 rerun (ETA ≈ 06:20Z) and U-s202 rerun (ETA ≈ 09:00–10:20Z), both L40S.
4. **Schedule.** H-s202 rounds 72–75 averaged 834 s; rounds 66→75 took 6,920 s against 6,281 s for H-s101 (×1.10). Projection:
   round 128 ≈ 20:30Z, RUN_COMPLETE ≈ 21:30Z (range 20:45–22:15Z). Six shards on 3 general GPUs ≈ 1.7 h → test ≈ 23:15Z (range
   22:30–00:00Z). Verdict: report on time; the H-s202 test cell has ≈ 1.25 h slack in the central case. The H-s202-r128 stress cell
   can run only after RUN_COMPLETE and will likely be incomplete. Report text for the H-s202 cells must be filled by script.
5. **GPUs.** General 3/3 running. Preempt 2 running + 4 pending = 6/6 slots. Within limits.
6. **Most important.** When R-s202 rerun ends, move stress R-s202-r128 (11112940) to general; then H-s101-r128, U-s202-r128,
   r64 cells. Stress endpoint cells take ≈ 2 h each (R/U-s101 at 1.8 h and still running).
7. **Reconciliation.** Live jobs 11112377, 11112396–8, 11112937/38/40, 11113928, 11115052 and standby 11112647 are ledger rows.
   Watcher bqye6eecl alive.

## Review 95 — 2026-11-11 06:50Z (1:50 AM EST Wed)
1. **Changes.** Common-GPU R-s202 rerun done (L40S pass@1 0.761 vs original L40 0.748; truncated 12.4% vs 13.1%); R − U against the
   original U-s202 (L40S) is −0.031 (original −0.044). Stress R-s202-r128 moved to general (11112940, running). Stress endpoints done:
   R-s101-r128 pass@1 0.695 (truncated 20.5%), U-s101-r128 0.653 (23.0%); base 0.373. R-s101-r64 11115377 and H-s202-r64 11115378
   submitted (preempt). H-s202 at round 79 (06:41Z).
2. **Competing explanation still plausible.** Hardware is no longer a candidate for either seed (R−U keeps its sign on L40S, with
   shifts of +0.003 and +0.013). Open: whether the seed-202 deficit is purely length/truncation (D142) or partly accuracy; the
   L40S U-s202 rerun gives the matched-hardware decomposition with intervals.
3. **Distinguishing task.** U-s202 rerun (general, ETA ≈ 09:00–10:20Z) → a17_test --cgpu-root (truncation term vs accuracy term on
   matched hardware). Stress U-s202-r128 against R-s202-r128 tests whether the s202 sign holds on harder problems.
4. **Schedule.** H-s202 rounds 76–79 averaged 851 s; rounds 66→79 took 10,325 s against 9,485 s for H-s101 (×1.09). Round 128 ≈
   20:20Z, RUN_COMPLETE ≈ 21:20Z (range 20:45–22:15Z); six-shard test ≈ 23:00Z. Verdict: on time, slack ≈ 1.5 h for the H-s202 cell.
   Preempt has started nothing since 04:02Z; the remaining stress cells (H-s101, U-s202 r128; two r64) and the U-s101 rerun need
   ≈ 14 GPU-h, and general has ≈ 24 GPU-h free between ≈ 08:30Z and 20:30Z.
5. **GPUs.** General 3/3 running. Preempt 0 running + 5 pending = 5/6 slots. Within limits.
6. **Most important.** Move stress cells to general as slots free (order: U-s202-r128, H-s101-r128, R-s101-r64, H-s202-r64,
   U-s101 rerun). Keep general free of new stress work from ≈ 20:30Z.
7. **Reconciliation.** Live jobs 11112377, 11112397/98, 11112940, 11113928, 11115052, 11115377/78 and standby 11112647 are ledger
   rows. Watcher re-armed after the U-s101 stress event.

## Review 96 — 2026-11-11 07:50Z (2:50 AM EST Wed)
1. **Changes.** Preempt started all five pending jobs at ≈ 07:02–07:23Z: stress U-s202-r128 11115052, H-s101-r128 11113928,
   R-s101-r64 11115377, H-s202-r64 11115378, and the U-s101 rerun 11112397 (restart from zero). H-s202 at round 83 (07:39Z).
2. **Competing explanation still plausible.** Seed-202 deficit: truncation only (D142) or truncation plus an accuracy loss.
3. **Distinguishing task.** U-s202 L40S rerun (2:04 h of ≈ 3.1–4.5 h) → a17_test --cgpu-root; stress R/U-s202 pair.
4. **Schedule.** H-s202 rounds 80–83 averaged 871 s; rounds 66→83 took 13,810 s against 12,552 s for H-s101 (×1.10). Round 128 ≈
   20:30Z, RUN_COMPLETE ≈ 21:30Z (range 20:45–22:15Z), six-shard test ≈ 23:15Z. Stress backlog now runs in parallel on preempt
   (≈ 2.5 h each → ≈ 09:30–10:00Z unless preempted). Verdict: on time; H-s202 test slack ≈ 1.25 h.
5. **GPUs.** General 3/3. Preempt 5 running + 0 pending = 5/6. Within limits.
6. **Most important.** Write the stress-panel analysis (per tier × checkpoint: pass@1, pass@8, truncation, solved-at-8, coverage
   against base, paired R − U intervals) so results drop into the report as cells finish.
7. **Reconciliation.** All live job IDs are ledger rows; standby 11112647. Watcher alive (pid 82847).

## Review 97 — 2026-11-11 08:50Z (3:50 AM EST Wed)
1. **Changes.** Stress R-s202-r128 done (pass@1 0.632, truncated 23.0%). Stress analysis draft (D-none; `a17_stress`, 3 cells):
   s101 R − U on stress 0.695 − 0.653 = +0.042 [+0.015, +0.068]. CPU standby 11112647 started. H-s202 at round 86 (08:42Z).
2. **Competing explanation still plausible.** Seed-202 deficit: truncation only (D142) or truncation plus accuracy.
3. **Distinguishing task.** U-s202 L40S rerun (3:04 h of ≈ 3.1–4.5 h) → a17_test --cgpu-root; stress U-s202-r128 (1:48 h).
4. **Schedule.** H-s202 responses grew to 6,000–6,800 tokens (H-s101: 3,500–4,500 at the same rounds); rounds 84–86 took 1,255 s
   each. RUN_COMPLETE ≈ 23:10Z (×1.21) to 01:15Z (×1.4). Verdict: the report is on time; the H-s202 endpoint test very likely is
   not. Fallback test of the latest checkpoint decided (D150); owner informed.
5. **GPUs.** General 2/3 (no runnable job waits). Preempt 5/6 running. Within limits.
6. **Most important.** D150 fallback; prepare the report so H-s202 cells fill by script.
7. **Reconciliation.** All live job IDs are ledger rows. Watcher alive (pid 13846).

## Review 98 — 2026-11-11 09:50Z (4:50 AM EST Wed)
1. **Changes.** U-s202 L40S rerun identical to the original (4,096/4,096 sequences); s202 R − U on a common GPU −0.031
   [−0.060, −0.004] (D151). Stress: U-s202-r128 and H-s202-r64 done; s202 stress R − U −0.027 [−0.051, −0.002] = truncation
   −0.063 + accuracy +0.036 (D152). Stress H-s101-r128 11113928 was preempted after ≈ 2 h (requeue; the watcher does not report
   requeues) and moved to general (pending, Resources: L40S). H-s202 at round 90 (09:42Z).
2. **Competing explanation still plausible.** Hardware is excluded (D151). Remaining: is the length difference between seeds a
   property of the run (seed-level drift in termination) or of the arm? With two seeds it cannot be separated (A9.4).
3. **Distinguishing task.** None within scope (needs more seeds; A17 forbids new seeds). Report it as the open question.
4. **Schedule.** H-s202 rounds 87–90 averaged 900 s (×1.23 of H-s101); rounds 66→90 ×1.215. Round 128 ≈ 22:15Z, RUN_COMPLETE ≈
   23:15Z (6:15 PM EST) → endpoint test after the deadline. D150 fallback at 18:00Z stands. Report: on time.
5. **GPUs.** General 1 running + 1 pending = 2/3. Preempt 2 running = 2/6. Within limits.
6. **Most important.** Get H-s101-r128 stress done (general). Then the REPORT fill: endpoints with common-GPU table, stress
   section, What we learned.
7. **Reconciliation.** Live job IDs 11112377, 11112397, 11113928, 11115377 and standby 11112647 are ledger rows. Watcher alive
   (pid 22415).
- 09:58Z correction to Review 98 item 1: 11113928 was not lost. Its evaluation wrote summary.json at 09:44:33Z, the moment of preemption; the requeued job skipped the completed phases in 26 s on general.

## Review 99 — 2026-11-11 10:51Z (5:51 AM EST Wed)
1. **Changes.** Stress R-s101-r64 11115377: its evaluation finished at 10:13:43Z (pass@1 0.643, 1,768/1,768 attempts, 0 unknown,
   ckpt hash verified). Slurm logged that same second as a preemption and requeued it. I moved it to general, where it skipped
   the finished phases and completed at 10:52Z. `a17_stress` rerun on 8 of 9 cells (H-s202-r128 still missing).
   - s101, all 221 stress problems: R 0.695, U 0.653, H 0.641. R − H +0.054 [+0.025, +0.082]; U − H +0.012 [−0.014, +0.038].
   - R-s101 r64 → r128: 0.643 → 0.695, +0.053 [+0.030, +0.076]. Truncation term +0.049, accuracy term +0.003: the late gain is
     fewer capped answers (26.1% → 20.5%), not better accuracy on finished answers.
   - Check from 8.5 done: equal solved counts across cells are coincidence. Raw records differ (distinct hashes and correct
     totals); solved sets differ by 7–16 problems per pair (independent script on the records).
2. **Competing explanation still plausible.** Seed-level length drift (run property) against an arm effect. Two seeds cannot
   separate them (A9.4). New this hour: within R-s101, the r64 → r128 gain is also a truncation change, so length is the main
   moving part in every stress contrast.
3. **Distinguishing task.** None within scope (needs more seeds). Report it as the open question.
4. **Schedule.** H-s202 round 93 at 10:36Z. Rounds 66→93 ran ×1.267 of H-s101; rounds 90→93 ×1.75 (len_mean 5,905–6,533).
   H-s101 needed 35,229 s for rounds 93→128. Round 128 ≈ 22:30Z (×1.215) to 23:20Z (×1.3); RUN_COMPLETE ≈ 23:30Z–00:20Z. The
   D150 fallback trigger (RUN_COMPLETE later than 22:15Z) is already met; the fallback runs at 18:00Z (expected round 112–116).
   cgpu U-s101 rerun: running since 07:02Z, ETA ≈ 11:30Z. Report: on time; H-s202 endpoint test after the deadline.
5. **GPUs.** General 1/3 (H-s202). Preempt 1/6 (cgpu U-s101). CPU standby 11112647. Within limits.
6. **Most important.** Prepare the D150 fallback recipes now (checkpoint copy + six shards), so 18:00Z needs only a submit. Then
   fill REPORT sections from D151/D152.
7. **Reconciliation.** Live IDs 11112377, 11112397 and standby 11112647 are ledger rows; 11115377 completed. Watcher re-armed
   after the R-s101-r64 event (task bqhmii0g8).

## Review 100 — 2026-11-11 11:50Z (6:50 AM EST Wed)
1. **Changes.**
   - Response audit done (D153): 96 responses, two blind reads each, category agreement 88/96. 29 of 70 failure reads judged
     fixable by more factoring practice. Grader limitation found (separate boxes → last box only); sealed test affected by
     ≤ 3/4,096 attempts per cell (≤ 0.07 points).
   - cgpu U-s101 11112397: eval finished 11:18Z at the moment of preemption (third job with this pattern, after 11113928
     and 11115377). Moved to general; completed 11:53Z; ckpt hash verified. U-s101 L40S 0.778 vs original 0.777.
   - Common-GPU table complete for R and U: s101 R − U 0.823 − 0.778 = +0.044 [+0.024, +0.065]; s202 −0.031 [−0.060,
     −0.004]; mean +0.007 [−0.010, +0.023]. New: same GPU type is not always bit-identical (U-s101 1,964/4,096 identical
     sequences; U-s202 4,096/4,096). pass@1 moved ≤ 1.3 points in every rerun. REPORT §1 filled; endpoints figure redrawn.
   - CPU standby 11112647 was preempted 11:14Z and requeued on babel-m5-24 (running).
2. **Competing explanation still plausible.** Unchanged: seed-level length drift against an arm effect (A9.4).
3. **Distinguishing task.** None within scope. H-s202's endpoint adds the seed-202 H cell; it cannot separate run from arm.
4. **Schedule.** H-s202 rounds 94–96 ran ×1.02 of H-s101 (len_mean fell to 4,292–4,344); round 96 at 11:13Z, then the r96
   eval. Cumulative ×1.24 since round 66. Round 128 ≈ 21:20Z (×1.1) to 22:50Z (×1.27); RUN_COMPLETE ≈ 1 h later
   (22:20Z–23:50Z); six-shard endpoint ≈ 1 h after that → 23:20Z–00:50Z. Endpoint before 00:30Z: possible, not likely.
   D150 fallback stays at 18:00Z (decide by the rule then). Report: on time.
5. **GPUs.** General 1/3 (H-s202). Preempt 0/6. Within limits.
6. **Most important.** REPORT fill continues (What we learned item 1–2 wording with cgpu; section 8 run/cost table;
   incomplete-cell list). Fix the endpoints legend overlap.
7. **Reconciliation.** Live IDs 11112377 and standby 11112647 are ledger rows. Watcher re-armed (task bo4i0sb9k).

## Review 101 — 2026-11-11 12:50Z (7:50 AM EST Wed)
1. **Changes.**
   - H-s202 11112377: round 98 at 12:43Z. The round-96 eval and save took 75.5 min (H-s101: 57.5 min). LATEST = round 96.
     len_mean rose again (r97 5,965; r98 5,354, against 4,292–4,344 at r95–96).
   - No other job ran. REPORT What we learned item 1 now cites the common-GPU rerun (50ac4ed).
2. **Competing explanation still plausible.** Unchanged: seed-level length drift against an arm effect (A9.4).
3. **Distinguishing task.** None within scope (no new seeds). H-s202 r128 adds the seed-202 H cell only.
4. **Schedule.** Rounds 96→98 ran ×1.30 of H-s101 (eval included); cumulative ×1.24 since round 66. H-s101 needed 481.6 min
   for rounds 98→128. Round 128 ≈ 20:45Z (×1.0) to 23:10Z (×1.30); RUN_COMPLETE ≈ 1 h later (21:45Z–00:10Z); six-shard
   endpoint ≈ 1 h after that (22:45Z–01:10Z). The central case (×1.24, RUN_COMPLETE ≈ 23:40Z) meets the D150 fallback trigger
   (> 22:15Z). Decide at 18:00Z by the rule; recipes ready. Report: on time; H-s202 r128 endpoint unlikely before 00:30Z.
5. **GPUs.** General 1/3 (H-s202). Preempt 0/6. Within limits. The fallback needs 2 general + 4 preempt: fits.
6. **Most important.** REPORT fill: fix the endpoints legend overlap, then the section 8 run/cost table and incomplete cells.
7. **Reconciliation.** Live IDs 11112377 and standby 11112647 are ledger rows; other running jobs are ae-dco-* (another
   project). Watcher alive (task bo4i0sb9k).

## Review 102 — 2026-11-11 13:50Z (8:50 AM EST Wed)
1. **Changes.**
   - H-s202 11112377: round 102 at 13:46Z; LATEST = round 100. len_mean 5,084–6,493 at rounds 99–102 (stays high).
   - REPORT section 8: provisional compute table (sacct over all 225 ledger jobs, 239 allocations): 283.0 GPU-h, 2,295
     CPU-h. A17 so far: common-GPU reruns 18.9, stress 18.1, H-s202 resume 12.2 GPU-h. `cost_table` gained the two A17
     categories (3c184ed). Endpoints legend fixed (d0fb202).
2. **Competing explanation still plausible.** Unchanged: seed-level length drift against an arm effect (A9.4).
3. **Distinguishing task.** None within scope (no new seeds).
4. **Schedule.** Rounds 98→102 ran ×1.11 of H-s101; ×1.21 since round 96 (eval included). Round 128 ≈ 21:35Z (×1.11) to
   23:00Z (×1.30); RUN_COMPLETE ≈ 22:35Z–00:00Z; six-shard endpoint ≈ 23:35Z–01:00Z. Central case (×1.24): RUN_COMPLETE
   23:30Z, past the D150 trigger (22:15Z). At 18:00Z LATEST is likely round 112 (round 116 projected ≈ 18:15Z). Report:
   on time.
5. **GPUs.** General 1/3, preempt 0/6. Within limits.
6. **Most important.** Section 8 compute table (done, provisional). Next: incomplete-cell list and manifest table.
7. **Reconciliation.** Live IDs 11112377 and standby 11112647 are ledger/standby rows; other running jobs are ae-dco-*.
   Watcher alive (task bo4i0sb9k).

## Review 103 — 2026-11-11 14:50Z (9:50 AM EST Wed)
1. **Changes.**
   - H-s202 11112377: round 105 at 14:38Z; LATEST = round 104. len_mean 5,418–6,660 at rounds 102–105.
   - CPU standby 11112647 was preempted again (≈ 14:17Z) and requeued on babel-o5-16 (11.4 h left).
   - REPORT reproduction: manifest table with full sha256 for every checkpoint and both frozen panels (rehashed: both
     match). Each checkpoint used by several evaluations has one hash. Source `research/evidence/a17/manifest/`.
2. **Competing explanation still plausible.** Unchanged: seed-level length drift against an arm effect (A9.4).
3. **Distinguishing task.** None within scope (no new seeds).
4. **Schedule.** Rounds 102→105 ran ×1.14 of H-s101 (×1.19 since round 96). Round 128 ≈ 21:50Z (×1.14) to 22:50Z (×1.30);
   RUN_COMPLETE ≈ 22:50Z–23:50Z; six-shard endpoint ≈ 23:50Z–00:50Z. Every case is past the D150 trigger (22:15Z). At
   18:00Z LATEST is likely round 112 (round 116 ≈ 18:05Z). Report: on time.
5. **GPUs.** General 1/3, preempt 0/6. Within limits.
6. **Most important.** Manifest (done). Next: incomplete-cell list; 18:00Z fallback decision.
7. **Reconciliation.** Live IDs 11112377 and standby 11112647 are ours; other running jobs are ae-dco-*. Watcher alive
   (task bo4i0sb9k).

## Review 104 — 2026-11-11 15:50Z (10:50 AM EST Wed)
1. **Changes.**
   - H-s202 11112377: round 109 at 15:40Z; LATEST = round 108. len_mean 5,388–6,195 at rounds 106–109.
   - CPU standby 11112647 preempted again; PENDING (Priority). preempt_cpu_qos has 77 pending jobs, preempt_qos 845.
     `sbatch --test-only` estimates (15–29 Nov) are backfill bounds, not forecasts. Read the metrics through an
     `srun --overlap` step in 11112377 instead (1 CPU, 1 GB, read-only).
   - Risk: the D150 copy job (`qis-h202-fallback-copy`, CPU pool) may not start at 18:00Z. Prepared
     `qis-h202-fallback-copy-g` (same body, 1 general GPU; `bash -n` ok). Use it if the CPU pool is still blocked.
2. **Competing explanation still plausible.** Unchanged: seed-level length drift against an arm effect (A9.4).
3. **Distinguishing task.** None within scope (no new seeds).
4. **Schedule.** Rounds 105→109 ran ×1.18 of H-s101 (×1.19 since round 96). Round 128 ≈ 22:05Z (×1.18) to 22:45Z (×1.30);
   RUN_COMPLETE ≈ 23:05Z–23:45Z; six-shard endpoint ≈ 00:05Z–00:45Z. Every case is past the D150 trigger (22:15Z). At
   18:00Z LATEST will be round 112 (the round-112 eval runs ≈ 16:35Z–17:45Z; round 116 ≈ 18:30Z or later). If preempt GPUs
   do not start, two general shards at a time finish the fallback in ≈ 3 h (≈ 21:15Z). Report: on time.
5. **GPUs.** General 1/3, preempt 0/6. Within limits.
6. **Most important.** Fallback copy path that does not depend on the CPU pool (done). Next: incomplete-cell list.
7. **Reconciliation.** 11112377 (ledger) running; 11112647 (our standby) pending; other jobs are ae-dco-*. Watcher alive
   (task bo4i0sb9k).

## Review 105 — 2026-11-11 16:50Z (11:50 AM EST Wed; first review under A19)
1. **Changes.**
   - A19 recorded (`project.md` §33, D154, CHECKLIST 0.26 + section 9, STATE). H-s202 gate kept: the round-128 dev eval
     precedes the save; `RUN_COMPLETE` follows within seconds.
   - H-s202 11112377: round 112 at 16:34Z (LATEST = 112; no eval at 112), round 113 at 16:48Z.
   - Length diagnostic (exact continuation 12,288 → 16,384): tranche 1 (64 problems × 4 attempts) done for R-s202 and
     U-s202; R-s101 generating (11118247, general); U-s101 queued on preempt (11118248). Canary 11118157 cancelled at the
     start of its full-panel phase (equal coverage first); U-s202 11118245 in its full-panel phase.
     | t1, 256 attempts | Correct at 12,288 | Capped | Recovered | Finished wrong | Still capped | Correct at 16,384 |
     |---|---|---|---|---|---|---|
     | R-s202 | 173 | 56 | 9 | 13 | 34 | 182 |
     | U-s202 | 178 | 41 | 9 | 11 | 21 | 187 |
     Seed 202 R − U on t1: (173 − 178)/256 = −2.0 points at 12,288 and (182 − 187)/256 = −2.0 at 16,384 (unchanged).
   - Verification: 18 cells + 9 contrasts recomputed with stdlib code, all match (`research/audits/a19/`); J and ΔJ
     figures match their tables. Audit counts resolved (35 responses, 70 reads, 4 n.a.). REPORT: Q1, section 9.2
     (divergence: sign appears in rounds 33–64 in both seeds), incomplete-cell table.
   - CPU standby 11112647 running again (babel-n5-20).
2. **Competing explanation still plausible.** Seed-level path dependence against an arm effect. New: in seed 202 the
   extra 4,096 tokens recover the same number of answers in R and U, so "R's answers only needed more room" does not
   explain the seed-202 deficit on t1.
3. **Distinguishing task.** Full-panel continuation for all four checkpoints (power), then R − U at both caps with the
   paired bootstrap. Seeds cannot be added (A9).
4. **Schedule.** Rounds 109→113 ran ×1.07 of H-s101 (×1.17 since round 96). Round 128 ≈ 21:30Z–22:30Z; RUN_COMPLETE ≈
   22:30Z–23:30Z (every case past the D150 trigger, 22:15Z); endpoint ≈ 23:30Z–00:30Z. Fallback at 18:00Z on LATEST
   (round 112, or 116 if saved by then ≈ 17:56Z). Length full panels ≈ 17:30–18:30Z. Report: on time.
5. **GPUs.** General 3/3 (H-s202, len U-s202, len R-s101). Preempt 1/6 queued (len U-s101). Within limits.
6. **Most important.** Equal coverage: when R-s101 t1 ends, stop it and move U-s101 to general.
7. **Reconciliation.** 11112377, 11118245, 11118247, 11118248 are ledger rows; 11112647 our standby; 11118157 cancelled
   by me. Watcher alive (task blzowz4dh).

## Review 106 — 2026-11-11 17:50Z (12:50 PM EST Wed)
1. **Changes.**
   - A20 (owner, ~17:15Z): 8 general GPUs until 21:00Z, then 3 (`project.md` §34, D155; guard time-limited). This review's
     "3 general" limit is superseded until 21:00Z.
   - H-s202 11112377: round 116 at 17:33Z (LATEST = 116), round 117 at 17:46Z. Rounds 113→117 ran ×0.78 of H-s101.
   - Length diagnostic: tranche 1 done for all four (R − U 12,288 → 16,384: s101 +3.9 → +5.9; s202 −2.0 → −2.0); blind
     read of 12 continuations (6 loops, 3 wrong, 1 sound, 1 repaired but capped, 1 productive). U-s202 stress full panel
     done: 1,768 attempts, capped 268, recovered 41, finished wrong 81, still capped 146; pass@1-equivalent correct
     1,164 → 1,205. U-s101, R-s202, R-s101 full panels running. Sealed-test continuation (D155) running on 4 general GPUs.
   - REPORT: Q1, Q3, 9.1 (tranche 1 + inspection), 9.2, incomplete-cell table; recomputation 27/27 verified; J/ΔJ figures
     verified; one broken path fixed.
2. **Competing explanation still plausible.** Seed-level path dependence (the runs split right after the cap change at
   round 33, in opposite directions) against an arm effect. "R-s202 only needed more room" is unlikely: equal recovery in
   both arms on tranche 1, and most capped answers loop or end wrong.
3. **Distinguishing task.** Sealed-test continuation (D155): if R-s202 − U-s202 stays ≈ −4.4 at 16,384, the deficit is not
   a budget artefact. Two seeds cannot separate path dependence from arm.
4. **Schedule.** D150 projection (cumulative ×1.164 since round 66): round 128 ≈ 21:26Z, RUN_COMPLETE ≈ 22:26Z → past
   22:15Z → fallback at 18:00Z on round 116 (preempt; idle). Range of projections: RUN_COMPLETE 21:15Z (×0.78) to 22:50Z
   (×1.30); endpoint ≈ 1 h later (22:15Z–23:50Z) on 3 general + preempt. Report: on time.
5. **GPUs.** General 8/8 (A20): H-s202, 3 stress full panels, 4 sealed-test continuations. Preempt 0/6 until the fallback.
6. **Most important.** Fallback at 18:00Z; then Q2/Q4 from the full-panel and sealed-test continuations.
7. **Reconciliation.** All 8 general jobs and standby 11112647 are ours (ledger / standby). Watcher alive (bfx1obf8f).
