# CHECKLIST — study s02 live execution plan (updated 2026-11-16 14:14 EST)

Marks: [x] done  [~] active  [!] blocked  [ ] pending. Definitions: PROJECT.md §2.

## 0. Owner instructions (mandate 2026-11-15; PROJECT.md, archive project-instructions.md)
- [~] A1 (2026-11-15 23:30): small diverse task set beyond math (code, puzzles, symbolic, instruction following,
      factual); reliable grading; behavioural objectives kept distinct from skills
- [x] New study ID and folders (research/, reports/, data/, envs/, instructions archive)
- [ ] NON-NEGOTIABLE: ceilings 2 general GPU / 6 preempt GPU / 20 CPU jobs; standbys count (`ops/qis-submit`)
- [ ] NON-NEGOTIABLE: separate selection (`sel`) and confirmation (`conf`) data; seed-level results kept
- [ ] NON-NEGOTIABLE: provenance of every result (commit, recipe, config, model rev, bundle SHA, seeds, budget)
- [~] Hourly reviews recorded in `state/direction.md`
- [x] Feasibility of the 9:00 PM EST 2026-11-16 target assessed at ~01:15 EST, revisited hourly
- [x] Final: report + evidence table + handover; dated snapshots of records in `research/<ID>/records-2026-11-16/`

## 1. Infrastructure (needed for family E1)
- [x] Babel clone, `ae setup` (held: all three QOS; home 37 GB free)
- [x] Environment venv-s02 + model @ 70d244cc (inside `qis-s02-calib`; CPU job 11167443 held as fallback)
- [x] Runner extensions: mix, loss divisor, n_per_prompt, n_minibatch, init_round, per-role groups (2e2bfdc, 6b31745)
- [x] Independent code review; 5 defects fixed (6b31745)
- [x] Unit tests pass on compute (61 + new graders) (`pytest.txt` of calib) — gate for training
- [x] CPU env job 11167443 cancelled (redundant)

## 2. E0 — calibration (selection data only)
- [x] `qis-s02-calib` + `calib2`: 26 pools × 64 × 16, cap 4096: pass@1 per stratum, length, truncation, grader audit
- [x] Grader audit (3 artefacts fixed, 602b151; regrade cancelled, superseded by in-run r0 evals): read wrong/no-answer examples; confirm no format artefact drives low add9 success
- [x] Chosen: caesar s4/s8/s12 and add9 d4/d6/d8 as stratified targets; terse k2, add10 d14, mbpp as C; cap 4,096;
      eval 64 problems x 4 samples per panel (E1), 64 x 16 on conf for probes

## 3. E1 — canary, single-task matrix (references R + off-target effects) and joint pairs
- [x] Record bundle `s02-v1` spec + SHA-256 in `data/<ID>/BUNDLES.md` (a761b981; same in all 4 harvested runs)
- [x] Canary (joint add9+countdown, 11168205): forced exit + resume ok; 105-130 s/round; 36 GiB
- [x] lr 2e-6 (study-00 D57)
- [x] R(caesar) F3; R(mbpp) F4; R(add9) F5 (harvested, analysed)
- [x] R(terse), R(add10): CANCELLED 06:48 (only C references; resubmit with new recipe if needed)
- [x] countdown / ARC single runs dropped (D4)
- [x] J(caesar,terse) harvested (F6): transient delay of caesar s8/s12 (D +.10..+.18 at r8-16, ~0 by r24); terse learned
- [x] E1c: R(caesar) s2 (F3 replicates) + J s2 (F11: delay of harder strata replicates; persists to r32 on s8 in s2). Decides F6 noise vs effect
- [x] J(add9,add10) harvested (F7): no harm to add9; R's d6 collapse absent in J (D -.48 at r32); D3 step confound noted
- [x] E1d J(caesar,mbpp) (F15: caesar lags +.09..+.12 at r32, MBPP unharmed, terse protected; step-sharing confound)
- [x] E1f caesar-terse seed 3 (F19): no delay; delay 2/3 seeds, tracks answer shortening
- [x] E1e R(caesar) lr 1e-6 (F17): halving lr costs .14-.53 → F15 lag explainable by step sharing
- [x] E1d J(add9,terse) (F18): no harm to add9; terse learned
- [ ] Analysis: D(t) per stratum with paired problem bootstrap; A0/mixed/A1 per role and stratum; lengths

## 4. E2 — local update effects (RQ1 "actual update" + RQ2 mechanism)
- [ ] DEPRIORITISED (F6: terse all-correct by r3, so Conly ≈ null at r8). J(caesar,terse) r8 probe: Tonly / Conly / null /
      null2, conf panels 64 x 16. Caveat: terse is all-correct by r3, so Conly may equal null (measured, not assumed)
- [ ] r16 probe (same recipe family) if r8 is informative

## 5. E3 — RQ2 discriminating runs
- [~] E3a add9 within-family decomposition (pending, preempt; first to general when free):
  - [x] seed-2 replicate: collapse replicates (F8; r16 checkpoint kept)  - [x] d6 alone: no collapse (F8)
  - [x] d8 alone (F12): harms d4 −.25, d6 −.15; d8 not learned  - [x] d4 alone (F9): d6 rises .15 → .36, no collapse
- [ ] If the collapse replicates and d6 alone learns: recovery branches from the mixed run at r24 (d6 all-wrong):
      continue mixed vs d6-only vs mixed with more d6 samples per prompt
- [x] DROPPED: R(add9) lr 1e-6 (F8: d6-only with full step does not collapse)
- [x] E3b recovery (F10: 0 successes → no recovery; 1 success (x32 samples) → .11; mixed → .07) from mix s2 r16 (baeda29; 11171949/53/94, preempt; move to general when free): 8 rounds of mixed /
      d6-only / d6-only x32 samples. One-round probe rejected (all-wrong d6 groups give zero gradient)
- [x] Storage incident resolved: 236G free (11171795); max 4 checkpointing runs at once
- [~] E3d d4+d6 without d8: s1, s2 no collapse (F16; 0/2 vs mixed 2/3, p≈0.4); s3 11174313 ( preempt → next general): predicts no collapse (F12) — now: a collapse here means d8 is not needed (F13, review)
- [x] E3c add9 mixed seed 3 (F14): no collapse → frequency 2/3; retained r16/r24 deleted 13:04
- [x] DROPPED (s3 did not collapse): repeat E3b branches from its collapse state (needs retained checkpoint — s3 retains none;
      would need a rerun with retain_rounds) — decide at s3 readout
- [ ] Caesar RQ2 test: J(caesar,terse) final D(t) per stratum; if harder strata lag, vary C weight (0.5/2)
- [ ] Order: C first then T, vs T first then C (only for a confirmed harmful pair)

## 6. Reporting
- [x] Second critical review of F8-F12 (done by 09:42): C1, C2 supported; C3, C4 partial; C5 d8-trigger not supported
- [x] Report final: reports/<ID>/report.md, runs.md (sacct states), handover.md, 5 figures; audit 18/18. Was: fill E1e, J(add9,terse), E3d s3;
      add sacct final state column to runs.md; handover.md; audit (DONE f4df3eb: 18/18 verified) before final; snapshots
      of state records into research/<ID>/
