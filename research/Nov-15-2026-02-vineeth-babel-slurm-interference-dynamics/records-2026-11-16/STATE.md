# STATE — study s02 (Nov-15-2026-02), updated 2026-11-16 13:37 EST

## Objective
RQ1: which task relationships give transfer / independence / transient harm / persistent delay, and why.
RQ2: does on-policy feedback amplify an initial disadvantage (adverse update → lower T success → fewer mixed T
groups → slower acquisition)? Definitions: PROJECT.md §2. Owner amendment A1: tasks beyond math in scope.
All results below: one seed, sel panels, EXPLORATORY.

## Established (this study)
- Base success of 26 candidate pools (F1); grader artefacts fixed before training (602b151); runner validated
  (canary 11168205: two roles, checkpoint, forced exit + resume).
- caesar alone (F3): all strata acquired (s12 .04 → .71). Off-target: terse −.34 via over-long answers (a
  competing behavioural demand, not lost arithmetic); add9 d6 +.11.
- MBPP alone (F4): +.14; arc −.04 (small harm); add10 +.07; terse unaffected.
- add9 alone on mixed strata (F5): d4 improves (.52 → .77); d6 rises to .49 (r16) then collapses to .05 with
  all-wrong training groups r26-31; d8 never gets signal. Terse −.34 via unboxed answers.

## Uncertain
- add9 d6 collapse: 2/3 mixed-strata seeds (F14), 0/3 single-stratum or +add10 runs. Persistence supported
  (review): once d6 groups are all-wrong there is no signal; recovery needs a nonzero update (rare success or d4
  transfer, F10). Trigger unknown: NOT d8 (F13, review); candidates d4×d6, one large update, sparse-update drift.
  d4+d6 without d8: 0/2 collapse (F16; s3 running) vs 2/3 with d8 — suggestive, p≈0.4. d8 alone harms d4/d6 (F12).
- caesar under terse (F6, F11): harder strata delayed (r16 D +.11 to +.18, 2/2 seeds); closed by r24 in seed 1,
  still +.12 on s8 at r32 in seed 2. Caesar answers 5-15% shorter in J during the lag (brevity spill-over, untested).
- caesar × MBPP (F15, F17): lag explained by Adam step sharing (half-lr costs .14-.53). add9 × terse (F18): no harm.
  add9 collapse: alone 2/3, with competitor 0/2, d4+d6 0/3 (F16). Report drafted (reports/<ID>/), handover drafted.
- vLLM sampling is not reproducible across jobs (±.01-.05); terse panel noise about ±.1.

## Decisions (and why)
- lr 2e-6, clip 0.2, 8 groups × 8 responses, 4 steps/round, divisor 32, cap 4,096 (study-00 selection; canary).
- Divisor 32 matches raw gradient weight, not Adam update size (D3).
- Countdown and ARC single runs dropped (D4). J(caesar,mbpp), J(add9,terse) deferred until E3a reads out.

## Blockers
- Storage: resolved (236G free at 06:39); recipes guard free space and prune checkpoints; ≤4 checkpointing runs.
- NFS writes slow since ~11:30 (checkpoint ~1 h at 6 MB/s): jobs finish late.
- Preempt starts slowly; babel-l9-32 excluded (CUDA devices unavailable). Hourly cron is session-only.

## Next actions
1. E1f caesar-terse seed 3 readout; update report.
2. runs.md state column; records snapshot; audit; final report by 21:00.
