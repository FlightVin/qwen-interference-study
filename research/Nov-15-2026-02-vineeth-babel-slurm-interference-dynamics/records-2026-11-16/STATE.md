# STATE — study s02 (Nov-15-2026-02), updated 2026-11-16 14:14 EST — FINAL (report delivered)

## Objective
RQ1: which task relationships give transfer / independence / transient harm / persistent delay, and why.
RQ2: does on-policy feedback amplify an initial disadvantage (adverse update → lower T success → fewer mixed T
groups → slower acquisition)? Definitions: PROJECT.md §2. Owner amendment A1: tasks beyond math in scope.
All results exploratory (1-3 seeds), sel panels unless "conf". Report: reports/<ID>/report.md.

## Established (this study)
- Caesar alone acquires all strata (3 seeds) and lowers untrained terse (3/3 seeds, −.11 to −.40) via long answers.
- caesar × terse: harder-strata delay at r16 in 2/3 seeds, exactly the seeds where joint-run Caesar answers were
  shorter (F6, F11, F19). Brevity spill-over fits; untested by intervention.
- caesar × MBPP: lag of .09-.12 explained by Adam step sharing (half-lr control costs .14-.53; F15, F17).
- add9 × add10 and add9 × terse: no harm to add9 (F7, F18).
- add9 mixed strata: d6 collapses in 2/3 seeds (F8, F14); once all d6 groups are all-wrong there is no signal; with
  zero successes no recovery in 8 rounds, one success or d4 transfer restarts it (F10; review: partial — the
  8-sample branch took no gradient; the 32-sample gain is one step and is confounded with step size).
- d4 training helps d6 (+.21); d8 training harms d4 (−.25) and d6 (−.15) (F9, F12).

## Uncertain
- Trigger of the d6 collapse: not d8 gradients (F13); without d8 0/3 collapse vs 2/3 with (p=0.4); with a
  competitor 0/2. Candidates: d4×d6 interaction, sparse-update drift under Adam, d8 group presence.
- Why d4-only lowers terse while d6-only raises it.
- vLLM sampling not reproducible across jobs (same-model evals differ ≤ .04).

## Decisions (and why)
- lr 2e-6, clip 0.2, 8 groups × 8 responses, 4 steps/round, divisor 32, cap 4,096 (study-00 selection; canary).
- Divisor 32 matches raw gradient weight, not Adam update size (D3) — the reason for the half-lr control.
- Countdown/ARC single runs dropped (D4). Random-reward control dropped (d8 no longer the suspect).

## Blockers
- None open. Incidents resolved: disk full 06:19 (recipes now guard space and prune), bad node babel-l9-32
  (excluded), NFS slowdown 11:30-12:45 (use no_ckpt where possible). Hourly cron is session-only.

## Next actions (for a successor; handover.md §4)
1. Collapse trigger: more seeds of mixed vs d4+d6; per-stratum gradient logging; random-reward stratum.
2. Test brevity spill-over by intervention (looser terse length limit).
3. Verify remaining storage under /data/user_data/$USER/qis/s02 before new work.
