# FINDINGS — append-only durable results

<!-- `ae harvest` appends one line per harvested experiment. Add a dated entry for every durable
     conclusion, with the evidence (result path, numbers, SE) and how it was verified. Correct an
     entry by appending a dated CORRECTION; never edit history silently. -->

> **Owner instruction, 2026-11-13 (kept on reset, 2026-11-15).** Keep durable records of every run in the repository
> under `research/<study-ID>/` (findings, decisions, job tables, recipes, handover), not in this harness folder.
> `ae harvest` and `ae submit` still append mechanical lines here and to `state/LEDGER.jsonl`; treat them as scratch.
> The previous workspace is archived in `research/harness-workspace-archive-2026-11-15.tgz`.
- 2026-11-16  **qis-s02-calib**  FAILED  sha=3416aa38765ae5513e570ca5b8ee728ce6a20a12  job=11167475  results=slurm-agentic-engineering/results/qis-s02-calib/
- 2026-11-16  **qis-s02-calib**  FAILED  sha=52f80816682a86acec71db41523015c0d812faa8  job=11167515  results=slurm-agentic-engineering/results/qis-s02-calib/  previous=slurm-agentic-engineering/results/qis-s02-calib/.prev-11167475-20261116T044931Z/
- 2026-11-16  **qis-s02-calib2**  COMPLETED  sha=627ad1361d7ba239d3419a6919d73cc980dc8b85  job=11167621  results=slurm-agentic-engineering/results/qis-s02-calib2/
- 2026-11-16  **qis-s02-calib**  COMPLETED  sha=9604281407eb11c5cc8bc455103e9fdc19f294ef  job=11167799  results=slurm-agentic-engineering/results/qis-s02-calib/  previous=slurm-agentic-engineering/results/qis-s02-calib/.prev-11167515-20261116T052939Z/

## 2026-11-16 00:40 EST — F1: base calibration of candidate tasks (E0; base model; sel split; exploratory)
Model Qwen3-1.7B @ 70d244cc, non-thinking (fixture ok). 64 sel problems x 16 samples, temperature 1, cap 4,096.
Jobs 11167799 (commit 9604281, bundle calib-v1) and 11167621 (commit 627ad13, bundle calib2-v1).
pass@1 (problems all-wrong / 64): add9 d3 .702 (2), d4 .502 (8), d5 .229 (20), d6 .181 (17), d8 .041 (48), d10 .000 (64);
add7 d4 .562, d6 .186, d8 .035; add10 d6 .957, d10 .664, d14 .396, d18 .302; mul10 k3 .887, k4 .715, k5 .374;
terse10 k2 .478 (477/1024 unboxed), k3 .031; caesar s4 .270 (4), s8 .082 (25), s12 .045 (40).
Mean length (tokens): add9 534-1,773 (grows with digits); add10 198-470; caesar 406-590; terse 18-80.
Truncation <= 2.4% except countdown n4/n5 (17-19%). add9 wrong answers are genuine (base subscripts accepted).
PENDING REGRADE (grader artefacts found, F2): mbpp .384, arc .688, countdown n3/n4/n5 .853/.520/.242.
Use: target strata for RQ2 with low but measurable success exist in two domains (caesar s8/s12, add9 d8).
- 2026-11-16  **qis-s02-canary**  COMPLETED  sha=0288b416fc1496b2dd124cbf295c5cbf17626c6e  job=11168205  results=slurm-agentic-engineering/results/qis-s02-canary/
- 2026-11-16  **qis-s02-e1-caesar-s1**  COMPLETED  sha=14be5a949ee612ba5ca23c0c92c48906a3af153b  job=11168342  results=slurm-agentic-engineering/results/qis-s02-e1-caesar-s1/

## 2026-11-16 02:40 EST — F3: caesar trained alone (E1, seed 1, sel panels; EXPLORATORY, one seed)
Run qis-s02-e1-caesar-s1 (job 11168342, commit 14be5a9, bundle s02-v1; lr 2e-6; 8 caesar groups/round with strata
s4:s8:s12 = 3:3:2; 32 rounds; evals 64 problems x 4 samples per panel). Delta pass@1 round 32 vs round 0 with 95%
bootstrap over problems (analysis: reports/<ID>/build/analyze.py):
- Own learning: caesar s4 .324 -> .957 (+.633 [+.566,+.699]); s8 .051 -> .883 (+.832 [+.777,+.883]);
  s12 .039 -> .707 (+.668 [+.598,+.734]). A target starting at 4-5% is acquired alone within 32 rounds.
- Off-target, harmful: terse10 k2 .41 -> .07 (-.336 [-.418,-.258]); already -.18 at round 4 and -.30 at round 8.
- Off-target, helpful: add9 d6 +.109 [+.047,+.172]; add9 d4 +.047 and d8 +.008 (intervals include 0).
- Off-target, none detected (|delta| <= .03, intervals within about +-.08): mbpp, arc, countdown n3, add10 d14.
Base rates in this run's round-0 eval (fixed grader): mbpp .590, arc .812, countdown n3 .855 (supersede F1 for these).
Cannot establish: seed robustness; mechanism of the terse drop (length vs format; checking); effect when both train.
F3 addendum (02:40): the terse drop is a length/behaviour failure, not an arithmetic one: over_len (correct but
> 24 tokens) .066 (r0) -> .207 (r4) -> .652 (r8) -> .891 (r32); mean length 19 -> 107 tokens; truncation 0.
- 2026-11-16  **qis-s02-e1-mbpp-s1**  COMPLETED  sha=14be5a949ee612ba5ca23c0c92c48906a3af153b  job=11168352  results=slurm-agentic-engineering/results/qis-s02-e1-mbpp-s1/

## 2026-11-16 02:55 EST — F4: MBPP trained alone (E1, seed 1, sel; EXPLORATORY)
Run qis-s02-e1-mbpp-s1 (job 11168352, commit 14be5a9). Own: mbpp .578 -> .719 (+.141 [+.066,+.223]) over 32 rounds.
Off-target: arc -.043 [-.082,-.012] (small harm); add10 d14 +.074 [+.012,+.145]; all others' intervals include 0
(caesar s4 -.070 [-.141,+.008]). Terse answers stayed 18-20 tokens: code training (which also writes explanations) did
NOT spill into the terse behaviour, unlike caesar training (F3). Noise note: the terse panel (64 x 4) moves .42-.63
within this run without trend and its round-0 value differs between runs (.41 vs .46): single terse readings carry
about +-.1 noise; caesar's -.34 (F3) is well beyond it.
- 2026-11-16  **qis-s02-e1-add9-s1**  COMPLETED  sha=14be5a949ee612ba5ca23c0c92c48906a3af153b  job=11168346  results=slurm-agentic-engineering/results/qis-s02-e1-add9-s1/

## 2026-11-16 03:10 EST — F5: add9 trained alone on mixed strata (E1, seed 1, sel; EXPLORATORY, one seed)
Run qis-s02-e1-add9-s1 (job 11168346, commit 14be5a9; strata d4:d6:d8 = 3:3:2 groups/round; lr 2e-6; 32 rounds).
- Eval pass@1 r0 -> r16 -> r24 -> r32: d4 .520 -> .621 -> .656 -> .766 (+.246 [+.137,+.355]);
  d6 .176 -> .488 -> .094 -> .051 (-.125 [-.195,-.051] vs r0; -.44 vs its r16 peak); d8 .035 -> .023 -> .000 -> .000.
- Training groups (records): d6 had mixed groups through round 25, then ALL-WRONG in rounds 26-31 (success 0; no
  corrective signal), .21 at round 32. d8 groups were all-wrong in 26 of 32 rounds. d4 groups were mixed every round;
  d4 answers lengthened ~600 -> ~770 tokens from round 22.
- Answers (records, decoded): the model's method stays the same (convert to decimal, add, convert back by repeated
  division by 9); r32 errors are wrong long divisions (d6 correct 125/256 at r16 vs 13/256 at r32). d8 truncation
  rises to 10% (r24-32).
- Off-target: terse -.340 [-.418,-.258] via format (r32: 238/256 unboxed "15 x 15 = 225."; length 19 -> 14 tokens),
  a different mechanism from caesar's spill-over (F3: over-long answers). countdown n3 +.047 [+.004,+.090].
Reading: the pattern (easy stratum improves; harder strata rise, then fall into all-wrong groups and stay there)
matches the RQ2 chain inside one task family. NOT yet established: it is one seed; instability at lr 2e-6 is an
alternative. Tests queued (E3a): seed-2 replicate; exposure-matched d6-only, d8-only, d4-only runs.
- 2026-11-16  **qis-s02-j-caesar-terse-s1**  COMPLETED  sha=8e70f56374d623ddd3b15dd45c58e2efe80201d6  job=11168639  results=slurm-agentic-engineering/results/qis-s02-j-caesar-terse-s1/

## 2026-11-16 05:05 EST — F6: J(caesar,terse) vs R(caesar): transient delay of harder strata (E1b, seed 1, sel; EXPLORATORY)
Run qis-s02-j-caesar-terse-s1 (job 11168639, commit 8e70f56, bundle s02-v1 sha a761b981; 8 caesar + 8 terse groups per
round; divisor 32; same T prompts and seeds as R = qis-s02-e1-caesar-s1). D = p_T(R) - p_T(J), paired bootstrap over
64 problems (analyze.py --pairs). Positive D = J behind R.
| stratum | r0 | r8 | r16 | r24 | r32 |
|---|---|---|---|---|---|
| caesar s4 | +.05 [-.02,+.11] | +.16 [+.10,+.23] | +.01 | .00 | +.02 |
| caesar s8 | -.06 [-.09,-.02] | +.10 [+.04,+.16] | +.18 [+.10,+.25] | +.02 | +.01 [-.05,+.07] |
| caesar s12 | .00 | -.01 | +.11 [+.03,+.18] | -.02 | -.07 [-.14,+.01] |
- Training groups (caesar role, R vs J): mixed groups r1-8 37 vs 32, r9-16 51 vs 44; all-wrong r1-8 26 vs 32, r9-16
  11 vs 18. From r17 J has as many or more mixed groups than R. The chain "fewer mixed groups → slower acquisition"
  is visible during the lag; it does not persist.
- Competitor signal: terse groups were mixed in r1-2 (7, 2 of 8), then nearly all-correct (0-3 mixed per round; brief
  length excursions r19-20, r28). Terse in J ends at .93 (+.52 vs r0); caesar training alone had pushed it to .07.
- Classification (one seed): TRANSIENT delay for caesar s8/s12 (and s4 at r8), recovered by r24; terse is protected
  by being trained (no persistent harm in either direction).
Caveats: r0 differs between the two runs by up to .06 on s8 although the model is identical — cross-run evaluation
noise (vLLM sampling not reproducible) of about this size is NOT inside the bootstrap interval. Only r8 (s4), r16
(s8, s12) exceed it clearly. One seed. Cannot establish: whether the delay comes from the r1-2 terse updates, from
the weak later terse signal, or from different rollouts. Tests: seed-2 R/J pair (specs/e1c.json); R(terse) evals of
caesar at r4.
- 2026-11-16  **qis-s02-j-add9-add10-s1**  COMPLETED  sha=8e70f56374d623ddd3b15dd45c58e2efe80201d6  job=11168641  results=slurm-agentic-engineering/results/qis-s02-j-add9-add10-s1/

## 2026-11-16 05:25 EST — F7: J(add9,add10) vs R(add9): no harm to add9; the R d6 collapse is absent in J (E1b, seed 1, sel; EXPLORATORY)
Run qis-s02-j-add9-add10-s1 (job 11168641, commit 8e70f56, bundle s02-v1; 8 add9 + 8 add10 groups/round; divisor 32;
same T prompts and seeds as R = qis-s02-e1-add9-s1). D = p_T(R) - p_T(J), paired bootstrap over 64 problems:
- add9 d4: |D| <= .05 at every eval (intervals include 0). d6: D ~0 to r16, then -.41 [-.52,-.30] (r24), -.48
  [-.57,-.39] (r32): J keeps d6 at .53 while R collapses to .05. d8: J ahead by .04-.12 at r4-r16, both ~0 later.
- Caesar off-target: D +.07 at r12/r16 (borderline); otherwise none. Terse: J falls less than R (D -.29..-.58 at
  r4-r16, ~0 at r32): add10 training (boxed decimal answers) slows the format drift seen in R.
- Training groups (add9 role), R vs J: mixed r25-32 22 vs 31, all-wrong 37 vs 21. add10 role: success .53 -> .73, mostly
  all-correct from r9 (C signal weakens).
- Classification for add9 under add10 (one seed): no harm detected (transfer or independence); the R collapse is
  NOT a joint-training effect.
Important confound (D3): with Adam, the update size does not scale with the loss divisor. In R all of the update
serves add9; in J add9 shares it with add10. So J takes a smaller effective add9 step than R. "J avoids the collapse"
fits both (a) R's collapse is seed noise and (b) R's larger effective step makes add9 training unstable. The
seed-2 replicate (11169764) and d6-only run (11169765) test (a); an R at lower lr would test (b).
- 2026-11-16  **qis-s02-e3-add9mix-s2**  FAILED  sha=102eae682133a7e0d40f80655316edf7bc8d5fa2  job=11169764  results=slurm-agentic-engineering/results/qis-s02-e3-add9mix-s2/
- 2026-11-16  **qis-s02-e3-add9d6-s1**  FAILED  sha=102eae682133a7e0d40f80655316edf7bc8d5fa2  job=11169765  results=slurm-agentic-engineering/results/qis-s02-e3-add9d6-s1/
- 2026-11-16  **qis-s02-e1-caesar-s2**  FAILED  sha=9a6f5601cb4e9d2a8affd9d61b23001a8b1e8f6a  job=11171036  results=slurm-agentic-engineering/results/qis-s02-e1-caesar-s2/
- 2026-11-16  **qis-s02-diag-storage**  FAILED  sha=9a6f5601cb4e9d2a8affd9d61b23001a8b1e8f6a  job=11171686  results=slurm-agentic-engineering/results/qis-s02-diag-storage/

## 2026-11-16 06:40 EST — F8: the add9 d6 collapse replicates in the mixed-strata run and is absent with d6 alone (E3a; sel; EXPLORATORY)
Runs (commit 102eae6, bundle s02-v1): qis-s02-e3-add9mix-s2 (job 11169764; R(add9) config, seed 2) and
qis-s02-e3-add9d6-s1 (job 11169765; 3 d6 groups/round only, divisor 32, seed 1). Both jobs ended FAILED at 06:19:21
because the compute disk filled (incident below) AFTER their round-32 evals were written: all eval rounds 0-32 are
complete. Final checkpoints are missing. add9 d6 pass@1 on sel (64 problems x 4), rounds 0/4/8/12/16/24/32:
| run | d6 | d4 at r32 | d8 at r32 |
|---|---|---|---|
| R mix seed 1 (F5) | .18 .31 .30 .24 .49 .09 .05 | .77 | .00 |
| R mix seed 2 | .15 .20 .28 .30 .00 .00 .07 | .67 | .00 |
| d6 only, seed 1 | .18 .29 .35 .40 .45 .51 .47 | .53 | .11 |
| J(add9,add10) seed 1 (F7) | .14 .27 .30 .36 .45 .50 .53 | .75 | .00 |
- The collapse replicates (2/2 seeds of the mixed run). It is absent when d6 trains alone and when add10 is added.
- Not length or truncation: d6 answer length stays 1,030-1,260 tokens; truncation <= 1%; no unknown statuses.
- d6-only gives all of each Adam step to d6 and does not collapse, so "a larger effective step" alone does not
  explain the collapse (weakens the D3 alternative in F7).
- Off-target: terse rises with d6 only (.40 -> .86) but falls with the mixed run (.46 -> .10, both seeds).
Reading: the collapse needs the mixture (d4 and/or d8 updates together with d6) and is removed by adding add10.
Leading hypothesis: d4 updates push a change that breaks d6 (within-family interference), after which d6 groups are
all-wrong and give no corrective signal (RQ2 chain). Tests: d4-only run (does it lower d6?); per-stratum training
groups of the mixed runs; decoded d6 answers at seed-2 r16 (what changes at the collapse).
Cannot establish yet: which stratum causes it; whether recovery is possible.
- 2026-11-16  **qis-s02-diag-du**  FAILED  sha=9a6f5601cb4e9d2a8affd9d61b23001a8b1e8f6a  job=11171696  results=slurm-agentic-engineering/results/qis-s02-diag-du/
- 2026-11-16  **qis-s02-prune-1**  COMPLETED  sha=9a6f5601cb4e9d2a8affd9d61b23001a8b1e8f6a  job=11171791  results=slurm-agentic-engineering/results/qis-s02-prune-1/
- 2026-11-16  **qis-s02-diag-df**  COMPLETED  sha=ea7363920e57e2be7a84804c3a6cdf0060c0921b  job=11171795  results=slurm-agentic-engineering/results/qis-s02-diag-df/
- 2026-11-16  **qis-s02-e3-add9d4-s1**  COMPLETED  sha=ea7363920e57e2be7a84804c3a6cdf0060c0921b  job=11171814  results=slurm-agentic-engineering/results/qis-s02-e3-add9d4-s1/

## 2026-11-16 07:45 EST — F9: d4-only training helps d6; it does not cause the collapse (E3a, seed 1, sel; EXPLORATORY)
Run qis-s02-e3-add9d4-s1 (job 11171814, commit ea73639, bundle s02-v1; 3 d4 groups/round, divisor 32, 32 rounds).
Delta pass@1 r32 vs r0, 95% problem bootstrap: d4 +.277 [+.176,+.379]; d6 +.211 [+.133,+.297] (.15 → .36, monotone
rise, no drop); d8 -.008 [-.039,+.020]; caesar s8 +.09 [+.05,+.13]; terse -.418 [-.504,-.336].
d4 training groups: mixed or all-correct every round (1-3 mixed of 3).
Reading: within-family transfer d4 → d6 is positive. With F8: the d6 collapse occurs in 2/2 mixed-strata runs and in
0/3 other add9 runs (d6 only, d4 only, mixed + add10). Remaining explanations: (i) d8 updates break d6 (d8-only run
pending); (ii) only the combination does it; (iii) chance (2/2 vs 0/3 is weak evidence by itself).
Terse note: d4-only also drops terse (-.42), d6-only raised it (+.46) — the format drift is not a simple function of
add9 training; unexplained.
- 2026-11-16  **qis-s02-e1-caesar-s2**  COMPLETED  sha=ea7363920e57e2be7a84804c3a6cdf0060c0921b  job=11171812  results=slurm-agentic-engineering/results/qis-s02-e1-caesar-s2/  previous=slurm-agentic-engineering/results/qis-s02-e1-caesar-s2/.prev-11171036-20261116T125513Z/

## 2026-11-16 07:58 EST — F3 replicated: caesar alone, seed 2 (E1c; sel; EXPLORATORY)
Run qis-s02-e1-caesar-s2 (job 11171812, commit ea73639). pass@1 r0 → r8 → r16 → r32: s4 .31 → .63 → .84 → .97;
s8 .07 → .23 → .57 → .82; s12 .04 → .13 → .39 → .64; terse .46 → .16 → .02 → .06. Seed 1 (F3): s4 .96, s8 .88,
s12 .71, terse .07 at r32. Both the acquisition of all caesar strata and the terse drop replicate across seeds.
Paired J(caesar,terse) seed 2 (11171816) started 07:54 on preempt; the F6 replication waits for it.
- 2026-11-16  **qis-s02-e3b-r16-d6n32**  COMPLETED  sha=baeda2978613560bde82d0ce1cf10831a1c5b6b4  job=11171994  results=slurm-agentic-engineering/results/qis-s02-e3b-r16-d6n32/
- 2026-11-16  **qis-s02-e3b-r16-mixed**  COMPLETED  sha=baeda2978613560bde82d0ce1cf10831a1c5b6b4  job=11171949  results=slurm-agentic-engineering/results/qis-s02-e3b-r16-mixed/
- 2026-11-16  **qis-s02-e3b-r16-d6**  COMPLETED  sha=baeda2978613560bde82d0ce1cf10831a1c5b6b4  job=11171953  results=slurm-agentic-engineering/results/qis-s02-e3b-r16-d6/

## 2026-11-16 08:47 EST — F10: recovery from the d6 collapse needs a success; scarce feedback keeps the deficit (E3b; EXPLORATORY)
Three 8-round branches from qis-s02-e3-add9mix-s2 round 16 (d6 sel .000, conf .004), commit baeda29, no checkpoints:
mixed = original mixture (11171949); d6 = 3 d6 groups x 8 samples (11171953); d6n32 = 3 d6 groups x 32 samples
(11171994). Same d6 prompts in all three. pass@1 at r16 / r20 / r24:
| branch | d6 conf (64 x 8) | d6 sel (64 x 4) | d4 sel | d6 groups with any success (rounds 17-24) |
|---|---|---|---|---|
| d6, 8 samples | .004 / .000 / .000 | .000 / .000 / .000 | .56 / .40 / .39 | 0 of 24 |
| d6n32 | .004 / .008 / .113 | .000 / .000 / .148 | .55 / .53 / .63 | 1 of 24 (round 20; 2 of 96 correct) |
| mixed | .004 / .016 / .072 | .000 / .000 / .070 | .53 / .68 / .81 | (d4 supplies 2-3 mixed groups per round) |
Paired differences at r24 (64 problems, 95% bootstrap): d6n32 − d6: conf +.113 [+.066,+.172], sel +.148 [+.086,+.215];
d6 − mixed: conf −.072 [−.117,−.035], sel −.070 [−.117,−.031]; d6n32 − mixed: conf +.041 [−.002,+.092].
Reading:
- With zero successes, d6 does not recover at all in 8 rounds (and d4 decays .56 → .39 without d4 training). Adam
  momentum carried over from the mixed state does not rescue it. This is the RQ2 chain's persistence step directly
  observed: no successes → all-wrong groups → zero advantage → no recovery.
- One rare success (found by sampling 4x more) is enough to start recovery: after a single mixed group in round 20,
  d6 rises to .11-.15 by round 24 while every later d6 group was again all-wrong (momentum amplifies that one step).
- Transfer from a related stratum (d4 updates, F9) also restarts d6 partly (+.07 vs d6-only).
Cannot establish: one collapse state, one branch per condition (round-20 success in d6n32 is a single event);
whether longer runs converge. Applies to Adam with these hyperparameters.
- 2026-11-16  **qis-s02-j-caesar-terse-s2**  COMPLETED  sha=ea7363920e57e2be7a84804c3a6cdf0060c0921b  job=11171816  results=slurm-agentic-engineering/results/qis-s02-j-caesar-terse-s2/

## 2026-11-16 08:56 EST — F11: F6 replicates — terse training delays the harder caesar strata (E1c, seed 2, sel; EXPLORATORY)
Pair qis-s02-e1-caesar-s2 (R, job 11171812) / qis-s02-j-caesar-terse-s2 (J, job 11171816), commit ea73639, same
seed-2 T prompts. D = p(R) − p(J), paired bootstrap over 64 problems. Here r0 agrees (|D| <= .02), unlike seed 1.
| stratum | r8 | r12 | r16 | r24 | r32 |
|---|---|---|---|---|---|
| s4 | +.04 | +.05 | +.09 [+.02,+.16] | −.04 | +.06 [+.01,+.11] |
| s8 | −.01 | +.07 | +.12 [+.05,+.20] | +.04 | +.12 [+.05,+.20] |
| s12 | +.04 | +.14 [+.07,+.21] | +.15 [+.07,+.23] | +.15 [+.07,+.22] | +.07 [−.01,+.16] |
Both seeds: J is behind R on s8 and s12 at r16 (seed 1: +.18, +.11; seed 2: +.12, +.15). Seed 1 closed the gap by
r24; seed 2 still shows it at r32 on s8 (+.12) — "transient" is not established for seed 2 within 32 rounds.
Training groups (caesar role, R vs J): mixed r9-16 51 vs 46, all-wrong 9 vs 13 (same direction as seed 1).
Competitor signal: terse groups mixed 13 (r1-8), 5, 2, 2 (later blocks): terse is learned early and goes quiet.
Answer length (caesar training responses, tokens, R vs J): seed 1 r5-8 562 vs 513, r9-16 603 vs 533; seed 2 r5-8
492 vs 449, r9-16 479 vs 456; equal from r17. Joint-run caesar answers are 5-15% shorter exactly during the lag.
Reading: a competing behavioural objective (short answers) slows a skill that needs written working, more for the
harder strata, while the competitor is learned. Mechanism suggested (brevity spill-over), not tested.
Terse is protected by training (J terse .9+ vs R terse ~.06).
- 2026-11-16  **qis-s02-e3-add9d8-s1**  COMPLETED  sha=ea7363920e57e2be7a84804c3a6cdf0060c0921b  job=11171817  results=slurm-agentic-engineering/results/qis-s02-e3-add9d8-s1/

## 2026-11-16 09:00 EST — F12: d8-only training harms d4 and d6 — the likely source of the collapse (E3a, seed 1, sel; EXPLORATORY)
Run qis-s02-e3-add9d8-s1 (job 11171817, commit ea73639; 2 d8 groups/round, divisor 32, 32 rounds). Delta r32 vs r0,
95% problem bootstrap: d4 −.246 [−.352,−.141] (.52 → .27); d6 −.148 [−.215,−.082] (.18 → .03); d8 −.027 [−.055,.000]
(not learned); terse +.383 [+.301,+.469]; add10 +.09 [.00,+.18]; caesar, mbpp, arc, countdown: none.
d8 training groups: mixed in 12 of 32 rounds (all in r4-r21; never both groups), all-wrong otherwise. So every
update came from groups with 1-2 correct of 8. d4/d6 answer length unchanged (646/1,033 → 630/848 tokens), no
truncation: the harm is to the arithmetic, not length or format.
Reading (combined with F8-F10): within add9, d4 updates help d6 (F9); updates from the near-impossible d8 stratum
(rare successes → large positive advantage on one sample) harm d4 and d6. In the mixed run both act: d6 first rises,
then the d8 harm wins; once d6's groups are all-wrong there is no corrective signal and the deficit persists (F10),
until a rare success or d4 transfer restarts it. This is the RQ2 chain with its source identified.
Cannot establish yet: one seed of d8-only; why adding add10 prevents the collapse (F7).
Test: d4+d6 mixture without d8 (predicts no collapse).
- 2026-11-16  **qis-s02-an-strata**  COMPLETED  sha=5b173e669a9d5819cb2ade1a1af1e242474e35e6  job=11172404  results=slurm-agentic-engineering/results/qis-s02-an-strata/

## 2026-11-16 09:40 EST — CORRECTION to F12 reading + F13: per-stratum groups at the collapse (an-strata, job 11172404)
Per-stratum training groups from the training records (results/qis-s02-an-strata/strata.txt):
- Seed 2 mixed run: d8 groups were mixed ONLY in rounds 1 and 4; all-wrong (zero gradient) in rounds 2-3 and 5-32.
  d6 training success: .50 (r12), .29 (r13: 1 mixed, 2 all-wrong), then .00 with 3 all-wrong groups in r14-26.
  So the seed-2 collapse (eval .30 at r12 → .00 at r16) happened while d8 contributed no gradient for 9 rounds.
- Seed 1 mixed run: d8 mixed in rounds 4, 6, 8, 16, 19, 21; d6 groups all-wrong from r26 to r31.
CORRECTION: F12's reading "d8 updates are the likely source of the collapse" does NOT hold for seed 2. d8 updates
in r1 and r4 did not prevent d6 from rising to .30 by r12. In seed 2 only d4 and d6 updates were active in the
rounds before the collapse. F12's direct result (d8 alone harms d4/d6) stands; its role in the mixed-run collapse is
unresolved. E3d (d4+d6 without d8, running) now discriminates: a collapse there means d8 is not needed.
Decoded answers (seed 2; answers.txt, few examples, not counted): at r16 the method is unchanged, but the decimal
value of the operands is wrong (e.g. sum 548,129 or 547,129 for a correct 533,129; a d4 operand 4,322 that should be
4,282); the final division by 9 is done correctly. At r12 errors are smaller (532,129 vs 533,129).

## 2026-11-16 09:40 EST — SECOND CRITICAL REVIEW of F8-F12 (independent agent, raw metrics only; mandate requirement)
Reviewer recomputed every number from metrics.jsonl (paired bootstrap over 64 problems, 10k draws); all pass@1 values
match to ±.001. Verdicts and the corrections they force:
- C1 collapse in 2/2 mixed runs, 0/3 controls: SUPPORTED (n=2). Collapse drops: seed 1 r16→r24 −.395 [−.48,−.31];
  seed 2 r12→r16 −.305 [−.39,−.22]. Caveat: the mixed+add10 run differs in more than data (32 responses/step,
  grad norms up to 63, 71/128 steps clipped) — the add9 share of each Adam step is diluted there.
- C2 d4 helps d6: SUPPORTED (+.211; +.184 [+.105,+.266] against the higher r0 of the same model).
- C3 d8 harms d4/d6: PARTIAL. CORRECTION to F12: "answer length unchanged" is true for d4 only; d6 answers shorten
  1,033 → 848 tokens and d8 1,610 → 1,099. Only 11/128 steps had nonzero gradient; harm appears mostly after r16.
- C4 recovery: PARTIAL. CORRECTIONS to F10: (1) the d6 8-sample branch had exactly zero gradient on all 32 steps, so
  it measured momentum-only drift from the checkpoint's Adam state, which alone cut d4 by −.164 [−.26,−.07];
  "stays at 0" follows from zero gradient by construction (that IS the scarcity point, but no d6 update was taken).
  (2) The d6n32 gain comes from ONE step (round 20; grad norm 1.26, clipped); with the loss divided by a fixed 32, a
  32-sample group carries 4x the loss weight of an 8-sample group, so "more samples" is confounded with "larger
  step". (3) d6n32 − mixed is not significant on conf (+.041 [−.004,+.09]).
- C5 mechanism: PARTIAL. Strongest alternative: Adam amplifies sparse noisy updates (with ~90% zero-gradient steps,
  a rare one-correct group gets a step ~3x larger than in a dense run and momentum carries it ~10 steps) — generic
  drift, not d8-specific. Also: untargeted drift harms skills that get no gradient. No existing run separates these.
  Reviewer's best discriminating run: d4:d6 = 3:3 plus 2 groups of a random-reward stratum matched in sparsity.
- Eval noise floor: same-model evals in different jobs differ by up to .043 (d4) / .035 (d6). Effects of .15-.44 are
  far above it; .07 effects are ~2x it.
Scripts: scratchpad review/a.py, review/b.py (session-local).
REVIEW ADDENDUM (09:42 EST, reviewer's second pass with strata.txt; per-stratum counts verified to sum to by_role.T
in all 32 rounds of all 5 runs): C5 NOT SUPPORTED for d8 as the collapse trigger; persistence part SUPPORTED.
Mixed d8 groups before collapse: mix-s2 2 (r1, r4); e1 6 (r4-r21); mixed+add10 16 (r1-23) with NO collapse — more
d8 gradient went with less collapse. mix-s2 persistence: 39 all-wrong d6 groups r14-26, then single mixed groups in
r27, r29, r31; eval d6 .07 at r32. Outlier: the largest gradient norm of mix-s2 (3.06, clipped) is round 12
minibatch 3, just before the collapse; round 13 has 1.19 (clipped); d8 was all-wrong in both, so these came from
d4/d6 groups. Alternatives left: d4×d6 interference; one large update destabilising d6; sparse-update drift.
Reviewer's best single run = d4:d6 3:3 without d8, seed 2 — already running (E3d, 11173240), plus seed 1 (11173237).
- 2026-11-16  **qis-s02-e3-add9mix-s3r**  COMPLETED  sha=8abdf9346a5be0e85d6600d812804704fb85ba7c  job=11173101  results=slurm-agentic-engineering/results/qis-s02-e3-add9mix-s3r/

## 2026-11-16 10:28 EST — F14: add9 mixed seed 3 does NOT collapse (E3c; sel; EXPLORATORY)
Run qis-s02-e3-add9mix-s3r (job 11173101, commit 8abdf93; R(add9) mixed-strata config, seed 3). pass@1 r0/4/8/12/16/
24/32: d6 .18 .28 .34 .24 .32 .45 .54; d4 .52 → .82; d8 .04 → .00; terse .40 → .03.
Collapse count in the mixed-strata configuration: 2 of 3 seeds (seed 1 r16→r24, seed 2 r12→r16; seed 3 none, with a
dip at r12 .34 → .24 that recovered). Reading: the collapse is a stochastic event of this configuration, not a fixed
consequence of the mixture. Retained checkpoints r16/r24 are not collapse states: E3b cannot be repeated from them.
- 2026-11-16  **qis-s02-j-caesar-mbpp-s1**  COMPLETED  sha=29cb12d80c109be80384d0d1237be40851525ea3  job=11173146  results=slurm-agentic-engineering/results/qis-s02-j-caesar-mbpp-s1/

## 2026-11-16 10:35 EST — F15: J(caesar,mbpp): caesar lags late; MBPP unharmed; terse protected (E1d, seed 1, sel; EXPLORATORY)
Run qis-s02-j-caesar-mbpp-s1 (job 11173146, commit 29cb12d; 8 caesar + 8 mbpp groups/round, divisor 32; same T
prompts and seeds as R = qis-s02-e1-caesar-s1). D = p(R) − p(J), paired bootstrap over 64 problems:
- caesar: r16 s4 +.05, s8 +.07, s12 +.13 [+.06,+.20]; r32 s4 +.10 [+.05,+.15], s8 +.09 [+.02,+.16], s12 +.12
  [+.05,+.21]. Own gains r0→r32: s4 +.58, s8 +.69, s12 +.55 (alone: +.63, +.83, +.67). Not closed by r32.
- MBPP vs R(mbpp) (qis-s02-e1-mbpp-s1): r32 D +.03 [−.02,+.07] — MBPP learning not detectably slowed.
- Terse (not trained): .40 → .68 (r32), answers stay 19-20 tokens. Caesar alone drove terse to .07 (F3); MBPP alone
  left it noisy around .4-.6 (F4). In the joint run terse is higher than in either single run (vs R(mbpp) at r32:
  −.26 [−.33,−.19], i.e. J higher). Panel noise ~±.1 (F4).
Confound (D3): MBPP gives active gradient throughout (mixed groups per 8-round block printed in the direction log),
so caesar shares each Adam step with MBPP; part of the caesar lag may be a smaller effective caesar step rather than
interference. (In J(caesar,terse) the competitor was silent after r2, so this confound is small there.)
Classification (one seed): caesar under MBPP — delay of all strata, persistent to r32; MBPP under caesar — none.
Cross-domain pair is asymmetric, not "independent". Terse spill-over from caesar is absent in the joint run.
- 2026-11-16  **qis-s02-e3-add9d4d6-s2**  COMPLETED  sha=4eb0bd40bd48a2e3b3ac7d1a0c341ceb7d58574b  job=11173240  results=slurm-agentic-engineering/results/qis-s02-e3-add9d4d6-s2/

## 2026-11-16 10:41 EST — F16 (provisional): d4+d6 without d8, seed 2 — no collapse (E3d; sel; EXPLORATORY)
Run qis-s02-e3-add9d4d6-s2 (job 11173240, commit 4eb0bd4; 3 d4 + 3 d6 groups, divisor 32, seed 2 = same d4/d6
prompt streams as qis-s02-e3-add9mix-s2). pass@1 r0/4/8/12/16/24/32: d6 .15 .23 .29 .32 .50 .57 .65; d4 .48 → .88;
d8 (never trained) .04 → .18. The mixed seed-2 run (same d4/d6 prompts, plus 2 d8 groups) collapsed d6 to .00 at
r16 and drove d8 to .00. Largest pre-clip gradient norms: d4+d6 1.04, 0.98, 0.90; mixed s2 3.06 (round index 11),
2.71, 1.71.
Reading: removing d8 coincided with no collapse and with positive transfer to d8, in the seed where d8 gave only 2
nonzero-advantage groups (r1, r4). Either those two early d8 updates changed the trajectory, or the difference is
chance (collapse rate 2/3 in the mixed configuration; vLLM sampling differs between any two runs). One pair cannot
separate these. Seed 1 pair (11173237) pending.
- 2026-11-16  **qis-s02-e3-add9d4d6-s1**  COMPLETED  sha=4eb0bd40bd48a2e3b3ac7d1a0c341ceb7d58574b  job=11173237  results=slurm-agentic-engineering/results/qis-s02-e3-add9d4d6-s1/
F16 UPDATE (11:09 EST): d4+d6 seed 1 (qis-s02-e3-add9d4d6-s1, job 11173237, 4eb0bd4) — no collapse either: d6 .16 .20
.30 .34 .38 .53 .50; d4 .49 → .81; d8 .04 → .01. Its mixed twin (seed 1) fell .49 → .09 → .05.
Collapse count: with d8 groups 2/3 seeds (mixed s1, s2; not s3) and 0/1 with d8 + add10; without d8 0/2 so far
(s3 running). Fisher exact for 2/3 vs 0/2: p ≈ 0.4 — suggestive only. The "single large update" lead weakens: the
d4+d6 s1 run had a larger pre-clip norm (2.12) than its collapsing mixed twin (max 1.19) and did not collapse.
How d8 could matter when it gives almost no gradient (seed 2: two nonzero-advantage groups) is unexplained; the
two runs of a pair also diverge through non-reproducible vLLM sampling, so the pairing is weak.
- 2026-11-16  **qis-s02-e1-caesar-lr1e6-s1**  COMPLETED  sha=2399deeee439f7727b5c2fcc96385d0424ec0dd1  job=11174276  results=slurm-agentic-engineering/results/qis-s02-e1-caesar-lr1e6-s1/

## 2026-11-16 12:45 EST — F17: step-size control — the caesar lag under MBPP is within what a smaller step explains (E1e)
Run qis-s02-e1-caesar-lr1e6-s1 (job 11174276, commit 2399dee; R(caesar) seed 1 at lr 1e-6 instead of 2e-6). Gains
r0→r32: s4 +.49 [+.41,+.58], s8 +.30 [+.24,+.36], s12 +.30 [+.24,+.36]; at lr 2e-6 (R): +.63, +.83, +.67. Halving
the step lowers caesar at r32 by .14 / .53 / .37. J(caesar,mbpp) lags R by only .10 / .09 / .12 (F15).
Reading: under Adam, sharing each step with an active competitor gives caesar an effective step between half and
full size. The MBPP-pair lag is well inside the range that a smaller step produces, so F15 cannot be attributed to
interference; "step sharing" is a sufficient explanation. Terse spill-over also scales with step size (−.10 at half
lr vs −.34 at full lr). For F6/F11 (terse competitor), the competitor was silent after round 2, so step sharing is
small there and this control does not explain that delay.
- 2026-11-16  **qis-s02-e3-add9d4d6-s3**  COMPLETED  sha=305dc56fcd067442385d7da0c7ec503ecac5470e  job=11174313  results=slurm-agentic-engineering/results/qis-s02-e3-add9d4d6-s3/
F16 FINAL (13:03 EST): d4+d6 seed 3 (qis-s02-e3-add9d4d6-s3, job 11174313, 305dc56; preempted twice, resumed from its
r16 checkpoint) — no collapse: d6 .17 .33 .34 .45 .42 .50 .59; d4 .52 → .74; d8 .04 → .07. Collapse count: with d8
groups 2/3, without d8 0/3 (Fisher exact two-sided p = 0.4, one-sided 0.2). Suggestive that d8 groups raise collapse
risk; not significant; how d8 matters when it gives so little gradient is unexplained. Note: metrics.jsonl of this run
holds round records 17-24 twice (replayed after the requeue); evals are not duplicated; use the last record.
- 2026-11-16  **qis-s02-j-add9-terse-s1**  COMPLETED  sha=29cb12d80c109be80384d0d1237be40851525ea3  job=11173148  results=slurm-agentic-engineering/results/qis-s02-j-add9-terse-s1/

## 2026-11-16 13:06 EST — F18: J(add9,terse): no harm to add9; terse learned (E1d, seed 1, sel; EXPLORATORY)
Run qis-s02-j-add9-terse-s1 (job 11173148, commit 29cb12d; 8 add9 groups (d4:d6:d8 3:3:2) + 8 terse groups; same add9
prompts and seed as R = qis-s02-e1-add9-s1). J pass@1 r0 → r32: d4 .52 → .82, d6 .18 → .52 (no collapse; dip .48
→ .40 at r12 reversed), d8 .04 → .11, terse .40 → .96 by r4, .93 at r32. D = p(R) − p(J) at r32: d4 −.06
[−.12,.00], d6 −.47 [−.55,−.38], d8 −.11 [−.15,−.06]: J is ahead, chiefly because R(add9) seed 1 collapsed (F5).
Classification (one seed): add9 under terse — no harm detected; terse under add9 — protected (learned), whereas
add9 alone drove terse to .06 (F5).
Collapse count for the add9 mixture: alone 2/3; with a competitor (add10, terse) 0/2; d4+d6 without d8 0/3.
A competitor shares the Adam step (F17 shows step size matters), but d4+d6 runs take a full add9 step and also did
not collapse, so step size alone does not explain the pattern. Unresolved.
- 2026-11-16  **qis-s02-e1-caesar-s3**  COMPLETED  sha=d2c519977c335cfa527f592a97cf03c8f9c0e99f  job=11175477  results=slurm-agentic-engineering/results/qis-s02-e1-caesar-s3/
- 2026-11-16  **qis-s02-j-caesar-terse-s3**  COMPLETED  sha=d2c519977c335cfa527f592a97cf03c8f9c0e99f  job=11175478  results=slurm-agentic-engineering/results/qis-s02-j-caesar-terse-s3/

## 2026-11-16 14:12 EST — F19: caesar × terse seed 3 — no delay; the delay is 2/3 seeds and tracks answer length (E1f)
Pair qis-s02-e1-caesar-s3 (job 11175477) / qis-s02-j-caesar-terse-s3 (job 11175478), commit d2c5199, no checkpoints.
R learns as before (r32: s4 .92, s8 .85, s12 .66; terse .40 → .29, a weaker drop than seeds 1-2).
D = p(R) − p(J): r16 s4 +.03, s8 +.07 [−.02,+.17], s12 +.02; r32 s8 −.02, s12 −.08 [−.17,+.01]; s4 at r8/r12 −.12/−.13
(J ahead). No delay. Training groups (caesar role) equal in r9-16 (mixed 51 vs 51, all-wrong 13 vs 13).
Caesar training-answer length, R vs J: r1-8 491 vs 522; r9-16 535 vs 566 (J LONGER); seeds 1-2 had J 5-15% shorter.
Terse (competitor) mixed groups: 11, 4, 3, 0 per 8-round block (as in seeds 1-2).
Combined (3 seeds): the harder-strata delay appears in 2/3 seeds (s1, s2) and is absent in s3. In all three seeds the
delay goes with shorter caesar answers in the joint run during rounds 1-16, and its absence with no shortening.
Supports brevity spill-over as the mechanism (correlational, n=3; not tested by intervention).
- 2026-11-16  **qis-s02-e3-add9d4d6-s4**  COMPLETED  sha=338c9942f5436045a4fab6f1cb1e5297cf5fd81d  job=11176693  results=slurm-agentic-engineering/results/qis-s02-e3-add9d4d6-s4/
- 2026-11-16  **qis-s02-e3-add9d4d6-s5**  COMPLETED  sha=338c9942f5436045a4fab6f1cb1e5297cf5fd81d  job=11176696  results=slurm-agentic-engineering/results/qis-s02-e3-add9d4d6-s5/
- 2026-11-16  **qis-s02-e3-add9mix-s5**  COMPLETED  sha=338c9942f5436045a4fab6f1cb1e5297cf5fd81d  job=11176692  results=slurm-agentic-engineering/results/qis-s02-e3-add9mix-s5/
- 2026-11-16  **qis-s02-e3-add9mix-s4**  COMPLETED  sha=338c9942f5436045a4fab6f1cb1e5297cf5fd81d  job=11176691  results=slurm-agentic-engineering/results/qis-s02-e3-add9mix-s4/
F16 FINAL-2 (16:25 EST, E3g seeds 4-5, commit 338c994, no checkpoints): mixed s4 (11176691) d6 .18 → .68, mixed s5
(11176692) d6 .16 → .62, d4+d6 s4 (11176693) .17 → .48, d4+d6 s5 (11176696) .17 → .57 — none collapsed.
Collapse count: mixed with d8 2/5; d4+d6 without d8 0/5 (Fisher exact two-sided p = 0.44); with a competitor 0/2.
Revised reading: the d6 collapse is an occasional event of the mixed configuration (2/5 seeds); its association with
d8 is not significant. RQ2 persistence/recovery findings (F10) are unaffected.
