# When does learning one skill impede another? Interference dynamics in GRPO on Qwen3-1.7B

Study `Nov-15-2026-02-vineeth-babel-slurm-interference-dynamics` (s02). Final report, 2026-11-16. Every result is
exploratory: one to three seeds per condition, selection-split panels unless marked "conf". Numbers were audited
(`research/<ID>/audit-2026-11-16.md`) and the add9 findings had an independent critical review.

## 1. Purpose and main results

This study asks two questions about reinforcement learning with GRPO on Qwen3-1.7B (non-thinking mode).

- **RQ1.** When two tasks train together, when does learning one impede the other? Classify each pair as transfer,
  independence, transient harm, or persistent deficit.
- **RQ2.** Does on-policy feedback amplify an initial disadvantage? The proposed chain is: an adverse update lowers
  target success, which yields fewer mixed groups, which slows acquisition.

Main results:

1. **A behavioural objective can delay a skill that needs written working, when it spills into its answers.**
   Training "terse" answers (no working, at most 24 tokens) together with Caesar decoding delayed the harder Caesar
   strata by 0.11 to 0.18 at round 16 in 2 of 3 seeds. One of those seeds closed the gap by round 24; the other
   still lagged by 0.12 at round 32 on one stratum. In both, Caesar answers were 5-15% shorter in the joint run
   during the lag. The third seed showed no shortening and no delay. (F6, F11, F19)
2. **Training one task often damages an untrained behaviour.** Caesar or base-9 addition (add9) training lowered
   the untrained terse task from about 0.4 to 0.03-0.29 (Caesar: 3 of 3 seeds; size varies by seed), in two
   different ways: over-long answers (Caesar) and unboxed answers (add9). Training Caesar together with MBPP, or
   training terse itself, protected it. (F3, F5, F15, F19)
3. **Inside one task family, a skill can collapse and stay collapsed because it stops receiving feedback.** When
   add9 trained on a mixture of 4-, 6- and 8-digit problems, the 6-digit stratum (d6) rose and then fell to near
   zero in 2 of 5 seeds. Once all d6 training groups were all-wrong, d6 received no gradient. From the collapsed
   state, a branch with zero d6 successes did not recover in 8 rounds; one rare success (found by sampling 4x more)
   or transfer from d4 training restarted recovery. This is the persistence step of the RQ2 chain. (F8, F10, F14)
4. **The trigger of that collapse is not identified; it is an occasional event.** It is not d8 gradients (d8 gave
   almost no gradient before the seed-2 collapse). Without d8 groups, 0 of 5 seeds collapsed, against 2 of 5 with
   them (Fisher exact p = 0.44); with a second task added (add10 or terse), 0 of 2 collapsed. No condition
   difference is significant. (F13, F16, F18, independent review)
5. **A joint-run lag is not by itself evidence of interference.** Caesar trained with MBPP lagged Caesar alone by
   0.09-0.12 at round 32 while MBPP was unaffected. Halving Caesar's learning rate alone costs 0.14-0.53, so the lag
   fits Adam simply sharing each step between two active tasks. Add9 trained with decimal addition (add10) showed no
   harm. The Caesar-terse delay (result 1) is the one cross-task delay that step sharing does not explain, because
   terse gave almost no gradient after round 2. (F7, F15, F17)

## 2. Setup

| Item | Value |
|---|---|
| Model | Qwen/Qwen3-1.7B @ `70d244cc`, `enable_thinking=False` |
| Algorithm | GRPO: 8 samples per prompt, std-normalised group advantage (all-equal groups get 0), clip 0.2, no KL |
| Optimiser | AdamW lr 2e-6, betas 0.9/0.999, weight decay 0, grad clip 1.0; 4 optimiser steps per round |
| Loss reduction | token mean per response, then sum divided by a fixed 32 responses per minibatch |
| Round | 8 target groups (+ 8 competitor groups in joint runs); 32 rounds per run |
| Sampling | vLLM BF16, temperature 1, cap 4,096 tokens |
| Evaluation | 64 problems x 4 samples per panel (sel); 64 x 8 or 16 (conf) for probes |
| Data | bundle `s02-v1`, SHA-256 `a761b981...` (`data/<ID>/BUNDLES.md`); sel, conf and train splits disjoint |
| Uncertainty | 95% percentile bootstrap over problems (2,000 draws); paired by problem for run comparisons |

Tasks (exact graders; artefacts found in calibration were fixed before training, commit `602b151`):

| Task | What it is | Base pass@1 (sel) |
|---|---|---|
| add9 d4 / d6 / d8 | add two base-9 numbers of 4 / 6 / 8 digits | .50 / .18 / .04 |
| caesar s4 / s8 / s12 | decode a Caesar cipher, shift 4 / 8 / 12 | .27-.32 / .05-.08 / .04 |
| terse10 k2 | multiply two 2-digit numbers, answer only, at most 24 tokens | .40-.48 |
| add10 d14 | add two 14-digit decimal numbers | .40-.60 |
| MBPP | Python functions, sandboxed unit tests | .59 |
| ARC, countdown n3 | multiple choice science; arithmetic target game | .81, .85 |

Terms. **Reference R(T)**: T trained alone with the same prompts and seeds as the joint run. **Joint J(T,C)**: 8 T
groups + 8 competitor groups per round. **D(t) = p_T(R) − p_T(J)**: positive means the joint run is behind. A group is
**all-wrong**, **mixed**, or **all-correct**; only mixed groups produce a gradient.

## 3. RQ1: pair interactions

![Caesar D(t)](figures/caesar_D.png)

*Figure 1. D(t) per Caesar stratum for three joint runs. Bands: 95% paired bootstrap over 64 problems.*

| Target T | Competitor C | Effect on T | Effect on C | Class (T under C) | Seeds | Source |
|---|---|---|---|---|---|---|
| caesar | terse | harder strata delayed in 2/3 seeds: D(s8, s12) +.11..+.18 at r16; none in seed 3 | terse learned (.4 → .9+), protected | delay in 2/3 seeds (transient in one), tracks answer shortening | 3 | F6, F11, F19 |
| caesar | MBPP | all strata behind at r32: D +.09..+.12 | MBPP unaffected (D +.03, n.s.) | lag explained by step sharing (half-lr control costs .14-.53) | 1 | F15, F17 |
| add9 | add10 | no harm; d6 collapse absent | not analysed | none detected | 1 | F7 |
| add9 | terse | no harm (J ahead of the collapsed R: d6 −.47) | terse learned (.4 → .93), protected | none detected | 1 | F18 |

Single-task off-target effects at round 32 (Figure 2) show that a task's own training can damage an untrained one:

![Off-target](figures/offtarget.png)

*Figure 2. Change in pass@1 from round 0 to 32 on every panel, per trained task. Bold: 95% CI excludes 0.*

- Caesar training lowers terse by 0.34-0.40 (2 seeds) through over-long answers (19 → 107 tokens); the
  multiplication itself is intact (F3).
- Mixed add9 and d4-only training lower terse by 0.34-0.42 through unboxed answers (F5, F9); d6-only and d8-only
  training raise it (+0.46, +0.38). This sign difference is unexplained.
- MBPP training slightly lowers ARC (−0.04) (F4).
- d8-only training lowers d4 (−0.25) and d6 (−0.15) while d8 itself is not learned (F12).

## 4. RQ2: does on-policy feedback amplify a disadvantage?

### 4.1 The add9 d6 collapse

![add9 d6](figures/add9_d6.png)

*Figure 3. add9 d6 pass@1. Colour = training condition; line style = seed.*

| Condition | Seeds | d6 collapse | d6 at r32 |
|---|---|---|---|
| mixed d4+d6+d8 (3:3:2 groups) | 5 | 2 of 5 (seed 1 r16→r24, seed 2 r12→r16) | .05, .07, .54, .68, .62 |
| d4+d6, no d8 | 5 | 0 of 5 | .50, .65, .59, .48, .57 |
| d6 only | 1 | no | .47 |
| d4 only | 1 | no (d6 rises .15 → .36) | .36 |
| d8 only | 1 | d6 falls .18 → .03 | .03 |
| mixed + add10 | 1 | no | .53 |
| mixed + terse | 1 | no | .52 |

- **The chain is visible.** In seed 2, d6 training success went .50 (r12), .29 (r13), then 0 with all three d6
  groups all-wrong in rounds 14-26: 39 consecutive groups with no gradient (training records, F13).
- **The method stays the same at the collapse.** Decoded answers still convert to decimal, add, and convert back;
  the errors appear in converting the operands (few examples inspected, not counted).
- **Not length or truncation.** d6 answer length stays 1,030-1,260 tokens; truncation ≤ 1%.

### 4.2 Recovery requires a success

![Recovery](figures/recovery.png)

*Figure 4. Eight-round branches from the seed-2 collapse state (round 16). Grey: the parent run before the branch.*

| Branch (8 rounds from r16) | d6 groups with any success | d6 conf at r24 | d4 sel at r24 |
|---|---|---|---|
| d6 only, 8 samples | 0 of 24 (zero gradient on all 32 steps) | .000 | .39 (from .55) |
| d6 only, 32 samples | 1 of 24 (round 20; 2 of 96 correct) | .113 | .63 |
| original mixture | d4 supplies 2-3 mixed groups per round | .072 | .81 |

Paired differences at r24 (conf): 32 samples − 8 samples +0.113 [+0.066, +0.17]; 8 samples − mixture −0.072
[−0.12, −0.035]. Reading: with no successes the deficit persists; a success from sampling or transfer from a related
stratum ends it. Caveats: the 8-sample branch took no gradient step at all (its weights moved only by Adam momentum
from the checkpoint, which alone lowered d4 by 0.16); the 32-sample gain comes from one step, and a 32-sample group
also carries 4x the loss weight of an 8-sample group, so "more samples" and "a larger step" are not separated.

### 4.3 Caesar under terse

During the lag (rounds 9-16), the joint runs had fewer mixed Caesar groups than the references (seed 1: 44 vs 51;
seed 2: 46 vs 51) and more all-wrong groups (18 vs 11; 13 vs 9). The chain direction holds, but the effect is small
and closes in one seed. Seed 3 had no delay and equal group counts (51 vs 51 mixed; 13 vs 13 all-wrong).

| Seed | D(s8) at r16 | D(s12) at r16 | Caesar answer length r9-16, alone vs joint |
|---|---|---|---|
| 1 | +.18 [+.10, +.25] | +.11 [+.03, +.18] | 603 vs 533 tokens (joint 12% shorter) |
| 2 | +.12 [+.05, +.20] | +.15 [+.07, +.23] | 479 vs 456 (5% shorter) |
| 3 | +.07 [−.02, +.17] | +.02 [−.05, +.08] | 535 vs 566 (6% longer) |

The delay appears only in the seeds where terse training shortened Caesar answers. This fits brevity spill-over;
with three seeds it is a correlation, not a test.

### 4.4 What the evidence supports

| RQ2 step | Status |
|---|---|
| adverse update lowers target success | observed (d6 collapse, Caesar lag); cause of the adverse update unresolved |
| lower success → fewer mixed groups | observed (add9 seed 2: 39 all-wrong d6 groups; Caesar: fewer mixed groups) |
| fewer mixed groups → slower acquisition / persistence | supported: zero successes → no recovery in 8 rounds; one success → recovery |
| amplification (the gap grows because of the feedback loop) | partial: the collapsed state persists, but recovers when any success appears |

## 5. Evidence table

| Claim | Experiments (expid) | Result | Status |
|---|---|---|---|
| C-1 Caesar acquires all strata alone | e1-caesar-s1, -s2, -s3 | s12 .04 → .64-.71 | replicated (3 seeds) |
| C-2 Caesar training lowers terse via long answers | e1-caesar-s1, -s2, -s3 | −.34, −.40, −.11 [−.19, −.04] | 3 of 3 seeds; size varies |
| C-3 terse delays harder Caesar strata | j-caesar-terse-s1/-s2/-s3 vs e1-caesar-s1/-s2/-s3 | D(r16) s8 +.18/+.12/+.07, s12 +.11/+.15/+.02 | 2 of 3 seeds; tracks answer shortening |
| C-4 add10 / terse do not harm add9 | j-add9-add10-s1, j-add9-terse-s1 vs e1-add9-s1 | |D(d4)| ≤ .06; d6 no collapse in either | 1 seed each |
| C-5 MBPP slows Caesar beyond step sharing | j-caesar-mbpp-s1, e1-caesar-s1, e1-mbpp-s1, e1-caesar-lr1e6-s1 | D(r32) +.09..+.12; half lr costs .14-.53 | not supported: step sharing suffices |
| C-6 mixed add9 can collapse d6 | e1-add9-s1, e3-add9mix-s2, -s3r, -s4, -s5 | 2 of 5 seeds | supported (reviewed); occasional |
| C-7 d4 training helps d6 | e3-add9d4-s1 | +.21 [+.13, +.30] | 1 seed (reviewed) |
| C-8 d8 training harms d4, d6 | e3-add9d8-s1 | −.25, −.15 | 1 seed; d6 answers shorten (reviewed) |
| C-9 d8 triggers the collapse | e3-add9mix-s2 strata; e3-add9mix-s1..s5; e3-add9d4d6-s1..s5 | 2 d8 updates before seed-2 collapse; 0/5 without d8 vs 2/5 with (p=0.44) | not supported; open |
| C-10 no successes → no recovery; a success restarts it | e3b-r16-d6 / -d6n32 / -mixed | .000 / .113 / .072 | 1 collapse state (reviewed, partial) |

## 6. What this study cannot establish

- **Seed robustness** beyond 2-3 seeds; most single-run results are one seed.
- **Generality** beyond Qwen3-1.7B, lr 2e-6, Adam, and these task families.
- **Which update causes the add9 collapse.** Candidates left: d4×d6 interaction, one large update, sparse-update
  drift under Adam, a weak effect of d8 groups.
- **Mechanism of the Caesar delay.** Brevity spill-over fits the length data in all three seeds but was not tested
  by intervention.
- **Step size versus interference** in joint runs with an active competitor. Adam gives each step about the same
  size whatever the number of tasks, so each task's share shrinks when another task is active; the loss divisor
  matches raw gradient weight only (decision D3). The half-lr control (F17) bounds this for Caesar only.
- **Absence** of an effect: "no harm detected" means the 95% interval includes 0 at about ±0.08.

## 7. Validity and incidents

- **Evaluation noise.** Same-model evaluations in different jobs differ by up to 0.04 (vLLM sampling is not
  reproducible). Effects of 0.15-0.45 are far above it; effects near 0.07 are about twice it.
- **Second critical review.** An independent agent recomputed F8-F12 from raw metrics (all values within ±0.001)
  and rejected "d8 triggers the collapse" (section 4, C-9). Corrections are in `state/FINDINGS.md`.
- **Disk full (06:19 EST).** Study checkpoints (256 GB) filled the shared 500 GB disk; three runs died after or
  during their final evaluations. Fix: recipes refuse to start below 60 GB free and prune checkpoints at the end.
- **Bad node.** babel-l9-32 failed every CUDA start; excluded.
- **Grader artefacts.** Three were found and fixed before training (MBPP multi-block code, countdown boxed value,
  ARC unboxed letter).

## 8. Deferred questions

Interventions (e.g. replay, per-task step normalisation), pre-training effects, model capacity, and optimiser
choice (SGD vs Adam; this study shows Adam momentum moves weights after zero-gradient steps) are deferred.

## 9. Runs

See `runs.md` for each experiment's job ID, commit, recipe hash and result path.
