# Does factoring practice help GRPO learn hard math? Qwen3-1.7B, arms H, R and U

Final report under owner amendment A17 (deadline Wed 11 Nov 2026, 7:30 PM EST). DRAFT — cells marked ⟨PENDING⟩ await
jobs that are still running. The first report (10 Nov) is kept unchanged in `archive/REPORT-v1-2026-11-10.md`.

**Status of the evidence.** The frozen 128-problem sealed test (section 1) is the primary result. The common-GPU rerun
(section 1) is a reproducibility check. Everything else — the seed-reversal analyses (section 3), the local-update audit
(section 4) and the two stress panels (section 5) — is exploratory: it was added after the design was frozen, and it
uses outputs that were already visible.

## Four practical answers (A19)

### Q1. Was related practice worth the training slots?

Not on this evidence. Training helped every arm by a wide margin, but factoring practice (R) against modular practice
(U) changed sign between seeds. Sealed test, pass@1, 128 problems × 32 attempts (95% paired problem bootstrap):

| Seed | Base | H | R | U | R − U | R − H | U − H |
|---|---|---|---|---|---|---|---|
| 101 | 52.6% | 77.0% | 81.9% | 77.7% | 81.9 − 77.7 = +4.2 [+2.1, +6.3] | 81.9 − 77.0 = +4.9 [+2.0, +8.0] | 77.7 − 77.0 = +0.7 [−2.0, +3.5] |
| 202 | 52.6% | ⟨PENDING⟩ | 74.8% | 79.2% | 74.8 − 79.2 = −4.4 [−7.2, −1.7] | ⟨PENDING⟩ | ⟨PENDING⟩ |

What replicates across seeds: trained − base (+22 to +29 points in all five finished runs), and the sign of R − U
on the stress panels following the seed (section 5). What does not replicate: the sign of R − U. R − H and U − H exist
for seed 101 only until the H-s202 endpoint lands. ⟨PENDING: H-s202.⟩

Cost per run on its canonical trajectory. Every run used 128 rounds × 16 prompts × 8 answers = 16,384 answers; H spends
all 16 prompts on hard problems, R and U spend 8 of 16 on practice.

| Run | Hard answers | Prompt tokens (M) | Generated tokens (M) | GPU-h |
|---|---|---|---|---|
| H-s101 | 16,384 | 1.67 | 68.6 | 29.5 |
| R-s101 | 8,192 | 1.45 | 65.1 | 27.9 |
| U-s101 | 8,192 | 1.44 | 60.0 | 22.6 |
| R-s202 | 8,192 | 1.48 * | 54.1 | 22.8 |
| U-s202 | 8,192 | 1.44 | 52.8 | 25.6 |
| H-s202 | ⟨running⟩ | 1.45 at round 111 | 62.3 at round 111 | ⟨final cost run⟩ |

Prompt tokens count each prompt once per answer. * R-s202's counter includes 3 rounds replayed after a resume (D127);
its generated tokens are from the attempt records. GPU-h mix GPU types (training nodes differed), so tokens are the
hardware-free cost. After round 32, 62–92% of R and U practice groups were all-correct and gave no learning signal.

### Q2. Did the policy lose accuracy, or did it stop finishing within budget?

⟨PENDING: descriptive decomposition (section 3.3) beside the within-trajectory length experiment (section 9), per seed.⟩

### Q3. When did the difference emerge, and did it reach harder problems?

**When.** In both seeds the endpoint sign of R − U appeared between rounds 32 and 64, the first window after the cap
rose from 8,192 to 12,288 tokens (section 9.2). Dev R − U at round 32 → 64: seed 101 −5.9 → +2.6 points; seed 202
−1.2 → −6.6 points. Training success on the shared hard prompts split in the same window (seed 101: R ahead from rounds
41–48 on; seed 202: U ahead in rounds 33–48). These are measured trajectories from training records. Checkpoint
coverage has a gap: no same-seed R/U pair survives below round 128, so no intermediate pair could be put on the stress
panels or the sealed test.

**Harder problems.** Yes, the pattern extends, and SuperHard was harder in fact, not only by label. Stress panels, pass@1
(share of attempts cut off at 12,288 tokens), 8 attempts per problem:

| Tier | Base | R-s101 | U-s101 | R − U s101 | R-s202 | U-s202 | R − U s202 |
|---|---|---|---|---|---|---|---|
| Harder (95) | 52.5% (0.1%) | 84.2% (9.9%) | 80.1% (11.4%) | 84.2 − 80.1 = +4.1 [+0.7, +7.8] | 76.3% (13.4%) | 79.5% (9.2%) | 76.3 − 79.5 = −3.2 [−6.7, +0.3] |
| SuperHard (126) | 25.8% (0.0%) | 58.4% (28.5%) | 54.2% (31.6%) | 58.4 − 54.2 = +4.3 [+0.5, +8.0] | 53.3% (30.3%) | 55.6% (19.6%) | 53.3 − 55.6 = −2.3 [−5.6, +1.0] |

The base model solved half as often on SuperHard as on Harder (25.8% vs 52.5%), and trained models hit the cap two to
three times as often there. R − U takes the sign of the seed in both tiers. In seed 202 on SuperHard, R was more
accurate on finished answers (76.4% vs 69.1%) but was cut off more often (30.3% vs 19.6%). The only round-64 stress
cells are R-s101 (Harder 78.6%, SuperHard 53.5%) and H-s202 (76.1%, 52.1%); they come from different arms and seeds and
cannot date the R/U split.

### Q4. What should the next experiment change?

⟨PENDING: after the length diagnostic.⟩

## What we learned

1. **Training gain versus the choice of practice.** On the sealed test, every trained policy beat the base model by a
   wide margin: the base model scored 52.6% pass@1, and the five finished trained policies scored 74.8%–81.9%, a gain
   of +22 to +29 points. The choice of practice changed that by much less and in no consistent direction: R − U was
   +4.2 points in seed 101 (R 81.9% vs U 77.7%) and −4.4 points in seed 202 (R 74.8% vs U 79.2%); their mean is −0.1
   points. Rerunning R and U on one GPU type keeps both signs (+4.4 and −3.1 points; section 1). ⟨PENDING: H-s202 endpoint; R − H and U − H for seed 202.⟩
2. **Which use of the eight practice slots?** The two seeds do not support a choice between factoring and modular
   practice. In seed 101, factoring practice (R) beat additional hard practice (H) by 4.9 points (R 81.9% vs H 77.0%),
   and modular practice (U) matched H (U 77.7% vs H 77.0%). ⟨PENDING: seed 202 H.⟩ Cost does not decide it either: per
   run, R and U generated 52.8–65.1 M tokens and used 22.6–27.9 GPU-h on their canonical trajectories, against 68.6 M
   tokens and 29.5 GPU-h for H-s101. After round 32, 62–92% of R and U practice groups were all-correct and carried no
   learning signal, so the practice slots mostly cost generation time without updating the model.
3. **What produced the seed reversal.** Mostly response length. In seed 202, R's answers were cut off at the
   12,288-token cap more often than U's (13.1% vs 8.1% of 4,096 attempts). That accounts for −4.3 of the −4.4-point
   R − U difference; on answers that finished, R and U were equally accurate (86.1% vs 86.2%). In seed 101 the pattern
   flips (R 10.7% vs U 12.2% truncated) and explains +1.4 of the +4.2 points. The difference is reliability on problems
   both arms can solve, not coverage: in seed 202, 122 of 128 problems were solved at least once by both arms, and they
   carry −4.2 of the −4.4 points. Band B1 (34 problems the base model solved 1–5 times in 16) carries −5.7 of the −8.6-point
   swing between seeds. Per-problem R − U values are unrelated across seeds (Spearman −0.09).
4. **The ray-interference mechanism.** Link 1 (an auxiliary update lowers hard success): not detected; one unrelated
   update did not lower the hard probe by 3 points or more relative to a related or hard update, at the base model or at
   R-s101 round 64. Link 2 (affected problems then produce fewer rewarding attempts): observed rarely and briefly; 2.5–5.0%
   of later appearances of previously solvable hard prompts were all-wrong, and most such prompts returned to mixed at
   their next appearance. Link 3 (learning then stalls): not observed. The competing explanation that fits the same
   evidence best is **length drift**: an arm-and-seed-specific tendency to write longer answers, which hits the cap,
   zeroes the reward and appears as "forgetting" in the records and as a seed-dependent R − U on the test. The long
   answers are unconverged searches, not loops, and training-time answer length on the shared hard prompts does not
   predict which arm truncates more on the test (R wrote shorter training answers than U in both seeds).
5. **Transfer to harder problems.** Training transfers; the arm choice again does not. On 221 new, harder factoring
   problems (section 5), the base model scored 37.3% pass@1 and the trained checkpoints 62.4–69.5%. SuperHard pass@1
   roughly doubled (25.8% → 52.1–58.4%). R − U again took the sign of the seed: 69.5 − 65.3 = +4.2 [+1.5, +6.8] in seed
   101 and 63.2 − 65.8 = −2.7 [−5.1, −0.2] in seed 202. In seed 202, R was more accurate on the answers it finished
   (82.1% vs 77.6%) but hit the cap more often (23.0% vs 15.2%). A blind audit of 96 responses found that more factoring
   practice would plausibly fix about 40% of the failures: both readers said so for 13 of 33 genuine failed responses
   (29 of 66 judged reads); the rest are later errors or cap hits. Two more failures were grader errors. ⟨PENDING: H-s202 r128.⟩
6. **The one larger experiment to run next.** ⟨Draft:⟩ Rerun R and U with more seeds (at least 6 per arm) **and** a
   length control (the same cap for every round, with length-matched reward or a larger cap for evaluation), and report
   R − U both overall and on finished answers. The dominant uncertainty is run-to-run variation that acts through answer
   length; more seeds without a length control would measure it but not remove it. The recommendation would change if
   R − U on finished answers (the accuracy term) agreed in sign across seeds and exceeded the seed spread: then the
   factoring effect is real and a larger single-cap R/U study is the right next step.

## 1. Endpoints: original and common-GPU evaluations

Sealed target_test: 128 MATH problems × 32 attempts, cap 12,288, frozen verifier. pass@1 weights each problem equally.
Intervals are 95% paired bootstraps over the 128 problems (10,000 draws). They describe these trained checkpoints. Two
training seeds say little about the next training run.

**Primary: original evaluations**

| Seed | H | R | U | R − U | R − H | U − H |
|---|---|---|---|---|---|---|
| 101 | 77.0% | 81.9% | 77.7% | 81.9 − 77.7 = +4.2 [+2.1, +6.3] | 81.9 − 77.0 = +4.9 [+2.0, +8.0] | 77.7 − 77.0 = +0.7 [−2.0, +3.5] |
| 202 | ⟨PENDING⟩ | 74.8% | 79.2% | 74.8 − 79.2 = −4.4 [−7.2, −1.7] | ⟨PENDING⟩ | ⟨PENDING⟩ |
| Mean | ⟨PENDING⟩ | 78.3% | 78.5% | 78.3 − 78.5 = −0.1 [−1.8, +1.5] | ⟨PENDING⟩ | ⟨PENDING⟩ |

Base model: 52.6% pass@1, 83.5% pass@8. Differences are in percentage points.

**Reproducibility check: one GPU type.** The original evaluations did not share hardware. In both seeds, R ran on an
L40 GPU and U on an L40S (`sacct` node lists, D141). The common recipe (frozen in D141 before any rerun score) uses
L40S for every cell, unchanged code, and the original per-item sampling seeds. Same seeds isolate the GPU type: a U
rerun should reproduce its original exactly, and an R rerun changes only the hardware. H-s101 and the base model
already ran on L40S under this recipe, so their originals are the common cells.

| Seed | H | R | U | R − U | R − H | U − H |
|---|---|---|---|---|---|---|
| 101 | 77.0% | 82.3% | 77.8% | 82.3 − 77.8 = +4.4 [+2.4, +6.5] | 82.3 − 77.0 = +5.3 [+2.5, +8.2] | 77.8 − 77.0 = +0.9 [−1.8, +3.5] |
| 202 | ⟨PENDING⟩ | 76.1% | 79.2% | 76.1 − 79.2 = −3.1 [−6.0, −0.4] | ⟨PENDING⟩ | ⟨PENDING⟩ |
| Mean | ⟨PENDING⟩ | 79.2% | 78.5% | 79.2 − 78.5 = +0.7 [−1.0, +2.3] | ⟨PENDING⟩ | ⟨PENDING⟩ |

How closely each rerun reproduced its original (4,096 attempts each; every request seed identical):

| Checkpoint | GPU: original → rerun | Identical token sequences | Identical score per problem (c/32) | pass@1 rerun − original |
|---|---|---|---|---|
| R-s101 | L40 → L40S | 48/4,096 | 62/128 | 82.3 − 81.9 = +0.4 [−0.8, +1.6] |
| R-s202 | L40 → L40S | 133/4,096 | 46/128 | 76.1 − 74.8 = +1.3 [+0.2, +2.5] |
| U-s101 | L40S → L40S | 1,964/4,096 | 79/128 | 77.8 − 77.7 = +0.1 [−1.0, +1.2] |
| U-s202 | L40S → L40S | 4,096/4,096 | 128/128 | 79.2 − 79.2 = +0.0 |

The GPU type does not explain the seed reversal. On one GPU type, R − U is still +4.4 points in seed 101 and −3.1
in seed 202. A rerun moved pass@1 by at most 1.3 points, even where most token sequences changed. Even the same GPU type
does not guarantee identical sequences: U-s202 reproduced exactly, but U-s101 matched on only 48% of attempts (a
different node and a run two days later; the cause is unknown).

![Original and common-GPU endpoints](figures/a17/endpoints.png)

## 2. Learning curves

![Learning curves](figures/a17/learning_curves.png)

Rows 1–2: held-out dev (53 problems × 8 attempts) and the three probes against training round and against cumulative
hard-prompt training responses. Row 3: success on the practised shared hard prompts in training (not a held-out
evaluation). The dotted line marks the cap change from 8,192 to 12,288 tokens after round 32. H trains on 16 hard prompts
per round and R/U on 8, so H reaches a given hard exposure in half the rounds. ⟨PENDING: H-s202 a2-r60 to round 128.⟩

**Probes.** `J_H`: 32 hard dev problems (held out). `J_R`: 28 factoring practice-style problems (held out from training
pools). `J_U`: 32 modular-arithmetic practice-style problems (held out). All at 8 attempts during training.

![J_H against J_R](figures/a17/pope_JH_vs_JR.png)
![J_H against J_U](figures/a17/pope_JH_vs_JU.png)

## 3. Why the two seeds disagree (exploratory)

### 3.1 Which problems carry the reversal

![Per-problem R − U, seed 101 against seed 202](figures/a17/reversal_scatter.png)

Each point is one test problem: its R − U success difference (c/32) in seed 101 against seed 202. Of 128 problems, 31
have the same sign in both seeds, 44 opposite signs, and 53 a tie in at least one seed (31 tie in both: 3 at 0/32 for
all four policies and 26 at 32/32). The per-problem differences are unrelated across seeds (Spearman −0.09 overall,
−0.04 within B1).

| Band (base k of 16) | Problems | R − U seed 101 | R − U seed 202 | Contribution to the swing (s202 − s101) |
|---|---|---|---|---|
| B0 (0) | 15 | 30.6 − 29.8 = +0.8 | 18.1 − 28.3 = −10.2 | −1.3 |
| B1 (1–5) | 34 | 75.5 − 65.8 = +9.7 | 59.2 − 71.0 = −11.8 | −5.7 |
| B2 (6–10) | 26 | 88.1 − 85.1 = +3.0 | 85.5 − 88.3 = −2.9 | −1.2 |
| B3 (11–15) | 32 | 95.9 − 93.5 = +2.4 | 93.8 − 91.2 = +2.6 | +0.0 |
| B4 (16) | 21 | 99.7 − 97.9 = +1.8 | 98.2 − 99.4 = −1.2 | −0.5 |
| All | 128 | 81.9 − 77.7 = +4.2 | 74.8 − 79.2 = −4.4 | −8.6 |

Bands use base attempts 0–15; a band's contribution is its share of 128 times its R − U. B1 carries two-thirds of the
swing. Within B1, the 3 largest item changes carry 30% of B1's share and the top 5 carry 45%; the rest is spread over
many items whose order changed. Example: `algebra/test/00213` went from R 23/32 vs U 13/32 in seed 101 to R 11/32 vs
U 24/32 in seed 202.

### 3.2 Coverage or reliability

| Policy | pass@1 | pass@8 | Solved ≥ 1 of 32 | 0/32 | Truncated attempts |
|---|---|---|---|---|---|
| base | 52.6% | 83.5% | 116/128 | 12/128 | 2/4,096 |
| H-s101 | 77.0% | 93.7% | 125/128 | 3/128 | 445/4,096 (10.9%) |
| R-s101 | 81.9% | 96.1% | 125/128 | 3/128 | 437/4,096 (10.7%) |
| U-s101 | 77.7% | 94.5% | 125/128 | 3/128 | 501/4,096 (12.2%) |
| R-s202 | 74.8% | 92.2% | 123/128 | 5/128 | 538/4,096 (13.1%) |
| U-s202 | 79.2% | 94.9% | 124/128 | 4/128 | 333/4,096 (8.1%) |
| H-s202 | ⟨PENDING⟩ | | | | |

Against an independent baseline (base attempts 16–31), training did both, but mostly improved reliability. 14 problems
were unreachable at baseline (0/16); training made 9–11 of them reachable, at low pass@1 (19–40%). On the 114 reachable
problems, pass@1 rose from 59.6% to 82.5–88.0%. Of each policy's total gain over base (+21.7 to +28.8 points), the
reachable problems carry +20.4 to +25.3. For R − U the split is the same: in seed 202 R − U is −4.4 points, of which
−4.2 comes from the 122 problems both arms reach and −0.2 from the 3 that only one arm reaches.

![Coverage](figures/a17/coverage.png)

### 3.3 Truncation: the main driver of R − U

A truncated answer scores 0. With t the truncated share and a the accuracy on finished answers, pass@1 = (1 − t)·a, and
R − U = −(t_R − t_U)·ā + (1 − t̄)·(a_R − a_U) exactly (bars are R/U means).

| Seed | R − U pass@1 | Truncation term | Accuracy term |
|---|---|---|---|
| 101 | 81.9 − 77.7 = +4.2 | t: 10.7% vs 12.2% → +1.4 [+0.0, +2.8] | a: 91.6% vs 88.5% → +2.8 [+0.9, +4.7] |
| 202 | 74.8 − 79.2 = −4.4 | t: 13.1% vs 8.1% → −4.3 [−6.1, −2.7] | a: 86.1% vs 86.2% → −0.1 [−2.4, +2.0] |

![Truncation and R − U](figures/a17/truncation.png)

Per problem, R − U success falls as R − U truncation rises (Pearson −0.51 in seed 101, −0.59 in seed 202). Counting
only answers that finish within 8,192 tokens, R − U is +5.1 [+3.1, +7.3] (R 76.5% vs U 71.4%) in seed 101 and −7.0
[−10.0, −4.2] (R 68.6% vs U 75.7%) in seed 202. The reversal is therefore not created by the 12,288 cap; it follows
which policy writes longer answers. The truncated answers are not degenerate loops: 0 of R-s202's 538 truncated answers repeat ≥ 50% of their final 32-token windows (median repeated share 4.7%). They are long searches that do not converge; for example, on `intermediate_algebra/test/00220`, a product that telescopes once n⁴ + ¼ is factored, R-s202 tried numeric patterns for 12,288 tokens and never factored. In training, R-s202 also generated more tokens than U-s202 late in training (16.4 M
vs 15.1 M in rounds 97–128); in seed 101 the two were close (19.0 M vs 19.1 M).

### 3.4 How much learning signal each arm received

Groups of 8 answers per prompt. Only mixed groups (some right, some wrong) carry a GRPO gradient. A mixed-group count
shows reward variation; it does not give the size or direction of the update.

| Run | Rounds | Shared hard: all-wrong / mixed / all-correct | Second pool: all-wrong / mixed / all-correct |
|---|---|---|---|
| R-s101 | 1–32 / 97–128 | 27 / 220 / 9 → 9 / 119 / 128 | factoring 0 / 180 / 76 → 0 / 25 / 231 |
| U-s101 | 1–32 / 97–128 | 22 / 222 / 12 → 13 / 122 / 121 | modular 0 / 168 / 88 → 0 / 20 / 236 |
| R-s202 | 1–32 / 97–128 | 27 / 214 / 15 → 11 / 130 / 115 | factoring 0 / 180 / 76 → 0 / 44 / 212 |
| U-s202 | 1–32 / 97–128 | 28 / 212 / 16 → 8 / 119 / 129 | modular 0 / 166 / 90 → 0 / 29 / 227 |
| H-s101 | 1–32 / 97–128 | 32 / 216 / 8 → 5 / 111 / 140 | extra hard 31 / 212 / 13 → 8 / 102 / 146 |

Each cell counts 256 groups (32 rounds × 8 prompts). The practice pools became easy: after round 32, the share of
all-correct practice groups was 62–90% for R and 74–92% for U, so most practice prompts gave no reward contrast.
The shared hard prompts gave similar signal in R and U. Full windows, tokens, gradient norms (logged per optimizer step)
and clip fractions (per round): `research/evidence/a17/analysis/train/signal.md`.

### 3.5 Did apparent forgetting persist?

Shared hard prompts, same prompt and round matched across arms. "Previously solvable" (PS) means ≥ 1 correct answer at an
earlier appearance in the same run.

| Run | Rounds | PS → all-wrong (of later PS appearances) | Capped answers in those groups: 0 / some / all 8 | Next appearance: all-wrong / mixed / all-correct / none |
|---|---|---|---|---|
| R-s101 | 33–128 | 29 / 765 (3.8%) | 12 / 13 / 4 | 18 / 10 / 0 / 1 |
| U-s101 | 33–128 | 38 / 768 (4.9%) | 7 / 23 / 8 | 17 / 18 / 0 / 3 |
| R-s202 | 33–128 | 19 / 765 (2.5%) | 13 / 3 / 3 | 7 / 10 / 0 / 2 |
| U-s202 | 33–128 | 26 / 768 (3.4%) | 18 / 6 / 2 | 13 / 11 / 1 / 1 |
| H-s101 | 33–128 | 13 / 768 (1.7%) | 11 / 2 / 0 | 8 / 4 / 0 / 1 |

Rounds 1–32 (cap 8,192) are in `persistence.md`. Matched cells, all-wrong in R only vs U only: seed 101 17 vs 24,
seed 202 11 vs 16 (rounds 33–128). Losses were rare, not much more frequent in U than R, and usually temporary: at the
next appearance most such prompts were mixed again or still all-wrong; almost none jumped to all-correct. One capped
answer in a group does not show that a longer allowance would have rescued it.

![Transitions on the shared hard prompts](figures/a17/transitions.png)

### 3.6 How often sampling alone creates an apparent loss

The first report described dev "losses" as "≥ 1 correct of 16 → 0". That was wrong: every scheduled dev evaluation used
**8** attempts per problem (all 28 dev rows and the per-attempt records show n = 8; `dev_transitions.md`). The loss
counts themselves were correct.

At an unchanged checkpoint, two disjoint groups of 8 attempts make a problem look lost (some → none) with probability
P = (1 − (1 − p)^8)(1 − p)^8, which peaks at 0.25 when p ≈ 0.08. On the 128 test problems, the expected count of
chance losses per comparison is 1.7–5.7 depending on the policy (observed with fixed splits: 0–5). On the dev panel,
each run lost 0–5 problems between consecutive evaluations (53 problems × 8 attempts), the same range that sampling
alone produces. Splits reuse the same attempts, so these rates are a reference, not independent experiments.

![Sampling-only loss rate](figures/a17/sampling_null.png)

## 4. Local evidence for interference (exploratory)

One GRPO/AdamW step on 4 prompts × 8 answers of one kind (hard, related = factoring, unrelated = modular), from the
base model (fresh optimizer) and from R-s101 round 64 (its saved optimizer), then `J_H`, `J_R`, `J_U` at 16 attempts per
problem. Four minibatch draws per branch. The U-s101 round-64 checkpoint was not retained, so that source is missing.

![One-step changes](figures/a17/delta_j.png)

| Checkpoint | No-update J_H (original / rerun) | Hard (mean of 4) | Related | Unrelated | Unrelated − related | Unrelated − hard |
|---|---|---|---|---|---|---|
| base (r0) | 40.0% / 43.2% | 41.5% | 42.7% | 40.5% | 40.5 − 42.7 = −2.2 | 40.5 − 41.5 = −0.9 |
| R-s101 r64 | 60.6% / 63.1% | 60.6% | 61.6% | 60.9% | 60.9 − 61.6 = −0.7 | 60.9 − 60.6 = +0.2 |

Full per-draw table with J_R and J_U, mixed groups and gradient norms: `delta_j_table.md` and `branches.md`.

- The no-update reference is itself noisy: two evaluations of the same base weights on different GPUs gave J_H 40.0% and
  43.2% (r64: 60.6% and 63.1%). A change measured against one reference therefore cannot be read as absolute; the direct
  branch comparisons above do not depend on it.
- Single draws spread by up to 4 points within a branch (unrelated at r64: 59.2%–63.3%). We do not claim a detection
  threshold; differences of 2–3 points between branch means are within this spread.
- At r64, three draws (related 4101, 4102; unrelated 4103) had no mixed group, so their reward gradient was exactly 0.
  AdamW still stepped from its stored moments, and the three produced identical scores (J_H 61.1%, J_R 89.5%, J_U 93.2%).
  This is a "momentum-only" step, not an unchanged checkpoint. At r0 the optimizer is fresh, and no draw had zero mixed
  groups.

**Mechanism, one link at a time.**

| Link | Evidence | Kind | Finding |
|---|---|---|---|
| 1. An auxiliary update lowers hard success | branch experiment (section 4) | direct, 2 checkpoints, 1 arm | not detected; unrelated vs related −2.2 / −0.7 points, within draw spread |
| 2. Affected problems then produce fewer rewarding attempts | training records (3.5) | observational | rare (2.5–4.9% of later PS appearances after round 32), often with capped answers, mostly temporary |
| 3. Their learning then stalls | training records, dev curves | observational | not observed; from round 0 to 128, dev pass@1 rose from 40–42% to 66–73% and practised shared-hard success (8-round bins) from 21–26% to 75–87% in every finished run, with dips of up to about 5 points between evaluations |
| Endpoint difference | sealed test (1, 3.3) | outcome only | R − U flips sign with seed and follows truncation; it cannot establish the chain |

## 5. Stress panels: Harder and SuperHard (exploratory)

**Construction.** Rubric first (`research/evidence/a17/panels/RUBRIC-stress-v1.md`, committed before any candidate was
exported). Harder: a substantive factoring step must be recognised inside a longer solution, with no cue. SuperHard:
Harder plus a documented obstacle (hidden substitution, approach choice, case analysis after factoring, non-obvious
grouping, or several factorizations). Candidates: 2,039 unused problems from MATH (pinned, Levels 4–5), Omni-MATH and
OlympiadBench; every candidate was annotated by an agent (labelled `agent:claude-sonnet`). Selection by hash order with
family, source and duplicate rules (`qis.data.stress_panels`); checks against every training, calibration, practice,
dev, probe, cue and sealed-test text. Frozen before any model was evaluated on them (D143, re-frozen in D145 after the
second read found one cross-source duplicate).

| Panel | Items | Sources | Notes |
|---|---|---|---|
| SuperHard-126 | 126 | MATH 60, Omni-MATH 52, OlympiadBench 14 | obstacles: case analysis 45, hidden substitution 29, non-obvious grouping 26, approach choice 14, several factorizations 12 |
| Harder-95 | 95 | MATH 69, Omni-MATH 15, OlympiadBench 11 | — |

Fewer than 128 items qualified in each panel: the whole 2,039-item pool was annotated, and only 126 SuperHard and 99
Harder items met the rubric and the checks. A blind second read (different agents, problem and solution only) agreed
on eligibility for 207 of 219 items, on the tier for 171 of 219 (78%), and confirmed every gold answer (219/219).
Absence from the study's sets does not show absence from pretraining data.

**Results.** 8 attempts per problem, cap 12,288, L40S, seed label stress-v1; every cell has 1,768/1,768 attempts and 0
ungradable answers, and every trained checkpoint's model.pt hash matched its record. H-s202 round 128 runs only after
that training run ends (section 8).

| Checkpoint | All 221 pass@1 | SuperHard-126 | Harder-95 | Truncated (all) | Accuracy on finished (all) | Solved ≥ 1 of 8 (all) |
|---|---|---|---|---|---|---|
| Base | 37.3% | 25.8% | 52.5% | 0.1% | 37.3% | 152/221 |
| H-s101 r128 | 64.1% | 52.6% | 79.5% | 17.7% | 77.9% | 183/221 |
| R-s101 r128 | 69.5% | 58.4% | 84.2% | 20.5% | 87.4% | 184/221 |
| U-s101 r128 | 65.3% | 54.2% | 80.1% | 23.0% | 84.8% | 184/221 |
| H-s202 r128 | ⟨PENDING⟩ | | | | | |
| R-s202 r128 | 63.2% | 53.3% | 76.3% | 23.0% | 82.1% | 180/221 |
| U-s202 r128 | 65.8% | 55.6% | 79.5% | 15.2% | 77.6% | 188/221 |
| R-s101 r64 | 64.3% | 53.5% | 78.6% | 26.1% | 87.0% | 183/221 |
| H-s202 r64 | 62.4% | 52.1% | 76.1% | 15.1% | 73.5% | 183/221 |

Contrasts on all 221 problems, in points, with the same exact split into a truncation term and an accuracy term as in
section 3.3 (95% paired bootstrap over problems, 10,000 draws):

| Contrast | pass@1 A − B | Truncation term | Accuracy term | pass@8 A, B |
|---|---|---|---|---|
| R − U, seed 101 | 69.5 − 65.3 = +4.2 [+1.5, +6.8] | t: 20.5% vs 23.0% → +2.1 [+0.4, +3.9] | a: 87.4% vs 84.8% → +2.0 [−0.1, +4.1] | 83.3%, 83.3% |
| R − U, seed 202 | 63.2 − 65.8 = −2.7 [−5.1, −0.2] | t: 23.0% vs 15.2% → −6.3 [−8.2, −4.3] | a: 82.1% vs 77.6% → +3.6 [+1.2, +6.1] | 81.4%, 85.1% |
| R − H, seed 101 | 69.5 − 64.1 = +5.4 [+2.5, +8.2] | t: 20.5% vs 17.7% → −2.3 [−4.6, +0.0] | a: 87.4% vs 77.9% → +7.7 [+5.2, +10.1] | 83.3%, 82.8% |
| U − H, seed 101 | 65.3 − 64.1 = +1.2 [−1.4, +3.8] | t: 23.0% vs 17.7% → −4.3 [−6.3, −2.2] | a: 84.8% vs 77.9% → +5.5 [+2.9, +8.0] | 83.3%, 82.8% |
| R-s101 r128 − r64 | 69.5 − 64.3 = +5.3 [+3.0, +7.6] | t: 20.5% vs 26.1% → +4.9 [+3.0, +6.9] | a: 87.4% vs 87.0% → +0.3 [−1.4, +2.0] | 83.3%, 82.8% |

![Stress panels](figures/a17/stress_panels.png)

What the stress panels show:
1. **Training transfers to harder problems.** Every trained checkpoint gains 25–32 points over the base model on all 221
   problems, and SuperHard pass@1 roughly doubles (25.8% → 52.1–58.4%). Each checkpoint solves 33–38 problems that the base
   model never solved in 8 attempts. It loses 2–5 that the base model solved, fewer than the ≈ 12 that a second 8-attempt
   draw from the base model would lose by chance.
2. **The arm order follows the seed again.** R − U is positive in seed 101 and negative in seed 202, as on the sealed test.
   pass@8 (solved at least once) differs by at most 3.7 points between arms, so the arms differ in reliability, not reach.
3. **Length again moves the result.** In seed 202, R loses because more of its attempts hit the cap (23.0% vs 15.2%),
   although it is more accurate on the answers it finishes. In R-s101, the gain from round 64 to round 128 is almost
   all fewer capped answers (26.1% → 20.5%); accuracy on finished answers stays at 87%.
4. **Caution on the accuracy term.** Accuracy on finished answers is not a pure skill measure. A policy that finishes
   more of its hardest attempts adds harder problems to its finished pool. That is why the two least-truncated policies
   (U-s202 and H) also show the lowest accuracy on finished answers.

**Response audit.** Would more direct factoring practice fix the failures that remain? 16 items (8 per tier, fixed by
hash before any stress output existed, D144) × attempt 0 × six policies (base and the five finished round-128
checkpoints) = 96 responses. Each response was shown under a random code, shuffled within its item, and read
independently by two annotating agents (Claude Sonnet) with the problem, gold, reference route and verifier result
(D153). Responses over 8,000 characters were shown as their first 5,000 and last 2,500 characters. The two reads agreed
on the category for 88 of 96 responses (29 of the 35 failures). Counts below are reads, two per response.

| Policy | Failures of 16 | Missed the factoring | Factored wrongly | Factored, failed later | Hit the cap | Grader | Fixable by factoring practice: yes / no / unclear / n.a. |
|---|---|---|---|---|---|---|---|
| Base | 8 | 6 | 5 | 4 | 0 | 0 | 8 / 5 / 3 / 0 |
| H-s101 | 5 | 4 | 2 | 1 | 0 | 3 | 4 / 3 / 1 / 2 |
| R-s101 | 4 | 2 | 1 | 4 | 1 | 0 | 2 / 5 / 1 / 0 |
| U-s101 | 4 | 6 | 0 | 0 | 2 | 0 | 5 / 1 / 2 / 0 |
| R-s202 | 8 | 4 | 0 | 5 | 4 | 3 | 6 / 8 / 0 / 2 |
| U-s202 | 6 | 4 | 1 | 3 | 4 | 0 | 4 / 5 / 3 / 0 |
| All | 35 | 26 | 9 | 17 | 11 | 6 | 29 / 27 / 10 / 4 |

(One base read chose "other method". The Grader column shows reads as given; 2 of its 6 reads, on `00352`, are
adjudicated below as wrong answers.) The answer is: partly. The audit holds 35 failed responses and 70 reads of them (two per
response). Of the 70 reads, 29 judged the failure fixable by more factoring practice, 27 not, 10 unclear, and 4 "not
applicable": both readers of 2 responses (R-s202 and H-s101 on `precalculus/train/00152`) called them grader errors (see
below). Per response, of the 33 genuine failures: both readers said fixable for 13, both said not fixable for 10, and 10
were split or unclear (no/unclear 6, unclear/yes 2, unclear/unclear 1, no/yes 1). After training, the readers put most failures after the factoring step (a later error
or the cap): 24 of 54 trained-policy failure reads, against 4 of 16 for the base model. Missed factoring steps remain in
every arm, R included (6 of 24 R failure reads, 10 of 20 U failure reads). With 4–8 failures per policy, these counts
describe the sample; they cannot rank the arms.

One grader limitation was confirmed. On `math:precalculus/train/00152` (gold `2,4`), a response that puts 2 and 4 in
separate boxes is extracted as `4` and scored wrong. That affects at most 7 of 1,768 attempts in any stress cell (≤ 0.4
points of pass@1) and at most 3 attempts between R and U in either seed (≤ 0.2 points). Scores are left as graded
(frozen verifier). The same pattern barely touches the sealed test: a scan of the 8 sealed items with several gold
values found 9 finished attempts scored wrong whose separate boxes match the gold, at most 3 of 4,096 in any cell
(≤ 0.07 points; H-s101 3, R-s101 2, base 3, R-s101 common-GPU 1; `research/evidence/a17/audit/twobox.py`). The two "0, 1" answers on `math:algebra/test/00352` list the integers instead of their count (gold 2):
they are wrong answers, not grader errors. Data: `research/evidence/a17/audit/`.

## 6. Strategy cue (from the first report; separate from the interference evidence)

24 unused MATH problems where factoring is central, as written and with one appended sentence ("Consider whether a
factorization could simplify the problem."), 8 attempts each, paired seeds. The cue changed what the models wrote (+9
to +15 points more mentions of "factor") but not how often they succeeded. Example: R-s202 scored 62.0% uncued and 57.3%
cued (cue gain −4.7 [−9.9, 0.0]); U-s202 scored 64.6% and 69.3% (+4.7 [−0.5, +10.9]). The difference in cue gain was
+2.1 [−6.3, +12.0] in seed 101 and −9.4 [−19.3, −1.6] in seed 202. Full table: `archive/REPORT-v1-2026-11-10.md` §5.

## 7. Design

| Arm | Prompts per round (16 × 8 answers) | Purpose |
|---|---|---|
| H | 8 shared hard + 8 extra hard | control: more hard practice |
| R | 8 shared hard + 8 factoring practice | treatment |
| U | 8 shared hard + 8 modular-arithmetic practice | control: same practice dose, unrelated skill |

Qwen3-1.7B (`70d244cc`), full-parameter GRPO, non-thinking mode, T = 1.0, lr 2e-6, symmetric clip 0.2, no KL, 4
optimizer steps per round, 128 rounds; cap 8,192 tokens for rounds 1–32 and 12,288 from round 33 in every run. H_train
46 hard problems; practice pools 63 + 63 matched items; dev 53; sealed test 128 MATH test problems.

## 8. What ran, compute, and incomplete cells

GPU-hours = elapsed time × GPUs allocated, summed over every allocation of every ledger job (225 jobs, 239
allocations; `sacct -D` keeps requeued allocations). Failed, cancelled and preempted allocations are charged. CPU
standby jobs are not ledger jobs and are not charged. "Canonical" = the job chains that produced the reported
trajectories. Source: `tables/cost.md`, from `reports/tables/sacct.txt`. PROVISIONAL (13:51Z, 11 Nov): H-s202 job
11112377 counts its elapsed time so far; the final run replaces this table.

| Category | Jobs | Allocations | GPU-h | of which canonical | CPU-h | Final states |
|---|---|---|---|---|---|---|
| CPU jobs | 126 | 131 | 0.0 | 0.0 | 31.0 | CANCELLED 12, COMPLETED 103, FAILED 11, PREEMPTED 5 |
| common-GPU rerun (A17) | 4 | 8 | 18.9 | 0.0 | 151.1 | COMPLETED 4, PREEMPTED 4 |
| local-update diagnostic | 8 | 8 | 14.9 | 0.0 | 118.9 | CANCELLED 4, COMPLETED 3, FAILED 1 |
| main H-s101 | 2 | 2 | 33.6 | 29.5 | 268.9 | CANCELLED 1, COMPLETED 1 |
| main H-s202 | 4 | 4 | 23.2 | 23.2 | 185.9 | CANCELLED 3, RUNNING 1 |
| main H-s303 | 2 | 2 | 0.0 | 0.0 | 0.0 | CANCELLED 2 |
| main R-s101 | 3 | 3 | 32.9 | 27.9 | 263.6 | CANCELLED 1, COMPLETED 1, FAILED 1 |
| main R-s202 | 3 | 3 | 22.8 | 22.8 | 182.3 | CANCELLED 2, COMPLETED 1 |
| main R-s303 | 1 | 1 | 0.0 | 0.0 | 0.0 | CANCELLED 1 |
| main U-s101 | 2 | 2 | 27.3 | 22.6 | 218.6 | CANCELLED 1, COMPLETED 1 |
| main U-s202 | 5 | 5 | 25.6 | 25.6 | 205.0 | CANCELLED 2, COMPLETED 1, FAILED 2 |
| main U-s303 | 1 | 1 | 0.0 | 0.0 | 0.0 | CANCELLED 1 |
| other GPU jobs (screening, calibration, smoke, A10/A11 studies) | 42 | 42 | 42.0 | 0.0 | 335.6 | CANCELLED 6, COMPLETED 29, FAILED 7 |
| sealed test evaluation | 8 | 8 | 21.5 | 0.0 | 171.8 | CANCELLED 2, COMPLETED 6 |
| strategy-cue panel (A15.4) | 2 | 2 | 2.2 | 0.0 | 17.7 | CANCELLED 1, COMPLETED 1 |
| stress evaluation (A17) | 12 | 17 | 18.1 | 0.0 | 144.8 | CANCELLED 4, COMPLETED 8, PREEMPTED 5 |
| **Total** | 225 | 239 | 283.0 | | 2295.2 | |

A17 so far: common-GPU reruns 18.9 GPU-h, stress evaluations 18.1 GPU-h, H-s202 resume 12.2 GPU-h. Preempted
allocations (4 rerun, 5 stress) are charged in full, including any work repeated after requeue. The H-s202 canonical
figure includes job 11082192 to round 64; the a2-r60 trajectory reuses only its rounds 1–60.

⟨PENDING: run table with H-s202 a2-orig / a2-r60; final cost run.⟩

**Incomplete cells** (status at the final commit; ⟨update at delivery⟩):

| Cell | Status | Reason | Where it would appear |
|---|---|---|---|
| H-s202 round-128 sealed test | ⟨pending⟩ | run finishes ≈ 6:05–6:45 PM EST; test ≈ 1 h after | section 1, Q1 |
| H-s202 round-128 stress cell | ⟨pending⟩ | queued after the sealed test | section 5 |
| H-s202 in the response audit | not done | audit sample fixed on r128 checkpoints before H-s202 existed (D144) | section 5 |
| R − H and U − H for seed 202 | ⟨pending⟩ | needs the H-s202 endpoint | section 1, Q1 |
| Same-seed R/U pair below round 128 | not possible | weights not retained (only R-s101 and H-s202 kept round 64) | section 9.2 |
| U-s101 round-64 local-update source | not possible | checkpoint not retained | section 4 |
| Seed 303, clipping sensitivity | not run | deferred by A9 | — |
| Stress panels of 128 items | not possible | candidate pool exhausted: Harder 95, SuperHard 126 (D145) | section 5 |

## 9. Length diagnostic and divergence timing (A19; exploratory)

### 9.1 Would capped answers succeed with more room?

Method (D154, fixed before any output). The question is what the capped answers would have done with more room. Each
stress attempt that hit the 12,288-token cap was continued from its own saved tokens, with 4,096 more tokens allowed,
so the total budget is 16,384 and the first 12,288 tokens are unchanged. Finished attempts were not regenerated. The
12,288 score is the original stored score (every selected attempt was re-graded and matched); the 16,384 score uses the
same grader on the full answer (an answer still capped scores 0). Decoding as in every evaluation (T 1.0, top_p 1.0);
continuation seeds are shared by all four checkpoints; L40S GPUs. Tranche 1 is a fixed sample drawn before generation:
64 problems (32 per tier, stratified by source), attempts 0–3, so 256 attempts per checkpoint.

**Tranche 1** (counts are attempts out of 256; pass@1 weights problems equally):

| Checkpoint | pass@1 at 12,288 | pass@1 at 16,384 | Capped at 12,288 | Recovered | Finished wrong | Still capped | Extra tokens per recovered answer |
|---|---|---|---|---|---|---|---|
| R-s101 | 71.5% | 75.8% | 54 | 11 | 5 | 38 | 16,805 |
| U-s101 | 67.6% | 69.9% | 47 | 6 | 10 | 31 | 27,348 |
| R-s202 | 67.6% | 71.1% | 56 | 9 | 13 | 34 | 20,331 |
| U-s202 | 69.5% | 73.0% | 41 | 9 | 11 | 21 | 12,910 |

| Seed | R − U at 12,288 | R − U at 16,384 | Change |
|---|---|---|---|
| 101 | 71.5 − 67.6 = +3.9 [−2.0, +10.2] | 75.8 − 69.9 = +5.9 [−0.4, +12.5] | +2.0 [−1.2, +5.1] |
| 202 | 67.6 − 69.5 = −2.0 [−9.0, +4.3] | 71.1 − 73.0 = −2.0 [−8.6, +4.3] | 0.0 [−3.1, +3.1] |

Few capped answers recover: 6–11 of 41–56 per checkpoint (13–22%). Most (51–70%) are still capped at 16,384 tokens,
and the rest finish with a wrong answer. Extra room does not close the seed-202 gap: R and U each recover 9 answers, and
R − U stays at −2.0 points. In seed 101, R recovers more (11 vs 6, mostly on SuperHard), so R − U widens by 2.0 points;
the interval includes zero. Per tier: `research/evidence/a19/length/t1/length.md`.

**Full stress panel** (all 221 problems × 8 attempts = 1,768 per checkpoint; tranche 1 included):

| Checkpoint | pass@1 at 12,288 | pass@1 at 16,384 | Capped at 12,288 | Recovered | Finished wrong | Still capped | Extra tokens per recovered answer |
|---|---|---|---|---|---|---|---|
| R-s101 | 69.5% | 72.3% | 362 | 49 (13.5%) | 61 | 252 (69.6%) | 25,435 |
| U-s101 | 65.3% | 68.2% | 406 | 51 (12.6%) | 64 | 291 (71.7%) | 28,075 |
| R-s202 | 63.2% | 66.0% | 407 | 50 (12.3%) | 96 | 261 (64.1%) | 27,437 |
| U-s202 | 65.8% | 68.2% | 268 | 41 (15.3%) | 81 | 146 (54.5%) | 19,914 |

| Seed | Tier | R − U at 12,288 | R − U at 16,384 | Change |
|---|---|---|---|---|
| 101 | all (221) | 69.5 − 65.3 = +4.2 [+1.6, +6.8] | 72.3 − 68.2 = +4.1 [+1.4, +6.7] | −0.1 [−1.1, +1.0] |
| 101 | Harder (95) | 84.2 − 80.1 = +4.1 [+0.5, +7.8] | 86.1 − 81.8 = +4.2 [+0.7, +7.8] | +0.1 [−1.1, +1.4] |
| 101 | SuperHard (126) | 58.4 − 54.2 = +4.3 [+0.6, +8.0] | 61.9 − 57.9 = +4.0 [+0.2, +7.8] | −0.3 [−1.8, +1.3] |
| 202 | all (221) | 63.2 − 65.8 = −2.7 [−5.1, −0.2] | 66.0 − 68.2 = −2.1 [−4.6, +0.3] | +0.5 [−0.5, +1.5] |
| 202 | Harder (95) | 76.3 − 79.5 = −3.2 [−6.6, +0.3] | 79.3 − 80.9 = −1.6 [−4.6, +1.4] | +1.6 [+0.1, +3.2] |
| 202 | SuperHard (126) | 53.3 − 55.6 = −2.3 [−5.6, +1.1] | 56.0 − 58.5 = −2.6 [−6.2, +1.0] | −0.3 [−1.7, +1.1] |

The full panel confirms tranche 1 and removes its one hint. Every checkpoint gains 2.4–2.9 points of pass@1 from 4,096
more tokens, and the gain is nearly the same in R and U. R − U moves by at most 0.5 points overall in either seed
(1.6 points on seed-202 Harder, where R had more capped answers to recover). The seed-101 widening seen in tranche 1
(+2.0) was sampling noise: on the full panel the change is −0.1. Only 12–15% of capped answers recover; 55–72% are
still capped at 16,384 tokens. R-s202 hits the cap far more than U-s202 (407 vs 268 attempts), but its capped answers
are no more recoverable (12.3% vs 15.3%). Source: `research/evidence/a19/length/full/length.md`.

What did the extra tokens do? An arm-blind reader classified 12 continuations, chosen by a hash rule before generation
(3 capped attempts per checkpoint): 6 kept looping (re-deriving the same steps or equation systems), 3 reached a wrong
answer, 1 finished sound reasoning correctly, and 1 found and fixed an error, wrote the correct boxed answer, and then
hit 16,384 tokens before its final write-up (scored 0 by the frozen cap rule). One was still making progress when cut
off. The reader judged 8 of the 12 off track before the 12,288-token cut (1 unclear). Loops were split evenly (R 3, U 3). Source:
`research/evidence/a19/length/t1/inspect_read.md`.

This measures inference allowance for these four checkpoints only. It does not show what training with a longer cap
would have produced.

### 9.2 When did R and U diverge?

No same-seed R/U checkpoint pair survives below round 128: the trainer kept only the latest checkpoint, plus round 64
for R-s101 and H-s202. So the timing comes from measurements taken during training, not from new evaluations. Dev =
53 problems × 8 attempts at fixed rounds (sd ≈ 2 points per value, D50). Shared hard prompts = the same hard prompts
in the same order in every arm of a seed (2,048 answers per arm per 32-round window).

| Seed | Dev round | R | U | R − U |
|---|---|---|---|---|
| 101 | 0 | 41.7% | 42.2% | −0.5 |
| 101 | 32 | 54.2% | 60.1% | −5.9 |
| 101 | 64 | 73.6% | 71.0% | +2.6 |
| 101 | 96 | 74.8% | 68.4% | +6.4 |
| 101 | 128 | 72.9% | 69.1% | +3.8 |
| 202 | 0 | 40.3% | 42.2% | −1.9 |
| 202 | 32 | 59.9% | 61.1% | −1.2 |
| 202 | 64 | 61.3% | 67.9% | −6.6 |
| 202 | 96 | 64.2% | 67.9% | −3.8 |
| 202 | 128 | 65.6% | 70.3% | −4.7 |

In both seeds the endpoint sign appeared between rounds 32 and 64, the first window after the cap rose from 8,192 to
12,288 tokens at round 33. Training success on the shared hard prompts shows the same timing. In seed 101, R led U by
+3.7 to +7.2 points in 10 of the 11 eight-round bins after round 40 (U led by 0.6 points in rounds 49–56). In seed 202,
U led by 4.5–6.1 points in rounds 33–48; after that, the gap stayed within ±7 points and changed sign often. Answer length on the shared hard prompts does not track the
sign. In seed 202, R wrote shorter answers than U in every window after round 32 (4,435 vs 4,755 tokens in rounds
33–64; 4,874 vs 5,590 in rounds 65–96), yet R was cut off more often on the test (section 3.3). Source:
`research/evidence/a19/divergence/divergence.md`.

What this cannot show: whether the split in rounds 33–64 is an arm effect or the point where two different runs of each
arm took different paths. Checkpoint coverage has a gap: no R/U pair exists at rounds 32, 64 or 96 to evaluate on the
stress panels.

## Reproduction

Code: branch of this repository at commit ⟨final commit⟩. Every job was pinned to a commit; the ledger
(`slurm-agentic-engineering/state/LEDGER.jsonl`, outside the repository) maps each job to its commit and recipe hash.
Run on a compute node, never on a login node, with `P=/data/user_data/$USER/qis`, the environment `$P/env/venv-v1`
(analysis) or `$P/env/venv-report` (figures), and `PYTHONPATH=src`. Model: `Qwen/Qwen3-1.7B` at `70d244cc`.

| Step | Command |
|---|---|
| Sealed-test endpoint (one checkpoint, L40S) | `python -m qis.evaluate_ckpt --model $MODEL --ckpt $P/ckpt/<run>/round_0128 --panel $P/data/main-v1/target_test_sealed.json --panel-key test --n 32 --out <dir> --seed-label test-final --max-tokens 12288` |
| Same, as six shards (H-s202) | `python -m qis.eval_merge --split 6 <panel> test <prefix>`, one `evaluate_ckpt` per shard panel, then `python -m qis.eval_merge <panel> test <out> <shard dirs>` |
| Stress panels (construction) | `python -m qis.data.stress_panels data/external/a17/candidates_v1.jsonl research/evidence/a17/panels/annotations <out> $P/data $PWD/data` |
| Stress evaluation (one checkpoint) | `python -m qis.evaluate_ckpt --model $MODEL --ckpt <ckpt or base> --panel $P/data/stress/stress_v1.json --panel-key stress --n 8 --out $P/stress-eval/<label> --seed-label stress-v1 --max-tokens 12288` |
| Test analyses (section 1, 3) | `python -m qis.analysis.a17_test $P/test-eval <out> --cgpu-root $P/test-eval-cgpu` |
| Training analyses (sections 2–4) | `python -m qis.analysis.a17_train $P/runs <out> --diag $P/diag-local/r0/diag_r0.jsonl $P/diag-local/r64/diag_r64.jsonl --diag-ref research/evidence/a15/diag_r0_ref.jsonl research/evidence/a15/diag_r64_ref.jsonl --gpu-hours reports/tables/cost.json` |
| Stress analysis (section 5) | `python -m qis.analysis.a17_stress $P/data/stress/stress_v1.json $P/stress-eval <out>`; figure: `python -m qis.analysis.a17_stress plot <out> <fig dir>` |
| Response audit packets (section 5) | `python research/evidence/a17/audit/build.py research/evidence/a17/panels/audit_sample_v1.json` |

Frozen inputs and their sha256: sealed test `2ffe70ddd5f81bee5c9feadee6f08335d8b189b33f38ccab474263182dec29ca`,
stress panel `757393bef80d5cf017d53bddbfcab84e049c77dee3b07b74898125379f76405d` (rehashed 14:51Z, 11 Nov).
Checkpoint hashes below come from each evaluation's `summary.json` (`ckpt_model_sha256`, recorded by
`evaluate_ckpt`). Every checkpoint used by more than one evaluation has the same hash in each. Source:
`research/evidence/a17/manifest/manifest.json`. ⟨PENDING: H-s202 round-128 or fallback row.⟩

| Checkpoint (`$P/ckpt/…`) | model.pt sha256 | Used in |
|---|---|---|
| `qis-main-H-s101-sym/round_0128` | `052c8a62b300385336538fb396ee537efd03cd081856130a19ac53cbf7c1c811` | sealed test, stress |
| `qis-main-H-s202-sym/round_0064` | `37963facc1cc5b087a2757c77992847b5385c0d63c9e81553f68aa2c8802e7ee` | stress |
| `qis-main-R-s101-sym/round_0064` | `6fe89540d80db2eb78b3b3b4b71eb4029ab2fa3c3d54a35528094153a6c69a5a` | stress |
| `qis-main-R-s101-sym/round_0128` | `aedcca6e6a443c880c153bf1dbcdefd1877632c0a7ba313735518a7c6e39689a` | sealed test, common GPU, stress |
| `qis-main-R-s202-sym/round_0128` | `22fd8364b310691203900a1667af292f5c119012d600d2af74cd99cb1ae97c71` | sealed test, common GPU, stress |
| `qis-main-U-s101-sym/round_0128` | `92aef0d2a53a93139a7759e488500126a324e000ffa47a17cca11ac87d919b70` | sealed test, common GPU, stress |
| `qis-main-U-s202-sym/round_0128` | `7158b0952ee34f3278c9260a66ddadf57aa25b79fc51f541e61f5e8f41ad1354` | sealed test, common GPU, stress |
| base `Qwen/Qwen3-1.7B` | HF revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | sealed test, stress |
