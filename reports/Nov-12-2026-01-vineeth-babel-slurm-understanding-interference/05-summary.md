# Summary: why the first study saw no interference, and where to go next (10-minute read)

Purpose: the direction of the work in one place, with a synopsis of each document of study
`Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`. Read this first. Every number below comes from the
document named beside it.

## 1. The answer

The first study trained Qwen3-1.7B with GRPO on hard algebra plus one of three practice domains and saw no
interference. **The domains did not interfere because their training signals did not oppose each other.** Each run
learned mostly one change that all domains share. Both preconditions of ray interference were absent: no negative
coupling, and no domain near zero success.

This study also found how to supply both preconditions on the same model. A brevity reward has a measured negative
coupling with the correctness reward. Base-9 addition of 12-digit strings has a measured success rate of 0 in 3,200 answers.

## 2. What we know, and how well

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | Gradients of different domains do not conflict at the base model. Mean cosine 0.0003–0.0007; conflict rates 48.7–49.8% (independent directions: 50%) | 1,512 group gradients, 64 answers per problem (`01`, §1.2) | measured; intervals for the hard pool are wide |
| 2 | Every trained model helped every domain: 24 of 24 effects positive, 17 with a 95% interval above zero | log-probability of base answers under six trained models (`01`, §1.4) | measured |
| 3 | Two runs share 4% of their weight update (cosine 0.03–0.05), whatever their arm. That shared part carries 75–96% of a run's effect | six stored checkpoints; averaged weight sets (`01`, §1.1, §1.5) | measured |
| 4 | The shared part also carries the accuracy gain. At equal norm the mean of six runs gains 22.2 points of dev pass@1, one run 11.3. With the shared part removed, a run gains 1.9 points against 31.0 | dev panel, 53 problems × 32 answers (`01`, §1.5) | measured; no intervals |
| 5 | The shared part formed in rounds 1–64. Later updates of two runs have cosine 0.004 | update halves of two runs (`01`, §1.1) | measured on two runs |
| 6 | The model that is sampled is a BF16-rounded copy. Rounding the weights explains why half of an update gives 8–9% of the output change; in FP32 it gives 26–27% | precision test (`01`, §1.6) | measured; cause confirmed |
| 7 | The learned change needs little of the model: 8 directions per weight matrix (0.6% of the parameters) carry 67–77% of the effect | rank-r parts of the update (`01`, §1.7) | measured; directions chosen after training |
| 8 | Capacity is untested. Every published test varies model size, width or the trainable subspace | literature review (`01`, §1.7) | one model size exists |
| 9 | Among 16 further domains, no pair has negative coupling under the correctness reward | base-model probe (`04`, §2) | measured; wide intervals for 4 domains |
| 10 | A brevity reward opposes the correctness reward: 16 of 16 cosines negative, 10 intervals below zero | same base answers, second reward (`04`, §4) | measured at the base model only |
| 11 | Success of the base model falls from 97.7% (2-digit products) to 5.5% (6-digit), 0.09% (8-digit) and 0 of 3,200 (12-digit base-9 addition) | probes (`04`, §2) | measured |

Verdicts on the hypotheses of the mandate (`01`, §2.6):

| Hypothesis | Verdict |
|---|---|
| Pre-training aligned the gradients | consistent in a weak form (no conflict); the cosines are near zero, not large |
| Capacity absorbs conflict | not needed here, because no conflict exists; untested as a claim about size |
| The domains are too similar | consistent: even unrelated subjects share the direction |
| Training too short, learning rate too small | not consistent as stated: dev pass@1 rose from 41% to 72% |
| The evaluation cannot detect interference | consistent for the first study's one-update test; not the reason for the main result |

## 3. Direction

The question changes from "why was there no interference" to "under which conditions does it appear, and do
pre-training and scale remove it". Three steps follow.

1. **Build the conflict, with a dial.** Train correctness together with a brevity reward of weight λ, and the
   contradictory-rule mixture of `04`, §4. Measure the coupling at the base model before each run. This gives a
   dose-response curve of harm against measured conflict (`04`, mixtures 7 and 10; `01`, E1).
2. **Add the zero-start domain.** Combine a conflict with a domain the model cannot solve (`04`, mixtures 5 and 11;
   `01`, E3). This is the direct test of ray interference.
3. **Then vary capacity.** Repeat the clearest conflict at Qwen3-0.6B and 4B (`01`, E2) and at update rank 1, 8 and
   64 (`01`, E6). Capacity can matter only where a conflict exists, so this step comes last.

Two cheaper items run beside these: early checkpoints of one rerun, to see when the shared signal forms and fades
(`01`, M1), and the control with shuffled rewards, to see whether the shared gain depends on the reward (`01`, E5).

| Step | Cost (GPU-hours; estimates from the first study's 19.5–29.5 per run) |
|---|---|
| Base-model coupling probe, per pair | about 3 |
| First training set of `04`, §7 (4 mixtures, references, 2 seeds) | about 450 |
| E1 without the code arm | about 190 |
| E3 | about 230 |
| E2 (0.6B half; full) | about 100; about 490 |
| E6 (first pass; full) | about 120; about 560 |
| M1 and mechanistic studies | about 120 |

No training run was started in this study.

## 4. Synopsis of each document

**[`00-initial-theory.md`](00-initial-theory.md): theory and literature.** Interference has seven definitions. To
first order they reduce to the inner product of two task gradients. Ray interference adds a second condition for
on-policy RL: coupling, where a task with low success gives almost no learning signal. Three factors reduce
interference in theory: high dimension with low-dimensional updates, over-parameterisation, and small steps from a
good starting point. Pre-training supplies all three. No paper found measures the gradient cosine between domains
under a policy-gradient loss against model size or pre-training. Short version:
[`02-theory-synopsis.md`](02-theory-synopsis.md).

**[`01-understanding.md`](01-understanding.md): measurements, explanation, proposed experiments.** Section 1 reports
seven measurements on the stored checkpoints (findings 1–8 above), with code and table locations. Section 2 gives
the explanation: condition C1 (opposing directions) fails, so size and persistence do not matter; a small common
signal adds over steps while noise does not; ray interference cannot start. Each claim is marked derived, supported
or speculation. Section 3 proposes experiments E1–E6 and measurements M1, each with predictions per hypothesis and
a cost. Short version: [`03-understanding-synopsis.md`](03-understanding-synopsis.md).

**[`02-theory-synopsis.md`](02-theory-synopsis.md).** The 10–15 minute version of `00`, with five figures and the
main formulas.

**[`03-understanding-synopsis.md`](03-understanding-synopsis.md).** The 10–15 minute version of `01`, with six
figures, the verdict table and one table of all proposed experiments.

**[`04-domains-to-explore.md`](04-domains-to-explore.md): domains and mixtures.** A reference table of 16 domains
with base pass rates and gradient coupling on Qwen3-1.7B, published reference points, three ways to obtain negative
coupling (one measured), 11 candidate mixtures ordered by expected interference and by starting success, and nine
further axes to vary. Its main point: hardness is easy to supply with procedural tasks; negative coupling must be
built into the reward.

## 5. Limits to keep in mind

- One model size and one training recipe. Nothing here tests scale.
- Coupling was measured at the base model and at five trained checkpoints, not in rounds 1–32.
- Effects on trained models are log-probability effects on base answers, plus dev accuracy for the weight sets of
  finding 4. They are not held-out test scores.
- The brevity result is a gradient measurement. No run has yet shown that it lowers accuracy.
- Published cases of harm between domains (`04`, §3) come from 4B and 7B models.

## 6. Where things are

| Item | Location |
|---|---|
| Reports | `reports/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/` |
| Figures, tables, analysis scripts | `figures/`, `tables/`, `build/` in that folder |
| Measurement code | `src/qis/interf/` |
| Job recipes, checklist, decisions, literature notes, handover | `research/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/` |
| Raw outputs on Babel | `/data/user_data/$USER/qis/ui/` |
