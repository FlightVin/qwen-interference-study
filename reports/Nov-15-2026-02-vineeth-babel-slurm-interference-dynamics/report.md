# When Does Learning One Skill Impede Another? Interference and Feedback Dynamics in GRPO Training of Qwen3-1.7B

## Abstract

Reinforcement learning with verifiable rewards is now routinely applied to several tasks at once, yet little is
known about how the tasks affect one another during training. We studied this question for Group Relative Policy
Optimization (GRPO) applied to Qwen3-1.7B, using small tasks with exact graders: Caesar-cipher decoding, Python
programming (MBPP), addition in base 9 and base 10, and a "terse" arithmetic task that rewards answers without
written working. We asked two questions. First, when two tasks are trained together, when does learning one slow
the other? Second, does the on-policy nature of GRPO amplify an initial disadvantage, because a skill that succeeds
less often also receives less learning signal?

Three results stand out. First, interference followed behaviour rather than domain. A task that rewards short
answers delayed the harder Caesar problems in two of three seeds, and in each of those seeds Caesar answers became
shorter during the delay. Training on one task also often damaged an untrained behaviour, here the terse format.
Second, a cross-domain lag (Caesar trained with MBPP) cannot be attributed to interference: a Caesar-only run at
half the learning rate showed that Caesar learning is highly sensitive to step size, which makes optimizer effects a
credible alternative explanation. Third, within base-9 addition, the medium-difficulty
problems sometimes collapsed to near-zero accuracy during training (2 of 5 seeds). Once every training sample on
those problems was wrong, GRPO produced no gradient for them and the skill stayed collapsed; a single rare correct
sample, or transfer from easier problems, was enough to restart learning. This confirms the persistence step of the
proposed feedback loop. The cause of the collapse itself remains unidentified. All results are exploratory: most
conditions have one to three seeds.

## 1. Introduction

Large language models are increasingly post-trained with reinforcement learning on several tasks at once. Each
task contributes reward signal and gradient updates to one shared set of weights. Whether these contributions
cooperate or conflict matters for how training mixtures are designed. A common intuition holds that similar tasks
help each other and dissimilar tasks interfere, but this intuition has rarely been tested in the regime of
policy-gradient fine-tuning, where the data a model learns from depends on its own current behaviour.

That dependence is the second motivation for this study. GRPO estimates the value of each response relative to
other responses to the same prompt. If all responses to a prompt receive the same reward, all wrong or all
correct, the prompt contributes no gradient at all. A skill whose success rate falls toward zero therefore loses
its learning signal at exactly the moment it most needs it. This suggests a feedback loop: an adverse update lowers
a skill's success rate, which produces fewer prompts with mixed outcomes, which slows further learning of that
skill. If the loop is real, small early disadvantages could grow into lasting deficits.

We therefore asked two research questions:

1. **Interference between tasks.** When two tasks are trained together, when does learning one impede the other?
   We classify each pair as transfer, independence, a temporary delay, or a lasting deficit.
2. **Amplification through on-policy feedback.** Does the loop described above occur, and does it turn an initial
   disadvantage into a persistent one?

The study had a budget of two non-preemptible and six preemptible GPUs for about one day. We therefore favoured
small, controlled comparisons over broad sweeps, and we replicated a finding only when it changed the direction of
the investigation.

## 2. Methods

### 2.1 Model and training

All experiments fine-tuned Qwen3-1.7B in non-thinking mode with GRPO. In each training round the model generated
eight responses per prompt. Each response received a binary reward from an exact grader. Rewards were normalised
within the group of eight, so a prompt with all-correct or all-wrong responses contributed zero advantage. We used
the clipped policy objective (clip range 0.2) without a KL penalty, and AdamW at learning rate 2×10⁻⁶ with four
optimizer steps per round. A typical run lasted 32 rounds with eight prompt groups per task per round. The
hyperparameters were selected in an earlier study on the same model and code base, and a canary run verified the
training, checkpointing and resume path before any experiment.

One detail of the loss matters for interpreting joint training. Each response's loss is averaged over its tokens,
and the sum over responses is divided by a fixed number (32 responses per optimizer step). This makes the raw
gradient contributed by a task the same whether it is trained alone or alongside another task. The Adam optimizer,
however, rescales updates by a running estimate of gradient magnitude. When a second task contributes gradient,
each task therefore receives a smaller share of a roughly fixed-size update. We return to this point in Section
3.3, where it explains one apparent case of interference.

### 2.2 Tasks

We chose tasks with exact graders, measurable but imperfect starting accuracy, and room to improve. Several tasks
were divided into difficulty levels, so that easy and hard versions of the same skill could be compared directly.

| Task | Description | Starting accuracy |
|---|---|---|
| Caesar (shift 4, 8, 12) | Decode a Caesar-shifted sentence | 0.30, 0.07, 0.04 |
| Base-9 addition (4, 6, 8 digits) | Add two base-9 numbers | 0.50, 0.18, 0.04 |
| Terse multiplication | Multiply two 2-digit numbers; give only the answer, in at most 24 tokens | about 0.45 |
| Decimal addition (14 digits) | Add two 14-digit base-10 numbers | about 0.50 |
| MBPP | Write a Python function that passes hidden unit tests | 0.59 |
| ARC-Challenge, Countdown | Multiple-choice science; reach a target number arithmetically | 0.81, 0.85 |

The terse task deserves comment. Its arithmetic is easy for the model; what is hard is the behavioural constraint
of answering without writing out steps. It therefore lets us distinguish a competing behavioural objective from a
competing skill. Caesar decoding and base-9 addition, in contrast, benefit from written working.

Before training, a calibration audit of base-model outputs found and fixed three grader faults (code split across
several blocks in MBPP answers, Countdown answers that boxed the target instead of the expression, and unboxed ARC
letters). Every task's data was split into disjoint training, selection and confirmation sets.

### 2.3 Experimental designs

The study combined five kinds of experiment.

- **Single-task references.** Each target task was trained alone. These runs measure its own learning curve and
  its side effects on every other task, which was evaluated but not trained.
- **Joint runs.** A target task and a competitor task were trained together, eight groups of each per round. The
  target's prompts and random seeds matched its single-task reference, so the two runs differ only in the presence
  of the competitor. We summarise each comparison by the gap D = (accuracy alone) − (accuracy in the joint run),
  computed per difficulty level; a positive gap means the joint run is behind.
- **Within-task mixtures.** Base-9 addition was trained on a mixture of its three difficulty levels, and on each
  level alone or in subsets, to see how difficulty levels interact inside one skill.
- **Branching experiments.** Starting from a saved checkpoint (model and optimizer state), several short
  continuations were run under different training conditions. All other factors, including prompts and seeds, were
  held fixed. This isolates the effect of the condition from the history that led to the checkpoint.
- **A step-size control.** One reference run used half the learning rate. This shows how much a smaller effective
  update alone would slow learning.

### 2.4 Measurement and statistics

Each evaluation used 64 held-out problems per task and difficulty level, with four samples per problem (eight for
the confirmation set used in branching experiments). We report pass@1, the fraction of correct samples. Uncertainty
is a 95% bootstrap interval over problems. When two runs are compared, the bootstrap pairs each problem across the
two runs, which removes variation due to problem difficulty. Evaluations of the same model in different jobs
differed by up to 0.04 because the inference engine's sampling is not bit-reproducible. Effects of 0.15 or more are
therefore well above evaluation noise, while effects near 0.07 are only about twice it.

These intervals have a limited meaning, and it matters throughout the results. They describe uncertainty over the
evaluation problems within a single training run. They say nothing about how much the result would change if the
whole training run were repeated with a different seed, and that between-run variation can be large: the
within-task collapse of Section 3.4 occurred in two of five otherwise identical runs. An interval that excludes zero
therefore shows that a difference is real for that pair of runs, not that it would replicate. Only replication
across training seeds addresses that, and we state the number of seeds behind each claim.

Results that changed the direction of the study were re-examined by an independent reviewer, who recomputed the
numbers from raw metrics and actively looked for alternative explanations. This review corrected one of our
conclusions (Section 3.4). A separate audit recomputed a sample of 18 reported numbers and found no discrepancies.

## 3. Results

### 3.1 Training a single task often damages untrained behaviour

Before asking how two trained tasks interact, we measured what training a single task does to tasks that are not
trained. Figure 1 shows the change in accuracy on every evaluated task after 32 rounds of training on one task.

![Off-target effects](figures/offtarget.png)

*Figure 1. Change in accuracy from round 0 to round 32, by trained task (rows) and evaluated task (columns). Bold
values have 95% intervals that exclude zero.*

Each task learned what it was trained on. Caesar decoding improved sharply at every difficulty level and in all
three seeds; even the hardest shift rose from 0.04 to between 0.64 and 0.71. MBPP improved by 0.14, and mixed
base-9 training improved the 4-digit level by 0.25.

The more striking pattern concerns the terse task, which none of these runs trained. Its arithmetic is easy for
the model; the base model got only 4 of 256 multiplications wrong. Terse answers fail almost entirely on format. At
the start, about half of them gave no boxed answer, and a few were correct but longer than the 24-token limit.
Training other tasks moved this format in one of two harmful directions, or in a helpful one (Figure 2).

![Terse task across runs](figures/terse_spillover.png)

*Figure 2. The terse task in every run. Left: accuracy. Right: mean answer length on a logarithmic scale (dashed:
the 24-token limit). Blue and purple: runs that trained terse; all other colours: runs that did not.*

Caesar training lowered terse accuracy in all three seeds, by 0.34, 0.40 and 0.11, but not in the same way. In
seed 1 the answers grew long: by round 32, 228 of 256 answers were correct but exceeded the length limit, and the
mean length rose from 19 to 107 tokens. In seed 2 the answers instead stopped carrying a boxed answer (239 of 256). In
seed 3 the format changed little. Base-9 training showed the same split between configurations. The mixed run lost
the box (238 of 256 unboxed), while training on 4-digit problems alone produced over-long answers (238 of 256).
Training on 6- or 8-digit problems alone did the opposite: the model learned to box its terse answers, and accuracy
rose by 0.46 and 0.38. MBPP training left the terse task essentially unchanged.

Two conclusions follow. First, the damage to terse comes from format and length conventions, not from lost
arithmetic. Second, which convention a task spreads is not a fixed property of the task. The same task, Caesar
decoding, spread long answers in one seed and unboxed answers in another, and different difficulty levels of
base-9 addition pushed the format in opposite directions. We cannot predict the direction from the task alone.

Other side effects were small. MBPP training lowered ARC accuracy by 0.04, and Caesar training raised 6-digit base-9
accuracy by 0.11. Within base-9 addition, the effects were large and of both signs. Training on 4-digit problems
alone raised 6-digit accuracy by 0.21, a clear case of transfer. Training on 8-digit problems alone lowered 4-digit
accuracy by 0.25 and 6-digit accuracy by 0.15, while 8-digit accuracy itself did not improve. On 8-digit problems
the model almost never succeeds, so nearly every gradient came from groups with a single correct answer out of
eight. Section 3.4 returns to this.

The lesson for multi-task training is that a behaviour that no task currently trains is not protected. Whether it
survives depends on the output conventions the other tasks reinforce, and these differ between closely related
tasks and even between training seeds of the same task.

### 3.2 A behavioural objective can delay a skill that needs written working

Our first joint pair trained Caesar decoding together with the terse task. Terse was learned within four rounds,
rising from about 0.4 to above 0.9. After that most of its prompt groups were all-correct, so it contributed much
less gradient than an active task: about one group per round in seed 1, and fewer still in seeds 2 and 3. Nonetheless, Caesar learning was slowed on the harder shifts. Figure 3
shows the gap to the single-task reference.

![Caesar gap in joint runs](figures/caesar_D.png)

*Figure 3. Gap between Caesar accuracy trained alone and in a joint run (positive: joint run behind), by shift
size. Blue: joint training with the terse task, three seeds. Orange: joint training with MBPP. Bands are 95% paired
bootstrap intervals.*

In seeds 1 and 2, the joint run fell behind on shifts 8 and 12 by 0.11 to 0.18 at round 16. In seed 1 the gap
closed by round 24. In seed 2 it persisted on shift 8 until the end of training (0.12 at round 32). Seed 3 showed
no delay at any point.

A candidate mechanism emerged from the length of Caesar answers during training. In the two seeds with a delay,
the joint run's Caesar answers were 5–12% shorter than the reference's during rounds 1–16. Caesar decoding benefits
from writing out the shifted letters, so shorter answers plausibly mean worse answers. In seed 3, the joint run's
answers were 6% longer, and there was no delay. This pattern makes brevity spill-over a promising explanation, but
it is not confirmed. It is a correlation across three seeds, and we did not test it by intervention, for example by
loosening the terse length limit.

A second explanation cannot be excluded either. Although terse was nearly solved after the first few rounds, it never
stopped producing gradient entirely. In rounds 1–8, 10 to 13 of its groups produced gradient. In later blocks of eight
rounds, seed 1 still had 6 to 9 such groups per block, and seeds 2 and 3 had 0 to 5 (Figure 5a). Those updates changed the model, and they also changed Adam's running estimates of
gradient direction and scale, which shape every later Caesar update. A delay could therefore arise from this
optimizer history rather than from brevity specifically. The two explanations are not exclusive.

During the delay, the joint runs also had fewer Caesar prompt groups with mixed outcomes, and more with all-wrong
outcomes, than the references. In seed 1, for example, rounds 9–16 had 44 mixed groups against 51. This is the first
link of the proposed feedback loop. The effect was small, however, and in seed 1 it disappeared once the gap closed.

The pairing also protected terse. When terse was trained alongside Caesar, its accuracy stayed above 0.9, whereas
Caesar training alone had driven it down.

### 3.3 A cross-domain lag that optimizer effects may explain

Our second pair crossed domains: Caesar decoding with Python programming. Figure 4 shows both tasks trained alone
and together. MBPP learned at the same rate in both settings; its accuracy at round 32 was 0.69 with Caesar and 0.72
alone, a difference well within the evaluation uncertainty. Caesar improved substantially in both settings, but
the joint run fell behind on all three shifts, by 0.09 to 0.12 at round 32, and had not caught up by the end of
training.

![Caesar and MBPP alone and together](figures/caesar_mbpp_curves.png)

*Figure 4. Accuracy of MBPP (left) and of Caesar decoding at each shift, trained alone (orange) and together (blue).
Grey dashed: Caesar trained alone at half the learning rate. Bands are 95% bootstrap intervals over the 64
evaluation problems of each run; they do not capture variation across training seeds.*

The lag could be interference, but there is a simpler alternative. MBPP remained an active learner throughout
training: it produced gradient in 100 groups over 32 rounds, whereas terse produced 18 to 34 (Figure 5a).
Adam scales each update by running estimates of gradient size, so when two tasks contribute gradient the update is
spread across both, and Caesar's learning can slow without any conflict between the tasks. To gauge how sensitive
Caesar is to such effects, we trained Caesar alone at half the learning rate.

![Step-size sensitivity](figures/step_sharing.png)

*Figure 5. (a) Cumulative number of competitor groups that produced gradient: MBPP stayed active throughout, while
terse produced most of its gradient in the first rounds. (b) Caesar accuracy at round 32 when trained alone at the full learning rate, with MBPP, and
alone at half the learning rate (95% intervals over evaluation problems). (c) Accuracy lost relative to Caesar trained
alone at the full rate, over training.*

Halving the learning rate cost Caesar 0.14, 0.53 and 0.37 at round 32 on the three shifts, far more than the joint
run's lag (Figure 5b and c). This shows that Caesar learning is highly sensitive to the size of its updates, which
makes optimizer effects a credible explanation for the joint-run lag. It does not show that the joint run is
equivalent to some smaller Caesar-only learning rate. Adam changes the direction and the per-parameter scaling of
updates, not just their overall size, so there is no general rule that places a shared update between half and full
step. The lag is therefore unresolved: it may reflect interference, optimizer effects, or both. Separating them
would need, for example, an optimizer that keeps separate moment estimates per task, or a joint run in which the
competitor's gradient is measured and removed.

The control has a broader implication. With Adam, a lag in joint training is not by itself evidence that one task
harms another, because any active competitor also changes the optimizer's state. This caveat applies with less
force to the Caesar-terse delay of Section 3.2, because terse contributed far less gradient than MBPP, but it does
not vanish: terse kept producing some gradient, and its early updates shaped the optimizer's history.

Two further pairs combined base-9 addition with a competitor: decimal addition, which shares surface form but uses
a different carry rule, and the terse task (Figure 6). Against the single-task reference with the same seed, we
detected no harm: 4-digit accuracy stayed within 0.06 of the reference throughout. That reference, however,
collapsed on 6-digit problems (Section 3.4), which makes it a poor comparison at that level. We therefore also
compare with three other single-task seeds that did not collapse. Against them, both joint runs ended at or just below
the range on 6-digit problems (0.52 and 0.53, against 0.54 to 0.68). On 4-digit problems the run with terse ended
inside the range (0.82), and the run with decimal addition ended below it (0.75, against 0.82 to 0.86). With one
seed per joint configuration, we cannot tell a small cost of joint training from ordinary variation between seeds.
What the runs do show is that neither competitor caused a large or lasting deficit. The competitors themselves were
learned. Terse rose from 0.40 to above 0.9 within four rounds. Decimal addition rose from 0.38 to between 0.56 and
0.75 during training (0.61 at round 32), against 0.41 to 0.53 when it was not trained.

![Base-9 addition with competitors](figures/add9_pairs.png)

*Figure 6. Base-9 addition at each difficulty level, trained alone (grey band: range of three seeds that did not
collapse; grey line: the collapsed seed 1) and together with decimal addition or terse (one seed each). Right: the
competitor tasks, trained and untrained.*

A related observation came from the Caesar-MBPP pair: there, terse accuracy rose from 0.40 to 0.68, although
neither task trained it and Caesar alone had driven it down (Section 3.1).

Across the four pairs, then, interference did not follow domain. Two arithmetic tasks with conflicting carry rules
coexisted without a large or lasting deficit. A cross-domain pair produced a lag that optimizer effects may explain. The delay
that is hardest to attribute to the optimizer came from a behavioural objective acting on a skill that needs
written working, though even there the optimizer's history is a possible contributor.

### 3.4 A skill can collapse within a single task

The clearest evidence on the second research question came from inside one task. When base-9 addition was trained
on a mixture of 4-, 6- and 8-digit problems, 6-digit accuracy usually rose steadily. In two of five seeds, however,
it first rose to between 0.30 and 0.49 and then fell to near zero within a few rounds (Figure 7). In one seed it
dropped from 0.30 to 0.00 between rounds 12 and 16; in the other, from 0.49 to 0.09 between rounds 16 and 24. Both
stayed below 0.1 until training ended.

![Base-9 6-digit accuracy](figures/add9_d6.png)

*Figure 7. Accuracy on 6-digit base-9 problems under each training condition. Colour shows the condition; line
style distinguishes seeds. The two mixed-difficulty seeds that collapsed are the blue lines that fall to near zero.*

The collapse was not caused by longer or truncated answers. Answer length stayed between about 1,000 and 1,250
tokens and fewer than 1% of answers were truncated. Decoded answers showed the model still following the same
method (convert both numbers to decimal, add, convert back), with errors appearing in the first conversion. We
inspected only a handful of answers, so this should be read as an illustration, not a measurement.

We tested several explanations for what triggers the collapse.

- **6-digit training alone.** Training on 6-digit problems alone did not collapse in the one seed tested, so the
  skill does not appear inherently unstable at this learning rate.
- **4-digit updates.** Training on 4-digit problems alone improved 6-digit accuracy (Section 3.1), so they are
  unlikely to cause it.
- **8-digit updates.** We first suspected the 8-digit problems, since training on them alone harmed the other
  levels. This explanation did not survive closer inspection. In the seed that collapsed earliest, the 8-digit
  problems produced a gradient in only two rounds, both at least nine rounds before the collapse; in all other
  rounds every 8-digit sample was wrong. The independent review confirmed this and rejected the explanation.
- **The presence of 8-digit problems.** Removing them from the mixture gave 0 collapses in 5 seeds, compared with 2
  in 5 when they were present. With five seeds per condition this difference is not significant (Fisher's exact
  test, p = 0.44). Adding a second task to the full mixture also gave no collapse (0 of 2).
- **Unusually large updates.** The earliest collapse was preceded by the largest update of its run. But a run
  without 8-digit problems had a larger update than the other collapsing seed ever had, and did not collapse.

The trigger therefore remains unidentified. The collapse is an occasional event of the full mixture, occurring in
about two of five seeds, and no condition we tested differs from it significantly. Candidate explanations that our
data cannot separate include an interaction between 4- and 6-digit updates, drift from sparse and noisy updates
under Adam, and a weak effect of the 8-digit problems.

### 3.5 Recovery requires a success

What happens after the collapse speaks directly to the second research question. In the earliest-collapsing seed,
6-digit training success fell from 0.50 at round 12 to zero by round 14. For the next 13 rounds, all 39 of its
6-digit prompt groups were all-wrong. Under GRPO such groups produce no gradient, so the skill received no direct
learning signal. Its accuracy recovered only slightly, to 0.07, by round 32.

To test whether the missing signal is what keeps the skill down, we branched from the collapsed checkpoint (round
16, 6-digit accuracy 0.00) and continued for eight rounds under three conditions (Figure 8).

![Recovery from the collapsed state](figures/recovery.png)

*Figure 8. Eight-round continuations from the collapsed checkpoint at round 16. Left: 6-digit accuracy on the
confirmation set. Right: 4-digit accuracy. Grey: the original run before the branch point.*

With 6-digit problems only and the usual eight samples per prompt, the model never produced a correct 6-digit
answer. Every group was all-wrong, the gradient was exactly zero on every step, and accuracy stayed at 0.00. With
four times as many samples per prompt (32), one group in round 20 contained two correct answers among 96. After
that single update, accuracy rose to 0.11 by round 24, even though all later groups were again all-wrong. The
difference between these two branches was 0.11 (95% interval 0.07 to 0.17). Continuing the original mixture also
restarted learning: the 4-digit problems kept supplying mixed groups and gradient, and 6-digit accuracy reached
0.07, which is 0.07 above the zero-gradient branch (interval 0.04 to 0.12).

These branches show the persistence step of the feedback loop directly. Without a correct sample, GRPO cannot
repair the skill. A single rare success, or transfer from a related skill that still learns, is enough to restart
it.

The independent review identified three qualifications. First, the zero-gradient branch took no learning step at
all. Its weights moved only through optimizer momentum carried over from the checkpoint, and that momentum alone
lowered 4-digit accuracy by 0.16. Second, the recovery in the 32-sample branch stems from one update. Because the
loss divides by a fixed number of responses, a group of 32 samples also carries four times the weight of a group of
eight. The branch therefore does not separate "more samples" from "a larger step". Third, the whole experiment
rests on one collapse state and one branch per condition.

### 3.6 With Adam, a round without gradient is not a round without change

GRPO gives zero advantage to every group whose samples all receive the same reward. It is tempting to treat a round
made only of such groups as a round in which nothing happens. With the AdamW optimizer used here, that is wrong. AdamW
keeps a running average of past gradients (momentum, decaying by a factor of 0.9 per step) and a running average of
their squares. Each update is the ratio of the two. When a new gradient is exactly zero, the momentum term shrinks by
10% but does not vanish, so the optimizer keeps moving the weights in the direction of the most recent updates. Our
trainer runs an optimizer step after every minibatch, so a minibatch whose groups are all-equal still triggers such
a momentum-only update. The only exception is a minibatch with no responses at all, which leaves the weights
untouched.

Figure 9 shows what this means in the recovery branches of Section 3.5. Its top row shows every optimizer step of
each branch.

![Optimizer steps in the recovery branches](figures/momentum_branches.png)

*Figure 9. The three recovery branches of Section 3.5, step by step. Top: pre-clip gradient norm of every optimizer
step (four per round). Ticks at zero mark steps whose gradient was exactly zero but which still applied a momentum
update; grey crosses mark empty minibatches. Bottom: 6-digit accuracy (confirmation set) and 4-digit accuracy, with
95% intervals over evaluation problems.*

In the 8-sample branch, none of the 32 optimizer steps had any gradient. Yet 4-digit accuracy, which this branch no
longer trained, fell from 0.55 to 0.39 (change −0.16, 95% interval −0.26 to −0.07). Since every gradient was zero,
only the momentum carried over from the checkpoint could have moved the weights. In the 32-sample branch, exactly one
step had a gradient (round 20, norm 1.26, clipped to 1). Six-digit accuracy was still 0.01 at the end of that round
and rose to 0.11 over the next four rounds, during which every gradient was again zero. That one step was followed
by many momentum-carried steps in the same direction.

These branch observations are indirect: the training logs record gradients and accuracy, not how far the weights
moved. MOMENTUM_CPU_PLACEHOLDER

The practical consequence is that an analysis treating rounds with all-equal groups as "null" updates will
misattribute change. Weight movement continues after the last informative group, its size depends on the gradients
that preceded it, and it can harm skills that are no longer trained.

## 4. Discussion

**Interference between tasks.** We found transfer, independence and a temporary or lasting delay, but these did not
follow the similarity of the tasks. Decimal and base-9 addition, which share surface form but follow conflicting
carry rules, coexisted without a large or lasting deficit. A pair from different domains, Caesar decoding and
programming, produced a lag that we cannot attribute to interference, because Caesar learning proved highly
sensitive to step size and optimizer effects are a credible alternative. The clearest interaction involved a
competing behavioural objective rather than a competing skill. Rewarding short answers slowed a skill that benefits
from written working in two of three seeds, and in exactly those seeds the brevity preference reached that skill's
answers. This explanation is promising but not confirmed; the competitor's earlier updates and their effect on the
optimizer's state are an alternative we could not exclude. The single-task results
point the same way. The largest side effects of training were on an untrained behaviour (the terse format), and
they arose through output conventions, long answers or unboxed answers, whose direction varied even between seeds
of the same task. These results caution
against assuming that different domains conflict or that similar ones cooperate. They suggest that the relevant
variable is what behaviour each task rewards.

**Amplification through on-policy feedback.** The proposed loop has three steps, and the evidence for each differs.
An adverse change in a skill's success rate was observed, both in the within-task collapse and in the Caesar delay,
but its cause is unknown in the first case and only correlationally supported in the second. Lower success
produced fewer informative prompt groups in both cases, strikingly so after the collapse, where 39 consecutive
groups carried no signal. The final step, that scarce signal prevents recovery, is the best supported. From the
collapsed state, the skill did not recover without a correct sample and did recover after one. We therefore
conclude that GRPO's dependence on mixed-outcome groups can lock a collapsed skill in place. We do not conclude that
it amplifies small disadvantages in general. Where any correct samples remained, or where a related skill kept
learning, the deficit closed.

**Practical implications.** Three follow from the results. First, in multi-task GRPO, behaviours that no task
trains need monitoring, because style learned on one task spreads to others. Second, lags in joint training should
be checked against optimizer effects before they are attributed to interference; a learning-rate control shows
how sensitive the target is, although it cannot by itself establish equivalence with joint training. Third, a skill whose success
rate has fallen to zero can be revived by increasing the number of samples per prompt or by training a related,
easier variant alongside it.

## 5. Limitations

All results are exploratory. Most single-run findings rest on one seed, and the multi-seed findings on three to
five. The confidence intervals reported throughout describe uncertainty over evaluation problems within a run;
they are not a substitute for replication across training seeds, which was limited. The study used one model (Qwen3-1.7B), one learning rate, the Adam optimizer, and a small set of synthetic and
benchmark tasks. "No harm detected" means that the 95% interval included zero, typically within about ±0.08, not
that harm is absent. The cause of the within-task collapse remains unidentified. The brevity explanation for the
Caesar delay is a correlation across three seeds and was not tested by intervention; effects of the competitor's
earlier updates on the optimizer's state remain an alternative. The recovery experiment uses
one collapse state, and it does not separate the number of samples from the size of the update. Finally, the
half-learning-rate control was run for Caesar only. It shows that Caesar learning is sensitive to step size, but it
does not show that joint training is equivalent to any particular reduced step, so the Caesar-MBPP lag remains
unresolved between interference and optimizer effects.

## 6. Conclusion

In this setting, how tasks interact under GRPO depended on behaviour more than on domain. The clearest cross-task
delay came from an objective that rewarded short answers acting on a skill that needs written working; it appeared in
two of three seeds, and brevity spill-over is a promising but unconfirmed explanation. The largest side effects of
training appeared on behaviours that no task was training. A cross-domain lag could not be attributed to
interference, because the optimizer offers a credible alternative explanation. Within a single task, a
skill could collapse, and once it produced no correct samples GRPO could not repair it; a single success, or
transfer from a related skill, was enough to restart learning. The open questions are what triggers such collapses,
and whether the brevity effect can be confirmed by direct intervention.

## Appendix A. Training configuration

| Setting | Value |
|---|---|
| Model | Qwen3-1.7B, non-thinking mode |
| Samples per prompt | 8 (32 in one branching condition) |
| Prompt groups per round | 8 per task (16 in joint runs) |
| Optimizer | AdamW, learning rate 2×10⁻⁶ (1×10⁻⁶ in the control), β = (0.9, 0.999), no weight decay, gradient clip 1.0 |
| Updates | 4 optimizer steps per round; 32 rounds per run |
| Objective | clipped policy ratio (0.2), no KL penalty; loss averaged over tokens, then divided by 32 responses |
| Generation | temperature 1.0, at most 4,096 tokens |
| Evaluation | 64 held-out problems per task and level; 4 samples (selection set) or 8 (confirmation set) |

## Appendix B. Validity checks

- **Grader audit.** Three grader faults found in base-model calibration were fixed before any training run.
- **Data separation.** Training, selection and confirmation problems were disjoint, including commuted duplicates
  of addition problems.
- **Pipeline validation.** A canary run exercised two tasks, checkpointing, and a forced interruption and resume
  before the main experiments.
- **Independent review.** The within-task results were recomputed from raw metrics by a reviewer who did not use
  the study's analysis code. The reviewer rejected the initial claim that 8-digit updates trigger the collapse;
  the report reflects that correction.
- **Numerical audit.** A sample of 18 reported numbers was recomputed independently, with no discrepancies, and
  every evaluation was confirmed to use 64 problems.
- **Operational events.** One run was resumed after preemption and replayed eight rounds. Its evaluations were
  unaffected, and duplicated training records were excluded from all counts. When shared storage filled, two
  runs ended after their final evaluations, so their results are complete; a third was rerun from the start. One
  faulty compute node was excluded.

The experiment list, with job identifiers, code versions and final states, is in the accompanying file `runs.md`.
