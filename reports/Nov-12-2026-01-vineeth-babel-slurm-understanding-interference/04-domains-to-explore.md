# Domains to explore: mixtures that can produce ray interference on Qwen3-1.7B

Purpose: a reference list of domains and dataset mixtures for the next experiments. The first study had neither
precondition of ray interference: no negative coupling between domains, and no domain near zero success. This
document lists mixtures that supply one or both, with the base performance of Qwen3-1.7B and the evidence for each.

**Main point.** We measured 16 domains at the base model and found no pair of naturally occurring domains with
negative coupling. Hardness is easy to supply: a procedural task gives any success rate from 98% to 0%. Negative
coupling must be built into the reward. Section 4 lists three ways to do it and one measured case.

Evidence labels: **M** = measured in this study on Qwen3-1.7B; **P** = published, verified against the source
(`research/Nov-12-2026-01-vineeth-babel-slurm-understanding-interference/LIT-NOTES.md`); **S** = our expectation,
not measured.

## 1. What a mixture must supply

Ray interference (Schaul et al. 2019; `00-initial-theory.md`, §3) needs two things at the same time.

| Precondition | Meaning for GRPO | First study |
|---|---|---|
| Negative coupling | improving domain A lowers the success of domain B; the expected gradients have negative cosine | not present: all cosines ≥ 0 |
| A component that starts near zero | domain B has almost no correct answers, so almost no group gives a gradient; its learning speed depends on its own success | not present: hard pool at 2–12 of 32 |

## 2. Measured: base performance and coupling of 16 domains

Method. Qwen3-1.7B base, the first study's prompt, sampler and grader (boxed final answer, 12,288-token limit).
100 problems per domain (AIME: 30), 16 answers per problem, 2 groups of 8. "Mixed groups" have at least one correct
and one wrong answer; only they give a gradient. Coupling is the mean cosine between group gradients of different
problems, in units of 10⁻⁴, with a 95% interval (bootstrap over problems). Jobs `qis-ui-probe`, `qis-ui-probe2`; code
`src/qis/interf/build_probe_panel.py`, `grad_geom.py`, `summ_reward.py`; data
`slurm-agentic-engineering/results/qis-ui-probe/`.

| Domain | Example | pass@1 | Mixed groups | Cosine inside the domain (10⁻⁴) | Cosine with hard algebra (10⁻⁴) | Band |
|---|---|---|---|---|---|---|
| `mult2`: product of two 2-digit integers | 18 × 40 | 97.7% | 27 of 200 | +121 [−160, +557] | +1.1 [−10.8, +11.4] | solved |
| `mult3` | 408 × 125 | 90.8% | 87 | +128 [+55, +239] | +8.3 [+0.1, +17.5] | solved |
| `mult4` | 4446 × 4505 | 76.6% | 148 | +102 [+67, +146] | +3.2 [−2.2, +8.6] | some success |
| `mult5` | 29090 × 13198 | 42.7% | 164 | +73 [+39, +113] | +4.5 [−0.4, +10.6] | some success |
| `mult6` | 788012 × 971041 | 5.5% | 59 | +104 [+33, +184] | +8.7 [+0.2, +18.3] | marginal |
ZERO-ROWS
| `add10`: sum of two 6-digit integers, digits 0–8 | 544127 + 112573 | 91.0% | 51 | +17 [−123, +249] | −1.9 [−10.2, +6.9] | solved |
| `add9`: the same strings, read and answered in base 9 | same, "written in base 9" | 19.9% | 130 | +185 [+113, +270] | +10.2 [−1.3, +21.3] | marginal |
| `mult10s`: product of two 2-digit strings, digits 0–8 | 65 × 20 | 96.3% | 32 | +304 [+46, +799] | +3.4 [−6.7, +14.1] | solved |
| `mult9`: the same strings in base 9 | same, "written in base 9" | 78.8% | 134 | +45 [+16, +79] | +3.1 [−2.3, +9.2] | some success |
| `gsm8k`: word problems (`openai/gsm8k`, test, first 100) | | 81.1% | 71 | −9 [−35, +26] | −0.3 [−7.4, +6.0] | some success |
| `arc`: science multiple choice (`allenai/ai2_arc`, ARC-Challenge test) | | 66.2% | 143 | +92 [+60, +129] | +1.8 [−3.0, +6.3] | some success |
| `aime24`: competition math (`Maxwell-Jia/AIME_2024`) | | 11.0% | 15 of 60 | +45 [−72, +138] | +8.5 [−12.2, +30.3] | marginal |
| Hard algebra, factoring drills, remainder drills, other subjects (first study pools) | | 41% on the dev set | 1,512 of 1,888 (64 answers) | +2.5 to +12.5 | +2.5 [−4.6, +10.8] (hard with hard) | some success |

Four results.

1. **No pair has negative coupling.** Among the 114 pairs that include a new domain, no interval lies below zero. The
   lowest values belong to `add10`: −36 [−93, +24] with `add9`, −65 [−242, +78] with `mult10s`, and −12.5
   [−31.6, +3.0] with the remainder drills. `add10` has only 51 mixed groups, so these intervals are wide.
2. **Procedural tasks have strong structure inside a family.** Multiplication pools of neighbouring size have
   cosines of +76 to +88 (the four pairs from `mult2`–`mult3` to `mult5`–`mult6`). This is 10 to 30 times the cosine between the math pools
   of the first study. Transfer inside the family should be large.
3. **A stated rule change decouples a task.** Base-9 multiplication has cosine −1.5 to +4.3 with every base-10
   multiplication pool, and +45 with itself. The model treats it as a separate task, not as an opposing one.
4. **Hardness is a clean dial.** pass@1 falls from 97.7% to 5.5% between 2 and 6 digits. ZERO-SENTENCE

## 3. Published reference points

| Fact | Source | Label |
|---|---|---|
| Qwen3-1.7B, non-thinking: MATH-500 73.0, AutoLogi 59.8, AIME'24 13.4, ZebraLogic 12.8, LiveCodeBench v5 11.6 | Qwen3 Technical Report, arXiv:2505.09388, Tables 18 and 20 | P |
| Qwen3-1.7B base model: GSM8K 75.44, MATH 43.50, MBPP 55.40, MMLU 62.63 | same, Tables 7 and 8 | P |
| GRPO on puzzles lowers code: CountDown-only training, code average 67.46 → 29.59 (Qwen2.5-7B-Base). The base model starts at CountDown 1.05% and Zebra 0.27% | Li et al. 2025, arXiv:2507.17512 | P |
| Order matters: science 38.3 → 25.4 when math training follows science training (Qwen3-4B-Base, GRPO) | Yang et al. 2026, arXiv:2602.01365 | P |
| Easy problems starve hard ones: hard pass@1 13.55% → 8.22% with 256 easy problems added, → 2.24% with 1,000 (Qwen3-4B-Instruct, GRPO) | Qu et al. 2026 (POPE), arXiv:2601.18779 | P (read in `00`) |
| Code gradients are larger: squared norm "up to 15× larger than MATH" | Wu et al. 2025, arXiv:2510.19178 | P |
| No published study of length or brevity rewards against correctness under GRPO | our search, 2026-11-12 | not found |

The published cases of harm are at 4B and 7B, and none reports a gradient cosine. Whether they reproduce at 1.7B is
unknown.

## 4. How to obtain negative coupling

BREVITY-SECTION

Two further constructions give a conflict by design. Neither is measured.

| Construction | How | Dial | Label |
|---|---|---|---|
| Contradictory rule on the same prompts | Use the `add10` strings with no cue. Domain A rewards the base-10 sum; domain B rewards the base-9 sum of the same prompt. The reward of B is zero for every answer that A rewards | share λ of prompts whose two answers differ; share of B in the batch | S: the gradients oppose exactly on shared prompts; the coupling across prompts is not known |
| Cue strength | Keep the two rules but state the rule in the prompt (`add9` as measured: decoupled). Then shorten or remove the cue | cue present / weak / absent | S: moves from "decoupled" (M, result 3) to "opposed" |

## 5. Candidate mixtures

Axis 1 is the expected scale of interference. Axis 2 is the starting success of the second domain. Domain A is the
hard algebra pool of the first study unless stated.

| # | Mixture (A + B) | Interference expected | Start of B | Evidence | Data |
|---|---|---|---|---|---|
| 1 | Hard algebra + factoring drills | none (reference) | some: 12–24 of 32 | M: cosine ≥ 0, first study outcome | first study pools |
| 2 | `mult4` + `mult5` | none; positive transfer | some: 43% | M: cosine +76 | procedural |
| 3 | `mult4` + `mult9` | none; no transfer (decoupled) | some: 79% | M: cosine +2.0 [−9.3, +14.3] | procedural |
| 4 | Hard algebra + `arc` or `gsm8k` | none; no transfer | some: 66%, 81% | M: cosine +1.8, −0.3 | `allenai/ai2_arc`, `openai/gsm8k` |
| 5 | `mult5` + `mult6`, then MULT-NEXT | starvation only: B gives few gradients, the cosine is positive | marginal: 5.5%; none: MULT-NEXT-RATE | M for the cosine; S for the outcome | procedural |
| 6 | Hard algebra + AIME 2024 | starvation only | marginal: 11.0% | M: cosine +8.5 [−12.2, +30.3] | `Maxwell-Jia/AIME_2024` (30 problems; use as a probe, not a training pool) |
| 7 | Correctness + brevity on the same problems | BREVITY-ROW | some | BREVITY-EVID | first study pools |
| 8 | Math + code | some (published at 7B: −3 to −7 points) | some: MBPP 55.4 (P) | P at 7B; not measured here | `google-research-datasets/mbpp`, `openai/openai_humaneval`; needs a sandboxed verifier |
| 9 | Code + CountDown puzzle | great (published at 7B: training on CountDown alone lowers the code average by 38 points) | none at 7B: 1.05% (P) | P at 7B; not measured here | `open-thought/reasoning-gym` (`countdown`); needs a verifier for expressions |
| 10 | `add10` rule against `add9` rule, no cue | great, by construction | some (A: 91%) and none (B answers are never sampled) | S | procedural (Section 4) |
| 11 | Mixture 10 with B = MULT-NEXT in place of `add9` rule, plus brevity | both preconditions at once | none | S | procedural |

Mixtures 1–4 are controls: they should show no interference, with and without transfer. Mixtures 5–6 supply the
zero-start precondition alone. Mixtures 7–10 supply negative coupling alone, at three strengths. Mixture 11 supplies
both and is the direct test of ray interference.

## 6. Other axes worth varying

| Axis | Why it matters for the question "what do pre-training and scale do?" | How to vary it |
|---|---|---|
| Conflict strength λ | gives a dose-response curve, not one point; a capacity effect would shift the curve with model size | share of prompts or reward weight that opposes (mixtures 7 and 10) |
| Familiarity from pre-training | pre-training can align only what it has seen. Base 10 against base 9 holds the task fixed and changes familiarity (91% against 20% on the same strings) | `add10`/`add9`, `mult10s`/`mult9`; counterfactual tasks of Wu et al. 2023, arXiv:2307.02477 |
| Cue in the prompt | with a cue the model can separate two rules (M: decoupled); without it the rules must compete | cue present, weak, absent |
| Task family distance | coupling inside a procedural family is 10–30 times the coupling between math subjects | neighbours in a family; across families |
| Model size | the capacity hypothesis predicts more harm in a smaller model at equal conflict | Qwen3-0.6B, 1.7B, 4B with pools rebuilt to equal start rates (`01`, E2) |
| Update rank | a capacity dial at one size | LoRA rank 1, 8, 64 (`01`, E6) |
| Mixture ratio and order | POPE and Yang et al. report effects of the easy share and of the training order | share of B in the batch: 1/8, 1/2, 7/8; A then B against mixed |
| Answer length | all trained models of the first study wrote 3.3 times longer answers; length is a shared resource under a token limit | token cap; brevity reward |
| Weight precision | BF16 rounding hides small updates (`01`, §1.6) and may hide or delay interference | FP32 against BF16 forward pass (`01`, E4) |

## 7. Recommended first set

1. Measure before training. For every candidate pair, run the base-model gradient probe (64 answers per problem,
   about 3 GPU-hours per pair). Train only pairs whose coupling is measured.
2. Train mixtures 1, 3, 7 and 10 with single-domain references, two seeds: one reference, one decoupled pair and
   two conflict strengths. Estimated cost: 4 mixtures × 2 seeds + 5 single-domain runs × 2 seeds = 18 runs of about
   25 GPU-hours = 450 GPU-hours. Procedural pools produce short answers, so these runs are likely cheaper; we did not
   measure this.
3. Then mixture 11 and the size and rank axes on the mixture with the clearest conflict.

## 8. Limits

- Coupling was measured at the base model only. It can change during training (`01`, §1.3).
- Each new domain has 100 problems and 16 answers. Intervals for domains with few mixed groups (`mult2`, `add10`,
  `mult10s`, `aime24`) are wide.
- The cosine uses a count sketch with noise of about 0.0007 per pair; means over thousands of pairs are precise.
- Rows labelled S are expectations. Rows labelled P come from larger models.
- Dataset ids were read from the Hugging Face hub on 2026-11-12. `open-thought/reasoning-gym` and
  `ZhaofengWu/counterfactual-evaluation` are GitHub repositories.
