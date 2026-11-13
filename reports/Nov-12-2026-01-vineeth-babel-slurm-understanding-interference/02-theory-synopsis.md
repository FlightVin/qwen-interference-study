# Interference: theory synopsis (10–15 minute read)

This is the synopsis of [`00-initial-theory.md`](00-initial-theory.md) for study `Nov-12-2026-01-vineeth-babel-slurm-understanding-interference`. See also [`03-understanding-synopsis.md`](03-understanding-synopsis.md), [`04-domains-to-explore.md`](04-domains-to-explore.md) and [`05-summary.md`](05-summary.md).

## Main points

*Interference* means that learning one task harms another task in the same network.

1. To first order, every definition reduces to one number: the inner product of two task gradients.
2. A negative inner product is not enough for harm. On-policy reinforcement learning (RL) also needs *coupling*: the learning signal for a task vanishes when success on that task is low.
3. Three factors reduce interference in theory: high dimension with low-dimensional task updates, over-parameterisation, and small steps from a good starting point. Pre-training supplies all three.
4. No paper the source found measures the gradient cosine between domains under a policy-gradient loss against model size or pre-training.

All figures are ours.

## 1. Definitions of interference

Notation: parameters θ, tasks k with objectives J_k and losses L_k = −J_k, gradients g_k = ∇_θ L_k, learning rate η.

| # | Name | What it measures |
|---|---|---|
| D1 | Gradient | Cosine ρ of two task gradients at one point θ. ρ < 0: interference. ρ > 0: transfer. |
| D2 | Outcome | A worse result on a task than a reference (negative transfer, forgetting). |
| D3 | One-step outcome | Change of task j's loss after one real step on task i. |
| D4 | Kernel | How much a step on input x moves the output on x'. |
| D5 | Representational | Whether two tasks use the same units or directions. |
| D6 | Weight-space | Whether separately learned updates can be added. |
| D7 | Ray interference | Negative D1 **and** coupling. A property of the whole training path. |

D1 is local: it says nothing about step size, curvature or the path. D2 matters in practice and is relative: "worse than what" must be stated. Only D7 needs RL.

Full text: `00-initial-theory.md`, Section 1.

## 2. The local theory: what one step does

**First order (standard).** Take one step on task A, θ' = θ − η g_A. The loss of task B becomes

    L_B(θ') = L_B(θ) − η ⟨g_B, g_A⟩ + (η²/2) g_Aᵀ H_B g_A + O(η³),   H_B = ∇²L_B(θ).        (1)

The step lowers L_B exactly when ⟨g_B, g_A⟩ > 0. To first order, for small η, D1, D3 and D4 are one quantity.

**Policy gradient.** The same rule holds for the log-probabilities of a language model. GRPO, the method of our runs, samples a group of answers per problem. A group whose answers all have the same reward gives no gradient. It cannot interfere with or help anything.

**Adam (our derivation).** Adam reweights coordinates, so the cross-task term can change sign. A raw-gradient cosine is not the last word for an Adam run.

**Second order.** Let h be the curvature of L_B along g_A. A step on A raises L_B, even when the gradients agree, when

    η h ‖g_A‖ / ‖g_B‖ > 2 cos(g_A, g_B).                                                          (4)

A small learning rate removes this curvature route, not the first-order route. In joint supervised training, harm also needs unequal gradient sizes and high curvature.

**Conflict rate.** The *conflict rate* is the share of gradient pairs with negative cosine. Its null value for independent random directions is 50%, not 0%. Reported rates of 32–42% between tasks are therefore mild alignment. A conflict rate needs a within-task baseline.

Full text: `00-initial-theory.md`, Section 2.

## 3. Ray interference: the on-policy RL theory

**Claim** (Schaul et al. 2019). A learning system suffers from plateaus if it has (a) negative interference between the components of its objective and (b) coupled performance and learning progress. Coupling arises because the agent generates its own training data.

**Exact model.** A bandit has two contexts and two actions. Success is J_1 = σ(θ_1 + θ_3) and J_2 = σ(θ_2 − θ_3). The shared action bias θ_3 is the only shared parameter. It gives ρ = −1/2 at every θ. Expected on-policy REINFORCE with small steps gives

    dJ_1/dt = 2 J_1²(1−J_1)² − J_1(1−J_1) J_2(1−J_2)                                             (5)
    dJ_2/dt = 2 J_2²(1−J_2)² − J_1(1−J_1) J_2(1−J_2)

The first term is a task's own progress. The second is the interference. The factor J_k(1−J_k) is the coupling.

**Results.**

1. *Winner-take-all (WTA) regions*: near the origin, J_2 > 2 J_1 makes context 1 get worse while context 2 improves. A random start direction near the origin lies in a WTA region with probability ≈ 59%.
2. Plateau: the path then passes close to a saddle point, (0,1) or (1,0), with speed ε = 2 J_1²(1−J_1)². The paper states a slowdown of order 1/ε.

![Flow of the two-context bandit](figures/t1_ray_flow.png)

*Figure T1. Left (on-policy): unbalanced starts run along the edges and pass a saddle corner before they reach (1,1). Middle (supervised): every start goes directly to (1,1).*

![Learning curves and slowdown](figures/t2_ray_curves.png)

*Figure T2. Left: on-policy curves stall at total performance 1, longer for less balanced starts. Right: time to reach J = 1.9. In our simulation it is t = 38 for a balanced start and t = 1,503 for ratio 1:9. Supervised learning needs t = 24 to 26.*

**The coupling function.** Schaul et al. write ∇J_k = f_k(J_k) v_k(θ), with f_k the *coupling function*. On-policy REINFORCE has f(u) = u(1−u). Supervised learning has f(u) = 1−u and the same ρ = −1/2, with no saddle and no plateau. The sufficient condition for a saddle is **f_k(0) = 0**.

Removing the shared parameter removes the WTA regions. 10% uniform random actions give f(0) > 0 and much weaker plateaus.

**Where it fails.** The paper does not show that real tasks have negative ρ. The condition f(0) = 0 fails with exploration noise, off-policy data, dense rewards or supervised signal. The case of more than two contexts and deep RL is not proven; all experiments are bandit simulations.

**GRPO coupling (our derivation).** Let p be the success probability of a problem and G the number of sampled answers. The expected update size is

    f_GRPO(p) = Σ_c Binom(c; G, p) √((c/G)(1 − c/G)).

f_GRPO(0) = 0, so the saddle condition still holds. The coupling is weaker than in REINFORCE: at p = 0.05 with G = 8, 0.117 against 0.0475. The derivation ignores ratio clipping and the token averaging of our implementation. Adam or gradient clipping weakens coupling further.

![Coupling functions](figures/t9_coupling.png)

*Figure T9. Left: the GRPO coupling lies above REINFORCE and still reaches zero at p = 0. Right: only mixed groups give a gradient.*

Full text: `00-initial-theory.md`, Section 3.

## 4. The kernel view: training near a starting point

A network is in the *kernel regime* when its output stays close to its first-order expansion around the starting point. Whether fine-tuning is in this regime is an empirical question for each setting. Nobody has tested GRPO.

**Forgetting** (Doan et al. 2021). Train on task A, then on task B, in the kernel regime:

    forgetting = ‖ U_A Σ_A (V_Aᵀ V_B) M_B ỹ_B ‖²,                                                 (6)

Three factors multiply: the high-variance directions of task A (Σ_A), the overlap of the two feature subspaces (V_Aᵀ V_B), and the error signal on task B (ỹ_B). Zero overlap or zero error signal gives zero forgetting.

**Task similarity.** In over-parameterised linear regression, forgetting is zero for identical and for orthogonal tasks. It is largest in between.

![Forgetting in linear models](figures/t6_linear_forgetting.png)

*Figure T6. Left: forgetting peaks at 45°. Middle: the peak at intermediate dissimilarity appears only for high over-parameterisation β. Right: forgetting falls as parameters per sample grow.*

"Very different domains interfere most" is only true for models with little spare capacity. Otherwise the danger zone is tasks that share features and need different outputs.

Full text: `00-initial-theory.md`, Section 4.

## 5. Why large pre-trained models may show little interference

| Argument | What the source states | Limit |
|---|---|---|
| High dimension (5.1) | Independent isotropic vectors in R^d have cosine of order 1/√d. For d = 1.72 × 10⁹ (Qwen3-1.7B without the tied output layer), 1/√d = 2.4 × 10⁻⁵. | A floor, not a prediction. Real gradients concentrate in few directions. |
| Capacity (5.2) | If the network can minimise all task losses at once, summed-loss training converges where every task gradient is zero. | This concerns the solution, not the path. Capacity reduces outcome interference without reducing gradient conflict. |
| Over-parameterisation (5.3) | Wider networks forget less. | — |
| Pre-training (5.4) | Pre-trained models forget far less, and the advantage grows with model size. | Causality not tested. Scale is contested for language models. |
| Small on-policy moves (5.5) | Theorem: policy gradient with binary reward converges to the optimal policy closest in KL divergence to the start. | Neural networks do not meet the convexity assumption exactly. |

Low-dimensional updates allow two opposite readings. *Room to avoid*: unrelated tasks use disjoint directions, so the cosine is ≈ 0, with no interference and no transfer. *Shared subspace*: the gradients concentrate in the same subspace, and the cosine can be large with either sign. A measured cosine distinguishes the two.

**Warning.** Random rewards gave Qwen2.5-Math-7B +21.4 points on MATH-500 (true rewards: +29.1). Uniform gains on all subjects can be this signature.

Full text: `00-initial-theory.md`, Section 5.

## 6. Known empirical results, by setting

| Finding | Setting | Source |
|---|---|---|
| Adding 256 easy problems to 256 hard ones lowers hard pass@1 from 13.55% to 8.22%; adding 1,000 lowers it to 2.24% | GRPO, Qwen3-4B-Instruct, math | Qu et al. 2026 (POPE), Table 2 |
| Math RL lowers code accuracy (67.46 → 60.60); code RL lowers math for the base model (56.40 → 50.80) and raises it for the instruct model (69.00 → 72.00) | GRPO, Qwen2.5-7B | Li et al. 2025 |
| Math-only RL raises other reasoning tasks (30.2 → 60.0) and non-reasoning tasks (45.7 → 53.2); supervised fine-tuning (SFT) lowers the latter (→ 21.1) | GRPO, Qwen3-14B-Base | Huan et al. 2025 |
| Cosine between task score functions about 10⁻¹ for SFT and 10⁻³ for RL | DeepSeek-R1-Distill-Qwen 1.5B and 7B | Zhu et al. 2026 |

![Reported results](figures/t8_reported.png)

*Figure T8. Top left: easy problems lower hard-set pass@1 in POPE. Bottom right: the conflict rate stays below the 50% null at every model size.*

POPE is the only study with easy-plus-hard mixtures under GRPO, and it reports an outcome, not a gradient cosine. No study covers two skills inside one subject, which is our case.

Full text: `00-initial-theory.md`, Section 6.

## 7. What we know about interference

| Status | Statements |
|---|---|
| **Proven** (under the stated assumptions) | Equations (1), (4) and (6). The bandit results, and f(0) = 0 as sufficient for a saddle. Linear-regression forgetting: zero for identical and orthogonal tasks, largest in between. A model that fits all tasks ends where all task gradients vanish. Convergence to the KL-closest optimal policy. Cosine of order 1/√d for independent directions. |
| **Empirically supported** | Outcome interference under on-policy RL exists in language models (easy against hard math; math against code). Larger and pre-trained models show less outcome interference in vision and in weight merging. Gradient cosine tracks task relatedness. RL post-training forgets less than SFT. Conflict rates are near 30–45%, similar within and between tasks. |
| **Conjecture or contested** | Ray interference explains plateaus in deep RL or language-model RL: proven only for bandits. Gradient conflict *causes* outcome interference. Scale by itself reduces interference. Language-model fine-tuning is linear in the weights: untested for RL. Why pre-training helps: correlation only. Superposition capacity governs interference between tasks: no direct evidence. How the cross-domain gradient cosine depends on model size: not measured. |

Full text: `00-initial-theory.md`, Section 7.

## 8. What this means for the study

A mixed training run shows **no** outcome interference under five conditions. `01-understanding.md` tests each one.

| Condition | Test |
|---|---|
| The domain gradients do not conflict (ρ ≥ 0) | cosine of per-domain GRPO gradients with within-domain baseline |
| The steps are small, so only first-order effects exist | linearity of the log-probability change |
| The coupling is weak: hard problems rarely fall to zero success | share of all-wrong groups |
| The interfering domain stops producing gradient | share of mixed practice groups over training |
| The evaluation cannot see an effect of the size that exists | predicted effect size against measurement noise |

Full text: `00-initial-theory.md`, Section 8.

## References

Label: **R** = text read for the cited facts.

- Doan, Bennani, Mazoure, Rabusseau, Alquier (2021). A theoretical analysis of catastrophic forgetting through the NTK overlap matrix. AISTATS. arXiv:2010.04003. **R**
- Huan et al. (2025). Does math reasoning improve general LLM capabilities? arXiv:2507.00432. **R**
- Li, Pan, Lin, Sun, He, Wu (2025). Can one domain help others? arXiv:2507.17512. **R**
- Qu, Setlur, Smith, Salakhutdinov, Kumar (2026). POPE: learning to reason on hard problems via privileged on-policy exploration. arXiv:2601.18779. **R**
- Schaul, Borsa, Modayil, Pascanu (2019). Ray interference: a source of plateaus in deep reinforcement learning. arXiv:1904.11455. **R** (full text, appendices checked)
- Zhu et al. (2026). SFT conflicts, RL coexists. arXiv:2608.03573. **R** (abstract and Section 4)
