# READING

Purpose: record what each source establishes, what stays uncertain, and what it means for our protocol. Access date for all web sources: 2026-11-07. These sources motivate the study. They are not results of this study.

## Project history (context/)

File: `context/_BC2RL__Can_Early_Behavior_Guide_Sparse_Reward_RL_Training_ (1).pdf` (27 pages, Bhat and Goyal, October 2026). Read by text extraction: synopsis, iteration transitions, conclusions, limitations.

- Iteration 1 (Qwen2.5-0.5B, LoRA, arithmetic): GRPO after extended BC raised overall success but lost more coverage on problems that BC had made intermittently solvable. The mechanism stayed unresolved; gradient assays failed validation.
- Iteration 2 (MiniGrid, PPO): demonstration-state starts and BC-preservation penalties gave no reliable benefit. Equivalence tests bounded effects within ±5 points. A subgroup effect chosen after seeing results did not replicate. No intervention-selection rule (RQ2) was found.
- Self-critique: the research questions outpaced the evidence, and follow-ups drifted into explaining one agent's failures.
- Iteration 3 (current): first test whether related easier practice helps hard problems relative to hard-only and unrelated practice. Match difficulty and length, define relatedness by a shared operation, and use paired short updates to separate dilution from damage. A reflection-based selector comes later.

## Ray Interference (Schaul et al., arXiv 1904.11455v1)

**Establishes.** Plateaus need two ingredients: negative interference between component gradients (cosine ρ < 0) and coupling of learning progress to current performance (on-policy sampling). In an exact 2×2 contextual-bandit toy (ρ = −1/2 everywhere), (0,1) and (1,0) are saddle points and learning stalls near them. The supervised analogue has no saddles and no plateaus. Removing either ingredient removes the plateaus (§2.6). §4 proposes controls: inspect individual-run curves (averaging hides plateaus); reduce interference (capacity, separate networks) or reduce coupling (fixed data, off-policy learning); a plateau that vanishes under these controls was likely ray interference.

**Uncertain.** The toy is a bandit with a shared bias, not an LLM. Interference can also be positive. Whether a trajectory hits a plateau depends on initialization.

**Consequences.** A flat or slower U curve does not establish ray interference. Report per-seed and per-problem curves, not only means. The local update diagnostic (§7) is our closest analogue of a control. It measures actual Adam/GRPO update effects, not gradient cosines.

## POPE (Qu et al., arXiv 2601.18779v1)

**Establishes.** On a hard set with near-zero base reward (no correct rollout in 128 samples, Qwen3-4B-Instruct-2507), mixing in easy problems plateaus below hard-only. Table 2 (hard pass@1 / pass@16): hard-only 13.55 / 32.89; hard + easy 8.22 / 23.81; hard + guide 15.50 / 42.53. In the two-problem study (§3.2, Fig. 5), an unrelated easy problem slowed hard-problem learning. A related easy problem, chosen by embedding cosine similarity and matched in base success, only partly mitigated this. If all rollouts fail, the GRPO gradient on that prompt is exactly zero.

**Uncertain.** Fig. 5 uses one hard problem; seed count and numbers are not given. The text says related mixing "slightly accelerates" training, while the caption says it is slower than hard-only. No table compares related + hard against hard-only. Appendix E optimizer details (lr 1e-6, batch 32) came from a fetched summary, not verbatim text. The recipe is asynchronous (Pipeline-RL, ring buffer of 128, ratio clip 0 to 5).

**Consequences.** Do not expect R > H. Our primary contrast is R − U. Our hard pool (2–8/32 successes) is easier than POPE's zero-success regime, so reward scarcity is milder. We use a synchronous, on-policy setup and do not copy the asynchronous recipe.

## Qwen3 Technical Report (arXiv 2505.09388v1) and Qwen3-1.7B model card

**Establishes.** Reasoning-RL queries (§4.2) were "learnable for the cold-start model" and "as challenging as possible": 3,995 query-verifier pairs, GRPO, large batch and many rollouts, entropy kept stable or rising. No numeric hyperparameters are given. §4.5: Qwen3-1.7B is one of the five dense models built by strong-to-weak distillation (off-policy from /think and /no_think teacher outputs, then on-policy logit KL to a teacher). The model card: non-thinking mode uses `enable_thinking=False`; recommended non-thinking sampling is T=0.7, top_p=0.8, top_k=20, min_p=0; do not use greedy decoding; transformers ≥ 4.51.0. The pinned `tokenizer_config.json` (main branch) emits `<think>\n\n</think>\n\n` after the generation prompt when `enable_thinking` is false.

**Uncertain.** The report does not state whether 1.7B received any direct RL. The card does not describe the empty think block; only the template shows it.

**Consequences.** Our checkpoint is distilled from RL-trained teachers. Our decoding (T=1.0, top_p=1.0, no top_k) is off the card's recommendation, which is deliberate for exploration; measure the truncation and repetition rate in the length probe. The rendering fixture must assert the exact empty think block. Rollout and training tokenization must treat it as prompt, not response, and the reward must ignore it.

## DAPO (arXiv 2503.14476v2)

**Establishes.** Clip-higher (§3.1): with ε=0.2, a token at probability 0.01 can only rise to 0.012, which restricts exploration and drives entropy collapse. DAPO sets ε_low=0.2, ε_high=0.28 and keeps ε_low to avoid suppressing tokens to zero. §4.1: AdamW, constant lr 1e-6 with 20-step linear warm-up, 512 prompts × 16 responses, minibatch 512 (16 updates per rollout step), max generation 20,480 tokens, no KL term, base model Qwen2.5-32B.

**Uncertain.** Values were tuned for a 32B model with token-level loss and dynamic sampling. Transfer to a 1.7B model with 4 updates per round is untested.

**Consequences.** Our grid {5e-7, 1e-6, 2e-6} × {(0.2,0.2), (0.2,0.28)} brackets DAPO's point. We omit warm-up, dynamic sampling, token-weighted loss and length shaping. Our loss is per-response token mean, then mean over responses (original GRPO style).

## Math-Verify (github.com/huggingface/Math-Verify, CHANGELOG top entry 0.8.1)

**Establishes.** API: `verify(gold, answer)`; order matters. `parse(..., extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()], parsing_timeout=5)`; `verify(..., float_rounding=6, numeric_precision=15, strict=True, timeout_seconds=5)`. Both default to `try_extract_without_anchor=True`; `boxed_match_priority=50` by default, and the README advises 0 for predictions. Timeouts use `signal.alarm()`; in threads `parse`/`verify` raise `ValueError`, and `timeout=None` disables protection. On error, `parse` returns `[]` and `verify` returns `False`.

**Uncertain.** No tagged release was visible; the `parse` docstring says timeout 3, the signature says 5. Default anchorless extraction can match a number in the reasoning.

**Consequences.** Pin a commit. Extract only the last `\boxed{}` with our own code, then call `parse` on that span. Set `boxed_match_priority=0` and disable anchorless extraction. The default returns `False` on timeout, which would score unknown as wrong. Run each verification in a separate process with an outer 5 s / 15 s limit, and record a timeout as unknown.

## Open questions to resolve before freeze

1. Does the pinned verl reduce loss as "token mean per response, then mean over responses", and do its minibatch fields count prompts or responses?
2. Does the verl/vLLM rollout path render the empty think block identically to `apply_chat_template`, and is it masked from the loss?
3. Can Math-Verify run with `signal.alarm()` in our worker processes, or must we disable its timeout and enforce our own?
4. Does T=1.0 without top_k cause the >5% truncation or repetition that triggers the 8k-cap test?
5. Pin exact revisions: Qwen3-1.7B commit hash, Math-Verify commit, verl commit, transformers and vLLM versions.
6. POPE Appendix E values were not read verbatim. Read them before citing them.
