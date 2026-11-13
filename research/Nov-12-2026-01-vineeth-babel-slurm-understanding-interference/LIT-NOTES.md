# Literature notes: capacity and candidate domains (study 01)

Purpose: verified facts that Section 1.7 of `01-understanding.md` and `04-domains-to-explore.md` cite. Each arXiv
entry marked **V** was checked on 2026-11-12 against the abstract page and the HTML full text (title, authors, the
quoted number). Entries marked **A** rest on the abstract only. Entries marked **NV** are not verified; do not cite
them as facts.

## 1. Capacity

Main point of the review: every paper that supports "larger models interfere less" varies model size, width or the
trainable subspace. No measurement on checkpoints of one size tests capacity.

| Ref | Fact |
|---|---|
| Aghajanyan, Zettlemoyer, Gupta 2020, arXiv:2012.13255 **V** | Intrinsic dimension d90 (SAID) on MRPC: BERT-Base 1608, BERT-Large 1037, RoBERTa-Base 896, RoBERTa-Large 207 (Table 1). d90 falls with pre-training and with model size. Needs training in a random subspace. |
| Li, Farkhoor, Liu, Yosinski 2018, arXiv:1804.08838 (R by the review agent, not re-verified) | Intrinsic dimension by random-subspace training; extra parameters add redundancy. |
| Hu et al. 2021 (LoRA), arXiv:2106.09685 (R by the review agent) | Rank 1 on query and value matrices nearly matches rank 8 on GPT-3 (WikiSQL 73.4 against 73.8). |
| Mukherjee, Yuan, Hakkani-Tur, Peng 2025, arXiv:2505.11711 **V** | "updating only a small subnetwork comprising just 5%-30% of the parameters". "All models in our experiments are in the bfloat16 data type." Values equal when absolute difference ≤ 1e-5. "the updates to almost all parameter matrices are nearly full-rank". |
| Shenfeld, Pari, Agrawal 2025, arXiv:2509.04259 **V** | Forward KL on the new task is the consistent predictor of forgetting. "the reason for the observed sparse updates was the use of bfloat16 for model training"; float32 training gave "identical performance but without any sparsity". |
| Zhu et al. 2025, arXiv:2511.08567 **V** | Sub-ULP changes are not representable in bf16. But: "sparsity cannot be explained by precision alone"; precision is "an amplifier for visible sparsity". |
| Miahi, Belilovsky 2026, arXiv:2602.03839 **V** | GRPO: "Mean per-step sparsity is approximately 99%"; "updates smaller than roughly \|w\|/256 cannot be represented". Gradients are dense. |
| Gao, Gao 2026, arXiv:2610.41073 **V** | FP32 master weights, BF16 served copy: "on Qwen3-1.7B only 0.4–0.5% of served weights change per step". Proposes unbiased rounding. |
| Ilharco et al. 2022, arXiv:2212.04089 (R by the review agent) | Cosine between task vectors of CLIP tasks is typically 0.01–0.06. |
| Ortiz-Jimenez, Favero, Frossard 2023, arXiv:2305.12827 **V** | Footnote 2: "larger models are more weight disentangled" (CLIP ViT, Appendix D.3). |
| Yadav et al. 2024, arXiv:2410.03617 **V** | "larger models are consistently easier to merge" (PaLM-2, 1B–64B). |
| Yadav et al. 2023 (TIES), arXiv:2306.01708 (R by the review agent) | Keeping the top 20% of a task vector by magnitude retains performance. |
| Luo et al. 2023, arXiv:2308.08747 **V** | "as the model scale increases from 1b to 7b parameters, the severity of forgetting intensifies" (BLOOMZ; the authors note higher initial performance of the larger model as a possible cause). |
| Mirzadeh et al. 2022, arXiv:2110.11526 (R by the review agent) | Width reduces forgetting, depth does not; wider networks keep task gradients closer to orthogonal. |
| Shaham et al. 2023, arXiv:2212.07530 **A** | Multilingual translation: substantial interference mainly when the model is very small relative to the data. |
| Ramasesh, Lewkowycz, Dyer 2022 (ICLR) **A** | Pre-trained models forget less and the effect grows with scale. |

## 2. Interference between domains under RL post-training

| Ref | Fact |
|---|---|
| Li et al. 2025, arXiv:2507.17512 **V** | GRPO on Qwen2.5-7B-Base and -Instruct. Base model starts at CountDown 1.05% and Zebra 0.27% (Table 4). CountDown-only training: code average 67.46 → 29.59 on the Base model. "Math" (DeepScaleR + CountDown): code average 64.23 (−3.23); DeepScaleR alone 60.60. Math+Puzzle: HumanEval 70.12 → 25.00 (Table 12). |
| Yang et al. 2026, arXiv:2602.01365 **V** | GRPO on Qwen3-4B-Base; math, science, logic, puzzle. Science 38.3 → 25.4 when math training FOLLOWS science training; math then science gives science 40.8. Mixed math+science+puzzle: math 79.92 against 82.76 single-domain (difference −2.84 by subtraction; not a number stated in the paper). |
| Wu et al. 2025, arXiv:2510.19178 **V** | Squared gradient norm of code "up to 15× larger than MATH". No gradient cosine measured. |
| Huan et al. 2025, arXiv:2507.00432 **V** | Qwen3-14B-Base, math-only data: non-reasoning average base 45.7, SFT 21.1 (think) / 29.0 (no-think), RL 53.2. The table's other-reasoning average for RL (60.0) does not equal the mean of its four listed scores. |
| Shao et al. 2025, arXiv:2506.10947 **V** | Qwen2.5-Math-7B, MATH-500: +21.4 points with random rewards, +29.1 with true rewards. |
| Qu et al. 2026 (POPE), arXiv:2601.18779 (R in `00-initial-theory.md`) | GRPO, Qwen3-4B-Instruct: adding 256 easy problems to 256 hard ones lowers hard pass@1 from 13.55% to 8.22%; adding 1,000 lowers it to 2.24%. |
| Not found | No verified study of length drift under GRPO, of concise-against-verbose objectives, or of refusal-against-compliance RL. |

## 3. Qwen3 small models (Qwen3 Technical Report, arXiv:2505.09388 **V**)

| Benchmark | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Mode |
|---|---|---|---|---|
| MMLU | 52.81 | 62.63 | 72.99 | base model (Tables 7, 8) |
| GSM8K | 59.59 | 75.44 | 87.79 | base model |
| MATH | 32.44 | 43.50 | 54.10 | base model |
| MBPP | 36.60 | 55.40 | 67.00 | base model |
| MATH-500 | 55.2 | 73.0 | 84.8 | post-trained, non-thinking (Tables 18, 20) |
| AIME'24 | 3.4 | 13.4 | 25.0 | post-trained, non-thinking |
| ZebraLogic | 4.2 | 12.8 | 35.2 | post-trained, non-thinking |
| AutoLogi | 37.4 | 59.8 | 76.3 | post-trained, non-thinking |
| LiveCodeBench v5 | 3.6 | 11.6 | 21.3 | post-trained, non-thinking |

Verified individually: all Qwen3-1.7B values, and ZebraLogic for 0.6B and 4B. The other 0.6B and 4B values were read
by the review agent only.

## 4. Datasets (Hugging Face ids read from the hub API by the review agent on 2026-11-12)

| Domain | Id | Size | Check |
|---|---|---|---|
| Easy math | `openai/gsm8k` (MIT) | test 1,319 | number |
| Hard math | `EleutherAI/hendrycks_math` (MIT); `HuggingFaceH4/MATH-500` | 12,500; 500 | boxed answer |
| Competition math | `Maxwell-Jia/AIME_2024` (MIT) | 30 | integer |
| Code | `google-research-datasets/mbpp` (CC-BY-4.0); `openai/openai_humaneval` (MIT); `livecodebench/code_generation_lite` | 974; 164; — | unit tests |
| Procedural tasks | `open-thought/reasoning-gym` on GitHub (Apache-2.0) **V**: `countdown`, `sudoku`, `zebra_puzzles`, `base_conversion`, `caesar_cipher`, `knights_knaves`, `rubiks_cube` | unlimited, tunable | `score_answer` per task |
| Zebra puzzles | `allenai/ZebraLogicBench` | 1,000 grid; 3,259 multiple choice | grid or letter |
| Multiple choice science | `allenai/ai2_arc` (CC-BY-SA-4.0) | Challenge test 1,172 | letter |
| Multiple choice knowledge | `cais/mmlu` (MIT) | test 14,042 | letter |
| Factual recall | `akariasai/PopQA` | test 14,267 | alias match |
| Counterfactual arithmetic | `ZhaofengWu/counterfactual-evaluation` on GitHub (MIT); Wu et al. 2023, arXiv:2307.02477 | generator | exact match |
| Sycophancy | `Anthropic/model-written-evals` (CC-BY-4.0), folder `sycophancy/` | 3 files | A/B match |
| Chess | `Lichess/chess-puzzles` (CC0) | 6,157,341 | move string |
