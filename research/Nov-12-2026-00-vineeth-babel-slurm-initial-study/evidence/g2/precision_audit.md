| item | measured |
|---|---|
| checkpoint tensors (disk) | `{"BF16": 311}` |
| rollout (vLLM params) | `{"torch.bfloat16": 226}` |
| learner compute (body output under autocast) | `"torch.float32"` |
| selected-token log probs | `"torch.float32"` |
| optimizer-owned params | `{"torch.float32": 310}` |
| gradients (accumulated) | `{"torch.float32": 310}` |
| AdamW exp_avg / exp_avg_sq | `{"exp_avg": {"torch.float32": 310}, "exp_avg_sq": {"torch.float32": 310}, "step": {"torch.float32": 310}}` |
| restart weights / moments | `[{"torch.float32": 310}, {"torch.float32": 620}]` |
| first-step grad norm | `1.1425613164901733` |
| updates accumulate (4 vs 1 step) | `true` |
| sync == bf16(learner) | `{"model.layers.13.post_attention_layernorm.weight": true, "model.layers.26.post_attention_layernorm.weight": true, "model.layers.26.self_attn.o_proj.weight": true, "model.layers.21.input_layernorm.weight": true}` |
| restart bit-exact (w, m) | `[true, true]` |
| peak GPU GiB | `44.58` |
