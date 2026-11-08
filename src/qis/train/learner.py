"""Learner: full-parameter Qwen3 policy with FP32 weights, BF16 autocast compute (addendum A1).

- Trainable parameters are FP32 and owned by torch AdamW (FP32 moments). No second FP32 copy.
- Transformer body runs under BF16 autocast with gradient checkpointing.
- The LM head runs in FP32 on chunks of hidden states (hidden cast to FP32, FP32 weight, TF32
  disabled), and log-softmax and selected-token log probs are FP32. Full-vocabulary logits are
  never materialised for the whole sequence: only `chunk` positions at a time.
- Gradients accumulate in the FP32 .grad buffers across micro-batches of one response.
- Dropout modules are forced to eval mode in every forward.
"""
import torch
import torch.utils.checkpoint

CHUNK = 512


def _chunk_lp(hc, w, tc):
    """FP32 selected-token log probs for one chunk of hidden states (TF32 off, autocast disabled)."""
    with torch.autocast("cuda", enabled=False):
        logits = hc.float() @ w.t()
        lse = torch.logsumexp(logits, dim=-1)
        sel = logits.gather(-1, tc[:, None])[:, 0]
    return sel - lse


def lm_head_logprobs(h, w, tgt, recompute: bool):
    """Chunked FP32 LM head. recompute=True (training, A11) recomputes each chunk's logits in backward instead of
    storing them (identical arithmetic; memory O(chunk) instead of O(sequence x vocab))."""
    out = []
    for c in range(0, h.shape[0], CHUNK):
        hc, tc = h[c:c + CHUNK], tgt[c:c + CHUNK]
        if recompute and torch.is_grad_enabled():
            out.append(torch.utils.checkpoint.checkpoint(_chunk_lp, hc, w, tc, use_reentrant=False))
        else:
            out.append(_chunk_lp(hc, w, tc))
    return torch.cat(out)


class Learner:
    def __init__(self, model_path: str, lr: float, device="cuda"):
        from transformers import AutoModelForCausalLM
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0, foreach=True, fused=False)
        self.device = device

    def _no_dropout(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

    def token_logprobs(self, prompt_ids: list, resp_ids: list, train: bool) -> torch.Tensor:
        """FP32 log probs of resp_ids given prompt_ids (teacher forcing). Shape [len(resp_ids)]."""
        self.model.train(train)
        self._no_dropout()
        ids = torch.tensor([prompt_ids + resp_ids], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = self.model.model(input_ids=ids).last_hidden_state[0]
        P = len(prompt_ids)
        h = hid[P - 1:-1]                       # positions predicting each response token
        tgt = ids[0, P:]
        w = self.model.lm_head.weight           # FP32 (tied to embeddings in Qwen3-1.7B)
        return lm_head_logprobs(h, w, tgt, recompute=train)

    def token_entropy(self, prompt_ids: list, resp_ids: list) -> torch.Tensor:
        """Mean token entropy (FP32) of the policy over the response; no grad."""
        with torch.no_grad():
            self.model.eval()
            ids = torch.tensor([prompt_ids + resp_ids], device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hid = self.model.model(input_ids=ids).last_hidden_state[0]
            h = hid[len(prompt_ids) - 1:-1]
            w = self.model.lm_head.weight
            tot = torch.zeros((), device=self.device)
            for c in range(0, h.shape[0], CHUNK):
                with torch.autocast("cuda", enabled=False):
                    lp = torch.log_softmax(h[c:c + CHUNK].float() @ w.t(), -1)
                tot += -(lp.exp() * lp).sum()
            return tot / max(1, h.shape[0])

    def grad_norm_and_clip(self, max_norm: float):
        params = [p for p in self.model.parameters() if p.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=2.0,
                                              error_if_nonfinite=False, foreach=True)
        return norm.float()

    def step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
