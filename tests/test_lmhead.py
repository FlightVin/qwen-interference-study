"""A11: LM-head chunk recomputation gives the same log probs and gradients as storing the logits."""
import torch

from qis.train.learner import CHUNK, lm_head_logprobs


def test_recompute_matches_stored():
    torch.manual_seed(0)
    T, H, V = CHUNK * 2 + 37, 16, 50
    h0 = torch.randn(T, H, dtype=torch.float32)
    w0 = torch.randn(V, H, dtype=torch.float32)
    tgt = torch.randint(0, V, (T,))
    res = []
    for rec in (False, True):
        h, w = h0.clone().requires_grad_(), w0.clone().requires_grad_()
        lp = lm_head_logprobs(h, w, tgt, recompute=rec)
        (lp * torch.linspace(-1, 1, T, dtype=torch.float32)).sum().backward()
        res.append((lp.detach(), h.grad, w.grad))
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=0, atol=1e-6)
