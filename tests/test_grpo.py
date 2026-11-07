import math

import numpy as np
import pytest
import torch

from qis.train import grpo


def test_population_std_advantage():
    r = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    a = grpo.group_advantages(r)
    std = math.sqrt(0.125 * 0.875)            # population std, not Bessel
    assert a.dtype == torch.float32
    assert a[0, 0].item() == pytest.approx(0.875 / (std + 1e-6), rel=1e-6)
    assert a[0, 1].item() == pytest.approx(-0.125 / (std + 1e-6), rel=1e-6)
    assert abs(a.sum().item()) < 1e-5


def test_all_equal_groups_exactly_zero_and_group_boundaries():
    r = torch.tensor([[1.0] * 8, [0.0] * 8, [1, 1, 0, 0, 1, 1, 0, 0]])
    a = grpo.group_advantages(r)
    assert torch.equal(a[0], torch.zeros(8)) and torch.equal(a[1], torch.zeros(8))
    # third group normalized on its own (boundaries respected): mean .5, pop std .5
    assert a[2, 0].item() == pytest.approx(0.5 / (0.5 + 1e-6), rel=1e-6)


def test_rewarded_response_signed_contribution():
    lp = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    obj, st = grpo.response_objective(lp, lp.detach().clone(), 2.0, 0.2, 0.28)
    assert obj.item() == pytest.approx(2.0)      # ratio 1 -> objective = A
    (-obj).backward()                            # loss = -objective
    assert torch.all(lp.grad < 0)                # descent raises rewarded token log probs
    assert lp.grad[0].item() == pytest.approx(-2.0 / 3)


@pytest.mark.parametrize("ratio,adv,expect,active", [
    (1.5, 1.0, 1.28, 1), (1.5, -1.0, -1.5, 0), (0.5, -1.0, -0.8, 1), (0.5, 1.0, 0.5, 0),
    (1.1, 1.0, 1.1, 0), (3.0, -1.0, -3.0, 0)])     # last: no dual clip for negative A
def test_clip_cases(ratio, adv, expect, active):
    old = torch.tensor([0.0])
    obj, st = grpo.response_objective(torch.log(torch.tensor([ratio])), old, adv, 0.2, 0.28)
    assert obj.item() == pytest.approx(expect, rel=1e-6)
    assert st["clip_active"].item() == active


def _toy(lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(4, generator=g, requires_grad=True)
    feats = [torch.randn(L, 4, generator=g) for L in lengths]
    adv = torch.randn(len(lengths), generator=g)
    return w, feats, adv


def test_accumulation_equals_full_batch_reduction():
    lengths = [1, 7, 30, 3, 12, 5, 9, 2]
    w, feats, adv = _toy(lengths)
    # micro-batch 1 accumulation, fixed denominator = minibatch size
    for f, a in zip(feats, adv):
        lp = torch.log_softmax(torch.stack([f @ w, torch.zeros(len(f))], -1), -1)[:, 0]
        obj, _ = grpo.response_objective(lp, lp.detach(), float(a), 0.2, 0.28)
        (-obj / len(lengths)).backward()
    g_acc = w.grad.clone()
    # reference: padded batch, mean over tokens per response, then mean over responses
    w.grad = None
    T = max(lengths)
    mat = torch.zeros(len(lengths), T); mask = torch.zeros(len(lengths), T)
    for i, (f, a) in enumerate(zip(feats, adv)):
        lp = torch.log_softmax(torch.stack([f @ w, torch.zeros(len(f))], -1), -1)[:, 0]
        mat[i, :len(f)] = -(torch.exp(lp - lp.detach()) * a); mask[i, :len(f)] = 1
    ref = ((mat * mask).sum(1) / mask.sum(1)).mean()
    ref.backward()
    assert torch.allclose(g_acc, w.grad, atol=1e-6)


def test_matches_verl_seq_mean_token_mean():
    core = pytest.importorskip("verl.trainer.ppo.core_algos")
    import inspect
    torch.manual_seed(0)
    mat = torch.randn(32, 50); mask = torch.zeros(32, 50)
    for i in range(32):
        mask[i, :1 + (i * 7) % 50] = 1
    kw = {}
    params = inspect.signature(core.agg_loss).parameters
    if "global_batch_size" in params:
        kw.update(global_batch_size=32, dp_size=1)
    theirs = core.agg_loss(loss_mat=mat, loss_mask=mask, loss_agg_mode="seq-mean-token-mean", **kw)
    ours = sum(((mat[i] * mask[i]).sum() / mask[i].sum()) / 32 for i in range(32))
    assert theirs.item() == pytest.approx(ours.item(), rel=1e-5, abs=1e-6)


def test_minibatch_plan_balanced_and_complete():
    roles = ["hard_common"] * 8 + ["aux"] * 8
    p1 = grpo.minibatch_plan(roles, 4, np.random.default_rng(5))
    p2 = grpo.minibatch_plan(roles, 4, np.random.default_rng(5))
    assert p1 == p2 and len(p1) == 4
    assert sorted(g for mb in p1 for g in mb) == list(range(16))
    for mb in p1:
        assert sum(roles[g] == "aux" for g in mb) == 2 and len(mb) == 4
    with pytest.raises(ValueError):
        grpo.minibatch_plan(["a"] * 6 + ["b"] * 10, 4, np.random.default_rng(0))


def test_pass_at_k():
    assert grpo.pass_at_k(32, 0, 8) == 0.0
    assert grpo.pass_at_k(32, 32, 8) == 1.0
    assert grpo.pass_at_k(32, 25, 8) == 1.0               # 32-c < 8
    assert grpo.pass_at_k(32, 1, 8) == pytest.approx(8 / 32)
    assert grpo.pass_at_k(32, 4, 1) == pytest.approx(4 / 32)
