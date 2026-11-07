import torch

from qis import diag_local as dl


def pools():
    return {"hard": [{"id": f"h{i}"} for i in range(46)],
            "related": [{"id": f"r{i}"} for i in range(63)], "unrelated": [{"id": f"u{i}"} for i in range(63)]}


def test_prompt_draws_deterministic_and_paired():
    P = pools()
    for s in dl.MB_SEEDS:
        r, ri = dl.branch_prompts(P, "related", s)
        u, ui = dl.branch_prompts(P, "unrelated", s)
        h, hi = dl.branch_prompts(P, "hard", s)
        assert ri == ui and len(r) == len(u) == len(h) == 4 and len(set(hi)) == 4
        assert [x["id"][1:] for x in r] == [x["id"][1:] for x in u]          # matched pairs
        assert dl.branch_prompts(P, "hard", s) == (h, hi)                      # deterministic
    assert len({tuple(dl.branch_prompts(P, "hard", s)[1]) for s in dl.MB_SEEDS}) > 1   # seeds differ


def _step(model, opt):
    loss = model(torch.ones(2, 3)).pow(2).sum(); loss.backward(); opt.step(); opt.zero_grad()


def test_snapshot_restore_model_and_optimizer():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2); opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _step(model, opt)                                   # source with non-empty moments
    snap = dl.snapshot(model, opt)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    ref_m = [opt.state[p]["exp_avg"].clone() for p in model.parameters()]
    for _ in range(2):                                  # two branches from the same source
        _step(model, opt)
        dl.restore(model, opt, snap)
        assert all(torch.equal(model.state_dict()[k], ref[k]) for k in ref)
        assert all(torch.equal(opt.state[p]["exp_avg"], m) for p, m in zip(model.parameters(), ref_m))


def test_restore_fresh_optimizer_clears_state():
    model = torch.nn.Linear(3, 2); opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    snap = dl.snapshot(model, opt)                      # base source: fresh optimizer
    _step(model, opt)
    dl.restore(model, opt, snap)
    assert len(opt.state) == 0
