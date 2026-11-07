"""A10 cap schedule: rounds 1-64 (0-based 0..63) at 8,192, rounds 65-128 at 16,384; evaluations use the cap of the
last trained round, so the round-64 evaluation stays at 8,192 for every run."""
from qis.train.run import cap_for, eval_cap

CFG = {"max_tokens": 16384, "cap_schedule": [[0, 8192], [64, 16384]]}


def test_training_caps():
    assert [cap_for(CFG, r) for r in (0, 1, 63)] == [8192] * 3
    assert [cap_for(CFG, r) for r in (64, 65, 127)] == [16384] * 3


def test_eval_caps():
    assert {d: eval_cap(CFG, d) for d in (0, 32, 64, 96, 128)} == {0: 8192, 32: 8192, 64: 8192, 96: 16384, 128: 16384}


def test_no_schedule_falls_back():
    assert cap_for({"max_tokens": 8192}, 100) == 8192 and eval_cap({"max_tokens": 8192}, 0) == 8192
