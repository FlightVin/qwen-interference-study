"""Study Nov-15-2026-02: task generators, digit grading, mixture schedule."""
import pytest

from qis.s02 import tasks
from qis.train.schedule import mix_plan
from qis.verify import CORRECT, OVER_LEN, TRUNCATED, WRONG, Verifier, normalize_digits


@pytest.mark.parametrize("raw,want", [
    ("1234", "1234"), ("1,234", "1234"), ("1\\,234", "1234"), ("1{,}234", "1234"), (" 1234 ", "1234"),
    ("1234_9", "1234"), ("1234_{9}", "1234"), ("1234_{(9)}", "1234"), ("1234_{\\text{9}}", "1234"),
    ("1234 (base 9)", "1234"), ("\\text{1234}", "1234"), ("$1234$", "1234"), ("001234", "1234"), ("0", "0"),
    ("12.5", None), ("x=3", None), ("1234_9 + 1", None), ("", None)])
def test_normalize_digits(raw, want):
    assert normalize_digits(raw) == want


@pytest.fixture(scope="module")
def v():
    ver = Verifier(1)
    yield ver
    ver.close()


def one(v, gold, resp, n_tokens=10, truncated=False, len_limit=None):
    return v.score([{"gold": gold, "response": resp, "truncated": truncated, "n_tokens": n_tokens,
                     "match": "digits", "len_limit": len_limit}])[0]


def test_digit_grading(v):
    assert one(v, "1234", "so \\boxed{1234_9}")["status"] == CORRECT
    assert one(v, "1234", "\\boxed{1234} no wait \\boxed{1235}")["status"] == WRONG
    assert one(v, "1234", "\\boxed{1234}", truncated=True)["status"] == TRUNCATED
    r = one(v, "1234", "\\boxed{1234}", n_tokens=30, len_limit=24)
    assert (r["status"], r["reward"]) == (OVER_LEN, 0)
    assert one(v, "1234", "\\boxed{1234}", n_tokens=24, len_limit=24)["reward"] == 1
    assert one(v, "1234", "\\boxed{1243}", n_tokens=30, len_limit=24)["status"] == WRONG


def test_golds_are_right():
    import random
    rng = random.Random(0)
    for fam, st, base in [("add9", "d6", 9), ("add7", "d5", 7), ("add10", "d8", 10)]:
        for _ in range(200):
            it = tasks.make(fam, st, rng)
            nums = [w for w in it["problem"].replace(".", " ").split() if w.isdigit() and len(w) == int(st[1:])]
            x, y = nums[0], nums[1]
            assert all(int(c) < base for c in x + y + it["gold"])
            assert int(it["gold"], base) == int(x, base) + int(y, base)
    it = tasks.make("terse10", "k2", rng)
    assert it["len_limit"] == tasks.TERSE_LEN and it["prefix"] == tasks.TERSE_PREFIX


def test_bundle_splits_disjoint_and_deterministic():
    spec = [("add9", "d4"), ("add10", "d4"), ("terse10", "k2")]
    a, b = tasks.build(spec, 300, 40), tasks.build(spec, 300, 40)
    assert a == b
    texts = {}
    for name, items in a["pools"].items():
        split = name.rsplit("_", 1)[1]
        for it in items:
            key = (it.get("prefix", ""), it["problem"])
            assert key not in texts or texts[key] == split
            texts[key] = split
    ids = [it["id"] for v in a["pools"].values() for it in v]
    assert len(ids) == len(set(ids))


def test_mix_plan_pairs_target_stream():
    sizes = {"T": 500, "C": 500, "D": 500}
    joint = [{"role": "T", "pool": "T", "slots": 8}, {"role": "C", "pool": "C", "slots": 8}]
    alone = [{"role": "T", "pool": "T", "slots": 8}]
    other = [{"role": "T", "pool": "T", "slots": 8}, {"role": "C", "pool": "D", "slots": 8}]
    for r in range(30):
        pj, pa, po = mix_plan(joint, 7, r, sizes), mix_plan(alone, 7, r, sizes), mix_plan(other, 7, r, sizes)
        assert pj[:8] == pa == po[:8]
        assert len({i for _, p, i in pj if p == "T"}) == 8
    seen = [i for r in range(10) for _, _, i in mix_plan(alone, 7, r, {"T": 80})]
    assert sorted(seen) == list(range(80))
