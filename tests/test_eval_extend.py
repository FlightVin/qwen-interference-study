import collections
import json
import os

import pytest

from qis.eval_extend import combine, plan, select

PANEL = os.path.join(os.path.dirname(__file__), "..", "data", "stress", "stress_v1.json")


def rec(pid, j, n, finish, reward, status="correct"):
    return {"prompt_id": pid, "sample": j, "n_tokens": n, "finish": finish, "reward": reward, "status": status, "ids": [0] * 3}


def test_select_frozen_panel():
    items = json.load(open(PANEL))["pools"]["stress"]
    s = select(items)
    assert len(s) == 64 * 4 and len(set(s)) == len(s)
    tier = {it["id"]: (it["tier"], it["source_collection"]) for it in items}
    by = collections.Counter(tier[p] for p, j in s if j == 0)
    assert sum(v for (t, _), v in by.items() if t == "harder") == 32
    assert sum(v for (t, _), v in by.items() if t == "superhard") == 32
    assert by[("harder", "MATH")] == 23 and by[("harder", "Omni-MATH")] == 5 and by[("harder", "OlympiadBench")] == 4
    assert by[("superhard", "MATH")] == 15 and by[("superhard", "Omni-MATH")] == 13 and by[("superhard", "OlympiadBench")] == 4
    assert {j for _, j in s} == {0, 1, 2, 3}
    assert select(items) == s and select(list(reversed(items))) == s      # order of the panel does not matter


def test_plan_boundaries():
    cap = 12288
    recs = {("a", 0): rec("a", 0, cap, "length", 0, "truncated"),      # hit the cap: extend
            ("a", 1): rec("a", 1, cap, "stop", 1),                       # finished on the last allowed token: keep
            ("a", 2): rec("a", 2, cap - 1, "stop", 0, "wrong"),          # finished below the cap: keep
            ("a", 3): rec("a", 3, 17, "stop", 1)}
    keep, ext = plan(recs, list(recs), cap)
    assert ext == [("a", 0)] and keep == [("a", 1), ("a", 2), ("a", 3)]
    with pytest.raises(AssertionError):                                  # capped record shorter than the cap
        plan({("b", 0): rec("b", 0, cap - 5, "length", 0)}, [("b", 0)], cap)
    with pytest.raises(AssertionError):                                  # finished record longer than the cap
        plan({("b", 0): rec("b", 0, cap + 1, "stop", 1)}, [("b", 0)], cap)


def test_combine_scores_both_caps():
    cap1, cap2 = 12288, 16384
    kept = combine(rec("a", 1, 900, "stop", 1), None, None, cap2)
    assert (kept["reward_1"], kept["reward_2"], kept["new_tokens"], kept["extended"]) == (1, 1, 0, False)
    r = rec("a", 0, cap1, "length", 0, "truncated")
    ok = combine(r, {"ids": [1] * 300, "finish": "stop"}, {"reward": 1, "status": "correct", "extracted": "2"}, cap2)
    assert (ok["reward_1"], ok["reward_2"], ok["n_tokens_2"], ok["new_tokens"]) == (0, 1, cap1 + 300, 300)
    still = combine(r, {"ids": [1] * (cap2 - cap1), "finish": "length"}, {"reward": 0, "status": "truncated", "extracted": None}, cap2)
    assert (still["reward_2"], still["n_tokens_2"], still["finish_2"]) == (0, cap2, "length")
    with pytest.raises(AssertionError):
        combine(r, {"ids": [1] * (cap2 - cap1 + 1), "finish": "length"}, {"reward": 0, "status": "truncated", "extracted": None}, cap2)
