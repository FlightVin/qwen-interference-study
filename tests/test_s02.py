"""Study Nov-15-2026-02: task generators, digit grading, mixture schedule."""
import pytest

from qis.s02 import tasks
from qis.train.schedule import mix_plan
from qis.verify import CORRECT, OVER_LEN, TRUNCATED, WRONG, Verifier, normalize_digits


@pytest.mark.parametrize("raw,base,want", [
    ("1234", None, "1234"), ("1,234", None, "1234"), ("1\\,234", None, "1234"), ("1{,}234", None, "1234"),
    (" 1234 ", None, "1234"), ("1234_9", 9, "1234"), ("1234_{9}", 9, "1234"), ("1234_{(9)}", 9, "1234"),
    ("1234_{\\text{9}}", 9, "1234"), ("1234 (base 9)", 9, "1234"), ("\\text{1234}", None, "1234"),
    ("$1234$", None, "1234"), ("001234", None, "1234"), ("0", None, "0"), ("(1234)_9", 9, "1234"),
    ("1234_{base9}", 9, "1234"), ("1234\\text{ (base 9)}", 9, "1234"), ("1234 \\text{base } 9", 9, "1234"),
    ("\\text{1234}_9", 9, "1234"),
    ("1234_{10}", 9, None), ("1234 base 7", 9, None), ("1234 9", None, None), ("1234 \\, 5678", None, None),
    ("12,34", None, None), ("1,2,3,4", None, None), ("12.5", None, None), ("x=3", None, None),
    ("1234_9 + 1", 9, None), ("", None, None), ("\u00b2", None, None)])
def test_normalize_digits(raw, base, want):
    assert normalize_digits(raw, base) == want


@pytest.fixture(scope="module")
def v():
    ver = Verifier(1)
    yield ver
    ver.close()


def one(v, gold, resp, n_tokens=10, truncated=False, len_limit=None, base=9):
    return v.score([{"gold": gold, "response": resp, "truncated": truncated, "n_tokens": n_tokens,
                     "match": "digits", "len_limit": len_limit, "base": base}])[0]


def test_digit_grading(v):
    assert one(v, "1234", "so \\boxed{1234_9}")["status"] == CORRECT
    assert one(v, "1234", "\\boxed{1234} no wait \\boxed{1235}")["status"] == WRONG
    assert one(v, "1234", "\\boxed{1234_{10}}")["status"] == WRONG
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
    spec = [("add9", "d4"), ("add10", "d4"), ("terse10", "k2"), ("mul10", "k2", 300)]
    a, b = tasks.build(spec, 300, 40), tasks.build(spec, 300, 40)
    assert a == b
    keys = set()
    for name, items in a["pools"].items():
        for it in items:
            key = (tasks.DEDUP_GROUP[it["task"]], tuple(it["operands"]))
            assert key not in keys            # no shared or commuted pair, also across mul10/terse10
            keys.add(key)
    ids = [it["id"] for v in a["pools"].values() for it in v]
    assert len(ids) == len(set(ids))
    assert len(a["pools"]["mul10_k2_train"]) == 300 and len(a["pools"]["terse10_k2_train"]) == 300


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


# ---- amendment A1 task families: countdown, caesar, code, choice -----------------------------------------------
from qis.verify import countdown_ok, normalize_choice, normalize_string, run_code_tests  # noqa: E402


@pytest.mark.parametrize("expr,ok", [
    ("(3 + 5) * 2", True), ("2 * (5 + 3)", True), ("(3+5) \\times 2", True), ("(3 + 5) * 2 = 16", True),
    ("16 = 2*(3+5)", True), ("\\left(3+5\\right)\\cdot 2", True), ("3 + 5 * 2", False), ("(3 + 5) * 2 = 17", False),
    ("(3 + 5) * 2 * 1", False), ("8 * 2", False), ("(3 + 5) ** 2", False), ("__import__('os')", False),
    ("(3 + 5) / 0 * 2", False), ("-(3 + 5) * -2", False), ("2.0 * (3 + 5)", False)])
def test_countdown_grader(expr, ok):
    assert countdown_ok(expr, [3, 5, 2], 16) is ok


def test_countdown_witness_and_caesar_gold():
    import random
    rng = random.Random(1)
    for n in (3, 4, 5):
        for _ in range(100):
            it = tasks.make("countdown", f"n{n}", rng)
            assert len(it["numbers"]) == n and 10 <= it["target"] <= 999
            assert countdown_ok(it["gold"], it["numbers"], it["target"])
    for _ in range(100):
        it = tasks.make("caesar", "s6", rng)
        w = it["problem"].split('"')[1]
        sh = int(it["problem"].split("shift of ")[1].split()[0])
        assert len(it["gold"]) == 6 and all((ord(b) - ord(a)) % 26 == sh for a, b in zip(w, it["gold"]))


def test_string_and_choice():
    assert normalize_string("\\text{AbC}") == "abc" and normalize_string('"abc"') == "abc"
    assert normalize_string("ab1") is None
    assert [normalize_choice(x) for x in ("B", "(b)", "\\text{C}", "D) foo", "A.", "E", "AB")] == \
        ["B", "B", "C", "D", "A", None, None]


def test_code_runner(v):
    good = "def f(x):\n    return x + 1\n"
    assert run_code_tests(good, ["assert f(1) == 2", "assert f(2) == 3"])["passed"]
    assert not run_code_tests(good, ["assert f(1) == 3"])["passed"]
    assert run_code_tests("while True:\n    pass\n", ["assert True"], timeout=3)["passed"] is False
    r = v.score([{"gold": "", "response": "Here:\n```python\n" + good + "```", "truncated": False, "n_tokens": 30,
                  "match": "code", "tests": ["assert f(4) == 5"]},
                 {"gold": "", "response": "no code", "truncated": False, "n_tokens": 3, "match": "code", "tests": []},
                 {"gold": "B", "response": "\\boxed{(B)}", "truncated": False, "n_tokens": 3, "match": "choice"},
                 {"gold": "", "response": "\\boxed{2*(3+5)}", "truncated": False, "n_tokens": 3, "match": "countdown",
                  "numbers": [3, 5, 2], "target": 16}])
    assert [x["status"] for x in r] == [CORRECT, "no_answer", CORRECT, CORRECT]


def test_extraction_fixes(v):
    from qis.verify import choice_fallback, last_code_block
    resp = "```python\ndef f(x):\n    return x + 1\n```\nExample:\n```python\nassert f(1) == 2\n```"
    assert "def f" in last_code_block(resp)
    r = v.score([{"gold": "", "response": resp, "truncated": False, "n_tokens": 30, "match": "code",
                  "tests": ["assert f(4) == 5"]},
                 {"gold": "", "response": "\\boxed{(22 + 1) + (15 - 9)} so the value is \\boxed{29}", "truncated": False,
                  "n_tokens": 9, "match": "countdown", "numbers": [22, 1, 15, 9], "target": 29},
                 {"gold": "D", "response": "### Final Answer:\n**(D) volcanic**", "truncated": False, "n_tokens": 9,
                  "match": "choice"}])
    assert [x["status"] for x in r] == [CORRECT, CORRECT, CORRECT]
    assert choice_fallback("the answer is a volcanic rock") is None
    assert choice_fallback("The answer is B.") == "B"
