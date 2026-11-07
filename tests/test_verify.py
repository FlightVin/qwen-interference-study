import pytest

from qis.verify import CORRECT, NO_ANSWER, TRUNCATED, UNKNOWN, WRONG, Verifier


@pytest.fixture(scope="module")
def v():
    ver = Verifier(2)
    yield ver
    ver.close()


def one(v, gold, response, truncated=False):
    return v.score([{"gold": gold, "response": response, "truncated": truncated}])[0]


def test_correct_integer(v):
    r = one(v, "2", "x+1=3 so x = 2. \\boxed{2}")
    assert (r["status"], r["reward"]) == (CORRECT, 1)


def test_only_last_boxed_counts(v):
    r = one(v, "2", "\\boxed{2} wait, actually \\boxed{3}")
    assert (r["status"], r["reward"]) == (WRONG, 0)


def test_number_in_reasoning_not_extracted(v):
    r = one(v, "5", "We get x = 5 in the first step, but the answer is \\boxed{4}")
    assert r["reward"] == 0 and r["extracted"] == "4"


def test_missing_box_is_no_answer(v):
    r = one(v, "5", "The answer is 5.")
    assert (r["status"], r["reward"]) == (NO_ANSWER, 0)


def test_equivalent_forms(v):
    assert one(v, "\\frac{1}{2}", "\\boxed{0.5}")["reward"] == 1
    assert one(v, "\\frac{1}{2}", "\\boxed{\\dfrac12}")["reward"] == 1
    assert one(v, "3, -2", "\\boxed{-2, 3}")["status"] in (CORRECT, WRONG)  # documented, not asserted


def test_truncated_correct_gets_zero(v):
    r = one(v, "2", "... so \\boxed{2", truncated=True)
    assert (r["status"], r["reward"]) == (TRUNCATED, 0)
    r = one(v, "2", "... \\boxed{2} and then the response keeps going", truncated=True)
    assert (r["status"], r["reward"]) == (TRUNCATED, 0) and r["would_match"] is True


def test_timeout_is_unknown_and_retried(v):
    res = v._run([("k", "__qis_test_sleep__", "30")], 1.0)
    assert res["k"]["ok"] is False and res["k"]["error"].startswith("timeout")
    # the pool recovers: a normal check still works after a kill
    assert one(v, "7", "\\boxed{7}")["reward"] == 1


def test_unknown_status_propagates(v, monkeypatch):
    import qis.verify as qv
    monkeypatch.setattr(qv, "TIMEOUT_S", 0.5)
    monkeypatch.setattr(qv, "RETRY_TIMEOUT_S", 0.5)
    orig = v._run

    def slow(jobs, timeout):
        return orig([(k, "__qis_test_sleep__", "5") for k, g, p in jobs], timeout)
    monkeypatch.setattr(v, "_run", slow)
    r = one(v, "2", "\\boxed{2}")
    assert r["status"] == UNKNOWN and r["reward"] is None and len(r["diagnostic"]) == 2
