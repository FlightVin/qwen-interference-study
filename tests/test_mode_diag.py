"""A6 grading rules for thinking mode (no GPU, no tokenizer)."""
from qis import mode_diag as md
from qis.verify import Verifier

END = 151668            # stand-in id for </think>; the job checks the real id via check_templates


def dec(ids):
    table = {1: "Reasoning: so the answer is \\boxed{7}.", 2: "Final answer: \\boxed{7}", 3: "Final answer: \\boxed{9}"}
    return " ".join(table[i] for i in ids if i in table)


def test_split_final_uses_first_close():
    assert md.split_final([1, END, 2], END) == ([1], [2])
    assert md.split_final([1, 1], END) == ([1, 1], None)


def test_boxed_inside_reasoning_cannot_earn_reward():
    v = Verifier(1)
    try:
        # correct box only inside reasoning, wrong box in the final answer -> 0
        vin, fixed, _, _ = md.grade_inputs("7", [1, END, 3], "stop", True, dec, END)
        assert fixed is None and v.score([vin])[0]["reward"] == 0
        # correct box only inside reasoning and the reasoning never closes (stopped) -> 0, no_final
        vin, fixed, _, _ = md.grade_inputs("7", [1, 1], "stop", True, dec, END)
        assert vin is None and fixed == {"status": md.NO_FINAL, "reward": 0, "extracted": None}
        # control: correct box after </think> earns reward
        vin, fixed, _, _ = md.grade_inputs("7", [1, END, 2], "stop", True, dec, END)
        assert v.score([vin])[0]["reward"] == 1
    finally:
        v.close()


def test_incomplete_response_gets_truncation_treatment():
    v = Verifier(1)
    try:
        # capped inside the reasoning: existing truncation treatment (reward 0, status truncated)
        vin, fixed, n, nr = md.grade_inputs("7", [1, 1, 1], "length", True, dec, END)
        assert vin is None and fixed["status"] == "truncated" and fixed["reward"] == 0 and n == nr == 3
        # capped during the final answer: verifier's own truncation path -> 0 even with a correct box
        vin, fixed, _, _ = md.grade_inputs("7", [1, END, 2], "length", True, dec, END)
        r = v.score([vin])[0]
        assert vin["truncated"] and r["reward"] == 0 and r["status"] == "truncated"
        # non-thinking mode is graded exactly as before
        vin, fixed, n, nr = md.grade_inputs("7", [2], "stop", False, dec, END)
        assert fixed is None and nr == 0 and v.score([vin])[0]["reward"] == 1
    finally:
        v.close()
