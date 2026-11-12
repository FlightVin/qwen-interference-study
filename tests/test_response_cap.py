"""One response cap for all rounds: config max_tokens, default MAX_TOKENS (12,288); cap schedules are rejected."""
import pytest

from qis.limits import MAX_TOKENS
from qis.train.run import response_cap


def test_default_is_12k():
    assert MAX_TOKENS == 12288
    assert response_cap({}) == 12288


def test_explicit_cap():
    assert response_cap({"max_tokens": 4096}) == 4096


def test_schedule_rejected():
    with pytest.raises(ValueError):
        response_cap({"max_tokens": 12288, "cap_schedule": [[0, 8192], [32, 12288]]})
