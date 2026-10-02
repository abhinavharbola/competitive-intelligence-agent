import time

import pytest

import config
from agent.guardrails import wall_clock_exceeded
from agent.tracing import traced_node


def test_wall_clock_not_exceeded_for_fresh_state(state):
    assert wall_clock_exceeded(state) is False


def test_wall_clock_exceeded_past_limit(expired_state):
    assert wall_clock_exceeded(expired_state) is True


def test_wall_clock_boundary_is_exclusive(state, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    state["start_time"] = 1000.0 - config.MAX_WALL_CLOCK_SECONDS
    assert wall_clock_exceeded(state) is False


def test_traced_node_returns_wrapped_result():
    @traced_node("n")
    def fn(s):
        s["x"] = 1
        return s

    assert fn({}) == {"x": 1}


def test_traced_node_propagates_exceptions():
    @traced_node("n")
    def fn(s):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fn({})




def test_add_stop_reason_sets_first_reason(state):
    from agent.guardrails import add_stop_reason
    add_stop_reason(state, "a")
    assert state["stop_reason"] == "a"


def test_add_stop_reason_appends_and_dedupes(state):
    from agent.guardrails import add_stop_reason
    add_stop_reason(state, "a")
    add_stop_reason(state, "b")
    add_stop_reason(state, "b")
    assert state["stop_reason"] == "a+b"


def test_stop_reasons_parses_combined_value():
    from agent.guardrails import stop_reasons
    assert stop_reasons("a+b") == {"a", "b"}
    assert stop_reasons("") == set()
    assert stop_reasons(None) == set()
