import time

import config
from agent import executor, llm


def _patch(monkeypatch, args, search=None, calc=None):
    seq = list(args) if isinstance(args, list) else None

    def fake_exec(system, user):
        if seq is not None:
            item = seq.pop(0)
        else:
            item = args
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm, "call_executor", fake_exec)
    monkeypatch.setattr(executor, "web_search", search or (lambda q: f"result for {q}"))
    monkeypatch.setattr(executor, "calculate", calc or (lambda e: "42"))


def test_execute_search_step_populates_scratchpad_and_cache(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "Acme Funding"})
    state["plan"] = [step()]
    out = executor.execute(state)
    assert out["plan"][0]["status"] == "done"
    assert out["tool_call_count"] == 1
    assert len(out["scratchpad"]) == 1
    e = out["scratchpad"][0]
    assert e["args"] == "Acme Funding"
    assert e["source"] == "Acme Funding"
    assert e["result"] == "result for Acme Funding"
    assert "search:acme funding" in out["tool_call_cache"]


def test_execute_calculator_step_uses_calculator_source(state, step, monkeypatch):
    _patch(monkeypatch, {"expression": "1+1"})
    state["plan"] = [step(tool="calculator")]
    out = executor.execute(state)
    assert out["scratchpad"][0]["source"] == "calculator: 1+1"
    assert out["scratchpad"][0]["result"] == "1+1 = 42"
    assert out["plan"][0]["status"] == "done"


def test_execute_skips_non_pending_steps(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "q"})
    state["plan"] = [step(status="done"), step(status="failed"), step(status="blocked")]
    out = executor.execute(state)
    assert out["scratchpad"] == []
    assert out["tool_call_count"] == 0


def test_execute_duplicate_query_is_blocked_but_still_recorded_for_new_field(state, step, monkeypatch):
    calls = []
    _patch(monkeypatch, {"query": "Same Query"}, search=lambda q: calls.append(q) or "res")
    state["plan"] = [step(field="risks"), step(field="competitors")]
    out = executor.execute(state)
    assert calls == ["Same Query"]
    assert out["tool_call_count"] == 1
    assert [s["status"] for s in out["plan"]] == ["done", "blocked"]
    assert [e["field"] for e in out["scratchpad"]] == ["risks", "competitors"]
    assert out["scratchpad"][1]["result"] == "res"


def test_execute_duplicate_detection_is_case_insensitive(state, step, monkeypatch):
    _patch(monkeypatch, [{"query": "Acme News"}, {"query": "acme news"}])
    state["plan"] = [step(), step(field="risks")]
    out = executor.execute(state)
    assert out["tool_call_count"] == 1
    assert out["plan"][1]["status"] == "blocked"


def test_execute_same_arg_different_tool_is_not_duplicate(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "5", "expression": "5"}, search=lambda q: "s", calc=lambda e: "c")
    state["plan"] = [step(tool="search"), step(tool="calculator")]
    out = executor.execute(state)
    assert out["tool_call_count"] == 2


def test_execute_stops_at_max_tool_calls(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "q"})
    state["tool_call_count"] = config.MAX_TOOL_CALLS
    state["plan"] = [step()]
    out = executor.execute(state)
    assert out["stop_reason"] == "max_tool_calls"
    assert out["plan"][0]["status"] == "pending"
    assert out["scratchpad"] == []


def test_execute_hits_cap_mid_plan(state, step, monkeypatch):
    counter = iter(range(100))
    _patch(monkeypatch, {"query": "x"}, search=lambda q: "r")

    def fake_exec(system, user):
        return {"query": f"q{next(counter)}"}

    monkeypatch.setattr(llm, "call_executor", fake_exec)
    state["tool_call_count"] = config.MAX_TOOL_CALLS - 1
    state["plan"] = [step(), step()]
    out = executor.execute(state)
    assert out["tool_call_count"] == config.MAX_TOOL_CALLS
    assert out["stop_reason"] == "max_tool_calls"
    assert [s["status"] for s in out["plan"]] == ["done", "pending"]


def test_execute_stops_on_wall_clock(expired_state, step, monkeypatch):
    _patch(monkeypatch, {"query": "q"})
    expired_state["plan"] = [step()]
    out = executor.execute(expired_state)
    assert out["stop_reason"] == "wall_clock"
    assert out["plan"][0]["status"] == "pending"


def test_execute_llm_failure_marks_step_failed_and_continues(state, step, monkeypatch):
    _patch(monkeypatch, [RuntimeError("down"), {"query": "ok"}])
    state["plan"] = [step(), step(field="risks")]
    out = executor.execute(state)
    assert [s["status"] for s in out["plan"]] == ["failed", "done"]
    assert out["tool_call_count"] == 1


def test_execute_invalid_args_shape_marks_failed(state, step, monkeypatch):
    _patch(monkeypatch, {"query": ["not", "a", "string"]})
    state["plan"] = [step()]
    assert executor.execute(state)["plan"][0]["status"] == "failed"


def test_execute_empty_arg_marks_failed(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "   "})
    state["plan"] = [step()]
    out = executor.execute(state)
    assert out["plan"][0]["status"] == "failed"
    assert out["tool_call_count"] == 0


def test_execute_no_args_at_all_marks_failed(state, step, monkeypatch):
    _patch(monkeypatch, {})
    state["plan"] = [step()]
    assert executor.execute(state)["plan"][0]["status"] == "failed"


def test_execute_tool_exception_marks_failed_and_not_counted(state, step, monkeypatch):
    def boom(q):
        raise RuntimeError("tavily down")
    _patch(monkeypatch, {"query": "q"}, search=boom)
    state["plan"] = [step()]
    out = executor.execute(state)
    assert out["plan"][0]["status"] == "failed"
    assert out["tool_call_count"] == 0
    assert out["tool_call_cache"] == {}
    assert out["scratchpad"] == []


def test_execute_failed_step_is_not_marked_blocked(state, step, monkeypatch):
    def boom(q):
        raise RuntimeError("x")
    _patch(monkeypatch, {"query": "q"}, search=boom)
    state["plan"] = [step()]
    assert executor.execute(state)["plan"][0]["status"] != "blocked"


def test_execute_calculator_error_marks_failed(state, step, monkeypatch):
    def boom(e):
        raise ZeroDivisionError()
    _patch(monkeypatch, {"expression": "1/0"}, calc=boom)
    state["plan"] = [step(tool="calculator")]
    assert executor.execute(state)["plan"][0]["status"] == "failed"


def test_execute_strips_whitespace_from_args(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "  spaced  "})
    state["plan"] = [step()]
    assert executor.execute(state)["scratchpad"][0]["args"] == "spaced"


def test_execute_prompt_includes_date_question_and_tool(state, step, monkeypatch):
    captured = {}

    def fake(system, user):
        captured["user"] = user
        return {"query": "q"}

    monkeypatch.setattr(llm, "call_executor", fake)
    monkeypatch.setattr(executor, "web_search", lambda q: "r")
    state["plan"] = [step(sub_question="the question")]
    executor.execute(state)
    assert state["today"] in captured["user"]
    assert "the question" in captured["user"]
    assert "Tool: search" in captured["user"]




def test_execute_same_field_duplicate_is_not_recorded_twice(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "Same"}, search=lambda q: "res")
    state["plan"] = [step(field="risks"), step(field="risks", sub_question="again")]
    out = executor.execute(state)
    assert [s["status"] for s in out["plan"]] == ["done", "blocked"]
    assert len(out["scratchpad"]) == 1
    assert out["tool_call_count"] == 1


def test_execute_uses_expression_key_for_calculator_and_query_key_for_search(state, step, monkeypatch):
    _patch(monkeypatch, {"query": "only a query"}, calc=lambda e: "1")
    state["plan"] = [step(tool="calculator")]
    assert executor.execute(state)["plan"][0]["status"] == "failed"

    _patch(monkeypatch, {"expression": "1+1"})
    state["plan"] = [step(tool="search")]
    assert executor.execute(state)["plan"][0]["status"] == "failed"


def test_execute_prompt_lists_already_run_queries(state, step, entry, monkeypatch):
    captured = {}

    def fake(system, user):
        captured["user"] = user
        return {"query": "fresh"}

    monkeypatch.setattr(llm, "call_executor", fake)
    monkeypatch.setattr(executor, "web_search", lambda q: "r")
    state["scratchpad"] = [entry(args="old query one"), entry(args="old query two")]
    state["plan"] = [step()]
    executor.execute(state)
    assert "Already-run queries, do not repeat: old query one; old query two" in captured["user"]


def test_execute_calculator_prompt_includes_findings_but_search_prompt_does_not(state, step, entry, monkeypatch):
    captured = []

    def fake(system, user):
        captured.append(user)
        return {"expression": "1+1", "query": "q"}

    monkeypatch.setattr(llm, "call_executor", fake)
    monkeypatch.setattr(executor, "web_search", lambda q: "r")
    monkeypatch.setattr(executor, "calculate", lambda e: "2")
    state["scratchpad"] = [entry(result="revenue was 100")]
    state["plan"] = [step(tool="calculator"), step(tool="search")]
    executor.execute(state)
    assert "Findings available for the calculation" in captured[0] and "revenue was 100" in captured[0]
    assert "Findings available for the calculation" not in captured[1]


def test_execute_stops_on_wall_clock_after_args_generated(state, step, monkeypatch):
    import time
    import config

    def fake(system, user):
        state["start_time"] = time.time() - config.MAX_WALL_CLOCK_SECONDS - 1
        return {"query": "q"}

    monkeypatch.setattr(llm, "call_executor", fake)
    called = []
    monkeypatch.setattr(executor, "web_search", lambda q: called.append(q) or "r")
    state["plan"] = [step()]
    out = executor.execute(state)
    assert out["stop_reason"] == "wall_clock"
    assert called == []


def test_executor_prompt_carries_untrusted_notice():
    from tools.results import UNTRUSTED_NOTICE
    assert UNTRUSTED_NOTICE in executor.SYSTEM
