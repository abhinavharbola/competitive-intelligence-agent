from agent import llm, planner


def _steps_response(*fields):
    return {"steps": [{"sub_question": f"q {f}", "field": f, "tool": "search"} for f in fields]}


def test_plan_builds_pending_steps(state, monkeypatch):
    monkeypatch.setattr(llm, "call_planner", lambda s, u: _steps_response("risks", "competitors"))
    out = planner.plan(state)
    assert [s["field"] for s in out["plan"]] == ["risks", "competitors"]
    assert all(s["status"] == "pending" for s in out["plan"])
    assert out["stop_reason"] == ""


def test_plan_drops_unknown_fields(state, monkeypatch):
    monkeypatch.setattr(llm, "call_planner", lambda s, u: _steps_response("risks", "bogus"))
    out = planner.plan(state)
    assert [s["field"] for s in out["plan"]] == ["risks"]


def test_plan_skips_llm_when_wall_clock_exceeded(expired_state, monkeypatch):
    def boom(*a):
        raise AssertionError("must not be called")
    monkeypatch.setattr(llm, "call_planner", boom)
    out = planner.plan(expired_state)
    assert out["plan"] == []
    assert out["stop_reason"] == "wall_clock"


def test_plan_preserves_existing_stop_reason_on_wall_clock(expired_state):
    expired_state["stop_reason"] = "max_tool_calls"
    out = planner.plan(expired_state)
    assert out["stop_reason"] == "max_tool_calls"


def test_plan_llm_failure_sets_planner_unavailable(state, monkeypatch):
    def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(llm, "call_planner", boom)
    out = planner.plan(state)
    assert out["plan"] == []
    assert out["stop_reason"] == "planner_unavailable"


def test_plan_invalid_response_shape_sets_planner_unavailable(state, monkeypatch):
    monkeypatch.setattr(llm, "call_planner", lambda s, u: {"wrong": []})
    out = planner.plan(state)
    assert out["stop_reason"] == "planner_unavailable"


def test_plan_failure_keeps_existing_stop_reason(state, monkeypatch):
    state["stop_reason"] = "max_replans"

    def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(llm, "call_planner", boom)
    assert planner.plan(state)["stop_reason"] == "max_replans"


def test_plan_prompt_includes_gaps_and_prior_findings(state, entry, monkeypatch):
    captured = {}

    def fake(system, user):
        captured["user"] = user
        return _steps_response("risks")

    monkeypatch.setattr(llm, "call_planner", fake)
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["scratchpad"] = [entry(field="competitors", result="x" * 500)]
    planner.plan(state)
    assert "Acme Corp" in captured["user"]
    assert state["today"] in captured["user"]
    assert "['risks']" in captured["user"]
    assert "competitors:" in captured["user"]
    assert "x" * 200 in captured["user"] and "x" * 201 not in captured["user"]


def test_plan_prompt_omits_gap_and_prior_sections_on_first_pass(state, monkeypatch):
    captured = {}

    def fake(system, user):
        captured["user"] = user
        return _steps_response("risks")

    monkeypatch.setattr(llm, "call_planner", fake)
    planner.plan(state)
    assert "Critic flagged" not in captured["user"]
    assert "already confirmed" not in captured["user"]


def test_plan_replaces_previous_plan(state, step, monkeypatch):
    state["plan"] = [step(status="done")]
    monkeypatch.setattr(llm, "call_planner", lambda s, u: _steps_response("risks"))
    out = planner.plan(state)
    assert len(out["plan"]) == 1 and out["plan"][0]["field"] == "risks"
