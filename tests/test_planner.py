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




def test_plan_excludes_empty_results_and_gap_fields_from_confirmed_summary(state, entry, monkeypatch):
    from tools.results import EMPTY_RESULT
    captured = {}

    def fake(system, user):
        captured["user"] = user
        return _steps_response("risks")

    monkeypatch.setattr(llm, "call_planner", fake)
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["scratchpad"] = [
        entry(field="risks", result="old risk text"),
        entry(field="competitors", result=EMPTY_RESULT),
        entry(field="what_it_does", result="does things"),
    ]
    planner.plan(state)
    assert "what_it_does:" in captured["user"]
    assert "old risk text" not in captured["user"]
    assert "competitors:" not in captured["user"]


def test_plan_lists_seeded_fields_as_confirmed_on_first_pass(state, entry, monkeypatch):
    captured = {}

    def fake(system, user):
        captured["user"] = user
        return _steps_response("recent_news")

    monkeypatch.setattr(llm, "call_planner", fake)
    state["scratchpad"] = [entry(field="risks", tool="memory", result="cached risk")]
    planner.plan(state)
    assert "already confirmed" in captured["user"]
    assert "risks: " in captured["user"]


def test_plan_confirmed_summary_is_rewrapped_after_truncation(state, entry, monkeypatch):
    from tools.results import wrap_untrusted
    captured = {}

    def fake(system, user):
        captured["user"] = user
        return _steps_response("risks")

    monkeypatch.setattr(llm, "call_planner", fake)
    state["scratchpad"] = [entry(field="competitors", result=wrap_untrusted("y" * 500))]
    planner.plan(state)
    assert captured["user"].count("<untrusted_web_content>") == captured["user"].count("</untrusted_web_content>") == 1


def test_plan_is_capped_at_max_plan_steps(state, monkeypatch):
    import config
    many = _steps_response(*(["risks"] * (config.MAX_PLAN_STEPS + 5)))
    monkeypatch.setattr(llm, "call_planner", lambda s, u: many)
    out = planner.plan(state)
    assert len(out["plan"]) == config.MAX_PLAN_STEPS


def test_planner_prompt_carries_untrusted_notice_and_step_cap():
    import config
    from tools.results import UNTRUSTED_NOTICE
    assert UNTRUSTED_NOTICE in planner.SYSTEM
    assert str(config.MAX_PLAN_STEPS) in planner.SYSTEM
