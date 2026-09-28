from agent import critic, llm


def _fake(response):
    def fn(model, system, user):
        if isinstance(response, Exception):
            raise response
        return response
    return fn


def test_critique_approved(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": True, "gaps": []}))
    out = critic.critique(state)
    assert out["critique"] == {"approved": True, "gaps": []}
    assert out["replan_count"] == 0


def test_critique_gaps_increment_replan_count(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": False, "gaps": ["risks", "recent_news"]}))
    out = critic.critique(state)
    assert out["critique"] == {"approved": False, "gaps": ["risks", "recent_news"]}
    assert out["replan_count"] == 1


def test_critique_drops_unrecognized_gap_fields(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": False, "gaps": ["risks", "pricing"]}))
    out = critic.critique(state)
    assert out["critique"]["gaps"] == ["risks"]
    assert out["critique"]["approved"] is False


def test_critique_only_unrecognized_gaps_becomes_approved(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": False, "gaps": ["pricing"]}))
    out = critic.critique(state)
    assert out["critique"] == {"approved": True, "gaps": []}
    assert out["replan_count"] == 0


def test_critique_empty_gaps_forces_approval_even_if_model_says_false(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": False, "gaps": []}))
    assert critic.critique(state)["critique"]["approved"] is True


def test_critique_llm_failure_sets_critic_unavailable(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake(RuntimeError("down")))
    out = critic.critique(state)
    assert out["stop_reason"] == "critic_unavailable"
    assert out["replan_count"] == 0


def test_critique_invalid_response_sets_critic_unavailable(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"nope": 1}))
    assert critic.critique(state)["stop_reason"] == "critic_unavailable"


def test_critique_noop_when_stop_reason_already_set(state, monkeypatch):
    def boom(*a):
        raise AssertionError("must not be called")
    monkeypatch.setattr(llm, "call_gemini", boom)
    state["stop_reason"] = "max_tool_calls"
    out = critic.critique(state)
    assert out["stop_reason"] == "max_tool_calls"
    assert out["critique"] == {"approved": False, "gaps": []}


def test_critique_wall_clock_sets_stop_reason_without_llm(expired_state, monkeypatch):
    def boom(*a):
        raise AssertionError("must not be called")
    monkeypatch.setattr(llm, "call_gemini", boom)
    assert critic.critique(expired_state)["stop_reason"] == "wall_clock"


def test_critique_uses_critic_model_and_includes_scratchpad(state, entry, monkeypatch):
    import config
    captured = {}

    def fake(model, system, user):
        captured.update(model=model, user=user)
        return {"approved": True, "gaps": []}

    monkeypatch.setattr(llm, "call_gemini", fake)
    state["scratchpad"] = [entry(field="risks", source="src1", result="finding text")]
    critic.critique(state)
    assert captured["model"] == config.CRITIC_MODEL
    assert "field=risks source=src1" in captured["user"]
    assert "finding text" in captured["user"]
    assert state["today"] in captured["user"]


def test_critique_empty_scratchpad_is_labelled_empty(state, monkeypatch):
    captured = {}

    def fake(model, system, user):
        captured["user"] = user
        return {"approved": False, "gaps": ["risks"]}

    monkeypatch.setattr(llm, "call_gemini", fake)
    critic.critique(state)
    assert "Scratchpad:\nempty" in captured["user"]


def test_critique_valid_gaps_never_approved(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": True, "gaps": ["risks"]}))
    out = critic.critique(state)
    assert out["critique"] == {"approved": False, "gaps": ["risks"]}
    assert out["replan_count"] == 1


def test_critique_sets_max_replans_when_budget_reached(state, monkeypatch):
    import config
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": False, "gaps": ["risks"]}))
    state["replan_count"] = config.MAX_REPLAN_CYCLES - 1
    out = critic.critique(state)
    assert out["replan_count"] == config.MAX_REPLAN_CYCLES
    assert out["stop_reason"] == "max_replans"


def test_critique_does_not_set_stop_reason_below_budget(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": False, "gaps": ["risks"]}))
    assert critic.critique(state)["stop_reason"] == ""


def test_critique_approval_at_budget_does_not_set_stop_reason(state, monkeypatch):
    import config
    monkeypatch.setattr(llm, "call_gemini", _fake({"approved": True, "gaps": []}))
    state["replan_count"] = config.MAX_REPLAN_CYCLES - 1
    assert critic.critique(state)["stop_reason"] == ""
