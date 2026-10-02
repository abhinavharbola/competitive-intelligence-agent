import config
import agent.graph as g
from agent.graph import (
    build_graph,
    build_initial_state,
    prepare_state,
    route_after_critic,
    run,
    save_results,
    seed_from_memory,
)


def test_build_initial_state_defaults():
    s = build_initial_state("Stripe")
    assert s["entity"] == "Stripe"
    assert len(s["today"]) == 10 and s["today"][4] == "-"
    assert s["plan"] == [] and s["scratchpad"] == []
    assert s["critique"] == {"approved": False, "gaps": []}
    assert s["replan_count"] == 0 and s["tool_call_count"] == 0
    assert s["tool_call_cache"] == {}
    assert s["stop_reason"] == "" and s["memory_note"] == ""
    assert s["report"] == "" and s["field_status"] == {}


def test_build_initial_state_returns_independent_containers():
    a, b = build_initial_state("A"), build_initial_state("B")
    a["scratchpad"].append(1)
    a["tool_call_cache"]["k"] = 1
    assert b["scratchpad"] == [] and b["tool_call_cache"] == {}


def test_route_approved_goes_to_synthesizer(state):
    state["critique"] = {"approved": True, "gaps": []}
    assert route_after_critic(state) == "synthesizer"


def test_route_stop_reason_goes_to_synthesizer(state):
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["stop_reason"] = "critic_unavailable"
    assert route_after_critic(state) == "synthesizer"


def test_route_gaps_below_limit_replans(state):
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["replan_count"] = config.MAX_REPLAN_CYCLES - 1
    assert route_after_critic(state) == "planner"


def test_route_gaps_at_limit_still_replans_until_critic_sets_stop_reason(state):
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["replan_count"] = config.MAX_REPLAN_CYCLES
    assert route_after_critic(state) == "planner"


def test_route_max_replans_stop_reason_goes_to_synthesizer(state):
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["replan_count"] = config.MAX_REPLAN_CYCLES
    state["stop_reason"] = "max_replans"
    assert route_after_critic(state) == "synthesizer"


def test_route_count_beyond_limit_goes_to_synthesizer(state):
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["replan_count"] = config.MAX_REPLAN_CYCLES + 1
    assert route_after_critic(state) == "synthesizer"


def _patch_nodes(monkeypatch, critic_gaps_forever=False):
    calls = {"planner": 0, "executor": 0, "critic": 0, "synthesizer": 0}

    def planner(s):
        calls["planner"] += 1
        return s

    def executor(s):
        calls["executor"] += 1
        return s

    def critic(s):
        calls["critic"] += 1
        if critic_gaps_forever:
            s["critique"] = {"approved": False, "gaps": ["risks"]}
            if s["replan_count"] >= config.MAX_REPLAN_CYCLES:
                s["stop_reason"] = "max_replans"
            else:
                s["replan_count"] += 1
        else:
            s["critique"] = {"approved": True, "gaps": []}
        return s

    def synth(s):
        calls["synthesizer"] += 1
        s["report"] = "final"
        return s

    monkeypatch.setattr(g, "plan", planner)
    monkeypatch.setattr(g, "execute", executor)
    monkeypatch.setattr(g, "critique", critic)
    monkeypatch.setattr(g, "synthesize", synth)
    return calls


def test_graph_with_critic_approved_runs_each_node_once(monkeypatch):
    calls = _patch_nodes(monkeypatch)
    out = build_graph(critic_enabled=True).invoke(build_initial_state("X"))
    assert calls == {"planner": 1, "executor": 1, "critic": 1, "synthesizer": 1}
    assert out["report"] == "final"


def test_graph_without_critic_has_no_critic_node(monkeypatch):
    calls = _patch_nodes(monkeypatch)
    compiled = build_graph(critic_enabled=False)
    assert "critic" not in compiled.get_graph().nodes
    compiled.invoke(build_initial_state("X"))
    assert calls["critic"] == 0
    assert calls["synthesizer"] == 1


def test_graph_with_critic_has_critic_node():
    assert "critic" in build_graph(critic_enabled=True).get_graph().nodes


def test_graph_replans_until_max_cycles_then_synthesizes(monkeypatch):
    calls = _patch_nodes(monkeypatch, critic_gaps_forever=True)
    out = build_graph().invoke(build_initial_state("X"))
    assert calls["planner"] == config.MAX_REPLAN_CYCLES + 1
    assert calls["critic"] == config.MAX_REPLAN_CYCLES + 1
    assert calls["synthesizer"] == 1
    assert out["replan_count"] == config.MAX_REPLAN_CYCLES


def test_graph_max_replans_stop_reason_survives_to_final_state(monkeypatch):
    from agent import critic as real_critic, llm

    calls = _patch_nodes(monkeypatch)
    monkeypatch.setattr(g, "critique", real_critic.critique)
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {"approved": False, "gaps": ["risks"]})
    out = build_graph().invoke(build_initial_state("X"))
    assert out["stop_reason"] == "max_replans"
    assert out["replan_count"] == config.MAX_REPLAN_CYCLES
    assert calls["planner"] == config.MAX_REPLAN_CYCLES + 1
    assert calls["synthesizer"] == 1


def test_route_does_not_mutate_state(state):
    state["critique"] = {"approved": False, "gaps": ["risks"]}
    state["replan_count"] = config.MAX_REPLAN_CYCLES
    route_after_critic(state)
    assert state["stop_reason"] == ""


def prior(**over):
    base = {
        "exact_match": True,
        "findings": {"what_it_does": "does x", "recent_news": "old news", "risks": "r"},
        "sources": {"what_it_does": ["s1", "s2"]},
        "ages": {"what_it_does": 2.4, "recent_news": 2.4, "risks": 2.4},
    }
    base.update(over)
    return base


def test_seed_no_prior_returns_unchanged(state, monkeypatch):
    monkeypatch.setattr(g, "find_prior_research", lambda e: None)
    out, note = seed_from_memory(state, "Acme")
    assert out["scratchpad"] == [] and note == ""


def test_seed_fuzzy_match_only_notes(state, monkeypatch):
    monkeypatch.setattr(g, "find_prior_research", lambda e: {
        "exact_match": False, "fuzzy_candidate": "meta financial", "score": 90,
    })
    out, note = seed_from_memory(state, "Meta")
    assert out["scratchpad"] == []
    assert "meta financial" in note and "90" in note and "not auto-used" in note


def test_seed_stale_fields_ignored(state, monkeypatch):
    stale = config.MEMORY_CACHE_DAYS + 0.5
    monkeypatch.setattr(g, "find_prior_research", lambda e: prior(ages={"what_it_does": stale, "risks": stale}))
    out, note = seed_from_memory(state, "Acme")
    assert out["scratchpad"] == [] and note == ""


def test_seed_boundary_age_is_still_fresh(state, monkeypatch):
    fresh = float(config.MEMORY_CACHE_DAYS)
    monkeypatch.setattr(g, "find_prior_research", lambda e: prior(ages={"what_it_does": fresh, "risks": fresh}))
    out, _ = seed_from_memory(state, "Acme")
    assert len(out["scratchpad"]) == 2


def test_seed_applies_freshness_per_field(state, monkeypatch):
    ages = {"what_it_does": 1.0, "risks": config.MEMORY_CACHE_DAYS + 1}
    monkeypatch.setattr(g, "find_prior_research", lambda e: prior(ages=ages))
    out, _ = seed_from_memory(state, "Acme")
    assert [e["field"] for e in out["scratchpad"]] == ["what_it_does"]


def test_seed_skips_fields_without_an_age_or_unknown_fields(state, monkeypatch):
    findings = {"what_it_does": "x", "bogus": "y"}
    monkeypatch.setattr(g, "find_prior_research", lambda e: prior(findings=findings, ages={"bogus": 1.0}))
    out, _ = seed_from_memory(state, "Acme")
    assert out["scratchpad"] == []


def test_seed_skips_recent_news_and_records_provenance(state, monkeypatch):
    monkeypatch.setattr(g, "find_prior_research", lambda e: prior())
    out, note = seed_from_memory(state, "Acme")
    fields = [e["field"] for e in out["scratchpad"]]
    assert "recent_news" not in fields
    assert set(fields) == {"what_it_does", "risks"}
    by_field = {e["field"]: e for e in out["scratchpad"]}
    assert by_field["what_it_does"]["source"] == "cache, 2d old, originally: s1; s2"
    assert by_field["risks"]["source"] == "cache, 2d old, originally: unrecorded"
    assert all(e["tool"] == "memory" for e in out["scratchpad"])
    assert note == ""


def test_prepare_state_seeds_and_sets_memory_note(monkeypatch):
    monkeypatch.setattr(g, "seed_from_memory", lambda s, e: (s, "a note"))
    assert prepare_state("Acme")["memory_note"] == "a note"


def test_prepare_state_without_memory_never_touches_it(monkeypatch):
    def boom(*a):
        raise AssertionError("memory must not be used")
    monkeypatch.setattr(g, "seed_from_memory", boom)
    out = prepare_state("Acme", use_memory=False)
    assert out["memory_note"] == "" and out["scratchpad"] == []


def _final(scratchpad, status):
    s = build_initial_state("Acme")
    s["scratchpad"] = scratchpad
    s["field_status"] = status
    return s


def test_save_results_only_confirmed_fields(monkeypatch, entry):
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(entity=e, f=f, s=s))
    final = _final(
        [entry(field="risks", result="r", source="a"), entry(field="competitors", result="c", source="b")],
        {"risks": "confirmed", "competitors": "insufficient information"},
    )
    save_results("Acme", final)
    assert saved["f"] == {"risks": "r"}
    assert saved["s"] == {"risks": ["a"]}


def test_save_results_merges_multiple_entries_and_dedupes_sources(monkeypatch, entry):
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f, s=s))
    final = _final(
        [entry(field="risks", result="r1", source="b"), entry(field="risks", result="r2", source="a"),
         entry(field="risks", result="r3", source="a")],
        {"risks": "confirmed"},
    )
    save_results("Acme", final)
    assert saved["f"]["risks"] == "r1\n\nr2\n\nr3"
    assert saved["s"]["risks"] == ["a", "b"]


def test_save_results_ignores_empty_search_entries(monkeypatch, entry):
    from tools.results import EMPTY_RESULT
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f, s=s))
    final = _final(
        [entry(field="risks", result=EMPTY_RESULT, source="empty"), entry(field="risks", result="real", source="ok")],
        {"risks": "confirmed"},
    )
    save_results("Acme", final)
    assert saved["f"] == {"risks": "real"}
    assert saved["s"] == {"risks": ["ok"]}


def test_save_results_skips_field_whose_entries_are_all_empty(monkeypatch, entry):
    from tools.results import EMPTY_RESULT
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f))
    save_results("Acme", _final([entry(field="risks", result=EMPTY_RESULT)], {"risks": "confirmed"}))
    assert saved == {}


def test_save_results_skips_confirmed_field_with_no_entries(monkeypatch):
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f))
    save_results("Acme", _final([], {"risks": "confirmed"}))
    assert saved == {}


def test_save_results_ignores_non_required_fields(monkeypatch, entry):
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f))
    save_results("Acme", _final([entry(field="bogus")], {"bogus": "confirmed"}))
    assert saved == {}


def test_save_results_never_resaves_cached_memory_entries(monkeypatch, entry):
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f, s=s))
    final = _final(
        [entry(field="risks", tool="memory", result="cached", source="cache, 6d old, originally: q0"),
         entry(field="risks", result="fresh", source="q1"),
         entry(field="competitors", tool="memory", result="cached c", source="cache")],
        {"risks": "confirmed", "competitors": "confirmed"},
    )
    save_results("Acme", final)
    assert saved["f"] == {"risks": "fresh"}
    assert saved["s"] == {"risks": ["q1"]}


def test_save_results_skips_entirely_when_only_cached_fields_exist(monkeypatch, entry):
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f))
    final = _final([entry(field="risks", tool="memory", result="cached")], {"risks": "confirmed"})
    save_results("Acme", final)
    assert saved == {}


def test_save_results_skips_fallback_reports(monkeypatch, entry):
    saved = {}
    monkeypatch.setattr(g, "save_research", lambda e, f, s: saved.update(f=f))
    final = _final([entry(field="risks", result="r")], {"risks": "confirmed"})
    final["stop_reason"] = "max_replans+synthesizer_unavailable"
    save_results("Acme", final)
    assert saved == {}


def test_run_with_memory_seeds_saves_and_sets_note(monkeypatch):
    class FakeApp:
        def invoke(self, s):
            s["field_status"] = {}
            return s

    seen = {}
    monkeypatch.setattr(g, "build_graph", lambda critic_enabled=True: seen.update(critic=critic_enabled) or FakeApp())
    monkeypatch.setattr(g, "seed_from_memory", lambda s, e: (s, "a note"))
    saves = []
    monkeypatch.setattr(g, "save_results", lambda e, s: saves.append(e))
    out = run("Acme", critic_enabled=False, use_memory=True)
    assert out["memory_note"] == "a note"
    assert saves == ["Acme"]
    assert seen["critic"] is False


def test_run_without_memory_never_touches_memory(monkeypatch):
    class FakeApp:
        def invoke(self, s):
            return s

    monkeypatch.setattr(g, "build_graph", lambda critic_enabled=True: FakeApp())

    def boom(*a):
        raise AssertionError("memory must not be used")

    monkeypatch.setattr(g, "seed_from_memory", boom)
    monkeypatch.setattr(g, "save_results", boom)
    out = run("Acme", use_memory=False)
    assert out["memory_note"] == ""


