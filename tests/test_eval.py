import json

import pytest

import config
from eval import judge, run_ablation
from eval.run_ablation import INFRA_FAILURE_REASONS, _avg, load_benchmark, run_condition, summarize


def _judge_raw(**over):
    raw = {"groundedness": 4, "groundedness_notes": " good ", "completeness": 5, "completeness_notes": "full"}
    raw.update(over)
    return raw


def _score(monkeypatch, raw):
    def fake(system, user):
        if isinstance(raw, Exception):
            raise raw
        return raw
    monkeypatch.setattr(judge.llm, "call_judge", fake)
    return judge.score_run("Acme", {"risks": "x"}, "report", {"risks": "confirmed"}, [])


def test_score_run_happy_path_strips_notes(monkeypatch):
    out = _score(monkeypatch, _judge_raw())
    assert out == {"groundedness": 4, "groundedness_notes": "good", "completeness": 5, "completeness_notes": "full"}


def test_score_run_call_failure_returns_all_none(monkeypatch):
    out = _score(monkeypatch, RuntimeError("down"))
    assert out == {"groundedness": None, "groundedness_notes": None, "completeness": None, "completeness_notes": None}


@pytest.mark.parametrize("bad", [None, "abc", [], {}])
def test_score_run_invalid_score_becomes_none(monkeypatch, bad):
    out = _score(monkeypatch, _judge_raw(groundedness=bad))
    assert out["groundedness"] is None
    assert out["completeness"] == 5


@pytest.mark.parametrize("value,expected", [(0, 0), (5, 5), (3, 3), (4.0, 4), ("2", 2)])
def test_score_run_accepts_boundary_and_integral_scores(monkeypatch, value, expected):
    assert _score(monkeypatch, _judge_raw(groundedness=value))["groundedness"] == expected


def test_score_run_numeric_string_is_coerced(monkeypatch):
    assert _score(monkeypatch, _judge_raw(groundedness="3"))["groundedness"] == 3


@pytest.mark.parametrize("bad", [None, "", "   ", 5])
def test_score_run_invalid_notes_become_none(monkeypatch, bad):
    out = _score(monkeypatch, _judge_raw(completeness_notes=bad))
    assert out["completeness_notes"] is None
    assert out["groundedness_notes"] == "good"


def test_score_run_missing_keys(monkeypatch):
    out = _score(monkeypatch, {})
    assert all(v is None for v in out.values())


@pytest.mark.parametrize("bad", [4.9, "4.9", 6, 9, -1, float("nan"), float("inf"), True])
def test_score_run_rejects_fractional_or_out_of_range(monkeypatch, bad):
    assert _score(monkeypatch, _judge_raw(groundedness=bad))["groundedness"] is None


def test_score_run_prompt_contains_scratchpad_and_report(monkeypatch):
    captured = {}
    monkeypatch.setattr(judge.llm, "call_judge", lambda s, u: captured.update(user=u) or _judge_raw())
    judge.score_run("Acme", {"g": 1}, "THE REPORT", {"risks": "confirmed"},
                    [{"field": "risks", "source": "SRC", "result": "RES"}])
    assert "THE REPORT" in captured["user"]
    assert "field=risks source=SRC" in captured["user"] and "RES" in captured["user"]


def test_score_run_empty_scratchpad_labelled_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(judge.llm, "call_judge", lambda s, u: captured.update(user=u) or _judge_raw())
    judge.score_run("Acme", {}, "r", {}, [])
    assert "findings used (source + content):\nnone" in captured["user"]


def test_load_benchmark_full_and_limit():
    assert len(load_benchmark()) == 15
    assert len(load_benchmark(limit=3)) == 3
    assert load_benchmark(limit=None) == load_benchmark()


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_load_benchmark_rejects_non_positive_limit(bad):
    with pytest.raises(ValueError):
        load_benchmark(limit=bad)


def test_judge_prompt_treats_ground_truth_as_snapshot_and_carries_notice():
    from tools.results import UNTRUSTED_NOTICE
    assert "dated snapshot" in judge.SYSTEM
    assert UNTRUSTED_NOTICE in judge.SYSTEM


def _res(stop="", g=4, c=5, tools=10, secs=30.0, entity="E"):
    return {"entity": entity, "groundedness": g, "completeness": c, "tool_call_count": tools,
            "elapsed_seconds": secs, "stop_reason": stop, "replan_count": 0}


def test_avg_ignores_none_and_handles_empty():
    assert _avg([{"a": 2}, {"a": None}, {"a": 4}], "a") == 3
    assert _avg([{"a": None}], "a") is None
    assert _avg([], "a") is None


def test_summarize_averages_and_counts():
    s = summarize([_res(g=4, c=4), _res(g=2, c=5)])
    assert s["avg_groundedness"] == 3 and s["avg_completeness"] == 4.5
    assert s["scored_entities"] == 2 and s["total_entities"] == 2
    assert s["excluded_infra_failures"] == []


@pytest.mark.parametrize("reason", sorted(INFRA_FAILURE_REASONS))
def test_summarize_excludes_infra_failures(reason):
    s = summarize([_res(g=4), _res(stop=reason, g=0, entity="Bad")])
    assert s["avg_groundedness"] == 4
    assert s["total_entities"] == 2 and s["scored_entities"] == 1
    assert s["excluded_infra_failures"] == [{"entity": "Bad", "stop_reason": reason}]


def test_summarize_keeps_research_limit_stops_in_averages():
    s = summarize([_res(stop="max_replans", g=2), _res(stop="max_tool_calls", g=4), _res(stop="wall_clock", g=3)])
    assert s["avg_groundedness"] == 3 and s["excluded_infra_failures"] == []


def test_summarize_all_excluded_gives_none_averages():
    s = summarize([_res(stop="planner_unavailable")])
    assert s["avg_groundedness"] is None and s["scored_entities"] == 0


def test_summarize_unscored_but_usable_run_not_counted_as_scored():
    s = summarize([_res(g=None, c=None), _res(g=4)])
    assert s["scored_entities"] == 1 and s["avg_groundedness"] == 4


def test_summarize_reports_avg_replan_count():
    s = summarize([dict(_res(), replan_count=1), dict(_res(), replan_count=3)])
    assert s["avg_replan_count"] == 2


def _state(**over):
    st = {"report": "r", "field_status": {}, "scratchpad": [], "tool_call_count": 7,
          "replan_count": 2, "stop_reason": ""}
    st.update(over)
    return st


def test_run_condition_success(monkeypatch):
    calls = []
    monkeypatch.setattr(run_ablation, "run", lambda e, critic_enabled, use_memory: calls.append((e, critic_enabled, use_memory)) or _state())
    monkeypatch.setattr(run_ablation, "score_run", lambda **kw: _judge_raw())
    bench = [{"entity": "A", "ground_truth": {}}, {"entity": "B", "ground_truth": {}}]
    out = run_condition(bench, critic_enabled=True)
    assert calls == [("A", True, False), ("B", True, False)]
    assert [r["entity"] for r in out] == ["A", "B"]
    assert out[0]["groundedness"] == 4 and out[0]["tool_call_count"] == 7
    assert out[0]["critic_enabled"] is True and out[0]["replan_count"] == 2
    assert out[0]["elapsed_seconds"] >= 0


def test_run_condition_isolates_failures(monkeypatch):
    def fake_run(e, critic_enabled, use_memory):
        if e == "A":
            raise RuntimeError("kaboom")
        return _state()

    monkeypatch.setattr(run_ablation, "run", fake_run)
    monkeypatch.setattr(run_ablation, "score_run", lambda **kw: _judge_raw())
    out = run_condition([{"entity": "A", "ground_truth": {}}, {"entity": "B", "ground_truth": {}}], critic_enabled=False)
    assert out[0]["stop_reason"] == "ablation_run_failed" and out[0]["groundedness"] is None
    assert out[1]["groundedness"] == 4 and out[1]["critic_enabled"] is False


def test_main_requires_judge_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "")
    monkeypatch.setattr("sys.argv", ["run_ablation"])
    with pytest.raises(SystemExit) as exc:
        run_ablation.main()
    assert "GROQ_JUDGE_API_KEY" in str(exc.value)


@pytest.mark.parametrize("bad", ["0", "-2", "abc"])
def test_main_rejects_invalid_limit(monkeypatch, bad):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "k")
    monkeypatch.setattr("sys.argv", ["run_ablation", "--limit", bad])
    with pytest.raises(SystemExit) as exc:
        run_ablation.main()
    assert exc.value.code == 2


def test_help_works_without_judge_key(monkeypatch, capsys):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "")
    monkeypatch.setattr("sys.argv", ["run_ablation", "--help"])
    with pytest.raises(SystemExit) as exc:
        run_ablation.main()
    assert exc.value.code == 0
    assert "--limit" in capsys.readouterr().out


def test_main_writes_results_and_delta(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "k")
    monkeypatch.setattr(run_ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("sys.argv", ["run_ablation", "--limit", "1"])

    def fake_condition(bench, critic_enabled):
        return [dict(_res(g=5 if critic_enabled else 3, c=5 if critic_enabled else 4,
                          tools=12 if critic_enabled else 6, secs=100.0 if critic_enabled else 40.0),
                     replan_count=1)]

    monkeypatch.setattr(run_ablation, "run_condition", fake_condition)
    run_ablation.main()
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["delta"] == {"avg_groundedness": 2, "avg_completeness": 1, "avg_tool_calls": 6, "avg_elapsed_seconds": 60.0}
    assert (tmp_path / "results" / "with_critic.json").exists()
    assert (tmp_path / "results" / "without_critic.json").exists()


def test_main_delta_none_when_a_side_has_no_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "k")
    monkeypatch.setattr(run_ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("sys.argv", ["run_ablation", "--limit", "1"])
    monkeypatch.setattr(run_ablation, "run_condition",
                        lambda b, critic_enabled: [_res(stop="critic_unavailable")] if critic_enabled else [_res()])
    run_ablation.main()
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["delta"]["avg_groundedness"] is None


def test_main_warns_on_unverified_entries(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "k")
    monkeypatch.setattr(run_ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("sys.argv", ["run_ablation"])
    monkeypatch.setattr(run_ablation, "load_benchmark", lambda limit=None: [{"entity": "Z", "ground_truth": {}}])
    monkeypatch.setattr(run_ablation, "run_condition", lambda b, critic_enabled: [_res()])
    run_ablation.main()
    assert "unverified" in capsys.readouterr().out




def test_summarize_treats_combined_stop_reason_with_infra_part_as_excluded():
    s = summarize([_res(g=4), _res(stop="max_replans+synthesizer_unavailable", g=1, entity="Bad")])
    assert s["avg_groundedness"] == 4
    assert [e["entity"] for e in s["excluded_infra_failures"]] == ["Bad"]


def test_summarize_restricts_to_paired_entities_and_reports_unpaired():
    s = summarize([_res(g=4, entity="A"), _res(g=0, entity="B")], only={"A"})
    assert s["avg_groundedness"] == 4
    assert s["excluded_unpaired"] == ["B"]
    assert s["total_entities"] == 2


def test_paired_entities_intersects_usable_runs():
    first = [_res(entity="A"), _res(entity="B"), _res(stop="planner_unavailable", entity="C")]
    second = [_res(entity="A"), _res(stop="critic_unavailable", entity="B"), _res(entity="C")]
    assert run_ablation.paired_entities(first, second) == {"A"}


def test_main_delta_is_computed_over_paired_entities_only(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "k")
    monkeypatch.setattr(run_ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("sys.argv", ["run_ablation"])
    monkeypatch.setattr(run_ablation, "load_benchmark", lambda limit=None: [{"entity": "A"}, {"entity": "B"}])

    def fake_condition(bench, critic_enabled):
        if critic_enabled:
            return [_res(g=4, entity="A"), _res(g=5, entity="B")]
        return [_res(g=2, entity="A"), _res(stop="planner_unavailable", g=0, entity="B")]

    monkeypatch.setattr(run_ablation, "run_condition", fake_condition)
    run_ablation.main()
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["paired_entities"] == ["A"]
    assert summary["with_critic"]["avg_groundedness"] == 4
    assert summary["delta"]["avg_groundedness"] == 2
    assert summary["with_critic"]["excluded_unpaired"] == ["B"]
