import config
from agent import llm, synthesizer


def _all(status):
    return {f: status for f in config.REQUIRED_FIELDS}


def test_synthesize_success(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "## What It Does\nx [1]",
        "field_status": _all("confirmed"),
    })
    out = synthesizer.synthesize(state)
    assert out["report"].startswith("## What It Does")
    assert out["field_status"] == _all("confirmed")
    assert out["stop_reason"] == ""


def test_synthesize_fills_missing_fields_as_insufficient(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "r",
        "field_status": {"risks": "confirmed"},
    })
    out = synthesizer.synthesize(state)
    assert out["field_status"]["risks"] == "confirmed"
    for f in config.REQUIRED_FIELDS:
        if f != "risks":
            assert out["field_status"][f] == "insufficient information"


def test_synthesize_uses_synthesizer_model_and_scratchpad(state, entry, monkeypatch):
    captured = {}

    def fake(model, system, user):
        captured.update(model=model, user=user)
        return {"report_markdown": "r", "field_status": {}}

    monkeypatch.setattr(llm, "call_gemini", fake)
    state["scratchpad"] = [entry(field="risks", source="s1", result="body")]
    synthesizer.synthesize(state)
    assert captured["model"] == config.SYNTHESIZER_MODEL
    assert "field=risks\nsource=s1\nbody" in captured["user"]


def test_synthesize_llm_failure_uses_fallback(state, entry, monkeypatch):
    def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(llm, "call_gemini", boom)
    state["scratchpad"] = [entry(field="risks", source="s1", result="risk body")]
    out = synthesizer.synthesize(state)
    assert out["stop_reason"] == "synthesizer_unavailable"
    assert "risk body" in out["report"]
    assert out["field_status"]["risks"] == "confirmed"
    assert out["field_status"]["competitors"] == "insufficient information"


def test_synthesize_failure_preserves_existing_stop_reason(state, monkeypatch):
    def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(llm, "call_gemini", boom)
    state["stop_reason"] = "max_replans"
    assert synthesizer.synthesize(state)["stop_reason"] == "max_replans"


def test_synthesize_invalid_status_value_triggers_fallback(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "r",
        "field_status": {"risks": "kinda"},
    })
    assert synthesizer.synthesize(state)["stop_reason"] == "synthesizer_unavailable"


def test_fallback_report_structure(state, entry):
    state["scratchpad"] = [
        entry(field="risks", source="a", result="r1"),
        entry(field="risks", source="b", result="r2"),
    ]
    report, status = synthesizer._fallback_report(state)
    assert report.startswith("# Acme Corp (auto-generated, synthesizer unavailable)")
    assert "- r1 (source: a)" in report
    assert "- r2 (source: b)" in report
    assert report.count("insufficient information") == len(config.REQUIRED_FIELDS) - 1
    assert status["risks"] == "confirmed"


def test_fallback_report_lists_fields_in_required_order(state):
    report, _ = synthesizer._fallback_report(state)
    positions = [report.index(f"## {f}") for f in config.REQUIRED_FIELDS]
    assert positions == sorted(positions)


def test_fallback_report_no_results_entry_is_not_confirmed(state, entry):
    from tools.results import EMPTY_RESULT
    state["scratchpad"] = [entry(field="risks", result=EMPTY_RESULT)]
    report, status = synthesizer._fallback_report(state)
    assert status["risks"] == "insufficient information"
    assert "no results" not in report


def test_fallback_report_keeps_real_entries_next_to_empty_ones(state, entry):
    from tools.results import EMPTY_RESULT
    state["scratchpad"] = [
        entry(field="risks", result=EMPTY_RESULT, source="empty"),
        entry(field="risks", result="real risk", source="ok"),
    ]
    report, status = synthesizer._fallback_report(state)
    assert status["risks"] == "confirmed"
    assert "real risk (source: ok)" in report
    assert "empty" not in report
