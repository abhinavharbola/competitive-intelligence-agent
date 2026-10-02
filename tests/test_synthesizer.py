import config
from agent import llm, synthesizer


def _all(status):
    return {f: status for f in config.REQUIRED_FIELDS}


def _covered(entry, fields=None):
    return [entry(field=f, result=f"body {f}") for f in (fields or config.REQUIRED_FIELDS)]


def test_synthesize_success(state, entry, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "## What It Does\nx [1]",
        "field_status": _all("confirmed"),
    })
    state["scratchpad"] = _covered(entry)
    out = synthesizer.synthesize(state)
    assert out["report"].startswith("## What It Does")
    assert out["field_status"] == _all("confirmed")
    assert out["stop_reason"] == ""


def test_synthesize_fills_missing_fields_as_insufficient(state, entry, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "r",
        "field_status": {"risks": "confirmed"},
    })
    state["scratchpad"] = _covered(entry, ["risks"])
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


def test_synthesize_failure_keeps_existing_reason_and_records_failure(state, monkeypatch):
    from agent.guardrails import stop_reasons
    def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(llm, "call_gemini", boom)
    state["stop_reason"] = "max_replans"
    out = synthesizer.synthesize(state)
    assert out["stop_reason"] == "max_replans+synthesizer_unavailable"
    assert stop_reasons(out["stop_reason"]) == {"max_replans", "synthesizer_unavailable"}


def test_synthesize_invalid_status_value_is_coerced_not_fatal(state, entry, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "r",
        "field_status": {"risks": "kinda", "competitors": "confirmed"},
    })
    state["scratchpad"] = _covered(entry)
    out = synthesizer.synthesize(state)
    assert out["stop_reason"] == ""
    assert out["report"] == "r"
    assert out["field_status"]["risks"] == "insufficient information"
    assert out["field_status"]["competitors"] == "confirmed"


def test_synthesize_extra_status_keys_are_dropped(state, entry, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "r",
        "field_status": {**_all("confirmed"), "bogus": "confirmed"},
    })
    state["scratchpad"] = _covered(entry)
    assert set(synthesizer.synthesize(state)["field_status"]) == set(config.REQUIRED_FIELDS)


def test_synthesize_downgrades_confirmed_fields_without_evidence(state, entry, monkeypatch):
    from tools.results import EMPTY_RESULT
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "r",
        "field_status": _all("confirmed"),
    })
    state["scratchpad"] = [entry(field="risks", result="real"), entry(field="competitors", result=EMPTY_RESULT)]
    out = synthesizer.synthesize(state)
    assert out["field_status"]["risks"] == "confirmed"
    assert out["field_status"]["competitors"] == "insufficient information"
    assert out["field_status"]["what_it_does"] == "insufficient information"


def test_synthesize_empty_report_uses_fallback(state, monkeypatch):
    monkeypatch.setattr(llm, "call_gemini", lambda m, s, u: {
        "report_markdown": "   ",
        "field_status": _all("confirmed"),
    })
    out = synthesizer.synthesize(state)
    assert out["stop_reason"] == "synthesizer_unavailable"
    assert out["report"].startswith("> Synthesizer unavailable")


def test_synthesizer_prompt_has_notice_and_no_em_dash():
    from tools.results import UNTRUSTED_NOTICE
    assert UNTRUSTED_NOTICE in synthesizer.SYSTEM
    assert chr(0x2014) not in synthesizer.SYSTEM


def test_fallback_report_structure(state, entry):
    state["scratchpad"] = [
        entry(field="risks", source="a", result="r1"),
        entry(field="risks", source="b", result="r2"),
    ]
    report, status = synthesizer._fallback_report(state)
    assert report.startswith("> Synthesizer unavailable")
    assert "# Acme Corp" not in report
    assert "- r1 (source: a)" in report
    assert "- r2 (source: b)" in report
    assert report.count("insufficient information") == len(config.REQUIRED_FIELDS) - 1
    assert status["risks"] == "confirmed"


def test_fallback_report_lists_fields_in_required_order(state):
    report, _ = synthesizer._fallback_report(state)
    titles = ["What It Does", "Funding & Ownership", "Recent News", "Competitors", "Risks"]
    positions = [report.index(f"## {t}") for t in titles]
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




def test_fallback_report_unwraps_and_flattens_results(state, entry):
    from tools.results import wrap_untrusted
    state["scratchpad"] = [entry(field="risks", source="s", result=wrap_untrusted("line one\n\nline two"))]
    report, _ = synthesizer._fallback_report(state)
    assert "untrusted_web_content" not in report
    assert "- line one line two (source: s)" in report


def test_fallback_report_truncates_long_results(state, entry):
    state["scratchpad"] = [entry(field="risks", source="s", result="z" * 2000)]
    report, _ = synthesizer._fallback_report(state)
    assert "z" * 500 in report and "z" * 501 not in report
