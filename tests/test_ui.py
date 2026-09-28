from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent.graph as g
UI_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def _entry():
    return {"sub_question": "q", "field": "risks", "tool": "search", "args": "a", "result": "r", "source": "src"}


def _plan():
    return [{"sub_question": "q", "field": "risks", "tool": "search", "status": "done"}]


class ApprovedApp:
    def stream(self, state, stream_mode="values"):
        s = dict(state)
        s["plan"] = _plan()
        s["scratchpad"] = [_entry()]
        yield dict(s)
        s["critique"] = {"approved": True, "gaps": []}
        s["report"] = "## Risks\ncost is $5 [1]"
        s["field_status"] = {"risks": "confirmed"}
        yield dict(s)


class ExhaustedApp:
    def stream(self, state, stream_mode="values"):
        s = dict(state)
        s["plan"] = _plan()
        s["scratchpad"] = [_entry()]
        yield dict(s)
        s["critique"] = {"approved": False, "gaps": ["recent_news"]}
        s["replan_count"] = 3
        s["stop_reason"] = "max_replans"
        yield dict(s)
        s["report"] = "## Risks\nx"
        s["field_status"] = {"risks": "confirmed"}
        yield dict(s)


class BrokenApp:
    def stream(self, state, stream_mode="values"):
        raise RuntimeError("provider down")
        yield


@pytest.fixture
def run_ui(monkeypatch):
    def _run(app_cls, entity="Acme Corp"):
        monkeypatch.setattr(g, "build_graph", lambda critic_enabled=True: app_cls())
        monkeypatch.setattr(g, "save_results", lambda e, s: None)
        monkeypatch.setattr(g, "seed_from_memory", lambda s, e: (s, ""))
        at = AppTest.from_file(UI_PATH, default_timeout=30)
        at.run()
        assert not at.exception
        at.text_input[0].set_value(entity)
        at.button[0].click()
        at.run()
        return at
    return _run


def _all_markdown(at):
    return "\n".join(m.value for m in at.markdown)


def test_empty_entity_shows_warning(monkeypatch):
    at = AppTest.from_file(UI_PATH, default_timeout=30)
    at.run()
    at.button[0].click()
    at.run()
    assert [w.value for w in at.warning] == ["Enter a company or product name first."]


def test_successful_run_renders_brief_and_status(run_ui):
    at = run_ui(ApprovedApp)
    assert not at.exception and not at.error
    text = _all_markdown(at)
    assert "CRITIC all fields confirmed, approved" in text
    assert "REPORT filed" in text
    assert "Risks" in text
    assert "stamp-confirmed" in text


def test_exhausted_replans_are_reported_as_exhausted_not_replanning(run_ui):
    at = run_ui(ExhaustedApp)
    text = _all_markdown(at)
    assert "replan budget exhausted (3/3)" in text
    assert "replanning (cycle 3/3)" not in text
    assert "stop_reason: max_replans" in text


def test_provider_failure_shows_error_and_no_brief(run_ui):
    at = run_ui(BrokenApp)
    assert len(at.error) == 1
    assert "provider down" in at.error[0].value
    assert "Filed brief" not in _all_markdown(at)
