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
    assert 'class="stamp stamp-confirmed"' in text


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


class EmptyResultApp:
    def stream(self, state, stream_mode="values"):
        from tools.results import EMPTY_RESULT
        s = dict(state)
        s["plan"] = _plan()
        s["scratchpad"] = [dict(_entry(), result=EMPTY_RESULT)]
        yield dict(s)
        s["report"] = "## Risks\ninsufficient information"
        s["field_status"] = {"risks": "insufficient information"}
        yield dict(s)


class LinkedApp:
    def stream(self, state, stream_mode="values"):
        s = dict(state)
        s["plan"] = _plan()
        s["scratchpad"] = [
            dict(_entry(), result="[A](https://a.com)\nx\n\n[B](https://b.com)"),
            dict(_entry(), args="b", result="[A](https://a.com)\ny"),
        ]
        s["report"] = "## Risks\nx"
        s["field_status"] = {"risks": "confirmed"}
        yield dict(s)


def test_brief_survives_a_rerun_such_as_a_download_click(run_ui):
    at = run_ui(ApprovedApp)
    assert "Filed brief" in _all_markdown(at)
    at.run()
    assert not at.exception
    text = _all_markdown(at)
    assert "Filed brief" in text
    assert "REPORT filed" in text
    assert 'class="stamp stamp-confirmed"' in text


def test_new_run_replaces_previous_saved_run(run_ui):
    at = run_ui(ApprovedApp, entity="First Co")
    at.text_input[0].set_value("Second Co")
    at.button[0].click()
    at.run()
    text = _all_markdown(at)
    assert "Second Co" in text and "First Co" not in text


def test_failed_run_does_not_store_a_brief_for_later_reruns(run_ui):
    at = run_ui(BrokenApp)
    at.run()
    assert "Filed brief" not in _all_markdown(at)


def test_live_status_does_not_mark_empty_results_confirmed(run_ui):
    at = run_ui(EmptyResultApp)
    text = _all_markdown(at)
    assert 'class="stamp stamp-confirmed"' not in text
    assert 'class="stamp stamp-insufficient"' in text


def test_source_count_counts_distinct_urls_not_queries(run_ui):
    at = run_ui(LinkedApp)
    assert "2 sources reviewed" in _all_markdown(at)


def test_stop_reason_is_logged_once(run_ui):
    at = run_ui(ExhaustedApp)
    assert _all_markdown(at).count("STOP   max_replans") == 1
