import pytest

from tools import search


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def search(self, query, max_results):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_web_search_wraps_output_in_untrusted_delimiters(monkeypatch):
    client = FakeClient([{"results": [{"title": "T", "url": "http://u", "content": "body"}]}])
    monkeypatch.setattr(search, "_client", client)
    out = search.web_search("q")
    assert out.startswith("<untrusted_web_content>\n")
    assert out.endswith("\n</untrusted_web_content>")
    assert "[T](http://u)" in out
    assert "body" in out


def test_web_search_no_results_still_wrapped(monkeypatch):
    monkeypatch.setattr(search, "_client", FakeClient([{"results": []}]))
    out = search.web_search("q")
    assert out == "<untrusted_web_content>\nno results\n</untrusted_web_content>"


def test_web_search_missing_results_key(monkeypatch):
    monkeypatch.setattr(search, "_client", FakeClient([{}]))
    assert "no results" in search.web_search("q")


def test_web_search_retries_once_then_succeeds(monkeypatch):
    client = FakeClient([RuntimeError("503"), {"results": []}])
    monkeypatch.setattr(search, "_client", client)
    search.web_search("q")
    assert client.calls == 2


def test_web_search_raises_after_retries_exhausted(monkeypatch):
    client = FakeClient([RuntimeError("a"), RuntimeError("b")])
    monkeypatch.setattr(search, "_client", client)
    with pytest.raises(RuntimeError, match="b"):
        search.web_search("q")
    assert client.calls == 2


def test_web_search_result_missing_field_raises_keyerror(monkeypatch):
    client = FakeClient([{"results": [{"title": "T", "url": "u"}]}])
    monkeypatch.setattr(search, "_client", client)
    with pytest.raises(KeyError):
        search.web_search("q")


def test_web_search_passes_max_results(monkeypatch):
    seen = {}

    class C:
        def search(self, query, max_results):
            seen["max"] = max_results
            return {"results": []}

    monkeypatch.setattr(search, "_client", C())
    search.web_search("q", max_results=3)
    assert seen["max"] == 3


