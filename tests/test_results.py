import pytest

from tools.results import EMPTY_RESULT, NO_RESULTS, is_empty_result, wrap_untrusted


def test_wrap_untrusted_format():
    assert wrap_untrusted("x") == "<untrusted_web_content>\nx\n</untrusted_web_content>"


def test_empty_result_is_wrapped_no_results():
    assert EMPTY_RESULT == wrap_untrusted(NO_RESULTS)


def test_is_empty_result_true_for_empty_marker():
    assert is_empty_result(EMPTY_RESULT)
    assert is_empty_result(f"  {EMPTY_RESULT}\n")


@pytest.mark.parametrize("value", ["", "no results", "42", wrap_untrusted("real content"), wrap_untrusted("no results found for acme")])
def test_is_empty_result_false_for_other_content(value):
    assert not is_empty_result(value)


def test_web_search_output_with_no_hits_is_detected_as_empty(monkeypatch):
    from tools import search

    class C:
        def search(self, query, max_results):
            return {"results": []}

    monkeypatch.setattr(search, "_client", C())
    assert is_empty_result(search.web_search("q"))
