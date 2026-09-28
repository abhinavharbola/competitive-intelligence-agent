import json
from types import SimpleNamespace

import pytest

import config
from agent import llm


class StatusError(Exception):
    def __init__(self, status_code, msg="err"):
        super().__init__(msg)
        self.status_code = status_code


@pytest.mark.parametrize("exc,expected", [
    (StatusError(401), False),
    (StatusError(403), False),
    (StatusError(503), True),
    (StatusError(429), True),
    (RuntimeError("Invalid_API_Key provided"), False),
    (RuntimeError("Incorrect API key"), False),
    (RuntimeError("PERMISSION_DENIED"), False),
    (RuntimeError("Unauthorized"), False),
    (RuntimeError("timeout"), True),
    (RuntimeError("503 service unavailable"), True),
])
def test_is_retryable(exc, expected):
    assert llm._is_retryable(exc) is expected


def test_with_retries_returns_first_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert llm._with_retries(fn) == "ok"
    assert len(calls) == 1


def test_with_retries_recovers_after_transient_failures():
    outcomes = [RuntimeError("503"), RuntimeError("503"), "ok"]

    def fn():
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    assert llm._with_retries(fn) == "ok"


def test_with_retries_raises_last_error_after_max_attempts():
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError(f"fail {len(calls)}")

    with pytest.raises(RuntimeError, match="fail 3"):
        llm._with_retries(fn)
    assert len(calls) == llm._RETRY_ATTEMPTS


def test_with_retries_non_retryable_fails_immediately():
    calls = []

    def fn():
        calls.append(1)
        raise StatusError(401)

    with pytest.raises(StatusError):
        llm._with_retries(fn)
    assert len(calls) == 1


def test_with_retries_uses_exponential_backoff(monkeypatch):
    delays = []
    monkeypatch.setattr(llm.time, "sleep", lambda d: delays.append(d))

    def fn():
        raise RuntimeError("503")

    with pytest.raises(RuntimeError):
        llm._with_retries(fn)
    assert delays == [2, 4]


class FakeOpenAI:
    def __init__(self, content):
        self.captured = {}
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
                )

        self.chat = SimpleNamespace(completions=Completions())


def test_call_planner_uses_planner_model_and_parses_json(monkeypatch):
    fake = FakeOpenAI(json.dumps({"steps": []}))
    monkeypatch.setattr(llm, "_nim", fake)
    assert llm.call_planner("sys", "usr") == {"steps": []}
    assert fake.captured["model"] == config.PLANNER_MODEL
    assert fake.captured["response_format"] == {"type": "json_object"}
    assert fake.captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_call_executor_uses_executor_model(monkeypatch):
    fake = FakeOpenAI('{"query": "x"}')
    monkeypatch.setattr(llm, "_groq_executor", fake)
    assert llm.call_executor("s", "u") == {"query": "x"}
    assert fake.captured["model"] == config.EXECUTOR_MODEL


def test_call_openai_compatible_invalid_json_raises(monkeypatch):
    monkeypatch.setattr(llm, "_nim", FakeOpenAI("not json"))
    with pytest.raises(json.JSONDecodeError):
        llm.call_planner("s", "u")


def test_call_judge_requires_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "")
    with pytest.raises(RuntimeError, match="GROQ_JUDGE_API_KEY"):
        llm.call_judge("s", "u")


def test_call_judge_uses_judge_model(monkeypatch):
    fake = FakeOpenAI('{"groundedness": 4}')
    monkeypatch.setattr(llm, "_groq_judge", fake)
    monkeypatch.setattr(config, "GROQ_JUDGE_API_KEY", "k")
    assert llm.call_judge("s", "u") == {"groundedness": 4}
    assert fake.captured["model"] == config.JUDGE_MODEL


class FakeGemini:
    def __init__(self, text, usage=True):
        self.captured = {}
        outer = self
        response = SimpleNamespace(
            text=text,
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2) if usage else None,
        )

        class Models:
            def generate_content(self, **kwargs):
                outer.captured.update(kwargs)
                return response

        self.models = Models()


def test_call_gemini_passes_config_and_parses(monkeypatch):
    fake = FakeGemini('{"approved": true, "gaps": []}')
    monkeypatch.setattr(llm, "_gemini", fake)
    out = llm.call_gemini("gem-model", "sys", "usr")
    assert out == {"approved": True, "gaps": []}
    assert fake.captured["model"] == "gem-model"
    assert fake.captured["contents"] == "usr"
    assert fake.captured["config"]["system_instruction"] == "sys"
    assert fake.captured["config"]["response_mime_type"] == "application/json"


def test_call_gemini_handles_missing_usage_metadata(monkeypatch):
    monkeypatch.setattr(llm, "_gemini", FakeGemini("{}", usage=False))
    assert llm.call_gemini("m", "s", "u") == {}


def test_call_gemini_invalid_json_raises(monkeypatch):
    monkeypatch.setattr(llm, "_gemini", FakeGemini("nope"))
    with pytest.raises(json.JSONDecodeError):
        llm.call_gemini("m", "s", "u")
