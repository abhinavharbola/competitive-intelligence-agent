import os
import time

for _key, _value in {
    "NIM_API_KEY": "test-nim",
    "GROQ_EXECUTOR_API_KEY": "test-groq-exec",
    "GROQ_JUDGE_API_KEY": "test-groq-judge",
    "GEMINI_API_KEY": "test-gemini",
    "TAVILY_API_KEY": "test-tavily",
    "NEON_DSN": "",
    "LOGFIRE_TOKEN": "",
    "API_ACCESS_KEY": "",
}.items():
    os.environ[_key] = _value

import pytest

import config
from agent.graph import build_initial_state


@pytest.fixture
def state():
    return build_initial_state("Acme Corp")


@pytest.fixture
def step():
    def _make(field="what_it_does", tool="search", sub_question="what does acme do", status="pending"):
        return {"sub_question": sub_question, "field": field, "tool": tool, "status": status}
    return _make


@pytest.fixture
def entry():
    def _make(field="what_it_does", source="acme query", result="acme makes anvils", tool="search", args="acme query"):
        return {
            "sub_question": "q",
            "field": field,
            "tool": tool,
            "args": args,
            "result": result,
            "source": source,
        }
    return _make


@pytest.fixture
def expired_state(state):
    state["start_time"] = time.time() - config.MAX_WALL_CLOCK_SECONDS - 1
    return state


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)


