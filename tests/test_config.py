import importlib

import pytest

import config


def test_required_fields_are_the_five_canonical_ones():
    assert config.REQUIRED_FIELDS == ["what_it_does", "funding_ownership", "recent_news", "competitors", "risks"]


def test_limits_match_documented_values():
    assert config.MAX_REPLAN_CYCLES == 3
    assert config.MAX_TOOL_CALLS == 15
    assert config.MAX_WALL_CLOCK_SECONDS == 480
    assert config.MEMORY_CACHE_DAYS == 7


def test_required_helper_raises_on_missing(monkeypatch):
    monkeypatch.delenv("SOME_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SOME_MISSING_KEY"):
        config._required("SOME_MISSING_KEY")


def test_required_helper_treats_empty_as_missing(monkeypatch):
    monkeypatch.setenv("EMPTY_KEY", "")
    with pytest.raises(RuntimeError):
        config._required("EMPTY_KEY")


def test_required_helper_returns_value(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "v")
    assert config._required("SOME_KEY") == "v"


def test_import_fails_without_required_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        importlib.reload(config)
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    importlib.reload(config)


def test_optional_keys_default_to_empty(monkeypatch):
    monkeypatch.delenv("GROQ_JUDGE_API_KEY")
    monkeypatch.delenv("NEON_DSN")
    monkeypatch.delenv("LOGFIRE_TOKEN")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    importlib.reload(config)
    assert config.GROQ_JUDGE_API_KEY == "" and config.NEON_DSN == "" and config.LOGFIRE_TOKEN == ""
    monkeypatch.setenv("GROQ_JUDGE_API_KEY", "test-groq-judge")
    monkeypatch.setenv("NEON_DSN", "")
    monkeypatch.setenv("LOGFIRE_TOKEN", "")
    importlib.reload(config)
