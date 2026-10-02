import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from agent.graph import build_initial_state


@pytest.fixture
def client():
    return TestClient(api_main.app)


def _final(**over):
    s = build_initial_state("Acme")
    s.update(report="rep", field_status={"risks": "confirmed"}, stop_reason="", replan_count=1,
             tool_call_count=4, scratchpad=[{"field": "risks"}], memory_note="")
    s.update(over)
    return s


def test_research_success(client, monkeypatch):
    monkeypatch.setattr(api_main, "run", lambda entity: _final())
    r = client.post("/research", json={"entity": "Acme"})
    assert r.status_code == 200
    body = r.json()
    assert body["entity"] == "Acme"
    assert body["report"] == "rep"
    assert body["field_status"] == {"risks": "confirmed"}
    assert body["replan_count"] == 1 and body["tool_call_count"] == 4
    assert body["scratchpad"] == [{"field": "risks"}]
    assert body["stop_reason"] == "" and body["memory_note"] == ""


@pytest.mark.parametrize("entity", ["", "   ", "\t\n"])
def test_research_blank_entity_returns_400(client, monkeypatch, entity):
    def boom(e):
        raise AssertionError("must not run")
    monkeypatch.setattr(api_main, "run", boom)
    r = client.post("/research", json={"entity": entity})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_research_missing_entity_returns_422(client):
    assert client.post("/research", json={}).status_code == 422


def test_research_wrong_type_returns_422(client):
    assert client.post("/research", json={"entity": 123}).status_code == 422


def test_research_internal_error_returns_500_without_leaking_details(client, monkeypatch):
    def boom(e):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(api_main, "run", boom)
    r = client.post("/research", json={"entity": "Acme"})
    assert r.status_code == 500
    assert "secret" not in r.text


def test_research_strips_entity_before_running_and_echoing(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(api_main, "run", lambda entity: seen.update(e=entity) or _final())
    body = client.post("/research", json={"entity": "  Acme  "}).json()
    assert seen["e"] == "Acme"
    assert body["entity"] == "Acme"


def test_research_passes_entity_to_run(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(api_main, "run", lambda entity: seen.update(e=entity) or _final())
    client.post("/research", json={"entity": "Stripe"})
    assert seen["e"] == "Stripe"


def test_research_overlong_entity_returns_422(client, monkeypatch):
    def boom(e):
        raise AssertionError("must not run")
    monkeypatch.setattr(api_main, "run", boom)
    r = client.post("/research", json={"entity": "a" * (api_main.MAX_ENTITY_LENGTH + 1)})
    assert r.status_code == 422


def test_research_entity_at_length_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(api_main, "run", lambda entity: _final())
    r = client.post("/research", json={"entity": "a" * api_main.MAX_ENTITY_LENGTH})
    assert r.status_code == 200


def test_auth_disabled_when_no_key_configured(client, monkeypatch):
    monkeypatch.setattr(api_main.config, "API_ACCESS_KEY", "")
    monkeypatch.setattr(api_main, "run", lambda entity: _final())
    assert client.post("/research", json={"entity": "Acme"}).status_code == 200


def test_auth_rejects_missing_and_wrong_key(client, monkeypatch):
    monkeypatch.setattr(api_main.config, "API_ACCESS_KEY", "secret")

    def boom(e):
        raise AssertionError("must not run")

    monkeypatch.setattr(api_main, "run", boom)
    assert client.post("/research", json={"entity": "Acme"}).status_code == 401
    r = client.post("/research", json={"entity": "Acme"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_auth_accepts_correct_key(client, monkeypatch):
    monkeypatch.setattr(api_main.config, "API_ACCESS_KEY", "secret")
    monkeypatch.setattr(api_main, "run", lambda entity: _final())
    r = client.post("/research", json={"entity": "Acme"}, headers={"X-API-Key": "secret"})
    assert r.status_code == 200


def test_get_not_allowed(client):
    assert client.get("/research").status_code == 405


