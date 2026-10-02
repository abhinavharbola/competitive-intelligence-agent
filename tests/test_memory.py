from datetime import datetime, timedelta, timezone

import pytest

import config
from tools import memory


class FakeCursor:
    def __init__(self, fetchone_rows, fetchall_rows, executed):
        self.fetchone_rows = list(fetchone_rows)
        self.fetchall_rows = list(fetchall_rows)
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_rows.pop(0) if self.fetchone_rows else None

    def fetchall(self):
        return self.fetchall_rows.pop(0) if self.fetchall_rows else []


class FakeConn:
    def __init__(self, fetchone_rows=(), fetchall_rows=()):
        self.executed = []
        self.committed = False
        self._cur = FakeCursor(fetchone_rows, fetchall_rows, self.executed)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


@pytest.fixture
def dsn(monkeypatch):
    monkeypatch.setattr(config, "NEON_DSN", "postgres://fake")


@pytest.mark.parametrize("raw,expected", [
    ("Stripe", "stripe"),
    ("Stripe, Inc.", "stripe"),
    ("  ACME  Corp  ", "acme"),
    ("Foo Bar Ltd", "foo bar"),
    ("Foo LLC", "foo"),
    ("Big Co Inc", "big"),
    ("Siemens GmbH", "siemens"),
    ("Vodafone PLC", "vodafone"),
    ("Some Company Limited", "some"),
    ("Meta", "meta"),
    ("Coca-Cola", "cocacola"),
    ("", ""),
])
def test_normalize_entity(raw, expected):
    assert memory.normalize_entity(raw) == expected


def test_normalize_entity_only_strips_trailing_suffixes():
    assert memory.normalize_entity("Inc Holdings") == "inc holdings"


def test_normalize_entity_suffix_only_name_collapses_to_empty():
    assert memory.normalize_entity("Inc") == ""


def test_find_prior_research_without_dsn_returns_none():
    assert memory.find_prior_research("Stripe") is None


def test_save_research_without_dsn_is_noop(monkeypatch):
    def boom():
        raise AssertionError("should not connect")
    monkeypatch.setattr(memory, "_connect", boom)
    memory.save_research("Stripe", {"a": "b"}, {"a": ["s"]})


def test_find_prior_research_exact_match(dsn, monkeypatch):
    created = datetime.now(timezone.utc) - timedelta(days=3, hours=1)
    conn = FakeConn(fetchall_rows=[[(created, {"risks": "r"}, {"risks": ["s"]})]])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    result = memory.find_prior_research("Stripe Inc")
    assert result["exact_match"] is True
    assert result["findings"] == {"risks": "r"}
    assert result["sources"] == {"risks": ["s"]}
    assert 3.0 < result["ages"]["risks"] < 3.1
    assert conn.executed[0][1][0] == "stripe"


def test_find_prior_research_keeps_fractional_age(dsn, monkeypatch):
    created = datetime.now(timezone.utc) - timedelta(days=7, hours=20)
    conn = FakeConn(fetchall_rows=[[(created, {"risks": "r"}, {})]])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    age = memory.find_prior_research("Stripe")["ages"]["risks"]
    assert age > config.MEMORY_CACHE_DAYS


def test_find_prior_research_merges_fields_across_rows_newest_first(dsn, monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [
        (now - timedelta(days=1), {"risks": "new risk"}, {"risks": ["n"]}),
        (now - timedelta(days=5), {"risks": "old risk", "competitors": "old comp"}, {"risks": ["o"], "competitors": ["oc"]}),
    ]
    conn = FakeConn(fetchall_rows=[rows])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    result = memory.find_prior_research("Stripe")
    assert result["findings"] == {"risks": "new risk", "competitors": "old comp"}
    assert result["sources"] == {"risks": ["n"], "competitors": ["oc"]}
    assert result["ages"]["risks"] < result["ages"]["competitors"]


def test_find_prior_research_null_sources_becomes_empty_list(dsn, monkeypatch):
    created = datetime.now(timezone.utc)
    conn = FakeConn(fetchall_rows=[[(created, {"risks": "r"}, None)]])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    assert memory.find_prior_research("Stripe")["sources"] == {"risks": []}


def test_find_prior_research_fuzzy_match(dsn, monkeypatch):
    conn = FakeConn(fetchall_rows=[[], [("stripe",), ("notion",)]])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    result = memory.find_prior_research("Stripee")
    assert result["exact_match"] is False
    assert result["fuzzy_candidate"] == "stripe"
    assert result["score"] >= config.FUZZY_MATCH_THRESHOLD


def test_find_prior_research_meta_vs_meta_financial_group_surfaces_warning(dsn, monkeypatch):
    conn = FakeConn(fetchall_rows=[[], [("meta financial group",)]])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    result = memory.find_prior_research("Meta")
    assert result is not None
    assert result["fuzzy_candidate"] == "meta financial group"


def test_find_prior_research_no_match_below_threshold(dsn, monkeypatch):
    conn = FakeConn(fetchall_rows=[[], [("zzzzzz",)]])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    assert memory.find_prior_research("Stripe") is None


def test_find_prior_research_empty_table(dsn, monkeypatch):
    conn = FakeConn(fetchall_rows=[[], []])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    assert memory.find_prior_research("Stripe") is None


def test_fuzzy_candidates_are_ordered_by_recency(dsn, monkeypatch):
    conn = FakeConn(fetchall_rows=[[], []])
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    memory.find_prior_research("Stripe")
    assert "MAX(created_at) DESC" in conn.executed[1][0]


def test_find_prior_research_swallows_db_errors(dsn, monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(memory, "_connect", boom)
    assert memory.find_prior_research("Stripe") is None


def test_save_research_inserts_and_commits(dsn, monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(memory, "_connect", lambda: conn)
    memory.save_research("Stripe, Inc.", {"risks": "r"}, {"risks": ["s"]})
    sql, params = conn.executed[0]
    assert "INSERT INTO research_runs" in sql
    assert params[0] == "stripe"
    assert params[1] == "Stripe, Inc."
    assert conn.committed


def test_save_research_swallows_db_errors(dsn, monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(memory, "_connect", boom)
    memory.save_research("Stripe", {}, {})




def test_save_research_with_empty_findings_is_noop(dsn, monkeypatch):
    def boom():
        raise AssertionError("should not connect")
    monkeypatch.setattr(memory, "_connect", boom)
    memory.save_research("Stripe", {}, {})
