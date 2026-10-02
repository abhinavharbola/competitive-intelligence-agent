import json
import re
from pathlib import Path

import pytest

import config

DATA = json.loads((Path(__file__).parent.parent / "eval" / "benchmark.json").read_text())


def test_has_fifteen_entries():
    assert len(DATA) == 15


def test_entities_are_unique():
    names = [d["entity"].lower() for d in DATA]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("item", DATA, ids=[d["entity"] for d in DATA])
def test_entry_shape(item):
    assert set(item) == {"entity", "ground_truth", "verified"}
    assert item["verified"] is True
    assert set(item["ground_truth"]) == set(config.REQUIRED_FIELDS)
    assert all(isinstance(v, str) and v.strip() for v in item["ground_truth"].values())


def test_schema_sql_defines_expected_table_and_index():
    sql = (Path(__file__).parent.parent / "memory" / "schema.sql").read_text()
    assert re.search(r"CREATE TABLE IF NOT EXISTS research_runs", sql)
    for col in ("entity_normalized", "entity_raw", "created_at", "findings", "sources"):
        assert col in sql
    assert "idx_research_runs_entity_normalized" in sql


