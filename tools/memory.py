import re
from datetime import datetime, timezone
import psycopg
from psycopg.types.json import Jsonb
from rapidfuzz import fuzz
import config

LEGAL_SUFFIXES = {
    "incorporated", "corporation", "limited", "company",
    "inc", "corp", "ltd", "llc", "co", "plc", "gmbh",
}

_FUZZY_CANDIDATE_LIMIT = 2000
_ROW_LIMIT = 20
_SECONDS_PER_DAY = 86400


def normalize_entity(name: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", name.lower()).strip()
    words = cleaned.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def _connect():
    return psycopg.connect(config.NEON_DSN)


def _merge_rows(rows: list) -> dict:
    now = datetime.now(timezone.utc)
    findings: dict[str, str] = {}
    sources: dict[str, list] = {}
    ages: dict[str, float] = {}
    for created_at, row_findings, row_sources in rows:
        age = (now - created_at).total_seconds() / _SECONDS_PER_DAY
        row_sources = row_sources or {}
        for field, text in (row_findings or {}).items():
            if field in findings:
                continue
            findings[field] = text
            sources[field] = row_sources.get(field, [])
            ages[field] = age
    return {"exact_match": True, "findings": findings, "sources": sources, "ages": ages}


def find_prior_research(entity_raw: str) -> dict | None:
    if not config.NEON_DSN:
        return None

    normalized = normalize_entity(entity_raw)
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at, findings, sources FROM research_runs
                    WHERE entity_normalized = %s
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (normalized, _ROW_LIMIT),
                )
                rows = cur.fetchall()
                if rows:
                    return _merge_rows(rows)

                cur.execute(
                    """
                    SELECT entity_normalized FROM research_runs
                    GROUP BY entity_normalized
                    ORDER BY MAX(created_at) DESC LIMIT %s
                    """,
                    (_FUZZY_CANDIDATE_LIMIT,),
                )
                candidates = [r[0] for r in cur.fetchall()]
    except Exception as e:
        print(f"  [memory] lookup failed, continuing without cache: {e}", flush=True)
        return None

    best_score, best_candidate = 0, None
    for candidate in candidates:
        score = fuzz.WRatio(normalized, candidate)
        if score > best_score:
            best_score, best_candidate = score, candidate

    if best_candidate and best_score >= config.FUZZY_MATCH_THRESHOLD:
        return {"exact_match": False, "fuzzy_candidate": best_candidate, "score": best_score}

    return None


def save_research(entity_raw: str, findings: dict, sources: dict) -> None:
    if not config.NEON_DSN or not findings:
        return

    normalized = normalize_entity(entity_raw)
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO research_runs (entity_normalized, entity_raw, findings, sources)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (normalized, entity_raw, Jsonb(findings), Jsonb(sources)),
                )
            conn.commit()
    except Exception as e:
        print(f"  [memory] save failed, result not cached: {e}", flush=True)
