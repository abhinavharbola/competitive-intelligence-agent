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


def normalize_entity(name: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", name.lower()).strip()
    words = cleaned.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def _connect():
    return psycopg.connect(config.NEON_DSN)


def find_prior_research(entity_raw: str) -> dict | None:
    if not config.NEON_DSN:
        return None

    normalized = normalize_entity(entity_raw)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, findings FROM research_runs
                WHERE entity_normalized = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (normalized,),
            )
            row = cur.fetchone()
            if row:
                created_at, findings = row
                age_days = (datetime.now(timezone.utc) - created_at).days
                return {"exact_match": True, "age_days": age_days, "findings": findings}

            cur.execute("SELECT DISTINCT entity_normalized FROM research_runs")
            candidates = [r[0] for r in cur.fetchall()]

    best_score, best_candidate = 0, None
    for candidate in candidates:
        score = fuzz.ratio(normalized, candidate)
        if score > best_score:
            best_score, best_candidate = score, candidate

    if best_candidate and best_score >= config.FUZZY_MATCH_THRESHOLD:
        return {"exact_match": False, "fuzzy_candidate": best_candidate, "score": best_score}

    return None


def save_research(entity_raw: str, findings: dict, sources: dict) -> None:
    if not config.NEON_DSN:
        return

    normalized = normalize_entity(entity_raw)
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
