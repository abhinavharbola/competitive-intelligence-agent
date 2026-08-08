CREATE TABLE IF NOT EXISTS research_runs (
    id SERIAL PRIMARY KEY,
    entity_normalized TEXT NOT NULL,
    entity_raw TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    findings JSONB NOT NULL,
    sources JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_research_runs_entity_normalized
    ON research_runs (entity_normalized);
