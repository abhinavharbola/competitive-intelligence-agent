# Competitive Intelligence Agent (CIA)

An autonomous research agent that takes a company or product name as input and delivers a structured, sourced intelligence brief covering what it does, funding and ownership, recent news, competitors, and risks. Every claim is traceable to a search result or a recorded calculation, and anything that could not be sourced is marked "insufficient information" instead of being guessed.

Built as a portfolio project on entirely free-tier infrastructure: no paid APIs, no GPU, no local model.

## Preview

<p align="center">
  <img src="assets/main_ui.png" width="720" alt="Streamlit UI showing the entity input and a live architecture diagram of the Planner, Executor, Critic, and Synthesizer pipeline">
  <br>
  <sub>Landing view, entity input, and the pipeline's own architecture rendered inline.</sub>
</p>

<p align="center">
  <img src="assets/research_log_and_dossier.png" width="720" alt="Live research log streaming node-by-node progress next to a dossier status panel showing all five fields confirmed">
  <br>
  <sub>A run in progress: the research log streams node-by-node on the left (including a live Critic replan cycle), while the dossier status panel on the right stamps each of the 5 required fields as it is sourced.</sub>
</p>

> Additional screenshots and example report in [`assets/`](assets/).

## What this is

Given an entity name, the agent:

1. **Plans** a small set of research sub-questions (at most 8 per plan) covering the 5 required fields, skipping any field already confirmed from cache.
2. **Executes** each one, as a web search or a calculation, and writes the results to a scratchpad.
3. **Critiques** its own coverage against the 5 fields, and can send itself back to re-plan up to 3 times if something is missing. Each replan targets only the flagged gaps and avoids repeating earlier queries.
4. **Synthesizes** a final brief from the scratchpad only, with numbered citations and a references list. Field statuses are checked against the scratchpad, so a field cannot be reported as confirmed without evidence behind it.

It remembers past runs per field (Neon/Postgres), traces every node, LLM call and tool call (Logfire), is exposed as both a FastAPI endpoint and a Streamlit UI, and ships with an evaluation harness that runs a paired ablation study on its own Critic loop.

## Architecture

```mermaid
flowchart TD
    start([entity name]) --> mem[memory lookup]
    mem -->|exact match, per-field fresh| seed[seed scratchpad\nfresh fields except recent_news]
    mem -->|fuzzy match| note[memory_note only\nno seeding]
    mem -->|no match| planner
    seed --> planner
    note --> planner

    planner[Planner\nNIM - Nemotron 3 Super 120B] --> executor[Executor\nGroq - gpt-oss-120b]
    executor --> critic[Critic\nGemini 3.5 Flash-Lite]
    critic -->|approved| synthesizer[Synthesizer\nGemini 3.5 Flash]
    critic -->|gaps, replans used < 3| planner
    critic -->|gaps, 3 replans already used\nor stop_reason set| synthesizer
    synthesizer --> save[save fresh fields to Neon]
    save --> report([final report])
```

A full run is at most 4 Planner calls and 4 Critic calls: the initial pass plus 3 replans. With the Critic loop disabled (used for the ablation study), Executor connects straight to Synthesizer. The `critic` node is not present in that graph at all, not just skipped at runtime.

Full node-by-node data flow and state schema: [`docs/architecture.md`](docs/architecture.md).

## Models
 
Models are matched to free-tier rate limits (RPM/TPM/RPD) by each role's call volume:
 
| Role | Model | Provider | Calls/run | Why this model, this provider |
|---|---|---|---|---|
| Planner | `nvidia/nemotron-3-super-120b-a12b` | NIM | 1 to 4 | NVIDIA's model for multi-step planning. Low-volume role. |
| Executor | `openai/gpt-oss-120b` | Groq | one per pending step, at most 8 steps per plan | Highest-volume, latency-sensitive role, so it gets Groq. Blocked and failed steps still cost a call (see Known limitations). |
| Critic | `gemini-3.5-flash-lite` | Gemini | 1 to 4 | Light JSON gap-classification; needs less quality, gets higher RPM/RPD. |
| Synthesizer | `gemini-3.5-flash` | Gemini | 1 | Writes what the user reads, so it gets full Flash. |
| Eval judge | `openai/gpt-oss-120b` | Groq | eval-only | Separate key keeps judge traffic off the Executor's quota. Grades the Gemini report, never Executor output. |
 
## Guardrails
 
| Guardrail | Behavior |
|---|---|
| Hard stops | 3 replans, 15 tool calls, 8 steps per plan, 8 minutes wall-clock. On a limit: return confirmed fields, mark the rest "insufficient information". |
| Clock checks | Before the Planner, Critic and each Executor step, and after tool input is generated. The Synthesizer always runs; an in-flight call may finish. |
| Loop detection | Executor sees past queries and must vary them. An exact repeat skips the call and reuses the cached result for that field, once. |
| Timeouts | 60s per LLM call (client-enforced for NIM and Groq, a watchdog thread for Gemini). |
| Retries | 3 attempts, exponential backoff; `429` waits 10s then 20s. Malformed JSON is retried, Tavily once. 400, 401, 403, 404, 422, invalid key and exhausted daily quota fail fast. Each failure logs a Logfire warning. |
| Failure isolation | Tool error: `failed`. Duplicate skipped: `blocked`. Logged separately. |
| Degradation | Critic fails: go to the Synthesizer. Synthesizer fails or returns empty: unverified scratchpad report, never cached. Stop reasons are joined (`max_replans+synthesizer_unavailable`), never overwritten. |
 
## Memory
 
Neon/Postgres, keyed by a normalized entity name (lowercased, legal suffixes like Inc/Ltd/Corp/LLC stripped).
 
| Lookup | Behavior |
|---|---|
| Exact match | Rows merged per field, newest wins, each with its own age in fractional days. Fields under 7 days old are seeded and not re-planned, except `recent_news`, which is always re-researched. |
| Fuzzy match only | Never seeded. `WRatio` is substring-aware, so "Meta" vs. "Meta Financial Group" should raise a `memory_note`. Candidates: the 2000 most recent entities. |
| No match | Fresh research. |
 
**Write-back**:
- Saved: fields the Synthesizer marked `"confirmed"` that have a non-empty, non-cached entry (all such entries kept).
- Cached text is never re-saved, so ages never reset and text never grows.
- A run with no new work, or a fallback report, writes nothing, so earlier rows are never shadowed.
- Unconfirmed fields, including any cut off by a hard stop, are never cached.
## Safety
 
| Area | Mitigation | Limit |
|---|---|---|
| Prompt injection | Tavily output is wrapped in `<untrusted_web_content>`; the Planner, Executor, Critic, Synthesizer and Judge prompts treat it as data. Embedded tags are neutralized, excerpts re-wrapped. | A mitigation, not a guarantee. |
| Calculator | Built from scratchpad findings; stored as `calculator: <expression>` with its result, so it is traceable. | "Use only these numbers" is a model instruction, not code-enforced. |
| API | Optional `API_ACCESS_KEY` requires `X-API-Key` on `POST /research`. Entity stripped, capped at 200 characters. | No rate limiting. |

## Project Structure

```
competitive-intelligence-agent/
├── agent/
│   ├── __init__.py
│   ├── state.py               # shared graph state schema
│   ├── schemas.py             # pydantic validation for every LLM JSON response
│   ├── guardrails.py          # wall-clock check, combined stop-reason helpers
│   ├── llm.py                 # provider clients, timeouts, retry classification
│   ├── tracing.py             # Logfire setup and node tracing
│   ├── planner.py
│   ├── executor.py
│   ├── critic.py
│   ├── synthesizer.py
│   └── graph.py               # StateGraph wiring, run setup, memory seed and save
│
├── tools/
│   ├── __init__.py
│   ├── search.py              # Tavily wrapper, injection-delimited output
│   ├── calculator.py          # simpleeval wrapper
│   ├── results.py             # untrusted-content wrapper, prompt notice, empty-result detection
│   └── memory.py              # entity normalization + Neon read/write
│
├── docs/architecture.md
├── memory/schema.sql          # Neon table DDL
│
├── ui/app.py                  # Streamlit live trace views
├── api/main.py                # FastAPI, research endpoint
│
├── eval/
│   ├── benchmark.json         # 15 companies + ground truth
│   ├── judge.py               # Groq (openai/gpt-oss-120b) judge calls, scores + notes
│   ├── run_ablation.py        # paired critic on/off runner
│   └── results/
│
├── tests/                     # offline pytest suite, all providers stubbed
├── pytest.ini
│
├── .streamlit/config.toml     # locked light theme
├── assets/
│
├── .gitignore
├── .env.example
├── config.py                  # env loading, model config, all limits (memory days, replans, tool calls, plan steps, wall-clock)
├── requirements.txt           # runtime dependencies
├── requirements-dev.txt       # adds pytest and httpx
└── README.md
```

## Getting started

Requires Python 3.10 or newer (the code uses `X | None` annotations).

1. **API keys**, you'll need:
   - NVIDIA NIM (Planner): https://build.nvidia.com
   - Groq (Executor and eval Judge): https://console.groq.com/keys. The two keys are split for free-tier quotas; on a paid plan both variables can hold the same key.
   - Gemini (Critic and Synthesizer, two models): https://aistudio.google.com/apikey
   - Tavily (free tier): https://tavily.com
   - Neon (free tier): https://neon.tech. `NEON_DSN` is the Postgres connection string from the Neon dashboard (it starts with `postgresql://`).
   - Logfire (optional, tracing no-ops without it): https://logfire.pydantic.dev
   - `API_ACCESS_KEY` (optional): any secret you choose; the API then requires it in `X-API-Key`

2. **Install**
   ```
   python3 -m venv venv && source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env   # fill in all keys except the optional ones (LOGFIRE_TOKEN, NEON_DSN, API_ACCESS_KEY, and GROQ_JUDGE_API_KEY unless running the eval)
   ```

3. **Database** (optional, no local `psql` needed): paste [`memory/schema.sql`](memory/schema.sql) into your Neon project's SQL Editor and run it. Without `NEON_DSN`, memory safely does nothing.

4. **Testing**
   ```
   pytest tests/ -v
   ```
   Tests run fully offline: `tests/conftest.py` injects fake API keys and stubs every LLM, Tavily and Postgres call. `tests/test_repo_consistency.py` also checks that file and function references in the docs resolve to real code. 

## Running it

Run every command from the repository root so `.env`, imports and `.streamlit/config.toml` are found.

```
uvicorn api.main:app --reload      # API on :8000, POST /research {"entity": "..."} (one request blocks for the whole run, up to about 8 minutes)
streamlit run ui/app.py            # live dossier console UI
python -m eval.run_ablation        # eval harness + critic on/off ablation study (add --limit N for a smaller slice)
```

Example API call (add `-H "X-API-Key: <your key>"` if `API_ACCESS_KEY` is set):

```
curl -X POST http://localhost:8000/research -H "Content-Type: application/json" -d '{"entity": "Stripe"}'
```

The API returns the report, per-field status, replan and tool-call counts, the scratchpad (execution trace) and any memory note. The Streamlit UI shows the same run live: a node-by-node research log, a dossier status panel with per-field stamps, and the filed brief with a download button.

## Evaluation
 
`/eval` is the project's core differentiator, not a checkbox.
 
| Component | What it does |
|---|---|
| [`benchmark.json`](eval/benchmark.json) | 15 companies, manually verified ground truth (`"verified": true`). Warns on any unverified entry. |
| [`judge.py`](eval/judge.py) | Groq `openai/gpt-oss-120b` judge on its own key. Scores groundedness and completeness (0 to 5) with a short note. Ground truth is a dated snapshot, so newer sourced `recent_news` is not penalized. Tool calls and time are measured directly. |
| [`run_ablation.py`](eval/run_ablation.py) | Critic on vs. off. Writes `with_critic.json`, `without_critic.json` and `summary.json` to `eval/results/`. |
 
- **Scores**: groundedness = every claim traces to a scratchpad source. Completeness = all 5 fields correct or marked insufficient.
- **Paired delta**: only entities that finished cleanly in both conditions count. Dropped ones go under `excluded_infra_failures` and `excluded_unpaired`.
- **`--limit N`**: positive integer, first N entries.
- **Noise**: judge variance dominates small slices (around n=3). Read the notes, and use all 15 when quotas allow.

## Evaluation Metrics (Local Run)

5-entity slice (Anthropic, Stripe, Notion, Figma, Databricks; hand-picked, so `--limit 5` does not reproduce it), Critic on vs. off:

| Metric | With Critic | Without Critic | Delta |
|---|---|---|---|
| `avg_groundedness` (0-5) | 4.6 | 3.8 | +0.8 |
| `avg_completeness` (0-5) | 4.8 | 3.8 | +1.0 |
| `avg_tool_calls` | 10.8 | 6.2 | +4.6 |
| `avg_elapsed_seconds` | ~96 | ~40 | +56 |
| `avg_replan_count` | 1.6 (of max 3) | 0, no critic node in this graph | n/a |
| `scored_entities` / `total_entities` | 5 / 5 | 5 / 5 | n/a |

Expected shape: the Critic costs about 1.7x the tool calls and 2.4x the time, but replans are more productive now. They target only gaps, avoid repeated queries and get a full 3 cycles instead of 2. Completeness was already near its ceiling, so the gain shows mainly in groundedness, which scratchpad-checked statuses and traceable calculations should also lift slightly without the Critic. Pairing removes bias from entities dropped in one condition only.

`n=5` is a smoke test: one noisy judge call can move an average by 0.2, so read the per-entity notes in `eval/results/` before trusting any delta.

## Known limitations

- **Groq quota**: 1,000 requests/day per key for `openai/gpt-oss-120b` on the free tier. Each pending step costs one Executor call, including `blocked` and `failed` ones. Typical use is about 10 to 15 per run, worst case 8 steps x 4 plans = 32, so budget roughly 30 to 90 runs/day.
- **Wall clock**: enforced between steps, so a run can overrun by one in-flight step.
- **Synthesizer**: `field_status` is checked against the scratchpad, but whether the report text matches those entries is judged by the model, not verified in code.
- **API**: no rate limiting, and each request holds a worker for the whole run. `API_ACCESS_KEY` is the only protection for the quotas.
- **Ablation**: small samples are dominated by judge variance.
- **Memory**: freshness is per field, so one entity's fields can be different ages.