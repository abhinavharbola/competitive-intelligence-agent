# Architecture

## Graph

```mermaid
flowchart TD
    start([entity name]) --> mem[memory lookup]
    mem -->|exact match, fresh| seed[seed scratchpad\nall fields except recent_news]
    mem -->|fuzzy match| note[memory_note only\nno seeding]
    mem -->|no match| planner
    seed --> planner
    note --> planner

    planner[Planner\nNIM - Llama-3.1-8B] --> executor[Executor\nGroq - gpt-oss-120b]
    executor --> critic[Critic\nGemini 3.5-flash]
    critic -->|approved| synthesizer[Synthesizer\nGemini 3.5-flash]
    critic -->|gaps, replan_count < 3| planner
    critic -->|gaps, replan_count = 3\nor stop_reason set| synthesizer
    synthesizer --> save[save to Neon]
    save --> report([final report])
```

With the Critic loop disabled (ablation), `executor` connects directly to `synthesizer`; the
`critic` node is not present in that graph at all, not just skipped.

## State

All nodes read and write a single `ResearchState` dict, passed through the LangGraph state graph:

| field | written by | read by |
|---|---|---|
| `plan` | planner | executor |
| `scratchpad` | executor, memory seed | critic, synthesizer |
| `critique` | critic | router |
| `replan_count` | critic | router |
| `tool_call_count`, `tool_call_log` | executor | executor (loop detection), router |
| `stop_reason` | executor, router | router, caller |
| `report`, `field_status` | synthesizer | caller |
| `memory_note` | memory seed step | caller |

Nothing is shared mutable state outside this dict — each node is a pure function of
`ResearchState -> ResearchState`, which is what makes the ablation variant (dropping the critic
node) a structural graph change rather than a conditional inside a monolithic function.

## Data flow per node

**Memory lookup** (`agent/graph.py:_seed_from_memory`, runs before the graph)
Normalizes the entity name, checks Neon for an exact match. Fresh exact match seeds the
scratchpad for every field except `recent_news`. Fuzzy match never seeds — it only sets
`memory_note` so the caller sees "possible match, not used" instead of silently getting the wrong
company's data.

**Planner** (`agent/planner.py`)
Input: entity, any Critic gaps from the previous cycle, and prior-run scratchpad summary (so it
doesn't re-plan fields already confirmed). Output: `plan`, a list of `{sub_question, field, tool}`.

**Executor** (`agent/executor.py`)
For each pending plan step: asks the Executor model for concrete tool args, checks
`tool_call_log` for an exact repeat (loop detection), calls the tool, wraps search results in
`<untrusted_web_content>` delimiters, appends to `scratchpad`. Stops early if `MAX_TOOL_CALLS` or
`MAX_WALL_CLOCK_SECONDS` is hit, setting `stop_reason`.

**Critic** (`agent/critic.py`)
Checks `scratchpad` against the 5 required fields. Returns `{approved, gaps}`. Every rejection
increments `replan_count`.

**Router** (`agent/graph.py:route_after_critic`)
`approved` -> synthesizer. Gaps and `replan_count < 3` and no `stop_reason` -> back to planner.
Gaps but replan budget exhausted, or executor already set `stop_reason` -> synthesizer anyway,
so the run always terminates with whatever was confirmed rather than looping or crashing.

**Synthesizer** (`agent/synthesizer.py`)
Builds the final report from `scratchpad` only — explicitly instructed not to introduce claims
absent from it. Unfilled fields get `"insufficient information"` in `field_status` rather than a
guess.

**Save** (`agent/graph.py:run`, after the graph completes)
Every run's final per-field findings are written back to `research_runs` in Neon, whether or not
the run itself started from a cache hit.

## Model families

Three distinct families across the pipeline, on purpose:

- Llama (Planner, via NIM)
- Gemini (Critic + Synthesizer)
- Qwen (eval Judge, via a second NIM account)

Groq hosts gpt-oss-120b for the Executor — a fourth family, on its own provider so Executor's
per-call volume (up to 15 calls/run) doesn't share a rate-limit budget with anything else.

Keeping Judge separate from Critic/Synthesizer's family matters specifically for the ablation
study: the study compares "Critic loop on" vs "off," and a judge from the same family as the
Critic would risk scoring outputs more favorably when they resemble its own family's reasoning
style, biasing the comparison it's supposed to be neutral about.

## Safety boundary

Every piece of tool output that originates from the open web (Tavily search results) is wrapped
in `<untrusted_web_content>` before it reaches any LLM prompt, with system instructions telling
the model that content inside is data, never instructions. Calculator output isn't wrapped —
it's derived from the model's own expression via `simpleeval`, not fetched from an external
source, so it isn't untrusted in the same sense.
