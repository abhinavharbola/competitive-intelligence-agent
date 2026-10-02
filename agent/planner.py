from agent import llm, schemas
from agent.state import ResearchState, PlanStep
from agent.guardrails import wall_clock_exceeded
from tools.results import UNTRUSTED_NOTICE, excerpt, is_empty_result
import config

_BASE = """You are the Planner for a Competitive Intelligence Agent.
You are given a company/product name, today's date, any findings already confirmed, and optionally gaps flagged by the Critic.
The required fields are: what_it_does, funding_ownership, recent_news, competitors, risks.
When gaps are listed, plan only for those fields. Otherwise plan for every required field that is not already confirmed. Never plan for a field that is already confirmed unless it is listed as a gap.
Each sub_question maps to exactly one field and one tool ("search" or "calculator").
Only use "calculator" for numeric verification (e.g. growth rate math), never for lookups.
You are given today's date. Use it to phrase recent_news sub-questions concretely (e.g. an actual
month/year or "in the last 30 days") rather than a vague "recent", and never assume your own
training cutoff is the current date.
"""

SYSTEM = (
    _BASE
    + f"Plan at most {config.MAX_PLAN_STEPS} steps in total.\n"
    + UNTRUSTED_NOTICE
    + '\nRespond as JSON: {"steps": [{"sub_question": str, "field": str, "tool": "search"|"calculator"}]}'
)


def _confirmed_summary(state: ResearchState, gaps: list[str]) -> str:
    lines = []
    for e in state.get("scratchpad", []):
        if e["field"] in gaps or is_empty_result(e["result"]):
            continue
        lines.append(f"- {e['field']}: {excerpt(e['result'], 200)}")
    return "\n".join(lines)


def plan(state: ResearchState) -> ResearchState:
    if wall_clock_exceeded(state):
        state["stop_reason"] = state.get("stop_reason") or "wall_clock"
        state["plan"] = []
        return state

    gaps = list(state["critique"]["gaps"]) if state.get("critique") else []
    prior = _confirmed_summary(state, gaps)

    user = f"Entity: {state['entity']}\nToday's date: {state['today']}\n"
    if gaps:
        user += f"Critic flagged these gaps, focus the plan on closing them: {gaps}\n"
    if prior:
        user += f"Findings already confirmed, do not repeat these:\n{prior}\n"

    try:
        response = llm.call_planner(SYSTEM, user)
        validated = schemas.valid_planner_steps(response)
        steps = [
            PlanStep(sub_question=s.sub_question, field=s.field, tool=s.tool, status="pending")
            for s in validated[: config.MAX_PLAN_STEPS]
        ]
    except Exception as e:
        print(f"  [planner] failed after retries: {e}", flush=True)
        state["stop_reason"] = state.get("stop_reason") or "planner_unavailable"
        steps = []

    state["plan"] = steps
    return state
