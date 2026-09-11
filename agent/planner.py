from agent import llm, schemas
from agent.state import ResearchState, PlanStep
from agent.guardrails import wall_clock_exceeded

SYSTEM = """You are the Planner for a Competitive Intelligence Agent.
Given a company/product name and, optionally, gaps flagged by the Critic, produce a research plan.
The plan must cover these required fields: what_it_does, funding_ownership, recent_news, competitors, risks.
Each sub_question maps to exactly one field and one tool ("search" or "calculator").
Only use "calculator" for numeric verification (e.g. growth rate math), never for lookups.
Respond as JSON: {"steps": [{"sub_question": str, "field": str, "tool": "search"|"calculator"}]}"""


def plan(state: ResearchState) -> ResearchState:
    if wall_clock_exceeded(state):
        state["stop_reason"] = state.get("stop_reason") or "wall_clock"
        state["plan"] = []
        return state

    gaps = state["critique"]["gaps"] if state.get("critique") else []
    prior = "\n".join(f"- {e['field']}: {e['result'][:200]}" for e in state.get("scratchpad", []))

    user = f"Entity: {state['entity']}\n"
    if gaps:
        user += f"Critic flagged these gaps, focus the plan on closing them: {gaps}\n"
    if prior:
        user += f"Findings already confirmed, do not repeat these:\n{prior}\n"

    try:
        response = llm.call_planner(SYSTEM, user)
        validated = schemas.valid_planner_steps(response)
        steps = [
            PlanStep(sub_question=s.sub_question, field=s.field, tool=s.tool, status="pending")
            for s in validated
        ]
    except Exception as e:
        print(f"  [planner] failed after retries: {e}", flush=True)
        state["stop_reason"] = state.get("stop_reason") or "planner_unavailable"
        steps = []

    state["plan"] = steps
    return state
