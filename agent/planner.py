from agent import llm
from agent.state import ResearchState, PlanStep

SYSTEM = """You are the Planner for a Competitive Intelligence Agent.
Given a company/product name and, optionally, gaps flagged by the Critic, produce a research plan.
The plan must cover these required fields: what_it_does, funding_ownership, recent_news, competitors, risks.
Each sub_question maps to exactly one field and one tool ("search" or "calculator").
Only use "calculator" for numeric verification (e.g. growth rate math), never for lookups.
Respond as JSON: {"steps": [{"sub_question": str, "field": str, "tool": "search"|"calculator"}]}"""


def plan(state: ResearchState) -> ResearchState:
    gaps = state["critique"]["gaps"] if state.get("critique") else []
    prior = "\n".join(f"- {e['field']}: {e['result'][:200]}" for e in state.get("scratchpad", []))

    user = f"Entity: {state['entity']}\n"
    if gaps:
        user += f"Critic flagged these gaps, focus the plan on closing them: {gaps}\n"
    if prior:
        user += f"Findings already confirmed, do not repeat these:\n{prior}\n"

    response = llm.call_planner(SYSTEM, user)
    steps = [
        PlanStep(sub_question=s["sub_question"], field=s["field"], tool=s["tool"], status="pending")
        for s in response["steps"]
    ]
    state["plan"] = steps
    return state
