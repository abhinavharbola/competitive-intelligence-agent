from agent import llm
from agent.state import ResearchState
import config

SYSTEM = """You are the Critic for a Competitive Intelligence Agent.
Check the scratchpad against the 5 required fields: what_it_does, funding_ownership, recent_news, competitors, risks.
A field is satisfied only if the scratchpad contains a sourced, on-topic finding for it.
Respond as JSON: {"approved": bool, "gaps": [field names still missing or insufficiently sourced]}
approved is true only when gaps is empty."""


def critique(state: ResearchState) -> ResearchState:
    scratchpad_summary = "\n".join(
        f"- field={e['field']} source={e['source']}\n  {e['result'][:400]}"
        for e in state["scratchpad"]
    )
    user = f"Entity: {state['entity']}\nScratchpad:\n{scratchpad_summary or 'empty'}"

    try:
        result = llm.call_gemini(config.CRITIC_MODEL, SYSTEM, user)
    except Exception as e:
        print(f"  [critic] failed after retries: {e}", flush=True)
        state["stop_reason"] = "critic_unavailable"
        return state

    state["critique"] = {"approved": result["approved"], "gaps": result["gaps"]}
    if not result["approved"]:
        state["replan_count"] += 1
    return state
