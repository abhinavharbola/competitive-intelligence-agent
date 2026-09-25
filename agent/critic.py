from agent import llm, schemas
from agent.state import ResearchState
from agent.guardrails import wall_clock_exceeded
import config

SYSTEM = """You are the Critic for a Competitive Intelligence Agent.
Check the scratchpad against the 5 required fields: what_it_does, funding_ownership, recent_news, competitors, risks.
A field is satisfied only if the scratchpad contains a sourced, on-topic finding for it.
You are given today's date. Judge recent_news against it, a finding dated well before today is
not sufficient on its own, flag recent_news as a gap if nothing in the scratchpad is genuinely
recent relative to the given date; never assume your own training cutoff is the current date.
Respond as JSON: {"approved": bool, "gaps": [field names still missing or insufficiently sourced]}
Every gap must be exactly one of the 5 field names above, nothing else.
approved is true only when gaps is empty."""


def critique(state: ResearchState) -> ResearchState:
    if state.get("stop_reason"):
        return state

    if wall_clock_exceeded(state):
        state["stop_reason"] = "wall_clock"
        return state

    scratchpad_summary = "\n".join(
        f"- field={e['field']} source={e['source']}\n  {e['result']}"
        for e in state["scratchpad"]
    )
    user = f"Entity: {state['entity']}\nToday's date: {state['today']}\nScratchpad:\n{scratchpad_summary or 'empty'}"

    try:
        result = llm.call_gemini(config.CRITIC_MODEL, SYSTEM, user)
        parsed = schemas.CriticResponse.model_validate(result)
    except Exception as e:
        print(f"  [critic] failed after retries: {e}", flush=True)
        state["stop_reason"] = "critic_unavailable"
        return state

    gaps = [g for g in parsed.gaps if g in config.REQUIRED_FIELDS]
    dropped = [g for g in parsed.gaps if g not in config.REQUIRED_FIELDS]
    if dropped:
        print(f"  [critic] dropping unrecognized gap field(s) {dropped}", flush=True)

    approved = parsed.approved or not gaps

    state["critique"] = {"approved": approved, "gaps": gaps}
    if not approved:
        state["replan_count"] += 1
    return state
