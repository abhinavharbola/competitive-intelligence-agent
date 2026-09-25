import time
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from agent.state import ResearchState, ScratchpadEntry
from agent.planner import plan
from agent.executor import execute
from agent.critic import critique
from agent.synthesizer import synthesize
from agent.tracing import traced_node
from tools.memory import find_prior_research, save_research
import config


def route_after_critic(state: ResearchState) -> str:
    if state["critique"]["approved"]:
        return "synthesizer"
    if state.get("stop_reason"):
        return "synthesizer"
    if state["replan_count"] >= config.MAX_REPLAN_CYCLES:
        state["stop_reason"] = "max_replans"
        return "synthesizer"
    return "planner"


def build_graph(critic_enabled: bool = True):
    graph = StateGraph(ResearchState)
    graph.add_node("planner", traced_node("planner")(plan))
    graph.add_node("executor", traced_node("executor")(execute))
    graph.add_node("synthesizer", traced_node("synthesizer")(synthesize))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")

    if critic_enabled:
        graph.add_node("critic", traced_node("critic")(critique))
        graph.add_edge("executor", "critic")
        graph.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "synthesizer": "synthesizer"})
    else:
        graph.add_edge("executor", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile()


def build_initial_state(entity: str) -> ResearchState:
    return {
        "entity": entity,
        "today": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "plan": [],
        "scratchpad": [],
        "critique": {"approved": False, "gaps": []},
        "replan_count": 0,
        "tool_call_count": 0,
        "tool_call_cache": {},
        "start_time": time.time(),
        "report": "",
        "field_status": {},
        "stop_reason": "",
        "memory_note": "",
    }


def seed_from_memory(state: ResearchState, entity: str) -> tuple[ResearchState, str]:
    prior = find_prior_research(entity)
    if not prior:
        return state, ""

    if not prior["exact_match"]:
        return state, (
            f"possible prior entry '{prior['fuzzy_candidate']}' "
            f"(similarity {prior['score']}) found but not auto-used, verify manually"
        )

    if prior["age_days"] > config.MEMORY_CACHE_DAYS:
        return state, ""

    for field, result in prior["findings"].items():
        if field == "recent_news":
            continue
        prior_sources = prior["sources"].get(field, [])
        provenance = "; ".join(prior_sources) if prior_sources else "unrecorded"
        state["scratchpad"].append(
            ScratchpadEntry(
                sub_question=f"cached finding for {field}",
                field=field,
                tool="memory",
                args="",
                result=result,
                source=f"cache, {prior['age_days']}d old, originally: {provenance}",
            )
        )
    return state, ""


def save_results(entity: str, final_state: ResearchState) -> None:
    findings: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    for field in config.REQUIRED_FIELDS:
        if final_state["field_status"].get(field) != "confirmed":
            continue
        entries = [e for e in final_state["scratchpad"] if e["field"] == field]
        if not entries:
            continue
        findings[field] = "\n\n".join(e["result"] for e in entries)
        sources[field] = sorted({e["source"] for e in entries})
    save_research(entity, findings, sources)


def run(entity: str, critic_enabled: bool = True, use_memory: bool = True) -> ResearchState:
    initial_state = build_initial_state(entity)

    memory_note = ""
    if use_memory:
        initial_state, memory_note = seed_from_memory(initial_state, entity)

    app = build_graph(critic_enabled=critic_enabled)
    final_state = app.invoke(initial_state)
    final_state["memory_note"] = memory_note

    if use_memory:
        save_results(entity, final_state)

    return final_state
