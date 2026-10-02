from agent import llm, schemas
from agent.guardrails import wall_clock_exceeded
from agent.state import ResearchState, ScratchpadEntry
from tools.search import web_search
from tools.calculator import calculate
from tools.results import UNTRUSTED_NOTICE, excerpt, is_empty_result
import config

SYSTEM = (
    """You are the Executor for a Competitive Intelligence Agent.
Given one research sub-question, its assigned tool, and optionally earlier queries and findings, produce the exact tool input.
Respond as JSON. For "search" respond {"query": str}. For "calculator" respond {"expression": str}.
Keep search queries specific and short. Never repeat a query or expression listed as already run, vary the wording or angle instead.
For "calculator", build the expression only from numbers that appear in the provided findings, never from memory, using plain arithmetic operators.
You are given today's date, use it to make a
recency-sensitive query concrete (an actual month/year) instead of relying on the word "recent"
alone, and never assume your own training cutoff is the current date.
"""
    + UNTRUSTED_NOTICE
)

_MAX_CONTEXT_ITEMS = 12


def _run_tool(tool: str, arg_value: str) -> str:
    return web_search(arg_value) if tool == "search" else calculate(arg_value)


def _build_user(state: ResearchState, step) -> str:
    lines = [
        f"Today's date: {state['today']}",
        f"Sub-question: {step['sub_question']}",
        f"Tool: {step['tool']}",
    ]
    previous = [e["args"] for e in state["scratchpad"] if e["tool"] == step["tool"] and e["args"]]
    if previous:
        unique = list(dict.fromkeys(previous))[-_MAX_CONTEXT_ITEMS:]
        noun = "queries" if step["tool"] == "search" else "expressions"
        lines.append(f"Already-run {noun}, do not repeat: " + "; ".join(unique))
    if step["tool"] == "calculator":
        findings = [
            f"- {e['field']}: {excerpt(e['result'], 400)}"
            for e in state["scratchpad"]
            if not is_empty_result(e["result"])
        ][-_MAX_CONTEXT_ITEMS:]
        if findings:
            lines.append("Findings available for the calculation:\n" + "\n".join(findings))
    return "\n".join(lines)


def execute(state: ResearchState) -> ResearchState:
    for step in state["plan"]:
        if step["status"] != "pending":
            continue
        if state["tool_call_count"] >= config.MAX_TOOL_CALLS:
            state["stop_reason"] = "max_tool_calls"
            break
        if wall_clock_exceeded(state):
            state["stop_reason"] = "wall_clock"
            break

        print(f"  [step] {step['field']}: {step['sub_question']}", flush=True)

        try:
            args_response = llm.call_executor(SYSTEM, _build_user(state, step))
            args = schemas.ExecutorArgs.model_validate(args_response)
            raw_arg = args.query if step["tool"] == "search" else args.expression
            arg_value = raw_arg.strip()
        except Exception as e:
            print(f"  [step] {step['field']} failed to produce tool args: {e}", flush=True)
            step["status"] = "failed"
            continue

        if not arg_value:
            print(f"  [step] {step['field']} produced an empty tool input, skipping", flush=True)
            step["status"] = "failed"
            continue

        if wall_clock_exceeded(state):
            state["stop_reason"] = "wall_clock"
            break

        call_key = f"{step['tool']}:{arg_value.lower()}"
        cached = state["tool_call_cache"].get(call_key)

        if cached is not None:
            step["status"] = "blocked"
            already_recorded = any(
                e["field"] == step["field"] and f"{e['tool']}:{e['args'].lower()}" == call_key
                for e in state["scratchpad"]
            )
            if not already_recorded:
                state["scratchpad"].append(
                    ScratchpadEntry(
                        sub_question=step["sub_question"],
                        field=step["field"],
                        tool=step["tool"],
                        args=arg_value,
                        result=cached["result"],
                        source=cached["source"],
                    )
                )
            continue

        try:
            result = _run_tool(step["tool"], arg_value)
        except Exception as e:
            print(f"  [step] {step['field']} tool call failed: {e}", flush=True)
            step["status"] = "failed"
            continue

        if step["tool"] == "search":
            source = arg_value
        else:
            result = f"{arg_value} = {result}"
            source = f"calculator: {arg_value}"
        state["tool_call_count"] += 1
        state["tool_call_cache"][call_key] = {"result": result, "source": source}
        state["scratchpad"].append(
            ScratchpadEntry(
                sub_question=step["sub_question"],
                field=step["field"],
                tool=step["tool"],
                args=arg_value,
                result=result,
                source=source,
            )
        )
        step["status"] = "done"

    return state
