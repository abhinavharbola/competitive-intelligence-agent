import time
from agent import llm, schemas
from agent.state import ResearchState, ScratchpadEntry
from tools.search import web_search
from tools.calculator import calculate
import config

SYSTEM = """You are the Executor for a Competitive Intelligence Agent.
Given one research sub-question and its assigned tool, produce the exact tool input.
Respond as JSON. For "search" respond {"query": str}. For "calculator" respond {"expression": str}.
Keep search queries specific and short. You are given today's date, use it to make a
recency-sensitive query concrete (an actual month/year) instead of relying on the word "recent"
alone, and never assume your own training cutoff is the current date."""


def _run_tool(tool: str, arg_value: str) -> str:
    return web_search(arg_value) if tool == "search" else calculate(arg_value)


def execute(state: ResearchState) -> ResearchState:
    for step in state["plan"]:
        if step["status"] != "pending":
            continue
        if state["tool_call_count"] >= config.MAX_TOOL_CALLS:
            state["stop_reason"] = "max_tool_calls"
            break
        if time.time() - state["start_time"] > config.MAX_WALL_CLOCK_SECONDS:
            state["stop_reason"] = "wall_clock"
            break

        print(f"  [step] {step['field']}: {step['sub_question']}", flush=True)

        try:
            args_response = llm.call_executor(
                SYSTEM,
                f"Today's date: {state['today']}\nSub-question: {step['sub_question']}\nTool: {step['tool']}",
            )
            args = schemas.ExecutorArgs.model_validate(args_response)
            arg_value = (args.query or args.expression).strip()
        except Exception as e:
            print(f"  [step] {step['field']} failed to produce tool args: {e}", flush=True)
            step["status"] = "failed"
            continue

        if not arg_value:
            print(f"  [step] {step['field']} produced an empty tool input, skipping", flush=True)
            step["status"] = "failed"
            continue

        call_key = f"{step['tool']}:{arg_value.lower()}"
        cached = state["tool_call_cache"].get(call_key)

        if cached is not None:
            step["status"] = "blocked"
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

        source = arg_value if step["tool"] == "search" else "calculator"
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
