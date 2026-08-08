import time
from agent import llm
from agent.state import ResearchState, ScratchpadEntry
from tools.search import web_search
from tools.calculator import calculate
import config

SYSTEM = """You are the Executor for a Competitive Intelligence Agent.
Given one research sub-question and its assigned tool, produce the exact tool input.
Respond as JSON. For "search" respond {"query": str}. For "calculator" respond {"expression": str}.
Keep search queries specific and short."""


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
                SYSTEM, f"Sub-question: {step['sub_question']}\nTool: {step['tool']}"
            )
            arg_value = args_response.get("query") or args_response.get("expression", "")
            call_key = f"{step['tool']}:{arg_value.strip().lower()}"

            if call_key in state["tool_call_log"]:
                step["status"] = "blocked"
                continue

            result = web_search(arg_value) if step["tool"] == "search" else calculate(arg_value)
        except Exception as e:
            print(f"  [step] {step['field']} failed: {e}", flush=True)
            step["status"] = "blocked"
            continue

        state["tool_call_log"].append(call_key)
        state["tool_call_count"] += 1
        state["scratchpad"].append(
            ScratchpadEntry(
                sub_question=step["sub_question"],
                field=step["field"],
                tool=step["tool"],
                args=arg_value,
                result=result,
                source=arg_value if step["tool"] == "search" else "calculator",
            )
        )
        step["status"] = "done"

    return state
