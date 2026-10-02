import time
import config
from agent.state import ResearchState


def wall_clock_exceeded(state: ResearchState) -> bool:
    return time.time() - state["start_time"] > config.MAX_WALL_CLOCK_SECONDS


def add_stop_reason(state: ResearchState, reason: str) -> None:
    current = state.get("stop_reason") or ""
    parts = current.split("+") if current else []
    if reason not in parts:
        parts.append(reason)
    state["stop_reason"] = "+".join(parts)


def stop_reasons(value: str | None) -> set[str]:
    return set(value.split("+")) if value else set()
