import time
import config
from agent.state import ResearchState


def wall_clock_exceeded(state: ResearchState) -> bool:
    return time.time() - state["start_time"] > config.MAX_WALL_CLOCK_SECONDS
