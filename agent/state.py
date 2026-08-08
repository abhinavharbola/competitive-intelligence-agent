from typing import TypedDict, Literal


class PlanStep(TypedDict):
    sub_question: str
    field: str
    tool: Literal["search", "calculator"]
    status: Literal["pending", "done", "blocked"]


class ScratchpadEntry(TypedDict):
    sub_question: str
    field: str
    tool: str
    args: str
    result: str
    source: str


class Critique(TypedDict):
    approved: bool
    gaps: list[str]


class ResearchState(TypedDict):
    entity: str
    plan: list[PlanStep]
    scratchpad: list[ScratchpadEntry]
    critique: Critique
    replan_count: int
    tool_call_count: int
    tool_call_log: list[str]
    start_time: float
    report: str
    field_status: dict[str, str]
    stop_reason: str
    memory_note: str
