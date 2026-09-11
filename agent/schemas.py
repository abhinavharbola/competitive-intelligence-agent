from typing import Literal
from pydantic import BaseModel

import config


class PlanStepOut(BaseModel):
    sub_question: str
    field: str
    tool: Literal["search", "calculator"]


class PlannerResponse(BaseModel):
    steps: list[PlanStepOut]


class CriticResponse(BaseModel):
    approved: bool
    gaps: list[str]


class ExecutorArgs(BaseModel):
    query: str = ""
    expression: str = ""


FieldStatusValue = Literal["confirmed", "insufficient information"]


class SynthesizerResponse(BaseModel):
    report_markdown: str
    field_status: dict[str, FieldStatusValue]


def valid_planner_steps(raw: dict) -> list[PlanStepOut]:
    """Validate the planner's raw JSON and silently drop any step whose field
    isn't one of the five canonical fields, rather than failing the whole
    response over one bad key.
    """
    parsed = PlannerResponse.model_validate(raw)
    steps = []
    for step in parsed.steps:
        if step.field not in config.REQUIRED_FIELDS:
            print(f"  [planner] dropping step with unknown field {step.field!r}", flush=True)
            continue
        steps.append(step)
    return steps
