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
    approved: bool | None = None
    gaps: list[str]


class ExecutorArgs(BaseModel):
    query: str = ""
    expression: str = ""


class SynthesizerResponse(BaseModel):
    report_markdown: str
    field_status: dict[str, str]


def valid_planner_steps(raw: dict) -> list[PlanStepOut]:
    parsed = PlannerResponse.model_validate(raw)
    steps = []
    for step in parsed.steps:
        if step.field not in config.REQUIRED_FIELDS:
            print(f"  [planner] dropping step with unknown field {step.field!r}", flush=True)
            continue
        steps.append(step)
    return steps


def clean_field_status(raw: dict) -> dict[str, str]:
    return {
        field: "confirmed" if raw.get(field) == "confirmed" else "insufficient information"
        for field in config.REQUIRED_FIELDS
    }
