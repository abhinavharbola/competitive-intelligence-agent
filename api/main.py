from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import run

app = FastAPI(title="Competitive Intelligence Agent")


class ResearchRequest(BaseModel):
    entity: str


class ResearchResponse(BaseModel):
    entity: str
    report: str
    field_status: dict
    stop_reason: str
    replan_count: int
    tool_call_count: int
    scratchpad: list
    memory_note: str


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    final_state = run(request.entity)
    return ResearchResponse(
        entity=request.entity,
        report=final_state["report"],
        field_status=final_state["field_status"],
        stop_reason=final_state["stop_reason"],
        replan_count=final_state["replan_count"],
        tool_call_count=final_state["tool_call_count"],
        scratchpad=final_state["scratchpad"],
        memory_note=final_state["memory_note"],
    )
