from fastapi import FastAPI, HTTPException
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
    if not request.entity.strip():
        raise HTTPException(status_code=400, detail="entity must not be empty")

    try:
        final_state = run(request.entity)
    except Exception as e:
        print(f"  [api] research failed for {request.entity!r}: {e}", flush=True)
        raise HTTPException(status_code=500, detail="research run failed, please try again")

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