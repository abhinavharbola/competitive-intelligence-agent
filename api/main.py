import hmac
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import config
from agent.graph import run

MAX_ENTITY_LENGTH = 200

app = FastAPI(title="Competitive Intelligence Agent")


def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = config.API_ACCESS_KEY
    if expected and not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class ResearchRequest(BaseModel):
    entity: str = Field(max_length=MAX_ENTITY_LENGTH)


class ResearchResponse(BaseModel):
    entity: str
    report: str
    field_status: dict
    stop_reason: str
    replan_count: int
    tool_call_count: int
    scratchpad: list
    memory_note: str


@app.post("/research", response_model=ResearchResponse, dependencies=[Depends(require_api_key)])
def research(request: ResearchRequest) -> ResearchResponse:
    entity = request.entity.strip()
    if not entity:
        raise HTTPException(status_code=400, detail="entity must not be empty")

    try:
        final_state = run(entity)
    except Exception as e:
        print(f"  [api] research failed for {entity!r}: {e}", flush=True)
        raise HTTPException(status_code=500, detail="research run failed, please try again")

    return ResearchResponse(
        entity=entity,
        report=final_state["report"],
        field_status=final_state["field_status"],
        stop_reason=final_state["stop_reason"],
        replan_count=final_state["replan_count"],
        tool_call_count=final_state["tool_call_count"],
        scratchpad=final_state["scratchpad"],
        memory_note=final_state["memory_note"],
    )
