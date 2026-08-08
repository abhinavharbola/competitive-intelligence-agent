from agent import llm
from agent.state import ResearchState
import config

SYSTEM = """You are the Synthesizer for a Competitive Intelligence Agent.
Produce the final brief using ONLY the scratchpad content provided. Never introduce claims not present in the scratchpad.
Do not include a top-level title or heading naming the entity itself — start directly with the first field section, the app renders its own title.
Cover exactly these fields, in this order, each as a level-2 markdown heading with this exact wording:
## What It Does
## Funding & Ownership
## Recent News
## Competitors
## Risks
For any field with no supporting scratchpad entry, write "insufficient information" under that heading instead of guessing.
Cite every claim with a numbered bracket marker, e.g. [1], placed immediately after the sentence it supports. Reuse the same number for the same source if it supports more than one claim. Do not use inline markdown links or embed source names inside sentences.
After the 5 field sections, add one more section:
## References
List every citation number used, one per line, in this exact format: [n] Source Title — URL
Use the title and URL exactly as they appear in the scratchpad's search results. For a calculator-derived figure, write: [n] Calculated from scratchpad figures
Respond as JSON: {"report_markdown": str, "field_status": {"what_it_does": "confirmed"|"insufficient information", "funding_ownership": "confirmed"|"insufficient information", "recent_news": "confirmed"|"insufficient information", "competitors": "confirmed"|"insufficient information", "risks": "confirmed"|"insufficient information"}}"""


def synthesize(state: ResearchState) -> ResearchState:
    scratchpad_full = "\n\n".join(
        f"field={e['field']}\nsource={e['source']}\n{e['result']}" for e in state["scratchpad"]
    )
    user = f"Entity: {state['entity']}\nScratchpad:\n{scratchpad_full or 'empty'}"

    try:
        result = llm.call_gemini(config.SYNTHESIZER_MODEL, SYSTEM, user)
        state["report"] = result["report_markdown"]
        state["field_status"] = result["field_status"]
    except Exception as e:
        print(f"  [synthesizer] failed after retries: {e}", flush=True)
        state["stop_reason"] = state["stop_reason"] or "synthesizer_unavailable"
        state["report"], state["field_status"] = _fallback_report(state)

    return state


def _fallback_report(state: ResearchState) -> tuple[str, dict[str, str]]:
    by_field: dict[str, list] = {}
    for entry in state["scratchpad"]:
        by_field.setdefault(entry["field"], []).append(entry)

    lines = [f"# {state['entity']} (auto-generated, synthesizer unavailable)", ""]
    field_status: dict[str, str] = {}
    for field in config.REQUIRED_FIELDS:
        entries = by_field.get(field)
        lines.append(f"## {field}")
        if entries:
            for e in entries:
                lines.append(f"- {e['result']} (source: {e['source']})")
            field_status[field] = "confirmed"
        else:
            lines.append("insufficient information")
            field_status[field] = "insufficient information"
        lines.append("")

    return "\n".join(lines), field_status
