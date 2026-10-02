from agent import llm, schemas
from agent.guardrails import add_stop_reason
from agent.state import ResearchState
import config
from tools.results import UNTRUSTED_NOTICE, is_empty_result, unwrap

SYSTEM = (
    """You are the Synthesizer for a Competitive Intelligence Agent.
Produce the final brief using ONLY the scratchpad content provided. Never introduce claims not present in the scratchpad.
You are given today's date. Use it to judge what genuinely counts as recent news and to phrase
relative time correctly ("last month", "in the past year"), never assume your own training
cutoff is the current date.
"""
    + UNTRUSTED_NOTICE
    + """
Do not include a top-level title or heading naming the entity itself, start directly with the first field section, the app renders its own title.
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
List every citation number used, one per line, in this exact format: [n] Source Title - URL
Use the title and URL exactly as they appear in the scratchpad's search results. For a finding whose source starts with "calculator:", write: [n] Calculated: followed by the expression and result exactly as shown in the scratchpad.
For a finding whose source starts with "cache,", write: [n] Cached finding from a prior run (not re-sourced this run)
Respond as JSON: {"report_markdown": str, "field_status": {"what_it_does": "confirmed"|"insufficient information", "funding_ownership": "confirmed"|"insufficient information", "recent_news": "confirmed"|"insufficient information", "competitors": "confirmed"|"insufficient information", "risks": "confirmed"|"insufficient information"}}"""
)

_FIELD_TITLES = {
    "what_it_does": "What It Does",
    "funding_ownership": "Funding & Ownership",
    "recent_news": "Recent News",
    "competitors": "Competitors",
    "risks": "Risks",
}
_FALLBACK_TEXT_LIMIT = 500


def _supported_fields(scratchpad: list) -> set[str]:
    return {e["field"] for e in scratchpad if not is_empty_result(e["result"])}


def _verified_status(raw: dict, scratchpad: list) -> dict[str, str]:
    supported = _supported_fields(scratchpad)
    cleaned = schemas.clean_field_status(raw)
    return {
        field: "confirmed" if cleaned[field] == "confirmed" and field in supported else "insufficient information"
        for field in config.REQUIRED_FIELDS
    }


def synthesize(state: ResearchState) -> ResearchState:
    scratchpad_full = "\n\n".join(
        f"field={e['field']}\nsource={e['source']}\n{e['result']}" for e in state["scratchpad"]
    )
    user = f"Entity: {state['entity']}\nToday's date: {state['today']}\nScratchpad:\n{scratchpad_full or 'empty'}"

    try:
        result = llm.call_gemini(config.SYNTHESIZER_MODEL, SYSTEM, user)
        parsed = schemas.SynthesizerResponse.model_validate(result)
        if not parsed.report_markdown.strip():
            raise ValueError("synthesizer returned an empty report")
        state["report"] = parsed.report_markdown
        state["field_status"] = _verified_status(parsed.field_status, state["scratchpad"])
    except Exception as e:
        print(f"  [synthesizer] failed after retries: {e}", flush=True)
        add_stop_reason(state, "synthesizer_unavailable")
        state["report"], state["field_status"] = _fallback_report(state)

    return state


def _fallback_report(state: ResearchState) -> tuple[str, dict[str, str]]:
    by_field: dict[str, list] = {}
    for entry in state["scratchpad"]:
        if is_empty_result(entry["result"]):
            continue
        by_field.setdefault(entry["field"], []).append(entry)

    lines = ["> Synthesizer unavailable. Raw scratchpad findings are listed below without verification.", ""]
    field_status: dict[str, str] = {}
    for field in config.REQUIRED_FIELDS:
        entries = by_field.get(field)
        lines.append(f"## {_FIELD_TITLES[field]}")
        if entries:
            for e in entries:
                text = " ".join(unwrap(e["result"]).split())[:_FALLBACK_TEXT_LIMIT]
                lines.append(f"- {text} (source: {e['source']})")
            field_status[field] = "confirmed"
        else:
            lines.append("insufficient information")
            field_status[field] = "insufficient information"
        lines.append("")

    return "\n".join(lines), field_status
