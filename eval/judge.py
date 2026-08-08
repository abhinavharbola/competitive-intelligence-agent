from agent import llm

SYSTEM = """You are the evaluation judge for a Competitive Intelligence Agent benchmark.
Score a single research run against manually-verified ground truth.
groundedness (0-5): does every claim in the report trace back to a scratchpad source? Penalize unsourced or fabricated claims.
completeness (0-5): are all 5 fields (what_it_does, funding_ownership, recent_news, competitors, risks) filled with information matching ground truth, or correctly marked "insufficient information" when the scratchpad had nothing relevant?
Respond as JSON: {"groundedness": int, "groundedness_notes": str, "completeness": int, "completeness_notes": str}"""


def score_run(entity: str, ground_truth: dict, report: str, field_status: dict, scratchpad: list[dict]) -> dict:
    sources = "\n".join(f"- field={e['field']} source={e['source']}" for e in scratchpad)
    user = (
        f"Entity: {entity}\n"
        f"Ground truth: {ground_truth}\n"
        f"Field status reported by agent: {field_status}\n"
        f"Scratchpad sources used:\n{sources or 'none'}\n\n"
        f"Final report:\n{report}"
    )
    return llm.call_judge(SYSTEM, user)
