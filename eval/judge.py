from agent import llm

SYSTEM = """You are the evaluation judge for a Competitive Intelligence Agent benchmark.
Score a single research run against manually-verified ground truth.
groundedness (0-5): does every claim in the report trace back to a scratchpad finding? Penalize unsourced or fabricated claims. You are given each scratchpad finding's source and its actual content, check claims against the content, not just the presence of a source label.
completeness (0-5): are all 5 fields (what_it_does, funding_ownership, recent_news, competitors, risks) filled with information matching ground truth, or correctly marked "insufficient information" when the scratchpad had nothing relevant?
Respond as JSON with exactly these four keys, no others: {"groundedness": int, "groundedness_notes": str, "completeness": int, "completeness_notes": str}
groundedness_notes and completeness_notes should each be one or two sentences explaining the score, specific enough that someone reading only the notes (not the report) understands why the score landed where it did."""


def score_run(entity: str, ground_truth: dict, report: str, field_status: dict, scratchpad: list[dict]) -> dict:
    findings = "\n".join(
        f"- field={e['field']} source={e['source']}\n  {e['result']}" for e in scratchpad
    )
    user = (
        f"Entity: {entity}\n"
        f"Ground truth: {ground_truth}\n"
        f"Field status reported by agent: {field_status}\n"
        f"Scratchpad findings used (source + content):\n{findings or 'none'}\n\n"
        f"Final report:\n{report}"
    )

    empty_result = {
        "groundedness": None, "groundedness_notes": None,
        "completeness": None, "completeness_notes": None,
    }

    try:
        raw = llm.call_judge(SYSTEM, user)
    except Exception as e:
        print(f"  [judge] call failed for {entity}: {e}", flush=True)
        return empty_result

    result = {}
    for key in ("groundedness", "completeness"):
        value = raw.get(key)
        try:
            result[key] = int(value)
        except (TypeError, ValueError):
            print(f"  [judge] missing/invalid '{key}' in response for {entity}: {raw}", flush=True)
            result[key] = None

    for key in ("groundedness_notes", "completeness_notes"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
        else:
            print(f"  [judge] missing/invalid '{key}' in response for {entity}: {raw}", flush=True)
            result[key] = None

    return result