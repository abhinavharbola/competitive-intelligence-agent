import argparse
import json
import time
from pathlib import Path
from agent.graph import run
from eval.judge import score_run

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_benchmark(limit: int | None = None) -> list[dict]:
    benchmark = json.loads(BENCHMARK_PATH.read_text())
    return benchmark[:limit] if limit else benchmark


def run_condition(benchmark: list[dict], critic_enabled: bool) -> list[dict]:
    results = []
    for item in benchmark:
        entity = item["entity"]
        print(f"\n=== {entity} (critic_enabled={critic_enabled}) ===", flush=True)
        try:
            start = time.time()
            final_state = run(entity, critic_enabled=critic_enabled)
            elapsed = time.time() - start

            judge_result = score_run(
                entity=entity,
                ground_truth=item["ground_truth"],
                report=final_state["report"],
                field_status=final_state["field_status"],
                scratchpad=final_state["scratchpad"],
            )

            results.append({
                "entity": entity,
                "critic_enabled": critic_enabled,
                "groundedness": judge_result["groundedness"],
                "completeness": judge_result["completeness"],
                "tool_call_count": final_state["tool_call_count"],
                "elapsed_seconds": elapsed,
                "replan_count": final_state["replan_count"],
                "stop_reason": final_state["stop_reason"],
            })
        except Exception as e:
            print(f"  [ablation] {entity} failed entirely, skipping: {e}", flush=True)
            results.append({
                "entity": entity, "critic_enabled": critic_enabled,
                "groundedness": None, "completeness": None,
                "tool_call_count": None, "elapsed_seconds": None,
                "replan_count": None, "stop_reason": "ablation_run_failed",
            })
    return results


def _avg(results: list[dict], key: str) -> float | None:
    values = [r[key] for r in results if r[key] is not None]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]) -> dict:
    scored = [r for r in results if r["groundedness"] is not None]
    return {
        "avg_groundedness": _avg(results, "groundedness"),
        "avg_completeness": _avg(results, "completeness"),
        "avg_tool_calls": _avg(results, "tool_call_count"),
        "avg_elapsed_seconds": _avg(results, "elapsed_seconds"),
        "scored_entities": len(scored),
        "total_entities": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N benchmark entities (useful given free-tier daily quotas)")
    args = parser.parse_args()

    benchmark = load_benchmark(limit=args.limit)
    unverified = [b["entity"] for b in benchmark if not b.get("verified")]
    if unverified:
        print(f"warning: {len(unverified)} entries have unverified ground truth: {unverified}")

    print(f"Running ablation on {len(benchmark)} entities (Critic and Synthesizer share Gemini's free-tier daily quota, currently 20 requests/day/model — reduce --limit if you hit RESOURCE_EXHAUSTED).")

    with_critic = run_condition(benchmark, critic_enabled=True)
    without_critic = run_condition(benchmark, critic_enabled=False)

    summary = {
        "with_critic": summarize(with_critic),
        "without_critic": summarize(without_critic),
    }
    summary["delta"] = {
        k: (summary["with_critic"][k] - summary["without_critic"][k])
        if summary["with_critic"][k] is not None and summary["without_critic"][k] is not None else None
        for k in ("avg_groundedness", "avg_completeness", "avg_tool_calls", "avg_elapsed_seconds")
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "with_critic.json").write_text(json.dumps(with_critic, indent=2))
    (RESULTS_DIR / "without_critic.json").write_text(json.dumps(without_critic, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
