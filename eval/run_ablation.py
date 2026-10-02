import argparse
import json
import time
from pathlib import Path
from agent.graph import run
from agent.guardrails import stop_reasons
from eval.judge import score_run
import config

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results"

INFRA_FAILURE_REASONS = {
    "planner_unavailable",
    "critic_unavailable",
    "synthesizer_unavailable",
    "ablation_run_failed",
}


def load_benchmark(limit: int | None = None) -> list[dict]:
    benchmark = json.loads(BENCHMARK_PATH.read_text())
    if limit is None:
        return benchmark
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    return benchmark[:limit]


def run_condition(benchmark: list[dict], critic_enabled: bool) -> list[dict]:
    results = []
    for item in benchmark:
        entity = item["entity"]
        print(f"\n=== {entity} (critic_enabled={critic_enabled}) ===", flush=True)
        try:
            start = time.time()
            final_state = run(entity, critic_enabled=critic_enabled, use_memory=False)
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
                "groundedness_notes": judge_result["groundedness_notes"],
                "completeness": judge_result["completeness"],
                "completeness_notes": judge_result["completeness_notes"],
                "tool_call_count": final_state["tool_call_count"],
                "elapsed_seconds": elapsed,
                "replan_count": final_state["replan_count"],
                "stop_reason": final_state["stop_reason"],
            })
        except Exception as e:
            print(f"  [ablation] {entity} failed entirely, skipping: {e}", flush=True)
            results.append({
                "entity": entity, "critic_enabled": critic_enabled,
                "groundedness": None, "groundedness_notes": None,
                "completeness": None, "completeness_notes": None,
                "tool_call_count": None, "elapsed_seconds": None,
                "replan_count": None, "stop_reason": "ablation_run_failed",
            })
    return results


def _avg(results: list[dict], key: str) -> float | None:
    values = [r[key] for r in results if r[key] is not None]
    return sum(values) / len(values) if values else None


def _is_infra_failure(result: dict) -> bool:
    return bool(stop_reasons(result["stop_reason"]) & INFRA_FAILURE_REASONS)


def paired_entities(first: list[dict], second: list[dict]) -> set[str]:
    def usable(results: list[dict]) -> set[str]:
        return {r["entity"] for r in results if not _is_infra_failure(r)}
    return usable(first) & usable(second)


def summarize(results: list[dict], only: set[str] | None = None) -> dict:
    usable_all = [r for r in results if not _is_infra_failure(r)]
    excluded = [r for r in results if _is_infra_failure(r)]
    usable = [r for r in usable_all if only is None or r["entity"] in only]
    unpaired = [r["entity"] for r in usable_all if only is not None and r["entity"] not in only]
    scored = [r for r in usable if r["groundedness"] is not None]
    return {
        "avg_groundedness": _avg(usable, "groundedness"),
        "avg_completeness": _avg(usable, "completeness"),
        "avg_tool_calls": _avg(usable, "tool_call_count"),
        "avg_elapsed_seconds": _avg(usable, "elapsed_seconds"),
        "avg_replan_count": _avg(usable, "replan_count"),
        "scored_entities": len(scored),
        "total_entities": len(results),
        "excluded_infra_failures": [
            {"entity": r["entity"], "stop_reason": r["stop_reason"]} for r in excluded
        ],
        "excluded_unpaired": unpaired,
    }


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=_positive_int, default=None, help="Only run the first N benchmark entities (useful given free-tier daily quotas)")
    args = parser.parse_args()

    if not config.GROQ_JUDGE_API_KEY:
        raise SystemExit(
            "GROQ_JUDGE_API_KEY is not set. The ablation judge is a separate Groq use case "
            "from the Executor, set it in .env before running this (it can be the same key "
            "as GROQ_EXECUTOR_API_KEY if you're not on the free tier, see README)."
        )

    benchmark = load_benchmark(limit=args.limit)
    unverified = [b["entity"] for b in benchmark if not b.get("verified")]
    if unverified:
        print(f"warning: {len(unverified)} entries have unverified ground truth: {unverified}")

    print(f"Running ablation on {len(benchmark)} entities (Critic and Synthesizer run on separate Gemini models with separate free-tier daily quotas, but each still has a cap, reduce --limit if you hit RESOURCE_EXHAUSTED).")

    with_critic = run_condition(benchmark, critic_enabled=True)
    without_critic = run_condition(benchmark, critic_enabled=False)

    paired = paired_entities(with_critic, without_critic)
    summary = {
        "paired_entities": sorted(paired),
        "with_critic": summarize(with_critic, paired),
        "without_critic": summarize(without_critic, paired),
    }
    summary["delta"] = {
        k: (summary["with_critic"][k] - summary["without_critic"][k])
        if summary["with_critic"][k] is not None and summary["without_critic"][k] is not None else None
        for k in ("avg_groundedness", "avg_completeness", "avg_tool_calls", "avg_elapsed_seconds")
    }

    all_excluded = summary["with_critic"]["excluded_infra_failures"] + summary["without_critic"]["excluded_infra_failures"]
    if all_excluded:
        print(f"warning: {len(all_excluded)} run(s) excluded from summary averages due to infra failure: {all_excluded}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "with_critic.json").write_text(json.dumps(with_critic, indent=2))
    (RESULTS_DIR / "without_critic.json").write_text(json.dumps(without_critic, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
