import json
import time
from pathlib import Path
from agent.graph import run
from eval.judge import score_run

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_benchmark() -> list[dict]:
    return json.loads(BENCHMARK_PATH.read_text())


def run_condition(benchmark: list[dict], critic_enabled: bool) -> list[dict]:
    results = []
    for item in benchmark:
        entity = item["entity"]
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
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    return {
        "avg_groundedness": sum(r["groundedness"] for r in results) / n,
        "avg_completeness": sum(r["completeness"] for r in results) / n,
        "avg_tool_calls": sum(r["tool_call_count"] for r in results) / n,
        "avg_elapsed_seconds": sum(r["elapsed_seconds"] for r in results) / n,
    }


def main():
    benchmark = load_benchmark()
    unverified = [b["entity"] for b in benchmark if not b.get("verified")]
    if unverified:
        print(f"warning: {len(unverified)} entries have unverified ground truth: {unverified}")

    with_critic = run_condition(benchmark, critic_enabled=True)
    without_critic = run_condition(benchmark, critic_enabled=False)

    summary = {
        "with_critic": summarize(with_critic),
        "without_critic": summarize(without_critic),
    }
    summary["delta"] = {
        k: summary["with_critic"][k] - summary["without_critic"][k]
        for k in summary["with_critic"]
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "with_critic.json").write_text(json.dumps(with_critic, indent=2))
    (RESULTS_DIR / "without_critic.json").write_text(json.dumps(without_critic, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
