"""Emit compact step-by-step JSON for poster/demo reporting."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def load_json(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8-sig"))


def main() -> None:
    poster_tests = load_json("poster_tests.json")
    mini = load_json("mini_benchmark.json")
    large = load_json("large_schema_benchmark.json")

    mini_methods = mini["summary_by_method"]
    large_methods = large["summary_by_method"]

    payload = {
        "steps": [
            {
                "step": 1,
                "name": "poster_tests",
                "purpose": "Validate graph policy rewrite, 100+ table schema, and benchmark improvement.",
                "status": poster_tests["status"],
                "tests_run": poster_tests["tests_run"],
                "failures": poster_tests["failures"],
                "errors": poster_tests["errors"],
            },
            {
                "step": 2,
                "name": "mini_benchmark",
                "schema": "22-table core schema",
                "cases_per_method": mini_methods["deterministic_agent"]["cases"],
                "metrics": {
                    "ours": metrics_for(mini_methods["deterministic_agent"]),
                    "agent_based": metrics_for(mini_methods["raw_graph_baseline"]),
                },
            },
            {
                "step": 3,
                "name": "large_schema_benchmark",
                "schema": large["schema"],
                "cases_per_method": large_methods["deterministic_agent"]["cases"],
                "relationship_metrics": large["relationship_metrics"],
                "metrics": {
                    "ours": metrics_for(large_methods["deterministic_agent"]),
                    "agent_based": metrics_for(large_methods["raw_graph_baseline"]),
                },
                "comparison_table": large["comparison_table"],
            },
        ]
    }
    print(json.dumps(payload, indent=2))


def metrics_for(summary: dict) -> dict:
    return {
        "join_acc": round(summary["avg_join_f1"] * 100, 1),
        "exec_acc": round(summary["result_match_accuracy"] * 100, 1),
        "em": round(summary["exact_match"] * 100, 1),
        "ves": summary["ves_like"],
        "retrieval_recall": summary["avg_retrieval_recall"],
        "validation_accuracy": summary["validation_accuracy"],
    }


if __name__ == "__main__":
    main()
