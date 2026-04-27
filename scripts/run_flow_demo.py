"""Print a poster-friendly block-by-block flow demo.

This is intentionally a script, not a unit test: it prints the query flow,
rewrite notes, final SQL, and metrics table for presentation/debugging.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text2sql_agent_prototype.large_schema_benchmark import (
    print_abstract_table,
    run_large_benchmark,
)
from text2sql_agent_prototype.prototype import (
    TextToSQLAgent,
    build_sample_database,
    build_sample_schema,
)


QUERIES = (
    "Show revenue by customer",
    "Show revenue by supplier country",
    "Show open support tickets by customer segment",
)


def main() -> None:
    schema = build_sample_schema()
    connection = build_sample_database()
    try:
        agent = TextToSQLAgent(schema, connection)
        for index, query in enumerate(QUERIES, start=1):
            trace = agent.run(query)
            print("=" * 88)
            print(f"FLOW DEMO {index}: {query}")
            print("=" * 88)
            print_block("1. Query", trace.query)
            print_block("2. Retrieval: Lexical + Embeddings", format_retrieval(trace))
            print_block("3. Candidate Tables T", ", ".join(trace.retrieval.candidate_tables))
            print_block("4. Graph Resolution G_T", format_join_plan(trace))
            print_block("5. Minimal Connecting Subgraph", format_join_edges(trace))
            print_block("6. LLM/SQL Generation", trace.generated.sql)
            print_block("7. Rewriter: DFC + Graph Join Policies", format_rewrite(trace))
            print_block("8. SQL Execution", format_execution(trace))
            print_block("9. Validation", f"{trace.validation.valid}: {trace.validation.reason}")
            print_block("10. Final SQL", trace.final_sql or "<none>")

        print("=" * 88)
        print("METRICS TABLE")
        print("=" * 88)
        print_abstract_table(run_large_benchmark())
    finally:
        connection.close()


def print_block(title: str, body: str) -> None:
    print(f"\n[{title}]")
    print(body)


def format_retrieval(trace) -> str:
    lines = []
    for match in trace.retrieval.matches[:6]:
        lines.append(
            f"- {match.table}: score={match.score}, "
            f"lexical={match.lexical_score}, semantic={match.semantic_score}, "
            f"terms={match.matched_terms}"
        )
    return "\n".join(lines)


def format_join_plan(trace) -> str:
    return "\n".join(
        [
            f"tables={trace.join_plan.tables}",
            f"unresolved={trace.join_plan.unresolved_tables}",
        ]
    )


def format_join_edges(trace) -> str:
    if not trace.join_plan.joins:
        return "No joins required."
    return "\n".join(f"- {join.join_condition()}" for join in trace.join_plan.joins)


def format_rewrite(trace) -> str:
    notes = "\n".join(f"- {note}" for note in trace.rewrite.notes)
    return "\n".join(
        [
            f"changed={trace.rewrite.changed}",
            "notes:",
            notes or "- <none>",
            "rewritten_sql:",
            trace.rewrite.sql,
        ]
    )


def format_execution(trace) -> str:
    preview = trace.execution.rows[:5]
    return "\n".join(
        [
            f"ok={trace.execution.ok}",
            f"error={trace.execution.error}",
            f"row_count={len(trace.execution.rows)}",
            f"preview={preview}",
        ]
    )


if __name__ == "__main__":
    main()
