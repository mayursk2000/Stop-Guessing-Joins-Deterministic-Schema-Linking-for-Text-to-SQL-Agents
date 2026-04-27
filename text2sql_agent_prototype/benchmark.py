"""Mini benchmark harness for deterministic schema-linking experiments."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass

from text2sql_agent_prototype.prototype import (
    ForeignKey,
    TextToSQLAgent,
    build_sample_database,
    build_sample_schema,
)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    db_id: str
    query: str
    hardness: str
    gold_sql: str
    expected_tables: tuple[str, ...]
    expected_joins: tuple[str, ...]


@dataclass
class BenchmarkResult:
    method: str
    id: str
    db_id: str
    query: str
    candidate_tables: list[str]
    plan_tables: list[str]
    plan_joins: list[str]
    missing_tables: list[str]
    extra_plan_tables: list[str]
    missing_joins: list[str]
    extra_plan_joins: list[str]
    retrieval_recall: float
    join_recall: float
    join_precision: float
    join_f1: float
    exact_match: bool
    execution_ok: bool
    gold_execution_ok: bool
    result_match: bool
    pred_ms: float
    gold_ms: float
    ves: float
    validation_ok: bool
    hardness: str
    gold_sql: str
    final_sql: str | None

    @property
    def passed(self) -> bool:
        return (
            not self.missing_tables
            and not self.missing_joins
            and self.execution_ok
            and self.gold_execution_ok
            and self.result_match
            and self.validation_ok
        )


CASES = (
    BenchmarkCase(
        id="revenue_by_customer",
        db_id="sales_ops_complex",
        query="Show revenue by customer",
        hardness="easy",
        gold_sql=(
            "SELECT customers.name, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.name"
        ),
        expected_tables=("orders", "customers"),
        expected_joins=("orders.customer_id = customers.id",),
    ),
    BenchmarkCase(
        id="revenue_by_region",
        db_id="sales_ops_complex",
        query="Show revenue by region",
        hardness="medium",
        gold_sql=(
            "SELECT regions.name, SUM(orders.amount) AS revenue "
            "FROM orders "
            "JOIN customers ON orders.customer_id = customers.id "
            "JOIN regions ON customers.region_id = regions.id "
            "GROUP BY regions.name"
        ),
        expected_tables=("orders", "customers", "regions"),
        expected_joins=(
            "orders.customer_id = customers.id",
            "customers.region_id = regions.id",
        ),
    ),
    BenchmarkCase(
        id="revenue_by_product_category",
        db_id="sales_ops_complex",
        query="Show revenue by product category",
        hardness="medium",
        gold_sql=(
            "SELECT products.category, SUM(order_items.quantity * order_items.unit_price) AS revenue "
            "FROM orders "
            "JOIN order_items ON order_items.order_id = orders.id "
            "JOIN products ON order_items.product_id = products.id "
            "GROUP BY products.category"
        ),
        expected_tables=("orders", "order_items", "products"),
        expected_joins=(
            "order_items.order_id = orders.id",
            "order_items.product_id = products.id",
        ),
    ),
    BenchmarkCase(
        id="revenue_by_supplier_country",
        db_id="sales_ops_complex",
        query="Show revenue by supplier country",
        hardness="hard",
        gold_sql=(
            "SELECT suppliers.country, SUM(order_items.quantity * order_items.unit_price) AS revenue "
            "FROM orders "
            "JOIN order_items ON order_items.order_id = orders.id "
            "JOIN products ON order_items.product_id = products.id "
            "JOIN suppliers ON products.supplier_id = suppliers.id "
            "GROUP BY suppliers.country"
        ),
        expected_tables=("orders", "order_items", "products", "suppliers"),
        expected_joins=(
            "order_items.order_id = orders.id",
            "order_items.product_id = products.id",
            "products.supplier_id = suppliers.id",
        ),
    ),
    BenchmarkCase(
        id="late_shipments_by_warehouse_region",
        db_id="sales_ops_complex",
        query="Show late shipments by warehouse region",
        hardness="medium",
        gold_sql=(
            "SELECT warehouses.name, regions.name, COUNT(shipments.id) AS shipments "
            "FROM shipments "
            "JOIN warehouses ON shipments.warehouse_id = warehouses.id "
            "JOIN regions ON warehouses.region_id = regions.id "
            "WHERE shipments.status = 'late' "
            "GROUP BY warehouses.name, regions.name"
        ),
        expected_tables=("shipments", "warehouses", "regions"),
        expected_joins=(
            "shipments.warehouse_id = warehouses.id",
            "warehouses.region_id = regions.id",
        ),
    ),
    BenchmarkCase(
        id="revenue_by_sales_rep",
        db_id="sales_ops_complex",
        query="Show revenue by sales rep",
        hardness="medium",
        gold_sql=(
            "SELECT employees.name, SUM(orders.amount) AS revenue "
            "FROM orders JOIN employees ON orders.sales_rep_id = employees.id "
            "GROUP BY employees.name"
        ),
        expected_tables=("orders", "employees"),
        expected_joins=("orders.sales_rep_id = employees.id",),
    ),
    BenchmarkCase(
        id="paid_amount_by_method",
        db_id="sales_ops_complex",
        query="Show paid amount by payment method",
        hardness="easy",
        gold_sql=(
            "SELECT payments.method, SUM(payments.amount) AS paid_amount "
            "FROM payments GROUP BY payments.method"
        ),
        expected_tables=("payments",),
        expected_joins=(),
    ),
    BenchmarkCase(
        id="returned_units_by_supplier",
        db_id="sales_ops_complex",
        query="Show returned units by supplier",
        hardness="hard",
        gold_sql=(
            "SELECT suppliers.name, SUM(returns.quantity) AS returned_units "
            "FROM returns "
            "JOIN order_items ON returns.order_item_id = order_items.id "
            "JOIN products ON order_items.product_id = products.id "
            "JOIN suppliers ON products.supplier_id = suppliers.id "
            "GROUP BY suppliers.name"
        ),
        expected_tables=("returns", "order_items", "products", "suppliers"),
        expected_joins=(
            "returns.order_item_id = order_items.id",
            "order_items.product_id = products.id",
            "products.supplier_id = suppliers.id",
        ),
    ),
    BenchmarkCase(
        id="stock_by_warehouse_and_category",
        db_id="sales_ops_complex",
        query="Show available stock by warehouse and product category",
        hardness="hard",
        gold_sql=(
            "SELECT warehouses.name, products.category, SUM(inventory.quantity_on_hand) AS stock "
            "FROM inventory "
            "JOIN warehouses ON inventory.warehouse_id = warehouses.id "
            "JOIN products ON inventory.product_id = products.id "
            "GROUP BY warehouses.name, products.category"
        ),
        expected_tables=("inventory", "warehouses", "products"),
        expected_joins=(
            "inventory.warehouse_id = warehouses.id",
            "inventory.product_id = products.id",
        ),
    ),
    BenchmarkCase(
        id="campaign_revenue_by_channel",
        db_id="sales_ops_complex",
        query="Show campaign revenue by marketing channel",
        hardness="extra",
        gold_sql=(
            "SELECT campaigns.channel, SUM(orders.amount) AS revenue "
            "FROM orders "
            "JOIN order_campaigns ON order_campaigns.order_id = orders.id "
            "JOIN campaigns ON order_campaigns.campaign_id = campaigns.id "
            "GROUP BY campaigns.channel"
        ),
        expected_tables=("orders", "order_campaigns", "campaigns"),
        expected_joins=(
            "order_campaigns.order_id = orders.id",
            "order_campaigns.campaign_id = campaigns.id",
        ),
    ),
    BenchmarkCase(
        id="open_invoice_amount_by_category",
        db_id="sales_ops_complex",
        query="Show open billed amount by product category",
        hardness="hard",
        gold_sql=(
            "SELECT products.category, SUM(invoice_lines.amount) AS billed_amount "
            "FROM invoices "
            "JOIN invoice_lines ON invoice_lines.invoice_id = invoices.id "
            "JOIN products ON invoice_lines.product_id = products.id "
            "WHERE invoices.status = 'open' "
            "GROUP BY products.category"
        ),
        expected_tables=("invoices", "invoice_lines", "products"),
        expected_joins=(
            "invoice_lines.invoice_id = invoices.id",
            "invoice_lines.product_id = products.id",
        ),
    ),
    BenchmarkCase(
        id="active_mrr_by_customer_region",
        db_id="sales_ops_complex",
        query="Show active subscription MRR by customer region",
        hardness="hard",
        gold_sql=(
            "SELECT regions.name, SUM(subscriptions.monthly_amount) AS mrr "
            "FROM subscriptions "
            "JOIN customers ON subscriptions.customer_id = customers.id "
            "JOIN regions ON customers.region_id = regions.id "
            "WHERE subscriptions.status = 'active' "
            "GROUP BY regions.name"
        ),
        expected_tables=("subscriptions", "customers", "regions"),
        expected_joins=(
            "subscriptions.customer_id = customers.id",
            "customers.region_id = regions.id",
        ),
    ),
    BenchmarkCase(
        id="open_tickets_by_segment",
        db_id="sales_ops_complex",
        query="Show open support tickets by customer segment",
        hardness="extra",
        gold_sql=(
            "SELECT customer_segments.name, COUNT(support_tickets.id) AS tickets "
            "FROM support_tickets "
            "JOIN customers ON support_tickets.customer_id = customers.id "
            "JOIN segment_members ON segment_members.customer_id = customers.id "
            "JOIN customer_segments ON segment_members.segment_id = customer_segments.id "
            "WHERE support_tickets.status = 'open' "
            "GROUP BY customer_segments.name"
        ),
        expected_tables=(
            "support_tickets",
            "customers",
            "segment_members",
            "customer_segments",
        ),
        expected_joins=(
            "support_tickets.customer_id = customers.id",
            "segment_members.customer_id = customers.id",
            "segment_members.segment_id = customer_segments.id",
        ),
    ),
)


def normalize_condition(condition: str) -> str:
    sides = sorted(part.strip().lower() for part in condition.split("="))
    return " = ".join(sides)


def normalize_join(join: ForeignKey) -> str:
    return normalize_condition(join.join_condition())


def canonical_rows(rows: list[dict[str, object]]) -> list[str]:
    return sorted(json.dumps(row, sort_keys=True) for row in rows)


def normalize_sql(sql: str | None) -> str:
    if not sql:
        return ""
    return re.sub(r"\s+", " ", sql.strip().lower())


def execute_timed(agent: TextToSQLAgent, sql: str):
    started = time.perf_counter()
    execution = agent.executor.execute(sql)
    elapsed_ms = max((time.perf_counter() - started) * 1000, 0.001)
    return execution, elapsed_ms


def ves_like(result_match: bool, pred_ms: float, gold_ms: float) -> float:
    if not result_match:
        return 0.0
    return round(min((gold_ms / max(pred_ms, 0.001)) ** 0.5, 2.0), 3)


def run_case(agent: TextToSQLAgent, method: str, case: BenchmarkCase) -> BenchmarkResult:
    trace = agent.run(case.query)
    pred_execution, pred_ms = execute_timed(agent, trace.final_sql or "SELECT 1")
    gold_execution, gold_ms = execute_timed(agent, case.gold_sql)

    expected_tables = set(case.expected_tables)
    candidate_tables = set(trace.retrieval.candidate_tables)
    plan_tables = set(trace.join_plan.tables)
    expected_joins = {normalize_condition(join) for join in case.expected_joins}
    plan_joins = {normalize_join(join) for join in trace.join_plan.joins}

    missing_tables = sorted(expected_tables - plan_tables)
    extra_plan_tables = sorted(plan_tables - expected_tables)
    missing_joins = sorted(expected_joins - plan_joins)
    extra_plan_joins = sorted(plan_joins - expected_joins)
    retrieval_recall = len(expected_tables & candidate_tables) / len(expected_tables)
    join_recall = (
        1.0
        if not expected_joins
        else len(expected_joins & plan_joins) / len(expected_joins)
    )
    join_precision = (
        1.0 if not plan_joins else len(expected_joins & plan_joins) / len(plan_joins)
    )
    join_f1 = (
        0.0
        if join_recall + join_precision == 0
        else (2 * join_recall * join_precision) / (join_recall + join_precision)
    )
    result_match = (
        pred_execution.ok
        and gold_execution.ok
        and canonical_rows(pred_execution.rows) == canonical_rows(gold_execution.rows)
    )

    return BenchmarkResult(
        method=method,
        id=case.id,
        db_id=case.db_id,
        query=case.query,
        candidate_tables=trace.retrieval.candidate_tables,
        plan_tables=trace.join_plan.tables,
        plan_joins=sorted(plan_joins),
        missing_tables=missing_tables,
        extra_plan_tables=extra_plan_tables,
        missing_joins=missing_joins,
        extra_plan_joins=extra_plan_joins,
        retrieval_recall=round(retrieval_recall, 3),
        join_recall=round(join_recall, 3),
        join_precision=round(join_precision, 3),
        join_f1=round(join_f1, 3),
        exact_match=normalize_sql(trace.final_sql) == normalize_sql(case.gold_sql),
        execution_ok=pred_execution.ok,
        gold_execution_ok=gold_execution.ok,
        result_match=result_match,
        pred_ms=round(pred_ms, 3),
        gold_ms=round(gold_ms, 3),
        ves=ves_like(result_match, pred_ms, gold_ms),
        validation_ok=trace.validation.valid,
        hardness=case.hardness,
        gold_sql=case.gold_sql,
        final_sql=trace.final_sql,
    )


def run_method(method: str, use_pruning: bool) -> list[BenchmarkResult]:
    schema = build_sample_schema()
    connection = build_sample_database()
    try:
        agent = TextToSQLAgent(schema, connection, use_pruning=use_pruning)
        return [run_case(agent, method, case) for case in CASES]
    finally:
        connection.close()


def run_benchmark() -> list[BenchmarkResult]:
    return [
        *run_method("raw_graph_baseline", use_pruning=False),
        *run_method("deterministic_agent", use_pruning=True),
    ]


def summarize(results: list[BenchmarkResult]) -> dict[str, float | int]:
    total = len(results)
    return {
        "cases": total,
        "passed": sum(result.passed for result in results),
        "avg_retrieval_recall": round(
            sum(result.retrieval_recall for result in results) / total, 3
        ),
        "avg_join_recall": round(sum(result.join_recall for result in results) / total, 3),
        "avg_join_precision": round(
            sum(result.join_precision for result in results) / total, 3
        ),
        "avg_join_f1": round(sum(result.join_f1 for result in results) / total, 3),
        "exact_match": round(sum(result.exact_match for result in results) / total, 3),
        "execution_accuracy": round(
            sum(result.execution_ok for result in results) / total, 3
        ),
        "result_match_accuracy": round(
            sum(result.result_match for result in results) / total, 3
        ),
        "ves_like": round(sum(result.ves for result in results) / total, 3),
        "validation_accuracy": round(
            sum(result.validation_ok for result in results) / total, 3
        ),
    }


def summarize_by_method(results: list[BenchmarkResult]) -> dict[str, dict[str, float | int]]:
    methods = sorted({result.method for result in results})
    return {
        method: summarize([result for result in results if result.method == method])
        for method in methods
    }


def summarize_by_hardness(results: list[BenchmarkResult]) -> dict[str, dict[str, float | int]]:
    groups = sorted({result.hardness for result in results})
    return {
        hardness: summarize([result for result in results if result.hardness == hardness])
        for hardness in groups
    }


def print_case_table(results: list[BenchmarkResult]) -> None:
    print("method               id                                  hard   EM   EX   VES   joinF")
    print("------               --                                  ----   --   --   ---   -----")
    for result in results:
        print(
            f"{result.method:<20} "
            f"{result.id:<35} "
            f"{result.hardness:<6} "
            f"{'yes' if result.exact_match else 'no ':<4} "
            f"{'yes' if result.result_match else 'no ':<4} "
            f"{result.ves:<5.2f} "
            f"{result.join_f1:<5.2f}"
        )


def print_summary_table(summary: dict[str, dict[str, float | int]]) -> None:
    print("method               n   EM    EX    VES   joinF  retR  valid")
    print("------               -   --    --    ---   -----  ----  -----")
    for method, metrics in summary.items():
        print(
            f"{method:<20} "
            f"{metrics['cases']:<3} "
            f"{metrics['exact_match']:<5.2f} "
            f"{metrics['result_match_accuracy']:<5.2f} "
            f"{metrics['ves_like']:<5.2f} "
            f"{metrics['avg_join_f1']:<6.2f} "
            f"{metrics['avg_retrieval_recall']:<5.2f} "
            f"{metrics['validation_accuracy']:<5.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mini schema-linking benchmark.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON results.")
    parser.add_argument(
        "--cases", action="store_true", help="Print per-case comparison table."
    )
    args = parser.parse_args()

    results = run_benchmark()
    if args.json:
        print(
            json.dumps(
                {
                    "summary_by_method": summarize_by_method(results),
                    "summary_by_hardness": summarize_by_hardness(results),
                    "results": [asdict(result) | {"passed": result.passed} for result in results],
                },
                indent=2,
            )
        )
        return

    print_summary_table(summarize_by_method(results))
    if args.cases:
        print()
        print_case_table(results)
    print()
    print(json.dumps({"by_hardness": summarize_by_hardness(results)}, indent=2))


if __name__ == "__main__":
    main()
