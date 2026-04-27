"""Large-schema benchmark with 100+ tables and paper-style metrics.

This module builds on the realistic core schema from ``prototype.py`` and adds
programmatic distractor tables to mimic Spider/BIRD-style large schemas:

- many tables
- shared hubs
- bridge tables
- near-miss schema names
- multi-hop paths

It stays dependency-free and uses SQLite in memory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict

from text2sql_agent_prototype.benchmark import (
    CASES,
    BenchmarkCase,
    BenchmarkResult,
    run_case,
    summarize_by_hardness,
    summarize_by_method,
)
from text2sql_agent_prototype.prototype import (
    Column,
    ForeignKey,
    Schema,
    Table,
    TextToSQLAgent,
    build_sample_database,
    build_sample_schema,
)


NOISE_GROUPS = (
    ("customer_profile", "customers", "customer_id", "customer account profile segment"),
    ("customer_score", "customers", "customer_id", "customer risk score cohort"),
    ("order_audit", "orders", "order_id", "order transaction audit revenue amount"),
    ("order_status_event", "orders", "order_id", "order status event history"),
    ("product_metric", "products", "product_id", "product category metric inventory"),
    ("product_alias", "products", "product_id", "product sku alias catalog"),
    ("region_quota", "regions", "region_id", "region sales quota territory"),
    ("warehouse_capacity", "warehouses", "warehouse_id", "warehouse capacity stock"),
    ("invoice_adjustment", "invoices", "invoice_id", "invoice billing adjustment amount"),
    ("support_note", "support_tickets", "ticket_id", "support ticket note event"),
)


def build_large_schema(tables_per_group: int = 10) -> Schema:
    base = build_sample_schema()
    tables = dict(base.tables)
    foreign_keys = list(base.foreign_keys)

    for prefix, parent_table, fk_column, description in NOISE_GROUPS:
        for index in range(1, tables_per_group + 1):
            table_name = f"{prefix}_{index:02d}"
            tables[table_name] = Table(
                name=table_name,
                description=f"{description} auxiliary distractor table {index}",
                aliases=(prefix.replace("_", " "), "auxiliary", "distractor"),
                columns=(
                    Column("id", "primary key"),
                    Column(fk_column, f"{parent_table} foreign key"),
                    Column("label", "descriptive label"),
                    Column("metric_value", "numeric metric value"),
                ),
            )
            foreign_keys.append(ForeignKey(table_name, fk_column, parent_table, "id"))

    return Schema(tables=tables, foreign_keys=foreign_keys)


def build_large_database(tables_per_group: int = 10) -> sqlite3.Connection:
    conn = build_sample_database()

    for prefix, parent_table, fk_column, _description in NOISE_GROUPS:
        parent_ids = [row[0] for row in conn.execute(f"SELECT id FROM {parent_table}")]
        for index in range(1, tables_per_group + 1):
            table_name = f"{prefix}_{index:02d}"
            conn.execute(
                f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY,
                    {fk_column} INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY ({fk_column}) REFERENCES {parent_table}(id)
                )
                """
            )
            rows = [
                (row_index, parent_id, f"{table_name}-{parent_id}", float(row_index + index))
                for row_index, parent_id in enumerate(parent_ids, start=1)
            ]
            conn.executemany(
                f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)",
                rows,
            )

    return conn


LARGE_CASES: tuple[BenchmarkCase, ...] = CASES + (
    BenchmarkCase(
        id="large_schema_revenue_by_customer",
        db_id="sales_ops_122_table",
        query="Show revenue by customer in the large warehouse schema",
        hardness="medium",
        gold_sql=(
            "SELECT customers.name, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.name"
        ),
        expected_tables=("orders", "customers"),
        expected_joins=("orders.customer_id = customers.id",),
    ),
    BenchmarkCase(
        id="large_schema_open_tickets_by_segment",
        db_id="sales_ops_122_table",
        query="Show open support tickets by customer segment in the large schema",
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
    BenchmarkCase(
        id="large_schema_billed_category",
        db_id="sales_ops_122_table",
        query="Show open billed amount by product category in the large schema",
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
        id="large_schema_revenue_by_region",
        db_id="sales_ops_122_table",
        query="Show revenue by region in the large enterprise schema",
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
        id="large_schema_revenue_by_supplier_country",
        db_id="sales_ops_122_table",
        query="Show revenue by supplier country in the large catalog schema",
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
        id="large_schema_late_shipments_by_region",
        db_id="sales_ops_122_table",
        query="Show late shipments by warehouse region in the large logistics schema",
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
        id="large_schema_stock_by_category",
        db_id="sales_ops_122_table",
        query="Show available stock by warehouse and product category in the large schema",
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
        id="large_schema_returned_units_by_supplier",
        db_id="sales_ops_122_table",
        query="Show returned units by supplier in the large commerce schema",
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
        id="large_schema_campaign_revenue",
        db_id="sales_ops_122_table",
        query="Show campaign revenue by marketing channel in the large schema",
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
        id="large_schema_active_mrr_by_region",
        db_id="sales_ops_122_table",
        query="Show active subscription MRR by customer region in the large schema",
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
        id="large_schema_payment_method",
        db_id="sales_ops_122_table",
        query="Show paid amount by payment method in the large finance schema",
        hardness="easy",
        gold_sql=(
            "SELECT payments.method, SUM(payments.amount) AS paid_amount "
            "FROM payments GROUP BY payments.method"
        ),
        expected_tables=("payments",),
        expected_joins=(),
    ),
)


def run_large_method(
    method: str,
    use_pruning: bool,
    tables_per_group: int = 10,
    cases: tuple[BenchmarkCase, ...] = LARGE_CASES,
    top_k: int = 12,
) -> list[BenchmarkResult]:
    schema = build_large_schema(tables_per_group=tables_per_group)
    connection = build_large_database(tables_per_group=tables_per_group)
    try:
        agent = TextToSQLAgent(schema, connection, use_pruning=use_pruning)
        agent.retriever.top_k = top_k
        return [run_case(agent, method, case) for case in cases]
    finally:
        connection.close()


def run_large_benchmark(tables_per_group: int = 10) -> list[BenchmarkResult]:
    return [
        *run_large_method(
            "raw_graph_baseline",
            use_pruning=False,
            tables_per_group=tables_per_group,
        ),
        *run_large_method(
            "deterministic_agent",
            use_pruning=True,
            tables_per_group=tables_per_group,
        ),
    ]


def abstract_table_rows(results: list[BenchmarkResult]) -> list[dict[str, float | str]]:
    summary = summarize_by_method(results)
    labels = {
        "raw_graph_baseline": "Agent-based",
        "deterministic_agent": "Ours",
    }
    rows = []
    for method, metrics in summary.items():
        method_results = [result for result in results if result.method == method]
        rows.append(
            {
                "Method": labels.get(method, method),
                "Join Acc.": round(float(metrics["avg_join_f1"]) * 100, 1),
                "Exec. Acc.": round(float(metrics["result_match_accuracy"]) * 100, 1),
                "EM": round(float(metrics["exact_match"]) * 100, 1),
                "VES": round(float(metrics["ves_like"]), 2),
                "Cases": int(metrics["cases"]),
                "Avg Joins": round(avg_gold_join_count(method_results), 1),
                "Max Joins": max_gold_join_count(method_results),
                "Tables": 122,
                "FKs": 126,
            }
        )
    return rows


def avg_gold_join_count(results: list[BenchmarkResult]) -> float:
    counts = [len(result.gold_sql.lower().split(" join ")) - 1 for result in results]
    return sum(counts) / len(counts)


def max_gold_join_count(results: list[BenchmarkResult]) -> int:
    return max(len(result.gold_sql.lower().split(" join ")) - 1 for result in results)


def relationship_metrics(results: list[BenchmarkResult]) -> dict[str, float | int]:
    expected_join_counts = [
        len(result.gold_sql.lower().split(" join ")) - 1
        for result in results
    ]
    return {
        "cases": len(results),
        "avg_gold_joins": round(avg_gold_join_count(results), 2),
        "max_gold_joins": max_gold_join_count(results),
        "multi_join_cases": sum(count >= 2 for count in expected_join_counts),
    }


REFERENCE_ROWS: tuple[dict[str, float | str], ...] = (
    {
        "Method": "Spider original best",
        "Join Acc.": "-",
        "Exec. Acc.": "-",
        "EM": 12.4,
        "VES": "-",
        "Cases": "10,181 Q",
        "Avg Joins": "-",
        "Max Joins": "-",
        "Tables": "200 DBs",
        "FKs": "-",
    },
    {
        "Method": "BIRD GPT-4 + EK",
        "Join Acc.": "-",
        "Exec. Acc.": 54.9,
        "EM": "-",
        "VES": "-",
        "Cases": "12,751 Q",
        "Avg Joins": "-",
        "Max Joins": "-",
        "Tables": "95 DBs",
        "FKs": "-",
    },
    {
        "Method": "BIRD human",
        "Join Acc.": "-",
        "Exec. Acc.": 93.0,
        "EM": "-",
        "VES": "-",
        "Cases": "-",
        "Avg Joins": "-",
        "Max Joins": "-",
        "Tables": "95 DBs",
        "FKs": "-",
    },
)


def comparison_rows(results: list[BenchmarkResult]) -> list[dict[str, float | str]]:
    return [*REFERENCE_ROWS, *abstract_table_rows(results)]


def print_abstract_table(results: list[BenchmarkResult]) -> None:
    print("Method                Join Acc.  Exec. Acc.  EM     VES   Cases     AvgJ  MaxJ  Tables   FKs")
    print("------                ---------  ----------  ----   ---   -----     ----  ----  ------   ---")
    for row in comparison_rows(results):
        print(
            f"{row['Method']:<21} "
            f"{format_metric(row['Join Acc.'], 8)}   "
            f"{format_metric(row['Exec. Acc.'], 8)}   "
            f"{format_metric(row['EM'], 4)}   "
            f"{format_metric(row['VES'], 4)}  "
            f"{format_metric(row['Cases'], 8)}  "
            f"{format_metric(row['Avg Joins'], 4)}  "
            f"{format_metric(row['Max Joins'], 4)}  "
            f"{format_metric(row['Tables'], 6)}  "
            f"{format_metric(row['FKs'], 4)}"
        )


def format_metric(value: float | str, width: int) -> str:
    if isinstance(value, str):
        return f"{value:>{width}}"
    if isinstance(value, int):
        return f"{value:>{width}d}"
    if width <= 4:
        return f"{value:>{width}.1f}"
    return f"{value:>{width}.1f}"


def print_schema_summary(schema: Schema) -> None:
    print(
        json.dumps(
            {
                "tables": len(schema.tables),
                "foreign_keys": len(schema.foreign_keys),
                "core_tables": len(build_sample_schema().tables),
                "distractor_tables": len(schema.tables) - len(build_sample_schema().tables),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 100+ table schema benchmark.")
    parser.add_argument("--json", action="store_true", help="Emit JSON results.")
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Only print the generated schema summary.",
    )
    parser.add_argument(
        "--tables-per-group",
        type=int,
        default=10,
        help="Number of generated distractor tables per group.",
    )
    args = parser.parse_args()

    schema = build_large_schema(tables_per_group=args.tables_per_group)
    if args.schema:
        print_schema_summary(schema)
        return

    results = run_large_benchmark(tables_per_group=args.tables_per_group)
    if args.json:
        print(
            json.dumps(
                {
                    "schema": {
                        "tables": len(schema.tables),
                        "foreign_keys": len(schema.foreign_keys),
                    },
                    "abstract_table": abstract_table_rows(results),
                    "comparison_table": comparison_rows(results),
                    "relationship_metrics": relationship_metrics(results),
                    "summary_by_method": summarize_by_method(results),
                    "summary_by_hardness": summarize_by_hardness(results),
                    "results": [
                        asdict(result) | {"passed": result.passed}
                        for result in results
                    ],
                },
                indent=2,
            )
        )
        return

    print_schema_summary(schema)
    print()
    print_abstract_table(results)
    print()
    print(json.dumps({"by_hardness": summarize_by_hardness(results)}, indent=2))


if __name__ == "__main__":
    main()
