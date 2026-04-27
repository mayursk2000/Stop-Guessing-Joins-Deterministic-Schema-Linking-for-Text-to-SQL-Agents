"""Large-schema benchmark with 100+ tables and paper-style metrics.

This module builds on the realistic core schema from ``prototype.py`` and adds
meaningful operational domains to mimic Spider/BIRD-style large schemas:

- many tables
- shared hubs
- bridge tables
- multi-hop paths
- realistic near-neighbor business concepts

It stays dependency-free and uses SQLite in memory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass

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


@dataclass(frozen=True)
class ExtensionTable:
    name: str
    description: str
    parents: tuple[tuple[str, str], ...]
    aliases: tuple[str, ...] = ()


EXTENSION_TABLES: tuple[ExtensionTable, ...] = (
    ExtensionTable("countries", "country master data for regions and suppliers", (("region_id", "regions"),)),
    ExtensionTable("region_countries", "bridge between regions and country markets", (("region_id", "regions"), ("country_id", "countries"))),
    ExtensionTable("sales_territories", "sales territory hierarchy by region", (("region_id", "regions"),)),
    ExtensionTable("territory_assignments", "employee assignments to sales territories", (("territory_id", "sales_territories"), ("employee_id", "employees"))),
    ExtensionTable("accounts", "commercial account record for a customer", (("customer_id", "customers"), ("region_id", "regions"))),
    ExtensionTable("contacts", "named contacts attached to customer accounts", (("account_id", "accounts"), ("customer_id", "customers"))),
    ExtensionTable("account_contacts", "bridge between accounts and contacts", (("account_id", "accounts"), ("contact_id", "contacts"))),
    ExtensionTable("leads", "pre-sale lead record by campaign and region", (("campaign_id", "campaigns"), ("region_id", "regions"))),
    ExtensionTable("opportunities", "pipeline opportunity linked to accounts and employees", (("account_id", "accounts"), ("employee_id", "employees"))),
    ExtensionTable("opportunity_products", "bridge between opportunities and products", (("opportunity_id", "opportunities"), ("product_id", "products"))),
    ExtensionTable("quotes", "quoted commercial proposal for an opportunity", (("opportunity_id", "opportunities"), ("account_id", "accounts"))),
    ExtensionTable("quote_lines", "product lines on a quote", (("quote_id", "quotes"), ("product_id", "products"))),
    ExtensionTable("contracts", "signed customer contract from an opportunity", (("account_id", "accounts"), ("opportunity_id", "opportunities"))),
    ExtensionTable("contract_lines", "contracted product lines", (("contract_id", "contracts"), ("product_id", "products"))),
    ExtensionTable("billing_accounts", "billing account tied to a commercial account", (("account_id", "accounts"), ("customer_id", "customers"))),
    ExtensionTable("payment_terms", "payment term catalog for billing accounts", (("billing_account_id", "billing_accounts"),)),
    ExtensionTable("invoice_adjustments", "invoice adjustment records", (("invoice_id", "invoices"), ("billing_account_id", "billing_accounts"))),
    ExtensionTable("credit_memos", "credit memo records against invoices", (("invoice_id", "invoices"), ("customer_id", "customers"))),
    ExtensionTable("refunds", "refund records for payments and credit memos", (("payment_id", "payments"), ("credit_memo_id", "credit_memos"))),
    ExtensionTable("tax_rates", "regional tax rates for invoice lines", (("region_id", "regions"),)),
    ExtensionTable("tax_jurisdictions", "country and regional tax jurisdiction", (("country_id", "countries"), ("region_id", "regions"))),
    ExtensionTable("invoice_taxes", "tax amounts attached to invoice lines", (("invoice_line_id", "invoice_lines"), ("tax_rate_id", "tax_rates"))),
    ExtensionTable("payment_allocations", "payment allocation to invoices", (("payment_id", "payments"), ("invoice_id", "invoices"))),
    ExtensionTable("collections_cases", "collections case for unpaid invoices", (("invoice_id", "invoices"), ("customer_id", "customers"))),
    ExtensionTable("collection_events", "events on a collections case", (("collections_case_id", "collections_cases"), ("employee_id", "employees"))),
    ExtensionTable("price_books", "price book catalog by region", (("region_id", "regions"),)),
    ExtensionTable("product_prices", "product price entries in price books", (("price_book_id", "price_books"), ("product_id", "products"))),
    ExtensionTable("product_attributes", "product attribute values", (("product_id", "products"),)),
    ExtensionTable("product_bundles", "sellable product bundle header", (("supplier_id", "suppliers"),)),
    ExtensionTable("bundle_items", "products contained in a product bundle", (("bundle_id", "product_bundles"), ("product_id", "products"))),
    ExtensionTable("supplier_contracts", "supplier contract by supplier and region", (("supplier_id", "suppliers"), ("region_id", "regions"))),
    ExtensionTable("supplier_scorecards", "supplier scorecard by supplier", (("supplier_id", "suppliers"),)),
    ExtensionTable("purchase_orders", "purchase order issued to suppliers", (("supplier_id", "suppliers"), ("warehouse_id", "warehouses"))),
    ExtensionTable("purchase_order_items", "product lines on purchase orders", (("purchase_order_id", "purchase_orders"), ("product_id", "products"))),
    ExtensionTable("receipts", "warehouse receipt against purchase orders", (("purchase_order_id", "purchase_orders"), ("warehouse_id", "warehouses"))),
    ExtensionTable("receipt_items", "received product lines", (("receipt_id", "receipts"), ("product_id", "products"))),
    ExtensionTable("carriers", "shipping carrier master data by region", (("region_id", "regions"),)),
    ExtensionTable("shipment_items", "items included in shipments", (("shipment_id", "shipments"), ("order_item_id", "order_items"))),
    ExtensionTable("shipment_events", "tracking events for shipments", (("shipment_id", "shipments"), ("carrier_id", "carriers"))),
    ExtensionTable("delivery_routes", "delivery route assigned to warehouse and carrier", (("warehouse_id", "warehouses"), ("carrier_id", "carriers"))),
    ExtensionTable("route_stops", "customer stops on a delivery route", (("route_id", "delivery_routes"), ("customer_id", "customers"))),
    ExtensionTable("warehouse_zones", "zones inside warehouses", (("warehouse_id", "warehouses"),)),
    ExtensionTable("bins", "storage bins inside warehouse zones", (("zone_id", "warehouse_zones"),)),
    ExtensionTable("inventory_lots", "inventory lots by product and warehouse", (("product_id", "products"), ("warehouse_id", "warehouses"))),
    ExtensionTable("inventory_movements", "inventory movement by lot and order", (("inventory_lot_id", "inventory_lots"), ("order_id", "orders"))),
    ExtensionTable("stock_reservations", "reserved stock for order items", (("order_item_id", "order_items"), ("inventory_lot_id", "inventory_lots"))),
    ExtensionTable("return_authorizations", "return authorization for orders", (("order_id", "orders"), ("customer_id", "customers"))),
    ExtensionTable("return_authorization_items", "return authorization lines", (("return_authorization_id", "return_authorizations"), ("order_item_id", "order_items"))),
    ExtensionTable("warranty_claims", "warranty claim for returned products", (("return_id", "returns"), ("product_id", "products"))),
    ExtensionTable("campaign_members", "customers targeted by campaigns", (("campaign_id", "campaigns"), ("customer_id", "customers"))),
    ExtensionTable("ad_spend", "advertising spend by campaign and region", (("campaign_id", "campaigns"), ("region_id", "regions"))),
    ExtensionTable("channel_touchpoints", "customer marketing touchpoints", (("campaign_id", "campaigns"), ("customer_id", "customers"))),
    ExtensionTable("attribution_events", "campaign attribution events for orders", (("order_id", "orders"), ("campaign_id", "campaigns"))),
    ExtensionTable("content_assets", "marketing content assets by campaign", (("campaign_id", "campaigns"),)),
    ExtensionTable("content_performance", "performance metrics for content assets", (("content_asset_id", "content_assets"), ("campaign_id", "campaigns"))),
    ExtensionTable("service_levels", "service-level policy by customer segment", (("segment_id", "customer_segments"),)),
    ExtensionTable("ticket_assignments", "ticket ownership by employee", (("ticket_id", "support_tickets"), ("employee_id", "employees"))),
    ExtensionTable("ticket_escalations", "support ticket escalations", (("ticket_id", "support_tickets"), ("employee_id", "employees"))),
    ExtensionTable("knowledge_articles", "knowledge base article by department", (("department_id", "departments"),)),
    ExtensionTable("article_feedback", "customer feedback on knowledge articles", (("article_id", "knowledge_articles"), ("customer_id", "customers"))),
    ExtensionTable("support_case_links", "links between support tickets and orders", (("ticket_id", "support_tickets"), ("order_id", "orders"))),
    ExtensionTable("plans", "subscription plan catalog", (("product_id", "products"),)),
    ExtensionTable("plan_features", "features included in subscription plans", (("plan_id", "plans"),)),
    ExtensionTable("subscription_events", "events for customer subscriptions", (("subscription_id", "subscriptions"), ("customer_id", "customers"))),
    ExtensionTable("renewals", "subscription renewal records", (("subscription_id", "subscriptions"), ("customer_id", "customers"))),
    ExtensionTable("usage_records", "usage records for active subscriptions", (("subscription_id", "subscriptions"), ("product_id", "products"))),
    ExtensionTable("usage_exports", "exports of subscription usage records", (("usage_record_id", "usage_records"), ("customer_id", "customers"))),
    ExtensionTable("entitlements", "customer entitlements by subscription", (("subscription_id", "subscriptions"), ("plan_id", "plans"))),
    ExtensionTable("customer_health_scores", "customer success health scores", (("customer_id", "customers"), ("employee_id", "employees"))),
    ExtensionTable("success_plans", "customer success plans", (("customer_id", "customers"), ("employee_id", "employees"))),
    ExtensionTable("success_plan_tasks", "tasks on customer success plans", (("success_plan_id", "success_plans"), ("employee_id", "employees"))),
    ExtensionTable("business_reviews", "customer business review meetings", (("customer_id", "customers"), ("employee_id", "employees"))),
    ExtensionTable("review_actions", "actions from customer business reviews", (("business_review_id", "business_reviews"), ("employee_id", "employees"))),
    ExtensionTable("user_accounts", "application user account for employees", (("employee_id", "employees"),)),
    ExtensionTable("roles", "role catalog by department", (("department_id", "departments"),)),
    ExtensionTable("user_roles", "bridge between application users and roles", (("user_account_id", "user_accounts"), ("role_id", "roles"))),
    ExtensionTable("access_logs", "application access log by user account", (("user_account_id", "user_accounts"),)),
    ExtensionTable("api_clients", "api client owner accounts", (("user_account_id", "user_accounts"),)),
    ExtensionTable("api_client_scopes", "authorization scopes for api clients", (("api_client_id", "api_clients"), ("role_id", "roles"))),
    ExtensionTable("audit_events", "audit events by user account and customer", (("user_account_id", "user_accounts"), ("customer_id", "customers"))),
    ExtensionTable("data_exports", "data export jobs by user account", (("user_account_id", "user_accounts"),)),
    ExtensionTable("export_files", "files produced by data export jobs", (("data_export_id", "data_exports"),)),
    ExtensionTable("forecast_versions", "forecast version by region and employee", (("region_id", "regions"), ("employee_id", "employees"))),
    ExtensionTable("forecast_lines", "forecast lines by product and forecast version", (("forecast_version_id", "forecast_versions"), ("product_id", "products"))),
    ExtensionTable("quota_plans", "quota plan by employee and region", (("employee_id", "employees"), ("region_id", "regions"))),
    ExtensionTable("quota_attainment", "quota attainment by quota plan and order", (("quota_plan_id", "quota_plans"), ("order_id", "orders"))),
)


def build_large_schema(tables_per_group: int = 10) -> Schema:
    _ = tables_per_group
    base = build_sample_schema()
    tables = dict(base.tables)
    foreign_keys = list(base.foreign_keys)

    for spec in EXTENSION_TABLES:
        tables[spec.name] = Table(
            name=spec.name,
            description=spec.description,
            aliases=spec.aliases or (spec.name.replace("_", " "),),
            columns=(
                Column("id", "primary key"),
                *(Column(fk_column, f"{parent_table} foreign key") for fk_column, parent_table in spec.parents),
                Column("name", "descriptive record name"),
                Column("status", "record status"),
                Column("amount", "numeric business amount"),
                Column("event_date", "business event date"),
                Column("metric_value", "numeric metric value"),
            ),
        )
        for fk_column, parent_table in spec.parents:
            foreign_keys.append(ForeignKey(spec.name, fk_column, parent_table, "id"))

    return Schema(tables=tables, foreign_keys=foreign_keys)


def build_large_database(tables_per_group: int = 10) -> sqlite3.Connection:
    _ = tables_per_group
    conn = build_sample_database()

    for spec_index, spec in enumerate(EXTENSION_TABLES, start=1):
        fk_columns = [
            f"{fk_column} INTEGER NOT NULL"
            for fk_column, _parent_table in spec.parents
        ]
        fk_constraints = [
            f"FOREIGN KEY ({fk_column}) REFERENCES {parent_table}(id)"
            for fk_column, parent_table in spec.parents
        ]
        ddl_parts = [
            "id INTEGER PRIMARY KEY",
            *fk_columns,
            "name TEXT NOT NULL",
            "status TEXT NOT NULL",
            "amount REAL NOT NULL",
            "event_date TEXT NOT NULL",
            "metric_value REAL NOT NULL",
            *fk_constraints,
        ]
        conn.execute(f"CREATE TABLE {spec.name} ({', '.join(ddl_parts)})")

        parent_ids = {
            parent_table: [row[0] for row in conn.execute(f"SELECT id FROM {parent_table}")]
            for _fk_column, parent_table in spec.parents
        }
        first_parent = spec.parents[0][1]
        row_count = min(3, len(parent_ids[first_parent]))
        rows = []
        for row_index in range(1, row_count + 1):
            fk_values = []
            for fk_column, parent_table in spec.parents:
                ids = parent_ids[parent_table]
                fk_values.append(ids[(row_index - 1) % len(ids)])
            rows.append(
                (
                    row_index,
                    *fk_values,
                    f"{spec.name}-{row_index}",
                    "active" if row_index % 2 else "open",
                    float(spec_index * 10 + row_index),
                    f"2024-01-{row_index:02d}",
                    float(spec_index + row_index),
                )
            )

        placeholders = ", ".join("?" for _ in range(6 + len(spec.parents)))
        conn.executemany(
            f"INSERT INTO {spec.name} VALUES ({placeholders})",
            rows,
        )

    return conn


LARGE_CASES: tuple[BenchmarkCase, ...] = CASES + (
    BenchmarkCase(
        id="large_schema_revenue_by_customer",
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
        db_id="sales_ops_108_table",
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
    schema = build_large_schema()
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
                "Tables": len(schema.tables),
                "FKs": len(schema.foreign_keys),
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
    core_tables = len(build_sample_schema().tables)
    print(
        json.dumps(
            {
                "tables": len(schema.tables),
                "foreign_keys": len(schema.foreign_keys),
                "core_tables": core_tables,
                "extension_tables": len(schema.tables) - core_tables,
                "extension_foreign_keys": len(schema.foreign_keys) - len(build_sample_schema().foreign_keys),
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
        help="Compatibility option retained from earlier generated-schema runs.",
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
