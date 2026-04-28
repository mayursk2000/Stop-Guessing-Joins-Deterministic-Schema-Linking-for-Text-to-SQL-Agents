"""Runnable prototype for deterministic schema linking in a text-to-SQL agent.

The code mirrors the architecture in the abstract:

Query -> Retrieval -> Candidate Tables -> Graph Resolution
-> Minimal Connecting Subgraph -> SQL Generation -> Rewriter
-> SQL Execution -> Validation -> Final SQL or Retry.

This prototype intentionally avoids third-party dependencies so it can run in a
fresh workspace. In production, the retrieval and SQL parsing components should
be replaced with embedding models and a parser such as sqlglot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import sqlglot
from sqlglot import exp


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "for",
    "from",
    "give",
    "in",
    "list",
    "me",
    "of",
    "on",
    "show",
    "the",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
        if token not in STOPWORDS
    ]


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def expanded_tokens(text: str) -> Counter[str]:
    tokens = tokenize(text)
    expanded = tokens + [singularize(token) for token in tokens]
    return Counter(expanded)


def cosine_like(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return numerator / (left_norm * right_norm)


@dataclass(frozen=True)
class Column:
    name: str
    description: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    description: str = ""
    aliases: tuple[str, ...] = ()

    def searchable_text(self) -> str:
        column_text = " ".join(
            f"{column.name} {column.description}" for column in self.columns
        )
        return f"{self.name} {self.description} {' '.join(self.aliases)} {column_text}"


@dataclass(frozen=True)
class ForeignKey:
    left_table: str
    left_column: str
    right_table: str
    right_column: str

    def reverse(self) -> "ForeignKey":
        return ForeignKey(
            self.right_table,
            self.right_column,
            self.left_table,
            self.left_column,
        )

    def join_condition(self) -> str:
        return (
            f"{self.left_table}.{self.left_column} = "
            f"{self.right_table}.{self.right_column}"
        )


@dataclass
class Schema:
    tables: dict[str, Table]
    foreign_keys: list[ForeignKey]


@dataclass
class RetrievalMatch:
    table: str
    lexical_score: float
    semantic_score: float
    score: float
    matched_terms: list[str]


@dataclass
class RetrievalResult:
    query: str
    candidate_tables: list[str]
    matches: list[RetrievalMatch]


@dataclass
class JoinPlan:
    tables: list[str]
    joins: list[ForeignKey]
    unresolved_tables: list[str] = field(default_factory=list)

    def prompt_context(self, schema: Schema) -> str:
        table_lines = []
        for table_name in self.tables:
            table = schema.tables[table_name]
            columns = ", ".join(column.name for column in table.columns)
            table_lines.append(f"- {table.name}({columns})")
        join_lines = [f"- {join.join_condition()}" for join in self.joins]
        return "\n".join(
            [
                "Allowed tables:",
                *table_lines,
                "Allowed joins:",
                *(join_lines or ["- no joins required"]),
            ]
        )


@dataclass
class GeneratedSQL:
    sql: str
    rationale: str


@dataclass
class RewriteResult:
    sql: str
    changed: bool
    notes: list[str]


@dataclass(frozen=True)
class GraphJoinPolicy:
    sources: tuple[str, str]
    constraint: str
    on_fail: str = "REWRITE"

    @classmethod
    def from_foreign_key(cls, foreign_key: ForeignKey) -> "GraphJoinPolicy":
        return cls(
            sources=(foreign_key.left_table, foreign_key.right_table),
            constraint=foreign_key.join_condition(),
        )


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[dict[str, Any]]
    error: str | None = None


@dataclass
class ValidationResult:
    valid: bool
    reason: str
    retry_type: str | None = None


@dataclass
class PipelineTrace:
    query: str
    retrieval: RetrievalResult
    join_plan: JoinPlan
    generated: GeneratedSQL
    rewrite: RewriteResult
    execution: ExecutionResult
    validation: ValidationResult
    final_sql: str | None


class HybridRetriever:
    """Lexical plus lightweight semantic retrieval over schema metadata."""

    def __init__(self, schema: Schema, top_k: int = 6, min_score: float = 0.2) -> None:
        self.schema = schema
        self.top_k = top_k
        self.min_score = min_score
        self.table_vectors = {
            name: expanded_tokens(table.searchable_text())
            for name, table in schema.tables.items()
        }

    def retrieve(self, query: str) -> RetrievalResult:
        query_vector = expanded_tokens(query)
        query_terms = set(query_vector)
        matches: list[RetrievalMatch] = []

        for table_name, table_vector in self.table_vectors.items():
            matched_terms = sorted(query_terms & set(table_vector))
            lexical_score = len(matched_terms) / max(len(query_terms), 1)
            semantic_score = cosine_like(query_vector, table_vector)
            score = (0.6 * lexical_score) + (0.4 * semantic_score)
            if score >= self.min_score:
                matches.append(
                    RetrievalMatch(
                        table=table_name,
                        lexical_score=round(lexical_score, 3),
                        semantic_score=round(semantic_score, 3),
                        score=round(score, 3),
                        matched_terms=matched_terms,
                    )
                )

        matches.sort(key=lambda item: item.score, reverse=True)
        candidate_tables = [match.table for match in matches[: self.top_k]]
        return RetrievalResult(query=query, candidate_tables=candidate_tables, matches=matches)


class SchemaGraph:
    """Graph of tables and foreign-key relationships."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.adjacent: dict[str, list[tuple[str, ForeignKey]]] = defaultdict(list)
        for edge in schema.foreign_keys:
            self.adjacent[edge.left_table].append((edge.right_table, edge))
            self.adjacent[edge.right_table].append((edge.left_table, edge.reverse()))

    def shortest_path(self, start: str, goal: str) -> list[ForeignKey] | None:
        queue: deque[tuple[str, list[ForeignKey]]] = deque([(start, [])])
        visited = {start}
        while queue:
            table, path = queue.popleft()
            if table == goal:
                return path
            for next_table, edge in self.adjacent[table]:
                if next_table not in visited:
                    visited.add(next_table)
                    queue.append((next_table, [*path, edge]))
        return None

    def minimal_connecting_subgraph(self, candidate_tables: Iterable[str]) -> JoinPlan:
        candidates = [
            table for table in dict.fromkeys(candidate_tables) if table in self.schema.tables
        ]
        if not candidates:
            return JoinPlan(tables=[], joins=[], unresolved_tables=[])

        selected_tables = {candidates[0]}
        joins: list[ForeignKey] = []
        unresolved: list[str] = []

        for target in candidates[1:]:
            best_path: list[ForeignKey] | None = None
            for source in sorted(selected_tables):
                path = self.shortest_path(source, target)
                if path is not None and (
                    best_path is None or len(path) < len(best_path)
                ):
                    best_path = path

            if best_path is None:
                unresolved.append(target)
                continue

            for edge in best_path:
                if edge not in joins:
                    joins.append(edge)
                selected_tables.add(edge.left_table)
                selected_tables.add(edge.right_table)
            selected_tables.add(target)

        ordered_tables = sorted(selected_tables, key=lambda name: candidates.index(name) if name in candidates else 999)
        return JoinPlan(tables=ordered_tables, joins=joins, unresolved_tables=unresolved)


class SQLGenerator:
    """LLM stand-in that generates SQL from a constrained join plan."""

    def generate(self, query: str, join_plan: JoinPlan, schema: Schema) -> GeneratedSQL:
        terms = set(expanded_tokens(query))
        table_set = set(join_plan.tables)

        if "shipment" in terms and "shipments" in table_set:
            sql = self._shipment_sql(query, join_plan, table_set)
            return GeneratedSQL(
                sql=sql,
                rationale="Shipment query using graph-approved logistics joins.",
            )

        if {"payment", "paid", "billing"} & terms and "payments" in table_set:
            sql = self._payment_sql(query, join_plan)
            return GeneratedSQL(
                sql=sql,
                rationale="Payment query using graph-approved billing joins.",
            )

        if {"invoice", "invoices", "billed", "billing"} & terms and "invoices" in table_set:
            sql = self._invoice_sql(query, join_plan, table_set)
            return GeneratedSQL(
                sql=sql,
                rationale="Invoice query using graph-approved billing joins.",
            )

        if {"subscription", "subscriptions", "recurring", "mrr"} & terms and "subscriptions" in table_set:
            sql = self._subscription_sql(query, join_plan, table_set)
            return GeneratedSQL(
                sql=sql,
                rationale="Subscription query using graph-approved customer/product joins.",
            )

        if {"ticket", "tickets", "support", "case", "cases"} & terms and "support_tickets" in table_set:
            sql = self._support_sql(query, join_plan, table_set)
            return GeneratedSQL(
                sql=sql,
                rationale="Support ticket query using graph-approved customer segment joins.",
            )

        if {"return", "returned", "refund"} & terms and "returns" in table_set:
            sql = self._returns_sql(query, join_plan, table_set)
            return GeneratedSQL(
                sql=sql,
                rationale="Return query using graph-approved order-item joins.",
            )

        if {"stock", "inventory", "available"} & terms and "inventory" in table_set:
            sql = self._inventory_sql(query, join_plan, table_set)
            return GeneratedSQL(
                sql=sql,
                rationale="Inventory query using graph-approved warehouse/product joins.",
            )

        if {"campaign", "promotion", "marketing"} & terms and "campaigns" in table_set:
            sql = self._campaign_sql(query, join_plan)
            return GeneratedSQL(
                sql=sql,
                rationale="Campaign attribution query using graph-approved bridge joins.",
            )

        if {"revenue", "sale", "sales", "amount"} & terms and "orders" in table_set:
            sql = self._orders_revenue_sql(query, join_plan, table_set)
            return GeneratedSQL(
                sql=sql,
                rationale="Revenue-style query using graph-approved order joins.",
            )

        if "customer" in terms and "customers" in table_set:
            sql = self._select_from_join_plan("customers.name", join_plan)
            return GeneratedSQL(sql=sql, rationale="Customer lookup from constrained graph.")

        first_table = join_plan.tables[0] if join_plan.tables else next(iter(schema.tables))
        return GeneratedSQL(
            sql=f"SELECT * FROM {first_table} LIMIT 20",
            rationale="Fallback query over highest-ranked candidate table.",
        )

    def _orders_revenue_sql(
        self, query: str, join_plan: JoinPlan, table_set: set[str]
    ) -> str:
        terms = set(expanded_tokens(query))
        revenue_expr = "SUM(orders.amount) AS revenue"
        if {"product", "category", "supplier", "country"} & terms and "order_items" in table_set:
            revenue_expr = "SUM(order_items.quantity * order_items.unit_price) AS revenue"

        group_fields: list[str] = []
        group_by_customer = "customers" in table_set and (
            "customer" in terms or "customers" in terms
        )
        if group_by_customer:
            group_fields.append("customers.name")
        if "region" in terms and "regions" in table_set:
            group_fields.append("regions.name")
        if "category" in terms and "products" in table_set:
            group_fields.append("products.category")
        if "country" in terms and "suppliers" in table_set:
            group_fields.append("suppliers.country")
        elif "supplier" in terms and "suppliers" in table_set:
            group_fields.append("suppliers.name")
        if {"rep", "representative", "salesperson"} & terms and "employees" in table_set:
            group_fields.append("employees.name")

        select_clause = ", ".join([*group_fields, revenue_expr])
        sql = self._select_from_join_plan(select_clause, join_plan)
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _payment_sql(self, query: str, join_plan: JoinPlan) -> str:
        terms = set(expanded_tokens(query))
        group_fields = ["payments.method"] if "method" in terms else []
        select_clause = ", ".join([*group_fields, "SUM(payments.amount) AS paid_amount"])
        sql = self._select_from_join_plan(select_clause, join_plan)
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _invoice_sql(
        self, query: str, join_plan: JoinPlan, table_set: set[str]
    ) -> str:
        terms = set(expanded_tokens(query))
        group_fields: list[str] = []
        if "category" in terms and "products" in table_set:
            group_fields.append("products.category")
        if "status" in terms:
            group_fields.append("invoices.status")
        metric = "SUM(invoice_lines.amount) AS billed_amount" if "invoice_lines" in table_set else "SUM(invoices.total_amount) AS billed_amount"
        sql = self._select_from_join_plan(", ".join([*group_fields, metric]), join_plan)
        if "open" in terms:
            sql += " WHERE invoices.status = 'open'"
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _subscription_sql(
        self, query: str, join_plan: JoinPlan, table_set: set[str]
    ) -> str:
        terms = set(expanded_tokens(query))
        group_fields: list[str] = []
        if "region" in terms and "regions" in table_set:
            group_fields.append("regions.name")
        if "category" in terms and "products" in table_set:
            group_fields.append("products.category")
        sql = self._select_from_join_plan(
            ", ".join([*group_fields, "SUM(subscriptions.monthly_amount) AS mrr"]),
            join_plan,
        )
        if "active" in terms:
            sql += " WHERE subscriptions.status = 'active'"
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _support_sql(
        self, query: str, join_plan: JoinPlan, table_set: set[str]
    ) -> str:
        terms = set(expanded_tokens(query))
        group_fields: list[str] = []
        if "segment" in terms and "customer_segments" in table_set:
            group_fields.append("customer_segments.name")
        if "product" in terms and "products" in table_set:
            group_fields.append("products.name")
        if "priority" in terms:
            group_fields.append("support_tickets.priority")
        sql = self._select_from_join_plan(
            ", ".join([*group_fields, "COUNT(support_tickets.id) AS tickets"]),
            join_plan,
        )
        if "open" in terms:
            sql += " WHERE support_tickets.status = 'open'"
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _returns_sql(
        self, query: str, join_plan: JoinPlan, table_set: set[str]
    ) -> str:
        terms = set(expanded_tokens(query))
        group_fields: list[str] = []
        if "supplier" in terms and "suppliers" in table_set:
            group_fields.append("suppliers.name")
        if "category" in terms and "products" in table_set:
            group_fields.append("products.category")
        select_clause = ", ".join([*group_fields, "SUM(returns.quantity) AS returned_units"])
        sql = self._select_from_join_plan(select_clause, join_plan)
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _inventory_sql(
        self, query: str, join_plan: JoinPlan, table_set: set[str]
    ) -> str:
        terms = set(expanded_tokens(query))
        group_fields: list[str] = []
        if "warehouse" in terms and "warehouses" in table_set:
            group_fields.append("warehouses.name")
        if "category" in terms and "products" in table_set:
            group_fields.append("products.category")
        select_clause = ", ".join([*group_fields, "SUM(inventory.quantity_on_hand) AS stock"])
        sql = self._select_from_join_plan(select_clause, join_plan)
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _campaign_sql(self, query: str, join_plan: JoinPlan) -> str:
        terms = set(expanded_tokens(query))
        group_fields = ["campaigns.channel"] if "channel" in terms else ["campaigns.name"]
        metric = "SUM(orders.amount) AS revenue" if "revenue" in terms else "COUNT(orders.id) AS orders"
        sql = self._select_from_join_plan(", ".join([*group_fields, metric]), join_plan)
        sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _shipment_sql(
        self, query: str, join_plan: JoinPlan, table_set: set[str]
    ) -> str:
        terms = set(expanded_tokens(query))
        group_fields: list[str] = []
        if "warehouse" in terms and "warehouses" in table_set:
            group_fields.append("warehouses.name")
        if "region" in terms and "regions" in table_set:
            group_fields.append("regions.name")

        select_clause = ", ".join([*group_fields, "COUNT(shipments.id) AS shipments"])
        sql = self._select_from_join_plan(select_clause, join_plan)
        if "late" in terms:
            sql += " WHERE shipments.status = 'late'"
        if group_fields:
            sql += " GROUP BY " + ", ".join(group_fields)
        return sql

    def _select_from_join_plan(self, select_clause: str, join_plan: JoinPlan) -> str:
        if not join_plan.tables:
            return "SELECT 1"
        base_table = join_plan.tables[0]
        sql = f"SELECT {select_clause} FROM {base_table}"
        used = {base_table}
        remaining = list(join_plan.joins)

        while remaining:
            next_index = None
            for index, join in enumerate(remaining):
                if join.left_table in used or join.right_table in used:
                    next_index = index
                    break
            if next_index is None:
                next_index = 0

            join = remaining.pop(next_index)
            next_table = join.right_table if join.left_table in used else join.left_table
            condition = join.join_condition()
            sql += f" JOIN {next_table} ON {condition}"
            used.add(next_table)
        return sql


class OpenAILLMGenerator(SQLGenerator):
    """LLM-backed SQL generator with deterministic generator fallback."""

    def __init__(
        self,
        model: str | None = None,
        fallback: SQLGenerator | None = None,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.fallback = fallback or SQLGenerator()

    def generate(self, query: str, join_plan: JoinPlan, schema: Schema) -> GeneratedSQL:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            fallback = self.fallback.generate(query, join_plan, schema)
            return GeneratedSQL(
                sql=fallback.sql,
                rationale=(
                    "OpenAI API key not set; used deterministic fallback generator."
                ),
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = self._build_prompt(query, join_plan, schema)
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Text-to-SQL agent. Return only SQL. "
                            "Use only the allowed tables and allowed joins."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            sql = self._extract_response_text(response)
            sql = self._strip_code_fence(sql)
            if not sql.lower().lstrip().startswith("select"):
                raise ValueError(f"LLM did not return a SELECT query: {sql[:80]}")
            return GeneratedSQL(
                sql=sql,
                rationale=f"OpenAI LLM SQL generation via {self.model}.",
            )
        except Exception as exc:
            fallback = self.fallback.generate(query, join_plan, schema)
            return GeneratedSQL(
                sql=fallback.sql,
                rationale=f"LLM generation failed ({exc}); used deterministic fallback.",
            )

    def _build_prompt(self, query: str, join_plan: JoinPlan, schema: Schema) -> str:
        return "\n".join(
            [
                f"Question: {query}",
                "",
                join_plan.prompt_context(schema),
                "",
                "Rules:",
                "- Return exactly one SQLite-compatible SELECT statement.",
                "- Do not use tables outside the allowed table list.",
                "- Do not invent join predicates.",
                "- Use aggregate aliases when helpful.",
            ]
        )

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()

    def _strip_code_fence(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:sql)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip().rstrip(";")


class DFCRewriterAdapter:
    """Adapter around the data-flow-control SQL rewriter project."""

    def __init__(
        self,
        project_path: str | None = None,
        enabled: bool = True,
    ) -> None:
        default_path = r"C:\Users\MK\Desktop\data-flow-control\sql_rewriter"
        self.project_path = Path(project_path or os.environ.get("DFC_SQL_REWRITER_PATH", default_path))
        self.enabled = enabled
        self.available = False
        self.error: str | None = None
        self._rewriter: Any | None = None

        if enabled:
            self._load()

    def _load(self) -> None:
        src_path = self.project_path / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        try:
            from sql_rewriter import SQLRewriter as DFCSQLRewriter

            self._rewriter = DFCSQLRewriter()
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def build_join_policies(self, join_plan: JoinPlan) -> list[GraphJoinPolicy]:
        return [GraphJoinPolicy.from_foreign_key(join) for join in join_plan.joins]

    def transform(
        self, sql: str, policies: list[GraphJoinPolicy] | None = None
    ) -> tuple[str, list[str]]:
        policies = policies or []
        if not self.enabled:
            return sql, ["DFC rewriter disabled.", *self._policy_notes(policies)]
        if not self.available or self._rewriter is None:
            return sql, [
                f"DFC rewriter unavailable: {self.error}",
                *self._policy_notes(policies),
            ]
        try:
            transformed = self._rewriter.transform_query(sql)
            return transformed, [
                "Transformed with data-flow-control SQLRewriter.",
                *self._policy_notes(policies),
            ]
        except Exception as exc:
            return sql, [f"DFC transform skipped: {exc}", *self._policy_notes(policies)]

    def _policy_notes(self, policies: list[GraphJoinPolicy]) -> list[str]:
        if not policies:
            return ["No graph join policies for this query."]
        return [
            (
                "Graph join policy "
                f"SOURCES {', '.join(policy.sources)} "
                f"CONSTRAINT {policy.constraint} "
                f"ON FAIL {policy.on_fail}"
            )
            for policy in policies
        ]


class SQLRewriter:
    """Reactive layer that enforces graph-approved join predicates."""

    def __init__(self, dfc_adapter: DFCRewriterAdapter | None = None) -> None:
        self.dfc_adapter = dfc_adapter or DFCRewriterAdapter()

    def rewrite(self, sql: str, join_plan: JoinPlan) -> RewriteResult:
        notes: list[str] = []
        original_sql = sql.strip()
        policies = self.dfc_adapter.build_join_policies(join_plan)

        transformed, dfc_notes = self.dfc_adapter.transform(original_sql, policies)
        notes.extend(dfc_notes)
        rewritten = self._repair_join_predicates(transformed, policies, notes)
        return RewriteResult(
            sql=rewritten.strip(),
            changed=rewritten.strip() != original_sql,
            notes=notes,
        )

    def _repair_join_predicates(
        self, sql: str, policies: list[GraphJoinPolicy], notes: list[str]
    ) -> str:
        try:
            parsed = sqlglot.parse_one(sql, read="sqlite")
        except Exception as exc:
            notes.append(f"AST join repair skipped: {exc}")
            return sql

        from_expr = parsed.args.get("from") or parsed.args.get("from_")
        used_tables: set[str] = set()
        if from_expr:
            first_table = next(from_expr.find_all(exp.Table), None)
            if first_table is not None:
                used_tables.add(first_table.name.lower())

        for join in parsed.find_all(exp.Join):
            table_expr = join.this
            if not isinstance(table_expr, exp.Table):
                continue
            table = table_expr.name.lower()
            existing_expr = join.args.get("on")
            existing = existing_expr.sql(dialect="sqlite") if existing_expr else ""
            allowed = self._allowed_condition_for_join(table, used_tables, policies)
            used_tables.add(table)
            if allowed is None:
                notes.append(f"Join to {table} has no approved graph edge.")
                continue
            if self._normalize_condition(existing) != self._normalize_condition(allowed):
                notes.append(f"Rewrote join predicate for {table}: {existing} -> {allowed}")
                join.set("on", sqlglot.parse_one(allowed, read="sqlite"))

        return parsed.sql(dialect="sqlite")

    def _allowed_condition_for_join(
        self, table: str, used_tables: set[str], policies: list[GraphJoinPolicy]
    ) -> str | None:
        for policy in policies:
            left, right = (source.lower() for source in policy.sources)
            if table == left and right in used_tables:
                return policy.constraint
            if table == right and left in used_tables:
                return policy.constraint
        return None

    def _normalize_condition(self, condition: str) -> str:
        sides = sorted(part.strip().lower() for part in condition.split("="))
        return "=".join(sides)


class SQLExecutor:
    def __init__(self, connection: sqlite3.Connection, row_limit: int = 2000) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.row_limit = row_limit

    def execute(self, sql: str) -> ExecutionResult:
        try:
            cursor = self.connection.execute(sql)
            rows = [dict(row) for row in cursor.fetchmany(self.row_limit)]
            return ExecutionResult(ok=True, rows=rows)
        except (sqlite3.Error, MemoryError) as exc:
            return ExecutionResult(ok=False, rows=[], error=str(exc))


class Validator:
    """Final gate for structural validity and semantic alignment."""

    def __init__(self, schema: Schema, min_alignment: float = 0.08) -> None:
        self.schema = schema
        self.min_alignment = min_alignment

    def validate(
        self,
        query: str,
        sql: str,
        join_plan: JoinPlan,
        execution: ExecutionResult,
    ) -> ValidationResult:
        if not execution.ok:
            return ValidationResult(False, execution.error or "execution failed", "repair")

        referenced_tables = self._referenced_tables(sql)
        unknown = referenced_tables - set(self.schema.tables)
        if unknown:
            return ValidationResult(False, f"Unknown tables: {sorted(unknown)}", "new_graph")

        allowed_conditions = {
            SQLRewriter()._normalize_condition(join.join_condition())
            for join in join_plan.joins
        }
        for condition in self._join_conditions(sql):
            if SQLRewriter()._normalize_condition(condition) not in allowed_conditions:
                return ValidationResult(False, f"Invalid join condition: {condition}", "repair")

        schema_text = " ".join(
            self.schema.tables[table].searchable_text() for table in referenced_tables
        )
        alignment = cosine_like(expanded_tokens(query), expanded_tokens(schema_text))
        if alignment < self.min_alignment:
            return ValidationResult(
                False,
                f"Low semantic alignment: {alignment:.3f}",
                "new_graph",
            )

        return ValidationResult(True, f"Validated with alignment {alignment:.3f}")

    def _referenced_tables(self, sql: str) -> set[str]:
        matches = re.findall(
            r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, flags=re.IGNORECASE
        )
        return {match.lower() for match in matches}

    def _join_conditions(self, sql: str) -> list[str]:
        try:
            parsed = sqlglot.parse_one(sql, read="sqlite")
        except Exception:
            return []
        conditions = []
        for join in parsed.find_all(exp.Join):
            on_expr = join.args.get("on")
            if on_expr is not None:
                conditions.append(" ".join(on_expr.sql(dialect="sqlite").split()))
        return conditions


class RetryController:
    """Routes failed attempts to repair or alternate graph computation."""

    def retry(
        self,
        query: str,
        retrieval: RetrievalResult,
        graph: SchemaGraph,
        generator: SQLGenerator,
        rewriter: SQLRewriter,
        executor: SQLExecutor,
        validator: Validator,
    ) -> PipelineTrace:
        alternate_tables = retrieval.candidate_tables[1:] + retrieval.candidate_tables[:1]
        join_plan = graph.minimal_connecting_subgraph(alternate_tables)
        generated = generator.generate(query, join_plan, graph.schema)
        rewrite = rewriter.rewrite(generated.sql, join_plan)
        execution = executor.execute(rewrite.sql)
        validation = validator.validate(query, rewrite.sql, join_plan, execution)
        return PipelineTrace(
            query=query,
            retrieval=retrieval,
            join_plan=join_plan,
            generated=generated,
            rewrite=rewrite,
            execution=execution,
            validation=validation,
            final_sql=rewrite.sql if validation.valid else None,
        )


class TextToSQLAgent:
    def __init__(
        self,
        schema: Schema,
        connection: sqlite3.Connection,
        use_pruning: bool = True,
        use_llm: bool | None = None,
    ) -> None:
        self.schema = schema
        self.use_pruning = use_pruning
        self.retriever = HybridRetriever(schema)
        self.graph = SchemaGraph(schema)
        self.use_llm = use_llm if use_llm is not None else os.environ.get("USE_LLM_AGENT") == "1"
        self.generator = OpenAILLMGenerator() if self.use_llm else SQLGenerator()
        self.rewriter = SQLRewriter()
        self.executor = SQLExecutor(connection)
        self.validator = Validator(schema)
        self.retry_controller = RetryController()

    def run(self, query: str) -> PipelineTrace:
        retrieval = self.retriever.retrieve(query)
        candidate_tables = (
            self._prune_candidate_tables(query, retrieval.candidate_tables)
            if self.use_pruning
            else retrieval.candidate_tables
        )
        join_plan = self.graph.minimal_connecting_subgraph(candidate_tables)
        generated = self.generator.generate(query, join_plan, self.schema)
        rewrite = self.rewriter.rewrite(generated.sql, join_plan)
        execution = self.executor.execute(rewrite.sql)
        validation = self.validator.validate(query, rewrite.sql, join_plan, execution)

        trace = PipelineTrace(
            query=query,
            retrieval=retrieval,
            join_plan=join_plan,
            generated=generated,
            rewrite=rewrite,
            execution=execution,
            validation=validation,
            final_sql=rewrite.sql if validation.valid else None,
        )

        if not validation.valid and validation.retry_type == "new_graph":
            return self.retry_controller.retry(
                query,
                retrieval,
                self.graph,
                self.generator,
                self.rewriter,
                self.executor,
                self.validator,
            )
        return trace

    def _prune_candidate_tables(self, query: str, candidate_tables: list[str]) -> list[str]:
        """Keep retrieval broad, then prune obvious off-intent tables.

        This mirrors a production reranker stage. The graph resolver is still
        allowed to add bridge tables needed to connect the remaining tables.
        """
        terms = set(expanded_tokens(query))
        candidates = list(candidate_tables)

        auxiliary_markers = (
            "_profile_",
            "_score_",
            "_audit_",
            "_status_event_",
            "_metric_",
            "_alias_",
            "_quota_",
            "_capacity_",
            "_adjustment_",
            "_note_",
        )
        blocked: set[str] = {
            table for table in candidates if any(marker in table for marker in auxiliary_markers)
        }
        core_tables = {
            "regions",
            "customers",
            "departments",
            "employees",
            "orders",
            "order_items",
            "products",
            "suppliers",
            "payments",
            "warehouses",
            "shipments",
            "returns",
            "campaigns",
            "order_campaigns",
            "inventory",
            "invoices",
            "invoice_lines",
            "subscriptions",
            "support_tickets",
            "ticket_events",
            "customer_segments",
            "segment_members",
        }
        generic_extension_terms = {
            "account",
            "amount",
            "customer",
            "employee",
            "invoice",
            "order",
            "payment",
            "product",
            "region",
            "sale",
            "sales",
            "shipment",
            "supplier",
            "support",
            "ticket",
            "warehouse",
        }
        for table in candidates:
            if table in core_tables:
                continue
            extension_terms = set(expanded_tokens(table.replace("_", " ")))
            extension_terms -= generic_extension_terms
            if extension_terms and not (extension_terms & terms):
                blocked.add(table)

        if {"revenue", "sale", "sales"} & terms:
            if not ({"stock", "inventory", "available"} & terms):
                blocked.add("inventory")
            if "shipment" not in terms:
                blocked.update({"shipments", "warehouses"})
            if not ({"payment", "paid", "billing"} & terms):
                blocked.add("payments")
            if not ({"return", "returned", "refund"} & terms):
                blocked.add("returns")
            if not ({"subscription", "subscriptions", "recurring", "mrr"} & terms):
                blocked.add("subscriptions")
            if not ({"invoice", "invoices", "billed"} & terms):
                blocked.update({"invoices", "invoice_lines"})
            if not ({"ticket", "tickets", "support", "case", "cases"} & terms):
                blocked.update({"support_tickets", "ticket_events"})
            if "segment" not in terms:
                blocked.update({"customer_segments", "segment_members"})
            if not ({"customer", "region"} & terms):
                blocked.update({"customers", "regions"})
            if not ({"supplier", "country"} & terms):
                blocked.add("suppliers")

        if {"payment", "paid", "billing"} & terms and "order" not in terms:
            blocked.update({"orders", "invoices", "invoice_lines"})

        if {"invoice", "invoices", "billed"} & terms:
            if "category" not in terms:
                blocked.update({"products", "invoice_lines"})
            blocked.update({"payments", "shipments", "returns", "inventory", "subscriptions"})

        if {"subscription", "subscriptions", "recurring", "mrr"} & terms:
            blocked.update({"orders", "order_items", "payments", "shipments", "invoices"})
            if "region" not in terms:
                blocked.add("regions")
            if "category" not in terms:
                blocked.add("products")

        if {"ticket", "tickets", "support", "case", "cases"} & terms:
            blocked.update(
                {
                    "orders",
                    "order_items",
                    "payments",
                    "shipments",
                    "invoices",
                    "ticket_events",
                }
            )
            if "segment" not in terms:
                blocked.update({"customer_segments", "segment_members"})
            if "product" not in terms:
                blocked.add("products")

        if {"shipment", "shipping", "delivery"} & terms:
            if "customer" not in terms:
                blocked.add("customers")
            if not ({"product", "category"} & terms):
                blocked.update({"products", "order_items"})

        if {"stock", "inventory", "available"} & terms:
            blocked.update({"orders", "order_items", "customers", "shipments"})

        pruned = [table for table in candidates if table not in blocked]
        return pruned or candidates


def build_sample_schema() -> Schema:
    return Schema(
        tables={
            "regions": Table(
                name="regions",
                description="geographic sales territory and warehouse region",
                aliases=("territory", "area", "market"),
                columns=(
                    Column("id", "primary key"),
                    Column("name", "region name"),
                ),
            ),
            "customers": Table(
                name="customers",
                description="people or accounts buying products in a sales region",
                aliases=("client", "buyer", "account"),
                columns=(
                    Column("id", "primary key"),
                    Column("name", "customer name"),
                    Column("region_id", "customer sales region foreign key"),
                ),
            ),
            "departments": Table(
                name="departments",
                description="employee organization departments",
                aliases=("team", "business unit"),
                columns=(
                    Column("id", "primary key"),
                    Column("name", "department name"),
                ),
            ),
            "employees": Table(
                name="employees",
                description="sales representatives and staff assigned to orders",
                aliases=("rep", "representative", "salesperson", "staff"),
                columns=(
                    Column("id", "primary key"),
                    Column("department_id", "department foreign key"),
                    Column("name", "employee name"),
                    Column("title", "job title"),
                ),
            ),
            "orders": Table(
                name="orders",
                description="sales transactions and revenue amounts placed by customers",
                aliases=("sale", "purchase", "transaction"),
                columns=(
                    Column("id", "primary key"),
                    Column("customer_id", "customer foreign key"),
                    Column("sales_rep_id", "employee sales representative foreign key"),
                    Column("order_date", "date of sale"),
                    Column("amount", "revenue amount"),
                    Column("status", "order status"),
                ),
            ),
            "order_items": Table(
                name="order_items",
                description="line items inside each order with product quantities and prices",
                aliases=("line item", "basket"),
                columns=(
                    Column("id", "primary key"),
                    Column("order_id", "order foreign key"),
                    Column("product_id", "product foreign key"),
                    Column("quantity", "units sold"),
                    Column("unit_price", "item sale price"),
                ),
            ),
            "products": Table(
                name="products",
                description="catalog of products, categories, and supplier links",
                aliases=("item", "sku"),
                columns=(
                    Column("id", "primary key"),
                    Column("supplier_id", "supplier foreign key"),
                    Column("name", "product name"),
                    Column("category", "product category"),
                ),
            ),
            "suppliers": Table(
                name="suppliers",
                description="vendors supplying products by country",
                aliases=("vendor", "manufacturer"),
                columns=(
                    Column("id", "primary key"),
                    Column("name", "supplier name"),
                    Column("country", "supplier country"),
                ),
            ),
            "payments": Table(
                name="payments",
                description="payments made against orders",
                aliases=("payment", "paid", "billing"),
                columns=(
                    Column("id", "primary key"),
                    Column("order_id", "order foreign key"),
                    Column("paid_at", "payment date"),
                    Column("amount", "paid amount"),
                    Column("method", "payment method"),
                ),
            ),
            "warehouses": Table(
                name="warehouses",
                description="fulfillment warehouses located in regions",
                aliases=("fulfillment center", "distribution center"),
                columns=(
                    Column("id", "primary key"),
                    Column("region_id", "warehouse region foreign key"),
                    Column("name", "warehouse name"),
                ),
            ),
            "shipments": Table(
                name="shipments",
                description="order shipments from warehouses with shipping status",
                aliases=("delivery", "fulfillment", "shipping"),
                columns=(
                    Column("id", "primary key"),
                    Column("order_id", "order foreign key"),
                    Column("warehouse_id", "warehouse foreign key"),
                    Column("shipped_at", "shipping date"),
                    Column("status", "shipment status such as delivered or late"),
                ),
            ),
            "returns": Table(
                name="returns",
                description="returned order items, refunds, and return reasons",
                aliases=("return", "refund", "returned"),
                columns=(
                    Column("id", "primary key"),
                    Column("order_item_id", "order item foreign key"),
                    Column("returned_at", "return date"),
                    Column("quantity", "returned unit count"),
                    Column("reason", "return reason"),
                ),
            ),
            "campaigns": Table(
                name="campaigns",
                description="marketing campaigns that influence orders",
                aliases=("campaign", "promotion", "marketing"),
                columns=(
                    Column("id", "primary key"),
                    Column("name", "campaign name"),
                    Column("channel", "marketing channel"),
                ),
            ),
            "order_campaigns": Table(
                name="order_campaigns",
                description="bridge table mapping orders to campaigns",
                aliases=("campaign attribution", "promotion attribution"),
                columns=(
                    Column("order_id", "order foreign key"),
                    Column("campaign_id", "campaign foreign key"),
                ),
            ),
            "inventory": Table(
                name="inventory",
                description="product stock levels by warehouse",
                aliases=("stock", "available", "on hand"),
                columns=(
                    Column("warehouse_id", "warehouse foreign key"),
                    Column("product_id", "product foreign key"),
                    Column("quantity_on_hand", "available stock quantity"),
                ),
            ),
            "invoices": Table(
                name="invoices",
                description="customer invoices issued for orders and billing",
                aliases=("invoice", "bill", "billed", "billing document"),
                columns=(
                    Column("id", "primary key"),
                    Column("order_id", "order foreign key"),
                    Column("issued_at", "invoice issue date"),
                    Column("status", "invoice status"),
                    Column("total_amount", "invoice total amount"),
                ),
            ),
            "invoice_lines": Table(
                name="invoice_lines",
                description="line items on invoices linked to products",
                aliases=("invoice item", "billing line"),
                columns=(
                    Column("id", "primary key"),
                    Column("invoice_id", "invoice foreign key"),
                    Column("product_id", "product foreign key"),
                    Column("amount", "line amount"),
                ),
            ),
            "subscriptions": Table(
                name="subscriptions",
                description="recurring customer subscriptions for products",
                aliases=("subscription", "recurring contract"),
                columns=(
                    Column("id", "primary key"),
                    Column("customer_id", "customer foreign key"),
                    Column("product_id", "product foreign key"),
                    Column("started_at", "subscription start date"),
                    Column("status", "subscription status"),
                    Column("monthly_amount", "monthly recurring revenue"),
                ),
            ),
            "support_tickets": Table(
                name="support_tickets",
                description="customer support cases and ticket priority",
                aliases=("ticket", "case", "support"),
                columns=(
                    Column("id", "primary key"),
                    Column("customer_id", "customer foreign key"),
                    Column("product_id", "product foreign key"),
                    Column("created_at", "ticket creation date"),
                    Column("priority", "ticket priority"),
                    Column("status", "ticket status"),
                ),
            ),
            "ticket_events": Table(
                name="ticket_events",
                description="status changes and events for support tickets",
                aliases=("ticket event", "case history"),
                columns=(
                    Column("id", "primary key"),
                    Column("ticket_id", "support ticket foreign key"),
                    Column("event_type", "ticket event type"),
                    Column("created_at", "event time"),
                ),
            ),
            "customer_segments": Table(
                name="customer_segments",
                description="named customer market segments",
                aliases=("segment", "cohort", "customer group"),
                columns=(
                    Column("id", "primary key"),
                    Column("name", "segment name"),
                ),
            ),
            "segment_members": Table(
                name="segment_members",
                description="bridge table mapping customers to segments",
                aliases=("segment membership", "cohort membership"),
                columns=(
                    Column("customer_id", "customer foreign key"),
                    Column("segment_id", "segment foreign key"),
                ),
            ),
        },
        foreign_keys=[
            ForeignKey("customers", "region_id", "regions", "id"),
            ForeignKey("employees", "department_id", "departments", "id"),
            ForeignKey("orders", "customer_id", "customers", "id"),
            ForeignKey("orders", "sales_rep_id", "employees", "id"),
            ForeignKey("order_items", "order_id", "orders", "id"),
            ForeignKey("order_items", "product_id", "products", "id"),
            ForeignKey("products", "supplier_id", "suppliers", "id"),
            ForeignKey("payments", "order_id", "orders", "id"),
            ForeignKey("warehouses", "region_id", "regions", "id"),
            ForeignKey("shipments", "order_id", "orders", "id"),
            ForeignKey("shipments", "warehouse_id", "warehouses", "id"),
            ForeignKey("returns", "order_item_id", "order_items", "id"),
            ForeignKey("order_campaigns", "order_id", "orders", "id"),
            ForeignKey("order_campaigns", "campaign_id", "campaigns", "id"),
            ForeignKey("inventory", "warehouse_id", "warehouses", "id"),
            ForeignKey("inventory", "product_id", "products", "id"),
            ForeignKey("invoices", "order_id", "orders", "id"),
            ForeignKey("invoice_lines", "invoice_id", "invoices", "id"),
            ForeignKey("invoice_lines", "product_id", "products", "id"),
            ForeignKey("subscriptions", "customer_id", "customers", "id"),
            ForeignKey("subscriptions", "product_id", "products", "id"),
            ForeignKey("support_tickets", "customer_id", "customers", "id"),
            ForeignKey("support_tickets", "product_id", "products", "id"),
            ForeignKey("ticket_events", "ticket_id", "support_tickets", "id"),
            ForeignKey("segment_members", "customer_id", "customers", "id"),
            ForeignKey("segment_members", "segment_id", "customer_segments", "id"),
        ],
    )


def build_sample_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region_id INTEGER NOT NULL,
            FOREIGN KEY (region_id) REFERENCES regions(id)
        );

        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            department_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            title TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            sales_rep_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (sales_rep_id) REFERENCES employees(id)
        );

        CREATE TABLE suppliers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT NOT NULL
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            supplier_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
        );

        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE payments (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            paid_at TEXT NOT NULL,
            amount REAL NOT NULL,
            method TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );

        CREATE TABLE warehouses (
            id INTEGER PRIMARY KEY,
            region_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            FOREIGN KEY (region_id) REFERENCES regions(id)
        );

        CREATE TABLE shipments (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            warehouse_id INTEGER NOT NULL,
            shipped_at TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (warehouse_id) REFERENCES warehouses(id)
        );

        CREATE TABLE returns (
            id INTEGER PRIMARY KEY,
            order_item_id INTEGER NOT NULL,
            returned_at TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            reason TEXT NOT NULL,
            FOREIGN KEY (order_item_id) REFERENCES order_items(id)
        );

        CREATE TABLE campaigns (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            channel TEXT NOT NULL
        );

        CREATE TABLE order_campaigns (
            order_id INTEGER NOT NULL,
            campaign_id INTEGER NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
        );

        CREATE TABLE inventory (
            warehouse_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity_on_hand INTEGER NOT NULL,
            FOREIGN KEY (warehouse_id) REFERENCES warehouses(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE invoices (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            issued_at TEXT NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );

        CREATE TABLE invoice_lines (
            id INTEGER PRIMARY KEY,
            invoice_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            FOREIGN KEY (invoice_id) REFERENCES invoices(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            status TEXT NOT NULL,
            monthly_amount REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE support_tickets (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE ticket_events (
            id INTEGER PRIMARY KEY,
            ticket_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (ticket_id) REFERENCES support_tickets(id)
        );

        CREATE TABLE customer_segments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE segment_members (
            customer_id INTEGER NOT NULL,
            segment_id INTEGER NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (segment_id) REFERENCES customer_segments(id)
        );

        INSERT INTO regions VALUES
            (1, 'East'),
            (2, 'West'),
            (3, 'EMEA');

        INSERT INTO customers VALUES
            (1, 'Acme Corp', 1),
            (2, 'Northwind', 2),
            (3, 'Globex', 1),
            (4, 'Initech', 3);

        INSERT INTO departments VALUES
            (1, 'Sales'),
            (2, 'Operations');

        INSERT INTO employees VALUES
            (1, 1, 'Ava Chen', 'Sales Rep'),
            (2, 1, 'Noah Smith', 'Senior Sales Rep'),
            (3, 2, 'Mira Patel', 'Logistics Manager');

        INSERT INTO suppliers VALUES
            (1, 'Contoso Supply', 'USA'),
            (2, 'Fabrikam GmbH', 'Germany'),
            (3, 'Nod Publishers', 'UK');

        INSERT INTO orders VALUES
            (1, 1, 1, '2026-01-12', 1200.0, 'closed'),
            (2, 1, 1, '2026-02-05', 850.0, 'closed'),
            (3, 2, 2, '2026-02-20', 620.0, 'closed'),
            (4, 3, 2, '2026-03-01', 1430.0, 'open'),
            (5, 4, 1, '2026-03-18', 980.0, 'closed');

        INSERT INTO products VALUES
            (1, 1, 'Data Warehouse', 'Platform'),
            (2, 2, 'Analytics Seat', 'Subscription'),
            (3, 3, 'Notebook Pack', 'Office');

        INSERT INTO order_items VALUES
            (1, 1, 1, 1, 700.0),
            (2, 1, 2, 5, 100.0),
            (3, 2, 2, 3, 100.0),
            (4, 2, 3, 10, 55.0),
            (5, 3, 1, 1, 620.0),
            (6, 4, 2, 7, 100.0),
            (7, 4, 3, 10, 73.0),
            (8, 5, 1, 1, 980.0);

        INSERT INTO payments VALUES
            (1, 1, '2026-01-15', 1200.0, 'card'),
            (2, 2, '2026-02-06', 850.0, 'wire'),
            (3, 3, '2026-02-21', 620.0, 'card'),
            (4, 5, '2026-03-20', 980.0, 'wire');

        INSERT INTO warehouses VALUES
            (1, 1, 'East Fulfillment'),
            (2, 2, 'West Fulfillment'),
            (3, 3, 'EMEA Fulfillment');

        INSERT INTO shipments VALUES
            (1, 1, 1, '2026-01-13', 'delivered'),
            (2, 2, 1, '2026-02-08', 'late'),
            (3, 3, 2, '2026-02-22', 'delivered'),
            (4, 4, 1, '2026-03-05', 'late'),
            (5, 5, 3, '2026-03-21', 'delivered');

        INSERT INTO returns VALUES
            (1, 2, '2026-01-25', 1, 'seat transfer'),
            (2, 6, '2026-03-10', 2, 'over purchase'),
            (3, 7, '2026-03-12', 1, 'damaged');

        INSERT INTO campaigns VALUES
            (1, 'Q1 Launch', 'email'),
            (2, 'Partner Push', 'partner'),
            (3, 'Renewal Nudge', 'email');

        INSERT INTO order_campaigns VALUES
            (1, 1),
            (2, 3),
            (4, 2),
            (5, 1);

        INSERT INTO inventory VALUES
            (1, 1, 8),
            (1, 2, 45),
            (1, 3, 120),
            (2, 1, 5),
            (2, 2, 30),
            (3, 1, 3),
            (3, 3, 75);

        INSERT INTO invoices VALUES
            (1, 1, '2026-01-14', 'paid', 1200.0),
            (2, 2, '2026-02-05', 'paid', 850.0),
            (3, 3, '2026-02-20', 'paid', 620.0),
            (4, 4, '2026-03-02', 'open', 1430.0),
            (5, 5, '2026-03-19', 'paid', 980.0);

        INSERT INTO invoice_lines VALUES
            (1, 1, 1, 700.0),
            (2, 1, 2, 500.0),
            (3, 2, 2, 300.0),
            (4, 2, 3, 550.0),
            (5, 3, 1, 620.0),
            (6, 4, 2, 700.0),
            (7, 4, 3, 730.0),
            (8, 5, 1, 980.0);

        INSERT INTO subscriptions VALUES
            (1, 1, 2, '2026-01-01', 'active', 250.0),
            (2, 2, 1, '2026-01-15', 'active', 620.0),
            (3, 3, 2, '2026-02-01', 'paused', 180.0),
            (4, 4, 3, '2026-03-01', 'active', 95.0);

        INSERT INTO support_tickets VALUES
            (1, 1, 2, '2026-01-20', 'high', 'open'),
            (2, 2, 1, '2026-02-18', 'low', 'closed'),
            (3, 3, 2, '2026-03-04', 'high', 'open'),
            (4, 4, 3, '2026-03-22', 'medium', 'open');

        INSERT INTO ticket_events VALUES
            (1, 1, 'created', '2026-01-20'),
            (2, 1, 'escalated', '2026-01-21'),
            (3, 2, 'closed', '2026-02-19'),
            (4, 3, 'created', '2026-03-04'),
            (5, 4, 'created', '2026-03-22');

        INSERT INTO customer_segments VALUES
            (1, 'Enterprise'),
            (2, 'SMB'),
            (3, 'Strategic');

        INSERT INTO segment_members VALUES
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 2);
        """
    )
    return conn


def trace_to_json(trace: PipelineTrace) -> str:
    return json.dumps(asdict(trace), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic text-to-SQL prototype.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Show revenue by customer",
        help="Natural-language query to run through the pipeline.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use OpenAI-backed SQL generation. Requires OPENAI_API_KEY.",
    )
    args = parser.parse_args()

    schema = build_sample_schema()
    connection = build_sample_database()
    trace = TextToSQLAgent(schema, connection, use_llm=args.llm).run(args.query)
    print(trace_to_json(trace))


if __name__ == "__main__":
    main()
