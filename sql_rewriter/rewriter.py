"""SQL rewriter that intercepts queries, transforms them, and executes against DuckDB."""

from decimal import Decimal
import json
import os
import tempfile
from typing import Any, Optional, Union

from botocore.exceptions import BotoCoreError, ClientError
import duckdb
import sqlglot
from sqlglot import exp

from .policy import AggregateDFCPolicy, DFCPolicy, Resolution
from .rewrite_rule import (
    _extract_sink_expressions_from_constraint,
    _extract_source_aggregates_from_constraint,
    _find_outer_aggregate_for_inner,
    apply_aggregate_policy_constraints_to_aggregation,
    apply_aggregate_policy_constraints_to_scan,
    apply_policy_constraints_to_aggregation,
    apply_policy_constraints_to_scan,
    apply_policy_constraints_to_update,
    ensure_subqueries_have_constraint_columns,
    get_policy_identifier,
    rewrite_exists_subqueries_as_joins,
    rewrite_in_subqueries_as_joins,
    wrap_query_with_limit_in_cte_for_remove_policy,
)
from .sqlglot_utils import get_column_name, get_table_name_from_column


class SQLRewriter:
    """SQL rewriter that intercepts queries, transforms them, and executes against DuckDB."""

    def __init__(
        self,
        conn: Optional[duckdb.DuckDBPyConnection] = None,
        stream_file_path: Optional[str] = None,
        bedrock_client: Optional[Any] = None,
        bedrock_model_id: Optional[str] = None,
        recorder: Optional[Any] = None,
    ) -> None:

        if conn is not None:
            self.conn = conn
        else:
            self.conn = duckdb.connect()

        self._policies: list[DFCPolicy] = []
        self._aggregate_policies: list[AggregateDFCPolicy] = []
        self._sc_families: dict = {}

        self._bedrock_client = bedrock_client
        self._bedrock_model_id = bedrock_model_id or os.environ.get(
            "BEDROCK_MODEL_ID",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

        self._recorder = recorder
        self._replay_manager = None

        if stream_file_path is None:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as stream_file:
                self._stream_file_path = stream_file.name
        else:
            self._stream_file_path = stream_file_path

        self._register_kill_udf()
        self._register_address_violating_rows_udf()

    def set_recorder(self, recorder: Optional[Any]) -> None:
        """Set the recorder for LLM responses."""
        self._recorder = recorder

    def set_replay_manager(self, replay_manager: Optional[Any]) -> None:
        """Set the replay manager for replaying recorded responses."""
        self._replay_manager = replay_manager

    # -------------------------------------------------------------------------
    # Dimension table injection helpers
    # -------------------------------------------------------------------------

    def _inject_dimension_tables(
        self, parsed: exp.Select, dimension_tables: set[str]
    ) -> None:
        """Add dimension tables to a SELECT's FROM clause as CROSS JOINs.

        Only adds a table if it is not already present in the query, so we
        never produce a duplicate-table error.

        Args:
            parsed: The SELECT statement to modify in-place.
            dimension_tables: Lowercase names of dimension tables to inject.
        """
        if not dimension_tables:
            return

        existing_tables = {
            table.name.lower()
            for table in parsed.find_all(exp.Table)
            if table.find_ancestor(exp.From) or table.find_ancestor(exp.Join)
        }

        for dim_table in sorted(dimension_tables):  # sorted for determinism
            if dim_table not in existing_tables:
                join_expr = exp.Join(
                    this=exp.Table(this=exp.Identifier(this=dim_table, quoted=False)),
                    kind="CROSS",
                )
                existing_joins = list(parsed.args.get("joins") or [])
                parsed.set("joins", [*existing_joins, join_expr])
                existing_tables.add(dim_table)

    def _collect_dimension_tables(self, policies: list[DFCPolicy]) -> set[str]:
        """Collect all dimension table names (lowercase) from a list of policies."""
        result: set[str] = set()
        for policy in policies:
            if hasattr(policy, "_dimension_lower"):
                result.update(policy._dimension_lower)
        return result

    # -------------------------------------------------------------------------
    # SafeCoarsen helpers
    # -------------------------------------------------------------------------

    def _run_probe_query(self, parsed: exp.Select, policy: DFCPolicy, source_tables: set[str]) -> list[dict]:
        """Run a probe query to get group summaries needed by SafeCoarsen.

        Automatically extracts aggregate functions from the policy constraint
        that reference source tables and injects them as extra SELECT columns
        (_sc_agg_0, _sc_agg_1, ...). Stores the mapping on the policy as
        policy._sc_probe_map so predicate/merge functions can reference the
        column names without knowing the original aggregate SQL.

        Returns a list of dicts, one per output group, with keys matching
        all SELECT column names including the injected aggregates.
        """
        probe = parsed.copy()
        self._inject_dimension_tables(probe, self._collect_dimension_tables([policy]))

        group_clause = probe.args.get("group")
        if not group_clause or not group_clause.expressions:
            return []

        # Extract source aggregates from the policy constraint
        # e.g. COUNT(DISTINCT compensation.employee_id) -> _sc_agg_0
        probe_map: dict[str, str] = {}  # {agg_sql: alias_name}
        injected: list[exp.Expression] = []

        for source in policy.sources:
            source_aggs = _extract_source_aggregates_from_constraint(
                policy._constraint_parsed, source
            )
            for agg_expr in source_aggs:
                agg_sql = agg_expr.sql(dialect="duckdb")
                if agg_sql in probe_map:
                    continue  # already mapped
                alias_name = f"_sc_agg_{len(probe_map)}"
                probe_map[agg_sql] = alias_name
                injected.append(
                    exp.Alias(
                        this=agg_expr.copy(),
                        alias=exp.Identifier(this=alias_name, quoted=False),
                    )
                )

        # Inject extra aggregates into the probe SELECT
        if injected:
            probe.set("expressions", list(probe.expressions) + injected)

        # Store the mapping on the policy so caller can reference _sc_agg_0 etc.
        policy._sc_probe_map = probe_map

        # Execute and return as list of dicts
        try:
            cursor = self.conn.execute(probe.sql(dialect="duckdb"))
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            return []

    def _inject_coarsening(
        self,
        parsed: exp.Select,
        coarsening_map: dict[str, str | None],
        group_col_name: str,
    ) -> exp.Select:
        """Rewrite a GROUP BY query by injecting a CASE expression.

        Replaces the grouping column reference with a CASE expression that
        maps original group values to their coarsened labels. Suppressed
        groups (None value in map) are excluded via a WHERE clause.

        Args:
            parsed: The original SELECT statement.
            coarsening_map: {original_value: coarsened_label | None}
            group_col_name: The name of the grouping column to coarsen.

        Returns:
            A new SELECT statement with the CASE expression injected.
        """
        rewritten = parsed.copy()

        # Build CASE expression: CASE WHEN col = 'v1' THEN 'label1' ... END
        ifs = []
        for orig, coarsened in coarsening_map.items():
            if coarsened is None:
                continue
            ifs.append(exp.If(
                this=exp.EQ(
                    this=exp.Column(this=exp.Identifier(this=group_col_name, quoted=False)),
                    expression=exp.Literal.string(orig),
                ),
                true=exp.Literal.string(coarsened),
            ))

        if not ifs:
            return rewritten

        case_expr = exp.Case(ifs=ifs, default=exp.Literal.string("__suppressed__"))

        # Replace the grouping column in SELECT expressions
        def replace_group_col(node):
            if (
                isinstance(node, exp.Column)
                and get_column_name(node).lower() == group_col_name.lower()
                and not node.find_ancestor(exp.AggFunc)
            ):
                return case_expr.copy()
            return node

        new_expressions = []
        for expr in rewritten.expressions:
            new_expressions.append(expr.transform(replace_group_col, copy=True))
        rewritten.set("expressions", new_expressions)

        # Replace GROUP BY with the CASE expression
        rewritten.set("group", exp.Group(expressions=[case_expr.copy()]))

        # Add WHERE to suppress None-mapped groups
        suppressed = [orig for orig, label in coarsening_map.items() if label is None]
        if suppressed:
            existing_where = rewritten.args.get("where")
            suppress_condition = exp.Not(
                this=exp.In(
                    this=exp.Column(this=exp.Identifier(this=group_col_name, quoted=False)),
                    expressions=[exp.Literal.string(v) for v in suppressed],
                )
            )
            if existing_where:
                new_where = exp.Where(
                    this=exp.And(this=existing_where.this, expression=suppress_condition)
                )
            else:
                new_where = exp.Where(this=suppress_condition)
            rewritten.set("where", new_where)

        return rewritten
    
    def _run_safecoarsen_rewrite(
        self,
        parsed: exp.Select,
        policy: DFCPolicy,
        source_tables: set[str],
    ) -> exp.Select:
        """Two-pass SafeCoarsen rewrite.

        Pass 1: run probe query to get group summaries.
        Pass 2: run SafeCoarsen to compute coarsening map.
        Pass 3: inject CASE expression into original query.

        If the original query has LIMIT/ORDER BY, those are stripped before
        coarsening and re-applied as an outer CTE wrapper so the LIMIT fires
        AFTER coarsening, not before.

        Falls back to original parsed query if SafeCoarsen cannot produce
        a safe repair (caller's HAVING safety net still applies).
        """
        from .safe_coarsen import safe_coarsen, build_hierarchy

        # Identify the grouping column
        group_clause = parsed.args.get("group")
        if not group_clause or not group_clause.expressions:
            return parsed

        group_expr = group_clause.expressions[0]
        group_col_name = (
            get_column_name(group_expr)
            if isinstance(group_expr, exp.Column)
            else group_expr.sql()
        )
        
        group_table = (
        get_table_name_from_column(group_expr)
        if isinstance(group_expr, exp.Column)
        else None
        )
        if group_table and group_table.lower() not in {s.lower() for s in policy.sources}:
            return parsed

        # Stash LIMIT and ORDER BY — strip them before probing and coarsening
        # They will be re-applied on the outer CTE after coarsening
        limit_expr = parsed.args.get("limit")
        order_expr = parsed.args.get("order")

        probe_base = parsed.copy()
        probe_base.set("limit", None)
        probe_base.set("order", None)
        probe_base.set("having", None) 

        # Pass 1: run probe query on the limit-stripped version
        probe_rows = self._run_probe_query(probe_base, policy, source_tables)
        if not probe_rows:
            return parsed

        # Load hierarchy from DB
        try:
            hier_rows = self.conn.execute(
                f"SELECT node, parent FROM {policy.hierarchy}"
            ).fetchall()
        except Exception:
            return parsed

        hierarchy = build_hierarchy(hier_rows)

        # Build groups list for SafeCoarsen — "value" is the grouping column
        groups = []
        for row in probe_rows:
            group = dict(row)
            group["value"] = row.get(group_col_name, row.get(group_col_name.lower()))
            groups.append(group)

        # Predicate, merge, label — attached to policy at demo/call time
        # Resolve predicate/merge/label — prefer registry over setattr
        family_name = getattr(policy, "predicate_family", None)
        if family_name and family_name in self._sc_families:
            family = self._sc_families[family_name]
            # Read current dimension values to instantiate predicate/merge
            try:
                dim_table = next(iter(policy._dimension_lower), "users")
                cursor = self.conn.execute(f"SELECT * FROM {dim_table} LIMIT 1")
                dim_cols = [d[0] for d in cursor.description]
                dim_row = cursor.fetchone()
                dim_values = dict(zip(dim_cols, dim_row)) if dim_row else {}
            except Exception:
                dim_values = {}
            predicate = family["predicate"](dim_values)
            merge_fn  = family["merge"](dim_values)
            label_fn  = family["label"]
        else:
            # Fall back to setattr (backwards compatible)
            predicate = getattr(policy, "_sc_predicate", None)
            merge_fn  = getattr(policy, "_sc_merge", None)
            label_fn  = getattr(policy, "_sc_label", None)

        if predicate is None or merge_fn is None or label_fn is None:
            return parsed

        coarsening_map = safe_coarsen(groups, hierarchy, predicate, merge_fn, label_fn)

        # If everything maps to None (all suppressed) — fall back to KILL
        if all(v is None for v in coarsening_map.values()):
            return parsed

        # Inject coarsening on the limit-stripped base
        coarsened = self._inject_coarsening(probe_base, coarsening_map, group_col_name)
        
        # Inject dimension tables (e.g. users) before HAVING so MAX(users.k_threshold) resolves
        self._inject_dimension_tables(coarsened, self._collect_dimension_tables([policy]))
        
        # Inject HAVING safety net — standard path is skipped for coarsen policies
        apply_policy_constraints_to_aggregation(
            coarsened,
            [policy],
            source_tables,
            stream_file_path=self._stream_file_path,
        )

        # If there was no LIMIT/ORDER BY, return directly
        if limit_expr is None and order_expr is None:
            return coarsened

        # Re-apply LIMIT/ORDER BY as an outer CTE so they fire AFTER coarsening
        # WITH coarsened_q AS (... CASE expression ... HAVING ...)
        # SELECT * FROM coarsened_q ORDER BY ... LIMIT n
        cte = exp.CTE(
            this=coarsened,
            alias=exp.TableAlias(
                this=exp.Identifier(this="coarsened_q", quoted=False)
            ),
        )

        # Build outer SELECT — project all columns from CTE
        outer_select = sqlglot.parse_one(
            "SELECT * FROM coarsened_q", read="duckdb"
        )
        if not isinstance(outer_select, exp.Select):
            return coarsened

        if order_expr is not None:
            # Strip table qualifiers from ORDER BY since we're now selecting from CTE
            def strip_table(node):
                if isinstance(node, exp.Column) and node.table:
                    return exp.Column(
                        this=exp.Identifier(
                            this=get_column_name(node), quoted=False
                        )
                    )
                return node
            outer_select.set(
                "order",
                order_expr.copy().transform(strip_table, copy=True)
            )

        if limit_expr is not None:
            outer_select.set("limit", limit_expr.copy())

        outer_select.set("with_", exp.With(expressions=[cte]))
        return outer_select

    # -------------------------------------------------------------------------
    # Query transformation entry points
    # -------------------------------------------------------------------------

    def transform_query(self, query: str, use_two_phase: bool = False) -> str:
        """Transform a SQL query according to the rewriter's rules."""
        parsed = sqlglot.parse_one(query, read="duckdb")
        if use_two_phase:
            transformed = self._transform_query_two_phase(parsed)
        else:
            transformed = self._transform_query_standard(parsed)
        return transformed.sql(pretty=True, dialect="duckdb")

    def _transform_query_standard(self, parsed: exp.Expression) -> exp.Expression:
        """Apply standard DFC rewriting rules to a parsed query."""
        return self._transform_query_common(parsed, use_two_phase=False)

    def _transform_query_two_phase(self, parsed: exp.Expression) -> exp.Expression:
        """Apply two-phase rewriting rules to a parsed query."""
        return self._transform_query_common(parsed, use_two_phase=True)

    def _transform_query_common(
        self, parsed: exp.Expression, use_two_phase: bool
    ) -> exp.Expression:
        """Apply rewriting rules shared by standard DFC and two-phase paths."""
        if isinstance(parsed, exp.Select):
            # _get_source_tables is called BEFORE we inject any dimension tables,
            # so dimension tables can never pollute from_tables and trigger
            # accidental policy matches.
            from_tables = self._get_source_tables(parsed)

            if from_tables:
                matching_policies = self._find_matching_policies(
                    source_tables=from_tables, sink_table=None
                )
                matching_aggregate_policies = self._find_matching_aggregate_policies(
                    source_tables=from_tables, sink_table=None
                )

                if matching_policies:
                    if not use_two_phase:
                        rewrite_in_subqueries_as_joins(parsed, matching_policies, from_tables)
                        rewrite_exists_subqueries_as_joins(parsed, matching_policies, from_tables)
                        # Second call still happens before dimension injection — safe.
                        from_tables = self._get_source_tables(parsed)

                    # SafeCoarsen: if any matching policy has repair_mode='coarsen'
                    # and query has aggregations, run two-pass rewrite before
                    # standard constraint application.
                    coarsen_policies = [
                        p for p in matching_policies
                        if getattr(p, "repair_mode", None) == "coarsen"
                        and p.hierarchy
                    ]
                    if coarsen_policies and self._has_group_by(parsed) and not use_two_phase:
                        original_parsed = parsed
                        parsed = self._run_safecoarsen_rewrite(
                            parsed, coarsen_policies[0], from_tables
                        )
                        # Only strip coarsen policy if SafeCoarsen actually rewrote the query.
                        # If it fell back (returned same object), keep it in matching_policies
                        # so standard enforcement still applies.
                        if parsed is not original_parsed:
                            matching_policies = [
                                p for p in matching_policies
                                if getattr(p, "repair_mode", None) != "coarsen"
                            ]

                    has_limit = parsed.args.get("limit") is not None
                    has_remove_policy = any(
                        p.on_fail == Resolution.REMOVE for p in matching_policies
                    )

                    if has_limit and has_remove_policy:
                        remove_policy = next(
                            p for p in matching_policies if p.on_fail == Resolution.REMOVE
                        )
                        if use_two_phase:
                            if self._has_aggregations(parsed):
                                parsed = self._rewrite_limit_aggregation_with_two_phase(
                                    parsed,
                                    matching_policies,
                                    from_tables,
                                    remove_policy,
                                )
                            else:
                                wrap_query_with_limit_in_cte_for_remove_policy(
                                    parsed,
                                    remove_policy,
                                    from_tables,
                                    is_aggregation=False,
                                )
                        else:
                            is_aggregation = self._has_aggregations(parsed)
                            wrap_query_with_limit_in_cte_for_remove_policy(
                                parsed, remove_policy, from_tables, is_aggregation
                            )
                    else:
                        if not (use_two_phase and self._has_aggregations(parsed)):
                            ensure_subqueries_have_constraint_columns(
                                parsed, matching_policies, from_tables
                            )

                        # Inject dimension tables into the main query right before
                        # constraint application (standard path only — two-phase
                        # handles injection inside its own rewrite methods).
                        if not use_two_phase:
                            dim_tables = self._collect_dimension_tables(matching_policies)
                            self._inject_dimension_tables(parsed, dim_tables)

                        if self._has_aggregations(parsed):
                            if use_two_phase:
                                parsed = self._rewrite_aggregation_with_two_phase(
                                    parsed,
                                    matching_policies,
                                    from_tables,
                                )
                            else:
                                apply_policy_constraints_to_aggregation(
                                    parsed,
                                    matching_policies,
                                    from_tables,
                                    stream_file_path=self._stream_file_path,
                                )
                        else:
                            apply_policy_constraints_to_scan(
                                parsed,
                                matching_policies,
                                from_tables,
                                stream_file_path=self._stream_file_path,
                            )

                if matching_aggregate_policies:
                    if self._has_aggregations(parsed):
                        apply_aggregate_policy_constraints_to_aggregation(
                            parsed, matching_aggregate_policies, from_tables
                        )
                    else:
                        apply_aggregate_policy_constraints_to_scan(
                            parsed, matching_aggregate_policies, from_tables
                        )

        elif isinstance(parsed, exp.Insert):
            sink_table = self._get_sink_table(parsed)
            source_tables = self._get_insert_source_tables(parsed)

            matching_policies = self._find_matching_policies(
                source_tables=source_tables, sink_table=sink_table
            )
            matching_aggregate_policies = self._find_matching_aggregate_policies(
                source_tables=source_tables, sink_table=sink_table
            )

            select_expr = parsed.find(exp.Select)

            sink_to_output_mapping = None
            if select_expr and sink_table:
                self._add_aliases_to_insert_select_outputs(parsed, select_expr)
                sink_to_output_mapping = self._get_insert_column_mapping(parsed, select_expr)

            if matching_policies:
                has_invalidate_with_sink = any(
                    p.on_fail == Resolution.INVALIDATE and p.sink
                    for p in matching_policies
                )
                has_invalidate_message_with_sink = any(
                    p.on_fail == Resolution.INVALIDATE_MESSAGE and p.sink
                    for p in matching_policies
                )

                if select_expr:
                    if has_invalidate_with_sink and sink_table:
                        self._add_valid_column_to_insert(parsed)
                    if has_invalidate_message_with_sink and sink_table:
                        self._add_invalid_string_column_to_insert(parsed)

                    if not sink_to_output_mapping and sink_table:
                        sink_to_output_mapping = self._get_insert_column_mapping(
                            parsed, select_expr
                        )

                    insert_columns = self._get_insert_column_list(parsed)

                    ensure_subqueries_have_constraint_columns(
                        select_expr, matching_policies, source_tables
                    )

                    # Inject dimension tables into the SELECT part of INSERT ... SELECT
                    # before applying constraints.
                    dim_tables = self._collect_dimension_tables(matching_policies)
                    self._inject_dimension_tables(select_expr, dim_tables)

                    insert_has_valid = False
                    insert_has_invalid_string = False
                    if (
                        hasattr(parsed, "this")
                        and isinstance(parsed.this, exp.Schema)
                        and hasattr(parsed.this, "expressions")
                        and parsed.this.expressions
                    ):
                        for col in parsed.this.expressions:
                            col_name = None
                            if isinstance(col, exp.Identifier):
                                col_name = col.name.lower()
                            elif isinstance(col, exp.Column):
                                col_name = get_column_name(col).lower()
                            elif isinstance(col, str):
                                col_name = col.lower()
                            if col_name == "valid":
                                insert_has_valid = True
                            if col_name == "invalid_string":
                                insert_has_invalid_string = True
                            if insert_has_valid and insert_has_invalid_string:
                                break

                    if self._has_aggregations(select_expr):
                        apply_policy_constraints_to_aggregation(
                            select_expr,
                            matching_policies,
                            source_tables,
                            stream_file_path=self._stream_file_path,
                            sink_table=sink_table,
                            sink_to_output_mapping=sink_to_output_mapping,
                            replace_existing_valid=insert_has_valid,
                            replace_existing_invalid_string=insert_has_invalid_string,
                            insert_columns=insert_columns,
                        )
                    else:
                        apply_policy_constraints_to_scan(
                            select_expr,
                            matching_policies,
                            source_tables,
                            stream_file_path=self._stream_file_path,
                            sink_table=sink_table,
                            sink_to_output_mapping=sink_to_output_mapping,
                            replace_existing_valid=insert_has_valid,
                            replace_existing_invalid_string=insert_has_invalid_string,
                            insert_columns=insert_columns,
                        )

            if matching_aggregate_policies and select_expr:
                has_aggs = self._has_aggregations(select_expr)
                if has_aggs:
                    apply_aggregate_policy_constraints_to_aggregation(
                        select_expr,
                        matching_aggregate_policies,
                        source_tables,
                        sink_table=sink_table,
                        sink_to_output_mapping=sink_to_output_mapping,
                    )
                else:
                    apply_aggregate_policy_constraints_to_scan(
                        select_expr,
                        matching_aggregate_policies,
                        source_tables,
                        sink_table=sink_table,
                        sink_to_output_mapping=sink_to_output_mapping,
                    )

                self._add_aggregate_temp_columns_to_insert(
                    parsed, matching_aggregate_policies, select_expr
                )

        elif isinstance(parsed, exp.Update):
            sink_table = self._get_update_target_table(parsed)
            source_tables = self._get_update_source_tables(parsed)

            matching_policies = self._find_matching_policies(
                source_tables=source_tables, sink_table=sink_table
            )
            matching_aggregate_policies = self._find_matching_aggregate_policies(
                source_tables=source_tables, sink_table=sink_table
            )

            if matching_aggregate_policies:
                raise ValueError("Aggregate policies are not supported for UPDATE statements")

            if matching_policies and sink_table:
                apply_policy_constraints_to_update(
                    parsed,
                    matching_policies,
                    source_tables,
                    sink_table=sink_table,
                    sink_assignments=self._get_update_assignment_mapping(parsed),
                    target_reference_name=self._get_update_target_reference_name(parsed),
                    stream_file_path=self._stream_file_path,
                )

        return parsed

    # -------------------------------------------------------------------------
    # Two-phase rewrite helpers
    # -------------------------------------------------------------------------

    def _extract_policy_comparison(
        self,
        policy: DFCPolicy,
    ) -> tuple[exp.Expression, exp.Expression, type[exp.Expression]]:
        """Extract left and right sides of the policy comparison expression."""
        constraint_expr = policy._constraint_parsed.copy()
        for op_class in (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ):
            comparisons = list(constraint_expr.find_all(op_class))
            if comparisons:
                comparison = comparisons[0]
                return comparison.this.copy(), comparison.expression.copy(), op_class
        raise ValueError(
            f"Unsupported constraint shape for two-phase LIMIT rewrite: {policy.constraint}"
        )

    def _build_comparison_expr(
        self,
        left: exp.Expression,
        right: exp.Expression,
        op_class: type[exp.Expression],
    ) -> exp.Expression:
        """Build a comparison expression from left/right operands and operator class."""
        if op_class == exp.GT:
            return exp.GT(this=left, expression=right)
        if op_class == exp.GTE:
            return exp.GTE(this=left, expression=right)
        if op_class == exp.LT:
            return exp.LT(this=left, expression=right)
        if op_class == exp.LTE:
            return exp.LTE(this=left, expression=right)
        if op_class == exp.EQ:
            return exp.EQ(this=left, expression=right)
        if op_class == exp.NEQ:
            return exp.NEQ(this=left, expression=right)
        raise ValueError(
            f"Unsupported comparison operator for two-phase LIMIT rewrite: {op_class}"
        )

    def _auto_alias_name_for_expression(self, expr: exp.Expression) -> str:
        """Generate a stable alias for non-column projection expressions."""
        alias_name = (
            expr.sql()
            .lower()
            .replace("(", "_")
            .replace(")", "")
            .replace(" ", "_")
            .replace(",", "_")
        )
        if alias_name and not alias_name[0].isalpha():
            alias_name = f"expr_{alias_name}"
        return alias_name[:50]

    def _ensure_projection_aliases(self, parsed: exp.Select) -> None:
        """Ensure projected expressions are columns or aliases with stable names."""
        aliased_exprs: list[exp.Expression] = []
        for expr in parsed.expressions:
            if isinstance(expr, (exp.Star, exp.Alias, exp.Column)):
                aliased_exprs.append(expr)
            else:
                aliased_exprs.append(
                    exp.Alias(
                        this=expr.copy(),
                        alias=exp.Identifier(
                            this=self._auto_alias_name_for_expression(expr),
                            quoted=False,
                        ),
                    )
                )
        parsed.set("expressions", aliased_exprs)

    def _outer_projection_from_original(
        self, parsed: exp.Select
    ) -> list[exp.Expression]:
        """Build outer SELECT projection that references CTE output columns by name."""
        outer_expressions: list[exp.Expression] = []
        for expr in parsed.expressions:
            if isinstance(expr, exp.Star):
                outer_expressions.append(expr.copy())
            elif isinstance(expr, exp.Alias):
                alias_name = get_column_name(expr.alias)
                outer_expressions.append(
                    exp.Column(this=exp.Identifier(this=alias_name, quoted=False))
                )
            elif isinstance(expr, exp.Column):
                col_name = get_column_name(expr)
                outer_expressions.append(
                    exp.Column(this=exp.Identifier(this=col_name, quoted=False))
                )
            else:
                raise ValueError("Outer projection must be normalized before building")
        return outer_expressions

    def _rewrite_limit_aggregation_with_two_phase(
        self,
        parsed: exp.Select,
        policies: list[DFCPolicy],
        source_tables: set[str],
        remove_policy: DFCPolicy,
    ) -> exp.Select:
        """Rewrite LIMIT+REMOVE aggregation queries using two-phase evaluation."""
        group_specs = self._group_by_join_specs(parsed)
        if group_specs is None:
            raise ValueError(
                "Two-phase LIMIT rewrite requires GROUP BY expressions to be projected in SELECT "
                "with stable output names"
            )
        group_keys = [key_name for key_name, _ in group_specs]

        projection_query = parsed.copy()
        self._ensure_projection_aliases(projection_query)

        base_query = projection_query.copy()
        base_query.set("order", None)
        base_query.set("limit", None)

        policy_eval = parsed.copy()
        rewrite_in_subqueries_as_joins(policy_eval, policies, source_tables)
        rewrite_exists_subqueries_as_joins(policy_eval, policies, source_tables)
        ensure_subqueries_have_constraint_columns(policy_eval, policies, source_tables)
        policy_eval.set("order", None)
        policy_eval.set("limit", None)
        policy_eval.set("having", None)

        # Inject dimension tables into policy_eval only, not base_query.
        dim_tables = self._collect_dimension_tables(policies)
        self._inject_dimension_tables(policy_eval, dim_tables)

        extra_dfc_aliases: list[exp.Expression] = []
        extra_dfc_filters: list[tuple[str, type[exp.Expression], exp.Expression]] = []
        if hasattr(policy_eval, "meta"):
            extra_dfc_aliases = [
                alias.copy() for alias in policy_eval.meta.get("extra_dfc_aliases", [])
            ]
            extra_dfc_filters = policy_eval.meta.get("extra_dfc_filters", [])

        dfc_expr, threshold_expr, op_class = self._extract_policy_comparison(remove_policy)

        policy_select_exprs: list[exp.Expression] = []
        if group_keys:
            for key_name, key_expr in group_specs:
                policy_select_exprs.append(
                    exp.Alias(
                        this=key_expr.copy(),
                        alias=exp.Identifier(this=key_name, quoted=False),
                    )
                )
        else:
            policy_select_exprs.append(
                exp.Alias(
                    this=exp.Literal.number(1),
                    alias=exp.Identifier(this="__dfc_two_phase_key", quoted=False),
                )
            )

        policy_select_exprs.append(
            exp.Alias(
                this=dfc_expr,
                alias=exp.Identifier(this="dfc", quoted=False),
            )
        )
        policy_select_exprs.extend(extra_dfc_aliases)
        policy_eval.set("expressions", policy_select_exprs)

        base_query_star = sqlglot.parse_one("SELECT base_query.*", read="duckdb").expressions[0]
        cte_select_exprs: list[exp.Expression] = [
            base_query_star,
            exp.Alias(
                this=exp.Column(
                    this=exp.Identifier(this="dfc", quoted=False),
                    table=exp.Identifier(this="policy_eval", quoted=False),
                ),
                alias=exp.Identifier(this="dfc", quoted=False),
            ),
        ]
        for extra_alias in extra_dfc_aliases:
            if not isinstance(extra_alias, exp.Alias) or not extra_alias.alias:
                continue
            extra_alias_name = get_column_name(extra_alias.alias)
            cte_select_exprs.append(
                exp.Alias(
                    this=exp.Column(
                        this=exp.Identifier(this=extra_alias_name, quoted=False),
                        table=exp.Identifier(this="policy_eval", quoted=False),
                    ),
                    alias=exp.Identifier(this=extra_alias_name, quoted=False),
                )
            )

        cte_body_sql = self._two_phase_join_clause(group_keys)
        cte_body = sqlglot.parse_one(f"SELECT 1 {cte_body_sql}", read="duckdb")
        if not isinstance(cte_body, exp.Select):
            raise ValueError("Two-phase LIMIT rewrite failed to build CTE body")
        cte_body.set("expressions", cte_select_exprs)
        cte_body.set(
            "order",
            self._qualify_two_phase_order_columns(parsed.args.get("order"), group_keys),
        )
        cte_body.set(
            "limit",
            parsed.args.get("limit").copy() if parsed.args.get("limit") else None,
        )

        cte = exp.CTE(
            this=cte_body,
            alias=exp.TableAlias(this=exp.Identifier(this="cte", quoted=False)),
        )

        outer_from = exp.From(
            this=exp.Table(this=exp.Identifier(this="cte", quoted=False))
        )
        combined_where = self._build_comparison_expr(
            exp.Column(this=exp.Identifier(this="dfc", quoted=False)),
            threshold_expr,
            op_class,
        )
        for dfc_name, extra_op_class, extra_threshold in extra_dfc_filters:
            combined_where = exp.And(
                this=combined_where,
                expression=self._build_comparison_expr(
                    exp.Column(this=exp.Identifier(this=dfc_name, quoted=False)),
                    extra_threshold.copy(),
                    extra_op_class,
                ),
            )

        outer_where = exp.Where(this=combined_where)
        outer_select = exp.Select(
            expressions=self._outer_projection_from_original(projection_query),
            from_=outer_from,
            where=outer_where,
        )

        existing_with = parsed.args.get("with_")
        with_exprs = []
        if existing_with:
            with_exprs.extend(existing_with.expressions)
        with_exprs.extend(
            [
                exp.CTE(
                    this=base_query,
                    alias=exp.TableAlias(
                        this=exp.Identifier(this="base_query", quoted=False)
                    ),
                ),
                exp.CTE(
                    this=policy_eval,
                    alias=exp.TableAlias(
                        this=exp.Identifier(this="policy_eval", quoted=False)
                    ),
                ),
                cte,
            ]
        )
        outer_select.set("with_", exp.With(expressions=with_exprs))
        return outer_select

    def _group_by_join_specs(
        self, parsed: exp.Select
    ) -> list[tuple[str, exp.Expression]] | None:
        """Map GROUP BY expressions to output key names and policy-eval expressions."""
        group_clause = parsed.args.get("group")
        if not group_clause or not group_clause.expressions:
            return []

        select_exprs = list(parsed.expressions or [])
        alias_expr_map: dict[str, exp.Expression] = {}
        for select_expr in select_exprs:
            if isinstance(select_expr, exp.Alias) and select_expr.alias:
                alias_expr_map[select_expr.alias.lower()] = select_expr.this.copy()
            elif isinstance(select_expr, exp.Column):
                alias_expr_map[get_column_name(select_expr).lower()] = select_expr.copy()

        keys: list[tuple[str, exp.Expression]] = []
        for group_expr in group_clause.expressions:
            target_sql = group_expr.sql(dialect="duckdb")
            matched_key = None
            matched_expr = None
            for select_expr in select_exprs:
                if isinstance(select_expr, exp.Alias):
                    if (
                        select_expr.this.sql(dialect="duckdb") == target_sql
                        and select_expr.alias
                    ):
                        matched_key = select_expr.alias
                        matched_expr = select_expr.this.copy()
                        break
                elif (
                    select_expr.sql(dialect="duckdb") == target_sql
                    and isinstance(select_expr, exp.Column)
                ):
                    matched_key = get_column_name(select_expr)
                    matched_expr = select_expr.copy()
                    break

            if matched_key is None:
                alias_ref = None
                if isinstance(group_expr, exp.Identifier):
                    alias_ref = group_expr.this
                elif (
                    isinstance(group_expr, exp.Column)
                    and get_table_name_from_column(group_expr) is None
                ):
                    alias_ref = get_column_name(group_expr)
                if alias_ref:
                    alias_expr = alias_expr_map.get(alias_ref.lower())
                    if alias_expr is not None:
                        matched_key = alias_ref
                        matched_expr = alias_expr.copy()

            if matched_key is None:
                return None
            keys.append(
                (matched_key, matched_expr if matched_expr is not None else group_expr.copy())
            )
        return keys

    def _rewrite_aggregation_with_two_phase(
        self,
        parsed: exp.Select,
        policies: list[DFCPolicy],
        source_tables: set[str],
    ) -> exp.Select:
        """Rewrite aggregation using two-phase policy evaluation."""
        group_specs = self._group_by_join_specs(parsed)
        if group_specs is None:
            raise ValueError(
                "Two-phase rewrite requires GROUP BY expressions to be projected in SELECT "
                "with stable output names"
            )
        group_keys = [key_name for key_name, _ in group_specs]

        include_valid = any(policy.on_fail == Resolution.INVALIDATE for policy in policies)
        include_invalid_string = any(
            policy.on_fail == Resolution.INVALIDATE_MESSAGE for policy in policies
        )
        base_query = parsed.copy()
        policy_eval = parsed.copy()
        rewrite_in_subqueries_as_joins(policy_eval, policies, source_tables)
        rewrite_exists_subqueries_as_joins(policy_eval, policies, source_tables)
        ensure_subqueries_have_constraint_columns(policy_eval, policies, source_tables)
        policy_eval.set("order", None)
        policy_eval.set("limit", None)
        policy_eval.set("having", None)

        # Inject dimension tables into policy_eval only, not base_query.
        dim_tables = self._collect_dimension_tables(policies)
        self._inject_dimension_tables(policy_eval, dim_tables)

        if group_keys:
            policy_select_exprs: list[exp.Expression] = []
            for key_name, key_expr in group_specs:
                policy_select_exprs.append(
                    exp.Alias(
                        this=key_expr.copy(),
                        alias=exp.Identifier(this=key_name, quoted=False),
                    )
                )
            policy_eval.set("expressions", policy_select_exprs)
        else:
            policy_eval.set(
                "expressions",
                [
                    exp.Alias(
                        this=exp.Literal.number(1),
                        alias=exp.Identifier(this="__dfc_two_phase_key", quoted=False),
                    )
                ],
            )

        apply_policy_constraints_to_aggregation(
            policy_eval,
            policies,
            source_tables,
            stream_file_path=self._stream_file_path,
        )

        select_list = "base_query.*"
        if include_valid:
            select_list += ", policy_eval.valid AS valid"
        if include_invalid_string:
            select_list += ", policy_eval.invalid_string AS invalid_string"
        outer_sql = f"SELECT {select_list} {self._two_phase_join_clause(group_keys)}"

        rewritten = sqlglot.parse_one(outer_sql, read="duckdb")
        if not isinstance(rewritten, exp.Select):
            raise ValueError("Two-phase aggregation rewrite must produce a SELECT statement")
        rewritten.set(
            "with_",
            exp.With(
                expressions=[
                    exp.CTE(
                        this=base_query,
                        alias=exp.TableAlias(
                            this=exp.Identifier(this="base_query", quoted=False)
                        ),
                    ),
                    exp.CTE(
                        this=policy_eval,
                        alias=exp.TableAlias(
                            this=exp.Identifier(this="policy_eval", quoted=False)
                        ),
                    ),
                ]
            ),
        )
        return rewritten

    def _two_phase_join_clause(self, key_names: list[str]) -> str:
        if not key_names:
            return "FROM base_query CROSS JOIN policy_eval"
        join_conditions = " AND ".join(
            [
                f"base_query.{key_name} = policy_eval.{key_name}"
                for key_name in key_names
            ]
        )
        return f"FROM base_query JOIN policy_eval ON {join_conditions}"

    def _qualify_two_phase_order_columns(
        self, order_expr: exp.Order | None, key_names: list[str]
    ) -> exp.Order | None:
        if order_expr is None or not key_names:
            return order_expr
        qualified = order_expr.copy()
        key_name_set = {key.lower() for key in key_names}
        for column in qualified.find_all(exp.Column):
            if column.table:
                continue
            column_name = get_column_name(column)
            if column_name.lower() in key_name_set:
                column.set("table", exp.Identifier(this="base_query", quoted=False))
        return qualified

    def _scan_join_keys(
        self, parsed: exp.Select
    ) -> list[tuple[str, exp.Expression]] | None:
        """Extract join keys from SELECT outputs for two-phase scan rewrites."""
        keys: list[tuple[str, exp.Expression]] = []
        for select_expr in parsed.expressions or []:
            if isinstance(select_expr, exp.Star):
                return None
            if isinstance(select_expr, exp.Alias):
                if not select_expr.alias:
                    return None
                key_name = select_expr.alias
                key_expr = select_expr.this.copy()
            elif isinstance(select_expr, exp.Column):
                key_name = get_column_name(select_expr)
                key_expr = select_expr.copy()
            else:
                return None
            if not key_name:
                return None
            keys.append((key_name, key_expr))
        return keys if keys else None

    def _can_use_rowid_scan_join(self, parsed: exp.Select) -> bool:
        """Whether we can safely use rowid as the two-phase scan join key."""
        if parsed.args.get("distinct") is not None:
            return False
        from_clause = parsed.args.get("from_")
        if not isinstance(from_clause, exp.From):
            return False
        joins = parsed.args.get("joins") or []
        return len(joins) == 0

    def _select_has_named_projection(self, parsed: exp.Select, name: str) -> bool:
        target = name.lower()
        for select_expr in parsed.expressions or []:
            if isinstance(select_expr, exp.Alias):
                if select_expr.alias and select_expr.alias.lower() == target:
                    return True
            elif (
                isinstance(select_expr, exp.Column)
                and get_column_name(select_expr).lower() == target
            ):
                return True
        return False

    def _rewrite_scan_with_two_phase(
        self,
        parsed: exp.Select,
        policies: list[DFCPolicy],
        source_tables: set[str],
    ) -> exp.Select:
        """Rewrite non-aggregation query using two-phase policy evaluation."""
        join_keys = self._scan_join_keys(parsed)
        use_rowid_join = self._can_use_rowid_scan_join(parsed) and (
            join_keys is None or len(join_keys) > 1
        )

        if join_keys is None and not use_rowid_join:
            raise ValueError(
                "Two-phase rewrite requires explicit projected join keys "
                "(no SELECT *, and computed expressions must be aliased)"
            )

        include_valid = any(policy.on_fail == Resolution.INVALIDATE for policy in policies)
        include_invalid_string = any(
            policy.on_fail == Resolution.INVALIDATE_MESSAGE for policy in policies
        )

        base_query = parsed.copy()
        policy_eval = parsed.copy()
        policy_eval.set("order", None)
        policy_eval.set("limit", None)
        base_has_star = False
        rowid_source_name = None

        if use_rowid_join:
            policy_eval.set("distinct", None)
            from_clause = parsed.args.get("from_")
            from_source = from_clause.this if isinstance(from_clause, exp.From) else None
            if isinstance(from_source, exp.Table):
                rowid_source_name = from_source.name.lower()

            if rowid_source_name is not None and rowid_source_name in source_tables:
                rowid_expr = exp.Column(this=exp.Identifier(this="rowid", quoted=False))
            else:
                for policy in policies:
                    for source in policy.sources:
                        source_lower = source.lower()
                        if source_lower in policy._source_columns_needed:
                            policy._source_columns_needed[source_lower].add("__dfc_rowid")
                ensure_subqueries_have_constraint_columns(policy_eval, policies, source_tables)
                ensure_subqueries_have_constraint_columns(base_query, policies, source_tables)
                rowid_expr = exp.Column(
                    this=exp.Identifier(this="__dfc_rowid", quoted=False)
                )

            rowid_alias = "__dfc_rowid"
            base_has_star = any(
                isinstance(expr, exp.Star) for expr in (base_query.expressions or [])
            )
            should_append_rowid = (
                not self._select_has_named_projection(base_query, rowid_alias)
                and not (rowid_source_name is None and base_has_star)
            )
            if should_append_rowid:
                base_exprs = list(base_query.expressions or [])
                base_exprs.append(
                    exp.Alias(
                        this=rowid_expr.copy(),
                        alias=exp.Identifier(this=rowid_alias, quoted=False),
                    )
                )
                base_query.set("expressions", base_exprs)

            policy_eval.set(
                "expressions",
                [
                    exp.Alias(
                        this=rowid_expr,
                        alias=exp.Identifier(this=rowid_alias, quoted=False),
                    )
                ],
            )
        else:
            policy_eval.set("distinct", exp.Distinct())
            policy_eval.set(
                "expressions",
                [
                    exp.Alias(
                        this=expr.copy(),
                        alias=exp.Identifier(this=key_name, quoted=False),
                    )
                    for key_name, expr in join_keys
                ],
            )

        # Inject dimension tables into policy_eval only.
        dim_tables = self._collect_dimension_tables(policies)
        self._inject_dimension_tables(policy_eval, dim_tables)

        apply_policy_constraints_to_scan(
            policy_eval,
            policies,
            source_tables,
            stream_file_path=self._stream_file_path,
        )

        if use_rowid_join:
            if base_has_star and rowid_source_name is None:
                select_list = "base_query.*"
            else:
                select_list = "base_query.* EXCLUDE (__dfc_rowid)"
        else:
            select_list = "base_query.*"
        if include_valid:
            select_list += ", policy_eval.valid AS valid"
        if include_invalid_string:
            select_list += ", policy_eval.invalid_string AS invalid_string"
        join_key_names = (
            ["__dfc_rowid"] if use_rowid_join else [key for key, _ in join_keys]
        )
        outer_sql = f"SELECT {select_list} {self._two_phase_join_clause(join_key_names)}"

        rewritten = sqlglot.parse_one(outer_sql, read="duckdb")
        if not isinstance(rewritten, exp.Select):
            raise ValueError("Two-phase scan rewrite must produce a SELECT statement")
        rewritten.set(
            "with_",
            exp.With(
                expressions=[
                    exp.CTE(
                        this=base_query,
                        alias=exp.TableAlias(
                            this=exp.Identifier(this="base_query", quoted=False)
                        ),
                    ),
                    exp.CTE(
                        this=policy_eval,
                        alias=exp.TableAlias(
                            this=exp.Identifier(this="policy_eval", quoted=False)
                        ),
                    ),
                ]
            ),
        )
        return rewritten

    # -------------------------------------------------------------------------
    # Execution helpers
    # -------------------------------------------------------------------------

    def _execute_transformed(self, query: str, use_two_phase: bool = False):
        """Execute a transformed query and return the cursor."""
        transformed_query = self.transform_query(query, use_two_phase=use_two_phase)
        return self.conn.execute(transformed_query)

    def execute(self, query: str, use_two_phase: bool = False) -> Any:
        """Execute a SQL query after transforming it."""
        return self._execute_transformed(query, use_two_phase=use_two_phase)

    def fetchall(self, query: str, use_two_phase: bool = False) -> list[tuple]:
        """Execute a query and fetch all results."""
        return self._execute_transformed(query, use_two_phase=use_two_phase).fetchall()

    def fetchone(self, query: str, use_two_phase: bool = False) -> Optional[tuple]:
        """Execute a query and fetch one result."""
        return self._execute_transformed(query, use_two_phase=use_two_phase).fetchone()

    # -------------------------------------------------------------------------
    # Database introspection helpers
    # -------------------------------------------------------------------------

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            result = self.conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_name = ?
                """,
                [table_name.lower()],
            ).fetchone()
            return result is not None
        except Exception:
            return False

    def _get_table_columns(self, table_name: str) -> set[str]:
        """Get all column names for a table."""
        if not self._table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        try:
            result = self.conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = ?
                """,
                [table_name.lower()],
            ).fetchall()
            return {row[0].lower() for row in result}
        except Exception as e:
            raise ValueError(f"Failed to get columns for table '{table_name}': {e}") from e

    def _create_aggregate_function(
        self, func_name: str, expressions: list[exp.Expression]
    ) -> exp.AggFunc:
        """Create an aggregate function expression using the proper sqlglot class."""
        func_name_upper = func_name.upper()
        agg_class_map = {
            "MAX": exp.Max,
            "MIN": exp.Min,
            "SUM": exp.Sum,
            "AVG": exp.Avg,
            "COUNT": exp.Count,
        }
        if hasattr(exp, "CountIf"):
            agg_class_map["COUNT_IF"] = exp.CountIf
        if hasattr(exp, "Stddev"):
            agg_class_map["STDDEV"] = exp.Stddev
        if hasattr(exp, "Variance"):
            agg_class_map["VARIANCE"] = exp.Variance
        if hasattr(exp, "AnyValue"):
            agg_class_map["ANY_VALUE"] = exp.AnyValue
        if hasattr(exp, "First"):
            agg_class_map["FIRST"] = exp.First
        if hasattr(exp, "Last"):
            agg_class_map["LAST"] = exp.Last
        if hasattr(exp, "StringAgg"):
            agg_class_map["STRING_AGG"] = exp.StringAgg
        if hasattr(exp, "ArrayAgg"):
            agg_class_map["ARRAY_AGG"] = exp.ArrayAgg

        if func_name_upper in agg_class_map:
            agg_class = agg_class_map[func_name_upper]
            if len(expressions) == 1:
                return agg_class(this=expressions[0])
            return agg_class(expressions=expressions)
        return exp.AggFunc(this=func_name, expressions=expressions)

    def _get_column_type(self, table_name: str, column_name: str) -> Optional[str]:
        """Get the data type of a column in a table."""
        try:
            result = self.conn.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = ? AND column_name = ?
                """,
                [table_name.lower(), column_name.lower()],
            ).fetchone()
            return result[0].upper() if result else None
        except Exception as e:
            raise ValueError(
                f"Failed to get column type for '{table_name}.{column_name}': {e}"
            ) from e

    def _validate_table_exists(self, table_name: str, table_type: str) -> None:
        """Validate that a table exists in the database."""
        if not self._table_exists(table_name):
            raise ValueError(
                f"{table_type} table '{table_name}' does not exist in the database"
            )

    # -------------------------------------------------------------------------
    # Policy registration and validation
    # -------------------------------------------------------------------------

    def _get_column_table_type(
        self, column: exp.Column, policy: DFCPolicy
    ) -> Optional[str]:
        """Determine which table type (source/sink/dimension) a column belongs to.

        Returns:
            "source", "sink", "dimension", or None.
        """
        table_name = get_table_name_from_column(column)
        if not table_name:
            return None

        if policy.sources and table_name in policy._sources_lower:
            return "source"
        if policy.sink and table_name in getattr(
            policy, "_sink_reference_names", {policy.sink.lower()}
        ):
            return "sink"
        if hasattr(policy, "_dimension_lower") and table_name in policy._dimension_lower:
            return "dimension"
        return None

    def _validate_column_in_table(
        self,
        column: exp.Column,
        table_name: str,
        table_columns: set[str],
        table_type: str,
    ) -> None:
        """Validate that a column exists in a specific table."""
        col_name = get_column_name(column).lower()
        if col_name not in table_columns:
            raise ValueError(
                f"Column '{table_name}.{col_name}' referenced in constraint "
                f"does not exist in {table_type} table '{table_name}'"
            )

    def register_policy(self, policy: Union[DFCPolicy, AggregateDFCPolicy]) -> None:
        """Register a DFC policy with the rewriter.

        Validates sources, sink, dimension, and hierarchy tables and their columns.
        """
        for source in policy.sources:
            self._validate_table_exists(source, "Source")
        if policy.sink:
            self._validate_table_exists(policy.sink, "Sink")

        if hasattr(policy, "dimension"):
            for dim in policy.dimension:
                self._validate_table_exists(dim, "Dimension")

        # Validate hierarchy table exists if repair_mode is coarsen
        if hasattr(policy, "hierarchy") and policy.hierarchy:
            self._validate_table_exists(policy.hierarchy, "Hierarchy")

        source_columns: Optional[dict[str, set[str]]] = None
        sink_columns: Optional[set[str]] = None
        dimension_columns: Optional[dict[str, set[str]]] = None

        if policy.sources:
            source_columns = {
                source.lower(): self._get_table_columns(source)
                for source in policy.sources
            }
        if policy.sink:
            sink_columns = self._get_table_columns(policy.sink)

        if hasattr(policy, "dimension") and policy.dimension:
            dimension_columns = {
                dim.lower(): self._get_table_columns(dim) for dim in policy.dimension
            }

        if (
            policy.on_fail == Resolution.INVALIDATE
            and policy.sink
            and not isinstance(policy, AggregateDFCPolicy)
        ):
            if sink_columns is None:
                raise ValueError(f"Sink table '{policy.sink}' has no columns")
            if "valid" not in sink_columns:
                raise ValueError(
                    f"Sink table '{policy.sink}' must have a boolean column named 'valid' "
                    f"for INVALIDATE resolution policies"
                )
            valid_column_type = self._get_column_type(policy.sink, "valid")
            if valid_column_type != "BOOLEAN":
                raise ValueError(
                    f"Column 'valid' in sink table '{policy.sink}' must be of type BOOLEAN, "
                    f"but found type '{valid_column_type}'"
                )

        if (
            policy.on_fail == Resolution.INVALIDATE_MESSAGE
            and policy.sink
            and not isinstance(policy, AggregateDFCPolicy)
        ):
            if sink_columns is None:
                raise ValueError(f"Sink table '{policy.sink}' has no columns")
            if "invalid_string" not in sink_columns:
                raise ValueError(
                    f"Sink table '{policy.sink}' must have a string column named "
                    f"'invalid_string' for INVALIDATE_MESSAGE resolution policies"
                )
            invalid_string_column_type = self._get_column_type(policy.sink, "invalid_string")
            if not any(
                token in invalid_string_column_type.upper()
                for token in ("CHAR", "VARCHAR", "STRING", "TEXT")
            ):
                raise ValueError(
                    f"Column 'invalid_string' in sink table '{policy.sink}' must be a "
                    f"string type, but found type '{invalid_string_column_type}'"
                )

        columns = list(policy._constraint_parsed.find_all(exp.Column))
        for column in columns:
            table_name = get_table_name_from_column(column)
            if not table_name:
                col_name = get_column_name(column).lower()

                if column.find_ancestor(exp.Filter) is not None:
                    continue

                parent = column.parent
                if (
                    isinstance(parent, exp.AggFunc)
                    and hasattr(parent, "this")
                    and parent.this == column
                    and policy.sink
                    and col_name in getattr(
                        policy, "_sink_reference_names", {policy.sink.lower()}
                    )
                ):
                    continue

                raise ValueError(
                    f"Column '{col_name}' in constraint is not qualified with a table name. "
                    "This should have been caught during policy creation."
                )

            table_type = self._get_column_table_type(column, policy)
            col_name = get_column_name(column).lower()

            if table_type == "source":
                if source_columns is None:
                    raise ValueError("Source tables have no columns")
                table_columns = source_columns.get(table_name)
                if table_columns is None:
                    raise ValueError(f"Source table '{table_name}' has no columns")
                self._validate_column_in_table(column, table_name, table_columns, "source")
            elif table_type == "sink":
                if sink_columns is None:
                    raise ValueError(f"Sink table '{policy.sink}' has no columns")
                self._validate_column_in_table(column, policy.sink, sink_columns, "sink")
            elif table_type == "dimension":
                if dimension_columns is None:
                    raise ValueError("Dimension tables have no columns")
                table_columns = dimension_columns.get(table_name)
                if table_columns is None:
                    raise ValueError(f"Dimension table '{table_name}' has no columns")
                self._validate_column_in_table(
                    column, table_name, table_columns, "dimension"
                )
            else:
                raise ValueError(
                    f"Column '{table_name}.{col_name}' referenced in constraint "
                    f"references table '{table_name}', which is not in sources "
                    f"({policy.sources}), sink ('{policy.sink}'), or dimension "
                    f"({getattr(policy, 'dimension', [])})"
                )

        if isinstance(policy, AggregateDFCPolicy):
            self._aggregate_policies.append(policy)
        else:
            self._policies.append(policy)
    
    def register_sc_family(
            self,
            name: str,
            predicate,
            merge,
            label,
        ) -> None:
        """Register a SafeCoarsen predicate family by name.

        Args:
            name:      Identifier matching policy.predicate_family
            predicate: Factory fn(dim_values: dict) -> predicate fn
            merge:     Factory fn(dim_values: dict) -> merge fn
            label:     Label fn(parent, children) -> str
        """
        self._sc_families[name] = {
            "predicate": predicate,
            "merge":     merge,
            "label":     label,
        }

    def get_dfc_policies(self) -> list[DFCPolicy]:
        """Get all registered DFC policies."""
        return self._policies.copy()

    def get_aggregate_policies(self) -> list[AggregateDFCPolicy]:
        """Get all registered AggregateDFCPolicy objects."""
        return self._aggregate_policies.copy()

    # -------------------------------------------------------------------------
    # Aggregate policy finalization
    # -------------------------------------------------------------------------

    def finalize_aggregate_policies(self, sink_table: str) -> dict[str, Optional[str]]:
        """Finalize aggregate policies by evaluating constraints after all data is processed."""
        violations = {}

        matching_policies = [
            p for p in self._aggregate_policies
            if p.sink and p.sink.lower() == sink_table.lower()
        ]

        if not matching_policies:
            return violations

        if not self._table_exists(sink_table):
            for policy in matching_policies:
                policy_id = get_policy_identifier(policy)
                violations[policy_id] = None
            return violations

        sink_columns = self._get_table_columns(sink_table)

        for policy in matching_policies:
            policy_id = get_policy_identifier(policy)
            violation_message = None

            try:
                temp_col_counter = 1
                source_temp_cols = []
                sink_temp_cols = []

                if policy.sources:
                    for source in policy.sources:
                        source_aggregates = _extract_source_aggregates_from_constraint(
                            policy._constraint_parsed, source
                        )
                        for _ in source_aggregates:
                            temp_col_name = f"_{policy_id}_tmp{temp_col_counter}"
                            if temp_col_name.lower() in sink_columns:
                                source_temp_cols.append(temp_col_name)
                            temp_col_counter += 1

                sink_expressions = _extract_sink_expressions_from_constraint(
                    policy._constraint_parsed, policy.sink
                )
                for _ in sink_expressions:
                    temp_col_name = f"_{policy_id}_tmp{temp_col_counter}"
                    if temp_col_name.lower() in sink_columns:
                        sink_temp_cols.append(temp_col_name)
                    temp_col_counter += 1

                if not source_temp_cols and not sink_temp_cols:
                    violations[policy_id] = None
                    continue

                constraint_expr = sqlglot.parse_one(policy.constraint, read="duckdb")
                replacement_map = {}
                temp_col_idx = 0

                if policy.sources:
                    for source in policy.sources:
                        source_aggregates = _extract_source_aggregates_from_constraint(
                            policy._constraint_parsed, source
                        )
                        for agg_expr in source_aggregates:
                            if temp_col_idx < len(source_temp_cols):
                                temp_col_name = source_temp_cols[temp_col_idx]
                                inner_agg_sql = agg_expr.sql()
                                outer_agg_name = _find_outer_aggregate_for_inner(
                                    policy._constraint_parsed, inner_agg_sql
                                )
                                if outer_agg_name:
                                    for outer_agg in policy._constraint_parsed.find_all(
                                        exp.AggFunc
                                    ):
                                        outer_agg_sql = outer_agg.sql()
                                        if (
                                            inner_agg_sql.upper() in outer_agg_sql.upper()
                                            and outer_agg_sql.upper() != inner_agg_sql.upper()
                                        ):
                                            temp_col_ref = exp.Column(
                                                this=exp.Identifier(
                                                    this=temp_col_name, quoted=False
                                                )
                                            )
                                            new_outer_agg = self._create_aggregate_function(
                                                outer_agg_name, [temp_col_ref]
                                            )
                                            replacement_map[outer_agg_sql] = new_outer_agg.sql()
                                            break
                                else:
                                    agg_name = (
                                        agg_expr.sql_name().upper()
                                        if hasattr(agg_expr, "sql_name")
                                        else "SUM"
                                    )
                                    temp_col_ref = exp.Column(
                                        this=exp.Identifier(
                                            this=temp_col_name, quoted=False
                                        )
                                    )
                                    outer_agg = self._create_aggregate_function(
                                        agg_name, [temp_col_ref]
                                    )
                                    replacement_map[inner_agg_sql] = outer_agg.sql()
                                temp_col_idx += 1

                temp_col_idx = 0
                sink_expressions = _extract_sink_expressions_from_constraint(
                    policy._constraint_parsed, policy.sink
                )
                for sink_expr in sink_expressions:
                    if temp_col_idx < len(sink_temp_cols):
                        temp_col_name = sink_temp_cols[temp_col_idx]
                        temp_col_ref = exp.Column(
                            this=exp.Identifier(this=temp_col_name, quoted=False)
                        )
                        sink_agg = self._create_aggregate_function("SUM", [temp_col_ref])
                        if isinstance(sink_expr, exp.Filter):
                            new_filter = exp.Filter(
                                this=sink_agg, expression=sink_expr.expression
                            )
                            replacement_map[sink_expr.sql()] = new_filter.sql()
                        else:
                            replacement_map[sink_expr.sql()] = sink_agg.sql()
                        temp_col_idx += 1

                constraint_expr = sqlglot.parse_one(policy.constraint, read="duckdb")
                expr_replacement_map = {}
                for old_expr_sql, new_expr_sql in replacement_map.items():
                    for node in constraint_expr.find_all(exp.AggFunc):
                        node_sql = node.sql()
                        if node_sql.upper() == old_expr_sql.upper():
                            new_expr = sqlglot.parse_one(new_expr_sql, read="duckdb")
                            expr_replacement_map[node] = new_expr
                            break
                    for node in constraint_expr.find_all(exp.Filter):
                        node_sql = node.sql()
                        if node_sql.upper() == old_expr_sql.upper():
                            new_expr = sqlglot.parse_one(new_expr_sql, read="duckdb")
                            expr_replacement_map[node] = new_expr
                            break

                def replace_node(node, expr_replacement_map=expr_replacement_map):
                    if node in expr_replacement_map:
                        return expr_replacement_map[node]
                    return node

                constraint_expr = constraint_expr.transform(replace_node, copy=False)
                constraint_sql = constraint_expr.sql()
                eval_query = (
                    f"SELECT ({constraint_sql}) AS constraint_result FROM {sink_table}"
                )

                result = self.conn.execute(eval_query).fetchone()
                if result and len(result) > 0:
                    constraint_passed = result[0]
                    if not constraint_passed:
                        violation_message = (
                            f"Aggregate policy constraint violated: {policy.constraint}"
                        )
                        if policy.description:
                            violation_message = (
                                f"{policy.description}: {violation_message}"
                            )
                else:
                    violation_message = None

            except Exception as e:
                violation_message = f"Error evaluating aggregate policy constraint: {e!s}"

            violations[policy_id] = violation_message

        return violations

    def delete_policy(
        self,
        sources: Optional[list[str]] = None,
        sink: Optional[str] = None,
        constraint: str = "",
        on_fail: Optional[Resolution] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Delete a DFC policy from the rewriter by matching all provided parameters."""
        if sources is None and sink is None and not constraint:
            raise ValueError(
                "At least one of sources, sink, or constraint must be provided"
            )

        normalized_sources = None
        if sources is not None:
            if not isinstance(sources, list):
                raise ValueError("Sources must be provided as a list of table names")
            normalized_sources = [source.strip() for source in sources]

        for i, policy in enumerate(self._policies):
            sources_match = (
                normalized_sources is None or policy.sources == normalized_sources
            )
            sink_match = sink is None or policy.sink == sink
            constraint_match = not constraint or policy.constraint == constraint
            on_fail_match = on_fail is None or policy.on_fail == on_fail
            description_match = description is None or policy.description == description

            if (
                sources_match
                and sink_match
                and constraint_match
                and on_fail_match
                and description_match
            ):
                del self._policies[i]
                return True

        for i, policy in enumerate(self._aggregate_policies):
            sources_match = (
                normalized_sources is None or policy.sources == normalized_sources
            )
            sink_match = sink is None or policy.sink == sink
            constraint_match = not constraint or policy.constraint == constraint
            on_fail_match = on_fail is None or policy.on_fail == on_fail
            description_match = description is None or policy.description == description

            if (
                sources_match
                and sink_match
                and constraint_match
                and on_fail_match
                and description_match
            ):
                del self._aggregate_policies[i]
                return True

        return False

    # -------------------------------------------------------------------------
    # Source/sink table extraction helpers
    # -------------------------------------------------------------------------

    def _get_source_tables(self, parsed: exp.Select) -> set[str]:
        """Extract source table names from a SELECT query.

        Always called BEFORE dimension injection so dimension tables never
        appear in from_tables and never trigger accidental policy matches.
        """
        from_tables = set()
        for table in parsed.find_all(exp.Table):
            if table.find_ancestor(exp.From) or table.find_ancestor(exp.Join):
                from_tables.add(table.name.lower())
        return from_tables

    def _get_sink_table(self, parsed: exp.Insert) -> Optional[str]:
        """Extract sink table name from an INSERT statement."""
        if not isinstance(parsed, exp.Insert):
            return None

        def _extract_table_name(table_expr) -> Optional[str]:
            if (
                isinstance(table_expr, exp.Schema)
                and hasattr(table_expr, "this")
                and isinstance(table_expr.this, exp.Table)
            ):
                return _extract_table_name(table_expr.this)
            if isinstance(table_expr, exp.Table):
                if hasattr(table_expr, "name") and table_expr.name:
                    return str(table_expr.name).lower()
                if hasattr(table_expr, "alias_or_name"):
                    return str(table_expr.alias_or_name).lower()
            return None

        if hasattr(parsed, "this") and parsed.this:
            result = _extract_table_name(parsed.this)
            if result:
                return result

        for table in parsed.find_all(exp.Table):
            if table.find_ancestor(exp.Select):
                continue
            if table.find_ancestor(exp.Join):
                continue
            result = _extract_table_name(table)
            if result:
                return result

        return None

    def _get_update_target_table(self, parsed: exp.Update) -> Optional[str]:
        if not isinstance(parsed, exp.Update):
            return None
        if isinstance(parsed.this, exp.Table) and parsed.this.name:
            return str(parsed.this.name).lower()
        return None

    def _get_update_target_reference_name(self, parsed: exp.Update) -> str:
        if not isinstance(parsed, exp.Update) or not isinstance(parsed.this, exp.Table):
            return ""
        return parsed.this.alias_or_name

    def _get_update_source_tables(self, parsed: exp.Update) -> set[str]:
        if not isinstance(parsed, exp.Update):
            return set()
        source_tables = set()
        target_node = parsed.this
        for table in parsed.find_all(exp.Table):
            if table is target_node:
                continue
            source_tables.add(table.name.lower())
        return source_tables

    def _get_update_assignment_mapping(
        self, parsed: exp.Update
    ) -> dict[str, exp.Expression]:
        mapping: dict[str, exp.Expression] = {}
        if not isinstance(parsed, exp.Update):
            return mapping
        for assignment in parsed.expressions or []:
            if not isinstance(assignment, exp.EQ):
                continue
            if not isinstance(assignment.this, exp.Column) or assignment.expression is None:
                continue
            mapping[get_column_name(assignment.this).lower()] = assignment.expression.copy()
        return mapping

    def _get_insert_source_tables(self, parsed: exp.Insert) -> set[str]:
        if not isinstance(parsed, exp.Insert):
            return set()
        select_expr = parsed.find(exp.Select)
        if select_expr:
            return self._get_source_tables(select_expr)
        return set()

    def _get_insert_column_mapping(
        self,
        insert_parsed: exp.Insert,
        select_parsed: exp.Select,
    ) -> dict[str, str]:
        """Get mapping from sink table column names to SELECT output column names/aliases."""
        mapping = {}
        insert_columns = []

        if (
            hasattr(insert_parsed, "this")
            and isinstance(insert_parsed.this, exp.Schema)
            and hasattr(insert_parsed.this, "expressions")
            and insert_parsed.this.expressions
        ):
            for col in insert_parsed.this.expressions:
                if isinstance(col, exp.Identifier):
                    insert_columns.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    insert_columns.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    insert_columns.append(col.lower())

        if not insert_columns and hasattr(insert_parsed, "columns") and insert_parsed.columns:
            for col in insert_parsed.columns:
                if isinstance(col, exp.Identifier):
                    insert_columns.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    insert_columns.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    insert_columns.append(col.lower())

        if (
            not insert_columns
            and hasattr(insert_parsed, "expressions")
            and insert_parsed.expressions
        ):
            for expr in insert_parsed.expressions:
                if isinstance(expr, exp.Identifier):
                    insert_columns.append(expr.name.lower())
                elif isinstance(expr, exp.Column):
                    insert_columns.append(get_column_name(expr).lower())

        select_outputs = []
        for expr in select_parsed.expressions:
            if isinstance(expr, exp.Alias):
                if isinstance(expr.alias, exp.Identifier):
                    alias_name = expr.alias.name.lower()
                elif isinstance(expr.alias, str):
                    alias_name = expr.alias.lower()
                else:
                    alias_name = str(expr.alias).lower()
                select_outputs.append(alias_name)
            elif isinstance(expr, exp.Column):
                select_outputs.append(get_column_name(expr).lower())
            elif isinstance(expr, exp.Star):
                return {}
            else:
                select_outputs.append(f"col{len(select_outputs) + 1}")

        if insert_columns:
            for i, sink_col in enumerate(insert_columns):
                if i < len(select_outputs):
                    mapping[sink_col] = select_outputs[i]
        else:
            for i, select_output in enumerate(select_outputs):
                mapping[f"col{i + 1}"] = select_output

        return mapping

    # -------------------------------------------------------------------------
    # INSERT column helpers
    # -------------------------------------------------------------------------

    def _add_valid_column_to_insert(self, insert_parsed: exp.Insert) -> None:
        if (
            hasattr(insert_parsed, "this")
            and isinstance(insert_parsed.this, exp.Schema)
            and hasattr(insert_parsed.this, "expressions")
            and insert_parsed.this.expressions
        ):
            column_names = []
            for col in insert_parsed.this.expressions:
                if isinstance(col, exp.Identifier):
                    column_names.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    column_names.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    column_names.append(col.lower())
            if "valid" not in column_names:
                insert_parsed.this.expressions.append(
                    exp.Identifier(this="valid", quoted=False)
                )
            return

        if hasattr(insert_parsed, "columns") and insert_parsed.columns:
            column_names = []
            for col in insert_parsed.columns:
                if isinstance(col, exp.Identifier):
                    column_names.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    column_names.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    column_names.append(col.lower())
            if "valid" not in column_names:
                insert_parsed.columns.append(exp.Identifier(this="valid", quoted=False))

    def _add_invalid_string_column_to_insert(self, insert_parsed: exp.Insert) -> None:
        if (
            hasattr(insert_parsed, "this")
            and isinstance(insert_parsed.this, exp.Schema)
            and hasattr(insert_parsed.this, "expressions")
            and insert_parsed.this.expressions
        ):
            column_names = []
            for col in insert_parsed.this.expressions:
                if isinstance(col, exp.Identifier):
                    column_names.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    column_names.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    column_names.append(col.lower())
            if "invalid_string" not in column_names:
                insert_parsed.this.expressions.append(
                    exp.Identifier(this="invalid_string", quoted=False)
                )
            return

        if hasattr(insert_parsed, "columns") and insert_parsed.columns:
            column_names = []
            for col in insert_parsed.columns:
                if isinstance(col, exp.Identifier):
                    column_names.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    column_names.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    column_names.append(col.lower())
            if "invalid_string" not in column_names:
                insert_parsed.columns.append(
                    exp.Identifier(this="invalid_string", quoted=False)
                )

    def _add_aggregate_temp_columns_to_insert(
        self,
        insert_parsed: exp.Insert,
        _policies: list[AggregateDFCPolicy],
        select_parsed: exp.Select,
    ) -> None:
        temp_column_names = []
        seen = set()
        for expr in select_parsed.expressions:
            if isinstance(expr, exp.Alias):
                alias_name = get_column_name(expr.alias).lower()
                if (
                    alias_name.startswith("_policy_")
                    and "_tmp" in alias_name
                    and alias_name not in seen
                ):
                    temp_column_names.append(alias_name)
                    seen.add(alias_name)

        if not temp_column_names:
            return

        if (
            hasattr(insert_parsed, "this")
            and isinstance(insert_parsed.this, exp.Schema)
            and hasattr(insert_parsed.this, "expressions")
            and insert_parsed.this.expressions
        ):
            existing_columns = []
            for col in insert_parsed.this.expressions:
                if isinstance(col, exp.Identifier):
                    existing_columns.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    existing_columns.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    existing_columns.append(col.lower())
            for temp_col in temp_column_names:
                if temp_col not in existing_columns:
                    insert_parsed.this.expressions.append(
                        exp.Identifier(this=temp_col, quoted=False)
                    )

        if hasattr(insert_parsed, "columns") and insert_parsed.columns:
            existing_columns = [
                col.lower() if isinstance(col, str) else get_column_name(col).lower()
                for col in insert_parsed.columns
            ]
            for temp_col in temp_column_names:
                if temp_col not in existing_columns:
                    insert_parsed.columns.append(temp_col)

    def _get_insert_column_list(self, insert_parsed: exp.Insert) -> list[str]:
        insert_columns = []

        if (
            hasattr(insert_parsed, "this")
            and isinstance(insert_parsed.this, exp.Schema)
            and hasattr(insert_parsed.this, "expressions")
            and insert_parsed.this.expressions
        ):
            for col in insert_parsed.this.expressions:
                if isinstance(col, exp.Identifier):
                    insert_columns.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    insert_columns.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    insert_columns.append(col.lower())

        if not insert_columns and hasattr(insert_parsed, "columns") and insert_parsed.columns:
            for col in insert_parsed.columns:
                if isinstance(col, exp.Identifier):
                    insert_columns.append(col.name.lower())
                elif isinstance(col, exp.Column):
                    insert_columns.append(get_column_name(col).lower())
                elif isinstance(col, str):
                    insert_columns.append(col.lower())

        if (
            not insert_columns
            and hasattr(insert_parsed, "expressions")
            and insert_parsed.expressions
        ):
            for expr in insert_parsed.expressions:
                if isinstance(expr, exp.Identifier):
                    insert_columns.append(expr.name.lower())
                elif isinstance(expr, exp.Column):
                    insert_columns.append(get_column_name(expr).lower())

        return insert_columns

    def _add_aliases_to_insert_select_outputs(
        self,
        insert_parsed: exp.Insert,
        select_parsed: exp.Select,
    ) -> None:
        insert_columns = self._get_insert_column_list(insert_parsed)
        if not insert_columns:
            return

        for i, expr in enumerate(select_parsed.expressions):
            if i >= len(insert_columns):
                break
            if isinstance(expr, exp.Alias):
                continue
            if isinstance(expr, exp.Star):
                continue
            sink_col_name = insert_columns[i]
            if isinstance(expr, exp.Column):
                col_name = get_column_name(expr).lower()
                if col_name == sink_col_name:
                    continue
            alias_expr = exp.Alias(
                this=expr,
                alias=exp.Identifier(this=sink_col_name, quoted=False),
            )
            select_parsed.expressions[i] = alias_expr

    # -------------------------------------------------------------------------
    # Aggregation detection and policy matching
    # -------------------------------------------------------------------------

    def _has_aggregations(self, parsed: exp.Select) -> bool:
        """Check if a SELECT query contains aggregations."""
        for expr in parsed.expressions:
            if isinstance(expr, exp.AggFunc):
                return True
            if isinstance(expr, exp.Alias) and isinstance(expr.this, exp.AggFunc):
                return True
            for node in expr.find_all(exp.AggFunc):
                subquery_ancestor = node.find_ancestor(exp.Subquery)
                if subquery_ancestor is None:
                    return True
        return False
    
    def _has_group_by(self, parsed: exp.Select) -> bool:
        """Check if a SELECT query has a GROUP BY clause."""
        group_clause = parsed.args.get("group")
        return bool(group_clause and group_clause.expressions)

    def _find_matching_policies(
        self,
        source_tables: set[str],
        sink_table: Optional[str] = None,
    ) -> list[DFCPolicy]:
        """Find policies that match the source and sink tables in the query.

        Dimension tables are never used for matching — they are metadata only.
        """
        matching = []
        for policy in self._policies:
            policy_sources = policy._sources_lower
            policy_sink = policy.sink.lower() if policy.sink else None

            if policy_sink and policy_sources:
                if (
                    sink_table is not None
                    and policy_sink == sink_table
                    and policy_sources.issubset(source_tables)
                ):
                    matching.append(policy)
            elif policy_sink:
                if sink_table is not None and policy_sink == sink_table:
                    matching.append(policy)
            elif policy_sources and policy_sources.issubset(source_tables):
                matching.append(policy)

        return matching

    def _find_matching_aggregate_policies(
        self,
        source_tables: set[str],
        sink_table: Optional[str] = None,
    ) -> list[AggregateDFCPolicy]:
        """Find aggregate policies that match the source and sink tables in the query."""
        matching = []
        for policy in self._aggregate_policies:
            policy_sources = policy._sources_lower
            policy_sink = policy.sink.lower() if policy.sink else None

            if policy_sink and policy_sources:
                if (
                    sink_table is not None
                    and policy_sink == sink_table
                    and policy_sources.issubset(source_tables)
                ):
                    matching.append(policy)
            elif policy_sink:
                if sink_table is not None and policy_sink == sink_table:
                    matching.append(policy)
            elif policy_sources and policy_sources.issubset(source_tables):
                matching.append(policy)

        return matching

    # -------------------------------------------------------------------------
    # UDF registration
    # -------------------------------------------------------------------------

    def _register_kill_udf(self) -> None:
        """Register the kill UDF."""
        def kill() -> bool:
            raise ValueError("KILLing due to dfc policy violation")
        self.conn.create_function("kill", kill, return_type="BOOLEAN")

    def _call_llm_to_fix_row(
        self,
        constraint: str,
        description: Optional[str],
        column_values: list[Any],
        column_names: Optional[list[str]] = None,
    ) -> Optional[list[Any]]:
        """Call LLM to try to fix a violating row based on the constraint."""
        if not self._bedrock_client:
            return None

        bedrock_client = self._bedrock_client

        def make_json_serializable(value):
            if isinstance(value, Decimal):
                return float(value)
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            return str(value)

        row_data = {}
        if column_names and len(column_names) == len(column_values):
            for name, value in zip(column_names, column_values):
                row_data[name] = make_json_serializable(value)
        else:
            for i, value in enumerate(column_values):
                row_data[f"col{i}"] = make_json_serializable(value)

        constraint_desc = description or "Policy constraint"
        prompt = f"""You are a data quality assistant. A row of data has violated a data flow control policy.

                    POLICY CONSTRAINT: {constraint}
                    POLICY DESCRIPTION: {constraint_desc}

                    VIOLATING ROW DATA:
                    {json.dumps(row_data, indent=2)}

                    Your task is to fix the violating row data so it satisfies the policy constraint. Return the fixed row data as a JSON object with the same keys as the input row data. Only modify values that need to be changed to satisfy the constraint. If you cannot fix the row, return null.

                    Return only the JSON object (or null), no additional text or explanation.
                """

        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }

            if self._replay_manager and self._replay_manager.is_enabled():
                response_body = self._replay_manager.get_llm_resolution_response(
                    constraint=constraint,
                    description=description,
                    row_data=row_data,
                    request_body=request_body,
                )
                if response_body is None:
                    response = bedrock_client.invoke_model(
                        modelId=self._bedrock_model_id,
                        body=json.dumps(request_body),
                    )
                    response_body = json.loads(response["body"].read())
            else:
                if self._recorder and self._recorder.is_enabled():
                    self._recorder.record_llm_resolution_request(
                        constraint=constraint,
                        description=description,
                        row_data=row_data,
                        request_body=request_body,
                    )
                response = bedrock_client.invoke_model(
                    modelId=self._bedrock_model_id,
                    body=json.dumps(request_body),
                )
                response_body = json.loads(response["body"].read())

            text_content = ""
            for content_block in response_body.get("content", []):
                if content_block.get("type") == "text":
                    text_content += content_block.get("text", "")

            if not text_content:
                return None

            text_content = text_content.strip()
            if text_content.lower() == "null":
                return None

            json_text = text_content
            if "```json" in text_content:
                start = text_content.find("```json") + 7
                end = text_content.find("```", start)
                if end != -1:
                    json_text = text_content[start:end].strip()
            elif "```" in text_content:
                start = text_content.find("```") + 3
                end = text_content.find("```", start)
                if end != -1:
                    json_text = text_content[start:end].strip()

            try:
                fixed_row_data = json.loads(json_text)
            except json.JSONDecodeError:
                fixed_row_data = None

            if self._recorder and self._recorder.is_enabled():
                self._recorder.record_llm_resolution_response(
                    constraint=constraint,
                    description=description,
                    response_body=response_body,
                    fixed_row_data=fixed_row_data,
                )

            if fixed_row_data is None:
                return None

            if column_names:
                fixed_values = [
                    fixed_row_data.get(name, val)
                    for name, val in zip(column_names, column_values)
                ]
            else:
                fixed_values = [
                    fixed_row_data.get(f"col{i}", val)
                    for i, val in enumerate(column_values)
                ]

            return fixed_values

        except (ClientError, BotoCoreError):
            return None
        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _register_address_violating_rows_udf(self) -> None:
        """Register the address_violating_rows UDF for LLM resolution policies."""
        def address_violating_rows(*args) -> bool:
            if not args or len(args) < 4:
                return False

            column_values = list(args[:-4]) if len(args) >= 4 else []
            constraint = args[-4] if len(args) >= 4 else ""
            description = args[-3] if len(args) >= 3 else ""
            column_names_json = args[-2] if len(args) >= 2 else ""
            stream_endpoint = args[-1] if len(args) >= 1 else ""

            if stream_endpoint:
                stream_endpoint = stream_endpoint.strip().strip("'").strip('"')

            column_names = None
            if column_names_json:
                try:
                    column_names_json_cleaned = (
                        column_names_json.strip().strip("'").strip('"')
                    )
                    column_names = json.loads(column_names_json_cleaned)
                except Exception:
                    column_names = None

            if constraint and self._bedrock_client:
                try:
                    fixed_values = self._call_llm_to_fix_row(
                        constraint,
                        description if description else None,
                        column_values,
                        column_names,
                    )
                    if fixed_values:
                        if stream_endpoint:
                            try:
                                row_data = "\t".join(
                                    str(val).lower() if isinstance(val, bool) else str(val)
                                    for val in fixed_values
                                )
                                with open(stream_endpoint, "a") as f:
                                    f.write(f"{row_data}\n")
                                    f.flush()
                                    os.fsync(f.fileno())
                            except Exception:
                                pass
                        return False
                except Exception:
                    pass

            return False

        self.conn.create_function(
            "address_violating_rows", address_violating_rows, return_type="BOOLEAN"
        )

    # -------------------------------------------------------------------------
    # Lifecycle helpers
    # -------------------------------------------------------------------------

    def get_stream_file_path(self) -> Optional[str]:
        """Get the path to the stream file for LLM-fixed rows."""
        return self._stream_file_path

    def reset_stream_file_path(self) -> None:
        """Reset the stream file path by creating a new temporary file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as stream_file:
            self._stream_file_path = stream_file.name

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()

    def __enter__(self) -> "SQLRewriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()