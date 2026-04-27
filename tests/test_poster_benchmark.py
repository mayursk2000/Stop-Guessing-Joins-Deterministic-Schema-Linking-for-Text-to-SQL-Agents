import unittest

from text2sql_agent_prototype.large_schema_benchmark import (
    abstract_table_rows,
    build_large_schema,
    comparison_rows,
    print_abstract_table,
    run_large_benchmark,
)
from text2sql_agent_prototype.prototype import (
    SQLRewriter,
    SchemaGraph,
    build_sample_schema,
)


class PosterBenchmarkTests(unittest.TestCase):
    def test_rewriter_repairs_invalid_join_using_schema_graph(self):
        schema = build_sample_schema()
        plan = SchemaGraph(schema).minimal_connecting_subgraph(["orders", "customers"])
        bad_sql = "SELECT * FROM orders JOIN customers ON orders.id = customers.id"

        rewritten = SQLRewriter().rewrite(bad_sql, plan)

        self.assertTrue(rewritten.changed)
        self.assertIn("orders.customer_id = customers.id", rewritten.sql)
        self.assertTrue(
            any("Graph join policy" in note for note in rewritten.notes),
            rewritten.notes,
        )

    def test_large_schema_fixture_has_100_plus_tables(self):
        schema = build_large_schema()

        self.assertGreaterEqual(len(schema.tables), 100)
        self.assertGreaterEqual(len(schema.foreign_keys), 100)
        self.assertIn("orders", schema.tables)
        self.assertIn("purchase_orders", schema.tables)
        self.assertIn("shipment_events", schema.tables)
        self.assertIn("opportunities", schema.tables)
        self.assertIn("quota_attainment", schema.tables)
        self.assertNotIn("customer_profile_01", schema.tables)

    def test_large_schema_benchmark_improves_join_and_execution_accuracy(self):
        results = run_large_benchmark()
        print()
        print_abstract_table(results)
        rows = {row["Method"]: row for row in abstract_table_rows(results)}

        self.assertIn("Ours", rows)
        self.assertIn("Agent-based", rows)
        self.assertGreater(rows["Ours"]["Join Acc."], rows["Agent-based"]["Join Acc."])
        self.assertGreaterEqual(rows["Ours"]["Exec. Acc."], rows["Agent-based"]["Exec. Acc."])

    def test_comparison_table_contains_spider_bird_and_abstract_references(self):
        rows = {row["Method"]: row for row in comparison_rows(run_large_benchmark())}

        self.assertEqual(rows["Spider original best"]["EM"], 12.4)
        self.assertEqual(rows["BIRD GPT-4 + EK"]["Exec. Acc."], 54.9)
        self.assertNotIn("Abstract prior", rows)


if __name__ == "__main__":
    unittest.main()
