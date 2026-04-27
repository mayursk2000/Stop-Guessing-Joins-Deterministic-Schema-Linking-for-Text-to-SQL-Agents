# Deterministic Schema-Linking Text-to-SQL Prototype

This workspace contains a runnable prototype for the abstract's architecture:
**Stop Guessing Joins: Deterministic Schema Linking for Text-to-SQL Agents**.

The prototype follows each block in the diagram:

```text
Query
  -> Retrieval: Lexical + Embeddings
  -> Candidate Tables T
  -> Graph Resolution G_T
  -> Minimal Connecting Subgraph
  -> LLM: SQL Gen
  -> Rewriter
  -> SQL Execution
  -> Validation
  -> Final SQL or Retry
```

## Quick Start

Open a terminal in this folder:

```powershell
cd C:\Users\MK\Desktop\NE_Agents
```

No package install is required. The prototype uses only Python standard-library
modules and an in-memory SQLite database.

Run one end-to-end query:

```powershell
python -m text2sql_agent_prototype.prototype "Show revenue by customer"
```

The output is a JSON trace with:

```text
query
retrieved candidate tables
resolved join graph
generated SQL
rewritten SQL
execution rows
validation result
final SQL
```

Useful single-query demos:

```powershell
python -m text2sql_agent_prototype.prototype "Show revenue by supplier country"
python -m text2sql_agent_prototype.prototype "Show open support tickets by customer segment"
python -m text2sql_agent_prototype.prototype "Show available stock by warehouse and product category"
```

## Poster Tests

Run poster-focused tests:

```powershell
python -m unittest discover -s tests
```

The cleaned test suite keeps only the checks that support the poster story:

- deterministic repair of an invalid join
- generation of a 100+ table schema with dense relationships
- side-by-side benchmark improvement for join and execution accuracy

Expected result:

```text
Ran 4 tests

OK
```

## Benchmarks

Run the mini schema-linking benchmark:

```powershell
python -m text2sql_agent_prototype.benchmark
```

Run the 100+ table large-schema benchmark:

```powershell
python -m text2sql_agent_prototype.large_schema_benchmark
```

This generates a synthetic Spider/BIRD-style schema with:

```text
122 tables
126 foreign keys
22 core tables
100 distractor tables
24 local evaluation questions
```

It prints a poster-style comparison table with benchmark references:

```text
Method                Join Acc.  Exec. Acc.  EM     VES   Cases     AvgJ  MaxJ  Tables   FKs
------                ---------  ----------  ----   ---   -----     ----  ----  ------   ---
Spider original best         -          -   12.4      -  10,181 Q     -     -  200 DBs     -
BIRD GPT-4 + EK              -       54.9      -      -  12,751 Q     -     -  95 DBs     -
BIRD human                   -       93.0      -      -         -     -     -  95 DBs     -
Ours                      68.8       50.0   29.2    0.6        24   2.0     3     122   126
Agent-based               39.3       50.0   12.5    0.4        24   2.0     3     122   126
```

The local rows compare `Ours` against `Agent-based`; reference rows include
Spider original best EM, BIRD GPT-4 execution accuracy, BIRD human execution
accuracy.

Metric meanings:

- `Join Acc.`: average join F1 over expected foreign-key chains.
- `Exec. Acc.`: execution result match against gold SQL, similar to Spider/BIRD EX.
- `EM`: exact SQL string match approximation.
- `VES`: BIRD-style valid efficiency score approximation.
- `Cases`: number of local evaluation questions or published benchmark size.
- `AvgJ`: average number of gold joins per local question.
- `MaxJ`: maximum number of gold joins in local questions.
- `Tables`: schema size.
- `FKs`: foreign-key relationship count.

Print only the generated schema size:

```powershell
python -m text2sql_agent_prototype.large_schema_benchmark --schema
```

Emit full large-schema benchmark JSON:

```powershell
python -m text2sql_agent_prototype.large_schema_benchmark --json
```

The JSON includes:

```text
schema
abstract_table
comparison_table
relationship_metrics
summary_by_method
summary_by_hardness
per-question results
```

Emit detailed benchmark JSON:

```powershell
python -m text2sql_agent_prototype.benchmark --json
```

## Evaluation Questions

The large benchmark currently evaluates 24 local questions across:

- revenue by customer, region, product category, supplier country, and sales rep
- payment amount by method
- late shipments by warehouse region
- returned units by supplier
- stock by warehouse and product category
- campaign revenue by marketing channel
- open billed amount by product category
- active subscription MRR by customer region
- open support tickets by customer segment

The same question family is repeated with large-schema wording to stress
retrieval against 100 distractor tables.

## Interpreting The Current Result

The deterministic agent improves structural correctness:

```text
Join Acc.: 68.8 vs 39.3
EM:        29.2 vs 12.5
VES:        0.6 vs 0.4
```

Execution accuracy is tied at `50.0` on the expanded 24-question large-schema
set. This is useful for the poster: deterministic graph resolution improves
join structure, but final execution still depends on retrieval finding the
right core tables in a noisy 122-table schema.

## Block-by-Block Implementation

### 1. Query

The input is a natural-language string. The pipeline keeps it in every trace so
later stages can validate semantic alignment against the original intent.

Code:

```text
TextToSQLAgent.run(query)
```

### 2. Retrieval: Lexical + Embeddings

Implemented by `HybridRetriever`.

The prototype uses:

- lexical matching over table names, column names, descriptions, and aliases
- a lightweight cosine-like score over token counters as a stand-in for
  embedding similarity
- a closed candidate set `T`

In production, replace this with:

- BM25 or database-native full-text search for lexical retrieval
- OpenAI/local embeddings over schema documentation
- reranking using query/table/column metadata

Output:

```json
{
  "candidate_tables": ["orders", "customers"],
  "matches": [
    {
      "table": "orders",
      "lexical_score": 0.5,
      "semantic_score": 0.2,
      "score": 0.38
    }
  ]
}
```

### 3. Candidate Tables `T`

This is the closed set passed to graph resolution. The SQL generator should not
freely invent tables outside this set unless the graph resolver adds bridge
tables required by foreign-key paths.

Code:

```text
RetrievalResult.candidate_tables
```

### 4. Graph Resolution `G_T`

Implemented by `SchemaGraph`.

The schema is modeled as:

- nodes: tables
- edges: foreign-key constraints
- edge metadata: left/right table and column

The graph can find shortest valid paths between candidate tables.

Code:

```text
SchemaGraph.shortest_path(start, goal)
```

### 5. Minimal Connecting Subgraph

Implemented by:

```text
SchemaGraph.minimal_connecting_subgraph(candidate_tables)
```

This computes the smallest available set of graph edges needed to connect the
candidate tables. Bridge tables are included when needed. For example,
connecting `customers` to `products` requires:

```text
customers <- orders <- order_items -> products
```

This is the proactive layer from the abstract: joins are valid before generation.

### 6. LLM: SQL Gen

Implemented by `SQLGenerator`.

This is a deterministic stand-in for an LLM. It only generates SQL using:

- allowed tables from the join plan
- allowed join predicates from the schema graph
- schema context from `JoinPlan.prompt_context(schema)`

In a real agent, this class becomes the LLM call boundary. The prompt should
include the join plan and say that joins must use only the provided predicates.

### 7. Rewriter

Implemented by `SQLRewriter`.

This is the reactive layer from the abstract. It treats generated SQL as an
intermediate program and repairs invalid join predicates using graph-approved
join conditions.

Example:

```sql
SELECT * FROM orders
JOIN customers ON orders.id = customers.id
```

is rewritten to:

```sql
SELECT * FROM orders
JOIN customers ON orders.customer_id = customers.id
```

In production, use a real SQL AST parser such as `sqlglot` instead of the
prototype regex matcher.

### 8. SQL Execution

Implemented by `SQLExecutor`.

The prototype uses an in-memory SQLite database with a richer sales schema:

- `regions`
- `customers`
- `departments`
- `employees`
- `orders`
- `order_items`
- `products`
- `suppliers`
- `payments`
- `warehouses`
- `shipments`
- `returns`
- `campaigns`
- `order_campaigns`
- `inventory`
- `invoices`
- `invoice_lines`
- `subscriptions`
- `support_tickets`
- `ticket_events`
- `customer_segments`
- `segment_members`

Execution returns structured rows or a structured error.

### 9. Validation

Implemented by `Validator`.

The validator checks:

- SQL executed successfully
- referenced tables exist
- join predicates match graph-approved constraints
- referenced schema semantically aligns with the query

Validation decides whether the final SQL is accepted or routed to retry.

### 10. Retry

Implemented by `RetryController`.

The prototype supports the diagram's retry shape:

- `repair`: invalid SQL can be fixed by the rewriter
- `new_graph`: low semantic alignment or missing schema elements can trigger
  alternate graph computation

The retry policy is intentionally simple so the control flow is easy to inspect.

## Files

```text
text2sql_agent_prototype/prototype.py                main pipeline
text2sql_agent_prototype/benchmark.py                mini benchmark
text2sql_agent_prototype/large_schema_benchmark.py   100+ table benchmark
tests/test_poster_benchmark.py                       poster-focused tests
README.md                                            runbook and architecture notes
```

## Next Production Steps

1. Replace sample schema with live database introspection.
2. Replace token-vector retrieval with real embeddings.
3. Replace regex SQL rewrite/validation with SQL AST parsing.
4. Add trace persistence for every pipeline stage.
5. Add adapters for BIRD/Spider-style schema and question files.

## Mini Benchmark

The included mini benchmark is intentionally small but shaped like a real
schema-linking benchmark. Each case defines:

- database id
- Spider/BIRD-style hardness label
- natural-language query
- gold SQL
- expected tables
- expected foreign-key joins

The runner reports:

- Spider-style exact match, labeled `EM`
- Spider/BIRD-style execution result match, labeled `EX`
- BIRD-style valid efficiency score approximation, labeled `VES`
- retrieval recall
- join recall
- join precision
- join F1
- execution accuracy
- result-match accuracy against gold SQL
- validation accuracy

Current cases include:

- revenue by customer
- revenue by region
- revenue by product category
- revenue by supplier country
- late shipments by warehouse region
- revenue by sales rep
- paid amount by payment method
- returned units by supplier
- available stock by warehouse and product category
- campaign revenue by marketing channel
- open billed amount by product category
- active subscription MRR by customer region
- open support tickets by customer segment

The complex sample schema includes multiple benchmark-style traps:

- bridge tables: `order_campaigns`, `order_items`
- shared hubs: `orders`, `products`, `regions`
- tempting but wrong nearby tables: `inventory` for product queries,
  `warehouses` for region queries, `payments` and `invoices` for amount queries
- multi-hop paths such as
  `returns -> order_items -> products -> suppliers`
- CRM/support paths such as
  `support_tickets -> customers -> segment_members -> customer_segments`
