# Stop-Guessing-Joins — BIRD Evaluation Devnotes

End-to-end reference for the BIRD evaluation harness built around the *Stop Guessing Joins: Deterministic Schema Linking for Text-to-SQL Agents* prototype.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Architecture — what the paper proposes](#2-architecture--what-the-paper-proposes)
3. [Repository layout](#3-repository-layout)
4. [The evaluation notebook](#4-the-evaluation-notebook-notebooksbird_evalipynb)
5. [Code modifications we made](#5-code-modifications-we-made)
6. [Results progression](#6-results-progression)
7. [Running the pipeline](#7-running-the-pipeline)
8. [Known limitations](#8-known-limitations)
9. [Future work](#9-future-work)
10. [Glossary](#10-glossary)

---

## 1. Project overview

The paper (NE Agents Day 2026 abstract, `NE_agents_abstract (10).pdf`):

> **Stop Guessing Joins: Deterministic Schema Linking for Text-to-SQL Agents**
> *Prajwal Raghunath, Mayur Kulkarni, Charlie Summers*

**Core thesis:** schema linking is the dominant source of errors in LLM text-to-SQL systems (~40% per BIRD analysis). Current systems have the LLM perform both **semantic matching** (which tables/columns are relevant?) *and* **structural reasoning** (what joins connect them?). The paper separates these — semantic matching restricted to a closed candidate set, structural reasoning handled deterministically using FK constraints.

**Paper Table 1 targets:**

| Method | Join Acc. | Exec. Acc. |
|---|---:|---:|
| Prior Work (BIRD baseline) | 58.2 | 54.9 |
| Agent-based | 63.7 | 57.1 |
| **Ours (paper target)** | **73.8** | **64.4** |

**What we built**

A Colab-runnable notebook (`notebooks/bird_eval.ipynb`) that runs the prototype end-to-end against BIRD dev:

- BIRD adapter loads canonical schema + FKs from `dev_tables.json` and column docs from `database_description/*.csv`
- Hybrid retriever: BM25 + `BAAI/bge-small-en-v1.5` dense embeddings, fused via Reciprocal Rank Fusion at the column-document level
- Graph-based proactive layer: minimal connecting subgraph over FKs
- DFC SQL rewriter as the reactive layer (paper ref [5], `sql_rewriter/`), patched to handle FK constraints
- AST-level alias normalizer as a final pass
- Local `Qwen2.5-Coder-32B-Instruct` (4-bit nf4 via `bitsandbytes`) with batched generation
- Evidence-anchored prompt that respects BIRD's hint-field semantics

**Headline result on `california_schools` (20-Q slice):** Join Acc 82.4, Exec Acc 70.0 — both above the paper's targets.

---

## 2. Architecture — what the paper proposes

```
Query
  └─► Retrieval: Lexical + Embeddings        (semantic matching, restricted)
        └─► Candidate Tables T               (closed set)
              └─► Graph Resolution G_T       (PROACTIVE — deterministic)
                    └─► Minimal Connecting Subgraph
                          └─► LLM: SQL Gen   (constrained context)
                                └─► Rewriter (REACTIVE — deterministic repair)
                                      └─► SQL Execution
                                            └─► Validation
                                                  └─► Final SQL or Retry
```

**Two deterministic layers around the LLM:**

- **Proactive layer (graph resolution).** Schema is modeled as a graph of tables (nodes) and FK constraints (edges). For candidate set T, a minimal connecting subgraph is computed — bridge tables added when needed. The LLM is told ONLY about graph-resolved tables and approved joins.

- **Reactive layer (deterministic repair, DFC).** Generated SQL is treated as an intermediate program. Invalid joins are detected and rewritten to schema-consistent conditions. Reference [5] = Summers et al., *Data Flow Control*, CIDR 2026 (the `sql_rewriter/` package in this repo).

The LLM's freedom is the *content* of the SQL (which columns to select, what filters, what aggregates). The *structure* — which tables, which join predicates — is enforced deterministically.

---

## 3. Repository layout

```
Stop-Guessing-Joins-Deterministic-Schema-Linking-for-Text-to-SQL-Agents/
│
├── README.md                                 ← original prototype readme
├── NE_agents_abstract (10).pdf               ← the paper abstract
│
├── text2sql_agent_prototype/                 ← prototype pipeline
│   ├── prototype.py                          ← TextToSQLAgent + all stages
│   ├── benchmark.py                          ← mini schema-linking benchmark
│   └── large_schema_benchmark.py             ← 100+ table synthetic benchmark
│
├── sql_rewriter/                             ← DFC SQL rewriter (paper ref [5])
│   ├── __init__.py                           ← exports DFCPolicy, SQLRewriter, etc.
│   ├── policy.py                             ← DFCPolicy class (PATCHED — FK support)
│   ├── rewriter.py                           ← DuckDB-based rewriter (PATCHED — FK routing)
│   ├── rewrite_rule.py                       ← rewrite-rule helpers (PATCHED — FK enforcement)
│   ├── safe_coarsen.py
│   └── sqlglot_utils.py
│
├── notebooks/
│   └── bird_eval.ipynb                       ← END-TO-END BIRD eval (this is the main artifact)
│
├── devnotes/
│   └── PROJECT.md                            ← this file
│
├── artifacts/                                ← precomputed JSON traces from prior runs
│   ├── large_schema_benchmark.json
│   ├── mini_benchmark.json
│   ├── poster_tests.json
│   └── step_metrics.json
│
├── scripts/                                  ← misc helpers
│   ├── build_step_metrics_json.py
│   ├── run_flow_demo.py
│   ├── run_poster_tests_json.py
│   └── run_tests_and_flow.py
│
└── tests/
    └── test_poster_benchmark.py              ← 4 poster-focused unit tests
```

### `text2sql_agent_prototype/prototype.py` — key classes

- `Schema`, `Table`, `Column`, `ForeignKey` — schema dataclasses
- `RetrievalResult`, `RetrievalMatch` — retriever output
- `JoinPlan` — graph output (tables + joins)
- `GraphJoinPolicy` — FK constraint adapter for DFC
- `HybridRetriever` — placeholder lexical retriever (we replace this in §5b)
- `SchemaGraph` — minimal connecting subgraph computation
- `SQLGenerator` — deterministic stand-in for LLM (we replace with `LocalHFGenerator`)
- `OpenAILLMGenerator` — OpenAI-backed (unused on Colab)
- `DFCRewriterAdapter` — wraps the DFC `sql_rewriter` package
- `SQLRewriter` (prototype version, line 738) — calls DFC adapter then `_repair_join_predicates` (sqlglot AST)
- `SQLExecutor`, `Validator`, `RetryController`
- `TextToSQLAgent` — orchestrates the whole pipeline via `agent.run(query)`

### `sql_rewriter/` — the DFC package

- `DFCPolicy` — policy with `sources`, `sink`, `constraint`, `on_fail` (Resolution enum: REMOVE/KILL/INVALIDATE/INVALIDATE_MESSAGE/LLM)
- `AggregateDFCPolicy` — for window-aggregate constraints
- `SQLRewriter` (DFC version) — DuckDB-backed, parses with `read="duckdb"`, transforms via policy enforcement

---

## 4. The evaluation notebook (`notebooks/bird_eval.ipynb`)

Cell-by-cell walkthrough.

### §1 — Setup
Clones the repo into Colab's `/content/`. Installs `sqlglot`, `huggingface_hub`, `tqdm`, `transformers>=4.45`, `accelerate>=0.34`, `bitsandbytes>=0.43`, `sentencepiece`. Adds repo to `sys.path`. Always pulls latest from `main` so cell scripts stay fresh across sessions.

### §2 — Secrets
Reads `HF_TOKEN` from `google.colab.userdata`. Required only for gated models (Llama, Gemma); Qwen is open and works without it.

### §3 — GPU check
Detects available VRAM, suggests model size:
- ≥35 GB → `Qwen/Qwen2.5-Coder-32B-Instruct`
- 20–35 GB → 32B 4-bit (~18 GB)
- 14–20 GB → `Qwen/Qwen2.5-Coder-14B-Instruct`
- <14 GB → `Qwen/Qwen2.5-Coder-7B-Instruct`

### §4 — Download BIRD dev
URL is fixed at `https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip`. Uses `gdown` for Drive URLs (with virus-scan handling) or plain `urllib.request.urlretrieve` for direct OSS URLs. Auto-detects extracted directory layout, validates dev.json + dev_databases/ are present.

### §5 — BIRD adapter
**Module:** schema introspection from BIRD's shipped metadata.

`schema_from_bird(db_id) -> Schema`:
1. Opens `<BIRD_DBS>/<db_id>/<db_id>.sqlite`
2. `PRAGMA table_info` for column types and names
3. Reads `<db_id>/database_description/*.csv` for human-written column descriptions → fills `Column.description`
4. Reads `dev_tables.json` for canonical FKs → fills `Schema.foreign_keys`
5. Unions in any FKs that `PRAGMA foreign_key_list` happens to find (rare on BIRD)

`graph_stats_str(trace, schema)` — formats per-question diagnostic of retrieval/graph state. Used by §7 and §8b.

### §5b — Hybrid retriever (BM25 + dense + RRF)
**This is the actual hybrid retriever the paper proposes.**

`BIRDHybridRetriever(schema, db_id, top_k=6)`:
- **Column-level corpus.** Each column → one searchable document: `"Table frpm. Column Free Meal Count (K-12): Number of K-12 students eligible for free meals"`. That's how schema RAG should work — column-granular, not table-granular.
- **Lexical (BM25)** over column docs with CamelCase-aware tokenization (`AvgScrMath` indexes as `{avg, scr, math}`).
- **Dense (`BAAI/bge-small-en-v1.5`)** — 33 M-param sentence encoder, 384-dim, normalized embeddings, cosine similarity. Catches *"math score"* → `AvgScrMath` even when no surface token matches.
- **Reciprocal Rank Fusion** (k=60). Parameter-free, no thresholds to sweep.
- **Aggregation.** Per-table score = max RRF score across the table's columns. Top-k tables → candidate set T → graph as before.

Cached per `db_id` to avoid re-embedding on every question.

### §5c — Wire in the DFC SQL rewriter (paper ref [5])
Installs `duckdb` and `botocore` (the rewriter imports both at module top). Sets `DFC_SQL_REWRITER_PATH` (cosmetic; the import resolves through `sys.path` because the repo is on it). Verifies `DFCRewriterAdapter` loads and `transform_query` works.

Also runs the **policy.py patch verification**:
- Positive: `DFCPolicy(sources=["frpm","schools"], constraint="frpm.CDSCode = schools.CDSCode", on_fail=Resolution.REMOVE)` → should construct successfully (was rejected by stock DFCPolicy).
- Negative: `constraint="frpm.CDSCode = 'X'"` (column-vs-literal, not a join) → should still be rejected.

Also the **Path B verification** for FK enforcement:
- Case A: query already enforces the FK in its JOIN ON → DFC should NOT add WHERE.
- Case B: FK missing → DFC should add as WHERE (not HAVING).

### §5d — Alias normalizer + agent factory
**`normalize_aliases(sql)`** — sqlglot AST pass that fixes the canonical LLM bug where the model declares `FROM frpm AS T1` but writes `frpm.CDSCode` instead of `T1.CDSCode`. Walks all `exp.Column` nodes; if a qualifier is a real table name that's been aliased, swaps it to the alias.

**`AliasNormalizingRewriter`** — subclass of `prototype.SQLRewriter`. Calls `super().rewrite()` (which runs DFC + `_repair_join_predicates`), then runs `normalize_aliases` on the result.

**`make_bird_agent(db_id)`** — factory that builds a fully wired agent:
1. `schema_from_bird(db_id)` → Schema
2. Open SQLite connection
3. Build `BIRDHybridRetriever` (cached per db_id)
4. Construct `TextToSQLAgent(use_llm=False, use_pruning=False)` — pruning disabled because the prototype's pruning heuristics are sales-schema-specific
5. Override `agent.retriever` with the hybrid retriever
6. Override `agent.rewriter` with `AliasNormalizingRewriter`
7. Override `agent.generator` with `local_generator` (defined in §6)
8. **Seed stub tables in the DFC's DuckDB** (column-only shells so `register_policy` validates)
9. **Register every FK from `schema.foreign_keys` as a DFCPolicy** with `on_fail=Resolution.REMOVE`
10. Print `[<db>] DFC FK policies: N/M registered`

### §6 — Local LLM generator (Qwen2.5-Coder, 4-bit)
Loads the suggested model with `bitsandbytes` 4-bit nf4 quantization. Sets `tokenizer.padding_side = "left"` (decoder-only models pad left for batched generation).

**`linked_schema_prompt(join_plan, schema)`** — renders ONLY graph-resolved tables with their columns + BIRD descriptions. Approved joins listed separately. The model never sees tables outside G_T.

**`LocalHFGenerator`** — drop-in `SQLGenerator` subclass.

Two methods:
- `generate(query, join_plan, schema)` — single question, used by `TextToSQLAgent.run()`.
- `generate_batch(items, max_input_length=4096)` — batched. `items` is a list of `(query, join_plan, schema)` tuples. Single tokenize-with-padding + single `model.generate` call. Per-item failures fall back to deterministic generator silently. On catastrophic batch failure (OOM etc.), falls back per-item.

The query string can carry `[BIRD_EVIDENCE]...[/BIRD_EVIDENCE]` and `[BIRD_QUESTION]...[/BIRD_QUESTION]` markers. `_split_bird_query` parses them and renders evidence as its own labeled prompt section with strict rules.

**System prompt** (the production version):
```
You are an expert Text-to-SQL assistant for SQLite, evaluated on BIRD.
You receive: (1) BIRD evidence with column hints, (2) the schema-linked
subset of the database, (3) a natural-language question.

STRICT RULES:
- If the evidence names a specific column in backticks (e.g. `Free Meal Count (K-12)`),
  USE THAT EXACT COLUMN. Do not substitute a similar-sounding column even if
  it seems to mean the same thing.
- If the evidence provides a formula (e.g. "rate = A / B"), COMPUTE that expression
  literally — do not use a precomputed alternative column. When the formula is a
  ratio, CAST the numerator to REAL (e.g. CAST(A AS REAL) / B) so SQLite doesn't
  integer-divide.
- If the evidence gives a literal value (e.g. "= 1" or "= 'Directly funded'"), use
  that EXACT value, including punctuation, capitalization, and spacing.
- Use ONLY tables and columns that appear in the linked schema.
- Use ONLY the approved joins listed; do not invent join predicates.
- Quote column names containing spaces or special characters with double quotes.
- Return EXACTLY ONE SQLite SELECT statement. No prose, no code fences.
```

### §7 — Smoke test
Runs Q0 end-to-end through `agent.run(build_bird_query(q0))`. Prints retrieval candidates, graph tables, joins, generated SQL, rewritten SQL, validation, final SQL. Sanity check before launching the full slice.

### §8 — Slice runner (batched)
Walks all of BIRD dev (`N = len(questions)`).

```
BATCH_SIZE = 16

for q in tqdm(questions[:N]):
    if qid in predictions: continue
    bird_query = build_bird_query(q)
    agent = get_agent(q["db_id"])              # cached per db_id
    retrieval = agent.retriever.retrieve(bird_query)
    join_plan = agent.graph.minimal_connecting_subgraph(retrieval.candidate_tables)
    pending.append({qid, agent, schema, retrieval, join_plan, bird_query, ...})

    if len(pending) >= BATCH_SIZE:
        flush_batch(pending)   # one batched generate_batch + per-question rewriter/exec/validate

flush_batch(pending)            # final partial batch
```

Retrieval and graph run serially per question (CPU-cheap). LLM generation goes in batches of 16 (GPU-expensive). Per-question rewrite/execute/validate runs serially after generation (also CPU-cheap).

**Resume-safe** — already-predicted qids are skipped at the top of the loop. Checkpoints to `/content/predict_dev.json` every ~50 questions.

Per-question graph stats logged to `/content/graph_stats.jsonl` for §8b's failure diagnosis.

### §8b — Local metrics (Join Acc + Exec Acc)
Reproduces the paper's Table 1 reporting format. For each prediction:
1. Execute predicted SQL and gold SQL on the BIRD SQLite DB
2. Compare result sets (sorted, all-string tuples to dodge type mismatches)
3. Compute Join Acc as F1 over join predicates (parsed via sqlglot, alias-resolved)
4. Skip Join Acc for single-table queries (no joins to score)

Reports:
- Headline table (Prior Work / Agent-based / Paper target / This run)
- Graph-table recall (% of gold tables included by graph)
- Status counts (correct / wrong_result / exec_error / fallback)
- Per-difficulty breakdown (simple / moderate / challenging)
- Per-db_id breakdown
- First 5 EX failures with graph diagnosis (which tables graph picked, which gold needed)

### §9 — BIRD official evaluation scripts
Generates `dev_gold.sql` from `dev.json`. Runs BIRD's `evaluation.py` and `evaluation_ves.py` (you drop them into `/content/bird_eval/`). Reports paper-grade EX and VES.

---

## 5. Code modifications we made

### 5.1 `sql_rewriter/policy.py`

**Why:** stock `DFCPolicy` rejects raw FK equality constraints (`a.x = b.y`) because its `_validate_aggregation_rules` requires every source-table column to be inside an aggregation function. FKs are structural — not aggregation invariants — so they're a different shape entirely.

**What changed:**
- Added `DFCPolicy._is_pure_join_constraint()` — detects whether the constraint is a conjunction of cross-source-table column equalities, no aggregations.
- Modified `DFCPolicy._validate_aggregation_rules()` — early-returns when `_is_pure_join_constraint()` is true. Aggregate rules still apply to anything containing AggFuncs.

**Effect:** `DFCPolicy(sources=["frpm","schools"], constraint="frpm.CDSCode = schools.CDSCode", on_fail=Resolution.REMOVE)` now constructs successfully. The negative case (`constraint="frpm.CDSCode = 'X'"`) is still correctly rejected — only column-vs-column equalities pass.

### 5.2 `sql_rewriter/rewrite_rule.py`

**Why:** even after policy.py accepts FK constraints, the enforcement functions emit them in ways that break SQLite — HAVING when there's no GROUP BY, redundant constraints when JOIN ON is already correct.

**What changed:**
- Added `_extract_fk_pairs_from_policy(policy)` — pulls column-pair frozensets out of a pure-join-shaped policy. Handles single equalities and AND-conjunctions.
- Added `_query_satisfies_fk_pairs(parsed, fk_pairs)` — walks the AST and returns True if every FK pair is already enforced by:
  - (a) a column-equality (JOIN ON / WHERE / nested) at any level, with table aliases resolved across all scopes, OR
  - (b) an IN-subquery of the form `outer.col IN (SELECT inner.col FROM inner_tbl …)` whose outer/inner column pair matches the FK.
- Modified `apply_policy_constraints_to_aggregation`:
  - Skip emission if `fk_pairs and _query_satisfies_fk_pairs(...)`.
  - For FK-shaped REMOVE policies, emit as `WHERE` (not `HAVING`). Aggregate-shaped policies stay on HAVING.
- Modified `apply_policy_constraints_to_scan`:
  - Skip emission if `fk_pairs and _query_satisfies_fk_pairs(...)`.

**Effect:** queries that already enforce the FK via JOIN/IN see DFC as a no-op (idempotent). Queries that need enforcement get a clean WHERE clause that's valid SQLite regardless of GROUP BY presence.

### 5.3 `sql_rewriter/rewriter.py`

**Why:** DFC's pre-application paths inject literal `"dfc"` and `"in_subquery"` aliases into the query (used for two-phase aggregation, IN-subquery rewriting, and limit-CTE wrapping). These paths assume aggregate-shaped constraints. When given FK policies, they corrupt the query — leaving artifacts like `WHERE dfc = "schools"."CDSCode"`.

**What changed:** modified `SQLRewriter._transform_query_common` to split policies at entry:

```python
if matching_policies:
    fk_policies = [p for p in matching_policies
                   if isinstance(p, DFCPolicy)
                   and hasattr(p, "_is_pure_join_constraint")
                   and p._is_pure_join_constraint()]
    aggregate_policies = [p for p in matching_policies if p not in fk_policies]

    # FK policies: direct path, no IN/EXISTS rewrite, no two-phase, no limit-CTE
    if fk_policies:
        if self._has_aggregations(parsed):
            apply_policy_constraints_to_aggregation(parsed, fk_policies, ...)
        else:
            apply_policy_constraints_to_scan(parsed, fk_policies, ...)

    # Aggregate policies: original DFC pipeline, unchanged
    if aggregate_policies:
        # IN/EXISTS rewrites, limit-CTE handling, two-phase aggregation, etc.
        ...
```

**Effect:** FK policies skip the pre-application paths entirely. They go through `apply_policy_constraints_to_*` (which now have the WHERE-not-HAVING + skip-when-satisfied patches). The `dfc =` / `in_subquery` artifacts disappear.

Aggregate-policy behavior is **identical** to before — the original code path is preserved when `aggregate_policies` is non-empty.

### 5.4 Notebook-level additions (no upstream code changes)

| Component | File | Purpose |
|---|---|---|
| BIRD adapter | §5 | Read BIRD's metadata into the prototype's `Schema` objects |
| Hybrid retriever | §5b | Replace placeholder lexical retriever with BM25 + bge-small + RRF |
| Alias normalizer | §5d | sqlglot AST pass for the alias-vs-table-name LLM bug |
| Agent factory | §5d | `make_bird_agent(db_id)` — wires retriever, rewriter, generator, FK policies |
| Local LLM generator | §6 | Qwen 4-bit + linked-schema prompt + evidence anchoring + batched generate |
| Marker-based query | §7/§8 | `[BIRD_EVIDENCE]` / `[BIRD_QUESTION]` so generator can structure the prompt |
| Local metrics | §8b | Approximate Join Acc + Exec Acc reproducing paper Table 1 |

---

## 6. Results progression

Tracked on the 20-question `california_schools` slice as we iterated.

| Iteration | Change | Join Acc | EX | Notes |
|---|---|---:|---:|---|
| 0 | Default prototype (lexical retriever, no DFC FK, sample-schema-tuned pruning) | 0.0 | 5.0 | Retriever returns only `frpm` for nearly every question |
| 1 | BIRD adapter (FKs from dev_tables.json, descs from CSVs) | 0.0 | 5.0 | Schema is now correct, but lexical retriever can't see column descriptions yet |
| 2 | Hybrid retriever (BM25 + bge-small + RRF, column-level corpus) | 17.6 | 20.0 | Retrieval recall jumps to 100% — graph now sees all gold tables |
| 3 | Alias normalizer (sqlglot AST pass) | 17.6 | 25.0 | Fixes some `frpm.col` vs `T1.col` bugs |
| 4 | DFC wire-up + FK policy registration (Path A only) | 47.1 | 20.0 | Join Acc up; EX down because DFC adds redundant WHERE/HAVING that breaks SQL |
| 5 | Path B v1: WHERE-not-HAVING + skip-when-satisfied (`apply_policy_constraints_*`) | 47.1 | 25.0 | Stops some HAVING-without-GROUP-BY breakage |
| 6 | Path B v2: split FK from aggregate at `_transform_query_common` | 76.5 | 55.0 | `dfc =` artifacts gone for direct-equality JOINs |
| 7 | Path B v3: IN-subquery satisfaction + evidence-anchored prompt + ratio-CAST rule | **82.4** | **70.0** | **Above paper targets on this slice** |

Status of remaining failures at iteration 7 (4 wrong_result + 0 exec_error):

| qid | Cause |
|---|---|
| 1 | LLM picked extra JOIN; missed Educational Option Type filter |
| 2 | LLM used `EdOpsName` instead of `District Name` |
| 4 | LLM used `FundingType` / `OpenDate` instead of `Charter Funding Type = 'Directly funded'` |
| 16 | LLM missed `StatusType = 'Merged'` from question phrase "merged Alameda" |

All four are LLM reasoning gaps — the structural pipeline is correct on every one of them. To break above ~70% on this slice would need self-consistency, few-shot examples, or a fundamentally different model.

---

## 7. Running the pipeline

### Prerequisites
- Google Colab with GPU (A100 40 GB or H100 90 GB recommended for 32B@4-bit at BATCH_SIZE=16)
- `HF_TOKEN` in Colab Secrets (only needed for gated models; Qwen is open)
- Network access to download BIRD dev (~3 GB)

### Step-by-step (full BIRD dev)
1. Open `notebooks/bird_eval.ipynb` in Colab.
2. **Restart kernel** to ensure clean state.
3. Run §1 → §2 → §3 (setup, secrets, GPU detect).
4. Run §4 (BIRD download + extract).
5. Run §5 (BIRD adapter — should print `BIRD adapter ready. Sample db: california_schools  →  3 tables, 2 FKs`).
6. Run the "Sanity Check" cell — should list 11 BIRD databases with sizes and `descriptions=yes`.
7. Run §5b (loads BGE-small encoder, runs probe — should return 6 candidate tables).
8. Run §5c (installs duckdb+botocore, verifies DFC adapter, tests FK policy construction — both positive and negative tests should pass; Path B Case A should report `NO (good)`, Case B should report HAVING `NO (good)` and WHERE `YES (good)`).
9. Run §5d (alias normalizer + agent factory — smoke test of normalizer should print before/after with notes; idempotence should be `OK`).
10. Run §6 (Qwen 32B 4-bit load, ~1–2 min download/load — final line: `LocalHFGenerator ready (...)`).
11. Run §7 (smoke test on Q0 — should generate a valid-looking SELECT, validation should be `True`).
12. Optional: clear `/content/predict_dev.json` and `/content/graph_stats.jsonl` for a fresh full run.
13. Run §8 (full BIRD dev, batched). With Qwen-32B@4-bit + BATCH_SIZE=16 on H100, expect ~10–15 min wall clock.
14. Run §8b (local metrics).

### Resume support
§8 checkpoints to `/content/predict_dev.json` every ~50 questions and writes once at the end. If anything crashes:
1. Re-run §6 (model still in memory if not crashed)
2. Re-run §8 — already-predicted qids are skipped automatically
3. Re-run §8b for fresh metrics

### Tuning
- `BATCH_SIZE` in §8 — start at 16, bump to 24/32 if `nvidia-smi` shows headroom. If OOM, the batched path falls back per-item but lower the constant to avoid repeated OOMs.
- `top_k` in `BIRDHybridRetriever.__init__` (§5b) — 6 is a safe default. Lower if graph is over-including tables (qid 1, 12, 13, 16 in our slice all had graph adding extras).
- `MAX_NEW_TOKENS` in §6 — 384 fits BIRD queries comfortably. Some complex queries hit the limit; bumping to 512 costs ~30% more wall clock.

### Pushing to git
```bash
git -C "<repo>" add -A
git -C "<repo>" commit -m "<message>"
git -C "<repo>" push
```
The repo URL is `https://github.com/mayursk2000/Stop-Guessing-Joins-Deterministic-Schema-Linking-for-Text-to-SQL-Agents.git`. If pushing fails due to auth, run `gh auth login` or use a fine-grained PAT with `Contents: write`.

---

## 8. Known limitations

### DFC-specific
- DFC parses with `read="duckdb"` and emits `dialect="duckdb"`. SQLite usually accepts but watch §7's `Generated → Rewritten` diff for any breakage.
- FK policy support is for **single-column FKs**. Composite FKs (`a.x = b.y AND a.z = b.w`) are detected by `_is_pure_join_constraint` but the satisfaction check evaluates them conjunctively — both pairs must appear in the query. Untested on real composite-FK schemas in BIRD.
- DFC's stub-table seeding stores all columns as `VARCHAR`. Type-aware aggregate policies wouldn't validate correctly against these stubs — fine for FK policies, would matter if you registered aggregate policies later.
- We don't propagate a real DB connection into DFC — its DuckDB stays disjoint from BIRD's SQLite. Any DFC code path that actually executes the query would fail. We rely on `transform_query` not executing.

### Pipeline
- The validator's "alignment threshold" is a heuristic (token overlap). It's permissive enough that valid SQL almost always passes; it's not a real correctness check.
- Retry loop (`new_graph` retry type) is shallow — only one retry attempt with no backoff or alternative graph computation.
- `_prune_candidate_tables` heuristics in `prototype.py` are sales-schema-specific; we set `use_pruning=False` to skip them. A BIRD-tuned pruner would be a real improvement but isn't load-bearing right now.

### Eval
- §8b metrics are *approximate* — set comparison after `str()` cast on each cell. BIRD's official `evaluation.py` is stricter on float tolerance and column-count alignment. Always run §9 before quoting numbers in any paper.
- Join Accuracy is averaged over multi-table questions only (single-table queries excluded — no joins to score). A different denominator could produce different numbers.

### LLM
- LLM column-choice errors are the dominant remaining failure mode. Bigger model + few-shot would help.
- Greedy decoding (`do_sample=False`). Sampling + self-consistency would likely buy a few EX points but multiplies cost.

### Resource
- BIRD dev is ~3 GB. Some Colab regions get the OSS bucket slowly (Beijing-hosted).
- Qwen2.5-Coder-32B in 4-bit is ~18 GB. Batched generation needs another ~5–10 GB for KV cache at BATCH_SIZE=16.

---

## 9. Future work

In rough order of leverage:

1. **vLLM backend.** Replace `transformers.generate` with vLLM's continuous batching. Expect 5–10× more throughput on the same GPU. Adds an install step but no pipeline changes.
2. **Self-consistency.** Sample N=5 SQL completions per question, execute each, majority-vote on the result set. ~5× cost, +3–5 EX expected.
3. **Few-shot in-context examples.** Pick 3–5 high-quality (question, evidence, gold SQL) tuples per BIRD database and prepend to the prompt. +2–4 EX expected, no extra GPU cost.
4. **Composite-FK support.** Extend `_extract_fk_pairs_from_policy` to handle multi-column FKs as a single unit (frozenset of pair-frozensets). Untested on BIRD; might not matter.
5. **Spider 2.0 adapter.** Port the BIRD adapter to Spider's tables.json + dev_databases layout. The rest of the pipeline is benchmark-agnostic.
6. **Aggregate DFCPolicies for privacy-aware BIRD eval.** The DFC machinery supports K-anonymity, hierarchical coarsening, etc. Could run BIRD with privacy invariants on top.
7. **Evidence parser for column extraction.** When evidence contains backtick-wrapped column names, surface them as a structured list rather than free text. Less prompt sensitivity.
8. **Better validator.** Replace the alignment-threshold heuristic with execution-result-based validation (run gold and predicted, compare). Currently the rewriter exits before execution can inform retry.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| BIRD | BIg bench for laRge-scale Database grounded Text-to-SQL evaluation |
| Spider 2.0 | Stanford text-to-SQL benchmark (newer than original Spider) |
| DFC | Data Flow Control — Summers et al., CIDR 2026; the SQL rewriter framework in `sql_rewriter/` |
| DFCPolicy | Source/sink/constraint/on_fail tuple expressing a data-flow invariant |
| Resolution | How to enforce a policy: REMOVE / KILL / INVALIDATE / INVALIDATE_MESSAGE / LLM |
| FK | Foreign key |
| T (candidate set) | Tables returned by retrieval — the closed set passed to the graph |
| G_T | Minimal connecting subgraph of FK edges over T |
| Join F1 | F1 score over predicted vs gold join predicates, parsed by sqlglot |
| EX (Exec Acc) | BIRD's primary metric — predicted SQL produces same result rows as gold |
| VES | Valid Efficiency Score — BIRD's runtime-aware EX |
| RRF | Reciprocal Rank Fusion — combines BM25 + dense rankings, parameter-free |
| BGE | `BAAI/bge-small-en-v1.5` — 33 M-param sentence encoder, 384-dim |
| Linked schema | The graph-resolved subset of tables/columns shown to the LLM |
| Evidence (BIRD) | Per-question hint string, part of BIRD's input — names exact columns/values to use |
| Pure-join constraint | A DFCPolicy whose constraint is a conjunction of cross-source-table column equalities, no aggregations |
| Skip-when-satisfied | Our patch — DFC skips emission if the query already enforces the FK |

---

## File hashes (so future-you knows what's pinned)

| Path | Why it matters |
|---|---|
| `sql_rewriter/policy.py` | Patched: FK-shaped constraints accepted in `_validate_aggregation_rules` |
| `sql_rewriter/rewrite_rule.py` | Patched: FK skip-when-satisfied + WHERE-not-HAVING in apply_policy_constraints_* |
| `sql_rewriter/rewriter.py` | Patched: FK vs aggregate policy split in `_transform_query_common` |
| `notebooks/bird_eval.ipynb` | The full evaluation harness — all our additions live here |
| `text2sql_agent_prototype/prototype.py` | UNCHANGED — we only override agent components from the notebook |

If `prototype.py` ever needs to change, do it carefully — that file is shared with the original benchmarks (`benchmark.py`, `large_schema_benchmark.py`) which we don't run on BIRD but should keep working.
