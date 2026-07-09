# LLM Memory Concepts Glossary

This glossary explains the core concepts underlying Cerebrum's two-tier memory system. Each section is grounded in the actual codebase implementation.

---

## 1. Why LLMs Need External Memory

Large language models operate with **stateless context windows** — each inference starts fresh with no memory of prior conversations or interactions. This creates two fundamental problems:

- **Conversation continuity:** Without external storage, an LLM cannot recall facts from earlier in a session or across sessions.
- **Knowledge accumulation:** An LLM cannot learn from user feedback or build a persistent knowledge base over time.

Cerebrum solves this by providing a persistent, queryable memory layer that agents can recall from before each inference. The agent embeds its query, retrieves relevant memories, and includes them in the prompt context — effectively extending the LLM's working memory beyond a single request.

---

## 2. Embeddings

An **embedding** is a dense vector representation of text in a high-dimensional space. Semantically similar texts map to nearby vectors; dissimilar texts map far apart.

Cerebrum uses **nomic-embed-text**, a 768-dimensional embedding model served by Ollama. When you call `embed("search_document: user preferences")`, the Ollama endpoint returns a 768-element vector of f32 values.

**Source:** `crates/cerebrum-core/src/fastembed_embedder.rs:48, 81, 100`

The embedding dimension (768) is validated at construction time and must match the LanceDB schema. Mismatches are caught before any data is written, preventing silent schema corruption.

---

## 3. Relevance vs. Salience

Two independent scoring dimensions determine memory ranking:

- **Relevance:** How semantically similar is the memory to the query? Computed as **cosine similarity** between the query embedding and the memory's stored embedding. Range: [0.0, 1.0].
- **Salience:** How important is this memory? A caller-supplied importance score (0.0–1.0) set when the memory is created. Reflects user intent, not semantic content.

**Blended ranking:** Cerebrum combines both signals with a weighted average:
```
score = 0.7 * relevance + 0.3 * salience
```

This means a highly relevant but low-salience memory can be outranked by a less relevant but high-salience memory. The 70/30 split prioritizes semantic match but respects explicit importance signals.

**Source:** `crates/cerebrum-core/src/lancedb_cortex.rs:retrieve()` — the blending formula is applied per-row during retrieval.

---

## 4. Two-Tier Architecture

Cerebrum splits memory into two tiers with different performance and persistence characteristics:

### Synapse (Short-Term, Session-Scoped)
- **Storage:** In-memory `Arc<RwLock<HashMap>>` — fast, volatile.
- **Scope:** Session-scoped; cleared on process exit.
- **Use case:** Rapid recall within a single agent session; no I/O latency.
- **Implementation:** `crates/cerebrum-core/src/synapse.rs`

### Cortex (Long-Term, Persistent)
- **Storage:** LanceDB on-disk vector database — durable, queryable.
- **Scope:** Global; survives process restarts.
- **Use case:** Persistent knowledge base; cross-session recall.
- **Implementation:** `crates/cerebrum-core/src/lancedb_cortex.rs`

The orchestrator queries both tiers and merges results, presenting a unified memory interface to agents.

---

## 5. Memory Lifecycle

A memory follows this lifecycle:

1. **Remember:** Agent calls `cerebrum_remember(content, salience)`. The memory is stored in Synapse immediately (fast, in-memory).
2. **Recall:** Agent calls `cerebrum_recall(query)`. The orchestrator embeds the query, retrieves from both Synapse and Cortex, blends results, and returns ranked memories.
3. **Promote (end_session):** When the agent session ends, the orchestrator calls `end_session()`. Memories with `salience >= auto_promote_threshold` are moved from Synapse to Cortex (persistent storage).
4. **Decay (future):** A planned feature (not yet implemented) will periodically score memories via a `DecayStrategy` and demote or delete low-scoring memories to prevent unbounded growth.

**Source:** `crates/cerebrum-core/src/orchestrator.rs:361` — inline promotion check during `end_session()`.

---

## 6. Scoping

Memories can be tagged with a **scope** to control visibility:

- **Global:** Visible to all agents and users. Default scope.
- **user:<id>:** Visible only to a specific user.
- **agent:<id>:** Visible only to a specific agent.
- **session:<id>:** Visible only within a specific session.

**Current state:** All memories created via the MCP tools land in the **global** scope. Per-agent and per-user scoping are defined in the schema but not yet exposed in the tool interface.

**Source:** `crates/cerebrum-core/src/models.rs` — `MemoryScope` enum and matching logic.

---

## 7. Retrieve-Then-Rerank

Cortex retrieval today uses a **brute-force full-scan** approach:

1. Fetch all rows from the LanceDB table into Rust memory.
2. Compute cosine similarity between the query embedding and each row's embedding.
3. Blend each row's similarity with its salience: `score = 0.7 * sim + 0.3 * salience`.
4. Sort by score (descending).
5. Return the top `limit` results.

**Complexity:** O(N) where N is the total number of memories in Cortex. For small datasets (< 100k rows), this is acceptable. For larger datasets, this becomes a bottleneck.

**Source:** `crates/cerebrum-core/src/lancedb_cortex.rs:retrieve()` — lines showing `table.query().execute()`, cosine computation, and blending.

---

## 8. Approximate Nearest Neighbors (ANN) Indexes

LanceDB supports **ANN indexes** via the `nearest_to()` API, which uses approximate similarity search to avoid full-table scans. Instead of comparing the query to every row, an ANN index narrows the search space to a small candidate set, then returns the top-k results.

**Contrast with current implementation:** Athenaeum (the sibling project) uses `store.rs:search()` with `nearest_to().distance_type(Cosine).limit(k)` — pushing the similarity computation to LanceDB and letting it manage the index.

**Why not yet in Cerebrum:** The blended ranking (0.7 * similarity + 0.3 * salience) cannot be pushed to LanceDB's distance metric. A future optimization (Deferred Refinement #1) will implement retrieve-then-rerank: fetch k*multiplier candidates via ANN, then exact blend-rerank in Rust.

**Source:** Athenaeum `crates/core/src/store.rs:search()` — reference implementation.

---

## 9. Write Path: Atomic Upsert

When storing a memory, Cerebrum uses **LanceDB `merge_insert`** keyed on the memory's `id`:

```rust
let mut mi = table.merge_insert(&["id"]);
mi.when_matched_update_all(None).when_not_matched_insert_all();
mi.execute(Box::new(reader)).await?;
```

This is an **atomic upsert:** if a memory with the same `id` already exists, it is updated in-place; otherwise, a new row is inserted. No duplicate rows are created, and the operation is transactional.

**Source:** `crates/cerebrum-core/src/lancedb_cortex.rs:store()` — the merge_insert call.

---

## 10. Evaluation: Qualitative Human Grading

Cerebrum does not yet have automated evaluation metrics (Recall@k, Precision, MRR, nDCG). The sibling project **Athenaeum** includes a qualitative evaluation tool (`relevance-eval`) that:

1. Loads a set of test queries from `queries.toml`.
2. For each query, retrieves results from the vector store.
3. Asks a human to grade each result as Green (relevant), Yellow (partially relevant), or Red (irrelevant).
4. Aggregates verdicts to assess retrieval quality.

This is a **manual, non-automated** process, excluded from CI. It is useful for understanding retrieval behavior on real data but does not scale to continuous validation.

**Source:** Athenaeum `crates/core/src/relevance_eval.rs` — reference implementation.

---

## 11. Operational Concerns

### LanceDB On-Disk Store

Cerebrum's Cortex tier stores data in a LanceDB database on disk. The database path is **CWD-relative** by default: `./data/cerebrum`. This means:

- The MCP server's working directory must be set to a durable location (e.g., `~/.local/share/cerebrum`).
- LanceDB creates the directory automatically on first write; no manual `mkdir` is needed.
- Data persists across process restarts as long as the directory is not deleted.

**Source:** `crates/cerebrum-core/src/config.rs:Config::default()` — `db_path: PathBuf::from("./data/cerebrum")`.

### Embedding Dimension Validation

The embedding dimension (768 for nomic-embed-text) is validated at construction time. If the embedder reports a different dimension than the schema expects, construction fails with a clear error message. This prevents silent schema mismatches.

**Source:** `crates/cerebrum-core/src/lancedb_cortex.rs:new()` — dimension assertion.

---

## 12. Quick Reference: Concepts → Source Files

| Concept | Primary Source | Key Lines |
|---------|---|---|
| Embeddings (768-dim) | `fastembed_embedder.rs` | 48, 81, 100, 273 |
| Blended ranking (0.7 sim + 0.3 salience) | `lancedb_cortex.rs` | `retrieve()` method |
| Synapse (in-memory) | `synapse.rs` | Entire module |
| Cortex (LanceDB) | `lancedb_cortex.rs` | Entire module |
| Memory lifecycle | `orchestrator.rs` | `end_session()` at line 361 |
| Scoping | `models.rs` | `MemoryScope` enum |
| Atomic upsert | `lancedb_cortex.rs` | `store()` method, `merge_insert` call |
| Dimension validation | `lancedb_cortex.rs` | `new()` constructor |
| Config (db_path) | `config.rs` | `Config::default()` |

---

## Maturity Note: Promotion and Decay Frameworks

### Current State

Cerebrum includes **two strategy frameworks** for advanced memory management:

1. **PromotionStrategy** (`crates/cerebrum-core/src/promotion.rs`)
   - Defines four implementations: `FrequencyBasedPromotion`, `RecencyBasedPromotion`, `ImportanceBasedPromotion`, `HybridPromotion`.
   - Each scores a memory to determine if it should be promoted from Synapse to Cortex.
   - All implementations are **fully unit-tested**.

2. **DecayStrategy** (`crates/cerebrum-core/src/decay.rs`)
   - Defines four implementations: `TimeBasedDecay`, `AccessBasedDecay`, `RelevanceBasedDecay`, `HybridDecay`.
   - Each scores a memory to determine if it should be demoted or deleted from Cortex.
   - All implementations are **fully unit-tested**.

### The Gap: Not Wired Into Runtime

**Despite being fully implemented and tested, neither framework is invoked at runtime:**

- **Promotion:** The `end_session()` method uses an **inline salience-threshold check** (`if entry.salience >= auto_promote_threshold`) at `orchestrator.rs:361`, not the `PromotionStrategy` framework.
- **Decay:** There is **no decay/prune path** in the runtime. Memories are never automatically demoted or deleted based on age, access count, or relevance.

### Access Count: Read But Never Incremented

Both `FrequencyBasedPromotion::score()` (`promotion.rs:60-66`) and `AccessBasedDecay::score()` (`decay.rs:97-103`) read an `access_count` field from memory metadata. However:

- **Nothing in the runtime increments `access_count`** when a memory is recalled.
- Only **test helpers** (`promotion.rs:212`, `decay.rs:224`) set this field for testing purposes.
- As a result, both strategies are **inert** — they would score all memories identically regardless of recall frequency.

### Why This Matters

The frameworks are **aspirational infrastructure** — they define the right abstractions and are proven correct by tests, but the runtime does not use them. This is intentional: the team prioritized shipping a working two-tier system with LanceDB persistence (Phase 6) before adding automatic promotion and decay logic.

### Next Steps

Five **Deferred Refinements** (documented in `docs/upgrades/`) will wire these frameworks into the runtime:

1. **Cortex Vector Search** — Optimize retrieval with ANN indexes.
2. **Access Count Tracking** — Increment on recall; shared precursor for promotion and decay.
3. **Wire Promotion Strategy** — Replace inline check with configurable strategy.
4. **Decay/Prune Pass** — Add periodic or end-session decay logic.
5. **Shared LanceDB Crate** — Extract generic `VectorStore<R>` for reuse across projects.

Until these are implemented, memory promotion is manual (salience-based) and decay does not occur.

---

## Summary

Cerebrum's memory system is a **two-tier, persistent, semantically-searchable store** with:

- **Fast recall** via in-memory Synapse (session-scoped).
- **Durable storage** via LanceDB Cortex (persistent, cross-session).
- **Semantic ranking** via embeddings and blended relevance/salience scoring.
- **Atomic writes** via merge_insert (no duplicates, no race conditions).
- **Extensible promotion/decay** via strategy frameworks (currently dormant, planned for future phases).

The system is production-ready for basic remember/recall workflows. Advanced features (automatic promotion, decay, ANN optimization) are deferred but architecturally sound.
