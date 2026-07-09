# Deferred Refinement #5: Shared LanceDB Crate — Cross-Project Reuse

## Current State

Cerebrum and Athenaeum (the sibling project) independently implement near-identical LanceDB access patterns:

**Cerebrum:** `crates/cerebrum-core/src/lancedb_cortex.rs`
- Connects to LanceDB: `lancedb::connect(path).execute().await`
- Creates schema: `conn.create_empty_table(table_name, schema(dim))`
- Builds Arrow batches: `RecordBatch::try_new(schema, vec![...])`
- Decodes batches: `batch_to_records()` with column extraction
- Performs upsert: `table.merge_insert(&["id"]).execute()`
- Retrieves: `table.query().execute()` with full-scan

**Athenaeum:** `crates/core/src/store.rs`
- Connects to LanceDB: `lancedb::connect(path).execute().await`
- Creates schema: `conn.create_empty_table(table_name, schema(...))`
- Builds Arrow batches: `RecordBatch::try_new(schema, vec![...])`
- Decodes batches: similar column extraction
- Performs upsert: `table.merge_insert(&["id"]).execute()`
- Retrieves: `table.search(query_embedding).distance_type(Cosine).limit(k)`

**Duplication:** ~60% of the code is identical. Both projects:
- Handle Arrow schema construction and validation.
- Manage RecordBatch building and decoding.
- Implement merge_insert upsert logic.
- Manage LanceDB connection lifecycle.

**Why this matters:** Every bug fix, optimization, or feature in one project must be manually ported to the other. This is error-prone and slows development.

---

## Proposed Change

Extract a **generic `VectorStore<R>` crate** that both projects consume:

1. Create a new crate: `crates/vector-store/` (or publish to crates.io as `cerebrum-vector-store`).
2. Define a generic trait `VectorStore<R>` where `R` is the record type (e.g., `LanceDBMemoryRecord`, `AthenaeumDocument`).
3. Implement `LanceDBVectorStore<R>` that handles:
   - Connection management.
   - Schema building (generic over record type).
   - Batch building and decoding (generic over record type).
   - Upsert via merge_insert.
   - Full-scan retrieval (current behavior).
   - ANN retrieval via `nearest_to()` (future, from Refinement #1).
4. Both Cerebrum and Athenaeum consume `LanceDBVectorStore<R>` instead of reimplementing.

---

## Design Notes

### Generic Record Type

The `VectorStore<R>` trait is generic over the record type:

```rust
pub trait VectorStore<R: Record>: Send + Sync {
    async fn store(&self, record: R) -> Result<()>;
    async fn retrieve(&self, query_vec: &[f32], limit: usize) -> Result<Vec<R>>;
    async fn delete(&self, id: &str) -> Result<()>;
    async fn list(&self) -> Result<Vec<R>>;
    async fn len(&self) -> Result<usize>;
}

pub trait Record: Send + Sync {
    fn id(&self) -> String;
    fn embedding(&self) -> &[f32];
    fn to_arrow_batch(&self, dim: usize) -> Result<RecordBatch>;
    fn from_arrow_batch(batch: &RecordBatch) -> Result<Vec<Self>>;
}
```

Each project implements `Record` for its own type:
- Cerebrum: `impl Record for LanceDBMemoryRecord`
- Athenaeum: `impl Record for AthenaeumDocument`

### Schema Building

Schema construction is generic over the record type. Each `Record` impl provides its own schema:

```rust
pub trait Record {
    fn schema(dim: usize) -> Arc<Schema>;
}
```

### Upsert Key

The upsert key (currently `["id"]`) is configurable:

```rust
pub struct LanceDBVectorStore<R: Record> {
    conn: Connection,
    table_name: String,
    upsert_keys: Vec<String>,
    // ...
}
```

### ANN Support (Future)

Once Refinement #1 (retrieve-then-rerank) lands in Cerebrum, the shared crate can add:

```rust
pub trait VectorStore<R: Record> {
    async fn retrieve_ann(&self, query_vec: &[f32], limit: usize, multiplier: usize) -> Result<Vec<R>>;
}
```

Both projects benefit from the optimization automatically.

### Location

Two options:

1. **Monorepo approach:** Create `crates/vector-store/` in the cerebrum-mcp repo, then import it in Athenaeum via git dependency.
2. **Separate crate:** Publish to crates.io as `cerebrum-vector-store` (or similar), both projects depend on it.

**Recommendation:** Start with monorepo approach (simpler), migrate to crates.io later if needed.

---

## Acceptance Criteria

- [ ] New crate `crates/vector-store/` created with `VectorStore<R>` trait and `Record` trait.
- [ ] `LanceDBVectorStore<R>` implements `VectorStore<R>` with full-scan retrieval.
- [ ] Cerebrum's `lancedb_cortex.rs` refactored to use `LanceDBVectorStore<LanceDBMemoryRecord>`.
- [ ] Athenaeum's `store.rs` refactored to use `LanceDBVectorStore<AthenaeumDocument>`.
- [ ] All existing tests pass (no behavioural change).
- [ ] New tests verify generic `VectorStore<R>` with both record types.
- [ ] Documentation updated to explain the shared crate and how to implement `Record`.
- [ ] Crate is published to crates.io (optional, can defer).

---

## Risks

### Abstraction Leakiness

**Risk:** The generic `VectorStore<R>` abstraction may not fit both projects perfectly, leading to awkward implementations or missing features.

**Mitigation:**
- Start with a minimal trait (store, retrieve, delete, list, len).
- Add features incrementally as both projects need them.
- Keep the abstraction close to LanceDB's API (do not over-abstract).

### Coordination Overhead

**Risk:** Changes to the shared crate require coordination between Cerebrum and Athenaeum teams.

**Mitigation:**
- Use semantic versioning. Breaking changes bump major version.
- Maintain backward compatibility where possible.
- Document the crate thoroughly so both projects can use it independently.

### Depends on #1

**Risk:** The shared crate should include ANN retrieval (retrieve-then-rerank), which is not yet implemented.

**Mitigation:** Implement #1 first. Once retrieve-then-rerank is stable in Cerebrum, add it to the shared crate. Both projects benefit automatically.

---

## Why Deferred

1. **Depends on #1:** The shared crate should include ANN retrieval. Better to defer until retrieve-then-rerank is stable.
2. **Cross-repo coordination:** Requires alignment with Athenaeum team. Better to defer than rush.
3. **Not a blocker:** Both projects work independently. Sharing is an optimization, not a correctness fix.
4. **Phase 6 scope:** Extracting a shared crate would have complicated the Phase 6 implementation. Better to ship both projects independently first, then extract the common pattern.

---

## Implementation Roadmap

1. **Design `VectorStore<R>` and `Record` traits** — 1 commit.
2. **Implement `LanceDBVectorStore<R>` with full-scan retrieval** — 1 commit.
3. **Refactor Cerebrum to use `LanceDBVectorStore`** — 1 commit.
4. **Refactor Athenaeum to use `LanceDBVectorStore`** — 1 commit (in Athenaeum repo).
5. **Add comprehensive tests** — 1 commit.
6. **Publish to crates.io** (optional) — 1 commit.

**Total effort:** ~5-6 commits, ~4-5 days of work (including cross-repo coordination).

---

## Related Tickets

- **#1: Cortex Vector Search** — Prerequisite. Once retrieve-then-rerank is stable, add ANN support to the shared crate.
- **#4: Access Count Tracking** — Orthogonal. Both projects can benefit from access count tracking independently.
