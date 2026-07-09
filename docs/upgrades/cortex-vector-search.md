# Deferred Refinement #1: Cortex Vector Search — Retrieve-Then-Rerank with ANN

## Current State

Cortex retrieval today uses a **brute-force full-scan** approach:

1. Fetch all rows from the LanceDB table into Rust memory via `table.query().execute()`.
2. Compute cosine similarity between the query embedding and each row's embedding.
3. Blend each row's similarity with its salience: `score = 0.7 * sim + 0.3 * salience`.
4. Sort by score (descending).
5. Return the top `limit` results.

**Complexity:** O(N) where N is the total number of memories in Cortex.

**Source:** `crates/cerebrum-core/src/lancedb_cortex.rs:retrieve()` — the full-scan pattern is visible in the `table.query().execute()` call followed by per-row cosine computation and blending.

**Why this works today:** For small datasets (< 100k rows), the latency is acceptable. The blended ranking (0.7 * similarity + 0.3 * salience) cannot be pushed to LanceDB's distance metric, so full-scan is necessary to compute the blend correctly.

---

## Proposed Change

Implement **retrieve-then-rerank** using LanceDB's Approximate Nearest Neighbors (ANN) index:

1. Use LanceDB `nearest_to().distance_type(Cosine).limit(k*multiplier)` to fetch a candidate set via ANN (fast, approximate).
2. In Rust, compute the exact blend `0.7 * similarity + 0.3 * salience` for each candidate.
3. Sort by blended score and return the top `limit` results.

**Complexity:** O(log N) for the ANN search + O(k*multiplier * log(k*multiplier)) for sorting candidates. For typical `limit=10` and `multiplier=10`, this is O(log N) + O(100 log 100) — orders of magnitude faster than O(N).

**Approximation cap:** The `multiplier` parameter controls the trade-off between speed and accuracy. A multiplier of 10 means we fetch 10x the requested limit via ANN, then rerank. This ensures we do not miss relevant memories due to ANN approximation.

---

## Design Notes

### Why Retrieve-Then-Rerank?

LanceDB's `nearest_to()` API computes distance using a single metric (e.g., Cosine). Cerebrum's blended ranking combines two independent signals (similarity + salience), which cannot be expressed as a single distance metric. Therefore:

- We cannot push the blend to LanceDB.
- We must fetch candidates and rerank in Rust.

### Reference Implementation

Athenaeum (the sibling project) uses a similar pattern in `crates/core/src/store.rs:search()`:

```rust
let results = table
    .search(query_embedding)
    .distance_type(DistanceType::Cosine)
    .limit(k)
    .execute()
    .await?;
```

Athenaeum does not blend with salience, so it can use LanceDB's result directly. Cerebrum must fetch candidates and rerank.

### Approximation Safety

The `multiplier` parameter ensures we do not miss relevant memories:

- **multiplier=1:** Fetch exactly `limit` results via ANN. Fastest, but may miss relevant memories due to ANN approximation.
- **multiplier=10:** Fetch 10x the limit via ANN, rerank in Rust. Slower, but much safer — the exact blend is computed over a larger candidate set.
- **multiplier=100:** Fetch 100x the limit. Approaches full-scan accuracy but loses most of the speed benefit.

Recommended default: **multiplier=10** (good balance of speed and accuracy).

### Schema Changes

No schema changes required. The embedding column already exists and is indexed by LanceDB.

### Configuration

Add to `crates/cerebrum-core/src/config.rs`:

```rust
pub struct Config {
    // ... existing fields ...
    
    /// Multiplier for ANN candidate set size (e.g., 10 means fetch 10x the limit).
    pub cortex_ann_multiplier: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            cortex_ann_multiplier: 10,
        }
    }
}
```

---

## Acceptance Criteria

- [ ] `LanceDBCortex::retrieve()` uses `nearest_to().distance_type(Cosine).limit(k*multiplier)` instead of full-scan.
- [ ] Blended ranking (0.7 * similarity + 0.3 * salience) is computed in Rust over the candidate set.
- [ ] Behavioural parity: results are identical to full-scan for small datasets (< 1000 rows).
- [ ] Performance: retrieval latency is sub-linear (O(log N)) for large datasets (> 100k rows).
- [ ] Configuration: `cortex_ann_multiplier` is configurable via `Config`.
- [ ] Tests: new tests verify behavioural parity and performance characteristics.
- [ ] Documentation: updated `docs/concepts/llm-memory.md` to explain ANN optimization.

---

## Risks

### Approximation Errors

**Risk:** ANN approximation may miss relevant memories if the multiplier is too small.

**Mitigation:** Default multiplier=10 provides a safety margin. Add observability (metrics) to track how often the exact blend would have ranked a memory differently than the ANN candidate set.

### Index Staleness

**Risk:** LanceDB's ANN index may become stale if not rebuilt after bulk inserts.

**Mitigation:** LanceDB rebuilds indexes automatically on write. Verify this in testing.

### Performance Regression

**Risk:** For very small datasets (< 100 rows), the ANN overhead may be slower than full-scan.

**Mitigation:** Add a threshold: if `row_count < 1000`, use full-scan; otherwise, use ANN. This is a micro-optimization but worth considering.

---

## Why Deferred

1. **Phase 6 scope:** LanceDB persistence and FastEmbed integration were already substantial. Adding ANN optimization would have delayed shipping.
2. **Not a blocker:** The system is production-ready with full-scan retrieval. ANN is an optimization, not a correctness fix.
3. **Depends on stable schema:** The embedding column and LanceDB integration must be stable before optimizing retrieval. Phase 6 delivered that stability.
4. **Unblocks #5:** Once retrieve-then-rerank is stable, the shared `VectorStore<R>` crate can be extracted with confidence.

---

## Implementation Roadmap

1. **Add `cortex_ann_multiplier` to Config** — 1 commit.
2. **Refactor `LanceDBCortex::retrieve()` to use ANN** — 1 commit.
3. **Add tests for behavioural parity and performance** — 1 commit.
4. **Update documentation** — 1 commit.

**Total effort:** ~4 commits, ~2-3 days of work.

---

## Related Tickets

- **#5: Shared LanceDB Crate** — Depends on this ticket. Once retrieve-then-rerank is stable, extract the generic `VectorStore<R>` pattern.
