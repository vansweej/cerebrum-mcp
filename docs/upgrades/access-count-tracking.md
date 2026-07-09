# Deferred Refinement #4: Access Count Tracking — Shared Precursor

## Current State

Both `FrequencyBasedPromotion` and `AccessBasedDecay` read an `access_count` field from memory metadata, but **nothing in the runtime increments it**:

- **FrequencyBasedPromotion::score()** (`crates/cerebrum-core/src/promotion.rs:60-66`):
  ```rust
  let access_count = entry
      .metadata
      .get("access_count")
      .and_then(|v| v.parse::<usize>().ok())
      .unwrap_or(0);
  ```

- **AccessBasedDecay::score()** (`crates/cerebrum-core/src/decay.rs:97-103`):
  ```rust
  let access_count = entry
      .metadata
      .get("access_count")
      .and_then(|v| v.parse::<usize>().ok())
      .unwrap_or(0);
  ```

**The problem:** Only test helpers (`promotion.rs:212`, `decay.rs:224`) set `access_count` for testing. The runtime never increments it on recall. As a result:

- `FrequencyBasedPromotion` always scores memories identically (all have `access_count=0`).
- `AccessBasedDecay` always scores memories identically (all have `access_count=0`).
- Both strategies are **inert** — they cannot distinguish between frequently-recalled and rarely-recalled memories.

**Impact:** Tickets #2 (Wire Promotion Strategy) and #3 (Decay/Prune Pass) cannot be meaningfully implemented without this ticket.

---

## Proposed Change

Increment `access_count` in memory metadata on each `recall()` hit in the orchestrator, and persist the updated count via merge_insert:

1. When `MemoryOrchestrator::recall()` retrieves a memory from either Synapse or Cortex, increment its `access_count` in metadata.
2. For Synapse: update the in-memory entry directly.
3. For Cortex: call `store()` with the updated entry to persist the increment via merge_insert (atomic upsert).
4. Ensure the increment is idempotent (multiple recalls in the same session do not cause issues).

---

## Design Notes

### Where to Increment

The orchestrator's `recall()` method is the single point where all recalls flow through. This is the right place to increment:

```rust
pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
    let query_vec = self.embedder.embed(query).await?;
    
    // Retrieve from both tiers
    let synapse_results = self.synapse.retrieve(&query_vec, limit).await?;
    let cortex_results = self.cortex.retrieve(&query_vec, limit).await?;
    
    // Merge and rank
    let mut merged = self.merge_and_rank(synapse_results, cortex_results, limit);
    
    // NEW: Increment access_count for each result
    for entry in &mut merged {
        let mut count = entry
            .metadata
            .get("access_count")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        count += 1;
        entry.metadata.insert("access_count".to_string(), count.to_string());
        
        // Persist the increment
        if entry.tier == MemoryTier::Cortex {
            self.cortex.store(entry.clone()).await?;
        } else {
            self.synapse.store(entry.clone()).await?;
        }
    }
    
    Ok(merged)
}
```

### Idempotency

Incrementing on every recall is safe because:

- Each recall is a separate operation.
- The increment is persisted atomically via merge_insert (Cortex) or direct update (Synapse).
- Multiple recalls in the same session each increment the count independently — this is correct behavior (we want to track total recall frequency).

### Performance Implications

- **Synapse:** Incrementing in-memory is O(1). No I/O.
- **Cortex:** Each recalled memory triggers a merge_insert. For a recall that returns 10 memories, this is 10 writes. This is acceptable for typical recall patterns (< 100 recalls per session).

If performance becomes an issue, a future optimization could batch the increments (e.g., flush every 100 recalls).

### Metadata Format

Store `access_count` as a string in the metadata JSON (consistent with existing patterns):

```json
{
  "access_count": "5",
  "other_field": "value"
}
```

---

## Acceptance Criteria

- [ ] `MemoryOrchestrator::recall()` increments `access_count` for each retrieved memory.
- [ ] Increments are persisted: Synapse updates in-memory, Cortex calls `store()` to persist.
- [ ] Existing tests pass (no regressions).
- [ ] New tests verify that `access_count` increments correctly across multiple recalls.
- [ ] New tests verify that `FrequencyBasedPromotion` and `AccessBasedDecay` now score memories differently based on access count.
- [ ] Documentation updated to explain access count tracking.

---

## Risks

### Write Amplification

**Risk:** Each recall triggers writes (Cortex merge_insert). For high-recall workloads, this could become a bottleneck.

**Mitigation:** 
- Monitor write latency in observability metrics.
- If needed, implement batching: accumulate increments in memory and flush periodically.
- Consider a "dirty flag" approach: only persist if access_count changed significantly (e.g., every 10th increment).

### Metadata Bloat

**Risk:** Storing `access_count` as a string in JSON adds overhead.

**Mitigation:** This is negligible (a few bytes per memory). If metadata becomes a concern, consider a dedicated column in the LanceDB schema.

### Concurrency

**Risk:** Multiple concurrent recalls of the same memory could race on the increment.

**Mitigation:** LanceDB's merge_insert is atomic. Synapse uses `RwLock` for thread-safe updates. No race condition.

---

## Why Deferred

1. **Shared precursor:** This ticket unblocks both #2 and #3. Deferring it defers both.
2. **Phase 6 scope:** Adding write-on-recall would have complicated the Phase 6 implementation. Better to ship persistence first, then add access tracking.
3. **Not a blocker:** The system works without access tracking. It is an enhancement for frequency-based promotion and decay.

---

## Implementation Roadmap

1. **Modify `MemoryOrchestrator::recall()` to increment and persist** — 1 commit.
2. **Add tests for access count increment** — 1 commit.
3. **Update documentation** — 1 commit.

**Total effort:** ~3 commits, ~1-2 days of work.

---

## Related Tickets

- **#2: Wire Promotion Strategy** — Depends on this ticket. `FrequencyBasedPromotion` needs meaningful access counts.
- **#3: Decay/Prune Pass** — Depends on this ticket. `AccessBasedDecay` needs meaningful access counts.
