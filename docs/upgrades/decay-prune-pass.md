# Deferred Refinement #3: Decay/Prune Pass — Automatic Memory Cleanup

## Current State

Cerebrum includes a full `DecayStrategy` framework (`crates/cerebrum-core/src/decay.rs`) with four implementations:

- `TimeBasedDecay` — Score based on age (older = lower score).
- `AccessBasedDecay` — Score based on access count (rarely accessed = lower score).
- `RelevanceBasedDecay` — Score based on salience (low importance = lower score).
- `HybridDecay` — Weighted combination of the above.

All four are **fully unit-tested** but **never invoked at runtime**. There is **no decay/prune path** in the orchestrator:

- Memories are never automatically demoted from Cortex to Synapse.
- Memories are never automatically deleted from Cortex.
- Cortex can grow unbounded over time.

**Impact:** Without decay, long-running agents accumulate stale, low-value memories indefinitely, increasing retrieval latency and storage costs.

---

## Proposed Change

Add a **decay/prune pass** that scores memories via a `DecayStrategy` and removes or demotes low-scoring memories:

1. Add a `decay_strategy: Box<dyn DecayStrategy>` field to `MemoryOrchestrator`.
2. Add a new method `prune_cortex()` that:
   - Retrieves all memories from Cortex.
   - Scores each via the decay strategy.
   - Deletes memories scoring below `decay_threshold`.
   - Optionally demotes low-scoring memories back to Synapse (if they are still relevant).
3. Call `prune_cortex()` either:
   - **Periodically:** Via a background task (e.g., every 24 hours).
   - **On end_session():** After promotion, before session ends.
4. Log all deletions for auditability (provenance).

---

## Design Notes

### Decay vs. Deletion

Two strategies for handling low-scoring memories:

1. **Deletion:** Remove permanently. Simplest, but irreversible.
2. **Demotion:** Move back to Synapse. Recoverable, but Synapse is session-scoped (lost on exit).

**Recommendation:** Start with deletion for simplicity. Add demotion in a future refinement if needed.

### Decay Threshold

Add to `Config`:

```rust
pub struct Config {
    // ... existing fields ...
    
    /// Which decay strategy to use during prune_cortex().
    pub decay_strategy: DecayStrategyKind,
    
    /// Threshold for decay (0.0-1.0). Memories scoring < threshold are deleted.
    pub decay_threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            decay_strategy: DecayStrategyKind::Hybrid,
            decay_threshold: 0.3,
        }
    }
}
```

### DecayContext

The `DecayStrategy::score()` method takes a `DecayContext` parameter:

```rust
pub struct DecayContext {
    pub now: SystemTime,
}
```

This provides the current time for age-based decay. Ensure it is populated correctly in `prune_cortex()`.

### Auditability & Provenance

Log every deletion with full context:

```rust
info!(
    "Decaying memory: id={}, content_preview={}, score={}, threshold={}",
    entry.id, &entry.content[..50.min(entry.content.len())], score, decay_threshold
);
```

This allows operators to audit what was deleted and why.

### Periodic vs. On-Demand

**Periodic (background task):**
- Pros: Automatic, does not block agent operations.
- Cons: Requires a background task (systemd service, launchd agent, or in-process timer).

**On-demand (end_session):**
- Pros: Simple, no background infrastructure.
- Cons: Only runs when sessions end; does not help long-running agents.

**Recommendation:** Start with on-demand (end_session). Add periodic pruning in a future refinement.

### Performance

Pruning all memories in Cortex is O(N). For large datasets (> 100k rows), this could be slow. Mitigations:

- Run pruning asynchronously (do not block the agent).
- Add a batch size limit: prune in chunks (e.g., 1000 memories at a time).
- Add observability: track pruning latency and memory count.

---

## Acceptance Criteria

- [ ] `MemoryOrchestrator` accepts a `DecayStrategy` (or creates one from `Config`).
- [ ] `prune_cortex()` method scores all Cortex memories and deletes low-scoring ones.
- [ ] Deletions are logged with full context (id, content preview, score, threshold).
- [ ] `end_session()` calls `prune_cortex()` after promotion.
- [ ] Decay strategy is configurable via `Config`.
- [ ] Existing tests pass (no regressions).
- [ ] New tests verify that different strategies produce different decay decisions.
- [ ] New tests verify that deletions are logged correctly.
- [ ] Documentation updated to explain decay strategies and auditability.

---

## Risks

### Irreversible Deletion

**Risk:** Deleting memories is permanent. If the decay threshold is too low, valuable memories may be lost.

**Mitigation:**
- Default threshold (0.3) is conservative. Memories must score very low to be deleted.
- Log all deletions. Operators can review logs to audit what was deleted.
- Add a "dry-run" mode: log what would be deleted without actually deleting.
- Consider demotion (move to Synapse) instead of deletion for a future refinement.

### Depends on #4

**Risk:** `AccessBasedDecay` and `HybridDecay` require meaningful access counts, which are not incremented until #4 is implemented.

**Mitigation:** Implement #4 first. Until then, `AccessBasedDecay` will score all memories identically (access_count=0).

### Performance

**Risk:** Pruning all memories in Cortex is O(N). For large datasets, this could be slow.

**Mitigation:**
- Run pruning asynchronously.
- Add batch size limits.
- Add observability to track pruning latency.

---

## Why Deferred

1. **Depends on #4:** Access-based decay requires meaningful access counts.
2. **Irreversible operation:** Deletion is permanent. Better to ship the framework (proven by tests) and activate it carefully later.
3. **Phase 6 scope:** Adding automatic pruning would have complicated the Phase 6 implementation. Better to ship persistence first, then add cleanup.
4. **Not a blocker:** The system works without decay. Unbounded growth is acceptable for typical use cases.

---

## Implementation Roadmap

1. **Add `decay_strategy` and `decay_threshold` to Config** — 1 commit.
2. **Implement `MemoryOrchestrator::prune_cortex()` method** — 1 commit.
3. **Call `prune_cortex()` from `end_session()`** — 1 commit.
4. **Add tests for decay and deletion** — 1 commit.
5. **Update documentation** — 1 commit.

**Total effort:** ~5 commits, ~3-4 days of work.

---

## Related Tickets

- **#4: Access Count Tracking** — Prerequisite. Must be implemented first for access-based decay.
- **#2: Wire Promotion Strategy** — Complementary. Promotion adds high-value memories; decay removes low-value ones.
