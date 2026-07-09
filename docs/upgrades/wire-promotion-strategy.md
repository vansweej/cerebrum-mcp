# Deferred Refinement #2: Wire Promotion Strategy — Activate the Framework

## Current State

The `end_session()` method uses an **inline salience-threshold check** to decide which memories to promote from Synapse to Cortex:

**Source:** `crates/cerebrum-core/src/orchestrator.rs:361`

```rust
if entry.salience >= auto_promote_threshold {
    // Promote to Cortex
}
```

This is a **hardcoded, single-criterion** promotion rule: promote if salience is high enough.

**The framework exists but is unused:** Cerebrum includes a full `PromotionStrategy` framework (`crates/cerebrum-core/src/promotion.rs`) with four implementations:

- `FrequencyBasedPromotion` — Score based on access count.
- `RecencyBasedPromotion` — Score based on age (newer = higher score).
- `ImportanceBasedPromotion` — Score based on salience (current behavior).
- `HybridPromotion` — Weighted combination of the above.

All four are **fully unit-tested** but **never called at runtime**. The framework is aspirational infrastructure, not active code.

---

## Proposed Change

Replace the inline salience-threshold check with a configurable `PromotionStrategy`:

1. Add a `promotion_strategy: Box<dyn PromotionStrategy>` field to `MemoryOrchestrator`.
2. In `end_session()`, replace the inline check with:
   ```rust
   let score = self.promotion_strategy.score(&entry, &context);
   if score >= self.promotion_threshold {
       // Promote to Cortex
   }
   ```
3. Default to `HybridPromotion` (combines frequency, recency, and importance).
4. Make the strategy configurable via `Config` or constructor parameter.

---

## Design Notes

### Why Hybrid by Default?

`HybridPromotion` combines three signals:

- **Frequency** (access count): Memories recalled often are valuable.
- **Recency** (age): Newer memories are more relevant to current context.
- **Importance** (salience): User-supplied importance is explicit signal.

This is more robust than salience-only promotion. It captures both explicit user intent and implicit usage patterns.

### PromotionContext

The `PromotionStrategy::score()` method takes a `PromotionContext` parameter:

```rust
pub struct PromotionContext {
    pub now: SystemTime,
    pub session_duration: Duration,
}
```

This provides temporal context for recency-based scoring. Ensure it is populated correctly in `end_session()`.

### Backward Compatibility

The current behavior (salience-only promotion) is equivalent to `ImportanceBasedPromotion`. To maintain backward compatibility:

- Default to `HybridPromotion` (better behavior).
- Allow users to opt into `ImportanceBasedPromotion` if they prefer the old behavior.

### Configuration

Add to `crates/cerebrum-core/src/config.rs`:

```rust
pub enum PromotionStrategyKind {
    Frequency,
    Recency,
    Importance,
    Hybrid,
}

pub struct Config {
    // ... existing fields ...
    
    /// Which promotion strategy to use during end_session().
    pub promotion_strategy: PromotionStrategyKind,
    
    /// Threshold for promotion (0.0-1.0). Memories scoring >= threshold are promoted.
    pub promotion_threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            promotion_strategy: PromotionStrategyKind::Hybrid,
            promotion_threshold: 0.5,
        }
    }
}
```

---

## Acceptance Criteria

- [ ] `MemoryOrchestrator` accepts a `PromotionStrategy` (or creates one from `Config`).
- [ ] `end_session()` uses the strategy instead of the inline check.
- [ ] Inline salience-threshold check is removed.
- [ ] Default strategy is `HybridPromotion`.
- [ ] Strategy is configurable via `Config`.
- [ ] Existing tests pass (behavior is equivalent for salience-only promotion).
- [ ] New tests verify that different strategies produce different promotion decisions.
- [ ] Documentation updated to explain promotion strategies.

---

## Risks

### Behavior Change

**Risk:** Switching from salience-only to hybrid promotion may promote different memories, changing agent behavior.

**Mitigation:**
- Run the change on a test dataset first.
- Add observability: log which memories are promoted and why.
- Provide a config option to revert to `ImportanceBasedPromotion` if needed.

### Performance

**Risk:** Computing strategy scores for every memory in `end_session()` could be slow.

**Mitigation:** Strategy scoring is O(1) per memory. For typical session sizes (< 1000 memories), this is negligible.

### Depends on #4

**Risk:** `FrequencyBasedPromotion` and `HybridPromotion` require meaningful access counts, which are not incremented until #4 is implemented.

**Mitigation:** Implement #4 first. Until then, `FrequencyBasedPromotion` will score all memories identically (access_count=0).

---

## Why Deferred

1. **Depends on #4:** Access count tracking must be implemented first for frequency-based promotion to be meaningful.
2. **Phase 6 scope:** Wiring the strategy framework would have complicated the Phase 6 implementation. Better to ship the framework (proven by tests) and activate it later.
3. **Not a blocker:** The current salience-only promotion works. Strategy-based promotion is an enhancement.

---

## Implementation Roadmap

1. **Add `promotion_strategy` and `promotion_threshold` to Config** — 1 commit.
2. **Modify `MemoryOrchestrator::end_session()` to use strategy** — 1 commit.
3. **Add tests for different promotion strategies** — 1 commit.
4. **Update documentation** — 1 commit.

**Total effort:** ~4 commits, ~2-3 days of work.

---

## Related Tickets

- **#4: Access Count Tracking** — Prerequisite. Must be implemented first.
- **#3: Decay/Prune Pass** — Complementary. Decay removes low-value memories; promotion adds high-value memories.
