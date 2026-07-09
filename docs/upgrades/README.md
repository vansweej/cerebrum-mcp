# Deferred Refinements: Upgrade Tickets

This directory contains five deferred-refinement tickets that extend Cerebrum's memory system with advanced features. Each ticket is architecturally sound and ready to implement, but was deferred to prioritize shipping Phase 6 (LanceDB persistence and real embeddings).

---

## Overview

| Ticket | File | Depends On | Status |
|--------|------|-----------|--------|
| #1: Cortex Vector Search | [cortex-vector-search.md](cortex-vector-search.md) | — | Deferred |
| #2: Wire Promotion Strategy | [wire-promotion-strategy.md](wire-promotion-strategy.md) | #4 | Deferred |
| #3: Decay/Prune Pass | [decay-prune-pass.md](decay-prune-pass.md) | #4 | Deferred |
| #4: Access Count Tracking | [access-count-tracking.md](access-count-tracking.md) | — | Deferred |
| #5: Shared LanceDB Crate | [shared-lancedb-crate.md](shared-lancedb-crate.md) | #1 | Deferred |

---

## Dependency Graph

```
#1 (Cortex Vector Search)
  └─> #5 (Shared LanceDB Crate)

#4 (Access Count Tracking)
  ├─> #2 (Wire Promotion Strategy)
  └─> #3 (Decay/Prune Pass)
```

---

## Recommended Sequencing

### Phase A: Foundation (Independent)
- **#1: Cortex Vector Search** — Optimize retrieval with ANN indexes. No dependencies; can ship independently.
- **#4: Access Count Tracking** — Increment on recall. Unblocks #2 and #3; can ship independently.

### Phase B: Promotion & Decay (Depends on Phase A)
- **#2: Wire Promotion Strategy** — Replace inline check with configurable strategy. Requires #4 for `FrequencyBasedPromotion` to be meaningful.
- **#3: Decay/Prune Pass** — Add periodic decay logic. Requires #4 for `AccessBasedDecay` to be meaningful.

### Phase C: Cross-Project Reuse (Depends on Phase A)
- **#5: Shared LanceDB Crate** — Extract generic `VectorStore<R>`. Requires #1 to stabilize the retrieval path (retrieve-then-rerank).

---

## Why These Were Deferred

1. **Scope creep:** Phase 6 (LanceDB + FastEmbed) was already substantial. Adding promotion/decay/ANN would have delayed shipping.
2. **Architectural soundness:** All five are well-designed and proven by tests. Deferring does not compromise correctness.
3. **Incremental value:** The system is production-ready without them. They are optimizations and advanced features, not blockers.
4. **Cross-repo coordination:** #5 requires alignment with Athenaeum. Better to defer than rush.

---

## Quick Links

- **Concepts & Maturity:** See [docs/concepts/llm-memory.md](../concepts/llm-memory.md) for the full glossary and maturity note explaining why these frameworks are dormant.
- **Roadmap:** See [docs/roadmap.md](../roadmap.md) for the full project roadmap and how these refinements fit into future phases.

---

## Ticket Descriptions

### #1: Cortex Vector Search — Retrieve-Then-Rerank with ANN

**File:** [cortex-vector-search.md](cortex-vector-search.md)

Optimize Cortex retrieval from O(N) brute-force to O(log N) approximate nearest neighbors. Fetch k*multiplier candidates via LanceDB ANN index, then exact blend-rerank in Rust to preserve the 0.7*similarity + 0.3*salience blending formula.

### #2: Wire Promotion Strategy — Activate the Framework

**File:** [wire-promotion-strategy.md](wire-promotion-strategy.md)

Replace the inline `if entry.salience >= auto_promote_threshold` check in `end_session()` with a configurable `PromotionStrategy` (default Hybrid). Requires #4 so `FrequencyBasedPromotion` can read meaningful access counts.

### #3: Decay/Prune Pass — Automatic Memory Cleanup

**File:** [decay-prune-pass.md](decay-prune-pass.md)

Add a periodic or end-session decay pass that scores memories via a `DecayStrategy` and demotes/deletes low-scoring memories. Prevents unbounded growth. Requires #4 for `AccessBasedDecay` to work.

### #4: Access Count Tracking — Shared Precursor

**File:** [access-count-tracking.md](access-count-tracking.md)

Increment `access_count` in memory metadata on each `recall()` hit, persisted via merge_insert. Unblocks both #2 and #3 by providing meaningful data for frequency-based and access-based strategies.

### #5: Shared LanceDB Crate — Cross-Project Reuse

**File:** [shared-lancedb-crate.md](shared-lancedb-crate.md)

Extract a generic `VectorStore<R>` crate from Cerebrum and Athenaeum's near-identical LanceDB implementations. Requires #1 to stabilize the retrieval path (retrieve-then-rerank with ANN).

---

## Implementation Notes

- Each ticket includes **Current State** (with verified file:line citations), **Proposed Change**, **Design Notes**, **Acceptance Criteria**, **Risks**, and **Why Deferred**.
- All citations are grounded in the actual codebase as of the main branch.
- Tickets are designed to be implemented in the recommended sequence, but #1 and #4 can ship independently.
- See the individual ticket files for detailed design and implementation guidance.
