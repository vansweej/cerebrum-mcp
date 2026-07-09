# Cerebrum Roadmap

This roadmap covers work **beyond Phase 5**. Phases 1-5 deliver a fully-featured two-tier memory system with scope filtering and LanceDB persistence. Phase 6 focuses on production hardening and real embeddings. **Note:** The PromotionStrategy and DecayStrategy frameworks are fully implemented and unit-tested but not yet wired into the runtime — see [docs/upgrades/](docs/upgrades/) for deferred refinements that will activate these frameworks and add automatic memory cleanup.

---

## Phase 6: Production Hardening & LanceDB Integration

**Status:** Planned (Ready to implement)

Phase 6 transitions Cerebrum from a prototype with in-memory storage to a production-ready system with persistent vector database storage.

### Deliverables

- **LanceDB Integration** — Replace in-memory Cortex with persistent vector database
- **FastEmbed Integration** — Real embeddings with BGE-small model (384-dim)
- **Embedding Migration** — Tooling to handle embedding model changes
- **Observability** — Structured logging, metrics, and tracing
- **Error Handling & Resilience** — Retry logic, circuit breaker, graceful degradation
- **Orchestrator Updates** — Support configurable backends
- **Integration Tests** — 20 comprehensive tests for Phase 6 features
- **Documentation** — Production deployment guide, LanceDB setup, observability guide

### Architecture Decisions

See `docs/adr-phase-6.md` for detailed Architecture Decision Records:
- **ADR-001:** LanceDB for Persistent Cortex Storage
- **ADR-002:** FastEmbed for Real Embeddings
- **ADR-003:** Embedding Migration Strategy
- **ADR-004:** Error Handling and Resilience

### Implementation Plan

See `docs/phase-6-plan.md` for detailed 8-step implementation plan:
1. LanceDB Integration Foundation
2. FastEmbed Integration
3. Embedding Migration Tooling
4. Observability & Logging
5. Error Handling & Resilience
6. Orchestrator Updates
7. Integration Tests
8. Documentation & Release

### Success Criteria

- ✅ LanceDB Cortex fully functional and tested
- ✅ FastEmbed embeddings working with BGE-small model
- ✅ Embedding migration tooling available and tested
- ✅ Comprehensive observability in place
- ✅ Error handling and resilience patterns implemented
- ✅ All 80 new tests passing
- ✅ 90%+ code coverage maintained
- ✅ Zero clippy warnings
- ✅ Production deployment guide complete
- ✅ Project at 100% completion (5 of 5 phases)

---

## Phase 7+: Advanced Features & Scaling

**Status:** Future (Post-Phase 6)

After Phase 6 completes production hardening, future phases may include:

### Intelligence Layer

- **Multi-model embedding support** — Support multiple embedding models simultaneously
- **Distributed deployment** — Multiple servers with shared Cortex
- **Advanced analytics** — Memory usage patterns, agent insights
- **Agent-specific optimization** — Per-agent memory tuning
- **Conflict resolution** — Handle contradictory memories
- **Semantic deduplication** — Identify and merge similar memories

### Operational Features

- **Docker containerization** — Container images for deployment
- **Kubernetes manifests** — Helm charts for orchestration
- **Monitoring and alerting** — Prometheus metrics, alerting rules
- **Performance tuning** — Benchmarking, optimization
- **Security hardening** — Authentication, encryption, audit logging

### Research Directions

- **Temporal reasoning** — Time-aware memory retrieval
- **Causal inference** — Understanding memory relationships
- **Active learning** — System-initiated clarification questions
- **Memory compression** — Lossless compression of long-term memories
- **Cross-agent learning** — Shared insights between agents

---

## Guiding Principles (All Phases)

1. **The agent must not see the seams.** Tiering stays an implementation detail; the tool surface should remain stable.
2. **Every durable memory carries provenance.** Automatic decisions must be auditable and reversible. *(Note: This is an aspirational goal for future phases. Currently, promotion is manual (salience-based) and decay does not occur. See [docs/upgrades/](docs/upgrades/) for planned refinements.)*
3. **Backward compatibility.** New phases must not break existing MCP tools or agent integrations.
4. **Production-ready quality.** All code must meet 90%+ coverage, zero clippy warnings, comprehensive tests.
5. **Clear documentation.** Every phase includes architecture docs, ADRs, and deployment guides.

---

## Deferred Refinements

Five refinements were deferred from Phase 6 to prioritize shipping LanceDB persistence and real embeddings. All are architecturally sound and proven by tests, but require additional coordination or depend on stabilizing other features first.

See [docs/upgrades/README.md](docs/upgrades/README.md) for the full index, dependency graph, and recommended sequencing.

### Refinement #1: Cortex Vector Search — Retrieve-Then-Rerank with ANN

**File:** [docs/upgrades/cortex-vector-search.md](docs/upgrades/cortex-vector-search.md)

Optimize Cortex retrieval from O(N) brute-force to O(log N) approximate nearest neighbors. Fetch k*multiplier candidates via LanceDB ANN index, then exact blend-rerank in Rust to preserve the 0.7*similarity + 0.3*salience blending formula.

### Refinement #2: Wire Promotion Strategy — Activate the Framework

**File:** [docs/upgrades/wire-promotion-strategy.md](docs/upgrades/wire-promotion-strategy.md)

Replace the inline `if entry.salience >= auto_promote_threshold` check in `end_session()` with a configurable `PromotionStrategy` (default Hybrid). Requires Refinement #4 so `FrequencyBasedPromotion` can read meaningful access counts.

### Refinement #3: Decay/Prune Pass — Automatic Memory Cleanup

**File:** [docs/upgrades/decay-prune-pass.md](docs/upgrades/decay-prune-pass.md)

Add a periodic or end-session decay pass that scores memories via a `DecayStrategy` and demotes/deletes low-scoring memories. Prevents unbounded growth. Requires Refinement #4 for `AccessBasedDecay` to work.

### Refinement #4: Access Count Tracking — Shared Precursor

**File:** [docs/upgrades/access-count-tracking.md](docs/upgrades/access-count-tracking.md)

Increment `access_count` in memory metadata on each `recall()` hit, persisted via merge_insert. Unblocks both Refinements #2 and #3 by providing meaningful data for frequency-based and access-based strategies.

### Refinement #5: Shared LanceDB Crate — Cross-Project Reuse

**File:** [docs/upgrades/shared-lancedb-crate.md](docs/upgrades/shared-lancedb-crate.md)

Extract a generic `VectorStore<R>` crate from Cerebrum and Athenaeum's near-identical LanceDB implementations. Requires Refinement #1 to stabilize the retrieval path (retrieve-then-rerank with ANN).

### Dependency Graph

```
Refinement #1 (Cortex Vector Search)
  └─> Refinement #5 (Shared LanceDB Crate)

Refinement #4 (Access Count Tracking)
  ├─> Refinement #2 (Wire Promotion Strategy)
  └─> Refinement #3 (Decay/Prune Pass)
```

### Recommended Sequencing

**Phase A (Independent):**
- Refinement #1 — Optimize retrieval with ANN indexes.
- Refinement #4 — Increment access count on recall.

**Phase B (Depends on Phase A):**
- Refinement #2 — Activate PromotionStrategy (requires #4).
- Refinement #3 — Activate DecayStrategy (requires #4).

**Phase C (Depends on Phase A):**
- Refinement #5 — Extract shared crate (requires #1).

For more details, see [docs/concepts/llm-memory.md](docs/concepts/llm-memory.md) for the full glossary and maturity note explaining why these frameworks are currently dormant.
