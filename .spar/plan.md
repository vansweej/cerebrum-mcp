# Feature: Memory-System Concepts Glossary + Deferred-Refinements Docs

## Phase 1: Concepts glossary

Commit message: docs: add LLM-memory concepts glossary with maturity note

### Step 1: Create docs/concepts/llm-memory.md

Create a new file `docs/concepts/llm-memory.md` in the cerebrum-mcp repo. It is a
learning-oriented glossary of the memory concepts this project implements. Write
these sections, each a short prose explanation grounded in THIS codebase:

1. Why LLMs need external memory (stateless context windows; recall across sessions).
2. Embeddings — text mapped to vectors; nomic-embed-text produces 384-dim vectors
   (cite crates/cerebrum-core/src/fastembed_embedder.rs).
3. Relevance vs. salience — relevance = cosine similarity to the query; salience =
   caller-supplied importance (0.0-1.0). Blended ranking is 0.7*similarity +
   0.3*salience (cite lancedb_cortex.rs retrieve()).
4. Two-tier architecture — Synapse (short-term, in-memory HashMap, session-scoped,
   synapse.rs) vs. Cortex (long-term, persistent LanceDB, lancedb_cortex.rs).
5. Memory lifecycle — remember -> recall -> promote (end_session) -> (future) decay.
6. Scoping — global / user:<id> / agent:<id> / session:<id>; currently all memories
   land in global scope.
7. Retrieve-then-rerank — how Cortex retrieval works today (brute-force full scan,
   cosine per row, blend, sort, take(limit)) — O(N).
8. ANN indexes — what LanceDB nearest_to() pushdown would enable (contrast with
   athenaeum-mcp store.rs which uses nearest_to().distance_type(Cosine).limit(k)).
9. Write path — merge_insert keyed on id (atomic upsert; no duplicate rows).
10. Evaluation — note athenaeum's relevance-eval is qualitative human-grading
    (Green/Yellow/Red), not automated Recall@k/MRR/nDCG.
11. Operational concerns — LanceDB on-disk store, cwd-relative db_path.
12. Quick-reference table mapping each concept to its source file.

End with a MATURITY NOTE section stating explicitly: the PromotionStrategy
framework (promotion.rs) and DecayStrategy framework (decay.rs) are fully defined
and unit-tested but NOT invoked at runtime. end_session uses an inline
`salience >= threshold` check at orchestrator.rs:361, not the strategy framework.
No decay/prune path exists. access_count is read by FrequencyBasedPromotion
(promotion.rs:60-66) and AccessBasedDecay (decay.rs:97-103) but never incremented
by the runtime — only test helpers set it. Keep the tone factual, not aspirational.

---

## Phase 2: Deferred-refinements upgrade tickets

Commit message: docs: add deferred-refinements upgrade tickets with dependency index

### Step 1: Create docs/upgrades/README.md

Create `docs/upgrades/README.md` as an index of the five deferred-refinement
tickets below. Include a short intro paragraph, a table (Ticket | File | Depends on
| Status = Deferred), and a recommended sequencing paragraph. Dependency facts:
#4 (access-count) is a shared precursor for #2 and #3; #5 (shared crate) depends on
#1 (cortex nearest_to). Cross-link every ticket file by relative path.

### Step 2: Create docs/upgrades/cortex-vector-search.md

Create ticket #1. Structure every ticket file with these headings: Current State
(with verified file:line), Proposed Change, Design Notes, Acceptance Criteria,
Risks, Why Deferred. Content for #1: Cortex retrieve() today calls
table.query().execute() pulling all rows into Rust, computes cosine per row, blends
salience, sorts, take(limit) — O(N) (cite lancedb_cortex.rs retrieve()). Proposed:
retrieve-then-rerank using LanceDB nearest_to().distance_type(Cosine).limit(k) with
an approximation cap (fetch k*multiplier candidates via ANN, then exact blend
rerank in Rust), mirroring athenaeum-mcp store.rs search(). Acceptance: bounded
memory, index-ready, behavioural parity within the candidate cap.

### Step 3: Create docs/upgrades/access-count-tracking.md

Create ticket #4. Current State: FrequencyBasedPromotion::score reads
metadata["access_count"] (promotion.rs:60-66) and AccessBasedDecay::score reads it
(decay.rs:97-103), but nothing in the runtime increments it — both strategies are
inert. Proposed: increment access_count in metadata on each recall() hit in the
orchestrator, persisted via merge_insert. Note this is the shared precursor that
unblocks tickets #2 and #3. Acceptance: recall increments and persists the count;
existing tests updated.

### Step 4: Create docs/upgrades/wire-promotion-strategy.md

Create ticket #2. Current State: end_session promotion uses an inline
`if entry.salience >= auto_promote_threshold` check at orchestrator.rs:361, NOT the
PromotionStrategy framework. The framework (promotion.rs) defines Frequency/Recency/
Importance/Hybrid impls, all unit-tested, but is never called at runtime. Proposed:
replace the inline check with a configurable PromotionStrategy (default Hybrid).
Depends on #4 for FrequencyBasedPromotion to be meaningful. Acceptance: end_session
delegates to a strategy; inline check removed; behaviour covered by tests.

### Step 5: Create docs/upgrades/decay-prune-pass.md

Create ticket #3. Current State: DecayStrategy (TimeBasedDecay, AccessBasedDecay,
RelevanceBasedDecay, HybridDecay in decay.rs) is defined and unit-tested but nothing
in the runtime invokes it; no purge/demote path exists. Proposed: add a periodic or
end_session decay/prune pass that scores memories via a DecayStrategy and
demotes/deletes below a threshold. Depends on #4 for access-based decay. Acceptance:
a runtime path invokes decay and removes/demotes low-score memories; covered by
tests. Risks: irreversible deletion — gate behind a threshold and log provenance.

### Step 6: Create docs/upgrades/shared-lancedb-crate.md

Create ticket #5. Current State: cerebrum-mcp lancedb_cortex.rs and athenaeum-mcp
store.rs independently implement near-identical LanceDB access (connect, schema,
merge_insert, batch<->record). Proposed: extract a shared generic VectorStore<R>
crate reused by both projects, and add an ANN index. Depends on #1 (the nearest_to
retrieval path must land first so the shared abstraction covers it). Acceptance:
both projects consume the shared crate; no behavioural change. Why Deferred:
cross-repo coordination; gate on #1 stabilising.

---

## Phase 3: Roadmap corrections

Commit message: docs: correct roadmap maturity claims and link deferred refinements

### Step 1: Correct two overstated lines in docs/roadmap.md

In `docs/roadmap.md`, make two exact-string replacements.

Replace line 3, currently:
"This roadmap covers work **beyond Phase 5**. Phases 1-5 deliver a fully-featured
two-tier memory system with advanced features (promotion, decay, summarization,
scope filtering). Phase 6 focuses on production hardening and persistence."

with a version stating that Phases 1-5 deliver the two-tier system with scope
filtering and LanceDB persistence, and that promotion is currently an inline
salience-threshold check while the promotion/decay/summarization STRATEGY
frameworks are implemented and unit-tested but NOT yet wired into the runtime (see
docs/upgrades/ and the maturity note in docs/concepts/llm-memory.md).

Replace line 95, currently:
"2. **Every durable memory carries provenance.** Automatic decisions must be
auditable and reversible."

with a version that keeps the provenance principle but marks the
auditable-and-reversible automatic-decision goal as aspirational / not yet met,
pending the deferred promotion and decay tickets.

### Step 2: Append a Deferred Refinements section to docs/roadmap.md

At the end of `docs/roadmap.md`, append a new "## Deferred Refinements" section
that briefly introduces the five tickets and links each by relative path into
docs/upgrades/ (cortex-vector-search, access-count-tracking, wire-promotion-strategy,
decay-prune-pass, shared-lancedb-crate), noting the #4->#2/#3 and #1->#5 dependencies.
