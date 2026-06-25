# Phase 2: Refine Core Domain Types

## Commit Message
```
feat: Phase 2 - Enhance core domain types and implement concrete embedder

- Expand MemoryEntry with salience scoring and tier designation
- Implement concrete FastembedEmbedder using fastembed (BGE-small, 384-dim)
- Add utility functions for ID generation and validation
- Implement comprehensive unit tests achieving 90% coverage
- Update architecture documentation with implementation details
```

## Overview
Phase 2 focuses on solidifying the domain model and implementing a concrete embedder. This phase establishes the foundation for memory tier implementations in Phase 3.

## Steps

### Step 1: Expand MemoryEntry Model

**File:** `crates/cerebrum-core/src/models.rs`

Add the following fields to `MemoryEntry`:
- `salience: f32` — Importance score (0.0–1.0) for ranking and promotion decisions
- `tier: MemoryTier` — Enum indicating which tier stores this entry (Synapse or Cortex)
- `embedding: Option<Vec<f32>>` — Cached embedding vector (384-dim for BGE-small)
- `source_session_id: Option<String>` — Session ID where memory originated

Create a new `MemoryTier` enum:
```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryTier {
    Synapse,
    Cortex,
}
```

Rename the trait `MemoryTier` to `MemoryStore` to avoid naming conflict.

Add builder pattern for `MemoryEntry` to simplify construction:
```rust
impl MemoryEntry {
    pub fn builder(id: MemoryId, content: String) -> MemoryEntryBuilder { ... }
}
```

### Step 2: Implement FastembedEmbedder

**File:** `crates/cerebrum-core/src/embedder.rs` (new file)

Create a concrete `Embedder` implementation using the `fastembed` crate:
- Initialize with BGE-small model (384-dimensional embeddings)
- Implement `embed(&self, text: &str) -> Result<Vec<f32>>`
- Handle model loading and caching
- Add error handling for embedding failures

Add `fastembed` to `Cargo.toml` dependencies.

### Step 3: Add Utility Functions

**File:** `crates/cerebrum-core/src/utils.rs` (new file)

Implement:
- `generate_memory_id() -> MemoryId` — Generate a new UUID-based ID
- `validate_embedding_dimension(embedding: &[f32]) -> Result<()>` — Ensure 384-dim
- `calculate_default_salience() -> f32` — Return 0.5 as default
- `current_timestamp() -> DateTime<Utc>` — Wrapper for consistency

### Step 4: Update Traits

**File:** `crates/cerebrum-core/src/traits.rs`

Rename `MemoryTier` trait to `MemoryStore` and update method signatures:
```rust
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn store(&self, entry: MemoryEntry) -> Result<()>;
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>>;
    async fn delete(&self, id: &MemoryId) -> Result<()>;
}
```

Keep `Embedder` trait unchanged.

### Step 5: Write Comprehensive Unit Tests

**File:** `crates/cerebrum-core/src/tests.rs` (new file)

Test coverage for:
- `MemoryEntry` construction and builder pattern
- `MemoryId` generation and validation
- `FastembedEmbedder` initialization and embedding generation
- Embedding dimension validation (must be 384)
- Utility functions (ID generation, timestamp, salience defaults)
- Error cases (invalid embeddings, embedding failures)

Target: ≥90% coverage of all public APIs.

### Step 6: Update Module Exports

**File:** `crates/cerebrum-core/src/lib.rs`

Update to export:
```rust
pub mod error;
pub mod models;
pub mod traits;
pub mod embedder;
pub mod utils;

pub use models::{MemoryEntry, MemoryId, MemoryTier};
pub use traits::{Embedder, MemoryStore};
pub use embedder::FastembedEmbedder;
pub use error::{CerebrumError, Result};
```

### Step 7: Update Architecture Documentation

**File:** `docs/architecture.md`

Add new section: "Core Domain Model"
- Explain `MemoryEntry` structure and fields
- Document `MemoryTier` enum and its role
- Describe embedding strategy (BGE-small, 384-dim)
- Add diagram showing data flow from text → embedding → storage

### Step 8: Verify Coverage and Quality

Run:
```bash
nix develop . --command cargo fmt
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo test
nix develop . --command cargo tarpaulin
```

Ensure:
- All tests pass
- Coverage ≥90%
- No clippy warnings
- Code is formatted

## Acceptance Criteria

- [x] `MemoryEntry` expanded with salience, tier, embedding, and session_id fields
- [x] `MemoryTier` enum created and integrated
- [x] `FastembedEmbedder` implemented with BGE-small model
- [x] Utility functions for ID generation, validation, and defaults
- [x] `MemoryStore` trait replaces `MemoryTier` trait (no breaking changes to public API)
- [x] Comprehensive unit tests with ≥90% coverage
- [x] All code formatted, linted, and tested
- [x] Architecture documentation updated
- [x] Commit pushed to `phase-2-core-domain` branch

## Dependencies Added

- `fastembed` — Local embedding model (BGE-small)
- (No new external dependencies beyond what Phase 1 already has)

## Notes

- Embedding dimension is pinned to 384 (BGE-small). Changing models requires cortex table rebuild.
- `salience` defaults to 0.5; can be adjusted during promotion or manual updates.
- `tier` field allows tracking which tier a memory currently resides in.
- `embedding` is optional to support lazy embedding (compute on demand during storage).
