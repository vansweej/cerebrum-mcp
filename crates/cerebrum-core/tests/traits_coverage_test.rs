//! Coverage tests for default trait impl bodies in `traits.rs` and
//! uncovered lines in `embedder.rs`.

use async_trait::async_trait;
use cerebrum_core::embedder::MockEmbedder;
use cerebrum_core::error::Result;
use cerebrum_core::models::{MemoryEntry, MemoryId, MemoryScope, MemoryTier, ScoredMemory};
use cerebrum_core::traits::{Embedder, MemoryStore};

// ── MockEmbedder default() and instance dimension() ─────────────────────────

#[test]
fn mock_embedder_default_constructs() {
    let e = MockEmbedder::default();
    assert_eq!(e.dimension(), 384);
}

// ── traits default methods: retrieve and retrieve_by_scope ───────────────────

/// Minimal MemoryStore implementation that records calls.
struct StubStore {
    entries: Vec<MemoryEntry>,
}

impl StubStore {
    fn new(entries: Vec<MemoryEntry>) -> Self { Self { entries } }
}

#[async_trait]
impl MemoryStore for StubStore {
    async fn store(&self, _entry: MemoryEntry) -> Result<()> { Ok(()) }

    async fn retrieve_scored(
        &self,
        _query_vec: &[f32],
        limit: usize,
        _prefer_project: Option<&str>,
    ) -> Result<Vec<ScoredMemory>> {
        Ok(self.entries.iter().take(limit).map(|e| ScoredMemory {
            entry: e.clone(),
            score: 1.0,
        }).collect())
    }

    async fn retrieve_by_scope_scored(
        &self,
        _query_vec: &[f32],
        _scope: &MemoryScope,
        limit: usize,
        _prefer_project: Option<&str>,
        _exact_scope: bool,
    ) -> Result<Vec<ScoredMemory>> {
        Ok(self.entries.iter().take(limit).map(|e| ScoredMemory {
            entry: e.clone(),
            score: 1.0,
        }).collect())
    }

    async fn delete(&self, _id: &MemoryId) -> Result<()> { Ok(()) }

    async fn list(&self) -> Result<Vec<MemoryEntry>> { Ok(self.entries.clone()) }

    async fn len(&self) -> Result<usize> { Ok(self.entries.len()) }

    async fn is_empty(&self) -> Result<bool> { Ok(self.entries.is_empty()) }
}

#[tokio::test]
async fn traits_retrieve_default_strips_score() {
    let entry = MemoryEntry::builder(MemoryId::new(), "hello".to_string())
        .tier(MemoryTier::Cortex)
        .build();
    let store = StubStore::new(vec![entry.clone()]);

    let results = store.retrieve(&[0.0_f32; 4], 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "hello");
}

#[tokio::test]
async fn traits_retrieve_by_scope_default_strips_score() {
    let entry = MemoryEntry::builder(MemoryId::new(), "world".to_string())
        .tier(MemoryTier::Cortex)
        .build();
    let store = StubStore::new(vec![entry.clone()]);
    let scope = MemoryScope::Global;

    let results = store
        .retrieve_by_scope(&[0.0_f32; 4], &scope, 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "world");
}
