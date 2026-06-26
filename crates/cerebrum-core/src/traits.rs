use crate::error::Result;
use crate::models::{MemoryEntry, MemoryId, MemoryScope};
use async_trait::async_trait;

/// Trait for embedding text into vector space.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed text into a vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Return the dimensionality of vectors produced by this embedder.
    fn dimension(&self) -> usize;
}

/// Trait for a memory storage tier (Synapse or Cortex).
///
/// Stores operate on query **vectors**, never raw text. The orchestrator owns
/// the [`Embedder`] and embeds the query exactly once before calling
/// `retrieve` / `retrieve_by_scope`, passing the resulting vector to both
/// tiers. This keeps embedding concerns out of the storage layer entirely.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory entry.
    async fn store(&self, entry: MemoryEntry) -> Result<()>;

    /// Retrieve memories matching a query vector, up to a limit.
    async fn retrieve(&self, query_vec: &[f32], limit: usize) -> Result<Vec<MemoryEntry>>;

    /// Retrieve memories matching a query vector and scope, up to a limit.
    async fn retrieve_by_scope(
        &self,
        query_vec: &[f32],
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>>;

    /// Delete a memory by ID.
    async fn delete(&self, id: &MemoryId) -> Result<()>;

    /// List all memories in the store.
    async fn list(&self) -> Result<Vec<MemoryEntry>>;

    /// Get the number of memories in the store.
    async fn len(&self) -> Result<usize>;

    /// Check if the store is empty.
    async fn is_empty(&self) -> Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal in-test store that implements MemoryStore with all required methods.
    /// `retrieve` ignores its query vector and returns a fixed Vec of two entries.
    struct DefaultStore;

    #[async_trait]
    impl MemoryStore for DefaultStore {
        async fn store(&self, _entry: MemoryEntry) -> Result<()> {
            Ok(())
        }

        async fn retrieve(&self, _query_vec: &[f32], _limit: usize) -> Result<Vec<MemoryEntry>> {
            // Return a fixed Vec of two entries regardless of input
            Ok(vec![
                MemoryEntry::new(MemoryId::new(), "content1".to_string()),
                MemoryEntry::new(MemoryId::new(), "content2".to_string()),
            ])
        }

        async fn retrieve_by_scope(
            &self,
            _query_vec: &[f32],
            _scope: &MemoryScope,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>> {
            Ok(vec![])
        }

        async fn delete(&self, _id: &MemoryId) -> Result<()> {
            Ok(())
        }

        async fn list(&self) -> Result<Vec<MemoryEntry>> {
            Ok(vec![
                MemoryEntry::new(MemoryId::new(), "content1".to_string()),
                MemoryEntry::new(MemoryId::new(), "content2".to_string()),
            ])
        }

        async fn len(&self) -> Result<usize> {
            Ok(self.list().await?.len())
        }

        async fn is_empty(&self) -> Result<bool> {
            Ok(self.len().await? == 0)
        }
    }

    #[tokio::test]
    async fn test_list_returns_entries() {
        let store = DefaultStore;
        let result = store.list().await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_len_counts_list() {
        let store = DefaultStore;
        let len = store.len().await.unwrap();
        assert_eq!(len, 2);
    }

    #[tokio::test]
    async fn test_is_empty_false_when_populated() {
        let store = DefaultStore;
        let is_empty = store.is_empty().await.unwrap();
        assert_eq!(is_empty, false);
    }
}
