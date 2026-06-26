use crate::error::Result;
use crate::models::{MemoryEntry, MemoryId, MemoryScope};
use async_trait::async_trait;

/// Trait for embedding text into vector space.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed text into a vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Trait for a memory storage tier (Synapse or Cortex).
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory entry.
    async fn store(&self, entry: MemoryEntry) -> Result<()>;

    /// Retrieve memories matching a query, up to a limit.
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>>;

    /// Retrieve memories matching a query and scope, up to a limit.
    async fn retrieve_by_scope(
        &self,
        query: &str,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>>;

    /// Delete a memory by ID.
    async fn delete(&self, id: &MemoryId) -> Result<()>;

    /// List all memories in the store.
    async fn list(&self) -> Result<Vec<MemoryEntry>> {
        // Default implementation: retrieve with a wildcard query and max limit
        // Use a non-empty query to avoid embedder validation errors
        self.retrieve("*", usize::MAX).await
    }

    /// Get the number of memories in the store.
    async fn len(&self) -> Result<usize> {
        // Default implementation: count items in list
        Ok(self.list().await?.len())
    }

    /// Check if the store is empty.
    async fn is_empty(&self) -> Result<bool> {
        // Default implementation: check if len is 0
        Ok(self.len().await? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal in-test store that implements MemoryStore with only required methods.
    /// Deliberately does NOT override list/len/is_empty to test default implementations.
    struct DefaultStore;

    #[async_trait]
    impl MemoryStore for DefaultStore {
        async fn store(&self, _entry: MemoryEntry) -> Result<()> {
            Ok(())
        }

        async fn retrieve(&self, _query: &str, _limit: usize) -> Result<Vec<MemoryEntry>> {
            // Return a fixed Vec of two entries regardless of input
            Ok(vec![
                MemoryEntry::new(MemoryId::new(), "content1".to_string()),
                MemoryEntry::new(MemoryId::new(), "content2".to_string()),
            ])
        }

        async fn retrieve_by_scope(
            &self,
            _query: &str,
            _scope: &MemoryScope,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>> {
            Ok(vec![])
        }

        async fn delete(&self, _id: &MemoryId) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_default_list_delegates_to_retrieve() {
        let store = DefaultStore;
        let result = store.list().await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_default_len_counts_list() {
        let store = DefaultStore;
        let len = store.len().await.unwrap();
        assert_eq!(len, 2);
    }

    #[tokio::test]
    async fn test_default_is_empty_false_when_populated() {
        let store = DefaultStore;
        let is_empty = store.is_empty().await.unwrap();
        assert_eq!(is_empty, false);
    }
}
