use crate::error::Result;
use crate::models::{MemoryEntry, MemoryId};
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

    /// Delete a memory by ID.
    async fn delete(&self, id: &MemoryId) -> Result<()>;
}
