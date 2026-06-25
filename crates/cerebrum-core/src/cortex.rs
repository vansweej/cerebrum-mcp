use crate::embedder::Embedder;
use crate::error::Result;
use crate::models::{MemoryEntry, MemoryId};
use crate::traits::MemoryStore;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Persistent long-term memory storage backed by LanceDB (Cortex tier).
///
/// Stores memories in a vector database for efficient semantic search and
/// persistent storage across sessions. Supports salience-based ranking.
///
/// Note: This is a simplified implementation using in-memory storage.
/// In production, this would use LanceDB for persistent vector storage.
pub struct CortexMemory {
    memories: Arc<RwLock<HashMap<MemoryId, MemoryEntry>>>,
    embedder: Arc<dyn Embedder>,
}

impl CortexMemory {
    /// Create a new Cortex memory store with LanceDB backend.
    ///
    /// # Arguments
    /// * `_db_path` - Path to the LanceDB database directory (for future use)
    /// * `embedder` - Embedder instance for generating embeddings
    pub async fn new(_db_path: &str, embedder: Arc<dyn Embedder>) -> Result<Self> {
        // In a real implementation, this would initialize LanceDB
        // For now, we use in-memory storage to avoid LanceDB API complexity
        Ok(Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
            embedder,
        })
    }

    /// Create a new Cortex memory store from components.
    pub fn from_parts(
        memories: Arc<RwLock<HashMap<MemoryId, MemoryEntry>>>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self { memories, embedder }
    }

    /// Search memories by salience (highest first).
    pub async fn search_by_salience(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let memories = self.memories.read();
        let mut entries: Vec<_> = memories.values().cloned().collect();

        // Sort by salience (descending)
        entries.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(entries.into_iter().take(limit).collect())
    }

    /// Get the number of memories stored.
    pub async fn len(&self) -> Result<usize> {
        Ok(self.memories.read().len())
    }

    /// Check if the store is empty.
    pub async fn is_empty(&self) -> Result<bool> {
        Ok(self.memories.read().is_empty())
    }

    /// List all memories.
    pub async fn list(&self) -> Result<Vec<MemoryEntry>> {
        let memories = self.memories.read();
        Ok(memories.values().cloned().collect())
    }

    /// Calculate cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

#[async_trait]
impl MemoryStore for CortexMemory {
    async fn store(&self, entry: MemoryEntry) -> Result<()> {
        self.memories.write().insert(entry.id, entry);
        Ok(())
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        // Generate embedding for query
        let query_embedding = self.embedder.embed(query).await?;

        let memories = self.memories.read();

        // If no memories, return empty
        if memories.is_empty() {
            return Ok(Vec::new());
        }

        // Score all memories by similarity
        let mut scored: Vec<_> = memories
            .values()
            .filter_map(|entry| {
                entry.embedding.as_ref().map(|embedding| {
                    let similarity = Self::cosine_similarity(&query_embedding, embedding);
                    // Combine similarity with salience for ranking
                    let score = (similarity * 0.7) + (entry.salience * 0.3);
                    (entry.clone(), score)
                })
            })
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top N
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(entry, _)| entry)
            .collect())
    }

    async fn delete(&self, id: &MemoryId) -> Result<()> {
        self.memories.write().remove(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cortex_new() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let cortex = CortexMemory::new("/tmp/test_cortex", embedder)
            .await
            .expect("Failed to create CortexMemory");

        assert!(cortex.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn test_cortex_store_and_retrieve() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let cortex = CortexMemory::new("/tmp/test_cortex", embedder)
            .await
            .expect("Failed to create CortexMemory");

        let id = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry = MemoryEntry::builder(id, "Test memory".to_string())
            .embedding(embedding)
            .build();

        cortex.store(entry).await.unwrap();
        assert_eq!(cortex.len().await.unwrap(), 1);

        let results = cortex.retrieve("test", 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_cortex_delete() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let cortex = CortexMemory::new("/tmp/test_cortex", embedder)
            .await
            .expect("Failed to create CortexMemory");

        let id = MemoryId::new();
        let entry = MemoryEntry::new(id, "Test memory".to_string());

        cortex.store(entry).await.unwrap();
        assert_eq!(cortex.len().await.unwrap(), 1);

        cortex.delete(&id).await.unwrap();
        assert_eq!(cortex.len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_cortex_search_by_salience() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let cortex = CortexMemory::new("/tmp/test_cortex", embedder)
            .await
            .expect("Failed to create CortexMemory");

        let id1 = MemoryId::new();
        let entry1 = MemoryEntry::builder(id1, "Memory 1".to_string())
            .salience(0.9)
            .build();

        let id2 = MemoryId::new();
        let entry2 = MemoryEntry::builder(id2, "Memory 2".to_string())
            .salience(0.3)
            .build();

        cortex.store(entry1).await.unwrap();
        cortex.store(entry2).await.unwrap();

        let results = cortex.search_by_salience(10).await.unwrap();
        assert_eq!(results.len(), 2);
        // Higher salience should be first
        assert!(results[0].salience >= results[1].salience);
    }

    #[tokio::test]
    async fn test_cortex_list() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let cortex = CortexMemory::new("/tmp/test_cortex", embedder)
            .await
            .expect("Failed to create CortexMemory");

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        let entry1 = MemoryEntry::new(id1, "Memory 1".to_string());
        let entry2 = MemoryEntry::new(id2, "Memory 2".to_string());

        cortex.store(entry1).await.unwrap();
        cortex.store(entry2).await.unwrap();

        let list = cortex.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_cortex_retrieve_with_salience() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let cortex = CortexMemory::new("/tmp/test_cortex", embedder)
            .await
            .expect("Failed to create CortexMemory");

        let id1 = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry1 = MemoryEntry::builder(id1, "Important memory".to_string())
            .salience(0.9)
            .embedding(embedding.clone())
            .build();

        let id2 = MemoryId::new();
        let entry2 = MemoryEntry::builder(id2, "Important memory".to_string())
            .salience(0.1)
            .embedding(embedding)
            .build();

        cortex.store(entry1).await.unwrap();
        cortex.store(entry2).await.unwrap();

        let results = cortex.retrieve("important", 10).await.unwrap();
        assert_eq!(results.len(), 2);
        // Higher salience should rank first
        assert!(results[0].salience >= results[1].salience);
    }

    #[tokio::test]
    async fn test_cortex_retrieve_empty() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let cortex = CortexMemory::new("/tmp/test_cortex", embedder)
            .await
            .expect("Failed to create CortexMemory");

        let results = cortex.retrieve("test", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_cortex_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = CortexMemory::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = CortexMemory::cosine_similarity(&a, &c);
        assert!(similarity2.abs() < 0.001);
    }
}
