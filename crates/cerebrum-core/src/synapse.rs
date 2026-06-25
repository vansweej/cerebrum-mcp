use crate::error::Result;
use crate::models::{MemoryEntry, MemoryId};
use crate::embedder::Embedder;
use crate::traits::MemoryStore;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// In-memory short-term memory storage (Synapse tier).
///
/// Stores memories in a thread-safe HashMap. Memories are volatile and cleared
/// when the session ends. Supports semantic search using embeddings.
#[derive(Clone)]
pub struct SynapseMemory {
    memories: Arc<RwLock<HashMap<MemoryId, MemoryEntry>>>,
}

impl SynapseMemory {
    /// Create a new empty Synapse memory store.
    pub fn new() -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the number of memories currently stored.
    pub fn len(&self) -> usize {
        self.memories.read().len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.memories.read().is_empty()
    }

    /// Clear all memories (typically called at session end).
    pub async fn clear(&self) -> Result<()> {
        self.memories.write().clear();
        Ok(())
    }

    /// List all memories (for debugging/inspection).
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

impl Default for SynapseMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for SynapseMemory {
    async fn store(&self, entry: MemoryEntry) -> Result<()> {
        self.memories.write().insert(entry.id, entry);
        Ok(())
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        // Generate embedding for query (before acquiring lock to avoid Send issues)
        let embedder = crate::embedder::MockEmbedder::new();
        let query_embedding: Vec<f32> = embedder.embed(query).await?;

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
        Ok(scored.into_iter().take(limit).map(|(entry, _)| entry).collect())
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
    async fn test_synapse_new() {
        let synapse = SynapseMemory::new();
        assert!(synapse.is_empty());
        assert_eq!(synapse.len(), 0);
    }

    #[tokio::test]
    async fn test_synapse_store_and_retrieve() {
        let synapse = SynapseMemory::new();
        let id = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry = MemoryEntry::builder(id, "Test memory".to_string())
            .embedding(embedding)
            .build();

        synapse.store(entry.clone()).await.unwrap();
        assert_eq!(synapse.len(), 1);

        let results = synapse.retrieve("test", 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_synapse_delete() {
        let synapse = SynapseMemory::new();
        let id = MemoryId::new();
        let entry = MemoryEntry::new(id, "Test memory".to_string());

        synapse.store(entry).await.unwrap();
        assert_eq!(synapse.len(), 1);

        synapse.delete(&id).await.unwrap();
        assert_eq!(synapse.len(), 0);
    }

    #[tokio::test]
    async fn test_synapse_clear() {
        let synapse = SynapseMemory::new();

        for i in 0..5 {
            let id = MemoryId::new();
            let entry = MemoryEntry::new(id, format!("Memory {}", i));
            synapse.store(entry).await.unwrap();
        }

        assert_eq!(synapse.len(), 5);
        synapse.clear().await.unwrap();
        assert_eq!(synapse.len(), 0);
    }

    #[tokio::test]
    async fn test_synapse_list() {
        let synapse = SynapseMemory::new();
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        let entry1 = MemoryEntry::new(id1, "Memory 1".to_string());
        let entry2 = MemoryEntry::new(id2, "Memory 2".to_string());

        synapse.store(entry1).await.unwrap();
        synapse.store(entry2).await.unwrap();

        let list = synapse.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_synapse_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = SynapseMemory::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = SynapseMemory::cosine_similarity(&a, &c);
        assert!(similarity2.abs() < 0.001);
    }

    #[tokio::test]
    async fn test_synapse_retrieve_empty() {
        let synapse = SynapseMemory::new();
        let results = synapse.retrieve("test", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_synapse_retrieve_with_salience() {
        let synapse = SynapseMemory::new();

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

        synapse.store(entry1).await.unwrap();
        synapse.store(entry2).await.unwrap();

        let results = synapse.retrieve("important", 10).await.unwrap();
        assert_eq!(results.len(), 2);
        // Higher salience should rank first
        assert!(results[0].salience >= results[1].salience);
    }
}
