use crate::error::Result;
use crate::models::{MemoryEntry, MemoryId, MemoryScope};
use crate::traits::MemoryStore;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// In-memory short-term memory storage (Synapse tier).
///
/// Stores memories in a thread-safe HashMap. Memories are volatile and cleared
/// when the session ends. Supports semantic search using embeddings.
///
/// # Vector-Based Operation
/// The Synapse store operates on query **vectors**, never raw text. The orchestrator owns
/// the [`Embedder`] and embeds the query exactly once before calling
/// `retrieve` / `retrieve_by_scope`, passing the resulting vector to the store.
/// This keeps embedding concerns out of the storage layer entirely.
///
/// # Similarity & Salience Blending
/// Memories are ranked using a blended score:
/// ```text
/// score = 0.7 * cosine_similarity(query_vec, memory_embedding) + 0.3 * memory_salience
/// ```
/// This formula balances semantic relevance (70%) with importance/salience (30%).
/// High-salience memories are boosted in rankings even if they're not perfect matches.
///
/// # Thread Safety
/// Uses `Arc<RwLock<HashMap>>` for thread-safe concurrent access.
/// Multiple threads can read simultaneously; writes are serialized.
///
/// # Session Scoping
/// Memories stored in Synapse are session-scoped and cleared when the session ends.
/// Use `memorize()` to promote high-salience memories to Cortex for persistence.
#[derive(Clone, Default)]
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

#[async_trait]
impl MemoryStore for SynapseMemory {
    async fn store(&self, entry: MemoryEntry) -> Result<()> {
        self.memories.write().insert(entry.id, entry);
        Ok(())
    }

    async fn retrieve(&self, query_vec: &[f32], limit: usize) -> Result<Vec<MemoryEntry>> {
        let memories = self.memories.read();

        // If no memories, return empty
        if memories.is_empty() {
            return Ok(Vec::new());
        }

        // Score all memories by similarity against the supplied query vector
        let mut scored: Vec<_> = memories
            .values()
            .filter_map(|entry| {
                entry.embedding.as_ref().map(|embedding| {
                    let similarity = Self::cosine_similarity(query_vec, embedding);
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

    async fn retrieve_by_scope(
        &self,
        query_vec: &[f32],
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let memories = self.memories.read();

        // If no memories, return empty
        if memories.is_empty() {
            return Ok(Vec::new());
        }

        // Score all memories by similarity, filtering by scope
        let mut scored: Vec<_> = memories
            .values()
            .filter(|entry| entry.scope.matches(scope))
            .filter_map(|entry| {
                entry.embedding.as_ref().map(|embedding| {
                    let similarity = Self::cosine_similarity(query_vec, embedding);
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

    /// List all memories. Delegates to the inherent [`SynapseMemory::list`].
    async fn list(&self) -> Result<Vec<MemoryEntry>> {
        Ok(self.memories.read().values().cloned().collect())
    }

    /// Count memories. Delegates to the inherent [`SynapseMemory::len`].
    async fn len(&self) -> Result<usize> {
        Ok(self.memories.read().len())
    }

    /// Whether the store is empty. Delegates to the inherent
    /// [`SynapseMemory::is_empty`].
    async fn is_empty(&self) -> Result<bool> {
        Ok(self.memories.read().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::MockEmbedder;
    use crate::traits::Embedder;

    fn create_synapse() -> SynapseMemory {
        SynapseMemory::new()
    }

    /// Helper function to generate embeddings from text using MockEmbedder.
    ///
    /// Tests now pass query **vectors** to `retrieve`, mirroring how the
    /// orchestrator embeds the query once before calling the store.
    async fn generate_embedding(text: &str) -> Vec<f32> {
        let embedder = MockEmbedder::new();
        embedder
            .embed(text)
            .await
            .unwrap_or_else(|_| vec![0.0; 384])
    }

    #[tokio::test]
    async fn test_synapse_new() {
        let synapse = create_synapse();
        assert!(synapse.is_empty());
        assert_eq!(synapse.len(), 0);
    }

    #[tokio::test]
    async fn test_synapse_store_and_retrieve() {
        let synapse = create_synapse();
        let id = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry = MemoryEntry::builder(id, "Test memory".to_string())
            .embedding(embedding)
            .build();

        synapse.store(entry.clone()).await.unwrap();
        assert_eq!(synapse.len(), 1);

        let results = synapse
            .retrieve(&generate_embedding("test").await, 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_synapse_delete() {
        let synapse = create_synapse();
        let id = MemoryId::new();
        let entry = MemoryEntry::new(id, "Test memory".to_string());

        synapse.store(entry).await.unwrap();
        assert_eq!(synapse.len(), 1);

        synapse.delete(&id).await.unwrap();
        assert_eq!(synapse.len(), 0);
    }

    #[tokio::test]
    async fn test_synapse_clear() {
        let synapse = create_synapse();

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
        let synapse = create_synapse();
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
        let synapse = create_synapse();
        let results = synapse
            .retrieve(&generate_embedding("test").await, 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_synapse_retrieve_with_salience() {
        let synapse = create_synapse();

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

        let results = synapse
            .retrieve(&generate_embedding("important").await, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        // Higher salience should rank first
        assert!(results[0].salience >= results[1].salience);
    }

    #[tokio::test]
    async fn test_synapse_retrieve_by_scope_global() {
        let synapse = create_synapse();

        let id1 = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry1 = MemoryEntry::builder(id1, "Global memory".to_string())
            .scope(MemoryScope::Global)
            .embedding(embedding.clone())
            .build();

        let id2 = MemoryId::new();
        let entry2 = MemoryEntry::builder(id2, "User memory".to_string())
            .scope(MemoryScope::User("user1".to_string()))
            .embedding(embedding)
            .build();

        synapse.store(entry1).await.unwrap();
        synapse.store(entry2).await.unwrap();

        // Global scope should match all
        let results = synapse
            .retrieve_by_scope(
                &generate_embedding("memory").await,
                &MemoryScope::Global,
                10,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_synapse_retrieve_by_scope_user() {
        let synapse = create_synapse();

        let id1 = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry1 = MemoryEntry::builder(id1, "User1 memory".to_string())
            .scope(MemoryScope::User("user1".to_string()))
            .embedding(embedding.clone())
            .build();

        let id2 = MemoryId::new();
        let entry2 = MemoryEntry::builder(id2, "User2 memory".to_string())
            .scope(MemoryScope::User("user2".to_string()))
            .embedding(embedding)
            .build();

        synapse.store(entry1).await.unwrap();
        synapse.store(entry2).await.unwrap();

        // User1 scope should only match user1 memories
        let results = synapse
            .retrieve_by_scope(
                &generate_embedding("memory").await,
                &MemoryScope::User("user1".to_string()),
                10,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "User1 memory");
    }

    #[tokio::test]
    async fn test_synapse_retrieve_by_scope_agent() {
        let synapse = create_synapse();

        let id1 = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry1 = MemoryEntry::builder(id1, "Agent1 memory".to_string())
            .scope(MemoryScope::Agent("agent1".to_string()))
            .embedding(embedding.clone())
            .build();

        let id2 = MemoryId::new();
        let entry2 = MemoryEntry::builder(id2, "Agent2 memory".to_string())
            .scope(MemoryScope::Agent("agent2".to_string()))
            .embedding(embedding)
            .build();

        synapse.store(entry1).await.unwrap();
        synapse.store(entry2).await.unwrap();

        // Agent1 scope should only match agent1 memories
        let results = synapse
            .retrieve_by_scope(
                &generate_embedding("memory").await,
                &MemoryScope::Agent("agent1".to_string()),
                10,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Agent1 memory");
    }

    #[tokio::test]
    async fn test_synapse_retrieve_by_scope_session() {
        let synapse = create_synapse();

        let id1 = MemoryId::new();
        let embedding = vec![0.1; 384];
        let entry1 = MemoryEntry::builder(id1, "Session1 memory".to_string())
            .scope(MemoryScope::Session("session1".to_string()))
            .embedding(embedding.clone())
            .build();

        let id2 = MemoryId::new();
        let entry2 = MemoryEntry::builder(id2, "Session2 memory".to_string())
            .scope(MemoryScope::Session("session2".to_string()))
            .embedding(embedding)
            .build();

        synapse.store(entry1).await.unwrap();
        synapse.store(entry2).await.unwrap();

        // Session1 scope should only match session1 memories
        let results = synapse
            .retrieve_by_scope(
                &generate_embedding("memory").await,
                &MemoryScope::Session("session1".to_string()),
                10,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Session1 memory");
    }

    // ============================================================================
    // Phase 2: Behavioral Relevance Tests
    // ============================================================================
    // These tests verify that semantic similarity and salience blending work
    // correctly with the blending formula: score = 0.7 * similarity + 0.3 * salience

    #[tokio::test]
    async fn test_synapse_semantic_similarity_ranking() {
        // Test that exact matches rank higher than non-matches
        // With MockEmbedder, same text produces same embedding (similarity = 1.0)
        let synapse = create_synapse();

        // Generate embeddings for each text
        let embedding_dog = generate_embedding("dog").await;
        let embedding_puppy = generate_embedding("puppy").await;
        let embedding_unrelated = generate_embedding("unrelated").await;

        // Store three memories with their actual embeddings
        let id_exact = MemoryId::new();
        let entry_exact = MemoryEntry::builder(id_exact, "dog".to_string())
            .embedding(embedding_dog)
            .salience(0.5) // Medium salience
            .build();

        let id_partial = MemoryId::new();
        let entry_partial = MemoryEntry::builder(id_partial, "puppy".to_string())
            .embedding(embedding_puppy)
            .salience(0.5) // Same salience as exact
            .build();

        let id_unrelated = MemoryId::new();
        let entry_unrelated = MemoryEntry::builder(id_unrelated, "unrelated".to_string())
            .embedding(embedding_unrelated)
            .salience(0.5) // Same salience as exact
            .build();

        synapse.store(entry_exact).await.unwrap();
        synapse.store(entry_partial).await.unwrap();
        synapse.store(entry_unrelated).await.unwrap();

        // Query with "dog" - exact match should rank first
        let results = synapse
            .retrieve(&generate_embedding("dog").await, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 3);

        // Exact match should be first (similarity = 1.0 to itself)
        // score = 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        assert_eq!(results[0].content, "dog");
    }

    #[tokio::test]
    async fn test_synapse_salience_override_blending() {
        // Test that high salience can boost ranking
        // Formula: score = 0.7 * similarity + 0.3 * salience
        let synapse = create_synapse();

        // Generate embeddings for each text
        let embedding_important = generate_embedding("important").await;
        let embedding_trivial = generate_embedding("trivial").await;

        // Create two entries with different salience values
        let id_high_sal = MemoryId::new();
        let entry_high_sal = MemoryEntry::builder(id_high_sal, "important".to_string())
            .embedding(embedding_important)
            .salience(0.9) // High salience
            .build();

        let id_low_sal = MemoryId::new();
        let entry_low_sal = MemoryEntry::builder(id_low_sal, "trivial".to_string())
            .embedding(embedding_trivial)
            .salience(0.1) // Low salience
            .build();

        synapse.store(entry_high_sal).await.unwrap();
        synapse.store(entry_low_sal).await.unwrap();

        // Query with "important" - should rank first due to exact match + high salience
        let results = synapse
            .retrieve(&generate_embedding("important").await, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        // "important" should rank first
        // score = 0.7 * 1.0 + 0.3 * 0.9 = 0.97
        assert_eq!(results[0].content, "important");
        assert_eq!(results[1].content, "trivial");
    }

    #[tokio::test]
    async fn test_synapse_salience_blending_weights() {
        // Test that the blending weights (0.7 similarity, 0.3 salience) are applied correctly
        // When two entries have same text (same embedding) but different salience
        let synapse = create_synapse();

        // Generate embedding for "memory"
        let embedding_memory = generate_embedding("memory").await;

        // Create two entries with same text but different salience
        let id_high_sal = MemoryId::new();
        let entry_high_sal = MemoryEntry::builder(id_high_sal, "memory".to_string())
            .embedding(embedding_memory.clone())
            .salience(0.9) // High salience
            .build();

        let id_low_sal = MemoryId::new();
        let entry_low_sal = MemoryEntry::builder(id_low_sal, "memory".to_string())
            .embedding(embedding_memory)
            .salience(0.1) // Low salience
            .build();

        synapse.store(entry_high_sal).await.unwrap();
        synapse.store(entry_low_sal).await.unwrap();

        // Query with "memory" - both have same similarity (1.0) but different salience
        let results = synapse
            .retrieve(&generate_embedding("memory").await, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        // High salience should rank first
        // score_high = 0.7 * 1.0 + 0.3 * 0.9 = 0.97
        // score_low = 0.7 * 1.0 + 0.3 * 0.1 = 0.73
        assert_eq!(results[0].salience, 0.9);
        assert_eq!(results[1].salience, 0.1);
    }

    #[tokio::test]
    async fn test_synapse_retrieve_respects_limit() {
        // Test that retrieve respects the limit parameter
        let synapse = create_synapse();

        // Generate embedding for "memory"
        let embedding_memory = generate_embedding("memory").await;

        // Store 5 memories with same text (same embedding)
        for _i in 0..5 {
            let id = MemoryId::new();
            let entry = MemoryEntry::builder(id, "memory".to_string())
                .embedding(embedding_memory.clone())
                .build();
            synapse.store(entry).await.unwrap();
        }

        // Query with limit 2
        let results = synapse
            .retrieve(&generate_embedding("memory").await, 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        // Query with limit 10 (more than stored)
        let results = synapse
            .retrieve(&generate_embedding("memory").await, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 5);

        // Query with limit 0
        let results = synapse
            .retrieve(&generate_embedding("memory").await, 0)
            .await
            .unwrap();
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_synapse_blending_formula_verification() {
        // Test that the blending formula (0.7 * similarity + 0.3 * salience) is applied
        // by verifying that higher salience boosts ranking when similarity is equal
        let synapse = create_synapse();

        // Generate embedding for "test"
        let embedding_test = generate_embedding("test").await;

        // Create two entries with same text (same embedding) but different salience
        let id_high_sal = MemoryId::new();
        let entry_high_sal = MemoryEntry::builder(id_high_sal, "test".to_string())
            .embedding(embedding_test.clone())
            .salience(0.9) // High salience
            .build();

        let id_low_sal = MemoryId::new();
        let entry_low_sal = MemoryEntry::builder(id_low_sal, "test".to_string())
            .embedding(embedding_test)
            .salience(0.1) // Low salience
            .build();

        synapse.store(entry_high_sal).await.unwrap();
        synapse.store(entry_low_sal).await.unwrap();

        // Query with "test" - both have same similarity (1.0) but different salience
        let results = synapse
            .retrieve(&generate_embedding("test").await, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        // High salience should rank first when similarity is equal
        // score_high = 0.7 * 1.0 + 0.3 * 0.9 = 0.97
        // score_low = 0.7 * 1.0 + 0.3 * 0.1 = 0.73
        assert_eq!(results[0].salience, 0.9);
        assert_eq!(results[1].salience, 0.1);
    }
}
