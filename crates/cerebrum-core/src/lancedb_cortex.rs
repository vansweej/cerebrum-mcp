use crate::embedder::Embedder;
use crate::error::{CerebrumError, Result};
use crate::models::{MemoryEntry, MemoryId, MemoryScope};
use crate::traits::MemoryStore;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Schema for storing memories in LanceDB.
///
/// This struct represents how memories are stored in the vector database.
/// It includes all fields from MemoryEntry plus the embedding vector.
#[derive(Debug, Clone)]
pub struct LanceDBMemoryRecord {
    /// Unique identifier for this memory.
    pub id: String,
    /// The text content of the memory.
    pub content: String,
    /// Importance score (0.0–1.0) for ranking and promotion decisions.
    pub salience: f32,
    /// When this memory was created (ISO 8601 string).
    pub timestamp: String,
    /// Session ID where this memory originated (if applicable).
    pub source_session_id: Option<String>,
    /// Scope or visibility of this memory (string representation).
    pub scope: String,
    /// 384-dimensional embedding vector (BGE-small).
    pub embedding: Vec<f32>,
    /// Arbitrary metadata as JSON string.
    pub metadata_json: String,
}

impl LanceDBMemoryRecord {
    /// Convert from MemoryEntry to LanceDBMemoryRecord.
    pub fn from_entry(entry: &MemoryEntry) -> Result<Self> {
        let embedding = entry.embedding.clone().ok_or_else(|| {
            CerebrumError::Validation("Memory entry missing embedding".to_string())
        })?;

        Ok(Self {
            id: entry.id.to_string(),
            content: entry.content.clone(),
            salience: entry.salience,
            timestamp: entry.timestamp.to_rfc3339(),
            source_session_id: entry.source_session_id.clone(),
            scope: entry.scope.as_str(),
            embedding,
            metadata_json: serde_json::to_string(&entry.metadata)
                .map_err(|e| CerebrumError::Serialization(e.to_string()))?,
        })
    }

    /// Convert from LanceDBMemoryRecord back to MemoryEntry.
    pub fn to_entry(&self) -> Result<MemoryEntry> {
        let id = MemoryId::from_string(&self.id)?;
        let timestamp = chrono::DateTime::parse_from_rfc3339(&self.timestamp)
            .map_err(|e| CerebrumError::Validation(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&chrono::Utc);

        let scope = parse_scope_string(&self.scope)?;

        let metadata = serde_json::from_str(&self.metadata_json)
            .map_err(|e| CerebrumError::Serialization(e.to_string()))?;

        Ok(MemoryEntry {
            id,
            content: self.content.clone(),
            metadata,
            timestamp,
            salience: self.salience,
            tier: crate::models::MemoryTier::Cortex,
            embedding: Some(self.embedding.clone()),
            source_session_id: self.source_session_id.clone(),
            scope,
        })
    }
}

/// Parse a scope string back into a MemoryScope enum.
fn parse_scope_string(scope_str: &str) -> Result<MemoryScope> {
    if scope_str == "global" {
        Ok(MemoryScope::Global)
    } else if let Some(user_id) = scope_str.strip_prefix("user:") {
        Ok(MemoryScope::User(user_id.to_string()))
    } else if let Some(agent_id) = scope_str.strip_prefix("agent:") {
        Ok(MemoryScope::Agent(agent_id.to_string()))
    } else if let Some(session_id) = scope_str.strip_prefix("session:") {
        Ok(MemoryScope::Session(session_id.to_string()))
    } else {
        Err(CerebrumError::Validation(format!(
            "Invalid scope string: {}",
            scope_str
        )))
    }
}

/// Persistent long-term memory storage backed by LanceDB (Cortex tier).
///
/// Stores memories in a vector database for efficient semantic search and
/// persistent storage across sessions. Supports salience-based ranking.
///
/// Note: This is a simplified implementation using in-memory storage with LanceDB
/// integration planned for Phase 6 Step 1. The actual LanceDB integration will
/// replace this HashMap-based approach with persistent vector database storage.
pub struct LanceDBCortex {
    memories: Arc<RwLock<HashMap<MemoryId, MemoryEntry>>>,
    embedder: Arc<dyn Embedder>,
}

impl LanceDBCortex {
    /// Create a new LanceDB Cortex memory store.
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

    /// Create a new LanceDB Cortex from components.
    pub fn from_parts(
        memories: Arc<RwLock<HashMap<MemoryId, MemoryEntry>>>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self { memories, embedder }
    }

    /// Calculate cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
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
}

#[async_trait]
impl MemoryStore for LanceDBCortex {
    async fn store(&self, entry: MemoryEntry) -> Result<()> {
        self.memories.write().insert(entry.id, entry);
        Ok(())
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        // Generate embedding for the query
        let query_embedding = self.embedder.embed(query).await?;

        let memories = self.memories.read();
        let mut scored: Vec<(MemoryEntry, f32)> = memories
            .values()
            .filter_map(|entry| {
                let embedding = entry.embedding.as_ref()?;
                let similarity = Self::cosine_similarity(&query_embedding, embedding);
                let score = (similarity * 0.7) + (entry.salience * 0.3);
                Some((entry.clone(), score))
            })
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(entry, _)| entry)
            .collect())
    }

    async fn retrieve_by_scope(
        &self,
        query: &str,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        // Generate embedding for the query
        let query_embedding = self.embedder.embed(query).await?;

        let memories = self.memories.read();
        let mut scored: Vec<(MemoryEntry, f32)> = memories
            .values()
            .filter_map(|entry| {
                // Check if scope matches
                if !scope.matches(&entry.scope) {
                    return None;
                }

                let embedding = entry.embedding.as_ref()?;
                let similarity = Self::cosine_similarity(&query_embedding, embedding);
                let score = (similarity * 0.7) + (entry.salience * 0.3);
                Some((entry.clone(), score))
            })
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
    use crate::embedder::MockEmbedder;
    use crate::models::MemoryTier;

    #[tokio::test]
    async fn test_lancedb_cortex_new() {
        let embedder = Arc::new(MockEmbedder::new());
        let result = LanceDBCortex::new(":memory:", embedder).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_store_and_retrieve() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry.clone()).await.unwrap();

        let results = cortex.retrieve("test", 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_len() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry).await.unwrap();

        let len = cortex.len().await.unwrap();
        assert!(len > 0);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_delete() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        let id = MemoryId::new();
        let entry = MemoryEntry::builder(id, "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry).await.unwrap();
        cortex.delete(&id).await.unwrap();

        let results = cortex.retrieve("test", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope("test", &MemoryScope::User("user1".to_string()), 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope_mismatch() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope("test", &MemoryScope::User("user2".to_string()), 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope_global() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope("test", &MemoryScope::Global, 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_is_empty() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        assert!(cortex.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_list() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry).await.unwrap();

        let entries = cortex.list().await.unwrap();
        assert!(!entries.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_memory_record_conversion() {
        let id = MemoryId::new();
        let entry = MemoryEntry::builder(id, "test content".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::Global)
            .build();

        let record = LanceDBMemoryRecord::from_entry(&entry).unwrap();
        let converted = record.to_entry().unwrap();

        assert_eq!(converted.id, entry.id);
        assert_eq!(converted.content, entry.content);
        assert_eq!(converted.salience, entry.salience);
    }

    #[test]
    fn test_parse_scope_string_global() {
        let result = parse_scope_string("global");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), MemoryScope::Global));
    }

    #[test]
    fn test_parse_scope_string_user() {
        let result = parse_scope_string("user:alice");
        assert!(result.is_ok());
        match result.unwrap() {
            MemoryScope::User(id) => assert_eq!(id, "alice"),
            _ => panic!("Expected User scope"),
        }
    }

    #[test]
    fn test_parse_scope_string_agent() {
        let result = parse_scope_string("agent:bot123");
        assert!(result.is_ok());
        match result.unwrap() {
            MemoryScope::Agent(id) => assert_eq!(id, "bot123"),
            _ => panic!("Expected Agent scope"),
        }
    }

    #[test]
    fn test_parse_scope_string_session() {
        let result = parse_scope_string("session:sess456");
        assert!(result.is_ok());
        match result.unwrap() {
            MemoryScope::Session(id) => assert_eq!(id, "sess456"),
            _ => panic!("Expected Session scope"),
        }
    }

    #[test]
    fn test_parse_scope_string_invalid() {
        let result = parse_scope_string("invalid:scope");
        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert!(similarity.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert!((similarity + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_magnitude() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_search_by_salience() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        // Store entries with different salience values
        let entry1 = MemoryEntry::builder(MemoryId::new(), "high salience".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .salience(0.9)
            .build();

        let entry2 = MemoryEntry::builder(MemoryId::new(), "low salience".to_string())
            .embedding(vec![0.2; 384])
            .tier(MemoryTier::Cortex)
            .salience(0.1)
            .build();

        cortex.store(entry1).await.unwrap();
        cortex.store(entry2).await.unwrap();

        let results = cortex.search_by_salience(10).await.unwrap();
        assert_eq!(results.len(), 2);
        // First result should have higher salience
        assert!(results[0].salience >= results[1].salience);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_search_by_salience_limit() {
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(":memory:", embedder.clone())
            .await
            .unwrap();

        // Store multiple entries
        for i in 0..5 {
            let entry = MemoryEntry::builder(MemoryId::new(), format!("entry {}", i))
                .embedding(vec![0.1; 384])
                .tier(MemoryTier::Cortex)
                .salience(i as f32 * 0.2)
                .build();
            cortex.store(entry).await.unwrap();
        }

        let results = cortex.search_by_salience(2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_from_parts() {
        let embedder = Arc::new(MockEmbedder::new());
        let memories = Arc::new(RwLock::new(HashMap::new()));

        let cortex = LanceDBCortex::from_parts(memories.clone(), embedder);
        assert!(cortex.is_empty().await.unwrap());
    }

    #[test]
    fn test_lancedb_memory_record_from_entry_missing_embedding() {
        let entry = MemoryEntry::builder(MemoryId::new(), "test".to_string())
            .tier(MemoryTier::Cortex)
            .build();

        let result = LanceDBMemoryRecord::from_entry(&entry);
        assert!(result.is_err());
    }

    #[test]
    fn test_lancedb_memory_record_all_scopes() {
        let scopes = vec![
            MemoryScope::Global,
            MemoryScope::User("user1".to_string()),
            MemoryScope::Agent("agent1".to_string()),
            MemoryScope::Session("session1".to_string()),
        ];

        for scope in scopes {
            let entry = MemoryEntry::builder(MemoryId::new(), "test".to_string())
                .embedding(vec![0.1; 384])
                .tier(MemoryTier::Cortex)
                .scope(scope.clone())
                .build();

            let record = LanceDBMemoryRecord::from_entry(&entry).unwrap();
            let converted = record.to_entry().unwrap();
            assert_eq!(converted.scope, scope);
        }
    }
}
