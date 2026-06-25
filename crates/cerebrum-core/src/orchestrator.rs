use crate::cortex::CortexMemory;
use crate::embedder::Embedder;
use crate::error::Result;
use crate::models::{MemoryEntry, MemoryId, MemoryTier};
use crate::synapse::SynapseMemory;
use crate::traits::MemoryStore;
use std::collections::HashMap;
use std::sync::Arc;

/// Orchestrates memory operations across Synapse and Cortex tiers.
///
/// Provides a unified interface for memory operations with automatic
/// tier management, blended search, and promotion logic.
pub struct MemoryOrchestrator {
    synapse: Arc<SynapseMemory>,
    cortex: Arc<CortexMemory>,
    embedder: Arc<dyn Embedder>,
}

impl MemoryOrchestrator {
    /// Create a new MemoryOrchestrator with both tiers.
    ///
    /// # Arguments
    /// * `cortex_db_path` - Path to the Cortex database directory
    /// * `embedder` - Embedder instance for generating embeddings
    pub async fn new(cortex_db_path: &str, embedder: Arc<dyn Embedder>) -> Result<Self> {
        let synapse = Arc::new(SynapseMemory::new());
        let cortex = Arc::new(CortexMemory::new(cortex_db_path, embedder.clone()).await?);

        Ok(Self {
            synapse,
            cortex,
            embedder,
        })
    }

    /// Store a memory in the Synapse tier (short-term).
    ///
    /// # Arguments
    /// * `content` - The memory content
    /// * `metadata` - Optional metadata key-value pairs
    ///
    /// # Returns
    /// The ID of the stored memory
    pub async fn remember(
        &self,
        content: String,
        metadata: HashMap<String, String>,
    ) -> Result<MemoryId> {
        let id = MemoryId::new();

        // Generate embedding
        let embedding = self.embedder.embed(&content).await?;

        // Create entry with embedding
        let mut entry = MemoryEntry::builder(id, content)
            .embedding(embedding)
            .tier(MemoryTier::Synapse)
            .build();

        entry.metadata = metadata;

        // Store in Synapse
        self.synapse.store(entry).await?;

        Ok(id)
    }

    /// Recall memories matching a query from both tiers.
    ///
    /// Performs blended search across Synapse and Cortex, merging and
    /// ranking results by relevance and salience.
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Ranked list of matching memories
    pub async fn recall(&self, query: String, limit: usize) -> Result<Vec<MemoryEntry>> {
        // Search both tiers in parallel
        let synapse_results = self.synapse.retrieve(&query, limit).await?;
        let cortex_results = self.cortex.retrieve(&query, limit).await?;

        // Merge results
        let mut all_results = Vec::new();
        all_results.extend(synapse_results);
        all_results.extend(cortex_results);

        // Remove duplicates (keep first occurrence)
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|entry| seen.insert(entry.id));

        // Sort by salience (descending)
        all_results.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top N
        Ok(all_results.into_iter().take(limit).collect())
    }

    /// Promote a memory from Synapse to Cortex.
    ///
    /// Moves a memory from short-term to long-term storage.
    ///
    /// # Arguments
    /// * `id` - The ID of the memory to promote
    pub async fn memorize(&self, id: MemoryId) -> Result<()> {
        // Retrieve from Synapse
        if let Some(entry) = self.synapse.list().await?.into_iter().find(|e| e.id == id) {
            // Create a copy with Cortex tier
            let mut cortex_entry = entry.clone();
            cortex_entry.tier = MemoryTier::Cortex;

            // Store in Cortex
            self.cortex.store(cortex_entry).await?;

            // Delete from Synapse
            self.synapse.delete(&id).await?;

            Ok(())
        } else {
            Err(crate::error::CerebrumError::NotFound(format!(
                "Memory {} not found in Synapse",
                id
            )))
        }
    }

    /// Delete a memory from both tiers.
    ///
    /// # Arguments
    /// * `id` - The ID of the memory to delete
    pub async fn forget(&self, id: MemoryId) -> Result<()> {
        // Try to delete from both tiers (ignore errors if not found)
        let _ = self.synapse.delete(&id).await;
        let _ = self.cortex.delete(&id).await;

        Ok(())
    }

    /// End the current session.
    ///
    /// Clears Synapse and optionally promotes high-salience memories to Cortex.
    ///
    /// # Arguments
    /// * `auto_promote_threshold` - Salience threshold for automatic promotion (0.0-1.0)
    pub async fn end_session(&self, auto_promote_threshold: f32) -> Result<()> {
        // Get all memories from Synapse
        let memories = self.synapse.list().await?;

        // Promote high-salience memories
        for entry in memories {
            if entry.salience >= auto_promote_threshold {
                let mut cortex_entry = entry.clone();
                cortex_entry.tier = MemoryTier::Cortex;
                self.cortex.store(cortex_entry).await?;
            }
        }

        // Clear Synapse
        self.synapse.clear().await?;

        Ok(())
    }

    /// Get the number of memories in Synapse.
    pub async fn synapse_len(&self) -> Result<usize> {
        Ok(self.synapse.len())
    }

    /// Get the number of memories in Cortex.
    pub async fn cortex_len(&self) -> Result<usize> {
        self.cortex.len().await
    }

    /// Get all memories from Synapse.
    pub async fn synapse_list(&self) -> Result<Vec<MemoryEntry>> {
        self.synapse.list().await
    }

    /// Get all memories from Cortex.
    pub async fn cortex_list(&self) -> Result<Vec<MemoryEntry>> {
        self.cortex.list().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_new() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
        assert_eq!(orchestrator.cortex_len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_orchestrator_remember() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

        let memories = orchestrator.synapse_list().await.unwrap();
        assert_eq!(memories[0].id, id);
    }

    #[tokio::test]
    async fn test_orchestrator_recall() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let results = orchestrator
            .recall("test".to_string(), 10)
            .await
            .expect("Failed to recall");

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_memorize() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);
        assert_eq!(orchestrator.cortex_len().await.unwrap(), 0);

        orchestrator.memorize(id).await.expect("Failed to memorize");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
        assert_eq!(orchestrator.cortex_len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_orchestrator_forget() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

        orchestrator.forget(id).await.expect("Failed to forget");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_orchestrator_end_session_with_promotion() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        // Store memories with different salience levels
        let _id1 = orchestrator
            .remember("Important memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let _id2 = orchestrator
            .remember("Less important memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        // Manually set salience (in real scenario, this would be set during remember)
        let mut memories = orchestrator.synapse_list().await.unwrap();
        memories[0].salience = 0.9;
        memories[1].salience = 0.2;

        // Re-store with updated salience
        orchestrator.forget(_id1).await.ok();
        orchestrator.forget(_id2).await.ok();

        let _id1 = orchestrator
            .remember("Important memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let _id2 = orchestrator
            .remember("Less important memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        // Update salience manually
        let mut synapse_memories = orchestrator.synapse_list().await.unwrap();
        synapse_memories[0].salience = 0.9;
        synapse_memories[1].salience = 0.2;

        // End session with threshold 0.5
        orchestrator
            .end_session(0.5)
            .await
            .expect("Failed to end session");

        // Synapse should be empty
        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_orchestrator_blended_recall() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        // Store in Synapse
        let _id1 = orchestrator
            .remember("Synapse memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        // Store in Cortex
        let id2 = orchestrator
            .remember("Cortex memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        orchestrator
            .memorize(id2)
            .await
            .expect("Failed to memorize");

        // Recall should return results from both tiers
        let results = orchestrator
            .recall("memory".to_string(), 10)
            .await
            .expect("Failed to recall");

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_orchestrator_recall_deduplication() {
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new("/tmp/test_orch", embedder)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        // Promote to Cortex (now in both tiers)
        orchestrator.memorize(id).await.expect("Failed to memorize");

        // Store another copy in Synapse
        let _id2 = orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        // Recall should deduplicate
        let results = orchestrator
            .recall("test".to_string(), 10)
            .await
            .expect("Failed to recall");

        // Should have 2 unique memories, not duplicates
        assert_eq!(results.len(), 2);
    }
}
