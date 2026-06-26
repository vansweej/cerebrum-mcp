use crate::embedder::Embedder;
use crate::error::{CerebrumError, Result};
#[allow(unused_imports)]
use crate::models::{MemoryEntry, MemoryId};
use crate::traits::MemoryStore;
use async_trait::async_trait;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Metadata about an embedding's provenance.
///
/// Tracks which model generated an embedding and when, enabling
/// migration strategies to make informed decisions about re-embedding.
#[derive(Debug, Clone)]
pub struct EmbeddingProvenance {
    /// Name of the model that generated this embedding (e.g., "bge-small-v1.5").
    pub model_name: String,
    /// ISO 8601 timestamp when the embedding was generated.
    pub generated_at: String,
    /// Version of the embedding model (for tracking breaking changes).
    pub model_version: String,
}

impl EmbeddingProvenance {
    /// Create a new embedding provenance record.
    pub fn new(model_name: String, model_version: String) -> Self {
        Self {
            model_name,
            generated_at: chrono::Utc::now().to_rfc3339(),
            model_version,
        }
    }
}

/// Strategy for handling embedding migrations.
///
/// Different strategies support different use cases:
/// - `Reembed`: Re-embed all memories with the new model (most accurate, slowest)
/// - `Preserve`: Keep old embeddings, add new ones alongside (preserves history, uses more storage)
/// - `Hybrid`: Re-embed high-salience memories, preserve low-salience ones (balanced approach)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MigrationStrategy {
    /// Re-embed all memories with the new model.
    Reembed,
    /// Keep old embeddings, add new ones alongside.
    Preserve,
    /// Re-embed high-salience memories, preserve low-salience ones.
    Hybrid,
}

/// Configuration for an embedding migration.
#[derive(Clone)]
pub struct MigrationConfig {
    /// Strategy to use for the migration.
    pub strategy: MigrationStrategy,
    /// New embedder to use for re-embedding.
    pub new_embedder: Arc<dyn Embedder>,
    /// Salience threshold for hybrid strategy (0.0-1.0).
    pub hybrid_threshold: f32,
    /// Whether to perform a dry-run (preview changes without applying).
    pub dry_run: bool,
    /// Maximum number of memories to migrate in a single batch.
    pub batch_size: usize,
}

impl fmt::Debug for MigrationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MigrationConfig")
            .field("strategy", &self.strategy)
            .field("hybrid_threshold", &self.hybrid_threshold)
            .field("dry_run", &self.dry_run)
            .field("batch_size", &self.batch_size)
            .field("new_embedder", &"<Arc<dyn Embedder>>")
            .finish()
    }
}

impl MigrationConfig {
    /// Create a new migration configuration.
    pub fn new(strategy: MigrationStrategy, new_embedder: Arc<dyn Embedder>) -> Self {
        Self {
            strategy,
            new_embedder,
            hybrid_threshold: 0.5,
            dry_run: false,
            batch_size: 100,
        }
    }

    /// Set the dry-run flag.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Set the hybrid threshold.
    pub fn with_hybrid_threshold(mut self, threshold: f32) -> Self {
        self.hybrid_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }
}

/// Result of an embedding migration.
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Total number of memories processed.
    pub total_memories: usize,
    /// Number of memories re-embedded.
    pub reembedded_count: usize,
    /// Number of memories preserved.
    pub preserved_count: usize,
    /// Number of memories that failed to migrate.
    pub failed_count: usize,
    /// Whether this was a dry-run (no changes applied).
    pub dry_run: bool,
}

impl MigrationResult {
    /// Create a new migration result.
    pub fn new(total_memories: usize, dry_run: bool) -> Self {
        Self {
            total_memories,
            reembedded_count: 0,
            preserved_count: 0,
            failed_count: 0,
            dry_run,
        }
    }

    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f32 {
        if self.total_memories == 0 {
            100.0
        } else {
            ((self.total_memories - self.failed_count) as f32 / self.total_memories as f32) * 100.0
        }
    }
}

/// Trait for embedding migration strategies.
#[async_trait]
pub trait EmbeddingMigration: Send + Sync {
    /// Execute the migration on a memory store.
    ///
    /// # Arguments
    /// * `store` - The memory store to migrate
    /// * `config` - Migration configuration
    ///
    /// # Returns
    /// A MigrationResult with statistics about the migration
    async fn migrate(
        &self,
        store: &dyn MemoryStore,
        config: &MigrationConfig,
    ) -> Result<MigrationResult>;
}

/// Reembed migration strategy: re-embed all memories with the new model.
pub struct ReembedMigration;

#[async_trait]
impl EmbeddingMigration for ReembedMigration {
    async fn migrate(
        &self,
        store: &dyn MemoryStore,
        config: &MigrationConfig,
    ) -> Result<MigrationResult> {
        // Retrieve all memories
        let all_memories = store.list().await?;
        let mut result = MigrationResult::new(all_memories.len(), config.dry_run);

        for memory in all_memories {
            match config.new_embedder.embed(&memory.content).await {
                Ok(new_embedding) => {
                    let mut updated = memory.clone();
                    updated.embedding = Some(new_embedding);

                    if !config.dry_run {
                        store.store(updated).await?;
                    }

                    result.reembedded_count += 1;
                }
                Err(_) => {
                    result.failed_count += 1;
                }
            }
        }

        Ok(result)
    }
}

/// Preserve migration strategy: keep old embeddings, add new ones alongside.
pub struct PreserveMigration;

#[async_trait]
impl EmbeddingMigration for PreserveMigration {
    async fn migrate(
        &self,
        store: &dyn MemoryStore,
        config: &MigrationConfig,
    ) -> Result<MigrationResult> {
        // Retrieve all memories
        let all_memories = store.list().await?;
        let mut result = MigrationResult::new(all_memories.len(), config.dry_run);

        for memory in all_memories {
            // Only add new embedding if one doesn't exist
            if memory.embedding.is_none() {
                match config.new_embedder.embed(&memory.content).await {
                    Ok(new_embedding) => {
                        let mut updated = memory.clone();
                        updated.embedding = Some(new_embedding);

                        if !config.dry_run {
                            store.store(updated).await?;
                        }

                        result.reembedded_count += 1;
                    }
                    Err(_) => {
                        result.failed_count += 1;
                    }
                }
            } else {
                result.preserved_count += 1;
            }
        }

        Ok(result)
    }
}

/// Hybrid migration strategy: re-embed high-salience, preserve low-salience.
pub struct HybridMigration;

#[async_trait]
impl EmbeddingMigration for HybridMigration {
    async fn migrate(
        &self,
        store: &dyn MemoryStore,
        config: &MigrationConfig,
    ) -> Result<MigrationResult> {
        // Retrieve all memories
        let all_memories = store.list().await?;
        let mut result = MigrationResult::new(all_memories.len(), config.dry_run);

        for memory in all_memories {
            // Re-embed if salience is above threshold
            if memory.salience >= config.hybrid_threshold {
                match config.new_embedder.embed(&memory.content).await {
                    Ok(new_embedding) => {
                        let mut updated = memory.clone();
                        updated.embedding = Some(new_embedding);

                        if !config.dry_run {
                            store.store(updated).await?;
                        }

                        result.reembedded_count += 1;
                    }
                    Err(_) => {
                        result.failed_count += 1;
                    }
                }
            } else {
                result.preserved_count += 1;
            }
        }

        Ok(result)
    }
}

/// Migration manager for coordinating embedding migrations.
pub struct MigrationManager {
    strategies: HashMap<MigrationStrategy, Arc<dyn EmbeddingMigration>>,
}

impl MigrationManager {
    /// Create a new migration manager with default strategies.
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert(
            MigrationStrategy::Reembed,
            Arc::new(ReembedMigration) as Arc<dyn EmbeddingMigration>,
        );
        strategies.insert(
            MigrationStrategy::Preserve,
            Arc::new(PreserveMigration) as Arc<dyn EmbeddingMigration>,
        );
        strategies.insert(
            MigrationStrategy::Hybrid,
            Arc::new(HybridMigration) as Arc<dyn EmbeddingMigration>,
        );

        Self { strategies }
    }

    /// Execute a migration using the configured strategy.
    pub async fn execute(
        &self,
        store: &dyn MemoryStore,
        config: &MigrationConfig,
    ) -> Result<MigrationResult> {
        let strategy = self
            .strategies
            .get(&config.strategy)
            .ok_or_else(|| CerebrumError::Validation("Unknown migration strategy".to_string()))?;

        strategy.migrate(store, config).await
    }
}

impl Default for MigrationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::MockEmbedder;
    use crate::models::MemoryTier;
    use crate::synapse::SynapseMemory;

    #[test]
    fn test_embedding_provenance_new() {
        let provenance = EmbeddingProvenance::new("bge-small".to_string(), "1.5".to_string());
        assert_eq!(provenance.model_name, "bge-small");
        assert_eq!(provenance.model_version, "1.5");
        assert!(!provenance.generated_at.is_empty());
    }

    #[test]
    fn test_migration_config_new() {
        let embedder = Arc::new(MockEmbedder::new());
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder);
        assert_eq!(config.strategy, MigrationStrategy::Reembed);
        assert!(!config.dry_run);
        assert_eq!(config.hybrid_threshold, 0.5);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_migration_config_with_dry_run() {
        let embedder = Arc::new(MockEmbedder::new());
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder).with_dry_run(true);
        assert!(config.dry_run);
    }

    #[test]
    fn test_migration_config_with_hybrid_threshold() {
        let embedder = Arc::new(MockEmbedder::new());
        let config =
            MigrationConfig::new(MigrationStrategy::Hybrid, embedder).with_hybrid_threshold(0.75);
        assert_eq!(config.hybrid_threshold, 0.75);
    }

    #[test]
    fn test_migration_config_threshold_clamping() {
        let embedder = Arc::new(MockEmbedder::new());
        let config =
            MigrationConfig::new(MigrationStrategy::Hybrid, embedder).with_hybrid_threshold(1.5); // Should clamp to 1.0
        assert_eq!(config.hybrid_threshold, 1.0);
    }

    #[test]
    fn test_migration_config_with_batch_size() {
        let embedder = Arc::new(MockEmbedder::new());
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder).with_batch_size(50);
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_migration_config_debug() {
        let embedder = Arc::new(MockEmbedder::new());
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("MigrationConfig"));
        assert!(debug_str.contains("Reembed"));
    }

    #[test]
    fn test_migration_manager_default() {
        let manager = MigrationManager::default();
        assert!(manager.strategies.contains_key(&MigrationStrategy::Reembed));
        assert!(manager
            .strategies
            .contains_key(&MigrationStrategy::Preserve));
        assert!(manager.strategies.contains_key(&MigrationStrategy::Hybrid));
    }

    #[test]
    fn test_migration_result_new() {
        let result = MigrationResult::new(100, false);
        assert_eq!(result.total_memories, 100);
        assert!(!result.dry_run);
        assert_eq!(result.success_rate(), 100.0);
    }

    #[test]
    fn test_migration_result_success_rate() {
        let mut result = MigrationResult::new(100, false);
        result.reembedded_count = 90;
        result.failed_count = 10;
        assert_eq!(result.success_rate(), 90.0);
    }

    #[test]
    fn test_migration_result_success_rate_empty() {
        let result = MigrationResult::new(0, false);
        assert_eq!(result.success_rate(), 100.0);
    }

    #[tokio::test]
    async fn test_reembed_migration() {
        let embedder = Arc::new(MockEmbedder::new());
        let store = SynapseMemory::new();

        // Add a memory
        let entry = MemoryEntry::builder(MemoryId::new(), "test content".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Synapse)
            .build();
        store.store(entry).await.unwrap();

        // Run reembed migration
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder);
        let migration = ReembedMigration;
        let result = migration.migrate(&store, &config).await.unwrap();

        assert_eq!(result.total_memories, 1);
        assert_eq!(result.reembedded_count, 1);
        assert_eq!(result.failed_count, 0);
    }

    #[tokio::test]
    async fn test_reembed_migration_dry_run() {
        let embedder = Arc::new(MockEmbedder::new());
        let store = SynapseMemory::new();

        // Add a memory
        let entry = MemoryEntry::builder(MemoryId::new(), "test content".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Synapse)
            .build();
        store.store(entry).await.unwrap();

        // Run reembed migration with dry-run
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder).with_dry_run(true);
        let migration = ReembedMigration;
        let result = migration.migrate(&store, &config).await.unwrap();

        assert!(result.dry_run);
        assert_eq!(result.reembedded_count, 1);
    }

    #[tokio::test]
    async fn test_preserve_migration() {
        let embedder = Arc::new(MockEmbedder::new());
        let store = SynapseMemory::new();

        // Add a memory with embedding
        let entry = MemoryEntry::builder(MemoryId::new(), "test content".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Synapse)
            .build();
        store.store(entry).await.unwrap();

        // Run preserve migration
        let config = MigrationConfig::new(MigrationStrategy::Preserve, embedder);
        let migration = PreserveMigration;
        let result = migration.migrate(&store, &config).await.unwrap();

        assert_eq!(result.total_memories, 1);
        assert_eq!(result.preserved_count, 1);
        assert_eq!(result.reembedded_count, 0);
    }

    #[tokio::test]
    async fn test_hybrid_migration() {
        let embedder = Arc::new(MockEmbedder::new());
        let store = SynapseMemory::new();

        // Add high-salience memory
        let high_salience = MemoryEntry::builder(MemoryId::new(), "important".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Synapse)
            .salience(0.8)
            .build();
        store.store(high_salience).await.unwrap();

        // Add low-salience memory
        let low_salience = MemoryEntry::builder(MemoryId::new(), "trivial".to_string())
            .embedding(vec![0.2; 384])
            .tier(MemoryTier::Synapse)
            .salience(0.2)
            .build();
        store.store(low_salience).await.unwrap();

        // Run hybrid migration with threshold 0.5
        let config =
            MigrationConfig::new(MigrationStrategy::Hybrid, embedder).with_hybrid_threshold(0.5);
        let migration = HybridMigration;
        let result = migration.migrate(&store, &config).await.unwrap();

        assert_eq!(result.total_memories, 2);
        assert_eq!(result.reembedded_count, 1); // High-salience
        assert_eq!(result.preserved_count, 1); // Low-salience
    }

    #[test]
    fn test_migration_manager_new() {
        let manager = MigrationManager::new();
        assert!(manager.strategies.contains_key(&MigrationStrategy::Reembed));
        assert!(manager
            .strategies
            .contains_key(&MigrationStrategy::Preserve));
        assert!(manager.strategies.contains_key(&MigrationStrategy::Hybrid));
    }

    #[tokio::test]
    async fn test_migration_manager_execute() {
        let embedder = Arc::new(MockEmbedder::new());
        let store = SynapseMemory::new();

        // Add a memory
        let entry = MemoryEntry::builder(MemoryId::new(), "test".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Synapse)
            .build();
        store.store(entry).await.unwrap();

        // Execute migration
        let manager = MigrationManager::new();
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder);
        let result = manager.execute(&store, &config).await.unwrap();

        assert_eq!(result.total_memories, 1);
        assert_eq!(result.reembedded_count, 1);
    }

    #[test]
    fn test_migration_config_clone() {
        let embedder = Arc::new(MockEmbedder::new());
        let config = MigrationConfig::new(MigrationStrategy::Reembed, embedder)
            .with_dry_run(true)
            .with_hybrid_threshold(0.7);
        let cloned = config.clone();

        assert_eq!(cloned.strategy, config.strategy);
        assert_eq!(cloned.dry_run, config.dry_run);
        assert_eq!(cloned.hybrid_threshold, config.hybrid_threshold);
    }
}
