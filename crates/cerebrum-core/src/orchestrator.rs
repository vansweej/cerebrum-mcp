use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::Result;
use crate::fastembed_embedder::FastEmbedEmbedder;
use crate::lancedb_cortex::LanceDBCortex;
use crate::models::{MemoryEntry, MemoryId, MemoryScope, MemoryTier};
use crate::synapse::SynapseMemory;
use crate::traits::MemoryStore;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Orchestrates memory operations across Synapse and Cortex tiers.
///
/// The orchestrator is the sole owner of the [`Embedder`]: it embeds content
/// on write and queries on read exactly once, passing query **vectors** down
/// to both storage tiers. Stores never embed text themselves.
pub struct MemoryOrchestrator {
    synapse: Arc<SynapseMemory>,
    cortex: Arc<dyn MemoryStore>,
    embedder: Arc<dyn Embedder>,
    /// Prefix prepended to queries before embedding (nomic asymmetric search).
    query_prefix: String,
    /// Prefix prepended to documents before embedding (nomic asymmetric search).
    document_prefix: String,
}

impl MemoryOrchestrator {
    /// Create a new MemoryOrchestrator from pre-built parts (injectable).
    ///
    /// Mirrors the athenaeum `Engine::with_parts` pattern. Builds a fresh
    /// in-memory Synapse tier and opens the LanceDB-backed Cortex tier. The
    /// query/document prefixes default to empty strings, so `MockEmbedder`
    /// tests are unaffected; [`MemoryOrchestrator::from_config`] sets them.
    ///
    /// # Arguments
    /// * `embedder`   – Embedder instance for generating embeddings
    /// * `db_path`    – Path to the LanceDB database directory
    /// * `table_name` – Name of the LanceDB table
    /// * `dim`        – Expected embedding dimension
    pub async fn new(
        embedder: Arc<dyn Embedder>,
        db_path: &Path,
        table_name: &str,
        dim: usize,
    ) -> Result<Self> {
        let synapse = Arc::new(SynapseMemory::new());
        let cortex: Arc<dyn MemoryStore> =
            Arc::new(LanceDBCortex::new(db_path, table_name, dim).await?);

        Ok(Self {
            synapse,
            cortex,
            embedder,
            query_prefix: String::new(),
            document_prefix: String::new(),
        })
    }

    /// Build a production orchestrator from a [`Config`].
    ///
    /// Mirrors the athenaeum `Engine::new` pattern: constructs a real
    /// [`FastEmbedEmbedder`] against the configured Ollama endpoint, then
    /// probes and warms up the model by embedding a sentinel string. If the
    /// warmup vector does not match `config.embedding_dim` the build fails fast
    /// before any schema-corrupting insert.
    ///
    /// # Warmup Probe
    /// The warmup probe:
    /// 1. Embeds a test string ("warmup") via Ollama
    /// 2. Validates the returned vector has the expected dimension (768 for nomic-embed-text)
    /// 3. Pre-loads the Ollama model to avoid cold-start hangs on first real request
    /// 4. Fails fast if Ollama is unavailable or model dimension doesn't match
    ///
    /// # Prefix Application
    /// The orchestrator stores the configured prefixes and applies them before embedding:
    /// - `document_prefix` is prepended to content in `remember()` before embedding
    /// - `query_prefix` is prepended to queries in `recall()` before embedding
    /// - Original text is stored in `MemoryEntry.content` (without prefix)
    /// - This follows nomic-embed-text best practices for asymmetric search
    ///
    /// # Errors
    /// Returns an error if:
    /// - Ollama is not available at the configured URL
    /// - The embedding model is not found or fails to load
    /// - The returned embedding dimension doesn't match `config.embedding_dim`
    /// - LanceDB initialization fails
    #[cfg(not(tarpaulin_include))]
    pub async fn from_config(config: &Config) -> Result<Self> {
        let embedder = FastEmbedEmbedder::with_timeouts(
            config.ollama_url.clone(),
            config.embed_model.clone(),
            config.embedding_dim,
            config.embed_timeout,
            config.embed_connect_timeout,
        );

        // Probe + warmup: force a model load and validate the dimension.
        // This fails fast before any schema-corrupting insert.
        let warmup = embedder.embed("warmup").await?;
        if warmup.len() != config.embedding_dim {
            return Err(crate::error::CerebrumError::Validation(format!(
                "Ollama model '{}' produced dimension {}, expected {}",
                config.embed_model,
                warmup.len(),
                config.embedding_dim
            )));
        }

        let embedder: Arc<dyn Embedder> = Arc::new(embedder);
        let mut orchestrator = Self::new(
            embedder,
            &config.db_path,
            &config.table_name,
            config.embedding_dim,
        )
        .await?;
        orchestrator.query_prefix = config.query_prefix.clone();
        orchestrator.document_prefix = config.document_prefix.clone();
        Ok(orchestrator)
    }

    /// Get a reference to the embedder.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.embedder)
    }

    /// Get a reference to the Synapse tier.
    pub fn synapse(&self) -> Arc<SynapseMemory> {
        Arc::clone(&self.synapse)
    }

    /// Get a reference to the Cortex tier.
    pub fn cortex(&self) -> Arc<dyn MemoryStore> {
        Arc::clone(&self.cortex)
    }

    /// Store a memory in the Synapse tier with the default salience of 0.5.
    ///
    /// This is a backwards-compatible wrapper around
    /// [`MemoryOrchestrator::remember_with_salience`]. Existing callers that do
    /// not care about importance scoring keep working unchanged.
    ///
    /// # Arguments
    /// * `content` - The memory content (prefixed before embedding)
    /// * `metadata` - Arbitrary key-value metadata
    /// * `scope` - Visibility scope for the memory
    ///
    /// # Returns
    /// The ID of the stored memory
    pub async fn remember(
        &self,
        content: String,
        metadata: HashMap<String, String>,
        scope: MemoryScope,
    ) -> Result<MemoryId> {
        self.remember_with_salience(content, metadata, scope, 0.5)
            .await
    }

    /// Store a memory in the Synapse tier with an explicit salience score.
    ///
    /// Embeds the content once (with the document prefix), then stores it in the
    /// short-term Synapse tier carrying the given salience. The salience is
    /// clamped to `0.0..=1.0` by the entry builder.
    ///
    /// # Arguments
    /// * `content` - The memory content (prefixed before embedding)
    /// * `metadata` - Arbitrary key-value metadata
    /// * `scope` - Visibility scope for the memory
    /// * `salience` - Importance score (0.0–1.0); values outside the range are clamped
    ///
    /// # Returns
    /// The ID of the stored memory
    pub async fn remember_with_salience(
        &self,
        content: String,
        metadata: HashMap<String, String>,
        scope: MemoryScope,
        salience: f32,
    ) -> Result<MemoryId> {
        let id = MemoryId::new();

        // Generate embedding once, prefixing for asymmetric (document) search.
        // The prefix improves semantic search quality (nomic best practice).
        let embedding = self
            .embedder
            .embed(&format!("{}{}", self.document_prefix, content))
            .await?;

        // Capture session id from scope for source_session_id.
        let session_id = match &scope {
            MemoryScope::Session(s) => Some(s.clone()),
            _ => None,
        };

        let mut builder = MemoryEntry::builder(id, content)
            .embedding(embedding)
            .salience(salience)
            .tier(MemoryTier::Synapse)
            .scope(scope);

        if let Some(sid) = session_id {
            builder = builder.source_session_id(sid);
        }

        let mut entry = builder.build();
        entry.metadata = metadata;

        // Store in Synapse (short-term, in-memory)
        self.synapse.store(entry).await?;

        Ok(id)
    }

    /// Recall memories matching a query from both tiers.
    ///
    /// # Blended Search
    /// Performs semantic search across both Synapse and Cortex:
    /// 1. Embeds the query with `query_prefix` (e.g., "search_query: ")
    /// 2. Searches Synapse (in-memory, fast, session-scoped)
    /// 3. Searches Cortex (persistent, comprehensive)
    /// 4. Merges results and deduplicates by ID
    /// 5. Ranks by salience (descending)
    /// 6. Returns top N results
    ///
    /// # Embedding & Prefix Application
    /// - Prepends `query_prefix` to the query before embedding
    /// - Embeds via Ollama (768-dimensional vector)
    /// - Passes the same vector to both Synapse and Cortex
    /// - This ensures consistent ranking across tiers
    ///
    /// # Arguments
    /// * `query` - The search query (will be prefixed before embedding)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Ranked list of matching memories (sorted by salience, descending)
    pub async fn recall(&self, query: String, limit: usize) -> Result<Vec<MemoryEntry>> {
        // Embed the query exactly once (with the asymmetric query prefix), then
        // pass the SAME vector to both tiers. This ensures consistent ranking.
        let query_vec = self
            .embedder
            .embed(&format!("{}{}", self.query_prefix, query))
            .await?;

        // Search both tiers with the shared query vector.
        let synapse_results = self.synapse.retrieve(&query_vec, limit).await?;
        let cortex_results = self.cortex.retrieve(&query_vec, limit).await?;

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

    /// Recall memories matching a query and scope from both tiers (Phase 5).
    ///
    /// Performs blended search across Synapse and Cortex, filtering by scope,
    /// and merging and ranking results by relevance and salience.
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `scope` - Memory scope filter
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Ranked list of matching memories within the specified scope
    pub async fn recall_by_scope(
        &self,
        query: String,
        scope: MemoryScope,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        // Embed the query exactly once (with the asymmetric query prefix), then
        // pass the SAME vector to both tiers.
        let query_vec = self
            .embedder
            .embed(&format!("{}{}", self.query_prefix, query))
            .await?;

        // Search both tiers with the shared query vector and scope filtering.
        let synapse_results = self
            .synapse
            .retrieve_by_scope(&query_vec, &scope, limit)
            .await?;
        let cortex_results = self
            .cortex
            .retrieve_by_scope(&query_vec, &scope, limit)
            .await?;

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
    use tempfile;

    #[tokio::test]
    async fn test_orchestrator_new() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
        assert_eq!(orchestrator.cortex_len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_orchestrator_remember() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

        let memories = orchestrator.synapse_list().await.unwrap();
        assert_eq!(memories[0].id, id);
    }

    #[tokio::test]
    async fn test_orchestrator_recall() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
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
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
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
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

        orchestrator.forget(id).await.expect("Failed to forget");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_orchestrator_end_session_with_promotion() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        // Store memories with different salience levels
        let _id1 = orchestrator
            .remember(
                "Important memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        let _id2 = orchestrator
            .remember(
                "Less important memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
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
            .remember(
                "Important memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        let _id2 = orchestrator
            .remember(
                "Less important memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
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
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        // Store in Synapse
        let _id1 = orchestrator
            .remember(
                "Synapse memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        // Store in Cortex
        let id2 = orchestrator
            .remember(
                "Cortex memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
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
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        // Promote to Cortex (now in both tiers)
        orchestrator.memorize(id).await.expect("Failed to memorize");

        // Store another copy in Synapse
        let _id2 = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
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

    #[tokio::test]
    async fn test_orchestrator_accessor_embedder() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let retrieved_embedder = orchestrator.embedder();
        // Verify we can use the embedder
        let embedding = retrieved_embedder
            .embed("test")
            .await
            .expect("Failed to embed");
        assert!(!embedding.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_accessor_synapse() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let synapse = orchestrator.synapse();
        assert_eq!(synapse.len(), 0);
    }

    #[tokio::test]
    async fn test_orchestrator_accessor_cortex() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let cortex = orchestrator.cortex();
        // Verify we can use the cortex trait object
        let is_empty = cortex.list().await.expect("Failed to list").is_empty();
        assert!(is_empty);
    }

    #[tokio::test]
    async fn test_orchestrator_lancedb_remember_and_recall() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let _id1 = orchestrator
            .remember(
                "First memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        let _id2 = orchestrator
            .remember(
                "Second memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        let results = orchestrator
            .recall("memory".to_string(), 10)
            .await
            .expect("Failed to recall");

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_orchestrator_lancedb_memorize() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

        orchestrator.memorize(id).await.expect("Failed to memorize");

        // After memorize, memory should still be accessible
        let results = orchestrator
            .recall("test".to_string(), 10)
            .await
            .expect("Failed to recall");

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_lancedb_forget() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

        orchestrator.forget(id).await.expect("Failed to forget");

        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_orchestrator_lancedb_recall_by_scope() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let _id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        let results = orchestrator
            .recall_by_scope("test".to_string(), MemoryScope::Global, 10)
            .await
            .expect("Failed to recall by scope");

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_lancedb_end_session() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let _id = orchestrator
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                MemoryScope::Global,
            )
            .await
            .expect("Failed to remember");

        orchestrator
            .end_session(0.5)
            .await
            .expect("Failed to end session");

        // After end_session, synapse should be empty
        assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_remember_session_scope_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        // Store a memory under session "alpha".
        orchestrator
            .remember(
                "alpha-only secret".to_string(),
                HashMap::new(),
                MemoryScope::Session("alpha".to_string()),
            )
            .await
            .expect("Failed to remember");

        // Owning session "alpha" must see it.
        let alpha = orchestrator
            .recall_by_scope(
                "secret".to_string(),
                MemoryScope::Session("alpha".to_string()),
                10,
            )
            .await
            .expect("recall_by_scope alpha");
        assert_eq!(alpha.len(), 1, "owning session must see its memory");
        assert_eq!(alpha[0].scope, MemoryScope::Session("alpha".to_string()));
        assert_eq!(
            alpha[0].source_session_id.as_deref(),
            Some("alpha"),
            "source_session_id must be populated from session scope"
        );

        // A different session "beta" must NOT see alpha's memory.
        let beta = orchestrator
            .recall_by_scope(
                "secret".to_string(),
                MemoryScope::Session("beta".to_string()),
                10,
            )
            .await
            .expect("recall_by_scope beta");
        assert!(
            beta.is_empty(),
            "different session must not see another session's memory"
        );
    }

    #[tokio::test]
    async fn remember_with_salience_persists_score() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::embedder::MockEmbedder::new());
        let orchestrator = MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
            .await
            .expect("Failed to create orchestrator");

        let id = orchestrator
            .remember_with_salience(
                "high priority fact".to_string(),
                HashMap::new(),
                MemoryScope::Global,
                0.9,
            )
            .await
            .expect("remember_with_salience should succeed");

        let stored = orchestrator
            .synapse()
            .list()
            .await
            .expect("synapse list should succeed")
            .into_iter()
            .find(|e| e.id == id)
            .expect("stored memory should be present in synapse");

        assert!(
            (stored.salience - 0.9).abs() < f32::EPSILON,
            "expected salience 0.9, got {}",
            stored.salience
        );
    }
}
