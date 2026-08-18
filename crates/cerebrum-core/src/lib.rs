pub mod config;
pub mod decay;
pub mod embedder;
pub mod ensure_manifest;
pub mod env_overlay;
pub mod error;
pub mod fastembed_embedder;
pub mod input_bound;
pub mod lancedb_cortex;
pub mod manifest;
pub mod migration;
pub mod models;
pub mod observability;
pub mod orchestrator;
pub mod promotion;
pub mod provenance;
pub mod reembed;
pub mod resilience;
pub mod schema_probe;
pub mod summarization;
pub mod synapse;
pub mod traits;
pub mod utils;

// Re-export commonly used types
pub use config::Config;
pub use decay::{
    AccessBasedDecay, DecayContext, DecayStrategy, HybridDecay, RelevanceBasedDecay, TimeBasedDecay,
};
pub use embedder::{Embedder, MockEmbedder};
pub use error::{CerebrumError, Result};
pub use fastembed_embedder::FastEmbedEmbedder;
pub use lancedb_cortex::LanceDBCortex;
pub use migration::{
    EmbeddingMigration, EmbeddingProvenance, HybridMigration, MigrationConfig, MigrationManager,
    MigrationResult, MigrationStrategy, PreserveMigration, ReembedMigration,
};
pub use models::{MemoryEntry, MemoryId, MemoryScope, MemoryTier};
pub use observability::{ObservabilityContext, OperationMetrics, OperationTimer};
pub use orchestrator::MemoryOrchestrator;
pub use promotion::{
    FrequencyBasedPromotion, HybridPromotion, ImportanceBasedPromotion, PromotionContext,
    PromotionStrategy, RecencyBasedPromotion,
};
pub use provenance::{
    parse_project_array, project_array_json, project_weight, status_weight, KEY_CONFIDENCE,
    KEY_PROJECT, KEY_STATUS, KEY_TYPE,
};
pub use resilience::{CircuitBreaker, CircuitBreakerConfig, CircuitState, RetryConfig};
pub use summarization::{
    IdentitySummarizer, KeywordSummarizer, LengthBasedSummarizer, SentenceBasedSummarizer,
    Summarizer,
};
pub use synapse::SynapseMemory;
pub use traits::MemoryStore;
