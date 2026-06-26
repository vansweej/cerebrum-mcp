pub mod config;
pub mod decay;
pub mod embedder;
pub mod error;
pub mod fastembed_embedder;
pub mod lancedb_cortex;
pub mod migration;
pub mod models;
pub mod observability;
pub mod orchestrator;
pub mod promotion;
pub mod resilience;
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
pub use resilience::{CircuitBreaker, CircuitBreakerConfig, CircuitState, RetryConfig};
pub use summarization::{
    IdentitySummarizer, KeywordSummarizer, LengthBasedSummarizer, SentenceBasedSummarizer,
    Summarizer,
};
pub use synapse::SynapseMemory;
pub use traits::MemoryStore;
