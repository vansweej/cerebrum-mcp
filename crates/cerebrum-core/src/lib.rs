pub mod cortex;
pub mod decay;
pub mod embedder;
pub mod error;
pub mod models;
pub mod orchestrator;
pub mod promotion;
pub mod synapse;
pub mod traits;
pub mod utils;

// Re-export commonly used types
pub use cortex::CortexMemory;
pub use decay::{AccessBasedDecay, DecayContext, DecayStrategy, HybridDecay, RelevanceBasedDecay, TimeBasedDecay};
pub use embedder::{Embedder, MockEmbedder};
pub use error::{CerebrumError, Result};
pub use models::{MemoryEntry, MemoryId, MemoryTier};
pub use orchestrator::MemoryOrchestrator;
pub use promotion::{
    FrequencyBasedPromotion, HybridPromotion, ImportanceBasedPromotion, PromotionContext,
    PromotionStrategy, RecencyBasedPromotion,
};
pub use synapse::SynapseMemory;
pub use traits::MemoryStore;
