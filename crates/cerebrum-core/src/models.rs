use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a memory entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// Generate a new random MemoryId.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Designates which memory tier a memory entry resides in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryTier {
    /// Short-term, volatile, in-memory storage.
    Synapse,
    /// Long-term, persistent, vector-backed storage.
    Cortex,
}

/// A single memory entry with content, metadata, and embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier for this memory.
    pub id: MemoryId,
    /// The text content of the memory.
    pub content: String,
    /// Arbitrary key-value metadata.
    pub metadata: std::collections::HashMap<String, String>,
    /// When this memory was created.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Importance score (0.0–1.0) for ranking and promotion decisions.
    pub salience: f32,
    /// Which tier this memory currently resides in.
    pub tier: MemoryTier,
    /// Cached 384-dimensional embedding vector (BGE-small).
    pub embedding: Option<Vec<f32>>,
    /// Session ID where this memory originated (if applicable).
    pub source_session_id: Option<String>,
}

impl MemoryEntry {
    /// Create a new MemoryEntry with sensible defaults.
    pub fn new(id: MemoryId, content: String) -> Self {
        Self {
            id,
            content,
            metadata: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            salience: 0.5,
            tier: MemoryTier::Synapse,
            embedding: None,
            source_session_id: None,
        }
    }

    /// Create a builder for constructing a MemoryEntry with custom fields.
    pub fn builder(id: MemoryId, content: String) -> MemoryEntryBuilder {
        MemoryEntryBuilder {
            id,
            content,
            metadata: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            salience: 0.5,
            tier: MemoryTier::Synapse,
            embedding: None,
            source_session_id: None,
        }
    }
}

/// Builder for constructing MemoryEntry with custom fields.
pub struct MemoryEntryBuilder {
    id: MemoryId,
    content: String,
    metadata: std::collections::HashMap<String, String>,
    timestamp: chrono::DateTime<chrono::Utc>,
    salience: f32,
    tier: MemoryTier,
    embedding: Option<Vec<f32>>,
    source_session_id: Option<String>,
}

impl MemoryEntryBuilder {
    /// Set the salience score (0.0–1.0).
    pub fn salience(mut self, salience: f32) -> Self {
        self.salience = salience.clamp(0.0, 1.0);
        self
    }

    /// Set the memory tier.
    pub fn tier(mut self, tier: MemoryTier) -> Self {
        self.tier = tier;
        self
    }

    /// Set the embedding vector.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the source session ID.
    pub fn source_session_id(mut self, session_id: String) -> Self {
        self.source_session_id = Some(session_id);
        self
    }

    /// Add a metadata key-value pair.
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set the timestamp.
    pub fn timestamp(mut self, timestamp: chrono::DateTime<chrono::Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Build the MemoryEntry.
    pub fn build(self) -> MemoryEntry {
        MemoryEntry {
            id: self.id,
            content: self.content,
            metadata: self.metadata,
            timestamp: self.timestamp,
            salience: self.salience,
            tier: self.tier,
            embedding: self.embedding,
            source_session_id: self.source_session_id,
        }
    }
}
