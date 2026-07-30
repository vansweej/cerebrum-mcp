use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique identifier for a memory entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// Generate a new random MemoryId.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create a MemoryId from a string representation.
    pub fn from_string(s: &str) -> crate::error::Result<Self> {
        let uuid = Uuid::parse_str(s)
            .map_err(|e| crate::error::CerebrumError::Validation(format!("Invalid UUID: {}", e)))?;
        Ok(Self(uuid))
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
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

/// Designates the scope or visibility of a memory entry.
///
/// Scopes determine who can access and retrieve a memory:
/// - Global: Accessible to all agents and users
/// - User: Accessible only to a specific user
/// - Agent: Accessible only to a specific agent
/// - Session: Accessible only within a specific session
/// - Plan: Addresses a stored plan by id (choragos plan-ref namespace)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryScope {
    /// Global scope: accessible to all agents and users.
    Global,
    /// User scope: accessible only to a specific user.
    User(String),
    /// Agent scope: accessible only to a specific agent.
    Agent(String),
    /// Session scope: accessible only within a specific session.
    Session(String),
    /// Plan scope: addresses a stored plan by id (choragos plan-ref
    /// namespace). Callers fetching a specific plan should use
    /// `exact_scope: true` on `recall_by_scope` to avoid competing against
    /// the global corpus for a limited result window.
    Plan(String),
}

impl MemoryScope {
    /// Check if this scope matches another scope.
    ///
    /// Global scope matches all scopes.
    /// Other scopes match only if they are identical.
    pub fn matches(&self, other: &MemoryScope) -> bool {
        match (self, other) {
            (MemoryScope::Global, _) | (_, MemoryScope::Global) => true,
            (MemoryScope::User(a), MemoryScope::User(b)) => a == b,
            (MemoryScope::Agent(a), MemoryScope::Agent(b)) => a == b,
            (MemoryScope::Session(a), MemoryScope::Session(b)) => a == b,
            (MemoryScope::Plan(a), MemoryScope::Plan(b)) => a == b,
            _ => false,
        }
    }

    /// Get a string representation of the scope.
    pub fn as_str(&self) -> String {
        match self {
            MemoryScope::Global => "global".to_string(),
            MemoryScope::User(id) => format!("user:{}", id),
            MemoryScope::Agent(id) => format!("agent:{}", id),
            MemoryScope::Session(id) => format!("session:{}", id),
            MemoryScope::Plan(id) => format!("plan:{}", id),
        }
    }
}

impl fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A memory paired with its blended-and-weighted rank score.
///
/// Carries the store-computed score (`sim*0.7 + salience*0.3`, multiplied by
/// the provenance status and project weights) across the tier boundary so the
/// orchestrator can merge results from both tiers by a single comparable score
/// instead of re-ranking by salience alone.
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    /// The underlying memory entry.
    pub entry: MemoryEntry,
    /// Blended similarity/salience score after provenance weighting.
    pub score: f32,
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
    /// Scope or visibility of this memory.
    pub scope: MemoryScope,
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
            scope: MemoryScope::Global,
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
            scope: MemoryScope::Global,
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
    scope: MemoryScope,
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

    /// Set the memory scope.
    pub fn scope(mut self, scope: MemoryScope) -> Self {
        self.scope = scope;
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
            scope: self.scope,
        }
    }
}
