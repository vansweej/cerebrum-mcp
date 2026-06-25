//! Memory decay strategies for detecting and handling stale memories.
//!
//! This module provides pluggable decay strategies that determine when memories
//! become stale and should be purged or demoted from long-term storage.

use crate::models::MemoryEntry;

/// Trait for memory decay strategies.
///
/// Implementations define different criteria for detecting stale memories
/// that should be purged or demoted.
pub trait DecayStrategy: Send + Sync {
    /// Calculate a decay score for a memory (0.0-1.0).
    ///
    /// Higher scores indicate more stale/decayed memories.
    /// A score of 1.0 means the memory is completely stale and should be purged.
    /// A score of 0.0 means the memory is fresh and should be kept.
    fn score(&self, entry: &MemoryEntry, context: &DecayContext) -> f32;

    /// Get the name of this strategy.
    fn name(&self) -> &str;
}

/// Context information for decay decisions.
///
/// Provides additional data that decay strategies may use to make decisions.
#[derive(Clone, Debug)]
pub struct DecayContext {
    /// Current timestamp for age calculations
    pub current_timestamp: chrono::DateTime<chrono::Utc>,
    /// Average salience of memories in storage
    pub avg_salience: f32,
    /// Maximum salience of memories in storage
    pub max_salience: f32,
    /// Minimum salience of memories in storage
    pub min_salience: f32,
}

/// Time-based decay strategy.
///
/// Decays memories based on their age. Older memories get higher decay scores.
pub struct TimeBasedDecay {
    /// Maximum age in seconds before a memory is considered fully decayed
    pub max_age_seconds: i64,
}

impl TimeBasedDecay {
    /// Create a new time-based decay strategy.
    ///
    /// # Arguments
    /// * `max_age_seconds` - Age in seconds at which decay score reaches 1.0
    pub fn new(max_age_seconds: i64) -> Self {
        Self { max_age_seconds }
    }
}

impl DecayStrategy for TimeBasedDecay {
    fn score(&self, entry: &MemoryEntry, context: &DecayContext) -> f32 {
        let age_seconds = (context.current_timestamp - entry.timestamp).num_seconds();

        if age_seconds <= 0 {
            0.0 // Future timestamp (shouldn't happen)
        } else if age_seconds >= self.max_age_seconds {
            1.0 // Fully decayed
        } else {
            // Linear decay based on age
            age_seconds as f32 / self.max_age_seconds as f32
        }
    }

    fn name(&self) -> &str {
        "TimeBasedDecay"
    }
}

/// Access-based decay strategy.
///
/// Decays memories based on how infrequently they are accessed.
/// Memories with low access counts decay faster.
pub struct AccessBasedDecay {
    /// Minimum access count to avoid decay
    pub min_access_count: usize,
}

impl AccessBasedDecay {
    /// Create a new access-based decay strategy.
    ///
    /// # Arguments
    /// * `min_access_count` - Access count below which decay begins
    pub fn new(min_access_count: usize) -> Self {
        Self { min_access_count }
    }
}

impl DecayStrategy for AccessBasedDecay {
    fn score(&self, entry: &MemoryEntry, _context: &DecayContext) -> f32 {
        let access_count = entry
            .metadata
            .get("access_count")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        if access_count >= self.min_access_count {
            0.0 // No decay for frequently accessed memories
        } else {
            // Decay based on how far below threshold
            let deficit = (self.min_access_count - access_count) as f32;
            let max_deficit = self.min_access_count as f32;
            (deficit / max_deficit).min(1.0)
        }
    }

    fn name(&self) -> &str {
        "AccessBasedDecay"
    }
}

/// Relevance-based decay strategy.
///
/// Decays memories based on their salience relative to the average.
/// Low-salience memories decay faster.
pub struct RelevanceBasedDecay {
    /// Salience threshold below which decay begins
    pub salience_threshold: f32,
}

impl RelevanceBasedDecay {
    /// Create a new relevance-based decay strategy.
    ///
    /// # Arguments
    /// * `salience_threshold` - Salience below which decay begins (0.0-1.0)
    pub fn new(salience_threshold: f32) -> Self {
        Self {
            salience_threshold: salience_threshold.clamp(0.0, 1.0),
        }
    }
}

impl DecayStrategy for RelevanceBasedDecay {
    fn score(&self, entry: &MemoryEntry, _context: &DecayContext) -> f32 {
        if entry.salience >= self.salience_threshold {
            0.0 // No decay for relevant memories
        } else {
            // Decay based on how far below threshold
            let deficit = self.salience_threshold - entry.salience;
            let max_deficit = self.salience_threshold;
            if max_deficit > 0.0 {
                (deficit / max_deficit).min(1.0)
            } else {
                0.0
            }
        }
    }

    fn name(&self) -> &str {
        "RelevanceBasedDecay"
    }
}

/// Hybrid decay strategy combining multiple strategies.
///
/// Combines multiple decay strategies with configurable weights.
pub struct HybridDecay {
    strategies: Vec<(Box<dyn DecayStrategy>, f32)>, // (strategy, weight)
}

impl HybridDecay {
    /// Create a new hybrid decay strategy.
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Add a strategy with a weight.
    ///
    /// Weights are normalized, so they don't need to sum to 1.0.
    pub fn add_strategy(mut self, strategy: Box<dyn DecayStrategy>, weight: f32) -> Self {
        self.strategies.push((strategy, weight));
        self
    }
}

impl Default for HybridDecay {
    fn default() -> Self {
        Self::new()
    }
}

impl DecayStrategy for HybridDecay {
    fn score(&self, entry: &MemoryEntry, context: &DecayContext) -> f32 {
        if self.strategies.is_empty() {
            return 0.0;
        }

        let total_weight: f32 = self.strategies.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        let weighted_score: f32 = self
            .strategies
            .iter()
            .map(|(strategy, weight)| strategy.score(entry, context) * weight)
            .sum();

        weighted_score / total_weight
    }

    fn name(&self) -> &str {
        "HybridDecay"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MemoryId, MemoryTier};
    use chrono::Utc;
    use std::collections::HashMap;

    fn create_test_entry(salience: f32, access_count: usize) -> MemoryEntry {
        let mut metadata = HashMap::new();
        metadata.insert("access_count".to_string(), access_count.to_string());

        MemoryEntry {
            id: MemoryId::new(),
            content: "Test memory".to_string(),
            metadata,
            timestamp: Utc::now(),
            salience,
            tier: MemoryTier::Cortex,
            embedding: None,
            source_session_id: None,
        }
    }

    fn create_test_context() -> DecayContext {
        DecayContext {
            current_timestamp: Utc::now(),
            avg_salience: 0.5,
            max_salience: 0.9,
            min_salience: 0.1,
        }
    }

    #[test]
    fn test_time_based_decay_fresh_memory() {
        let strategy = TimeBasedDecay::new(86400); // 1 day
        let entry = create_test_entry(0.5, 0);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert!(score < 0.01, "Fresh memories should have minimal decay");
    }

    #[test]
    fn test_time_based_decay_old_memory() {
        let strategy = TimeBasedDecay::new(3600); // 1 hour
        let mut entry = create_test_entry(0.5, 0);
        entry.timestamp = Utc::now() - chrono::Duration::hours(2);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert!(score > 0.5, "Old memories should have significant decay");
    }

    #[test]
    fn test_time_based_decay_fully_decayed() {
        let strategy = TimeBasedDecay::new(3600); // 1 hour
        let mut entry = create_test_entry(0.5, 0);
        entry.timestamp = Utc::now() - chrono::Duration::hours(3);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert_eq!(score, 1.0, "Very old memories should be fully decayed");
    }

    #[test]
    fn test_access_based_decay_frequently_accessed() {
        let strategy = AccessBasedDecay::new(5);
        let entry = create_test_entry(0.5, 10);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert_eq!(score, 0.0, "Frequently accessed memories should not decay");
    }

    #[test]
    fn test_access_based_decay_rarely_accessed() {
        let strategy = AccessBasedDecay::new(5);
        let entry = create_test_entry(0.5, 1);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert!(score > 0.0, "Rarely accessed memories should decay");
    }

    #[test]
    fn test_relevance_based_decay_high_salience() {
        let strategy = RelevanceBasedDecay::new(0.7);
        let entry = create_test_entry(0.8, 0);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert_eq!(score, 0.0, "High-salience memories should not decay");
    }

    #[test]
    fn test_relevance_based_decay_low_salience() {
        let strategy = RelevanceBasedDecay::new(0.7);
        let entry = create_test_entry(0.3, 0);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert!(score > 0.0, "Low-salience memories should decay");
    }

    #[test]
    fn test_hybrid_decay_single_strategy() {
        let time_strategy = Box::new(TimeBasedDecay::new(3600));
        let hybrid = HybridDecay::new().add_strategy(time_strategy, 1.0);

        let entry = create_test_entry(0.5, 0);
        let context = create_test_context();

        let score = hybrid.score(&entry, &context);
        assert!(score >= 0.0 && score <= 1.0, "Hybrid should work with single strategy");
    }

    #[test]
    fn test_hybrid_decay_multiple_strategies() {
        let time_strategy = Box::new(TimeBasedDecay::new(3600));
        let access_strategy = Box::new(AccessBasedDecay::new(5));

        let hybrid = HybridDecay::new()
            .add_strategy(time_strategy, 0.5)
            .add_strategy(access_strategy, 0.5);

        let entry = create_test_entry(0.5, 2);
        let context = create_test_context();

        let score = hybrid.score(&entry, &context);
        assert!(score >= 0.0 && score <= 1.0, "Hybrid should combine multiple strategies");
    }

    #[test]
    fn test_decay_strategy_name() {
        let time = TimeBasedDecay::new(3600);
        assert_eq!(time.name(), "TimeBasedDecay");

        let access = AccessBasedDecay::new(5);
        assert_eq!(access.name(), "AccessBasedDecay");

        let relevance = RelevanceBasedDecay::new(0.7);
        assert_eq!(relevance.name(), "RelevanceBasedDecay");

        let hybrid = HybridDecay::new();
        assert_eq!(hybrid.name(), "HybridDecay");
    }
}
