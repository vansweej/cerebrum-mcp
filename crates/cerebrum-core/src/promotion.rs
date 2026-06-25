//! Memory promotion strategies for automatic Synapse → Cortex promotion.
//!
//! This module provides pluggable promotion strategies that determine when and how
//! memories should be promoted from short-term (Synapse) to long-term (Cortex) storage.

use crate::models::MemoryEntry;

/// Trait for memory promotion strategies.
///
/// Implementations define different criteria for promoting memories from Synapse to Cortex.
pub trait PromotionStrategy: Send + Sync {
    /// Calculate a promotion score for a memory (0.0-1.0).
    ///
    /// Higher scores indicate stronger candidates for promotion.
    /// A score of 1.0 means the memory should definitely be promoted.
    /// A score of 0.0 means the memory should not be promoted.
    fn score(&self, entry: &MemoryEntry, context: &PromotionContext) -> f32;

    /// Get the name of this strategy.
    fn name(&self) -> &str;
}

/// Context information for promotion decisions.
///
/// Provides additional data that promotion strategies may use to make decisions.
#[derive(Clone, Debug)]
pub struct PromotionContext {
    /// Total number of memories in Synapse
    pub synapse_total: usize,
    /// Total number of memories in Cortex
    pub cortex_total: usize,
    /// Average salience of memories in Synapse
    pub avg_salience: f32,
    /// Maximum salience of memories in Synapse
    pub max_salience: f32,
    /// Minimum salience of memories in Synapse
    pub min_salience: f32,
}

/// Frequency-based promotion strategy.
///
/// Promotes memories that have been accessed frequently.
/// Requires `access_count` field in MemoryEntry.
pub struct FrequencyBasedPromotion {
    /// Threshold for access count (memories with count >= threshold are promoted)
    pub threshold: usize,
}

impl FrequencyBasedPromotion {
    /// Create a new frequency-based promotion strategy.
    pub fn new(threshold: usize) -> Self {
        Self { threshold }
    }
}

impl PromotionStrategy for FrequencyBasedPromotion {
    fn score(&self, entry: &MemoryEntry, _context: &PromotionContext) -> f32 {
        // Score based on access count (normalized to 0.0-1.0)
        // Assuming access_count is stored in metadata under "access_count"
        let access_count = entry
            .metadata
            .get("access_count")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        if access_count >= self.threshold {
            // Normalize: score increases with access count
            // Cap at 1.0 for very high access counts
            ((access_count as f32) / (self.threshold as f32 * 2.0)).min(1.0)
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "FrequencyBased"
    }
}

/// Recency-based promotion strategy.
///
/// Promotes memories that have been accessed recently.
/// Uses the `timestamp` field to determine recency.
pub struct RecencyBasedPromotion {
    /// Maximum age in seconds for a memory to be considered recent
    pub max_age_seconds: i64,
}

impl RecencyBasedPromotion {
    /// Create a new recency-based promotion strategy.
    pub fn new(max_age_seconds: i64) -> Self {
        Self { max_age_seconds }
    }
}

impl PromotionStrategy for RecencyBasedPromotion {
    fn score(&self, entry: &MemoryEntry, _context: &PromotionContext) -> f32 {
        use chrono::Utc;

        let now = Utc::now();
        let age_seconds = (now - entry.timestamp).num_seconds();

        if age_seconds <= self.max_age_seconds {
            // Score decreases with age
            // Recent memories get higher scores
            1.0 - (age_seconds as f32 / self.max_age_seconds as f32)
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "RecencyBased"
    }
}

/// Importance-based promotion strategy.
///
/// Promotes memories with high salience scores.
pub struct ImportanceBasedPromotion {
    /// Salience threshold for promotion (0.0-1.0)
    pub threshold: f32,
}

impl ImportanceBasedPromotion {
    /// Create a new importance-based promotion strategy.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }
}

impl PromotionStrategy for ImportanceBasedPromotion {
    fn score(&self, entry: &MemoryEntry, _context: &PromotionContext) -> f32 {
        if entry.salience >= self.threshold {
            entry.salience
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "ImportanceBased"
    }
}

/// Hybrid promotion strategy combining multiple strategies.
///
/// Combines multiple strategies with configurable weights.
pub struct HybridPromotion {
    strategies: Vec<(Box<dyn PromotionStrategy>, f32)>, // (strategy, weight)
}

impl HybridPromotion {
    /// Create a new hybrid promotion strategy.
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Add a strategy with a weight.
    ///
    /// Weights are normalized, so they don't need to sum to 1.0.
    pub fn add_strategy(mut self, strategy: Box<dyn PromotionStrategy>, weight: f32) -> Self {
        self.strategies.push((strategy, weight));
        self
    }
}

impl Default for HybridPromotion {
    fn default() -> Self {
        Self::new()
    }
}

impl PromotionStrategy for HybridPromotion {
    fn score(&self, entry: &MemoryEntry, context: &PromotionContext) -> f32 {
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
        "Hybrid"
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
            tier: MemoryTier::Synapse,
            embedding: None,
            source_session_id: None,
        }
    }

    fn create_test_context() -> PromotionContext {
        PromotionContext {
            synapse_total: 10,
            cortex_total: 50,
            avg_salience: 0.5,
            max_salience: 0.9,
            min_salience: 0.1,
        }
    }

    #[test]
    fn test_frequency_based_promotion_above_threshold() {
        let strategy = FrequencyBasedPromotion::new(5);
        let entry = create_test_entry(0.5, 10);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert!(score > 0.0, "Should promote high-frequency memories");
    }

    #[test]
    fn test_frequency_based_promotion_below_threshold() {
        let strategy = FrequencyBasedPromotion::new(5);
        let entry = create_test_entry(0.5, 2);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert_eq!(score, 0.0, "Should not promote low-frequency memories");
    }

    #[test]
    fn test_importance_based_promotion_above_threshold() {
        let strategy = ImportanceBasedPromotion::new(0.7);
        let entry = create_test_entry(0.8, 0);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert_eq!(score, 0.8, "Should promote high-salience memories");
    }

    #[test]
    fn test_importance_based_promotion_below_threshold() {
        let strategy = ImportanceBasedPromotion::new(0.7);
        let entry = create_test_entry(0.5, 0);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert_eq!(score, 0.0, "Should not promote low-salience memories");
    }

    #[test]
    fn test_recency_based_promotion_recent() {
        let strategy = RecencyBasedPromotion::new(3600); // 1 hour
        let entry = create_test_entry(0.5, 0);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert!(score > 0.9, "Should promote very recent memories");
    }

    #[test]
    fn test_recency_based_promotion_old() {
        let strategy = RecencyBasedPromotion::new(60); // 1 minute
        let mut entry = create_test_entry(0.5, 0);
        // Simulate an old memory by setting timestamp to 2 hours ago
        entry.timestamp = Utc::now() - chrono::Duration::hours(2);
        let context = create_test_context();

        let score = strategy.score(&entry, &context);
        assert_eq!(score, 0.0, "Should not promote old memories");
    }

    #[test]
    fn test_hybrid_promotion_single_strategy() {
        let freq_strategy = Box::new(FrequencyBasedPromotion::new(5));
        let hybrid = HybridPromotion::new().add_strategy(freq_strategy, 1.0);

        let entry = create_test_entry(0.5, 10);
        let context = create_test_context();

        let score = hybrid.score(&entry, &context);
        assert!(score > 0.0, "Hybrid should work with single strategy");
    }

    #[test]
    fn test_hybrid_promotion_multiple_strategies() {
        let freq_strategy = Box::new(FrequencyBasedPromotion::new(5));
        let importance_strategy = Box::new(ImportanceBasedPromotion::new(0.7));

        let hybrid = HybridPromotion::new()
            .add_strategy(freq_strategy, 0.5)
            .add_strategy(importance_strategy, 0.5);

        let entry = create_test_entry(0.8, 10);
        let context = create_test_context();

        let score = hybrid.score(&entry, &context);
        assert!(score > 0.0, "Hybrid should combine multiple strategies");
    }

    #[test]
    fn test_promotion_strategy_name() {
        let freq = FrequencyBasedPromotion::new(5);
        assert_eq!(freq.name(), "FrequencyBased");

        let importance = ImportanceBasedPromotion::new(0.7);
        assert_eq!(importance.name(), "ImportanceBased");

        let recency = RecencyBasedPromotion::new(3600);
        assert_eq!(recency.name(), "RecencyBased");

        let hybrid = HybridPromotion::new();
        assert_eq!(hybrid.name(), "Hybrid");
    }
}
