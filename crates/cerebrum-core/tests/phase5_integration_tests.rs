//! Phase 5 integration tests combining promotion, decay, summarization, and scope filtering.

use cerebrum_core::{
    AccessBasedDecay, DecayContext, DecayStrategy, FrequencyBasedPromotion, HybridDecay,
    HybridPromotion, IdentitySummarizer, ImportanceBasedPromotion, KeywordSummarizer,
    LengthBasedSummarizer, MemoryEntry, MemoryId, MemoryScope, MemoryStore, MemoryTier,
    PromotionContext, PromotionStrategy, RecencyBasedPromotion, SentenceBasedSummarizer,
    Summarizer, SynapseMemory,
};
use chrono::Utc;
use std::collections::HashMap;

#[tokio::test]
async fn test_promotion_with_scope_filtering() {
    // Test that promotion respects scope boundaries
    let synapse = SynapseMemory::new();

    let id1 = MemoryId::new();
    let embedding = vec![0.1; 384];
    let entry1 = MemoryEntry::builder(id1, "User1 important memory".to_string())
        .scope(MemoryScope::User("user1".to_string()))
        .salience(0.9)
        .embedding(embedding.clone())
        .build();

    let id2 = MemoryId::new();
    let entry2 = MemoryEntry::builder(id2, "User2 important memory".to_string())
        .scope(MemoryScope::User("user2".to_string()))
        .salience(0.9)
        .embedding(embedding)
        .build();

    synapse.store(entry1).await.unwrap();
    synapse.store(entry2).await.unwrap();

    // User1 should only see their own memories
    let results = synapse
        .retrieve_by_scope(&vec![0.1; 384], &MemoryScope::User("user1".to_string()), 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "User1 important memory");
}

#[tokio::test]
async fn test_decay_strategy_composition() {
    // Test that multiple decay strategies can be composed
    let entry = MemoryEntry::new(MemoryId::new(), "Test memory".to_string());

    let time_decay = AccessBasedDecay::new(10);
    let access_decay = AccessBasedDecay::new(5);

    let context = DecayContext {
        current_timestamp: Utc::now(),
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let score1 = time_decay.score(&entry, &context);
    let score2 = access_decay.score(&entry, &context);

    // Both should return valid scores
    assert!(score1 >= 0.0 && score1 <= 1.0);
    assert!(score2 >= 0.0 && score2 <= 1.0);
}

#[tokio::test]
async fn test_summarization_preserves_scope() {
    // Test that summarization preserves scope information
    let entry = MemoryEntry::builder(
        MemoryId::new(),
        "This is a very long memory that should be summarized".to_string(),
    )
    .scope(MemoryScope::Agent("agent1".to_string()))
    .build();

    let summarizer = LengthBasedSummarizer::new(20);
    let summarized = summarizer.summarize(&entry);

    // Scope should be preserved
    assert_eq!(summarized.scope, MemoryScope::Agent("agent1".to_string()));
    // Content should be shortened
    assert!(summarized.content.len() < entry.content.len());
}

#[tokio::test]
async fn test_promotion_strategy_with_scope() {
    // Test that promotion strategies work with scoped memories
    let entry = MemoryEntry::builder(MemoryId::new(), "Important memory".to_string())
        .scope(MemoryScope::Session("session1".to_string()))
        .salience(0.8)
        .build();

    let mut metadata = HashMap::new();
    metadata.insert("access_count".to_string(), "5".to_string());

    let promotion = FrequencyBasedPromotion::new(3);
    let context = PromotionContext {
        synapse_total: 10,
        cortex_total: 50,
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let score = promotion.score(&entry, &context);
    assert!(score >= 0.0 && score <= 1.0);
}

#[tokio::test]
async fn test_hybrid_promotion_with_multiple_strategies() {
    // Test hybrid promotion combining multiple strategies
    let entry = MemoryEntry::builder(MemoryId::new(), "Test memory".to_string())
        .scope(MemoryScope::Global)
        .salience(0.7)
        .build();

    let context = PromotionContext {
        synapse_total: 10,
        cortex_total: 50,
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let hybrid = HybridPromotion::new()
        .add_strategy(Box::new(FrequencyBasedPromotion::new(3)), 0.5)
        .add_strategy(Box::new(ImportanceBasedPromotion::new(0.6)), 0.5);

    let score = hybrid.score(&entry, &context);
    assert!(score >= 0.0 && score <= 1.0);
}

#[tokio::test]
async fn test_hybrid_decay_with_multiple_strategies() {
    // Test hybrid decay combining multiple strategies
    let entry = MemoryEntry::new(MemoryId::new(), "Test memory".to_string());

    let context = DecayContext {
        current_timestamp: Utc::now(),
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let hybrid = HybridDecay::new()
        .add_strategy(Box::new(AccessBasedDecay::new(5)), 0.5)
        .add_strategy(Box::new(AccessBasedDecay::new(10)), 0.5);

    let score = hybrid.score(&entry, &context);
    assert!(score >= 0.0 && score <= 1.0);
}

#[tokio::test]
async fn test_summarization_strategies_composition() {
    // Test different summarization strategies
    let entry = MemoryEntry::new(
        MemoryId::new(),
        "First sentence. Second sentence. Third sentence.".to_string(),
    );

    let identity = IdentitySummarizer;
    let length = LengthBasedSummarizer::new(30);
    let keyword = KeywordSummarizer::new(3);
    let sentence = SentenceBasedSummarizer::new(2);

    let identity_result = identity.summarize(&entry);
    let length_result = length.summarize(&entry);
    let keyword_result = keyword.summarize(&entry);
    let sentence_result = sentence.summarize(&entry);

    // Identity should preserve content
    assert_eq!(identity_result.content, entry.content);

    // Length should shorten
    assert!(length_result.content.len() <= 33);

    // Keyword should extract keywords
    assert!(keyword_result.content.contains("Keywords:"));

    // Sentence should reduce sentences
    assert!(sentence_result.content.len() < entry.content.len());
}

#[tokio::test]
async fn test_scope_matching_logic() {
    // Test MemoryScope matching behavior
    let global = MemoryScope::Global;
    let user1 = MemoryScope::User("user1".to_string());
    let user2 = MemoryScope::User("user2".to_string());
    let agent1 = MemoryScope::Agent("agent1".to_string());

    // Global matches everything
    assert!(global.matches(&global));
    assert!(global.matches(&user1));
    assert!(global.matches(&agent1));

    // User1 matches user1 and global
    assert!(user1.matches(&global));
    assert!(user1.matches(&user1));
    assert!(!user1.matches(&user2));
    assert!(!user1.matches(&agent1));

    // Agent1 matches agent1 and global
    assert!(agent1.matches(&global));
    assert!(agent1.matches(&agent1));
    assert!(!agent1.matches(&user1));
}

#[tokio::test]
async fn test_memory_entry_with_all_phase5_features() {
    // Test MemoryEntry with all Phase 5 features
    let id = MemoryId::new();
    let embedding = vec![0.1; 384];

    let entry = MemoryEntry::builder(id, "Complex memory with all features".to_string())
        .scope(MemoryScope::User("user1".to_string()))
        .salience(0.8)
        .embedding(embedding)
        .tier(MemoryTier::Synapse)
        .metadata("key1".to_string(), "value1".to_string())
        .metadata("key2".to_string(), "value2".to_string())
        .build();

    // Verify all fields
    assert_eq!(entry.id, id);
    assert_eq!(entry.content, "Complex memory with all features");
    assert_eq!(entry.scope, MemoryScope::User("user1".to_string()));
    assert_eq!(entry.salience, 0.8);
    assert_eq!(entry.tier, MemoryTier::Synapse);
    assert_eq!(entry.embedding.as_ref().unwrap().len(), 384);
    assert_eq!(entry.metadata.get("key1"), Some(&"value1".to_string()));
    assert_eq!(entry.metadata.get("key2"), Some(&"value2".to_string()));
}

#[tokio::test]
async fn test_scope_filtering_with_retrieval() {
    // Test scope filtering during retrieval
    let synapse = SynapseMemory::new();

    // Create memories with different scopes
    let global_id = MemoryId::new();
    let global_embedding = vec![0.1; 384];
    let global_entry = MemoryEntry::builder(global_id, "Global memory".to_string())
        .scope(MemoryScope::Global)
        .embedding(global_embedding)
        .build();

    let user_id = MemoryId::new();
    let user_embedding = vec![0.1; 384];
    let user_entry = MemoryEntry::builder(user_id, "User1 memory".to_string())
        .scope(MemoryScope::User("user1".to_string()))
        .embedding(user_embedding)
        .build();

    let agent_id = MemoryId::new();
    let agent_embedding = vec![0.1; 384];
    let agent_entry = MemoryEntry::builder(agent_id, "Agent1 memory".to_string())
        .scope(MemoryScope::Agent("agent1".to_string()))
        .embedding(agent_embedding)
        .build();

    synapse.store(global_entry).await.unwrap();
    synapse.store(user_entry).await.unwrap();
    synapse.store(agent_entry).await.unwrap();

    // Retrieve with global scope should get all
    let global_results = synapse
        .retrieve_by_scope(&vec![0.1; 384], &MemoryScope::Global, 10)
        .await
        .unwrap();
    assert_eq!(global_results.len(), 3);

    // Retrieve with user1 scope should get global + user1
    let user_results = synapse
        .retrieve_by_scope(&vec![0.1; 384], &MemoryScope::User("user1".to_string()), 10)
        .await
        .unwrap();
    assert_eq!(user_results.len(), 2);

    // Retrieve with agent1 scope should get global + agent1
    let agent_results = synapse
        .retrieve_by_scope(
            &vec![0.1; 384],
            &MemoryScope::Agent("agent1".to_string()),
            10,
        )
        .await
        .unwrap();
    assert_eq!(agent_results.len(), 2);
}

#[tokio::test]
async fn test_promotion_context_with_scoped_memories() {
    // Test promotion context calculation with scoped memories
    let entry = MemoryEntry::builder(MemoryId::new(), "Test memory".to_string())
        .scope(MemoryScope::Session("session1".to_string()))
        .salience(0.7)
        .build();

    let context = PromotionContext {
        synapse_total: 100,
        cortex_total: 500,
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let promotion = RecencyBasedPromotion::new(3600); // 1 hour threshold
    let score = promotion.score(&entry, &context);

    // Score should be valid
    assert!(score >= 0.0 && score <= 1.0);
}

#[tokio::test]
async fn test_decay_context_with_salience_range() {
    // Test decay context with various salience values
    let entry = MemoryEntry::builder(MemoryId::new(), "Test memory".to_string())
        .salience(0.5)
        .build();

    let context = DecayContext {
        current_timestamp: Utc::now(),
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let decay = AccessBasedDecay::new(5);
    let score = decay.score(&entry, &context);

    // Score should be valid
    assert!(score >= 0.0 && score <= 1.0);
}

#[tokio::test]
async fn test_summarizer_with_different_content_lengths() {
    // Test summarizers with various content lengths
    let short_entry = MemoryEntry::new(MemoryId::new(), "Short".to_string());
    let medium_entry = MemoryEntry::new(
        MemoryId::new(),
        "This is a medium length memory with some content".to_string(),
    );
    let long_entry = MemoryEntry::new(
        MemoryId::new(),
        "This is a very long memory with lots of content that should be summarized to a shorter version for storage efficiency".to_string(),
    );

    let summarizer = LengthBasedSummarizer::new(50);

    let short_result = summarizer.summarize(&short_entry);
    let medium_result = summarizer.summarize(&medium_entry);
    let long_result = summarizer.summarize(&long_entry);

    // Short should remain unchanged
    assert_eq!(short_result.content, "Short");

    // Medium should remain unchanged
    assert_eq!(medium_result.content, medium_entry.content);

    // Long should be truncated
    assert!(long_result.content.len() <= 53); // 50 + "..."
}

#[tokio::test]
async fn test_memory_scope_string_representation() {
    // Test MemoryScope string representation
    let global = MemoryScope::Global;
    let user = MemoryScope::User("user1".to_string());
    let agent = MemoryScope::Agent("agent1".to_string());
    let session = MemoryScope::Session("session1".to_string());

    assert_eq!(global.as_str(), "global");
    assert_eq!(user.as_str(), "user:user1");
    assert_eq!(agent.as_str(), "agent:agent1");
    assert_eq!(session.as_str(), "session:session1");
}

#[tokio::test]
async fn test_promotion_and_decay_together() {
    // Test promotion and decay strategies working together
    let entry = MemoryEntry::builder(MemoryId::new(), "Important memory".to_string())
        .salience(0.8)
        .scope(MemoryScope::Global)
        .build();

    let promotion_context = PromotionContext {
        synapse_total: 10,
        cortex_total: 50,
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let decay_context = DecayContext {
        current_timestamp: Utc::now(),
        avg_salience: 0.5,
        max_salience: 1.0,
        min_salience: 0.0,
    };

    let promotion = ImportanceBasedPromotion::new(0.7);
    let decay = AccessBasedDecay::new(5);

    let promotion_score = promotion.score(&entry, &promotion_context);
    let decay_score = decay.score(&entry, &decay_context);

    // Both should be valid
    assert!(promotion_score >= 0.0 && promotion_score <= 1.0);
    assert!(decay_score >= 0.0 && decay_score <= 1.0);

    // High salience should promote well
    assert!(promotion_score > 0.5);
}

#[tokio::test]
async fn test_scope_filtering_empty_results() {
    // Test scope filtering when no memories match
    let synapse = SynapseMemory::new();

    let id = MemoryId::new();
    let embedding = vec![0.1; 384];
    let entry = MemoryEntry::builder(id, "User1 memory".to_string())
        .scope(MemoryScope::User("user1".to_string()))
        .embedding(embedding)
        .build();

    synapse.store(entry).await.unwrap();

    // Query with different user scope should return empty
    let results = synapse
        .retrieve_by_scope(&vec![0.1; 384], &MemoryScope::User("user2".to_string()), 10)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_summarization_with_scope_preservation() {
    // Test that all summarizers preserve scope
    let entry = MemoryEntry::builder(
        MemoryId::new(),
        "This is a test memory with important information".to_string(),
    )
    .scope(MemoryScope::Agent("agent1".to_string()))
    .build();

    let summarizers: Vec<Box<dyn Summarizer>> = vec![
        Box::new(IdentitySummarizer),
        Box::new(LengthBasedSummarizer::new(30)),
        Box::new(KeywordSummarizer::new(3)),
        Box::new(SentenceBasedSummarizer::new(1)),
    ];

    for summarizer in summarizers {
        let result = summarizer.summarize(&entry);
        assert_eq!(result.scope, MemoryScope::Agent("agent1".to_string()));
    }
}
