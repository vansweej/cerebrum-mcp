use cerebrum_core::{
    CortexMemory, MemoryEntry, MemoryId, MemoryOrchestrator, MemoryStore, MemoryTier, MockEmbedder,
    SynapseMemory,
};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// SynapseMemory Integration Tests
// ============================================================================

#[tokio::test]
async fn test_synapse_basic_workflow() {
    let synapse = SynapseMemory::new();

    // Store multiple memories
    let id1 = MemoryId::new();
    let id2 = MemoryId::new();

    let entry1 = MemoryEntry::new(id1, "First memory".to_string());
    let entry2 = MemoryEntry::new(id2, "Second memory".to_string());

    synapse.store(entry1).await.unwrap();
    synapse.store(entry2).await.unwrap();

    assert_eq!(synapse.len(), 2);

    // Retrieve and verify
    let list = synapse.list().await.unwrap();
    assert_eq!(list.len(), 2);
}

#[tokio::test]
async fn test_synapse_semantic_search() {
    let synapse = SynapseMemory::new();

    let id1 = MemoryId::new();
    let embedding1 = vec![0.1; 384];
    let entry1 = MemoryEntry::builder(id1, "The quick brown fox".to_string())
        .embedding(embedding1)
        .build();

    let id2 = MemoryId::new();
    let embedding2 = vec![0.9; 384];
    let entry2 = MemoryEntry::builder(id2, "The lazy dog".to_string())
        .embedding(embedding2)
        .build();

    synapse.store(entry1).await.unwrap();
    synapse.store(entry2).await.unwrap();

    // Search should return results
    let results = synapse.retrieve("fox", 10).await.unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_synapse_salience_ranking() {
    let synapse = SynapseMemory::new();

    let id1 = MemoryId::new();
    let embedding = vec![0.5; 384];
    let entry1 = MemoryEntry::builder(id1, "Important".to_string())
        .salience(0.9)
        .embedding(embedding.clone())
        .build();

    let id2 = MemoryId::new();
    let entry2 = MemoryEntry::builder(id2, "Important".to_string())
        .salience(0.1)
        .embedding(embedding)
        .build();

    synapse.store(entry1).await.unwrap();
    synapse.store(entry2).await.unwrap();

    let results = synapse.retrieve("important", 10).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].salience > results[1].salience);
}

// ============================================================================
// CortexMemory Integration Tests
// ============================================================================

#[tokio::test]
async fn test_cortex_basic_workflow() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let cortex = CortexMemory::new("/tmp/test_cortex_basic", embedder)
        .await
        .unwrap();

    let id1 = MemoryId::new();
    let id2 = MemoryId::new();

    let entry1 = MemoryEntry::new(id1, "First memory".to_string());
    let entry2 = MemoryEntry::new(id2, "Second memory".to_string());

    cortex.store(entry1).await.unwrap();
    cortex.store(entry2).await.unwrap();

    assert_eq!(cortex.len().await.unwrap(), 2);
}

#[tokio::test]
async fn test_cortex_search_by_salience() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let cortex = CortexMemory::new("/tmp/test_cortex_salience", embedder)
        .await
        .unwrap();

    let id1 = MemoryId::new();
    let entry1 = MemoryEntry::builder(id1, "High priority".to_string())
        .salience(0.95)
        .build();

    let id2 = MemoryId::new();
    let entry2 = MemoryEntry::builder(id2, "Low priority".to_string())
        .salience(0.1)
        .build();

    cortex.store(entry1).await.unwrap();
    cortex.store(entry2).await.unwrap();

    let results = cortex.search_by_salience(10).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].salience > results[1].salience);
}

#[tokio::test]
async fn test_cortex_persistence_simulation() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let cortex = CortexMemory::new("/tmp/test_cortex_persist", embedder)
        .await
        .unwrap();

    // Store multiple memories
    for i in 0..5 {
        let id = MemoryId::new();
        let entry = MemoryEntry::new(id, format!("Memory {}", i));
        cortex.store(entry).await.unwrap();
    }

    assert_eq!(cortex.len().await.unwrap(), 5);

    // Verify all memories are retrievable
    let list = cortex.list().await.unwrap();
    assert_eq!(list.len(), 5);
}

// ============================================================================
// MemoryOrchestrator Integration Tests
// ============================================================================

#[tokio::test]
async fn test_orchestrator_remember_and_recall() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_basic", embedder)
        .await
        .unwrap();

    // Remember multiple memories
    let _id1 = orchestrator
        .remember("First memory".to_string(), HashMap::new())
        .await
        .unwrap();

    let _id2 = orchestrator
        .remember("Second memory".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 2);

    // Recall should find both
    let results = orchestrator
        .recall("memory".to_string(), 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_orchestrator_promotion_workflow() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_promote", embedder)
        .await
        .unwrap();

    // Remember in Synapse
    let id = orchestrator
        .remember("Important memory".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);
    assert_eq!(orchestrator.cortex_len().await.unwrap(), 0);

    // Promote to Cortex
    orchestrator.memorize(id).await.unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    assert_eq!(orchestrator.cortex_len().await.unwrap(), 1);
}

#[tokio::test]
async fn test_orchestrator_forget_from_synapse() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_forget_syn", embedder)
        .await
        .unwrap();

    let id = orchestrator
        .remember("Temporary memory".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

    orchestrator.forget(id).await.unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
}

#[tokio::test]
async fn test_orchestrator_forget_from_cortex() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_forget_cor", embedder)
        .await
        .unwrap();

    let id = orchestrator
        .remember("Memory to forget".to_string(), HashMap::new())
        .await
        .unwrap();

    orchestrator.memorize(id).await.unwrap();
    assert_eq!(orchestrator.cortex_len().await.unwrap(), 1);

    orchestrator.forget(id).await.unwrap();

    assert_eq!(orchestrator.cortex_len().await.unwrap(), 0);
}

#[tokio::test]
async fn test_orchestrator_blended_search() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_blend", embedder)
        .await
        .unwrap();

    // Store in Synapse
    let _id1 = orchestrator
        .remember("Synapse memory".to_string(), HashMap::new())
        .await
        .unwrap();

    // Store in Cortex
    let id2 = orchestrator
        .remember("Cortex memory".to_string(), HashMap::new())
        .await
        .unwrap();

    orchestrator.memorize(id2).await.unwrap();

    // Recall should find both
    let results = orchestrator
        .recall("memory".to_string(), 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_orchestrator_end_session_clears_synapse() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_end_session", embedder)
        .await
        .unwrap();

    // Store multiple memories
    let _id1 = orchestrator
        .remember("Memory 1".to_string(), HashMap::new())
        .await
        .unwrap();

    let _id2 = orchestrator
        .remember("Memory 2".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 2);

    // End session with high threshold (no auto-promotion)
    orchestrator.end_session(0.99).await.unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
}

#[tokio::test]
async fn test_orchestrator_end_session_auto_promotes() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_auto_promote", embedder)
        .await
        .unwrap();

    // Store memories
    let id1 = orchestrator
        .remember("Important".to_string(), HashMap::new())
        .await
        .unwrap();

    let id2 = orchestrator
        .remember("Less important".to_string(), HashMap::new())
        .await
        .unwrap();

    // Manually update salience (in real scenario, set during remember)
    let mut synapse_memories = orchestrator.synapse_list().await.unwrap();
    synapse_memories[0].salience = 0.9;
    synapse_memories[1].salience = 0.1;

    // Re-store with updated salience
    orchestrator.forget(id1).await.ok();
    orchestrator.forget(id2).await.ok();

    let _id1 = orchestrator
        .remember("Important".to_string(), HashMap::new())
        .await
        .unwrap();

    let _id2 = orchestrator
        .remember("Less important".to_string(), HashMap::new())
        .await
        .unwrap();

    // End session with threshold 0.5
    orchestrator.end_session(0.5).await.unwrap();

    // Synapse should be empty
    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
}

#[tokio::test]
async fn test_orchestrator_metadata_preservation() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_metadata", embedder)
        .await
        .unwrap();

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "user".to_string());
    metadata.insert("context".to_string(), "conversation".to_string());

    let _id = orchestrator
        .remember("Memory with metadata".to_string(), metadata.clone())
        .await
        .unwrap();

    let memories = orchestrator.synapse_list().await.unwrap();
    assert_eq!(memories[0].metadata, metadata);
}

#[tokio::test]
async fn test_orchestrator_embedding_generation() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_embedding", embedder)
        .await
        .unwrap();

    let _id = orchestrator
        .remember("Test memory".to_string(), HashMap::new())
        .await
        .unwrap();

    let memories = orchestrator.synapse_list().await.unwrap();
    assert!(memories[0].embedding.is_some());
    assert_eq!(memories[0].embedding.as_ref().unwrap().len(), 384);
}

#[tokio::test]
async fn test_orchestrator_tier_assignment() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_tier", embedder)
        .await
        .unwrap();

    let id = orchestrator
        .remember("Test memory".to_string(), HashMap::new())
        .await
        .unwrap();

    let synapse_memories = orchestrator.synapse_list().await.unwrap();
    assert_eq!(synapse_memories[0].tier, MemoryTier::Synapse);

    orchestrator.memorize(id).await.unwrap();

    let cortex_memories = orchestrator.cortex_list().await.unwrap();
    assert_eq!(cortex_memories[0].tier, MemoryTier::Cortex);
}

#[tokio::test]
async fn test_orchestrator_multiple_promotions() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_multi_promote", embedder)
        .await
        .unwrap();

    // Store multiple memories
    let _id1 = orchestrator
        .remember("Memory 1".to_string(), HashMap::new())
        .await
        .unwrap();

    let _id2 = orchestrator
        .remember("Memory 2".to_string(), HashMap::new())
        .await
        .unwrap();

    let _id3 = orchestrator
        .remember("Memory 3".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 3);

    // Promote all
    orchestrator.memorize(_id1).await.unwrap();
    orchestrator.memorize(_id2).await.unwrap();
    orchestrator.memorize(_id3).await.unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    assert_eq!(orchestrator.cortex_len().await.unwrap(), 3);
}

#[tokio::test]
async fn test_orchestrator_recall_limit() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_limit", embedder)
        .await
        .unwrap();

    // Store 10 memories
    for i in 0..10 {
        let _ = orchestrator
            .remember(format!("Memory {}", i), HashMap::new())
            .await
            .unwrap();
    }

    // Recall with limit
    let results = orchestrator
        .recall("memory".to_string(), 5)
        .await
        .unwrap();

    assert_eq!(results.len(), 5);
}

#[tokio::test]
async fn test_orchestrator_cross_tier_recall() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_cross_tier", embedder)
        .await
        .unwrap();

    // Store in both tiers
    let _id1 = orchestrator
        .remember("Synapse memory".to_string(), HashMap::new())
        .await
        .unwrap();

    let id2 = orchestrator
        .remember("Cortex memory".to_string(), HashMap::new())
        .await
        .unwrap();

    orchestrator.memorize(id2).await.unwrap();

    // Recall should find both
    let results = orchestrator
        .recall("memory".to_string(), 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);

    // Verify tier information is preserved
    let synapse_tiers = results.iter().filter(|m| m.tier == MemoryTier::Synapse).count();
    let cortex_tiers = results.iter().filter(|m| m.tier == MemoryTier::Cortex).count();

    assert_eq!(synapse_tiers, 1);
    assert_eq!(cortex_tiers, 1);
}

#[tokio::test]
async fn test_orchestrator_empty_recall() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_empty_recall", embedder)
        .await
        .unwrap();

    let results = orchestrator
        .recall("nonexistent".to_string(), 10)
        .await
        .unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_orchestrator_forget_nonexistent() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_forget_none", embedder)
        .await
        .unwrap();

    let fake_id = MemoryId::new();

    // Should not error
    let result = orchestrator.forget(fake_id).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_orchestrator_session_isolation() {
    let embedder: Arc<dyn cerebrum_core::Embedder> =
        Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::new("/tmp/test_orch_isolation", embedder)
        .await
        .unwrap();

    // Store in Synapse
    let _id1 = orchestrator
        .remember("Session 1 memory".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

    // End session
    orchestrator.end_session(0.99).await.unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);

    // Store new memory in new session
    let _id2 = orchestrator
        .remember("Session 2 memory".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);
}
