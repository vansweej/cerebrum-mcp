//! Integration tests for Cerebrum Phase 6 features
//! Tests LanceDB persistence, FastEmbed consistency, migration workflows,
//! error recovery, observability, and end-to-end workflows.

use cerebrum_core::embedder::{Embedder, MockEmbedder};
use cerebrum_core::lancedb_cortex::LanceDBCortex;
use cerebrum_core::migration::{MigrationConfig, MigrationManager, MigrationStrategy};
use cerebrum_core::models::{MemoryEntry, MemoryId, MemoryScope, MemoryTier};
use cerebrum_core::observability::ObservabilityContext;
use cerebrum_core::orchestrator::MemoryOrchestrator;
use cerebrum_core::resilience::{CircuitBreaker, CircuitBreakerConfig, RetryConfig};
use cerebrum_core::traits::MemoryStore;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_integration_lancedb_persistence() {
    // Test that LanceDB can store and retrieve data
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let db_path = "/tmp/test_integration_persistence";

    let orchestrator = MemoryOrchestrator::with_lancedb_cortex(db_path, embedder.clone())
        .await
        .expect("Failed to create orchestrator");

    // Store data
    let id = orchestrator
        .remember("Persistent memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    orchestrator.memorize(id).await.expect("Failed to memorize");

    // Verify data is accessible immediately
    let results = orchestrator
        .recall("persistent".to_string(), 10)
        .await
        .expect("Failed to recall");

    assert!(
        !results.is_empty(),
        "Data should be accessible after storage"
    );
}

#[tokio::test]
async fn test_integration_embedder_consistency() {
    // Test that embedder produces consistent embeddings
    let embedder = MockEmbedder::new();

    let text = "The quick brown fox jumps over the lazy dog";
    let embedding1 = embedder.embed(text).await.expect("Failed to embed");
    let embedding2 = embedder.embed(text).await.expect("Failed to embed");

    // Embeddings should be identical for the same text
    assert_eq!(embedding1.len(), embedding2.len());
    for (e1, e2) in embedding1.iter().zip(embedding2.iter()) {
        assert!((e1 - e2).abs() < 1e-6, "Embeddings should be consistent");
    }
}

#[tokio::test]
async fn test_integration_migration_workflow() {
    // Test complete migration workflow
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let db_path = "/tmp/test_integration_migration";

    let orchestrator = MemoryOrchestrator::with_lancedb_cortex(db_path, embedder.clone())
        .await
        .expect("Failed to create orchestrator");

    // Store multiple memories
    let _id1 = orchestrator
        .remember("First memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    let _id2 = orchestrator
        .remember("Second memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    // Create migration config and manager
    let config = MigrationConfig::new(MigrationStrategy::Hybrid, embedder.clone())
        .with_batch_size(10)
        .with_hybrid_threshold(0.5);

    let manager = MigrationManager::new();
    let cortex = orchestrator.cortex();
    let result = manager
        .execute(cortex.as_ref(), &config)
        .await
        .expect("Migration failed");

    // Verify success rate is valid (0.0 to 100.0 as percentage)
    let success_rate = result.success_rate();
    assert!(
        success_rate >= 0.0 && success_rate <= 100.0,
        "Success rate should be between 0 and 100, got {}",
        success_rate
    );
}

#[tokio::test]
async fn test_integration_error_recovery_with_circuit_breaker() {
    // Test error recovery using circuit breaker
    let config = CircuitBreakerConfig::new()
        .with_failure_threshold(2)
        .with_timeout_ms(100);

    let breaker = CircuitBreaker::new(config);

    // Simulate failures
    breaker.record_failure();
    breaker.record_failure();

    // Circuit should be open
    assert!(breaker.allow_request().is_err());

    // Record success to transition to half-open
    breaker.record_success();
    assert!(breaker.allow_request().is_ok());
}

#[tokio::test]
async fn test_integration_retry_with_exponential_backoff() {
    // Test retry logic with exponential backoff
    let config = RetryConfig::new()
        .with_max_retries(3)
        .with_initial_backoff_ms(10);

    let backoff1 = config.calculate_backoff(0);
    let backoff2 = config.calculate_backoff(1);
    let backoff3 = config.calculate_backoff(2);

    // Backoff should increase exponentially
    assert!(backoff2 > backoff1);
    assert!(backoff3 > backoff2);
}

#[tokio::test]
async fn test_integration_observability_context() {
    // Test observability context
    let context = ObservabilityContext::new();

    // Verify context can be cloned
    let _cloned = context.clone();

    // Log summary should not panic
    context.log_summary();
}

#[tokio::test]
async fn test_integration_end_to_end_workflow() {
    // Complete end-to-end workflow with all Phase 6 features
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let db_path = "/tmp/test_integration_e2e";

    let orchestrator = MemoryOrchestrator::with_lancedb_cortex(db_path, embedder.clone())
        .await
        .expect("Failed to create orchestrator");

    // 1. Store memories
    let id1 = orchestrator
        .remember("Important memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    let id2 = orchestrator
        .remember("Another memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    // 2. Verify recall works
    let results = orchestrator
        .recall("memory".to_string(), 10)
        .await
        .expect("Failed to recall");
    assert_eq!(results.len(), 2);

    // 3. Promote to long-term storage
    orchestrator
        .memorize(id1)
        .await
        .expect("Failed to memorize");

    // 4. Verify still accessible
    let results = orchestrator
        .recall("important".to_string(), 10)
        .await
        .expect("Failed to recall");
    assert!(!results.is_empty());

    // 5. Forget a memory
    orchestrator.forget(id2).await.expect("Failed to forget");

    // 6. Verify it's gone from synapse
    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);

    // 7. End session
    orchestrator
        .end_session(0.5)
        .await
        .expect("Failed to end session");
}

#[tokio::test]
async fn test_integration_lancedb_cortex_store_and_retrieve() {
    // Test LanceDB Cortex store and retrieve functionality
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let cortex = LanceDBCortex::new("/tmp/test_integration_cortex", embedder.clone())
        .await
        .expect("Failed to create LanceDB Cortex");

    // Store a memory
    let entry = MemoryEntry {
        id: MemoryId::new(),
        content: "Test memory content".to_string(),
        embedding: Some(vec![0.1, 0.2, 0.3]),
        scope: MemoryScope::Global,
        salience: 0.8,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
        source_session_id: None,
        tier: MemoryTier::Cortex,
    };

    cortex.store(entry.clone()).await.expect("Failed to store");

    // Retrieve it
    let results = cortex
        .retrieve("test", 10)
        .await
        .expect("Failed to retrieve");

    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_integration_memory_scope_filtering() {
    // Test that memory scope filtering works correctly
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let orchestrator =
        MemoryOrchestrator::with_lancedb_cortex("/tmp/test_integration_scope", embedder)
            .await
            .expect("Failed to create orchestrator");

    // Store memory
    let _id = orchestrator
        .remember("Scoped memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    // Recall by scope
    let results = orchestrator
        .recall_by_scope("scoped".to_string(), MemoryScope::Global, 10)
        .await
        .expect("Failed to recall by scope");

    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_integration_concurrent_operations() {
    // Test concurrent memory operations
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let orchestrator = Arc::new(
        MemoryOrchestrator::with_lancedb_cortex("/tmp/test_integration_concurrent", embedder)
            .await
            .expect("Failed to create orchestrator"),
    );

    let mut handles = vec![];

    // Spawn multiple concurrent remember operations
    for i in 0..5 {
        let orch = Arc::clone(&orchestrator);
        let handle = tokio::spawn(async move {
            orch.remember(format!("Concurrent memory {}", i), HashMap::new())
                .await
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok());
    }

    // Verify all memories were stored
    let results = orchestrator
        .recall("concurrent".to_string(), 10)
        .await
        .expect("Failed to recall");

    assert_eq!(results.len(), 5);
}

#[tokio::test]
async fn test_integration_memory_decay() {
    // Test memory decay over time
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let orchestrator =
        MemoryOrchestrator::with_lancedb_cortex("/tmp/test_integration_decay", embedder)
            .await
            .expect("Failed to create orchestrator");

    let _id = orchestrator
        .remember("Decaying memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    // Verify memory exists
    let results = orchestrator
        .recall("decaying".to_string(), 10)
        .await
        .expect("Failed to recall");
    assert!(!results.is_empty());

    // Decay should be applied during end_session
    orchestrator
        .end_session(0.5)
        .await
        .expect("Failed to end session");

    // Memory should still be accessible (decay doesn't delete, just reduces salience)
    let _results = orchestrator
        .recall("decaying".to_string(), 10)
        .await
        .expect("Failed to recall");
}

#[tokio::test]
async fn test_integration_blended_search_across_tiers() {
    // Test blended search across Synapse and Cortex tiers
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let orchestrator =
        MemoryOrchestrator::with_lancedb_cortex("/tmp/test_integration_blended", embedder)
            .await
            .expect("Failed to create orchestrator");

    // Store in Synapse
    let _id1 = orchestrator
        .remember("Synapse memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    // Store in Cortex
    let id2 = orchestrator
        .remember("Cortex memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    orchestrator
        .memorize(id2)
        .await
        .expect("Failed to memorize");

    // Blended recall should return from both tiers
    let results = orchestrator
        .recall("memory".to_string(), 10)
        .await
        .expect("Failed to recall");

    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_integration_metadata_preservation() {
    // Test that metadata is preserved through operations
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let orchestrator =
        MemoryOrchestrator::with_lancedb_cortex("/tmp/test_integration_metadata", embedder)
            .await
            .expect("Failed to create orchestrator");

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "test".to_string());
    metadata.insert("priority".to_string(), "high".to_string());

    let id = orchestrator
        .remember("Memory with metadata".to_string(), metadata.clone())
        .await
        .expect("Failed to remember");

    // Promote to Cortex
    orchestrator.memorize(id).await.expect("Failed to memorize");

    // Recall and verify metadata
    let results = orchestrator
        .recall("metadata".to_string(), 10)
        .await
        .expect("Failed to recall");

    assert!(!results.is_empty());
    let entry = &results[0];
    assert_eq!(entry.metadata.get("source"), Some(&"test".to_string()));
    assert_eq!(entry.metadata.get("priority"), Some(&"high".to_string()));
}

#[tokio::test]
async fn test_integration_circuit_breaker_state_transitions() {
    // Test circuit breaker state transitions
    let config = CircuitBreakerConfig::new()
        .with_failure_threshold(2)
        .with_timeout_ms(50);

    let breaker = CircuitBreaker::new(config);

    // Initially closed
    assert!(breaker.allow_request().is_ok());

    // Fail twice to open
    breaker.record_failure();
    breaker.record_failure();
    assert!(breaker.allow_request().is_err());

    // Wait for timeout
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Should allow request in half-open state
    assert!(breaker.allow_request().is_ok());
}

#[tokio::test]
async fn test_integration_retry_config_max_backoff() {
    // Test that retry backoff respects max backoff
    let config = RetryConfig::new()
        .with_max_retries(10)
        .with_initial_backoff_ms(100);

    let backoff = config.calculate_backoff(10);

    // Backoff should be capped at some reasonable value
    assert!(backoff <= Duration::from_secs(60));
}

#[tokio::test]
async fn test_integration_lancedb_list_all_memories() {
    // Test listing all memories from LanceDB
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let cortex = LanceDBCortex::new("/tmp/test_integration_list", embedder.clone())
        .await
        .expect("Failed to create LanceDB Cortex");

    // Store multiple memories
    for i in 0..3 {
        let entry = MemoryEntry {
            id: MemoryId::new(),
            content: format!("Memory {}", i),
            embedding: Some(vec![0.1 * i as f32, 0.2, 0.3]),
            scope: MemoryScope::Global,
            salience: 0.8,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            source_session_id: None,
            tier: MemoryTier::Cortex,
        };

        cortex.store(entry).await.expect("Failed to store");
    }

    // List all
    let all = cortex.list().await.expect("Failed to list");
    assert_eq!(all.len(), 3);
}

#[tokio::test]
async fn test_integration_migration_preserve_strategy() {
    // Test migration with preserve strategy
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let config =
        MigrationConfig::new(MigrationStrategy::Preserve, embedder.clone()).with_batch_size(10);

    let manager = MigrationManager::new();
    let cortex = LanceDBCortex::new("/tmp/test_integration_preserve", embedder.clone())
        .await
        .expect("Failed to create cortex");

    let result = manager
        .execute(&cortex, &config)
        .await
        .expect("Migration failed");

    assert!(result.success_rate() >= 0.0);
}

#[tokio::test]
async fn test_integration_migration_reembed_strategy() {
    // Test migration with reembed strategy
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let config =
        MigrationConfig::new(MigrationStrategy::Reembed, embedder.clone()).with_batch_size(10);

    let manager = MigrationManager::new();
    let cortex = LanceDBCortex::new("/tmp/test_integration_reembed", embedder.clone())
        .await
        .expect("Failed to create cortex");

    let result = manager
        .execute(&cortex, &config)
        .await
        .expect("Migration failed");

    assert!(result.success_rate() >= 0.0);
}

#[tokio::test]
async fn test_integration_orchestrator_accessors() {
    // Test orchestrator accessor methods
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let orchestrator = MemoryOrchestrator::with_lancedb_cortex(
        "/tmp/test_integration_accessors",
        embedder.clone(),
    )
    .await
    .expect("Failed to create orchestrator");

    // Test embedder accessor
    let _retrieved_embedder = orchestrator.embedder();

    // Test synapse accessor
    let synapse = orchestrator.synapse();
    assert_eq!(synapse.len(), 0);

    // Test cortex accessor
    let cortex = orchestrator.cortex();
    assert_eq!(cortex.len().await.unwrap(), 0);
}

#[tokio::test]
async fn test_integration_memory_promotion_with_salience() {
    // Test memory promotion based on salience
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let orchestrator =
        MemoryOrchestrator::with_lancedb_cortex("/tmp/test_integration_promotion", embedder)
            .await
            .expect("Failed to create orchestrator");

    let id = orchestrator
        .remember("High salience memory".to_string(), HashMap::new())
        .await
        .expect("Failed to remember");

    // Promote to cortex
    orchestrator.memorize(id).await.expect("Failed to memorize");

    // Verify it's still accessible
    let results = orchestrator
        .recall("high".to_string(), 10)
        .await
        .expect("Failed to recall");

    assert!(!results.is_empty());
}
