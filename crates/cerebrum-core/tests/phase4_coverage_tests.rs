//! Phase 4: Comprehensive Coverage Tests
//!
//! Tests for edge cases, concurrent access patterns, large-scale operations,
//! and boundary conditions to ensure production readiness.

use cerebrum_core::embedder::{Embedder, MockEmbedder};
use cerebrum_core::models::{MemoryEntry, MemoryId, MemoryScope};
use cerebrum_core::observability::OperationMetrics;
use cerebrum_core::orchestrator::MemoryOrchestrator;
use cerebrum_core::resilience::{CircuitBreaker, CircuitBreakerConfig};
use cerebrum_core::synapse::SynapseMemory;
use cerebrum_core::traits::MemoryStore;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::task::JoinHandle;

// ============================================================================
// EDGE CASE TESTS - Embedder Failures
// ============================================================================

#[tokio::test]
async fn test_embedder_with_empty_text() {
    let embedder = Arc::new(MockEmbedder::new());
    // MockEmbedder returns error for empty text
    let result = embedder.embed("").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_embedder_with_very_long_text() {
    let embedder = Arc::new(MockEmbedder::new());
    let long_text = "a".repeat(10_000);
    let embedding = embedder.embed(&long_text).await.unwrap();
    assert_eq!(embedding.len(), 384);
    assert!(embedding.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[tokio::test]
async fn test_embedder_with_unicode_text() {
    let embedder = Arc::new(MockEmbedder::new());
    let unicode_text = "Hello 世界 🌍 مرحبا мир";
    let embedding = embedder.embed(unicode_text).await.unwrap();
    assert_eq!(embedding.len(), 384);
    assert!(embedding.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[tokio::test]
async fn test_embedder_with_special_characters() {
    let embedder = Arc::new(MockEmbedder::new());
    let special_text = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
    let embedding = embedder.embed(special_text).await.unwrap();
    assert_eq!(embedding.len(), 384);
    assert!(embedding.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[tokio::test]
async fn test_embedder_deterministic_output() {
    let embedder = Arc::new(MockEmbedder::new());
    let text = "test deterministic output";
    let embedding1 = embedder.embed(text).await.unwrap();
    let embedding2 = embedder.embed(text).await.unwrap();
    assert_eq!(embedding1, embedding2);
}

#[tokio::test]
async fn test_embedder_different_inputs_different_outputs() {
    let embedder = Arc::new(MockEmbedder::new());
    let embedding1 = embedder.embed("text one").await.unwrap();
    let embedding2 = embedder.embed("text two").await.unwrap();
    assert_ne!(embedding1, embedding2);
}

// ============================================================================
// BOUNDARY CONDITION TESTS - Circuit Breaker
// ============================================================================

#[test]
fn test_circuit_breaker_exactly_at_failure_threshold() {
    let config = CircuitBreakerConfig::new().with_failure_threshold(5);
    let cb = CircuitBreaker::new(config);

    // Record exactly 5 failures
    for _ in 0..5 {
        cb.record_failure();
    }

    // Circuit should now be open
    assert!(cb.allow_request().is_err());
}

#[test]
fn test_circuit_breaker_just_below_failure_threshold() {
    let config = CircuitBreakerConfig::new().with_failure_threshold(5);
    let cb = CircuitBreaker::new(config);

    // Record 4 failures (one less than threshold)
    for _ in 0..4 {
        cb.record_failure();
    }

    // Circuit should still allow requests
    assert!(cb.allow_request().is_ok());
}

#[test]
fn test_circuit_breaker_just_above_failure_threshold() {
    let config = CircuitBreakerConfig::new().with_failure_threshold(5);
    let cb = CircuitBreaker::new(config);

    // Record 6 failures (one more than threshold)
    for _ in 0..6 {
        cb.record_failure();
    }

    // Circuit should be open
    assert!(cb.allow_request().is_err());
}

#[test]
fn test_circuit_breaker_success_resets_failure_count() {
    let config = CircuitBreakerConfig::new().with_failure_threshold(5);
    let cb = CircuitBreaker::new(config);

    // Record 3 failures
    for _ in 0..3 {
        cb.record_failure();
    }

    // Record a success
    cb.record_success();

    // Record 3 more failures (total 6, but counter was reset)
    for _ in 0..3 {
        cb.record_failure();
    }

    // Circuit should still allow requests (only 3 failures since reset)
    assert!(cb.allow_request().is_ok());
}

// ============================================================================
// BOUNDARY CONDITION TESTS - Metrics
// ============================================================================

#[test]
fn test_metrics_with_zero_operations() {
    let metrics = OperationMetrics::new();
    assert_eq!(metrics.total_operations(), 0);
    assert_eq!(metrics.successful_operations(), 0);
    assert_eq!(metrics.failed_operations(), 0);
}

#[test]
fn test_metrics_success_rate_with_all_successes() {
    let metrics = OperationMetrics::new();
    for _ in 0..10 {
        metrics.record_success(100);
    }
    assert_eq!(metrics.success_rate(), 100.0);
}

#[test]
fn test_metrics_success_rate_with_all_failures() {
    let metrics = OperationMetrics::new();
    for _ in 0..10 {
        metrics.record_failure(100);
    }
    assert_eq!(metrics.success_rate(), 0.0);
}

#[test]
fn test_metrics_success_rate_with_mixed_operations() {
    let metrics = OperationMetrics::new();
    for _ in 0..7 {
        metrics.record_success(100);
    }
    for _ in 0..3 {
        metrics.record_failure(100);
    }
    assert_eq!(metrics.success_rate(), 70.0);
}

#[test]
fn test_metrics_average_time_calculation() {
    let metrics = OperationMetrics::new();
    metrics.record_success(100);
    metrics.record_success(200);
    metrics.record_success(300);
    let avg = metrics.average_time_ms();
    assert!((avg - 200.0).abs() < 0.01); // Should be 200ms
}

// ============================================================================
// BOUNDARY CONDITION TESTS - Salience Range
// ============================================================================

#[tokio::test]
async fn test_synapse_with_minimum_salience() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();
    let embedding = embedder.embed("test content").await.unwrap();
    let entry = MemoryEntry::builder(MemoryId::new(), "test content".to_string())
        .salience(0.0)
        .embedding(embedding)
        .build();
    synapse.store(entry).await.unwrap();
    assert_eq!(synapse.len(), 1);
}

#[tokio::test]
async fn test_synapse_with_maximum_salience() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();
    let embedding = embedder.embed("test content").await.unwrap();
    let entry = MemoryEntry::builder(MemoryId::new(), "test content".to_string())
        .salience(1.0)
        .embedding(embedding)
        .build();
    synapse.store(entry).await.unwrap();
    assert_eq!(synapse.len(), 1);
}

#[tokio::test]
async fn test_synapse_with_mid_range_salience() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();
    let embedding = embedder.embed("test content").await.unwrap();
    let entry = MemoryEntry::builder(MemoryId::new(), "test content".to_string())
        .salience(0.5)
        .embedding(embedding)
        .build();
    synapse.store(entry).await.unwrap();
    assert_eq!(synapse.len(), 1);
}

// ============================================================================
// LARGE-SCALE OPERATION TESTS
// ============================================================================

#[tokio::test]
async fn test_synapse_with_1000_memories() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();

    for i in 0..1000 {
        let embedding = embedder.embed(&format!("content {}", i)).await.unwrap();
        let entry = MemoryEntry::builder(MemoryId::new(), format!("content {}", i))
            .salience((i as f32 % 100.0) / 100.0)
            .embedding(embedding)
            .build();
        synapse.store(entry).await.unwrap();
    }

    assert_eq!(synapse.len(), 1000);
}

#[tokio::test]
async fn test_synapse_retrieve_with_large_limit() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();

    // Store 100 memories
    for i in 0..100 {
        let embedding = embedder.embed(&format!("content {}", i)).await.unwrap();
        let entry = MemoryEntry::builder(MemoryId::new(), format!("content {}", i))
            .salience(0.5)
            .embedding(embedding)
            .build();
        synapse.store(entry).await.unwrap();
    }

    // Retrieve with limit larger than stored memories
    let results = synapse.retrieve(&vec![0.1; 384], 500).await.unwrap();
    assert_eq!(results.len(), 100);
}

#[tokio::test]
async fn test_synapse_retrieve_with_zero_limit() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();

    let embedding = embedder.embed("test content").await.unwrap();
    let entry = MemoryEntry::builder(MemoryId::new(), "test content".to_string())
        .salience(0.5)
        .embedding(embedding)
        .build();
    synapse.store(entry).await.unwrap();

    // Retrieve with limit 0
    let results = synapse.retrieve(&vec![0.1; 384], 0).await.unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_synapse_retrieve_with_limit_one() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();

    // Store 10 memories
    for i in 0..10 {
        let embedding = embedder.embed(&format!("content {}", i)).await.unwrap();
        let entry = MemoryEntry::builder(MemoryId::new(), format!("content {}", i))
            .salience(0.5)
            .embedding(embedding)
            .build();
        synapse.store(entry).await.unwrap();
    }

    // Retrieve with limit 1
    let results = synapse.retrieve(&vec![0.1; 384], 1).await.unwrap();
    assert_eq!(results.len(), 1);
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

#[tokio::test]
async fn test_concurrent_synapse_stores() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = Arc::new(SynapseMemory::new());

    let mut handles: Vec<JoinHandle<()>> = vec![];

    // Spawn 10 concurrent tasks, each storing 10 memories
    for task_id in 0..10 {
        let synapse_clone = synapse.clone();
        let embedder_clone = embedder.clone();

        let handle = tokio::spawn(async move {
            for _i in 0..10 {
                let embedding = embedder_clone
                    .embed(&format!("content from task {}", task_id))
                    .await
                    .unwrap();
                let entry =
                    MemoryEntry::builder(MemoryId::new(), format!("content from task {}", task_id))
                        .salience(0.5)
                        .embedding(embedding)
                        .build();
                synapse_clone.store(entry).await.unwrap();
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all 100 memories were stored
    assert_eq!(synapse.len(), 100);
}

#[tokio::test]
async fn test_concurrent_synapse_retrieves() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = Arc::new(SynapseMemory::new());

    // Store 50 memories
    for i in 0..50 {
        let embedding = embedder.embed(&format!("content {}", i)).await.unwrap();
        let entry = MemoryEntry::builder(MemoryId::new(), format!("content {}", i))
            .salience(0.5)
            .embedding(embedding)
            .build();
        synapse.store(entry).await.unwrap();
    }

    let mut handles: Vec<JoinHandle<usize>> = vec![];

    // Spawn 10 concurrent retrieve tasks
    for _ in 0..10 {
        let synapse_clone = synapse.clone();

        let handle = tokio::spawn(async move {
            let results = synapse_clone.retrieve(&vec![0.1; 384], 10).await.unwrap();
            results.len()
        });

        handles.push(handle);
    }

    // Wait for all tasks and verify results
    for handle in handles {
        let result_count = handle.await.unwrap();
        assert_eq!(result_count, 10);
    }
}

#[tokio::test]
async fn test_concurrent_circuit_breaker_state_transitions() {
    let config = CircuitBreakerConfig::new().with_failure_threshold(5);
    let cb = Arc::new(CircuitBreaker::new(config));
    let failure_count = Arc::new(AtomicUsize::new(0));

    let mut handles: Vec<JoinHandle<()>> = vec![];

    // Spawn 10 concurrent tasks, each recording failures
    for _ in 0..10 {
        let cb_clone = cb.clone();
        let failure_count_clone = failure_count.clone();

        let handle = tokio::spawn(async move {
            for _ in 0..5 {
                cb_clone.record_failure();
                failure_count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify total failures recorded
    let total_failures = failure_count.load(Ordering::SeqCst);
    assert_eq!(total_failures, 50);

    // Circuit should be open (50 failures >> 5 threshold)
    assert!(cb.allow_request().is_err());
}

#[tokio::test]
async fn test_concurrent_metrics_updates() {
    let metrics = Arc::new(OperationMetrics::new());
    let mut handles: Vec<JoinHandle<()>> = vec![];

    // Spawn 10 concurrent tasks, each recording operations
    for _ in 0..10 {
        let metrics_clone = metrics.clone();

        let handle = tokio::spawn(async move {
            for i in 0..10 {
                let success = i % 2 == 0;
                if success {
                    metrics_clone.record_success(100 + i as u64);
                } else {
                    metrics_clone.record_failure(100 + i as u64);
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify total operations recorded (10 tasks * 10 ops each = 100)
    assert_eq!(metrics.total_operations(), 100);

    // Verify success rate (50% of operations should succeed)
    assert_eq!(metrics.success_rate(), 50.0);
}

// ============================================================================
// STRESS TESTS - Memory Pressure
// ============================================================================

#[tokio::test]
async fn test_synapse_stress_many_deletes() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();

    // Store 100 memories
    let mut ids = vec![];
    for i in 0..100 {
        let embedding = embedder.embed(&format!("content {}", i)).await.unwrap();
        let id = MemoryId::new();
        let entry = MemoryEntry::builder(id, format!("content {}", i))
            .salience(0.5)
            .embedding(embedding)
            .build();
        ids.push(id);
        synapse.store(entry).await.unwrap();
    }

    assert_eq!(synapse.len(), 100);

    // Delete all memories
    for id in ids {
        synapse.delete(&id).await.unwrap();
    }

    assert_eq!(synapse.len(), 0);
}

#[tokio::test]
async fn test_synapse_stress_many_clears() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();

    for _iteration in 0..10 {
        // Store 50 memories
        for i in 0..50 {
            let embedding = embedder.embed(&format!("content {}", i)).await.unwrap();
            let entry = MemoryEntry::builder(MemoryId::new(), format!("content {}", i))
                .salience(0.5)
                .embedding(embedding)
                .build();
            synapse.store(entry).await.unwrap();
        }

        assert_eq!(synapse.len(), 50);

        // Clear all
        synapse.clear().await.unwrap();
        assert_eq!(synapse.len(), 0);
    }
}

// ============================================================================
// INTEGRATION TESTS - Combined Scenarios
// ============================================================================

#[tokio::test]
async fn test_orchestrator_with_concurrent_mixed_operations() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = Arc::new(SynapseMemory::new());
    let dir = tempfile::tempdir().unwrap();
    let orchestrator = Arc::new(
        MemoryOrchestrator::new(embedder.clone(), dir.path(), "memories", 384)
            .await
            .unwrap(),
    );

    let mut handles: Vec<JoinHandle<()>> = vec![];

    // Task 1: Store memories in synapse
    let synapse_clone = synapse.clone();
    let embedder_clone = embedder.clone();
    let handle1 = tokio::spawn(async move {
        for i in 0..20 {
            let embedding = embedder_clone
                .embed(&format!("content {}", i))
                .await
                .unwrap();
            let entry = MemoryEntry::builder(MemoryId::new(), format!("content {}", i))
                .salience(0.5)
                .embedding(embedding)
                .build();
            synapse_clone.store(entry).await.unwrap();
        }
    });
    handles.push(handle1);

    // Task 2: Recall memories
    let orch2 = orchestrator.clone();
    let handle2 = tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        for _ in 0..10 {
            let _ = orch2.recall("content".to_string(), 5).await.unwrap();
        }
    });
    handles.push(handle2);

    // Task 3: Forget memories
    let orch3 = orchestrator.clone();
    let handle3 = tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        for i in 0..10 {
            let id = MemoryId::from_string(&format!("00000000-0000-0000-0000-{:012}", i)).ok();
            if let Some(id) = id {
                let _ = orch3.forget(id).await;
            }
        }
    });
    handles.push(handle3);

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_concurrent_orchestrator_operations() {
    let embedder = Arc::new(MockEmbedder::new());
    let dir = tempfile::tempdir().unwrap();
    let orchestrator = Arc::new(
        MemoryOrchestrator::new(embedder.clone(), dir.path(), "memories", 384)
            .await
            .unwrap(),
    );

    let mut handles: Vec<JoinHandle<()>> = vec![];

    // Spawn 5 concurrent tasks, each recalling and forgetting
    for task_id in 0..5 {
        let orchestrator_clone = orchestrator.clone();

        let handle = tokio::spawn(async move {
            for _i in 0..5 {
                let content = format!("memory content from task {}", task_id);

                // Recall
                let results = orchestrator_clone.recall(content.clone(), 5).await.unwrap();

                // If we got results, try to forget the first one
                if !results.is_empty() {
                    let id = results[0].id;
                    let _ = orchestrator_clone.forget(id).await;
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_synapse_retrieve_by_scope_with_mixed_scopes() {
    let embedder = Arc::new(MockEmbedder::new());
    let synapse = SynapseMemory::new();

    // Store memories with different scopes
    let scopes = vec![
        MemoryScope::Global,
        MemoryScope::User("user1".to_string()),
        MemoryScope::Session("session1".to_string()),
        MemoryScope::Agent("agent1".to_string()),
    ];

    for (_idx, scope) in scopes.iter().enumerate() {
        for i in 0..5 {
            let embedding = embedder.embed(&format!("content {}", i)).await.unwrap();
            let entry = MemoryEntry::builder(MemoryId::new(), format!("content {}", i))
                .salience(0.5)
                .scope(scope.clone())
                .embedding(embedding)
                .build();
            synapse.store(entry).await.unwrap();
        }
    }

    // Verify total count
    assert_eq!(synapse.len(), 20);

    // Retrieve by specific user scope (should only match user1 entries)
    let user_results = synapse
        .retrieve_by_scope(&vec![0.1; 384], &MemoryScope::User("user1".to_string()), 10)
        .await
        .unwrap();
    // Should get 5 from user1 + 5 from Global (since Global matches all)
    assert_eq!(user_results.len(), 10);

    // Retrieve by specific session scope (should only match session1 entries)
    let session_results = synapse
        .retrieve_by_scope(
            &vec![0.1; 384],
            &MemoryScope::Session("session1".to_string()),
            10,
        )
        .await
        .unwrap();
    // Should get 5 from session1 + 5 from Global (since Global matches all)
    assert_eq!(session_results.len(), 10);
}
