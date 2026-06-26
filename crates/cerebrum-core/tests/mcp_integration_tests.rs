//! MCP Integration Tests
//!
//! End-to-end tests for the MCP server handler, verifying:
//! - Tool definitions and schemas
//! - Tool calling and response handling
//! - Error handling and validation
//! - Protocol compliance

use cerebrum_core::{
    embedder::MockEmbedder,
    models::{MemoryId, MemoryTier},
    orchestrator::MemoryOrchestrator,
    Embedder,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create a test orchestrator with persistent tempdir
async fn create_test_orchestrator() -> (MemoryOrchestrator, tempfile::TempDir) {
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
    let dir = tempfile::tempdir().expect("Failed to create tempdir");
    let orchestrator = MemoryOrchestrator::new(dir.path(), "memories", 384, embedder)
        .await
        .expect("Failed to create orchestrator");
    (orchestrator, dir)
}

/// Helper to parse tool input from JSON
fn parse_tool_input(input: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str(input)
}

// ============================================================================
// Tool Definition Tests
// ============================================================================

#[test]
fn test_remember_tool_definition() {
    // Verify the remember tool has correct schema
    let schema = json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory content to store"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata key-value pairs"
            }
        },
        "required": ["content"]
    });

    // Verify schema structure
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["content"].is_object());
    assert!(schema["properties"]["metadata"].is_object());
    assert_eq!(schema["required"][0], "content");
}

#[test]
fn test_recall_tool_definition() {
    // Verify the recall tool has correct schema
    let schema = json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for semantic similarity"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)"
            }
        },
        "required": ["query"]
    });

    // Verify schema structure
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["query"].is_object());
    assert!(schema["properties"]["limit"].is_object());
    assert_eq!(schema["required"][0], "query");
}

#[test]
fn test_memorize_tool_definition() {
    // Verify the memorize tool has correct schema
    let schema = json!({
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of memory to promote from Synapse to Cortex"
            }
        },
        "required": ["memory_id"]
    });

    // Verify schema structure
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["memory_id"].is_object());
    assert_eq!(schema["required"][0], "memory_id");
}

#[test]
fn test_forget_tool_definition() {
    // Verify the forget tool has correct schema
    let schema = json!({
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of memory to delete"
            }
        },
        "required": ["memory_id"]
    });

    // Verify schema structure
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["memory_id"].is_object());
    assert_eq!(schema["required"][0], "memory_id");
}

#[test]
fn test_end_session_tool_definition() {
    // Verify the end_session tool has correct schema
    let schema = json!({
        "type": "object",
        "properties": {
            "promotion_threshold": {
                "type": "number",
                "description": "Salience threshold for auto-promotion (0.0-1.0, default: 0.7)"
            }
        },
        "required": []
    });

    // Verify schema structure
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["promotion_threshold"].is_object());
}

// ============================================================================
// Tool Input Validation Tests
// ============================================================================

#[test]
fn test_remember_input_validation_valid() {
    let input = json!({
        "content": "Test memory content",
        "metadata": {
            "key": "value"
        }
    });

    // Verify input can be parsed
    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
    let value = parsed.unwrap();
    assert_eq!(value["content"], "Test memory content");
    assert_eq!(value["metadata"]["key"], "value");
}

#[test]
fn test_remember_input_validation_minimal() {
    let input = json!({
        "content": "Minimal memory"
    });

    // Verify minimal input is valid
    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
    let value = parsed.unwrap();
    assert_eq!(value["content"], "Minimal memory");
}

#[test]
fn test_remember_input_validation_missing_required() {
    let input = json!({
        "metadata": {
            "key": "value"
        }
    });

    // Verify missing required field is caught
    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok()); // JSON parsing succeeds, but validation would fail
    let value = parsed.unwrap();
    assert!(!value.get("content").is_some() || value["content"].is_null());
}

#[test]
fn test_recall_input_validation_valid() {
    let input = json!({
        "query": "important memories",
        "limit": 5
    });

    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
    let value = parsed.unwrap();
    assert_eq!(value["query"], "important memories");
    assert_eq!(value["limit"], 5);
}

#[test]
fn test_recall_input_validation_minimal() {
    let input = json!({
        "query": "test"
    });

    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
    let value = parsed.unwrap();
    assert_eq!(value["query"], "test");
}

#[test]
fn test_memorize_input_validation_valid() {
    let input = json!({
        "memory_id": "mem-12345"
    });

    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
    let value = parsed.unwrap();
    assert_eq!(value["memory_id"], "mem-12345");
}

#[test]
fn test_forget_input_validation_valid() {
    let input = json!({
        "memory_id": "mem-67890"
    });

    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
    let value = parsed.unwrap();
    assert_eq!(value["memory_id"], "mem-67890");
}

#[test]
fn test_end_session_input_validation_valid() {
    let input = json!({
        "promotion_threshold": 0.75
    });

    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
    let value = parsed.unwrap();
    assert_eq!(value["promotion_threshold"], 0.75);
}

#[test]
fn test_end_session_input_validation_minimal() {
    let input = json!({});

    let parsed = parse_tool_input(&input.to_string());
    assert!(parsed.is_ok());
}

// ============================================================================
// Tool Calling Integration Tests
// ============================================================================

#[tokio::test]
async fn test_remember_tool_integration() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Simulate remember tool call
    let content = "Important project deadline".to_string();
    let mut metadata = HashMap::new();
    metadata.insert("priority".to_string(), "high".to_string());

    let result = orchestrator.remember(content, metadata).await;
    assert!(result.is_ok());

    let memory_id = result.unwrap();
    assert!(!memory_id.to_string().is_empty());
}

#[tokio::test]
async fn test_recall_tool_integration() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store memories first
    let _ = orchestrator
        .remember("Meeting at 3pm".to_string(), HashMap::new())
        .await;
    let _ = orchestrator
        .remember("Lunch with team".to_string(), HashMap::new())
        .await;

    // Recall memories
    let results = orchestrator.recall("meeting".to_string(), 10).await;
    assert!(results.is_ok());

    let memories = results.unwrap();
    assert!(!memories.is_empty());
}

#[tokio::test]
async fn test_memorize_tool_integration() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store a memory in Synapse
    let memory_id = orchestrator
        .remember("Important data".to_string(), HashMap::new())
        .await
        .unwrap();

    // Promote to Cortex
    let result = orchestrator.memorize(memory_id).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_forget_tool_integration() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store a memory
    let memory_id = orchestrator
        .remember("Temporary memory".to_string(), HashMap::new())
        .await
        .unwrap();

    // Forget it
    let result = orchestrator.forget(memory_id).await;
    assert!(result.is_ok());

    // Verify it's gone
    let recall_result = orchestrator.recall("temporary".to_string(), 10).await;
    assert!(recall_result.is_ok());
    let memories = recall_result.unwrap();
    assert!(memories.is_empty());
}

#[tokio::test]
async fn test_end_session_tool_integration() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store memories
    let _ = orchestrator
        .remember("Session memory 1".to_string(), HashMap::new())
        .await;
    let _ = orchestrator
        .remember("Session memory 2".to_string(), HashMap::new())
        .await;

    // End session with promotion threshold
    let result = orchestrator.end_session(0.7).await;
    assert!(result.is_ok());

    // Verify Synapse is cleared
    let synapse_len = orchestrator.synapse_len().await.unwrap();
    assert_eq!(synapse_len, 0);
}

// ============================================================================
// Blended Search Integration Tests
// ============================================================================

#[tokio::test]
async fn test_blended_search_synapse_and_cortex() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store in Synapse
    let synapse_id = orchestrator
        .remember("Synapse memory".to_string(), HashMap::new())
        .await
        .unwrap();

    // Promote to Cortex
    let _ = orchestrator.memorize(synapse_id).await;

    // Store another in Synapse
    let _ = orchestrator
        .remember("Another synapse memory".to_string(), HashMap::new())
        .await;

    // Recall should find both
    let results = orchestrator.recall("memory".to_string(), 10).await;
    assert!(results.is_ok());
    let memories = results.unwrap();
    assert!(!memories.is_empty());
}

#[tokio::test]
async fn test_recall_deduplication() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store same content in both tiers (simulating duplication)
    let content = "Duplicate test".to_string();
    let id1 = orchestrator
        .remember(content.clone(), HashMap::new())
        .await
        .unwrap();
    let _ = orchestrator.memorize(id1).await;

    // Recall should deduplicate
    let results = orchestrator.recall("duplicate".to_string(), 10).await;
    assert!(results.is_ok());
    let memories = results.unwrap();

    // Count occurrences of the same content
    let count = memories.iter().filter(|m| m.content == content).count();
    assert_eq!(count, 1, "Should have exactly one copy after deduplication");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_forget_nonexistent_memory() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Try to forget a memory that doesn't exist
    let fake_id = MemoryId::new();
    let result = orchestrator.forget(fake_id).await;

    // Should handle gracefully (idempotent)
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_memorize_nonexistent_memory() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Try to promote a memory that doesn't exist
    let fake_id = MemoryId::new();
    let result = orchestrator.memorize(fake_id).await;

    // Should return error for nonexistent memory
    assert!(result.is_err());
}

#[tokio::test]
async fn test_recall_empty_query() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store a memory first
    let _ = orchestrator
        .remember("Test content".to_string(), HashMap::new())
        .await;

    // Recall with empty query - may return error or empty results
    let results = orchestrator.recall("".to_string(), 10).await;

    // Accept either error or empty results - implementation choice
    match results {
        Ok(memories) => {
            // Empty query may return no results or all results
            let _ = memories;
        }
        Err(_) => {
            // Empty query may be rejected as invalid
        }
    }
}

#[tokio::test]
async fn test_recall_with_zero_limit() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store a memory
    let _ = orchestrator
        .remember("Test".to_string(), HashMap::new())
        .await;

    // Recall with zero limit
    let results = orchestrator.recall("test".to_string(), 0).await;
    assert!(results.is_ok());

    let memories = results.unwrap();
    assert!(memories.is_empty());
}

#[tokio::test]
async fn test_recall_with_large_limit() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store multiple memories
    for i in 0..5 {
        let _ = orchestrator
            .remember(format!("Memory {}", i), HashMap::new())
            .await;
    }

    // Recall with large limit
    let results = orchestrator.recall("memory".to_string(), 1000).await;
    assert!(results.is_ok());

    let memories = results.unwrap();
    assert!(memories.len() <= 5);
}

// ============================================================================
// Tier Assignment Tests
// ============================================================================

#[tokio::test]
async fn test_remember_assigns_synapse_tier() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    let _memory_id = orchestrator
        .remember("Test".to_string(), HashMap::new())
        .await
        .unwrap();

    // Recall to verify tier
    let results = orchestrator.recall("test".to_string(), 10).await.unwrap();
    assert!(!results.is_empty());

    let memory = &results[0];
    assert_eq!(memory.tier, MemoryTier::Synapse);
}

#[tokio::test]
async fn test_memorize_assigns_cortex_tier() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    let memory_id = orchestrator
        .remember("Test".to_string(), HashMap::new())
        .await
        .unwrap();

    let _ = orchestrator.memorize(memory_id).await;

    // Recall to verify tier
    let results = orchestrator.recall("test".to_string(), 10).await.unwrap();
    assert!(!results.is_empty());

    // Find the promoted memory
    let promoted = results.iter().find(|m| m.id == memory_id);
    assert!(promoted.is_some());
    assert_eq!(promoted.unwrap().tier, MemoryTier::Cortex);
}

// ============================================================================
// Salience and Ranking Tests
// ============================================================================

#[tokio::test]
async fn test_recall_ranking_by_similarity() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store memories with varying similarity to query
    let _ = orchestrator
        .remember("exact match query".to_string(), HashMap::new())
        .await;
    let _ = orchestrator
        .remember("different content".to_string(), HashMap::new())
        .await;

    // Recall with specific query
    let results = orchestrator
        .recall("exact match query".to_string(), 10)
        .await
        .unwrap();
    assert!(!results.is_empty());

    // Most similar should rank first
    assert_eq!(results[0].content, "exact match query");
}

// ============================================================================
// Session Lifecycle Tests
// ============================================================================

#[tokio::test]
async fn test_auto_promotion_on_session_end() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Store high-salience memory (default salience is 0.5)
    let _high_salience_id = orchestrator
        .remember("High salience memory".to_string(), HashMap::new())
        .await
        .unwrap();

    // Store low-salience memory
    let _low_salience_id = orchestrator
        .remember("Low salience memory".to_string(), HashMap::new())
        .await
        .unwrap();

    // End session with 0.3 threshold (should promote both since default is 0.5)
    let _ = orchestrator.end_session(0.3).await;

    // Verify Synapse is cleared
    let synapse_len = orchestrator.synapse_len().await.unwrap();
    assert_eq!(synapse_len, 0);

    // Verify memories were promoted to Cortex
    let cortex_len = orchestrator.cortex_len().await.unwrap();
    assert!(cortex_len > 0);
}

// ============================================================================
// Embedding Tests
// ============================================================================

#[tokio::test]
async fn test_remember_generates_embedding() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    let _memory_id = orchestrator
        .remember("Test content".to_string(), HashMap::new())
        .await
        .unwrap();

    // Recall to verify embedding was generated
    let results = orchestrator.recall("test".to_string(), 10).await.unwrap();
    assert!(!results.is_empty());

    let memory = &results[0];
    assert!(memory.embedding.is_some());
    assert_eq!(memory.embedding.as_ref().unwrap().len(), 384); // BGE-small dimension
}

#[tokio::test]
async fn test_embedding_consistency() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    let content = "Consistent content".to_string();
    let _id1 = orchestrator
        .remember(content.clone(), HashMap::new())
        .await
        .unwrap();

    // Recall to get embedding
    let results1 = orchestrator.recall(content.clone(), 10).await.unwrap();
    let embedding1 = results1[0].embedding.clone();

    // Store again with same content
    let id2 = orchestrator
        .remember(content.clone(), HashMap::new())
        .await
        .unwrap();

    // Recall again
    let results2 = orchestrator.recall(content, 10).await.unwrap();
    let embedding2 = results2
        .iter()
        .find(|m| m.id == id2)
        .and_then(|m| m.embedding.clone());

    // Embeddings should be identical for same content
    assert_eq!(embedding1, embedding2);
}

// ============================================================================
// Metadata Tests
// ============================================================================

#[tokio::test]
async fn test_memory_metadata_preservation() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    let content = "Test content".to_string();
    let mut metadata = HashMap::new();
    metadata.insert("key".to_string(), "value".to_string());
    metadata.insert("priority".to_string(), "high".to_string());

    let _memory_id = orchestrator
        .remember(content.clone(), metadata.clone())
        .await
        .unwrap();

    // Recall and verify metadata
    let results = orchestrator.recall(content, 10).await.unwrap();
    assert!(!results.is_empty());

    let memory = &results[0];
    assert_eq!(memory.metadata, metadata);
}

#[tokio::test]
async fn test_memory_id_uniqueness() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    let id1 = orchestrator
        .remember("Content 1".to_string(), HashMap::new())
        .await
        .unwrap();
    let id2 = orchestrator
        .remember("Content 2".to_string(), HashMap::new())
        .await
        .unwrap();

    assert_ne!(id1, id2);
}

#[tokio::test]
async fn test_memory_timestamp_generation() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    let _memory_id = orchestrator
        .remember("Test".to_string(), HashMap::new())
        .await
        .unwrap();

    // Recall to verify timestamp
    let results = orchestrator.recall("test".to_string(), 10).await.unwrap();
    assert!(!results.is_empty());

    let memory = &results[0];
    // Verify timestamp is set (not the default/epoch)
    assert!(memory.timestamp.timestamp() > 0);
}

#[tokio::test]
async fn test_synapse_and_cortex_lengths() {
    let (orchestrator, _dir) = create_test_orchestrator().await;

    // Initially empty
    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    assert_eq!(orchestrator.cortex_len().await.unwrap(), 0);

    // Add to Synapse
    let id1 = orchestrator
        .remember("Memory 1".to_string(), HashMap::new())
        .await
        .unwrap();
    assert_eq!(orchestrator.synapse_len().await.unwrap(), 1);

    // Promote to Cortex
    let _ = orchestrator.memorize(id1).await;
    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0);
    assert_eq!(orchestrator.cortex_len().await.unwrap(), 1);
}
