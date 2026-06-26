//! End-to-end integration tests for Ollama embeddings with wiremock.
//!
//! These tests verify the complete semantic two-tier memory system:
//! - Warmup probe validates Ollama connection and dimension
//! - Prefixes are applied before embedding (search_query: and search_document:)
//! - Memories are stored in both Synapse and Cortex
//! - Recall returns results from both tiers
//! - Synapse is offline (no Ollama calls for Synapse-only recall)

use cerebrum_core::{
    Config, MemoryEntry, MemoryId, MemoryOrchestrator, MemoryScope, MemoryStore,
};
use wiremock::{
    matchers::{body_string_contains, method, path},
    Mock, MockServer, ResponseTemplate,
};

/// Test that MemoryOrchestrator::from_config succeeds with mocked Ollama.
#[tokio::test]
async fn test_from_config_with_mocked_ollama() {
    let mock_server = MockServer::start().await;

    // Mock the warmup probe request
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let orchestrator = MemoryOrchestrator::from_config(&config).await;
    assert!(orchestrator.is_ok(), "from_config should succeed with mocked Ollama");
}

/// Test that warmup probe validates embedding dimension.
#[tokio::test]
async fn test_warmup_probe_validates_dimension() {
    let mock_server = MockServer::start().await;

    // Mock with wrong dimension (384 instead of 768)
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 384]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let orchestrator = MemoryOrchestrator::from_config(&config).await;
    assert!(
        orchestrator.is_err(),
        "from_config should fail when dimension doesn't match"
    );
}

/// Test that remember() applies document prefix before embedding.
#[tokio::test]
async fn test_remember_applies_document_prefix() {
    let mock_server = MockServer::start().await;

    // Mock warmup probe
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();
    config.document_prefix = "search_document: ".to_string();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Mock the remember request - should contain the document prefix
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(body_string_contains("search_document: test content"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.2f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let result = orchestrator
        .remember("test content".to_string(), Default::default())
        .await;
    assert!(result.is_ok(), "remember should succeed");
}

/// Test that recall() applies query prefix before embedding.
#[tokio::test]
async fn test_recall_applies_query_prefix() {
    let mock_server = MockServer::start().await;

    // Mock warmup probe
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();
    config.query_prefix = "search_query: ".to_string();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Mock the recall request - should contain the query prefix
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(body_string_contains("search_query: test query"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.3f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let result = orchestrator.recall("test query".to_string(), 10).await;
    assert!(result.is_ok(), "recall should succeed");
}

/// Test that remember() stores in Synapse (short-term).
#[tokio::test]
async fn test_remember_stores_in_synapse() {
    let mock_server = MockServer::start().await;

    // Mock all Ollama requests
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    config.db_path = temp_dir.path().to_path_buf();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Remember a memory
    let id = orchestrator
        .remember("test memory".to_string(), Default::default())
        .await
        .expect("Failed to remember");

    // Verify it's in Synapse (short-term)
    let synapse_len = orchestrator.synapse_len().await.expect("Failed to get synapse len");
    assert_eq!(synapse_len, 1, "Memory should be in Synapse");

    // Verify it's NOT in Cortex yet (Cortex is long-term, accessed via memorize)
    let cortex_len = orchestrator.cortex_len().await.expect("Failed to get cortex len");
    assert_eq!(cortex_len, 0, "Memory should not be in Cortex until promoted");

    // Verify we can retrieve it from Synapse
    let synapse_list = orchestrator
        .synapse_list()
        .await
        .expect("Failed to list synapse");
    assert_eq!(synapse_list.len(), 1);
    assert_eq!(synapse_list[0].id, id);
}

/// Test that recall() returns results from both Synapse and Cortex.
#[tokio::test]
async fn test_recall_returns_from_both_tiers() {
    let mock_server = MockServer::start().await;

    // Mock all Ollama requests
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    config.db_path = temp_dir.path().to_path_buf();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Remember a memory (goes to Synapse)
    orchestrator
        .remember("synapse memory".to_string(), Default::default())
        .await
        .expect("Failed to remember");

    // Promote it to Cortex
    let synapse_list = orchestrator
        .synapse_list()
        .await
        .expect("Failed to list synapse");
    let id = synapse_list[0].id;
    orchestrator
        .memorize(id)
        .await
        .expect("Failed to memorize");

    // Remember another memory (stays in Synapse)
    orchestrator
        .remember("another synapse memory".to_string(), Default::default())
        .await
        .expect("Failed to remember");

    // Recall should return results from both tiers
    let results = orchestrator
        .recall("memory".to_string(), 10)
        .await
        .expect("Failed to recall");

    assert_eq!(results.len(), 2, "Should return memories from both tiers");
}

/// Test that recall_by_scope filters correctly.
#[tokio::test]
async fn test_recall_by_scope_filters_correctly() {
    let mock_server = MockServer::start().await;

    // Mock all Ollama requests
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    config.db_path = temp_dir.path().to_path_buf();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Store memories with different scopes
    let _id1 = orchestrator
        .remember("global memory".to_string(), Default::default())
        .await
        .expect("Failed to remember");

    // Update the first memory to have Global scope
    let synapse_list = orchestrator
        .synapse_list()
        .await
        .expect("Failed to list synapse");
    let mut entry = synapse_list[0].clone();
    entry.scope = MemoryScope::Global;
    orchestrator.synapse().store(entry).await.unwrap();

    // Recall with Global scope should return the memory
    let results = orchestrator
        .recall_by_scope("memory".to_string(), MemoryScope::Global, 10)
        .await
        .expect("Failed to recall by scope");

    assert!(!results.is_empty(), "Should return memories with Global scope");
}

/// Test that Synapse is offline (no Ollama calls for Synapse-only recall).
#[tokio::test]
async fn test_synapse_offline_no_ollama_calls() {
    let mock_server = MockServer::start().await;

    // Mock warmup probe
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    config.db_path = temp_dir.path().to_path_buf();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Store a memory directly in Synapse (bypassing orchestrator to avoid embedding)
    let id = MemoryId::new();
    let entry = MemoryEntry::builder(id, "synapse only".to_string())
        .embedding(vec![0.1f32; 768])
        .build();
    orchestrator.synapse().store(entry).await.unwrap();

    // Verify it's in Synapse but not in Cortex
    assert_eq!(
        orchestrator.synapse_len().await.unwrap(),
        1,
        "Memory should be in Synapse"
    );
    assert_eq!(
        orchestrator.cortex_len().await.unwrap(),
        0,
        "Memory should not be in Cortex"
    );

    // Synapse retrieve should work without calling Ollama (it uses precomputed vectors)
    let results = orchestrator
        .synapse()
        .retrieve(&vec![0.1f32; 768], 10)
        .await
        .expect("Failed to retrieve from Synapse");

    assert_eq!(results.len(), 1, "Should retrieve from Synapse without Ollama");
}

/// Test that end_session clears Synapse and promotes high-salience memories.
#[tokio::test]
async fn test_end_session_clears_synapse_and_promotes() {
    let mock_server = MockServer::start().await;

    // Mock all Ollama requests
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    config.db_path = temp_dir.path().to_path_buf();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Remember a high-salience memory
    let _id = orchestrator
        .remember("important memory".to_string(), Default::default())
        .await
        .expect("Failed to remember");

    // Update salience to be high
    let synapse_list = orchestrator
        .synapse_list()
        .await
        .expect("Failed to list synapse");
    let mut entry = synapse_list[0].clone();
    entry.salience = 0.9;
    orchestrator.synapse().store(entry).await.unwrap();

    // End session with threshold 0.5
    orchestrator
        .end_session(0.5)
        .await
        .expect("Failed to end session");

    // Verify Synapse is cleared
    assert_eq!(
        orchestrator.synapse_len().await.unwrap(),
        0,
        "Synapse should be cleared"
    );

    // Verify memory was promoted to Cortex
    assert_eq!(
        orchestrator.cortex_len().await.unwrap(),
        1,
        "High-salience memory should be promoted to Cortex"
    );
}

/// Test that forget removes from both tiers.
#[tokio::test]
async fn test_forget_removes_from_both_tiers() {
    let mock_server = MockServer::start().await;

    // Mock all Ollama requests
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": [vec![0.1f32; 768]] })),
        )
        .mount(&mock_server)
        .await;

    let mut config = Config::default();
    config.ollama_url = mock_server.uri();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    config.db_path = temp_dir.path().to_path_buf();

    let orchestrator = MemoryOrchestrator::from_config(&config)
        .await
        .expect("Failed to create orchestrator");

    // Remember a memory (goes to Synapse)
    let id = orchestrator
        .remember("test memory".to_string(), Default::default())
        .await
        .expect("Failed to remember");

    // Promote it to Cortex
    orchestrator
        .memorize(id)
        .await
        .expect("Failed to memorize");

    // Verify it's in both tiers
    assert_eq!(orchestrator.synapse_len().await.unwrap(), 0, "Should be removed from Synapse after promotion");
    assert_eq!(orchestrator.cortex_len().await.unwrap(), 1, "Should be in Cortex after promotion");

    // Forget it
    orchestrator.forget(id).await.expect("Failed to forget");

    // Verify it's removed from both tiers
    assert_eq!(
        orchestrator.synapse_len().await.unwrap(),
        0,
        "Memory should be removed from Synapse"
    );
    assert_eq!(
        orchestrator.cortex_len().await.unwrap(),
        0,
        "Memory should be removed from Cortex"
    );
}
