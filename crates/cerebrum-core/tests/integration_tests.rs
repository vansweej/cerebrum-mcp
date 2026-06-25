use cerebrum_core::{utils, Embedder, MemoryEntry, MemoryId, MemoryTier, MockEmbedder};

#[test]
fn test_memory_id_generation() {
    let id1 = MemoryId::new();
    let id2 = MemoryId::new();

    assert_ne!(id1, id2);
}

#[test]
fn test_memory_id_default() {
    let id1 = MemoryId::default();
    let id2 = MemoryId::default();

    assert_ne!(id1, id2);
}

#[test]
fn test_memory_entry_new() {
    let id = MemoryId::new();
    let content = "Test memory".to_string();
    let entry = MemoryEntry::new(id, content.clone());

    assert_eq!(entry.id, id);
    assert_eq!(entry.content, content);
    assert_eq!(entry.salience, 0.5);
    assert_eq!(entry.tier, MemoryTier::Synapse);
    assert!(entry.embedding.is_none());
    assert!(entry.source_session_id.is_none());
    assert!(entry.metadata.is_empty());
}

#[test]
fn test_memory_entry_builder() {
    let id = MemoryId::new();
    let content = "Test memory".to_string();
    let embedding = vec![0.1; 384];
    let session_id = "session-123".to_string();

    let entry = MemoryEntry::builder(id, content.clone())
        .salience(0.8)
        .tier(MemoryTier::Cortex)
        .embedding(embedding.clone())
        .source_session_id(session_id.clone())
        .metadata("key".to_string(), "value".to_string())
        .build();

    assert_eq!(entry.id, id);
    assert_eq!(entry.content, content);
    assert_eq!(entry.salience, 0.8);
    assert_eq!(entry.tier, MemoryTier::Cortex);
    assert_eq!(entry.embedding, Some(embedding));
    assert_eq!(entry.source_session_id, Some(session_id));
    assert_eq!(entry.metadata.get("key"), Some(&"value".to_string()));
}

#[test]
fn test_memory_entry_builder_salience_clamping() {
    let id = MemoryId::new();
    let content = "Test".to_string();

    let entry1 = MemoryEntry::builder(id, content.clone())
        .salience(1.5)
        .build();
    assert_eq!(entry1.salience, 1.0);

    let entry2 = MemoryEntry::builder(id, content).salience(-0.5).build();
    assert_eq!(entry2.salience, 0.0);
}

#[test]
fn test_memory_tier_enum() {
    assert_eq!(MemoryTier::Synapse, MemoryTier::Synapse);
    assert_eq!(MemoryTier::Cortex, MemoryTier::Cortex);
    assert_ne!(MemoryTier::Synapse, MemoryTier::Cortex);
}

#[test]
fn test_utils_generate_memory_id() {
    let id1 = utils::generate_memory_id();
    let id2 = utils::generate_memory_id();

    assert_ne!(id1, id2);
}

#[test]
fn test_utils_validate_embedding_dimension() {
    let valid = vec![0.1; 384];
    assert!(utils::validate_embedding_dimension(&valid).is_ok());

    let invalid = vec![0.1; 256];
    assert!(utils::validate_embedding_dimension(&invalid).is_err());
}

#[test]
fn test_utils_default_salience() {
    assert_eq!(utils::default_salience(), 0.5);
}

#[test]
fn test_utils_current_timestamp() {
    let before = chrono::Utc::now();
    let timestamp = utils::current_timestamp();
    let after = chrono::Utc::now();

    assert!(timestamp >= before);
    assert!(timestamp <= after);
}

#[tokio::test]
async fn test_mock_embedder_creation() {
    let _embedder = MockEmbedder::new();
    assert_eq!(MockEmbedder::dimension(), 384);
}

#[tokio::test]
async fn test_mock_embedder_dimension() {
    assert_eq!(MockEmbedder::dimension(), 384);
}

#[tokio::test]
async fn test_mock_embedder_embed() {
    let embedder = MockEmbedder::new();
    let embedding = embedder
        .embed("Hello, world!")
        .await
        .expect("Failed to embed");

    assert_eq!(embedding.len(), 384);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_mock_embedder_empty_text() {
    let embedder = MockEmbedder::new();
    let result = embedder.embed("").await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_mock_embedder_consistency() {
    let embedder = MockEmbedder::new();
    let text = "The quick brown fox jumps over the lazy dog";

    let embedding1 = embedder.embed(text).await.expect("Failed to embed");
    let embedding2 = embedder.embed(text).await.expect("Failed to embed");

    assert_eq!(embedding1, embedding2);
}

#[tokio::test]
async fn test_mock_embedder_different_texts() {
    let embedder = MockEmbedder::new();

    let embedding1 = embedder
        .embed("Hello, world!")
        .await
        .expect("Failed to embed");
    let embedding2 = embedder
        .embed("Goodbye, world!")
        .await
        .expect("Failed to embed");

    // Different texts should produce different embeddings
    assert_ne!(embedding1, embedding2);
}

#[tokio::test]
async fn test_mock_embedder_normalized() {
    let embedder = MockEmbedder::new();
    let embedding = embedder.embed("Test text").await.expect("Failed to embed");

    // Check that embedding is approximately unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.001, "Embedding should be normalized");
}

#[test]
fn test_mock_embedder_default() {
    let _embedder = MockEmbedder::default();
    assert_eq!(MockEmbedder::dimension(), 384);
}

#[test]
fn test_memory_entry_builder_timestamp() {
    let id = MemoryId::new();
    let content = "Test".to_string();
    let timestamp = chrono::Utc::now();

    let entry = MemoryEntry::builder(id, content)
        .timestamp(timestamp)
        .build();

    assert_eq!(entry.timestamp, timestamp);
}

#[test]
fn test_memory_entry_builder_multiple_metadata() {
    let id = MemoryId::new();
    let content = "Test".to_string();

    let entry = MemoryEntry::builder(id, content)
        .metadata("key1".to_string(), "value1".to_string())
        .metadata("key2".to_string(), "value2".to_string())
        .metadata("key3".to_string(), "value3".to_string())
        .build();

    assert_eq!(entry.metadata.len(), 3);
    assert_eq!(entry.metadata.get("key1"), Some(&"value1".to_string()));
    assert_eq!(entry.metadata.get("key2"), Some(&"value2".to_string()));
    assert_eq!(entry.metadata.get("key3"), Some(&"value3".to_string()));
}
