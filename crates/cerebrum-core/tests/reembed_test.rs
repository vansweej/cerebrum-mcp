//! Tests for the `reembed` table-to-table converter.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use cerebrum_core::config::Config;
use cerebrum_core::error::Result;
use cerebrum_core::models::{MemoryEntry, MemoryId, MemoryTier};
use cerebrum_core::reembed::reembed;
use cerebrum_core::traits::{Embedder, MemoryStore};

/// An embedder that records every input string it was asked to embed and
/// returns deterministic 1024-dim vectors.
struct MockEmbedder1024 {
    seen: Mutex<Vec<String>>,
}

impl MockEmbedder1024 {
    fn new() -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
        }
    }

    fn seen_inputs(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Embedder for MockEmbedder1024 {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.seen.lock().unwrap().push(text.to_string());
        Ok(vec![0.5_f32; 1024])
    }

    fn dimension(&self) -> usize {
        1024
    }
}

fn test_config(dir: &std::path::Path) -> Config {
    let mut c = Config::default();
    c.db_path = dir.to_path_buf();
    c.embedding_dim = 1024;
    c
}

#[tokio::test]
async fn missing_source_table_errors_and_creates_no_dest() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let embedder = MockEmbedder1024::new();

    let result = reembed("missing_src", 768, "dest_table", &config, &embedder).await;
    assert!(
        result.is_err(),
        "reembed must fail for a missing source table"
    );

    let path = dir.path().to_str().unwrap();
    let conn = lancedb::connect(path).execute().await.unwrap();
    let names = conn.table_names().execute().await.unwrap();
    assert!(
        !names.contains(&"dest_table".to_string()),
        "dest table must not be created when source is missing"
    );
}

#[tokio::test]
async fn reembeds_all_rows_and_verifies() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let embedder = MockEmbedder1024::new();

    // Populate a 768-dim source table directly via LanceDBCortex.
    let src = cerebrum_core::lancedb_cortex::LanceDBCortex::new(dir.path(), "src_table", 768)
        .await
        .unwrap();
    for i in 0..3 {
        let entry = MemoryEntry::builder(MemoryId::new(), format!("content {i}"))
            .embedding(vec![0.1_f32; 768])
            .tier(MemoryTier::Cortex)
            .build();
        src.store(entry).await.unwrap();
    }

    let report = reembed("src_table", 768, "dest_table", &config, &embedder)
        .await
        .expect("reembed should succeed");

    assert_eq!(report.total, 3);
    assert_eq!(report.written, 3);
    assert!(report.verified, "report must be verified");

    let dest = cerebrum_core::lancedb_cortex::LanceDBCortex::new(dir.path(), "dest_table", 1024)
        .await
        .unwrap();
    let rows = dest.list().await.unwrap();
    assert_eq!(rows.len(), 3);
    for row in rows {
        let emb = row.embedding.expect("row must have embedding");
        assert_eq!(emb.len(), 1024);
    }
}

#[tokio::test]
async fn embed_input_matches_document_prefix_plus_content() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let embedder = MockEmbedder1024::new();

    let src = cerebrum_core::lancedb_cortex::LanceDBCortex::new(dir.path(), "src_table", 768)
        .await
        .unwrap();
    let entry = MemoryEntry::builder(MemoryId::new(), "known content".to_string())
        .embedding(vec![0.1_f32; 768])
        .tier(MemoryTier::Cortex)
        .build();
    src.store(entry).await.unwrap();

    reembed("src_table", 768, "dest_table", &config, &embedder)
        .await
        .expect("reembed should succeed");

    let expected = format!("{}{}", config.document_prefix, "known content");
    let seen = embedder.seen_inputs();
    assert!(
        seen.iter().any(|s| s == &expected),
        "expected embed input {expected:?} to be observed, got {seen:?}"
    );
}

#[tokio::test]
async fn long_content_is_counted_as_truncated() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let embedder = MockEmbedder1024::new();

    let src = cerebrum_core::lancedb_cortex::LanceDBCortex::new(dir.path(), "src_table", 768)
        .await
        .unwrap();

    let long_content = "a".repeat(96_000);
    let entry = MemoryEntry::builder(MemoryId::new(), long_content)
        .embedding(vec![0.1_f32; 768])
        .tier(MemoryTier::Cortex)
        .build();
    src.store(entry).await.unwrap();

    let report = reembed("src_table", 768, "dest_table", &config, &embedder)
        .await
        .expect("reembed should succeed");

    assert_eq!(report.truncated, 1);
}
