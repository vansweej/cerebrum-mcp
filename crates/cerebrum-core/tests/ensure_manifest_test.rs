//! Tests for manifest, schema_probe, and ensure_manifest.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    builder::{FixedSizeListBuilder, Float32Builder, StringBuilder},
    RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use tempfile::TempDir;

use cerebrum_core::config::Config;
use cerebrum_core::ensure_manifest::ensure_manifest;
use cerebrum_core::manifest::{manifest_path, read_manifest, write_manifest, Manifest};
use cerebrum_core::schema_probe::read_embedding_width;
use cerebrum_core::error::CerebrumError;

/// Build a minimal Arrow schema for a table with the given embedding dimension.
fn make_schema(dim: usize) -> Arc<Schema> {
    let vector_field = Field::new("item", DataType::Float32, true);
    Arc::new(Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(vector_field), dim as i32),
            false,
        ),
    ])))
}

async fn open_conn(path: &Path) -> lancedb::Connection {
    let s = path.to_str().expect("path is UTF-8");
    lancedb::connect(s)
        .execute()
        .await
        .expect("lancedb connect")
}

// ── manifest round-trip ──────────────────────────────────────────────────────

#[tokio::test]
async fn manifest_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = manifest_path(dir.path(), "memories");
    let m = Manifest { model: "nomic-embed-text".to_string(), dim: 768 };
    write_manifest(&path, &m).unwrap();
    let read = read_manifest(&path).unwrap();
    assert_eq!(read, Some(m));
}

// ── ensure_manifest: absent table, no manifest → writes manifest ─────────────

#[tokio::test]
async fn absent_table_no_manifest_writes_manifest() {
    let dir = TempDir::new().unwrap();
    let conn = open_conn(dir.path()).await;
    let mut config = Config::default();
    config.db_path = dir.path().to_path_buf();
    config.table_name = "memories".to_string();
    config.embedding_dim = 768;
    config.embed_model = "nomic-embed-text".to_string();

    ensure_manifest(&conn, dir.path(), &config).await.unwrap();

    let mpath = manifest_path(dir.path(), "memories");
    let m = read_manifest(&mpath).unwrap().expect("manifest written");
    assert_eq!(m.dim, 768);
    assert_eq!(m.model, "nomic-embed-text");
}

// ── schema_probe: zero-row table → Ok(Some(768)) ─────────────────────────────

#[tokio::test]
async fn zero_row_table_probe_returns_width() {
    let dir = TempDir::new().unwrap();
    let conn = open_conn(dir.path()).await;

    // Create a zero-row table with dim=768.
    conn.create_empty_table("memories", make_schema(768))
        .execute()
        .await
        .expect("create table");

    let width = read_embedding_width(&conn, "memories").await.unwrap();
    assert_eq!(width, Some(768));

    // Also verify ensure_manifest stamps the probed width.
    let mut config = Config::default();
    config.db_path = dir.path().to_path_buf();
    config.table_name = "memories".to_string();
    config.embedding_dim = 768;
    config.embed_model = "nomic-embed-text".to_string();

    ensure_manifest(&conn, dir.path(), &config).await.unwrap();

    let mpath = manifest_path(dir.path(), "memories");
    let m = read_manifest(&mpath).unwrap().expect("manifest written");
    assert_eq!(m.dim, 768, "manifest must be stamped with the probed width");
}

// ── ensure_manifest: dim mismatch → pinned Validation error ─────────────────

#[tokio::test]
async fn dim_mismatch_returns_validation_error() {
    let dir = TempDir::new().unwrap();
    let conn = open_conn(dir.path()).await;

    // Create a 768-dim table.
    conn.create_empty_table("memories", make_schema(768))
        .execute()
        .await
        .expect("create table");

    // First call: no manifest present — stamps probed width (768) and returns Ok.
    let mut config = Config::default();
    config.db_path = dir.path().to_path_buf();
    config.table_name = "memories".to_string();
    config.embedding_dim = 768;
    config.embed_model = "nomic-embed-text".to_string();
    ensure_manifest(&conn, dir.path(), &config).await.unwrap();

    // Now manifest exists with dim=768; reconfigure to dim=1024 → mismatch.
    config.embedding_dim = 1024;
    config.embed_model = "qwen3-embedding:0.6b".to_string();

    let err = ensure_manifest(&conn, dir.path(), &config).await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("cerebrum-reembed"), "error must mention cerebrum-reembed: {msg}");
    assert!(msg.contains("CEREBRUM_TABLE_NAME"), "error must mention CEREBRUM_TABLE_NAME: {msg}");
    assert!(matches!(err, CerebrumError::Validation(_)));
}
