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
use cerebrum_core::error::CerebrumError;
use cerebrum_core::manifest::{manifest_path, read_manifest, write_manifest, Manifest};
use cerebrum_core::schema_probe::read_embedding_width;

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
    let m = Manifest {
        model: "nomic-embed-text".to_string(),
        dim: 768,
    };
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

    let err = ensure_manifest(&conn, dir.path(), &config)
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("cerebrum-reembed"),
        "error must mention cerebrum-reembed: {msg}"
    );
    assert!(
        msg.contains("CEREBRUM_TABLE_NAME"),
        "error must mention CEREBRUM_TABLE_NAME: {msg}"
    );
    assert!(matches!(err, CerebrumError::Validation(_)));
}

// ── ensure_manifest: absent table + existing manifest with wrong dim → error ──

#[tokio::test]
async fn absent_table_existing_manifest_dim_mismatch_errors() {
    let dir = TempDir::new().unwrap();
    let conn = open_conn(dir.path()).await;

    // Write a manifest claiming dim=768 before the table exists.
    let mpath = manifest_path(dir.path(), "memories");
    write_manifest(&mpath, &Manifest { model: "nomic-embed-text".to_string(), dim: 768 }).unwrap();

    let mut config = Config::default();
    config.db_path = dir.path().to_path_buf();
    config.table_name = "memories".to_string();
    config.embedding_dim = 1024; // disagrees with manifest
    config.embed_model = "qwen3-embedding:0.6b".to_string();

    let err = ensure_manifest(&conn, dir.path(), &config).await.unwrap_err();
    assert!(matches!(err, CerebrumError::Validation(_)));
    assert!(err.to_string().contains("cerebrum-reembed"));
}

// ── ensure_manifest: table exists + manifest present + dims match → Ok ────────

#[tokio::test]
async fn table_exists_manifest_present_matching_dim_ok() {
    let dir = TempDir::new().unwrap();
    let conn = open_conn(dir.path()).await;

    conn.create_empty_table("memories", make_schema(768))
        .execute()
        .await
        .expect("create table");

    // Pre-write a matching manifest.
    let mpath = manifest_path(dir.path(), "memories");
    write_manifest(&mpath, &Manifest { model: "nomic-embed-text".to_string(), dim: 768 }).unwrap();

    let mut config = Config::default();
    config.db_path = dir.path().to_path_buf();
    config.table_name = "memories".to_string();
    config.embedding_dim = 768;
    config.embed_model = "nomic-embed-text".to_string();

    ensure_manifest(&conn, dir.path(), &config).await.unwrap();
}

// ── manifest: corrupt file returns Err ───────────────────────────────────────

#[test]
fn read_manifest_corrupt_file_errors() {
    let dir = TempDir::new().unwrap();
    let path = manifest_path(dir.path(), "bad");
    std::fs::write(&path, b"not json at all {{{{").unwrap();
    let result = read_manifest(&path);
    assert!(result.is_err());
}

// ── manifest: write to unwritable path returns Err ────────────────────────────

#[test]
fn write_manifest_bad_path_errors() {
    let path = std::path::PathBuf::from("/nonexistent/dir/x.manifest.json");
    let m = Manifest { model: "m".to_string(), dim: 1 };
    let result = write_manifest(&path, &m);
    assert!(result.is_err());
}

// ── schema_probe: table with non-FixedSizeList embedding field → Err ──────────

#[tokio::test]
async fn schema_probe_wrong_field_type_errors() {
    let dir = TempDir::new().unwrap();
    let conn = open_conn(dir.path()).await;

    // Create a table where "embedding" is plain Utf8, not FixedSizeList.
    let bad_schema = Arc::new(Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("embedding", DataType::Utf8, false),
    ])));
    conn.create_empty_table("bad_table", bad_schema)
        .execute()
        .await
        .expect("create table");

    let err = read_embedding_width(&conn, "bad_table").await.unwrap_err();
    assert!(matches!(err, CerebrumError::Validation(_)));
}

// ── schema_probe: table with no embedding field → Err ────────────────────────

#[tokio::test]
async fn schema_probe_missing_embedding_field_errors() {
    let dir = TempDir::new().unwrap();
    let conn = open_conn(dir.path()).await;

    let no_embed_schema = Arc::new(Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
    ])));
    conn.create_empty_table("no_embed", no_embed_schema)
        .execute()
        .await
        .expect("create table");

    let err = read_embedding_width(&conn, "no_embed").await.unwrap_err();
    assert!(matches!(err, CerebrumError::Validation(_)));
}
