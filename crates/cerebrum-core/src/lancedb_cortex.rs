use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    array::ArrayRef,
    builder::{FixedSizeListBuilder, Float32Builder, StringBuilder},
    cast::AsArray,
    types::Float32Type,
    RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType, Table};
use serde::{Deserialize, Serialize};

use crate::embedder::Embedder;
use crate::error::{CerebrumError, Result};
use crate::models::{MemoryEntry, MemoryId, MemoryScope};
use crate::traits::MemoryStore;

/// Escape single quotes for safe insertion in a LanceDB SQL filter predicate.
///
/// Replaces every `'` with `''` and wraps the result in single quotes, making
/// the string safe to embed in `DELETE WHERE id = …` or `WHERE scope = …` filters.
fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// Build the Arrow schema for the `memories` table.
///
/// Column order must match the `RecordBatch` built in `store()`.
fn schema(dim: usize) -> Arc<Schema> {
    let vector_field = Field::new("item", DataType::Float32, true);
    let fields = vec![
        Field::new("id",                DataType::Utf8, false),
        Field::new("content",           DataType::Utf8, false),
        Field::new("salience",          DataType::Float32, false),
        Field::new("timestamp",         DataType::Utf8, false),
        Field::new("source_session_id", DataType::Utf8, true),   // nullable
        Field::new("scope",             DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(vector_field), dim as i32),
            false,
        ),
        Field::new("metadata_json",     DataType::Utf8, false),
    ];
    Arc::new(Schema::new(Fields::from(fields)))
}

/// Schema for storing memories in LanceDB.
///
/// This struct represents how memories are stored in the vector database.
/// It includes all fields from MemoryEntry plus the embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanceDBMemoryRecord {
    /// Unique identifier for this memory.
    pub id: String,
    /// The text content of the memory.
    pub content: String,
    /// Importance score (0.0–1.0) for ranking and promotion decisions.
    pub salience: f32,
    /// When this memory was created (ISO 8601 string).
    pub timestamp: String,
    /// Session ID where this memory originated (if applicable).
    pub source_session_id: Option<String>,
    /// Scope or visibility of this memory (string representation).
    pub scope: String,
    /// 384-dimensional embedding vector (BGE-small).
    pub embedding: Vec<f32>,
    /// Arbitrary metadata as JSON string.
    pub metadata_json: String,
}

impl LanceDBMemoryRecord {
    /// Convert from MemoryEntry to LanceDBMemoryRecord.
    pub fn from_entry(entry: &MemoryEntry) -> Result<Self> {
        let embedding = entry.embedding.clone().ok_or_else(|| {
            CerebrumError::Validation("Memory entry missing embedding".to_string())
        })?;

        Ok(Self {
            id: entry.id.to_string(),
            content: entry.content.clone(),
            salience: entry.salience,
            timestamp: entry.timestamp.to_rfc3339(),
            source_session_id: entry.source_session_id.clone(),
            scope: entry.scope.as_str(),
            embedding,
            metadata_json: serde_json::to_string(&entry.metadata)
                .map_err(|e| CerebrumError::Serialization(e.to_string()))?,
        })
    }

    /// Convert from LanceDBMemoryRecord back to MemoryEntry.
    pub fn to_entry(&self) -> Result<MemoryEntry> {
        let id = MemoryId::from_string(&self.id)?;
        let timestamp = chrono::DateTime::parse_from_rfc3339(&self.timestamp)
            .map_err(|e| CerebrumError::Validation(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&chrono::Utc);

        let scope = parse_scope_string(&self.scope)?;

        let metadata = serde_json::from_str(&self.metadata_json)
            .map_err(|e| CerebrumError::Serialization(e.to_string()))?;

        Ok(MemoryEntry {
            id,
            content: self.content.clone(),
            metadata,
            timestamp,
            salience: self.salience,
            tier: crate::models::MemoryTier::Cortex,
            embedding: Some(self.embedding.clone()),
            source_session_id: self.source_session_id.clone(),
            scope,
        })
    }
}

/// Parse a scope string back into a MemoryScope enum.
fn parse_scope_string(scope_str: &str) -> Result<MemoryScope> {
    if scope_str == "global" {
        Ok(MemoryScope::Global)
    } else if let Some(user_id) = scope_str.strip_prefix("user:") {
        Ok(MemoryScope::User(user_id.to_string()))
    } else if let Some(agent_id) = scope_str.strip_prefix("agent:") {
        Ok(MemoryScope::Agent(agent_id.to_string()))
    } else if let Some(session_id) = scope_str.strip_prefix("session:") {
        Ok(MemoryScope::Session(session_id.to_string()))
    } else {
        Err(CerebrumError::Validation(format!(
            "Invalid scope string: {}",
            scope_str
        )))
    }
}

/// Persistent long-term memory storage backed by LanceDB (Cortex tier).
///
/// Stores memories in a vector database for efficient semantic search and
/// persistent storage across sessions. Supports salience-based ranking.
///
/// The LanceDB `Connection` is held for the lifetime of the store; the
/// `Table` handle is re-opened on each operation to avoid stale snapshots.
pub struct LanceDBCortex {
    /// Held LanceDB connection.
    conn: Connection,
    /// Table name for storing memories.
    table_name: String,
    /// Embedding dimension (384 for nomic-embed-text).
    embedding_dim: usize,
    /// Embedder for generating query embeddings.
    embedder: Arc<dyn Embedder>,
}

impl LanceDBCortex {
    /// Open (or create) the memories table at `db_path`.
    ///
    /// Mirrors athenaeum `Store::open`. Asserts that `embedder.dimension() == dim`
    /// at construction time to fail-fast before any schema-corrupting insert.
    ///
    /// # Arguments
    /// * `db_path`    – Path to the LanceDB directory (relative or absolute).
    /// * `table_name` – Name of the table within the database.
    /// * `dim`        – Expected embedding dimension; must match `embedder.dimension()`.
    /// * `embedder`   – Embedder used for query embedding during retrieval.
    pub async fn new(
        db_path: &Path,
        table_name: &str,
        dim: usize,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self> {
        // Fail-fast: embedder dimension must match schema dimension.
        let embedder_dim = embedder.dimension();
        if embedder_dim != dim {
            return Err(CerebrumError::Validation(format!(
                "Embedder dimension ({}) does not match schema dimension ({})",
                embedder_dim, dim
            )));
        }

        let path = db_path.to_str().ok_or_else(|| {
            CerebrumError::Database("non-UTF-8 db_path".to_string())
        })?;

        let conn = lancedb::connect(path)
            .execute()
            .await
            .map_err(|e| CerebrumError::Database(format!("Failed to connect to LanceDB: {}", e)))?;

        // Create the table if it does not yet exist.
        let existing = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| CerebrumError::Database(format!("Failed to list tables: {}", e)))?;

        if !existing.contains(&table_name.to_string()) {
            conn.create_empty_table(table_name, schema(dim))
                .execute()
                .await
                .map_err(|e| CerebrumError::Database(format!("Failed to create table: {}", e)))?;
        }

        Ok(Self {
            conn,
            table_name: table_name.to_string(),
            embedding_dim: dim,
            embedder,
        })
    }

    /// Open the table. Re-opened per operation to avoid stale snapshots.
    ///
    /// Mirrors athenaeum `Store::table()`.
    async fn table(&self) -> Result<Table> {
        self.conn
            .open_table(&self.table_name)
            .execute()
            .await
            .map_err(|e| CerebrumError::Database(format!("Failed to open table: {}", e)))
    }

    /// Calculate cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Build a single-row RecordBatch from a LanceDBMemoryRecord.
    fn record_to_batch(record: &LanceDBMemoryRecord, dim: usize) -> Result<RecordBatch> {
        let schema = schema(dim);

        let mut id_b           = StringBuilder::new();
        let mut content_b      = StringBuilder::new();
        let mut salience_b     = arrow_array::builder::Float32Builder::new();
        let mut timestamp_b    = StringBuilder::new();
        let mut session_b      = StringBuilder::new();
        let mut scope_b        = StringBuilder::new();
        let mut embedding_b    = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
        let mut metadata_b     = StringBuilder::new();

        id_b.append_value(&record.id);
        content_b.append_value(&record.content);
        salience_b.append_value(record.salience);
        timestamp_b.append_value(&record.timestamp);
        match &record.source_session_id {
            Some(s) => session_b.append_value(s),
            None    => session_b.append_null(),
        }
        scope_b.append_value(&record.scope);
        for &v in &record.embedding {
            embedding_b.values().append_value(v);
        }
        embedding_b.append(true);
        metadata_b.append_value(&record.metadata_json);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_b.finish())        as ArrayRef,
                Arc::new(content_b.finish())   as ArrayRef,
                Arc::new(salience_b.finish())  as ArrayRef,
                Arc::new(timestamp_b.finish()) as ArrayRef,
                Arc::new(session_b.finish())   as ArrayRef,
                Arc::new(scope_b.finish())     as ArrayRef,
                Arc::new(embedding_b.finish()) as ArrayRef,
                Arc::new(metadata_b.finish())  as ArrayRef,
            ],
        )
        .map_err(|e| CerebrumError::Database(format!("Failed to build RecordBatch: {}", e)))?;

        Ok(batch)
    }

    /// Decode a RecordBatch into a Vec of LanceDBMemoryRecord.
    fn batch_to_records(batch: &RecordBatch) -> Result<Vec<LanceDBMemoryRecord>> {
        let n = batch.num_rows();
        if n == 0 {
            return Ok(vec![]);
        }

        let id_col       = batch.column_by_name("id").ok_or_else(|| CerebrumError::Database("missing 'id' column".into()))?.as_string::<i32>();
        let content_col  = batch.column_by_name("content").ok_or_else(|| CerebrumError::Database("missing 'content' column".into()))?.as_string::<i32>();
        let salience_col = batch.column_by_name("salience").ok_or_else(|| CerebrumError::Database("missing 'salience' column".into()))?.as_primitive::<Float32Type>();
        let ts_col       = batch.column_by_name("timestamp").ok_or_else(|| CerebrumError::Database("missing 'timestamp' column".into()))?.as_string::<i32>();
        let session_col  = batch.column_by_name("source_session_id").ok_or_else(|| CerebrumError::Database("missing 'source_session_id' column".into()))?.as_string::<i32>();
        let scope_col    = batch.column_by_name("scope").ok_or_else(|| CerebrumError::Database("missing 'scope' column".into()))?.as_string::<i32>();
        let emb_col      = batch.column_by_name("embedding").ok_or_else(|| CerebrumError::Database("missing 'embedding' column".into()))?.as_fixed_size_list();
        let meta_col     = batch.column_by_name("metadata_json").ok_or_else(|| CerebrumError::Database("missing 'metadata_json' column".into()))?.as_string::<i32>();

        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let emb_values = emb_col.value(i);
            let emb_f32 = emb_values.as_primitive::<Float32Type>();
            let embedding: Vec<f32> = (0..emb_f32.len()).map(|j| emb_f32.value(j)).collect();

            records.push(LanceDBMemoryRecord {
                id:                id_col.value(i).to_string(),
                content:           content_col.value(i).to_string(),
                salience:          salience_col.value(i),
                timestamp:         ts_col.value(i).to_string(),
                source_session_id: if session_col.is_null(i) { None } else { Some(session_col.value(i).to_string()) },
                scope:             scope_col.value(i).to_string(),
                embedding,
                metadata_json:     meta_col.value(i).to_string(),
            });
        }
        Ok(records)
    }

    /// Search memories by salience (highest first).
    pub async fn search_by_salience(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let table = self.table().await?;
        let row_count = table.count_rows(None).await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        let stream = table.query().execute().await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        let batches: Vec<RecordBatch> = stream.try_collect().await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        let mut records: Vec<LanceDBMemoryRecord> = batches.iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .collect();

        records.sort_by(|a, b| b.salience.partial_cmp(&a.salience).unwrap_or(std::cmp::Ordering::Equal));

        records.iter()
            .take(limit)
            .map(|r| r.to_entry())
            .collect()
    }
}

#[async_trait]
impl MemoryStore for LanceDBCortex {
    /// Store a memory entry using an atomic upsert keyed on `id`.
    ///
    /// Uses LanceDB `merge_insert` so re-storing the same `MemoryId` updates
    /// the existing row rather than duplicating it, with no crash window.
    async fn store(&self, entry: MemoryEntry) -> Result<()> {
        let record = LanceDBMemoryRecord::from_entry(&entry)?;
        let schema = schema(self.embedding_dim);
        let batch  = Self::record_to_batch(&record, self.embedding_dim)?;

        let reader = RecordBatchIterator::new(
            vec![Ok(batch)],
            schema,
        );

        let table = self.table().await?;
        let mut mi = table.merge_insert(&["id"]);
        mi.when_matched_update_all(None).when_not_matched_insert_all();
        mi.execute(Box::new(reader))
            .await
            .map_err(|e| CerebrumError::Database(format!("merge_insert failed: {}", e)))?;

        Ok(())
    }

    /// Retrieve memories by semantic similarity, blended with salience.
    ///
    /// Performs an exact full-scan so that the blend `0.7*similarity + 0.3*salience`
    /// is computed over every row — no memory can be dropped by a vector pre-filter.
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let query_embedding = self.embedder.embed(query).await?;

        let table = self.table().await?;
        let row_count = table.count_rows(None).await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        let stream = table.query().execute().await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        let batches: Vec<RecordBatch> = stream.try_collect().await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        let mut scored: Vec<(LanceDBMemoryRecord, f32)> = batches.iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .map(|record| {
                let sim   = Self::cosine_similarity(&query_embedding, &record.embedding);
                let score = sim * 0.7 + record.salience * 0.3;
                (record, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter()
            .take(limit)
            .map(|(r, _)| r.to_entry())
            .collect()
    }

    /// Retrieve memories filtered by scope, then blended-score ranked.
    ///
    /// Pushes a coarse SQL predicate to LanceDB (reducing rows fetched),
    /// then applies the precise `MemoryScope::matches` logic in Rust to
    /// handle the bidirectional Global-matches-all semantic.
    async fn retrieve_by_scope(
        &self,
        query: &str,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let query_embedding = self.embedder.embed(query).await?;

        let table = self.table().await?;
        let row_count = table.count_rows(None).await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        // Coarse SQL pushdown: global matches all, specific scopes match themselves + global.
        let stream = match scope {
            MemoryScope::Global => {
                // Global scope matches everything — no filter needed.
                table.query().execute().await
                    .map_err(|e| CerebrumError::Database(e.to_string()))?
            }
            _ => {
                let predicate = format!(
                    "scope = 'global' OR scope = {}",
                    sql_quote(&scope.as_str())
                );
                table.query().only_if(predicate).execute().await
                    .map_err(|e| CerebrumError::Database(e.to_string()))?
            }
        };

        let batches: Vec<RecordBatch> = stream.try_collect().await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        let mut scored: Vec<(LanceDBMemoryRecord, f32)> = batches.iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .filter_map(|record| {
                // Precise scope match in Rust (handles bidirectional Global logic).
                let entry = record.to_entry().ok()?;
                if !scope.matches(&entry.scope) {
                    return None;
                }
                let sim   = Self::cosine_similarity(&query_embedding, &record.embedding);
                let score = sim * 0.7 + record.salience * 0.3;
                Some((record, score))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter()
            .take(limit)
            .map(|(r, _)| r.to_entry())
            .collect()
    }

    /// Delete a memory by ID.
    async fn delete(&self, id: &MemoryId) -> Result<()> {
        let predicate = format!("id = {}", sql_quote(&id.to_string()));
        self.table().await?
            .delete(&predicate)
            .await
            .map_err(|e| CerebrumError::Database(format!("delete failed: {}", e)))?;
        Ok(())
    }

    /// List all memories in the store.
    ///
    /// Explicit override — the trait default would embed the literal `"*"` string.
    async fn list(&self) -> Result<Vec<MemoryEntry>> {
        let table = self.table().await?;
        let row_count = table.count_rows(None).await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        let stream = table.query().execute().await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        let batches: Vec<RecordBatch> = stream.try_collect().await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        batches.iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .map(|r| r.to_entry())
            .collect()
    }

    /// Get the number of memories in the store.
    ///
    /// Explicit override — the trait default would call list() which embeds `"*"`.
    async fn len(&self) -> Result<usize> {
        self.table().await?
            .count_rows(None)
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))
    }

    /// Check if the store is empty.
    ///
    /// Explicit override — the trait default would call len() via list() via retrieve("*"...).
    async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::MockEmbedder;
    use crate::models::MemoryTier;
    use tempfile;

    #[tokio::test]
    async fn test_lancedb_cortex_new() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let result = LanceDBCortex::new(dir.path(), "memories", 384, embedder).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_store_and_retrieve() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry.clone()).await.unwrap();

        let results = cortex.retrieve("test", 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_len() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry).await.unwrap();

        let len = cortex.len().await.unwrap();
        assert!(len > 0);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_delete() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        let id = MemoryId::new();
        let entry = MemoryEntry::builder(id, "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry).await.unwrap();
        cortex.delete(&id).await.unwrap();

        let results = cortex.retrieve("test", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope("test", &MemoryScope::User("user1".to_string()), 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope("test", &MemoryScope::User("user2".to_string()), 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope_global() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope("test", &MemoryScope::Global, 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        assert!(cortex.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_list() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry).await.unwrap();

        let entries = cortex.list().await.unwrap();
        assert!(!entries.is_empty());
    }

    #[tokio::test]
    async fn test_cortex_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let id = MemoryId::new();

        {
            let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
                .await
                .unwrap();
            let entry = MemoryEntry::builder(id, "persistent memory".to_string())
                .embedding(vec![0.1; 384])
                .tier(MemoryTier::Cortex)
                .build();
            cortex.store(entry).await.unwrap();
        }
        // LanceDBCortex dropped here — connection closed.

        {
            let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
                .await
                .unwrap();
            let results = cortex.list().await.unwrap();
            assert_eq!(results.len(), 1, "memory must survive process restart");
            assert_eq!(results[0].id, id);
            assert_eq!(results[0].content, "persistent memory");
        }
    }

    #[tokio::test]
    async fn test_cortex_dimension_mismatch_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new()); // reports dimension 384
        // Request dim=768 — must be rejected before any LanceDB call.
        let result = LanceDBCortex::new(dir.path(), "memories", 768, embedder).await;
        assert!(result.is_err(), "mismatched dimension must error at construction");
    }

    #[tokio::test]
    async fn test_cortex_search_empty_table_returns_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder)
            .await
            .unwrap();
        // All read operations on a fresh store must succeed with empty results.
        assert_eq!(cortex.retrieve("anything", 10).await.unwrap(), vec![]);
        assert_eq!(cortex.list().await.unwrap(), vec![]);
        assert_eq!(cortex.len().await.unwrap(), 0);
        assert!(cortex.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn test_cortex_store_upserts_on_same_id() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder)
            .await
            .unwrap();
        let id = MemoryId::new();

        cortex.store(
            MemoryEntry::builder(id, "version 1".to_string())
                .embedding(vec![0.1; 384])
                .tier(MemoryTier::Cortex)
                .build(),
        ).await.unwrap();

        cortex.store(
            MemoryEntry::builder(id, "version 2".to_string())
                .embedding(vec![0.2; 384])
                .tier(MemoryTier::Cortex)
                .build(),
        ).await.unwrap();

        let entries = cortex.list().await.unwrap();
        assert_eq!(entries.len(), 1, "upsert must not duplicate rows");
        assert_eq!(entries[0].content, "version 2");
    }

    #[tokio::test]
    async fn test_cortex_list_and_len_work_without_embedding_query() {
        // MockEmbedder rejects empty strings. If list() or len() internally called
        // retrieve("*", usize::MAX), this test would fail. It passing proves the
        // explicit overrides are in place and the defaults are not used.
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder)
            .await
            .unwrap();
        // These must succeed without embedding anything.
        let _ = cortex.list().await.expect("list() must not call embed()");
        let _ = cortex.len().await.expect("len() must not call embed()");
        let _ = cortex.is_empty().await.expect("is_empty() must not call embed()");
    }

    #[tokio::test]
    async fn test_lancedb_memory_record_conversion() {
        let id = MemoryId::new();
        let entry = MemoryEntry::builder(id, "test content".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::Global)
            .build();

        let record = LanceDBMemoryRecord::from_entry(&entry).unwrap();
        let converted = record.to_entry().unwrap();

        assert_eq!(converted.id, entry.id);
        assert_eq!(converted.content, entry.content);
        assert_eq!(converted.salience, entry.salience);
    }

    #[test]
    fn test_parse_scope_string_global() {
        let result = parse_scope_string("global");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), MemoryScope::Global));
    }

    #[test]
    fn test_parse_scope_string_user() {
        let result = parse_scope_string("user:alice");
        assert!(result.is_ok());
        match result.unwrap() {
            MemoryScope::User(id) => assert_eq!(id, "alice"),
            _ => panic!("Expected User scope"),
        }
    }

    #[test]
    fn test_parse_scope_string_agent() {
        let result = parse_scope_string("agent:bot123");
        assert!(result.is_ok());
        match result.unwrap() {
            MemoryScope::Agent(id) => assert_eq!(id, "bot123"),
            _ => panic!("Expected Agent scope"),
        }
    }

    #[test]
    fn test_parse_scope_string_session() {
        let result = parse_scope_string("session:sess456");
        assert!(result.is_ok());
        match result.unwrap() {
            MemoryScope::Session(id) => assert_eq!(id, "sess456"),
            _ => panic!("Expected Session scope"),
        }
    }

    #[test]
    fn test_parse_scope_string_invalid() {
        let result = parse_scope_string("invalid:scope");
        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert!(similarity.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert!((similarity + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_magnitude() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = LanceDBCortex::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_search_by_salience() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        // Store entries with different salience values
        let entry1 = MemoryEntry::builder(MemoryId::new(), "high salience".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .salience(0.9)
            .build();

        let entry2 = MemoryEntry::builder(MemoryId::new(), "low salience".to_string())
            .embedding(vec![0.2; 384])
            .tier(MemoryTier::Cortex)
            .salience(0.1)
            .build();

        cortex.store(entry1).await.unwrap();
        cortex.store(entry2).await.unwrap();

        let results = cortex.search_by_salience(10).await.unwrap();
        assert_eq!(results.len(), 2);
        // First result should have higher salience
        assert!(results[0].salience >= results[1].salience);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_search_by_salience_limit() {
        let dir = tempfile::tempdir().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384, embedder.clone())
            .await
            .unwrap();

        // Store multiple entries
        for i in 0..5 {
            let entry = MemoryEntry::builder(MemoryId::new(), format!("entry {}", i))
                .embedding(vec![0.1; 384])
                .tier(MemoryTier::Cortex)
                .salience(i as f32 * 0.2)
                .build();
            cortex.store(entry).await.unwrap();
        }

        let results = cortex.search_by_salience(2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_lancedb_memory_record_from_entry_missing_embedding() {
        let entry = MemoryEntry::builder(MemoryId::new(), "test".to_string())
            .tier(MemoryTier::Cortex)
            .build();

        let result = LanceDBMemoryRecord::from_entry(&entry);
        assert!(result.is_err());
    }

    #[test]
    fn test_lancedb_memory_record_all_scopes() {
        let scopes = vec![
            MemoryScope::Global,
            MemoryScope::User("user1".to_string()),
            MemoryScope::Agent("agent1".to_string()),
            MemoryScope::Session("session1".to_string()),
        ];

        for scope in scopes {
            let entry = MemoryEntry::builder(MemoryId::new(), "test".to_string())
                .embedding(vec![0.1; 384])
                .tier(MemoryTier::Cortex)
                .scope(scope.clone())
                .build();

            let record = LanceDBMemoryRecord::from_entry(&entry).unwrap();
            let converted = record.to_entry().unwrap();
            assert_eq!(converted.scope, scope);
        }
    }
}
