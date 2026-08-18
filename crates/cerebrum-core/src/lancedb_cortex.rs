use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    array::ArrayRef,
    builder::{FixedSizeListBuilder, Float32Builder, StringBuilder},
    cast::AsArray,
    types::Float32Type,
    Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};
use serde::{Deserialize, Serialize};

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
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("salience", DataType::Float32, false),
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("source_session_id", DataType::Utf8, true), // nullable
        Field::new("scope", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(vector_field), dim as i32),
            false,
        ),
        Field::new("metadata_json", DataType::Utf8, false),
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
    } else if let Some(plan_id) = scope_str.strip_prefix("plan:") {
        Ok(MemoryScope::Plan(plan_id.to_string()))
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
/// # Vector-Based Operation
/// The LanceDB `Cortex` store operates on query **vectors**, never raw text.
/// The orchestrator owns the embedder and passes a pre-computed query vector
/// into `retrieve()` / `retrieve_by_scope()`. This keeps embedding concerns
/// out of the storage layer entirely.
///
/// # Persistence
/// Unlike Synapse (in-memory), Cortex persists memories to disk via LanceDB.
/// Memories survive session restarts and can be searched across multiple sessions.
/// The LanceDB table is stored at `{db_path}/data/{table_name}.lance`.
///
/// # Schema & Dimension
/// The table schema is fixed at creation time and includes:
/// - `id`: Unique memory identifier
/// - `content`: Original memory text (without prefixes)
/// - `salience`: Importance score (0.0-1.0)
/// - `timestamp`: Creation time (ISO 8601)
/// - `source_session_id`: Session where memory originated
/// - `scope`: Memory visibility (Global, User, Agent, Session)
/// - `embedding`: 1024-dimensional vector (qwen3-embedding:0.6b)
/// - `metadata_json`: Arbitrary metadata as JSON
///
/// **Important:** Changing `embedding_dim` requires wiping the table schema.
/// See README.md "Schema Migration" section for details.
///
/// # Connection Management
/// The LanceDB `Connection` is held for the lifetime of the store; the
/// `Table` handle is re-opened on each operation to avoid stale snapshots.
/// This ensures we always read the latest data.
pub struct LanceDBCortex {
    /// Held LanceDB connection (persistent across operations).
    conn: Connection,
    /// Table name for storing memories (default: "memories").
    table_name: String,
    /// Embedding dimension (1024 for qwen3-embedding:0.6b).
    /// Must match the dimension of vectors passed to `retrieve()`.
    embedding_dim: usize,
}

impl LanceDBCortex {
    /// Open (or create) the memories table at `db_path`.
    ///
    /// Creates the directory (and any missing parents) if it does not already
    /// exist before connecting to LanceDB, so callers never need to create the
    /// directory themselves.
    ///
    /// Mirrors athenaeum `Store::open`. The store operates purely on vectors —
    /// it holds no embedder. The orchestrator owns the embedder and validates
    /// the embedding dimension against `dim` before constructing the store.
    ///
    /// # Arguments
    /// * `db_path`    – Path to the LanceDB directory (relative or absolute).
    /// * `table_name` – Name of the table within the database.
    /// * `dim`        – Expected embedding dimension for the table schema.
    pub async fn new(db_path: &Path, table_name: &str, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(db_path)
            .map_err(|e| CerebrumError::Database(format!("Failed to create db_path: {}", e)))?;

        let path = db_path
            .to_str()
            .ok_or_else(|| CerebrumError::Database("non-UTF-8 db_path".to_string()))?;

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

        // Fail-closed dimension guard: if the table already existed, probe its
        // actual embedding width and refuse to open it when it disagrees with
        // the requested `dim`. Prevents silently operating on a wrong-dimension
        // table (e.g. an old 768-dim `memories` table while the embedder now
        // produces 1024-dim vectors). A freshly created table probes as its
        // requested `dim`; an absent table (None) is skipped.
        if let Some(stored) = crate::schema_probe::read_embedding_width(&conn, table_name).await? {
            if stored != dim {
                return Err(CerebrumError::Validation(format!(
                    "table '{table_name}' stores {stored}-dim embeddings but this process expects {dim}-dim; run `cerebrum-reembed` and set CEREBRUM_TABLE_NAME to the migrated table"
                )));
            }
        }

        Ok(Self {
            conn,
            table_name: table_name.to_string(),
            embedding_dim: dim,
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
    ///
    /// On a non-empty length mismatch the function warns once per process
    /// (via a `std::sync::Once` guard) and returns `0.0`, so a mixed-dimension
    /// table degrades loudly-but-once rather than flooding the logs. An empty
    /// input returns `0.0` silently.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        if a.len() != b.len() {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            let query = a.len();
            let stored = b.len();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    "cosine_similarity dimension mismatch: query={query} stored={stored} — table may need cerebrum-reembed"
                );
            });
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

        let mut id_b = StringBuilder::new();
        let mut content_b = StringBuilder::new();
        let mut salience_b = arrow_array::builder::Float32Builder::new();
        let mut timestamp_b = StringBuilder::new();
        let mut session_b = StringBuilder::new();
        let mut scope_b = StringBuilder::new();
        let mut embedding_b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
        let mut metadata_b = StringBuilder::new();

        id_b.append_value(&record.id);
        content_b.append_value(&record.content);
        salience_b.append_value(record.salience);
        timestamp_b.append_value(&record.timestamp);
        match &record.source_session_id {
            Some(s) => session_b.append_value(s),
            None => session_b.append_null(),
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
                Arc::new(id_b.finish()) as ArrayRef,
                Arc::new(content_b.finish()) as ArrayRef,
                Arc::new(salience_b.finish()) as ArrayRef,
                Arc::new(timestamp_b.finish()) as ArrayRef,
                Arc::new(session_b.finish()) as ArrayRef,
                Arc::new(scope_b.finish()) as ArrayRef,
                Arc::new(embedding_b.finish()) as ArrayRef,
                Arc::new(metadata_b.finish()) as ArrayRef,
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

        let id_col = batch
            .column_by_name("id")
            .ok_or_else(|| CerebrumError::Database("missing 'id' column".into()))?
            .as_string::<i32>();
        let content_col = batch
            .column_by_name("content")
            .ok_or_else(|| CerebrumError::Database("missing 'content' column".into()))?
            .as_string::<i32>();
        let salience_col = batch
            .column_by_name("salience")
            .ok_or_else(|| CerebrumError::Database("missing 'salience' column".into()))?
            .as_primitive::<Float32Type>();
        let ts_col = batch
            .column_by_name("timestamp")
            .ok_or_else(|| CerebrumError::Database("missing 'timestamp' column".into()))?
            .as_string::<i32>();
        let session_col = batch
            .column_by_name("source_session_id")
            .ok_or_else(|| CerebrumError::Database("missing 'source_session_id' column".into()))?
            .as_string::<i32>();
        let scope_col = batch
            .column_by_name("scope")
            .ok_or_else(|| CerebrumError::Database("missing 'scope' column".into()))?
            .as_string::<i32>();
        let emb_col = batch
            .column_by_name("embedding")
            .ok_or_else(|| CerebrumError::Database("missing 'embedding' column".into()))?
            .as_fixed_size_list();
        let meta_col = batch
            .column_by_name("metadata_json")
            .ok_or_else(|| CerebrumError::Database("missing 'metadata_json' column".into()))?
            .as_string::<i32>();

        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let emb_values = emb_col.value(i);
            let emb_f32 = emb_values.as_primitive::<Float32Type>();
            let embedding: Vec<f32> = (0..emb_f32.len()).map(|j| emb_f32.value(j)).collect();

            records.push(LanceDBMemoryRecord {
                id: id_col.value(i).to_string(),
                content: content_col.value(i).to_string(),
                salience: salience_col.value(i),
                timestamp: ts_col.value(i).to_string(),
                source_session_id: if session_col.is_null(i) {
                    None
                } else {
                    Some(session_col.value(i).to_string())
                },
                scope: scope_col.value(i).to_string(),
                embedding,
                metadata_json: meta_col.value(i).to_string(),
            });
        }
        Ok(records)
    }

    /// Search memories by salience (highest first).
    pub async fn search_by_salience(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let table = self.table().await?;
        let row_count = table
            .count_rows(None)
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        let stream = table
            .query()
            .execute()
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        let mut records: Vec<LanceDBMemoryRecord> = batches
            .iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .collect();

        records.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        records.iter().take(limit).map(|r| r.to_entry()).collect()
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
        let batch = Self::record_to_batch(&record, self.embedding_dim)?;

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let table = self.table().await?;
        let mut mi = table.merge_insert(&["id"]);
        mi.when_matched_update_all(None)
            .when_not_matched_insert_all();
        mi.execute(Box::new(reader))
            .await
            .map_err(|e| CerebrumError::Database(format!("merge_insert failed: {}", e)))?;

        Ok(())
    }

    /// Retrieve memories by semantic similarity blended with salience and provenance weights.
    ///
    /// Performs an exact full-scan so that the blend is computed over every row —
    /// no memory can be dropped by a vector pre-filter.
    async fn retrieve_scored(
        &self,
        query_vec: &[f32],
        limit: usize,
        prefer_project: Option<&str>,
    ) -> Result<Vec<crate::models::ScoredMemory>> {
        use crate::models::ScoredMemory;

        let table = self.table().await?;
        let row_count = table
            .count_rows(None)
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        let stream = table
            .query()
            .execute()
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        let mut scored: Vec<ScoredMemory> = batches
            .iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .filter_map(|record| {
                let entry = record.to_entry().ok()?;
                let sim = Self::cosine_similarity(query_vec, &record.embedding);
                let base = sim * 0.7 + entry.salience * 0.3;
                let score = base
                    * crate::provenance::status_weight(&entry.metadata)
                    * crate::provenance::project_weight(&entry.metadata, prefer_project);
                Some(ScoredMemory { entry, score })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored.into_iter().take(limit).collect())
    }

    /// Retrieve memories filtered by scope, then blended-score ranked with provenance weights.
    ///
    /// Pushes a coarse SQL predicate to LanceDB (reducing rows fetched),
    /// then applies the precise `MemoryScope::matches` logic in Rust to
    /// handle the bidirectional Global-matches-all semantic — UNLESS
    /// `exact_scope` is `true`, in which case both the SQL predicate and the
    /// Rust-side filter restrict to rows whose scope is *exactly* `scope`
    /// (global rows are excluded from the candidate set entirely, not just
    /// deprioritized). See [`crate::traits::MemoryStore::retrieve_by_scope_scored`]
    /// for why this matters.
    async fn retrieve_by_scope_scored(
        &self,
        query_vec: &[f32],
        scope: &MemoryScope,
        limit: usize,
        prefer_project: Option<&str>,
        exact_scope: bool,
    ) -> Result<Vec<crate::models::ScoredMemory>> {
        use crate::models::ScoredMemory;

        let table = self.table().await?;
        let row_count = table
            .count_rows(None)
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        // Coarse SQL pushdown: global matches all UNLESS exact_scope narrows
        // the candidate set to the requested scope only.
        let stream = match scope {
            MemoryScope::Global => {
                // Global scope matches everything — no filter needed.
                table
                    .query()
                    .execute()
                    .await
                    .map_err(|e| CerebrumError::Database(e.to_string()))?
            }
            _ if exact_scope => {
                let predicate = format!("scope = {}", sql_quote(&scope.as_str()));
                table
                    .query()
                    .only_if(predicate)
                    .execute()
                    .await
                    .map_err(|e| CerebrumError::Database(e.to_string()))?
            }
            _ => {
                let predicate =
                    format!("scope = 'global' OR scope = {}", sql_quote(&scope.as_str()));
                table
                    .query()
                    .only_if(predicate)
                    .execute()
                    .await
                    .map_err(|e| CerebrumError::Database(e.to_string()))?
            }
        };

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        let mut scored: Vec<ScoredMemory> = batches
            .iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .filter_map(|record| {
                let entry = record.to_entry().ok()?;
                // Precise scope match in Rust: exact-scope mode requires
                // strict equality (excludes Global bleed); normal mode uses
                // the bidirectional MemoryScope::matches semantic.
                let scope_ok = if exact_scope {
                    entry.scope == *scope
                } else {
                    scope.matches(&entry.scope)
                };
                if !scope_ok {
                    return None;
                }
                let sim = Self::cosine_similarity(query_vec, &record.embedding);
                let base = sim * 0.7 + entry.salience * 0.3;
                let score = base
                    * crate::provenance::status_weight(&entry.metadata)
                    * crate::provenance::project_weight(&entry.metadata, prefer_project);
                Some(ScoredMemory { entry, score })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored.into_iter().take(limit).collect())
    }

    /// Delete a memory by ID.
    async fn delete(&self, id: &MemoryId) -> Result<()> {
        let predicate = format!("id = {}", sql_quote(&id.to_string()));
        self.table()
            .await?
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
        let row_count = table
            .count_rows(None)
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        if row_count == 0 {
            return Ok(vec![]);
        }

        let stream = table
            .query()
            .execute()
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| CerebrumError::Database(e.to_string()))?;

        batches
            .iter()
            .flat_map(|b| Self::batch_to_records(b).unwrap_or_default())
            .map(|r| r.to_entry())
            .collect()
    }

    /// Get the number of memories in the store.
    ///
    /// Explicit override — the trait default would call list() which embeds `"*"`.
    async fn len(&self) -> Result<usize> {
        self.table()
            .await?
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
    use crate::models::MemoryTier;
    use tempfile;

    /// Constant query vector matching the stored test embeddings (dim 384).
    ///
    /// Tests pass query vectors directly now that the store no longer embeds.
    fn qvec() -> Vec<f32> {
        vec![0.1; 384]
    }

    #[tokio::test]
    async fn test_lancedb_cortex_new() {
        let dir = tempfile::tempdir().unwrap();
        let result = LanceDBCortex::new(dir.path(), "memories", 384).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_new_rejects_dim_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        // First construction creates a 768-dim table.
        let first = LanceDBCortex::new(dir.path(), "memories", 768).await;
        assert!(first.is_ok(), "creating a fresh 768-dim table must succeed");

        // Second construction on the SAME dir + table at 1024 must fail closed.
        let second = LanceDBCortex::new(dir.path(), "memories", 1024).await;
        match second {
            Ok(_) => panic!("expected a dimension-mismatch Validation error, got Ok"),
            Err(CerebrumError::Validation(msg)) => {
                assert!(
                    msg.contains("cerebrum-reembed"),
                    "validation message must mention cerebrum-reembed, got: {msg}"
                );
            }
            Err(other) => panic!("expected Validation error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_lancedb_cortex_new_reopens_matching_dim() {
        let dir = tempfile::tempdir().unwrap();
        let first = LanceDBCortex::new(dir.path(), "memories", 768).await;
        assert!(first.is_ok());
        // Re-opening the existing 768-dim table at the same dim must succeed.
        let second = LanceDBCortex::new(dir.path(), "memories", 768).await;
        assert!(
            second.is_ok(),
            "re-opening an existing table at the matching dim must succeed"
        );
    }

    #[test]
    fn test_cosine_similarity_dim_mismatch_returns_zero() {
        let a = vec![0.1_f32; 4];
        let b = vec![0.1_f32; 8];
        assert_eq!(LanceDBCortex::cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty_returns_zero() {
        let a: Vec<f32> = Vec::new();
        let b = vec![0.1_f32; 4];
        assert_eq!(LanceDBCortex::cosine_similarity(&a, &b), 0.0);
    }

    #[tokio::test]
    async fn test_lancedb_cortex_new_creates_nested_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        assert!(
            !nested.exists(),
            "nested path must not exist before the test"
        );
        let result = LanceDBCortex::new(&nested, "memories", 384).await;
        assert!(
            result.is_ok(),
            "LanceDBCortex::new should succeed for a nested path"
        );
        assert!(
            nested.exists(),
            "LanceDBCortex::new must create the nested directory"
        );
    }

    #[tokio::test]
    async fn test_lancedb_cortex_store_and_retrieve() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry.clone()).await.unwrap();

        let results = cortex.retrieve(&qvec(), 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_len() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
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
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();

        let id = MemoryId::new();
        let entry = MemoryEntry::builder(id, "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .build();

        cortex.store(entry).await.unwrap();
        cortex.delete(&id).await.unwrap();

        let results = cortex.retrieve(&qvec(), 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope(&qvec(), &MemoryScope::User("user1".to_string()), 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope(&qvec(), &MemoryScope::User("user2".to_string()), 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope_global() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();

        let entry = MemoryEntry::builder(MemoryId::new(), "test memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .build();

        cortex.store(entry).await.unwrap();

        let results = cortex
            .retrieve_by_scope(&qvec(), &MemoryScope::Global, 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_retrieve_by_scope_scored_exact_excludes_global() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();

        let global_entry = MemoryEntry::builder(MemoryId::new(), "global memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::Global)
            .salience(0.95)
            .build();
        let scoped_entry = MemoryEntry::builder(MemoryId::new(), "scoped memory".to_string())
            .embedding(vec![0.1; 384])
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::User("user1".to_string()))
            .salience(0.1)
            .build();

        cortex.store(global_entry).await.unwrap();
        cortex.store(scoped_entry).await.unwrap();

        // Non-exact: global is a candidate alongside the scoped entry.
        let non_exact = cortex
            .retrieve_by_scope_scored(
                &qvec(),
                &MemoryScope::User("user1".to_string()),
                10,
                None,
                false,
            )
            .await
            .unwrap();
        assert_eq!(
            non_exact.len(),
            2,
            "non-exact mode includes global as a candidate"
        );

        // Exact: global must be excluded entirely, only the scoped entry remains.
        let exact = cortex
            .retrieve_by_scope_scored(
                &qvec(),
                &MemoryScope::User("user1".to_string()),
                10,
                None,
                true,
            )
            .await
            .unwrap();
        assert_eq!(exact.len(), 1, "exact mode must exclude global entirely");
        assert_eq!(exact[0].entry.content, "scoped memory");
    }

    #[tokio::test]
    async fn test_lancedb_cortex_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();

        assert!(cortex.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn test_lancedb_cortex_list() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
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
        let id = MemoryId::new();

        {
            let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
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
            let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
                .await
                .unwrap();
            let results = cortex.list().await.unwrap();
            assert_eq!(results.len(), 1, "memory must survive process restart");
            assert_eq!(results[0].id, id);
            assert_eq!(results[0].content, "persistent memory");
        }
    }

    #[tokio::test]
    async fn test_cortex_search_empty_table_returns_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();
        // All read operations on a fresh store must succeed with empty results.
        assert!(cortex.retrieve(&qvec(), 10).await.unwrap().is_empty());
        assert!(cortex.list().await.unwrap().is_empty());
        assert_eq!(cortex.len().await.unwrap(), 0);
        assert!(cortex.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn test_cortex_store_upserts_on_same_id() {
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();
        let id = MemoryId::new();

        cortex
            .store(
                MemoryEntry::builder(id, "version 1".to_string())
                    .embedding(vec![0.1; 384])
                    .tier(MemoryTier::Cortex)
                    .build(),
            )
            .await
            .unwrap();

        cortex
            .store(
                MemoryEntry::builder(id, "version 2".to_string())
                    .embedding(vec![0.2; 384])
                    .tier(MemoryTier::Cortex)
                    .build(),
            )
            .await
            .unwrap();

        let entries = cortex.list().await.unwrap();
        assert_eq!(entries.len(), 1, "upsert must not duplicate rows");
        assert_eq!(entries[0].content, "version 2");
    }

    #[tokio::test]
    async fn test_cortex_list_and_len_work_without_embedding_query() {
        // list()/len()/is_empty() must never embed or scan via a query vector.
        // These explicit overrides count rows directly instead of going through
        // the trait default that would call retrieve.
        let dir = tempfile::tempdir().unwrap();
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
            .await
            .unwrap();
        // These must succeed without embedding anything.
        let _ = cortex.list().await.expect("list() must not call embed()");
        let _ = cortex.len().await.expect("len() must not call embed()");
        let _ = cortex
            .is_empty()
            .await
            .expect("is_empty() must not call embed()");
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
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
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
        let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
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
            MemoryScope::Plan("plan1".to_string()),
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
