//! Table-to-table re-embedding converter.
//!
//! Reads every row from an existing LanceDB source table, re-embeds its
//! content with a new embedder/model, and atomically upserts the result into
//! a destination table. Never creates the source table (it must already
//! exist); refuses to claim success on an empty destination while the source
//! is non-empty.

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{CerebrumError, Result};
use crate::input_bound::bound_input;
use crate::lancedb_cortex::LanceDBCortex;
use crate::traits::{Embedder, MemoryStore};

/// Summary report produced by [`reembed`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReembedReport {
    /// Total number of rows read from the source table.
    pub total: usize,
    /// Number of rows successfully re-embedded and written to the destination table.
    pub written: usize,
    /// Count of entries whose embed input was head-bounded (see `input_bound::bound_input`).
    pub truncated: usize,
    /// Whether the destination table was verified to contain valid, correctly-dimensioned vectors.
    pub verified: bool,
}

/// Re-embed every row of `src_table` into `dest_table` using `embedder`.
///
/// # Errors
///
/// Returns `Err(CerebrumError::Validation(_))` if `src_table` does not exist in
/// the LanceDB connection at `config`'s data directory. Never creates the
/// source table. Also returns `Err` on any underlying LanceDB or embedding
/// failure.
pub async fn reembed(
    src_table: &str,
    src_dim: usize,
    dest_table: &str,
    config: &Config,
    embedder: &dyn Embedder,
) -> Result<ReembedReport> {
    let db_path = &config.db_path;
    std::fs::create_dir_all(db_path)
        .map_err(|e| CerebrumError::Database(format!("Failed to create db_path: {e}")))?;
    let path = db_path
        .to_str()
        .ok_or_else(|| CerebrumError::Database("non-UTF-8 db_path".to_string()))?;
    let conn = lancedb::connect(path)
        .execute()
        .await
        .map_err(|e| CerebrumError::Database(format!("Failed to connect to LanceDB: {e}")))?;

    let existing = conn
        .table_names()
        .execute()
        .await
        .map_err(|e| CerebrumError::Database(format!("Failed to list tables: {e}")))?;

    if !existing.contains(&src_table.to_string()) {
        return Err(CerebrumError::Validation(format!(
            "source table '{src_table}' does not exist; refusing to create it"
        )));
    }

    let src = LanceDBCortex::new(db_path, src_table, src_dim).await?;
    let dest = LanceDBCortex::new(db_path, dest_table, config.embedding_dim).await?;

    let rows = src.list().await?;
    let total = rows.len();
    let mut written = 0usize;
    let mut truncated = 0usize;

    for entry in rows {
        let input = format!("{}{}", config.document_prefix, entry.content);

        let (_head, was_truncated) = bound_input(&input, config.max_input_chars);
        if was_truncated {
            truncated += 1;
        }

        let embedding = embedder.embed(&input).await?;

        let mut updated = entry.clone();
        updated.embedding = Some(embedding);

        dest.store(updated).await?;
        written += 1;
    }

    let verified =
        verify_dest(&dest, total, written, config.embedding_dim).await? && written == total;

    Ok(ReembedReport {
        total,
        written,
        truncated,
        verified,
    })
}

/// Verify that the destination table contains valid, correctly-dimensioned
/// vectors. Returns `false` when `total > 0` (source non-empty) but the
/// destination is empty, refusing to silently claim success.
async fn verify_dest(
    dest: &LanceDBCortex,
    total: usize,
    _written: usize,
    expected_dim: usize,
) -> Result<bool> {
    let dest_rows = dest.list().await?;

    if total > 0 && dest_rows.is_empty() {
        return Ok(false);
    }

    for entry in &dest_rows {
        match &entry.embedding {
            Some(v) => {
                if v.len() != expected_dim {
                    return Ok(false);
                }
                if v.iter().any(|x| !x.is_finite()) {
                    return Ok(false);
                }
                if v.iter().all(|x| *x == 0.0) {
                    return Ok(false);
                }
            }
            None => return Ok(false),
        }
    }

    Ok(true)
}
