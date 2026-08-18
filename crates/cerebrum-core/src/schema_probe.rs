//! Schema probe — read the embedding vector width from a live LanceDB table.
//!
//! This module is pure read-only I/O: it never creates or modifies tables.
//! It is used by `ensure_manifest` to determine the actual stored dimension
//! before trusting any compiled default.

use arrow_schema::DataType;
use lancedb::Connection;

use crate::error::{CerebrumError, Result};

/// Read the `embedding` field's `FixedSizeList` dimension from `table_name`.
///
/// Returns:
/// - `Ok(None)` — the table is absent from the connection.
/// - `Ok(Some(width))` — the table exists (even with zero rows) and its
///   `embedding` field is a `FixedSizeList` of `width` floats.
/// - `Err(CerebrumError::Validation(_))` — the table exists but the
///   `embedding` field is not a `FixedSizeList`.
///
/// # Errors
///
/// Returns `Err` when the schema cannot be fetched or when the `embedding`
/// field exists with an unexpected type.
pub async fn read_embedding_width(conn: &Connection, table_name: &str) -> Result<Option<usize>> {
    let names = conn
        .table_names()
        .execute()
        .await
        .map_err(|e| CerebrumError::Persistence(format!("table_names failed: {e}")))?;

    if !names.contains(&table_name.to_string()) {
        return Ok(None);
    }

    let table = conn
        .open_table(table_name)
        .execute()
        .await
        .map_err(|e| CerebrumError::Persistence(format!("open_table({table_name}) failed: {e}")))?;

    let schema = table
        .schema()
        .await
        .map_err(|e| CerebrumError::Persistence(format!("schema() failed: {e}")))?;

    let field = schema.field_with_name("embedding").map_err(|_| {
        CerebrumError::Validation(format!(
            "table '{table_name}' has no 'embedding' field in its schema"
        ))
    })?;

    match field.data_type() {
        DataType::FixedSizeList(_, size) => Ok(Some(*size as usize)),
        other => Err(CerebrumError::Validation(format!(
            "table '{table_name}': 'embedding' field has unexpected type {other:?}; expected FixedSizeList"
        ))),
    }
}

