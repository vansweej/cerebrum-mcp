//! Fail-closed guard that ensures the per-table manifest is consistent with
//! the configured embedding dimension before any reads or writes are attempted.
//!
//! Call `ensure_manifest` once after opening a LanceDB connection and before
//! performing any memory operations. If the physical table and the manifest
//! disagree with `config.embedding_dim`, the function returns a
//! `CerebrumError::Validation` error that names the exact remediation steps,
//! preventing silent dimension corruption.

use std::path::Path;

use lancedb::Connection;

use crate::config::Config;
use crate::error::{CerebrumError, Result};
use crate::manifest::{self, Manifest};
use crate::schema_probe;

/// Ensure the per-table manifest exists and is consistent with `config`.
///
/// # Branch: "table absent"
///
/// When the table does not yet exist in `conn`:
/// - If a manifest exists and its `dim` differs from `config.embedding_dim`,
///   return the pinned mismatch error (see below).
/// - Otherwise, write a manifest stamped with `config.embed_model` /
///   `config.embedding_dim` and return `Ok(())`.
///
/// # Branch: "table exists"
///
/// When the table already exists in `conn` (even with zero rows):
/// - If no manifest exists, write one stamped with the **probed** physical
///   width and `config.embed_model` (never trust config over a real table —
///   ADR A5) and return `Ok(())`.
/// - If a manifest exists and the probed width differs from
///   `config.embedding_dim`, return the pinned mismatch error (see below).
/// - Otherwise return `Ok(())`.
///
/// # Pinned mismatch error
///
/// `CerebrumError::Validation` with the exact message:
///
/// ```text
/// Embedding dimension mismatch for table '<table>': stored manifest dim <manifest_dim>
/// does not match configured dim <config_dim>. Run the `cerebrum-reembed` binary to
/// migrate the table, then set CEREBRUM_TABLE_NAME to the new table before starting.
/// ```
///
/// # Errors
///
/// Returns `Err` on dimension mismatch or on I/O failure in the schema probe
/// or manifest read/write.
pub async fn ensure_manifest(conn: &Connection, data_dir: &Path, config: &Config) -> Result<()> {
    let mpath = manifest::manifest_path(data_dir, &config.table_name);
    let existing_manifest = manifest::read_manifest(&mpath)?;

    let probed_width = schema_probe::read_embedding_width(conn, &config.table_name).await?;

    match probed_width {
        // table absent
        None => {
            if let Some(ref m) = existing_manifest {
                if m.dim != config.embedding_dim {
                    return Err(mismatch_err(
                        &config.table_name,
                        m.dim,
                        config.embedding_dim,
                    ));
                }
            }
            manifest::write_manifest(
                &mpath,
                &Manifest {
                    model: config.embed_model.clone(),
                    dim: config.embedding_dim,
                },
            )?;
            Ok(())
        }
        // table exists
        Some(width) => {
            if existing_manifest.is_none() {
                // Stamp manifest with the PROBED width (A5 — never trust config).
                manifest::write_manifest(
                    &mpath,
                    &Manifest {
                        model: config.embed_model.clone(),
                        dim: width,
                    },
                )?;
                return Ok(());
            }
            if width != config.embedding_dim {
                return Err(mismatch_err(
                    &config.table_name,
                    width,
                    config.embedding_dim,
                ));
            }
            Ok(())
        }
    }
}

fn mismatch_err(table: &str, manifest_dim: usize, config_dim: usize) -> CerebrumError {
    CerebrumError::Validation(format!(
        "Embedding dimension mismatch for table '{table}': stored manifest dim {manifest_dim} \
does not match configured dim {config_dim}. Run the `cerebrum-reembed` binary to migrate the \
table, then set CEREBRUM_TABLE_NAME to the new table before starting."
    ))
}
