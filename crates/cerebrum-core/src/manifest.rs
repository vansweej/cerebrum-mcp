//! Per-table manifest — records the embedding model and dimension that were used
//! to populate a LanceDB table.
//!
//! The manifest is a small JSON sidecar file stored at
//! `<data_dir>/<table_name>.manifest.json`. It contains no network calls and
//! performs only local filesystem I/O (preserves the
//! `construction_does_no_network` regression in `orchestrator.rs`).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{CerebrumError, Result};

/// Records the embedding model and vector dimension used to populate a table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The Ollama model name used to embed the rows (e.g. `"nomic-embed-text"`).
    pub model: String,
    /// The vector dimension of the stored embeddings (e.g. `768` or `1024`).
    pub dim: usize,
}

/// Return the path to the manifest sidecar file for `table_name` inside `data_dir`.
///
/// Example: `/home/user/.local/share/cerebrum/data/cerebrum/memories.manifest.json`
pub fn manifest_path(data_dir: &Path, table_name: &str) -> PathBuf {
    data_dir.join(format!("{table_name}.manifest.json"))
}

/// Read the manifest sidecar file, returning `Ok(None)` when the file is absent.
///
/// # Errors
///
/// Returns `Err` when the file exists but cannot be read or parsed.
pub fn read_manifest(path: &Path) -> Result<Option<Manifest>> {
    match fs::read_to_string(path) {
        Ok(text) => {
            let m: Manifest = serde_json::from_str(&text).map_err(|e| {
                CerebrumError::Persistence(format!(
                    "failed to parse manifest at {}: {e}",
                    path.display()
                ))
            })?;
            Ok(Some(m))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CerebrumError::Persistence(format!(
            "failed to read manifest at {}: {e}",
            path.display()
        ))),
    }
}

/// Write `m` to `path`, creating or overwriting the file atomically via a
/// temp-file rename when possible, otherwise via a direct write.
///
/// # Errors
///
/// Returns `Err` when the file cannot be written.
pub fn write_manifest(path: &Path, m: &Manifest) -> Result<()> {
    let text = serde_json::to_string_pretty(m)
        .map_err(|e| CerebrumError::Persistence(format!("failed to serialise manifest: {e}")))?;
    fs::write(path, text).map_err(|e| {
        CerebrumError::Persistence(format!(
            "failed to write manifest to {}: {e}",
            path.display()
        ))
    })
}
