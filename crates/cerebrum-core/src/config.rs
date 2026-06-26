//! Hardcoded configuration defaults for the single-user local build.
//!
//! `Config` holds all runtime parameters for the LanceDB storage path.
//! These are compile-time defaults — override any field by constructing the
//! struct directly (e.g. in tests, set `db_path` to a `tempdir()` path).

use std::path::PathBuf;

/// Configuration for `cerebrum-core` storage.
///
/// All fields have hardcoded defaults suitable for a single-user local
/// deployment where the MCP client controls the working directory.
/// Tests override `db_path` via `tempfile::tempdir()` to avoid
/// touching the production store.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the LanceDB database directory (CWD-relative by default).
    ///
    /// Resolved against the process working directory at startup.
    /// Set the MCP server's `cwd` to a durable project folder so
    /// `./data/cerebrum` lands somewhere persistent.
    pub db_path: PathBuf,
    /// Name of the LanceDB table that holds memories.
    pub table_name: String,
    /// Expected dimension of the embedding vectors.
    pub embedding_dim: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./data/cerebrum"),
            table_name: "memories".to_string(),
            embedding_dim: 384,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_db_path_is_relative() {
        let config = Config::default();
        assert!(config.db_path.is_relative());
        assert_eq!(config.db_path, PathBuf::from("./data/cerebrum"));
    }

    #[test]
    fn default_table_name_is_memories() {
        assert_eq!(Config::default().table_name, "memories");
    }

    #[test]
    fn default_embedding_dim_is_384() {
        assert_eq!(Config::default().embedding_dim, 384);
    }
}
