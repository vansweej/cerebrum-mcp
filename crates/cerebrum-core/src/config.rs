//! Hardcoded configuration defaults for the single-user local build.
//!
//! `Config` holds all runtime parameters for the LanceDB storage path.
//! These are compile-time defaults — override any field by constructing the
//! struct directly (e.g. in tests, set `db_path` to a `tempdir()` path).

use std::path::PathBuf;
use std::time::Duration;

/// Total deadline for the Ollama embed request (generous enough to tolerate a
/// cold model load).
pub const DEFAULT_EMBED_TIMEOUT: Duration = Duration::from_secs(60);

/// TCP connect deadline for the Ollama embed request (short because "cannot
/// connect" is unambiguous).
pub const DEFAULT_EMBED_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// Expected dimension of the embedding vectors (768 for nomic-embed-text).
    pub embedding_dim: usize,
    /// Base URL of the local Ollama instance (no trailing slash).
    pub ollama_url: String,
    /// Name of the Ollama embedding model to use.
    pub embed_model: String,
    /// Prefix prepended to queries before embedding (nomic asymmetric search).
    pub query_prefix: String,
    /// Prefix prepended to documents before embedding (nomic asymmetric search).
    pub document_prefix: String,
    /// Total deadline for the Ollama embed request (connect + response).
    pub embed_timeout: Duration,
    /// TCP connect deadline for the Ollama embed request.
    pub embed_connect_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./data/cerebrum"),
            table_name: "memories".to_string(),
            embedding_dim: 768,
            ollama_url: "http://localhost:11434".to_string(),
            embed_model: "nomic-embed-text".to_string(),
            query_prefix: "search_query: ".to_string(),
            document_prefix: "search_document: ".to_string(),
            embed_timeout: DEFAULT_EMBED_TIMEOUT,
            embed_connect_timeout: DEFAULT_EMBED_CONNECT_TIMEOUT,
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
    fn default_embedding_dim_is_768() {
        assert_eq!(Config::default().embedding_dim, 768);
    }

    #[test]
    fn default_ollama_url_is_localhost() {
        assert_eq!(Config::default().ollama_url, "http://localhost:11434");
    }

    #[test]
    fn default_embed_model_is_nomic() {
        assert_eq!(Config::default().embed_model, "nomic-embed-text");
    }
}
