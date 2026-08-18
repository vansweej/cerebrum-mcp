//! Hardcoded configuration defaults for the single-user local build.
//!
//! `Config` holds all runtime parameters for the LanceDB storage path.
//! These are compile-time defaults — override any field by constructing the
//! struct directly (e.g. in tests, set `db_path` to a `tempdir()` path).

use std::path::PathBuf;
use std::time::Duration;

/// Resolve the data directory for `app` using XDG / platform conventions.
///
/// Precedence (each candidate is joined as `<base>/<app>/data/<app>`):
/// 1. `XDG_DATA_HOME` — when set and non-empty, used as the base.
/// 2. `LOCALAPPDATA` (Windows only) — when set and non-empty, used as the base.
/// 3. `HOME` — when set and non-empty, base is `<HOME>/.local/share`.
/// 4. Relative fallback `./data/<app>` (no path joining, returned as-is).
///
/// This is a pure function: all environment access is performed through the
/// `get_env` closure, making it fully testable without touching the process
/// environment.
fn resolve_data_dir(app: &str, get_env: impl Fn(&str) -> Option<String>) -> PathBuf {
    // 1. XDG_DATA_HOME
    if let Some(xdg) = get_env("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join(app).join("data").join(app);
    }

    // 2. LOCALAPPDATA (Windows only)
    #[cfg(windows)]
    if let Some(local) = get_env("LOCALAPPDATA").filter(|v| !v.is_empty()) {
        return PathBuf::from(local).join(app).join("data").join(app);
    }

    // 3. HOME → ~/.local/share
    if let Some(home) = get_env("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join(app)
            .join("data")
            .join(app);
    }

    // 4. Relative fallback
    PathBuf::from(format!("./data/{}", app))
}

/// Return the default LanceDB data directory for the `cerebrum` application.
///
/// The binary now self-locates its data directory without depending on the
/// working directory. On a typical Linux/macOS system this resolves to
/// `~/.local/share/cerebrum/data/cerebrum`, which is identical to the path
/// produced by the previous flake wrapper (`cd ~/.local/share/cerebrum &&
/// exec cerebrum`) combined with the old relative default `./data/cerebrum`.
/// Existing on-disk stores therefore require **zero migration**.
pub fn default_data_dir() -> PathBuf {
    resolve_data_dir("cerebrum", |k| std::env::var(k).ok())
}

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
    /// Maximum number of characters to send to the embedder per call.
    ///
    /// Inputs longer than this limit are head-truncated before embedding.
    /// The raw `content` is always stored verbatim; only the index vector
    /// is affected. Default: 96,000 characters (~24k tokens).
    pub max_input_chars: usize,
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
            db_path: default_data_dir(),
            table_name: "memories".to_string(),
            embedding_dim: 768,
            max_input_chars: 96_000,
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

    // ── resolve_data_dir unit tests ──────────────────────────────────────────

    /// XDG_DATA_HOME set and non-empty: must be used as the base directory.
    #[test]
    fn resolve_xdg_data_home_set_and_nonempty() {
        let path = resolve_data_dir("cerebrum", |k| match k {
            "XDG_DATA_HOME" => Some("/xdg".to_string()),
            "HOME" => Some("/home/user".to_string()),
            _ => None,
        });
        assert_eq!(path, PathBuf::from("/xdg/cerebrum/data/cerebrum"));
    }

    /// XDG_DATA_HOME present but empty: must fall through to HOME.
    #[test]
    fn resolve_empty_xdg_falls_through_to_home() {
        let path = resolve_data_dir("cerebrum", |k| match k {
            "XDG_DATA_HOME" => Some("".to_string()), // empty → skip
            "HOME" => Some("/home/user".to_string()),
            _ => None,
        });
        assert_eq!(
            path,
            PathBuf::from("/home/user/.local/share/cerebrum/data/cerebrum")
        );
    }

    /// HOME set with XDG absent: result must end with
    /// `<HOME>/.local/share/cerebrum/data/cerebrum`.
    #[test]
    fn resolve_home_used_when_xdg_absent() {
        let home = "/home/alice";
        let path = resolve_data_dir("cerebrum", |k| match k {
            "HOME" => Some(home.to_string()),
            _ => None,
        });
        assert_eq!(
            path,
            PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("cerebrum")
                .join("data")
                .join("cerebrum"),
        );
        // Also assert the suffix explicitly.
        let s = path.to_string_lossy();
        assert!(
            s.ends_with("cerebrum/data/cerebrum"),
            "path must end with cerebrum/data/cerebrum, got: {s}"
        );
    }

    /// Neither XDG nor HOME set: must return the relative fallback.
    #[test]
    fn resolve_relative_fallback_when_no_env() {
        let path = resolve_data_dir("cerebrum", |_| None);
        assert_eq!(path, PathBuf::from("./data/cerebrum"));
    }

    /// `default_data_dir()` must return a non-empty path whose final
    /// component is "cerebrum".
    #[test]
    fn default_data_dir_final_component_is_cerebrum() {
        let p = default_data_dir();
        assert!(!p.as_os_str().is_empty(), "path must not be empty");
        let last = p
            .file_name()
            .expect("path must have a file_name component")
            .to_string_lossy();
        assert_eq!(last, "cerebrum", "final path component must be 'cerebrum'");
    }

    // ── existing Config field tests ──────────────────────────────────────────

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
