//! Closure-injected environment overlay for `Config`.
//!
//! Provides two functions for applying environment variable overrides to a
//! `Config` value:
//!
//! - [`apply_env_overlay_with`] — pure, closure-injected; suitable for tests.
//! - [`apply_env_overlay`] — thin wrapper that reads from the process environment.
//!
//! Neither function performs network calls or modifies global state.

use crate::config::Config;

/// Apply environment variable overrides to `config` using the `get` closure.
///
/// Each variable is applied only when the closure returns `Some(_)` and the
/// value parses to the target type (numeric variables are silently ignored on
/// parse failure, preserving the existing default).
///
/// Supported variables:
/// - `CEREBRUM_EMBED_MODEL`      → `embed_model`
/// - `CEREBRUM_EMBEDDING_DIM`    → `embedding_dim` (`usize`; ignored on parse error)
/// - `CEREBRUM_TABLE_NAME`       → `table_name`
/// - `CEREBRUM_MAX_INPUT_CHARS`  → `max_input_chars` (`usize`; ignored on parse error)
/// - `CEREBRUM_QUERY_PREFIX`     → `query_prefix`
/// - `CEREBRUM_DOCUMENT_PREFIX`  → `document_prefix`
pub fn apply_env_overlay_with<F: Fn(&str) -> Option<String>>(config: Config, get: F) -> Config {
    let mut c = config;

    if let Some(v) = get("CEREBRUM_EMBED_MODEL") {
        c.embed_model = v;
    }
    if let Some(v) = get("CEREBRUM_EMBEDDING_DIM") {
        if let Ok(n) = v.parse::<usize>() {
            c.embedding_dim = n;
        }
    }
    if let Some(v) = get("CEREBRUM_TABLE_NAME") {
        c.table_name = v;
    }
    if let Some(v) = get("CEREBRUM_MAX_INPUT_CHARS") {
        if let Ok(n) = v.parse::<usize>() {
            c.max_input_chars = n;
        }
    }
    if let Some(v) = get("CEREBRUM_QUERY_PREFIX") {
        c.query_prefix = v;
    }
    if let Some(v) = get("CEREBRUM_DOCUMENT_PREFIX") {
        c.document_prefix = v;
    }

    c
}

/// Apply environment variable overrides to `config` by reading from the process
/// environment.
///
/// This is a thin wrapper around [`apply_env_overlay_with`] that uses
/// `std::env::var` as the lookup closure. It never touches global state beyond
/// reading environment variables.
pub fn apply_env_overlay(config: Config) -> Config {
    apply_env_overlay_with(config, |k| std::env::var(k).ok())
}
