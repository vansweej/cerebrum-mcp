//! Tests for the env-overlay module.

use std::collections::HashMap;

use cerebrum_core::config::Config;
use cerebrum_core::env_overlay::apply_env_overlay_with;

/// Overlay with EMBED_MODEL, EMBEDDING_DIM, and TABLE_NAME present.
#[test]
fn overlay_applies_known_vars() {
    let mut map = HashMap::<String, String>::new();
    map.insert(
        "CEREBRUM_EMBED_MODEL".to_string(),
        "qwen3-embedding:0.6b".to_string(),
    );
    map.insert("CEREBRUM_EMBEDDING_DIM".to_string(), "1024".to_string());
    map.insert(
        "CEREBRUM_TABLE_NAME".to_string(),
        "memories_qwen3".to_string(),
    );

    let result = apply_env_overlay_with(Config::default(), |k| map.get(k).cloned());

    assert_eq!(result.embed_model, "qwen3-embedding:0.6b");
    assert_eq!(result.embedding_dim, 1024);
    assert_eq!(result.table_name, "memories_qwen3");
}

/// Empty map — result must equal Config::default().
#[test]
fn empty_map_yields_default() {
    let map = HashMap::<String, String>::new();
    let result = apply_env_overlay_with(Config::default(), |k| map.get(k).cloned());
    let default = Config::default();

    assert_eq!(result.embed_model, default.embed_model);
    assert_eq!(result.embedding_dim, default.embedding_dim);
    assert_eq!(result.table_name, default.table_name);
    assert_eq!(result.max_input_chars, default.max_input_chars);
    assert_eq!(result.query_prefix, default.query_prefix);
    assert_eq!(result.document_prefix, default.document_prefix);
}

/// Non-numeric CEREBRUM_EMBEDDING_DIM is silently ignored — default is retained.
#[test]
fn invalid_dim_is_silently_ignored() {
    let mut map = HashMap::<String, String>::new();
    map.insert(
        "CEREBRUM_EMBEDDING_DIM".to_string(),
        "notanumber".to_string(),
    );

    let result = apply_env_overlay_with(Config::default(), |k| map.get(k).cloned());

    assert_eq!(result.embedding_dim, Config::default().embedding_dim);
}
