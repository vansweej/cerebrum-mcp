//! `cerebrum-reembed` — migrate a LanceDB table to a new embedding model.
//!
//! Reads configuration from environment variables via `apply_env_overlay`,
//! then re-embeds every row of the source table into a destination table.
//!
//! # Usage
//!
//! ```sh
//! CEREBRUM_EMBED_MODEL=qwen3-embedding:0.6b \
//! CEREBRUM_EMBEDDING_DIM=1024 \
//! CEREBRUM_SOURCE_TABLE=memories \
//! CEREBRUM_TABLE_NAME=memories_qwen3 \
//! cerebrum-reembed
//! ```
//!
//! See `docs/runbooks/reembed-migration.md` for the full migration guide.

use cerebrum_core::config::Config;
use cerebrum_core::env_overlay::apply_env_overlay;
use cerebrum_core::fastembed_embedder::FastEmbedEmbedder;
use cerebrum_core::manifest::{self, read_manifest};
use cerebrum_core::reembed::reembed;
use cerebrum_core::schema_probe::read_embedding_width;

#[cfg(not(tarpaulin_include))]
#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("cerebrum-reembed: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(tarpaulin_include))]
async fn run() -> anyhow::Result<()> {
    let config = apply_env_overlay(Config::default());

    let src_table =
        std::env::var("CEREBRUM_SOURCE_TABLE").unwrap_or_else(|_| "memories".to_string());
    let dest_table = config.table_name.clone();

    // Resolve source dimension from schema probe, falling back to source manifest.
    let db_path = &config.db_path;
    std::fs::create_dir_all(db_path)?;
    let db_path_str = db_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 db_path"))?;
    let conn = lancedb::connect(db_path_str)
        .execute()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect to LanceDB: {e}"))?;

    let probed_dim = read_embedding_width(&conn, &src_table)
        .await
        .map_err(|e| anyhow::anyhow!("Schema probe failed: {e}"))?;

    let src_manifest_path = manifest::manifest_path(db_path, &src_table);
    let src_manifest = read_manifest(&src_manifest_path)
        .map_err(|e| anyhow::anyhow!("Failed to read source manifest: {e}"))?;

    let src_dim = match probed_dim.or_else(|| src_manifest.as_ref().map(|m| m.dim)) {
        Some(d) => d,
        None => {
            anyhow::bail!(
                "Cannot determine source table '{src_table}' dimension: \
                 table is absent and no manifest found. \
                 Ensure the source table exists before running cerebrum-reembed."
            );
        }
    };

    let src_model = src_manifest
        .as_ref()
        .map(|m| m.model.as_str())
        .unwrap_or("unknown");

    // No-op refusal (ADR A8): refuse a same-model, same-dim migration unless forced.
    if src_dim == config.embedding_dim
        && config.embed_model == src_model
        && std::env::var("CEREBRUM_ALLOW_SAME_DIM").as_deref() != Ok("1")
    {
        anyhow::bail!(
            "No-op migration detected: source dim={src_dim} model={src_model} matches \
             configured dim={} model={}. \
             Set CEREBRUM_EMBED_MODEL and CEREBRUM_EMBEDDING_DIM to target values, \
             or set CEREBRUM_ALLOW_SAME_DIM=1 to force.",
            config.embedding_dim,
            config.embed_model,
        );
    }

    let embedder = FastEmbedEmbedder::with_config(
        config.ollama_url.clone(),
        config.embed_model.clone(),
        config.embedding_dim,
    );

    println!(
        "Re-embedding: {src_table} (dim={src_dim}) → {dest_table} (dim={})",
        config.embedding_dim
    );

    let report = reembed(&src_table, src_dim, &dest_table, &config, &embedder)
        .await
        .map_err(|e| anyhow::anyhow!("reembed failed: {e}"))?;

    println!(
        "Done: total={} written={} truncated={} verified={}",
        report.total, report.written, report.truncated, report.verified
    );

    if !report.verified {
        anyhow::bail!("Verification failed — destination table may be incomplete.");
    }

    Ok(())
}
