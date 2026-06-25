#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Cerebrum MCP server...");

    // TODO: Initialize memory tiers and orchestrator

    Ok(())
}
