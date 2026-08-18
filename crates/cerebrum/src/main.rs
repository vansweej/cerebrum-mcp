mod mcp_server;

use cerebrum_core::orchestrator::MemoryOrchestrator;
use mcp_server::CerebrumHandler;
use rmcp::ServiceExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Cerebrum MCP server...");

    // Initialize memory orchestrator with real Ollama embeddings from Config.
    let config = cerebrum_core::env_overlay::apply_env_overlay(cerebrum_core::Config::default());
    let orchestrator = Arc::new(MemoryOrchestrator::from_config(&config).await?);

    // Read optional project preference from the environment.
    // CEREBRUM_PROJECT may be a comma-separated list of project names.
    let projects: Vec<String> = std::env::var("CEREBRUM_PROJECT")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let handler = CerebrumHandler::with_default_project(orchestrator, projects);

    tracing::info!("Cerebrum MCP server initialized");
    tracing::info!("Available tools: remember, recall, memorize, forget, end_session");

    tracing::info!("Starting MCP server with stdio transport");
    let service = handler.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;

    tracing::info!("Cerebrum MCP server stopped");
    Ok(())
}
