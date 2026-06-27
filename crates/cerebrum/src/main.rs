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
    let config = cerebrum_core::Config::default();
    let orchestrator = Arc::new(MemoryOrchestrator::from_config(&config).await?);
    let handler = CerebrumHandler::new(orchestrator);

    tracing::info!("Cerebrum MCP server initialized");
    tracing::info!("Available tools: remember, recall, memorize, forget, end_session");

    tracing::info!("Starting MCP server with stdio transport");
    let service = handler.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;

    tracing::info!("Cerebrum MCP server stopped");
    Ok(())
}
