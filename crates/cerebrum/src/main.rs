mod mcp_server;

use cerebrum_core::orchestrator::MemoryOrchestrator;
use mcp_server::McpServer;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Cerebrum MCP server...");

    // Initialize memory orchestrator
    let orchestrator = Arc::new(MemoryOrchestrator::new());
    let _server = McpServer::new(orchestrator);

    tracing::info!("Cerebrum MCP server initialized");
    tracing::info!("Available tools: remember, recall, memorize, forget, end_session");

    // TODO: Implement MCP server event loop and stdio communication

    Ok(())
}
