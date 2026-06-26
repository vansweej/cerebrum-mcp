mod mcp_server;

use cerebrum_core::embedder::MockEmbedder;
use cerebrum_core::orchestrator::MemoryOrchestrator;
use mcp_server::CerebrumHandler;
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::RoleServer;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Cerebrum MCP server...");

    // Initialize embedder
    let embedder: Arc<dyn cerebrum_core::Embedder> = Arc::new(MockEmbedder::new());

    // Initialize memory orchestrator with Config
    let config = cerebrum_core::Config::default();
    let orchestrator = Arc::new(
        MemoryOrchestrator::new(
            &config.db_path,
            &config.table_name,
            config.embedding_dim,
            embedder,
        )
        .await?,
    );
    let handler = CerebrumHandler::new(orchestrator);

    tracing::info!("Cerebrum MCP server initialized");
    tracing::info!("Available tools: remember, recall, memorize, forget, end_session");

    // Start MCP server with stdio transport
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let transport = AsyncRwTransport::<RoleServer, _, _>::new(stdin, stdout);

    tracing::info!("Starting MCP server with stdio transport");
    rmcp::serve_server(handler, transport).await?;

    tracing::info!("Cerebrum MCP server stopped");
    Ok(())
}
