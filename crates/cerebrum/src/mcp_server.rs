use cerebrum_core::models::MemoryId;
use cerebrum_core::orchestrator::MemoryOrchestrator;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    Annotated, CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams,
    RawContent, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info};

/// Cerebrum MCP Server Handler
/// Implements the ServerHandler trait to expose memory tools via MCP protocol
pub struct CerebrumHandler {
    orchestrator: Arc<MemoryOrchestrator>,
}

impl CerebrumHandler {
    /// Create a new Cerebrum handler with the given orchestrator
    pub fn new(orchestrator: Arc<MemoryOrchestrator>) -> Self {
        Self { orchestrator }
    }

    /// Get the remember tool definition
    fn remember_tool() -> Tool {
        let schema = json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store"
                },
                "salience": {
                    "type": "number",
                    "description": "Importance score (0.0-1.0), defaults to 0.5",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["content"]
        });
        let map = schema.as_object().cloned().unwrap_or_default();
        Tool::new(
            "remember",
            "Store a memory in the Synapse (short-term) tier with automatic embedding generation",
            map,
        )
        .with_title("Remember")
    }

    /// Get the recall tool definition
    fn recall_tool() -> Tool {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["query"]
        });
        let map = schema.as_object().cloned().unwrap_or_default();
        Tool::new(
            "recall",
            "Search memories across both Synapse and Cortex tiers using semantic similarity",
            map,
        )
        .with_title("Recall")
    }

    /// Get the memorize tool definition
    fn memorize_tool() -> Tool {
        let schema = json!({
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The ID of the memory to promote"
                }
            },
            "required": ["memory_id"]
        });
        let map = schema.as_object().cloned().unwrap_or_default();
        Tool::new(
            "memorize",
            "Promote a memory from Synapse (short-term) to Cortex (long-term) storage",
            map,
        )
        .with_title("Memorize")
    }

    /// Get the forget tool definition
    fn forget_tool() -> Tool {
        let schema = json!({
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The ID of the memory to delete"
                }
            },
            "required": ["memory_id"]
        });
        let map = schema.as_object().cloned().unwrap_or_default();
        Tool::new(
            "forget",
            "Delete a memory from both Synapse and Cortex tiers",
            map,
        )
        .with_title("Forget")
    }

    /// Get the end_session tool definition
    fn end_session_tool() -> Tool {
        let schema = json!({
            "type": "object",
            "properties": {
                "promotion_threshold": {
                    "type": "number",
                    "description": "Salience threshold for auto-promotion (0.0-1.0, default: 0.7)",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": []
        });
        let map = schema.as_object().cloned().unwrap_or_default();
        Tool::new(
            "end_session",
            "End a session: clear Synapse and auto-promote memories above salience threshold to Cortex",
            map,
        )
        .with_title("End Session")
    }

    /// Get the recall_by_scope tool definition (Phase 5)
    fn recall_by_scope_tool() -> Tool {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "scope": {
                    "type": "string",
                    "description": "Memory scope filter: 'global', 'user:<id>', 'agent:<id>', or 'session:<id>'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["query", "scope"]
        });
        let map = schema.as_object().cloned().unwrap_or_default();
        Tool::new(
            "recall_by_scope",
            "Search memories filtered by scope (Phase 5 feature)",
            map,
        )
        .with_title("Recall by Scope")
    }

    /// Handle remember tool call
    async fn handle_remember(&self, arguments: Option<Value>) -> Result<CallToolResult, String> {
        let args = arguments.ok_or("Missing arguments for remember tool")?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: content")?
            .to_string();

        let _salience = args.get("salience").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;

        // Build metadata HashMap (empty for now, can be extended)
        let metadata = HashMap::new();

        match self.orchestrator.remember(content.clone(), metadata).await {
            Ok(memory_id) => {
                info!("Memory stored: {}", memory_id);
                let response = json!({
                    "success": true,
                    "memory_id": memory_id.to_string(),
                    "message": format!("Memory stored with ID: {}", memory_id)
                });
                Ok(CallToolResult::success(vec![Annotated::new(
                    RawContent::text(response.to_string()),
                    None,
                )]))
            }
            Err(e) => {
                error!("Failed to store memory: {:?}", e);
                Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(format!("Failed to store memory: {}", e)),
                    None,
                )]))
            }
        }
    }

    /// Handle recall tool call
    async fn handle_recall(&self, arguments: Option<Value>) -> Result<CallToolResult, String> {
        let args = arguments.ok_or("Missing arguments for recall tool")?;

        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: query")?
            .to_string();

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        match self.orchestrator.recall(query, limit).await {
            Ok(results) => {
                info!("Recall found {} results", results.len());
                let result_json: Vec<Value> = results
                    .iter()
                    .map(|entry| {
                        json!({
                            "id": entry.id.to_string(),
                            "content": entry.content,
                            "salience": entry.salience,
                            "tier": format!("{:?}", entry.tier),
                            "timestamp": entry.timestamp
                        })
                    })
                    .collect();

                let response = json!({
                    "success": true,
                    "count": results.len(),
                    "results": result_json
                });
                Ok(CallToolResult::success(vec![Annotated::new(
                    RawContent::text(response.to_string()),
                    None,
                )]))
            }
            Err(e) => {
                error!("Failed to recall memories: {:?}", e);
                Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(format!("Failed to recall memories: {}", e)),
                    None,
                )]))
            }
        }
    }

    /// Handle memorize tool call
    async fn handle_memorize(&self, arguments: Option<Value>) -> Result<CallToolResult, String> {
        let args = arguments.ok_or("Missing arguments for memorize tool")?;

        let memory_id_str = args
            .get("memory_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: memory_id")?;

        let memory_id = MemoryId::from_string(memory_id_str)
            .map_err(|e| format!("Invalid memory ID: {}", e))?;

        match self.orchestrator.memorize(memory_id).await {
            Ok(_) => {
                info!("Memory promoted to Cortex: {}", memory_id);
                let response = json!({
                    "success": true,
                    "memory_id": memory_id.to_string(),
                    "message": format!("Memory {} promoted to Cortex", memory_id)
                });
                Ok(CallToolResult::success(vec![Annotated::new(
                    RawContent::text(response.to_string()),
                    None,
                )]))
            }
            Err(e) => {
                error!("Failed to promote memory: {:?}", e);
                Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(format!("Failed to promote memory: {}", e)),
                    None,
                )]))
            }
        }
    }

    /// Handle forget tool call
    async fn handle_forget(&self, arguments: Option<Value>) -> Result<CallToolResult, String> {
        let args = arguments.ok_or("Missing arguments for forget tool")?;

        let memory_id_str = args
            .get("memory_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: memory_id")?;

        let memory_id = MemoryId::from_string(memory_id_str)
            .map_err(|e| format!("Invalid memory ID: {}", e))?;

        match self.orchestrator.forget(memory_id).await {
            Ok(_) => {
                info!("Memory forgotten: {}", memory_id);
                let response = json!({
                    "success": true,
                    "memory_id": memory_id.to_string(),
                    "message": format!("Memory {} deleted from all tiers", memory_id)
                });
                Ok(CallToolResult::success(vec![Annotated::new(
                    RawContent::text(response.to_string()),
                    None,
                )]))
            }
            Err(e) => {
                error!("Failed to forget memory: {:?}", e);
                Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(format!("Failed to forget memory: {}", e)),
                    None,
                )]))
            }
        }
    }

    /// Handle end_session tool call
    async fn handle_end_session(&self, arguments: Option<Value>) -> Result<CallToolResult, String> {
        let threshold = arguments
            .as_ref()
            .and_then(|args| args.get("promotion_threshold"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        match self.orchestrator.end_session(threshold).await {
            Ok(_) => {
                info!("Session ended with promotion threshold: {}", threshold);
                let response = json!({
                    "success": true,
                    "threshold": threshold,
                    "message": format!("Session ended, memories with salience >= {} promoted to Cortex", threshold)
                });
                Ok(CallToolResult::success(vec![Annotated::new(
                    RawContent::text(response.to_string()),
                    None,
                )]))
            }
            Err(e) => {
                error!("Failed to end session: {:?}", e);
                Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(format!("Failed to end session: {}", e)),
                    None,
                )]))
            }
        }
    }

    /// Handle recall_by_scope tool call (Phase 5)
    async fn handle_recall_by_scope(
        &self,
        arguments: Option<Value>,
    ) -> Result<CallToolResult, String> {
        let args = arguments.ok_or("Missing arguments for recall_by_scope tool")?;

        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: query")?
            .to_string();

        let scope_str = args
            .get("scope")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: scope")?;

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        // Parse scope string
        let scope = if scope_str == "global" {
            cerebrum_core::MemoryScope::Global
        } else if let Some(user_id) = scope_str.strip_prefix("user:") {
            cerebrum_core::MemoryScope::User(user_id.to_string())
        } else if let Some(agent_id) = scope_str.strip_prefix("agent:") {
            cerebrum_core::MemoryScope::Agent(agent_id.to_string())
        } else if let Some(session_id) = scope_str.strip_prefix("session:") {
            cerebrum_core::MemoryScope::Session(session_id.to_string())
        } else {
            return Ok(CallToolResult::error(vec![Annotated::new(
                RawContent::text("Invalid scope format. Use 'global', 'user:<id>', 'agent:<id>', or 'session:<id>'".to_string()),
                None,
            )]));
        };

        match self.orchestrator.recall_by_scope(query, scope, limit).await {
            Ok(results) => {
                info!("Recall by scope found {} results", results.len());
                let result_json: Vec<Value> = results
                    .iter()
                    .map(|entry| {
                        json!({
                            "id": entry.id.to_string(),
                            "content": entry.content,
                            "salience": entry.salience,
                            "scope": entry.scope.as_str(),
                            "tier": format!("{:?}", entry.tier),
                            "timestamp": entry.timestamp
                        })
                    })
                    .collect();

                let response = json!({
                    "success": true,
                    "count": results.len(),
                    "results": result_json
                });
                Ok(CallToolResult::success(vec![Annotated::new(
                    RawContent::text(response.to_string()),
                    None,
                )]))
            }
            Err(e) => {
                error!("Failed to recall memories by scope: {:?}", e);
                Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(format!("Failed to recall memories by scope: {}", e)),
                    None,
                )]))
            }
        }
    }
}

impl ServerHandler for CerebrumHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }

    #[allow(clippy::manual_async_fn)]
    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + '_ {
        async move {
            debug!("Listing available tools");
            Ok(ListToolsResult {
                tools: vec![
                    Self::remember_tool(),
                    Self::recall_tool(),
                    Self::memorize_tool(),
                    Self::forget_tool(),
                    Self::end_session_tool(),
                    Self::recall_by_scope_tool(),
                ],
                ..Default::default()
            })
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + '_ {
        async move {
            debug!("Calling tool: {}", request.name);
            let result = match request.name.as_ref() {
                "remember" => {
                    self.handle_remember(request.arguments.map(Value::Object))
                        .await
                }
                "recall" => {
                    self.handle_recall(request.arguments.map(Value::Object))
                        .await
                }
                "memorize" => {
                    self.handle_memorize(request.arguments.map(Value::Object))
                        .await
                }
                "forget" => {
                    self.handle_forget(request.arguments.map(Value::Object))
                        .await
                }
                "end_session" => {
                    self.handle_end_session(request.arguments.map(Value::Object))
                        .await
                }
                "recall_by_scope" => {
                    self.handle_recall_by_scope(request.arguments.map(Value::Object))
                        .await
                }
                _ => Err("Unknown tool".to_string()),
            };

            result.map_err(|e| ErrorData::internal_error(e, None))
        }
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        match name {
            "remember" => Some(Self::remember_tool()),
            "recall" => Some(Self::recall_tool()),
            "memorize" => Some(Self::memorize_tool()),
            "forget" => Some(Self::forget_tool()),
            "end_session" => Some(Self::end_session_tool()),
            "recall_by_scope" => Some(Self::recall_by_scope_tool()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile;

    #[tokio::test]
    async fn test_handler_creation() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let _handler = CerebrumHandler::new(orchestrator);
    }

    #[test]
    fn test_remember_tool_definition() {
        let tool = CerebrumHandler::remember_tool();
        assert_eq!(tool.name, "remember");
    }

    #[test]
    fn test_recall_tool_definition() {
        let tool = CerebrumHandler::recall_tool();
        assert_eq!(tool.name, "recall");
    }

    #[test]
    fn test_memorize_tool_definition() {
        let tool = CerebrumHandler::memorize_tool();
        assert_eq!(tool.name, "memorize");
    }

    #[test]
    fn test_forget_tool_definition() {
        let tool = CerebrumHandler::forget_tool();
        assert_eq!(tool.name, "forget");
    }

    #[test]
    fn test_end_session_tool_definition() {
        let tool = CerebrumHandler::end_session_tool();
        assert_eq!(tool.name, "end_session");
    }

    #[tokio::test]
    async fn test_get_tool() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        assert!(handler.get_tool("remember").is_some());
        assert!(handler.get_tool("recall").is_some());
        assert!(handler.get_tool("memorize").is_some());
        assert!(handler.get_tool("forget").is_some());
        assert!(handler.get_tool("end_session").is_some());
        assert!(handler.get_tool("unknown").is_none());
    }

    #[tokio::test]
    async fn test_handle_remember_missing_content() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler.handle_remember(Some(json!({}))).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_remember_success() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler
            .handle_remember(Some(json!({
                "content": "Test memory",
                "salience": 0.8
            })))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_recall_missing_query() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler.handle_recall(Some(json!({}))).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_recall_success() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );

        orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let handler = CerebrumHandler::new(orchestrator);
        let result = handler
            .handle_recall(Some(json!({
                "query": "Test",
                "limit": 10
            })))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_memorize_success() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );

        let memory_id = orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let handler = CerebrumHandler::new(orchestrator);
        let result = handler
            .handle_memorize(Some(json!({
                "memory_id": memory_id.to_string()
            })))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_memorize_invalid_id() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );

        let handler = CerebrumHandler::new(orchestrator);
        let result = handler
            .handle_memorize(Some(json!({
                "memory_id": "invalid-id"
            })))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_forget_success() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );

        let memory_id = orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let handler = CerebrumHandler::new(orchestrator);
        let result = handler
            .handle_forget(Some(json!({
                "memory_id": memory_id.to_string()
            })))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_end_session_success() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );

        orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let handler = CerebrumHandler::new(orchestrator);
        let result = handler
            .handle_end_session(Some(json!({
                "promotion_threshold": 0.7
            })))
            .await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_info() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(MemoryOrchestrator::new(
                    embedder,
                    dir.path(),
                    "memories",
                    384,
                ))
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        let info = handler.get_info();
        // Verify get_info returns a valid ServerInfo
        assert!(!info.server_info.name.is_empty());
    }

    #[test]
    fn test_recall_by_scope_tool_definition() {
        let tool = CerebrumHandler::recall_by_scope_tool();
        assert_eq!(tool.name, "recall_by_scope");
    }

    #[tokio::test]
    async fn test_handle_recall_by_scope_missing_query() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler
            .handle_recall_by_scope(Some(json!({
                "scope": "global"
            })))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_recall_by_scope_invalid_scope() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler
            .handle_recall_by_scope(Some(json!({
                "query": "test",
                "scope": "invalid_scope"
            })))
            .await;
        assert!(result.is_ok()); // Should return error in response, not Err
    }

    #[tokio::test]
    async fn test_handle_recall_by_scope_global() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );

        orchestrator
            .remember("Test memory".to_string(), HashMap::new())
            .await
            .expect("Failed to remember");

        let handler = CerebrumHandler::new(orchestrator);
        let result = handler
            .handle_recall_by_scope(Some(json!({
                "query": "test",
                "scope": "global",
                "limit": 10
            })))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_tool_recall_by_scope() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        assert!(handler.get_tool("recall_by_scope").is_some());
    }
}
