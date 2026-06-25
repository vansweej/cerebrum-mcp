use cerebrum_core::orchestrator::MemoryOrchestrator;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, InitializeRequestParams, InitializeResult,
    ListToolsResult, PaginatedRequestParams, Tool, TextContent,
};
use rmcp::service::RequestContext;
use rmcp::{McpError, RoleServer};
use serde_json::{json, Value};
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
        Tool {
            name: "remember".into(),
            description: Some(
                "Store a memory in the Synapse (short-term) tier with automatic embedding generation"
                    .into(),
            ),
            inputSchema: json!({
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
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session identifier for tracking context"
                    }
                },
                "required": ["content"]
            }),
            ..Default::default()
        }
    }

    /// Get the recall tool definition
    fn recall_tool() -> Tool {
        Tool {
            name: "recall".into(),
            description: Some(
                "Search memories across both Synapse and Cortex tiers using semantic similarity"
                    .into(),
            ),
            inputSchema: json!({
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
            }),
            ..Default::default()
        }
    }

    /// Get the memorize tool definition
    fn memorize_tool() -> Tool {
        Tool {
            name: "memorize".into(),
            description: Some(
                "Promote a memory from Synapse (short-term) to Cortex (long-term) storage"
                    .into(),
            ),
            inputSchema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to promote"
                    }
                },
                "required": ["memory_id"]
            }),
            ..Default::default()
        }
    }

    /// Get the forget tool definition
    fn forget_tool() -> Tool {
        Tool {
            name: "forget".into(),
            description: Some("Delete a memory from both Synapse and Cortex tiers".into()),
            inputSchema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to delete"
                    }
                },
                "required": ["memory_id"]
            }),
            ..Default::default()
        }
    }

    /// Get the end_session tool definition
    fn end_session_tool() -> Tool {
        Tool {
            name: "end_session".into(),
            description: Some(
                "End a session: clear Synapse and auto-promote memories above salience threshold to Cortex"
                    .into(),
            ),
            inputSchema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session ID to end"
                    },
                    "promotion_threshold": {
                        "type": "number",
                        "description": "Salience threshold for auto-promotion (0.0-1.0, default: 0.7)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["session_id"]
            }),
            ..Default::default()
        }
    }

    /// Handle remember tool call
    fn handle_remember(&self, arguments: Option<Value>) -> Result<CallToolResult, McpError> {
        let args = arguments.ok_or_else(|| {
            McpError::invalid_params("Missing arguments for remember tool", None)
        })?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                McpError::invalid_params("Missing required field: content", None)
            })?;

        let salience = args
            .get("salience")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        match self.orchestrator.remember(content, salience, session_id) {
            Ok(memory_id) => {
                info!("Memory stored: {}", memory_id);
                let response = json!({
                    "success": true,
                    "memory_id": memory_id.to_string(),
                    "message": format!("Memory stored with ID: {}", memory_id)
                });
                Ok(CallToolResult::success(TextContent {
                    text: response.to_string(),
                    ..Default::default()
                }))
            }
            Err(e) => {
                error!("Failed to store memory: {:?}", e);
                Ok(CallToolResult::error(TextContent {
                    text: format!("Failed to store memory: {}", e),
                    ..Default::default()
                }))
            }
        }
    }

    /// Handle recall tool call
    fn handle_recall(&self, arguments: Option<Value>) -> Result<CallToolResult, McpError> {
        let args = arguments.ok_or_else(|| {
            McpError::invalid_params("Missing arguments for recall tool", None)
        })?;

        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                McpError::invalid_params("Missing required field: query", None)
            })?;

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        match self.orchestrator.recall(query, limit) {
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
                            "created_at": entry.created_at
                        })
                    })
                    .collect();

                let response = json!({
                    "success": true,
                    "count": results.len(),
                    "results": result_json
                });
                Ok(CallToolResult::success(TextContent {
                    text: response.to_string(),
                    ..Default::default()
                }))
            }
            Err(e) => {
                error!("Failed to recall memories: {:?}", e);
                Ok(CallToolResult::error(TextContent {
                    text: format!("Failed to recall memories: {}", e),
                    ..Default::default()
                }))
            }
        }
    }

    /// Handle memorize tool call
    fn handle_memorize(&self, arguments: Option<Value>) -> Result<CallToolResult, McpError> {
        let args = arguments.ok_or_else(|| {
            McpError::invalid_params("Missing arguments for memorize tool", None)
        })?;

        let memory_id_str = args
            .get("memory_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                McpError::invalid_params("Missing required field: memory_id", None)
            })?;

        let memory_id = cerebrum_core::models::MemoryId::from_string(memory_id_str)
            .map_err(|e| McpError::invalid_params(format!("Invalid memory ID: {}", e), None))?;

        match self.orchestrator.memorize(&memory_id) {
            Ok(_) => {
                info!("Memory promoted to Cortex: {}", memory_id);
                let response = json!({
                    "success": true,
                    "memory_id": memory_id.to_string(),
                    "message": format!("Memory {} promoted to Cortex", memory_id)
                });
                Ok(CallToolResult::success(TextContent {
                    text: response.to_string(),
                    ..Default::default()
                }))
            }
            Err(e) => {
                error!("Failed to promote memory: {:?}", e);
                Ok(CallToolResult::error(TextContent {
                    text: format!("Failed to promote memory: {}", e),
                    ..Default::default()
                }))
            }
        }
    }

    /// Handle forget tool call
    fn handle_forget(&self, arguments: Option<Value>) -> Result<CallToolResult, McpError> {
        let args = arguments.ok_or_else(|| {
            McpError::invalid_params("Missing arguments for forget tool", None)
        })?;

        let memory_id_str = args
            .get("memory_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                McpError::invalid_params("Missing required field: memory_id", None)
            })?;

        let memory_id = cerebrum_core::models::MemoryId::from_string(memory_id_str)
            .map_err(|e| McpError::invalid_params(format!("Invalid memory ID: {}", e), None))?;

        match self.orchestrator.forget(&memory_id) {
            Ok(_) => {
                info!("Memory forgotten: {}", memory_id);
                let response = json!({
                    "success": true,
                    "memory_id": memory_id.to_string(),
                    "message": format!("Memory {} deleted from all tiers", memory_id)
                });
                Ok(CallToolResult::success(TextContent {
                    text: response.to_string(),
                    ..Default::default()
                }))
            }
            Err(e) => {
                error!("Failed to forget memory: {:?}", e);
                Ok(CallToolResult::error(TextContent {
                    text: format!("Failed to forget memory: {}", e),
                    ..Default::default()
                }))
            }
        }
    }

    /// Handle end_session tool call
    fn handle_end_session(&self, arguments: Option<Value>) -> Result<CallToolResult, McpError> {
        let args = arguments.ok_or_else(|| {
            McpError::invalid_params("Missing arguments for end_session tool", None)
        })?;

        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                McpError::invalid_params("Missing required field: session_id", None)
            })?;

        let threshold = args
            .get("promotion_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        match self.orchestrator.end_session(session_id, threshold) {
            Ok(promoted_count) => {
                info!(
                    "Session {} ended, {} memories promoted",
                    session_id, promoted_count
                );
                let response = json!({
                    "success": true,
                    "session_id": session_id,
                    "promoted_count": promoted_count,
                    "message": format!("Session ended, {} memories promoted to Cortex", promoted_count)
                });
                Ok(CallToolResult::success(TextContent {
                    text: response.to_string(),
                    ..Default::default()
                }))
            }
            Err(e) => {
                error!("Failed to end session: {:?}", e);
                Ok(CallToolResult::error(TextContent {
                    text: format!("Failed to end session: {}", e),
                    ..Default::default()
                }))
            }
        }
    }
}

impl ServerHandler for CerebrumHandler {
    fn initialize(
        &self,
        request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<InitializeResult, McpError>> + '_ {
        async move {
            context.peer.set_peer_info(request);
            info!("Cerebrum MCP server initialized");
            Ok(self.get_info())
        }
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, McpError>> + '_ {
        async move {
            debug!("Listing available tools");
            Ok(ListToolsResult {
                tools: vec![
                    Self::remember_tool(),
                    Self::recall_tool(),
                    Self::memorize_tool(),
                    Self::forget_tool(),
                    Self::end_session_tool(),
                ],
                nextCursor: None,
            })
        }
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, McpError>> + '_ {
        async move {
            debug!("Calling tool: {}", request.name);
            match request.name.as_ref() {
                "remember" => self.handle_remember(request.arguments.map(|obj| {
                    serde_json::Value::Object(obj)
                })),
                "recall" => self.handle_recall(request.arguments.map(|obj| {
                    serde_json::Value::Object(obj)
                })),
                "memorize" => self.handle_memorize(request.arguments.map(|obj| {
                    serde_json::Value::Object(obj)
                })),
                "forget" => self.handle_forget(request.arguments.map(|obj| {
                    serde_json::Value::Object(obj)
                })),
                "end_session" => self.handle_end_session(request.arguments.map(|obj| {
                    serde_json::Value::Object(obj)
                })),
                _ => Err(McpError::method_not_found::<rmcp::model::CallToolRequestMethod>()),
            }
        }
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        match name {
            "remember" => Some(Self::remember_tool()),
            "recall" => Some(Self::recall_tool()),
            "memorize" => Some(Self::memorize_tool()),
            "forget" => Some(Self::forget_tool()),
            "end_session" => Some(Self::end_session_tool()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_creation() {
        let orchestrator = Arc::new(MemoryOrchestrator::new());
        let _handler = CerebrumHandler::new(orchestrator);
    }

    #[test]
    fn test_remember_tool_definition() {
        let tool = CerebrumHandler::remember_tool();
        assert_eq!(tool.name, "remember");
        assert!(tool.description.is_some());
    }

    #[test]
    fn test_recall_tool_definition() {
        let tool = CerebrumHandler::recall_tool();
        assert_eq!(tool.name, "recall");
        assert!(tool.description.is_some());
    }

    #[test]
    fn test_memorize_tool_definition() {
        let tool = CerebrumHandler::memorize_tool();
        assert_eq!(tool.name, "memorize");
        assert!(tool.description.is_some());
    }

    #[test]
    fn test_forget_tool_definition() {
        let tool = CerebrumHandler::forget_tool();
        assert_eq!(tool.name, "forget");
        assert!(tool.description.is_some());
    }

    #[test]
    fn test_end_session_tool_definition() {
        let tool = CerebrumHandler::end_session_tool();
        assert_eq!(tool.name, "end_session");
        assert!(tool.description.is_some());
    }

    #[test]
    fn test_get_tool() {
        let orchestrator = Arc::new(MemoryOrchestrator::new());
        let handler = CerebrumHandler::new(orchestrator);

        assert!(handler.get_tool("remember").is_some());
        assert!(handler.get_tool("recall").is_some());
        assert!(handler.get_tool("memorize").is_some());
        assert!(handler.get_tool("forget").is_some());
        assert!(handler.get_tool("end_session").is_some());
        assert!(handler.get_tool("unknown").is_none());
    }

    #[test]
    fn test_handle_remember_missing_content() {
        let orchestrator = Arc::new(MemoryOrchestrator::new());
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler.handle_remember(Some(json!({})));
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_remember_success() {
        let orchestrator = Arc::new(MemoryOrchestrator::new());
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler.handle_remember(Some(json!({
            "content": "Test memory",
            "salience": 0.8
        })));
        assert!(result.is_ok());
    }

    #[test]
    fn test_handle_recall_missing_query() {
        let orchestrator = Arc::new(MemoryOrchestrator::new());
        let handler = CerebrumHandler::new(orchestrator);

        let result = handler.handle_recall(Some(json!({})));
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_recall_success() {
        let orchestrator = Arc::new(MemoryOrchestrator::new());
        let _ = orchestrator.remember("Test memory", 0.8, None);

        let handler = CerebrumHandler::new(orchestrator);
        let result = handler.handle_recall(Some(json!({
            "query": "Test",
            "limit": 10
        })));
        assert!(result.is_ok());
    }
}
