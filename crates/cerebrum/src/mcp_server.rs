use cerebrum_core::models::MemoryId;
use cerebrum_core::orchestrator::MemoryOrchestrator;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    Annotated, CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams,
    RawContent, ServerCapabilities, ServerInfo, Tool,
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
    default_project: Vec<String>,
}

impl CerebrumHandler {
    /// Create a new Cerebrum handler with the given orchestrator.
    ///
    /// `default_project` defaults to an empty vec (no project preference).
    #[allow(dead_code)]
    pub fn new(orchestrator: Arc<MemoryOrchestrator>) -> Self {
        Self {
            orchestrator,
            default_project: Vec::new(),
        }
    }

    /// Create a new Cerebrum handler with the given orchestrator and a list of
    /// default project names.
    ///
    /// When `default_project` is non-empty the first entry is passed as
    /// `prefer_project` to `recall_with_project` / `recall_by_scope_with_project`
    /// so that memories belonging to those projects are ranked higher.
    pub fn with_default_project(
        orchestrator: Arc<MemoryOrchestrator>,
        default_project: Vec<String>,
    ) -> Self {
        Self {
            orchestrator,
            default_project,
        }
    }

    /// Parse a scope string into a MemoryScope. Defaults to Global when None.
    /// Returns Err(message) for malformed non-empty scope strings.
    fn parse_scope(scope_str: Option<&str>) -> Result<cerebrum_core::MemoryScope, String> {
        match scope_str {
            None | Some("global") => Ok(cerebrum_core::MemoryScope::Global),
            Some(s) => {
                if let Some(id) = s.strip_prefix("user:") {
                    Ok(cerebrum_core::MemoryScope::User(id.to_string()))
                } else if let Some(id) = s.strip_prefix("agent:") {
                    Ok(cerebrum_core::MemoryScope::Agent(id.to_string()))
                } else if let Some(id) = s.strip_prefix("session:") {
                    Ok(cerebrum_core::MemoryScope::Session(id.to_string()))
                } else {
                    Err("Invalid scope format. Use 'global', 'user:<id>', 'agent:<id>', or 'session:<id>'".to_string())
                }
            }
        }
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
                },
                "scope": {
                    "type": "string",
                    "description": "Memory scope: 'global', 'user:<id>', 'agent:<id>', or 'session:<id>'. Defaults to 'global'."
                },
                "type": {
                    "type": "string",
                    "description": "Provenance type. Recommended (not enforced): decision, finding, idea, done, plan, gotcha, convention, context."
                },
                "status": {
                    "type": "string",
                    "description": "Lifecycle status. Recommended: active, parked, done. Defaults to 'active'."
                },
                "confidence": {
                    "type": "string",
                    "description": "Optional confidence: proposed, confirmed, verified."
                },
                "project": {
                    "description": "Project tag(s): a string or an array of strings. Merged with the server's CEREBRUM_PROJECT default.",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
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
                },
                "prefer_project": {
                    "type": "string",
                    "description": "Optional project name to gently boost matching memories in ranking (never excludes others). Defaults to the server's CEREBRUM_PROJECT."
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
                },
                "prefer_project": {
                    "type": "string",
                    "description": "Optional project name to gently boost matching memories in ranking (never excludes others). Defaults to the server's CEREBRUM_PROJECT."
                },
                "exact_scope": {
                    "type": "boolean",
                    "description": "When true, restrict results to memories whose scope is EXACTLY the given scope — global memories are excluded entirely rather than merely deprioritized. Use this when you already know the precise scope you want (e.g. fetching a specific plan or session by its scope id), to avoid a large corpus of high-salience global memories crowding the target out of the result window. Defaults to false."
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

        let salience = args.get("salience").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;

        // Parse optional scope (defaults to Global).
        let scope = match Self::parse_scope(args.get("scope").and_then(|v| v.as_str())) {
            Ok(s) => s,
            Err(msg) => {
                return Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(msg),
                    None,
                )]));
            }
        };

        // Build provenance metadata from the tool arguments.
        use cerebrum_core::provenance;
        let mut metadata: HashMap<String, String> = HashMap::new();

        // status: trim + lowercase; default to "active"
        let status_val = args
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "active".to_string());
        metadata.insert(provenance::KEY_STATUS.to_string(), status_val);

        // type: insert only when present
        if let Some(type_val) = args
            .get("type")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
        {
            metadata.insert(provenance::KEY_TYPE.to_string(), type_val);
        }

        // confidence: insert only when present
        if let Some(conf_val) = args
            .get("confidence")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
        {
            metadata.insert(provenance::KEY_CONFIDENCE.to_string(), conf_val);
        }

        // project: collect provided values (string or array), prepend defaults, dedup
        let provided_projects: Vec<String> = match args.get("project") {
            Some(serde_json::Value::String(s)) => {
                let t = s.trim().to_string();
                if t.is_empty() {
                    vec![]
                } else {
                    vec![t]
                }
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            _ => vec![],
        };

        let mut project_vec: Vec<String> = self.default_project.clone();
        project_vec.extend(provided_projects);

        // de-duplicate preserving order
        let mut seen_projects = std::collections::HashSet::new();
        project_vec.retain(|p| seen_projects.insert(p.clone()));

        if !project_vec.is_empty() {
            metadata.insert(
                provenance::KEY_PROJECT.to_string(),
                provenance::project_array_json(&project_vec),
            );
        }

        match self
            .orchestrator
            .remember_with_salience(content.clone(), metadata, scope, salience)
            .await
        {
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

        let prefer: Option<String> = args
            .get("prefer_project")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| self.default_project.first().cloned());

        match self
            .orchestrator
            .recall_with_project(query, limit, prefer.as_deref())
            .await
        {
            Ok(results) => {
                info!("Recall found {} results", results.len());
                let result_json: Vec<Value> = results
                    .iter()
                    .map(|entry| {
                        json!({
                            "id": entry.id.to_string(),
                            "content": entry.content,
                            "salience": entry.salience,
                            "scope": entry.scope.as_str(),
                            "tier": format!("{:?}", entry.tier),
                            "timestamp": entry.timestamp,
                            "metadata": json!(entry.metadata)
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

        // Parse scope string (shared with handle_remember).
        let scope = match Self::parse_scope(Some(scope_str)) {
            Ok(s) => s,
            Err(msg) => {
                return Ok(CallToolResult::error(vec![Annotated::new(
                    RawContent::text(msg),
                    None,
                )]));
            }
        };

        let prefer: Option<String> = args
            .get("prefer_project")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| self.default_project.first().cloned());

        let exact_scope = args
            .get("exact_scope")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        match self
            .orchestrator
            .recall_by_scope_with_project(query, scope, limit, prefer.as_deref(), exact_scope)
            .await
        {
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
                            "timestamp": entry.timestamp,
                            "metadata": json!(entry.metadata)
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
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Cerebrum two-tier memory: remember, recall, memorize, forget, end_session, recall_by_scope.",
        )
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
    async fn remember_persists_caller_salience() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        let store_result = handler
            .handle_remember(Some(json!({
                "content": "high priority note",
                "salience": 0.9
            })))
            .await;
        assert!(store_result.is_ok(), "remember should succeed");

        let recall_result = handler
            .handle_recall(Some(json!({
                "query": "high priority note",
                "limit": 5
            })))
            .await
            .expect("recall should succeed");

        let text = match &recall_result.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text content from recall"),
        };
        let parsed: Value = serde_json::from_str(&text).expect("recall payload should be JSON");
        let results = parsed["results"]
            .as_array()
            .expect("results should be an array");
        let entry = results
            .iter()
            .find(|e| e["content"] == "high priority note")
            .expect("stored memory should be returned by recall");
        let salience = entry["salience"]
            .as_f64()
            .expect("salience should be a number");

        assert!(
            (salience - 0.9).abs() < 1e-6,
            "expected salience 0.9, got {salience}"
        );
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
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                cerebrum_core::MemoryScope::Global,
            )
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
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                cerebrum_core::MemoryScope::Global,
            )
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
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                cerebrum_core::MemoryScope::Global,
            )
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
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                cerebrum_core::MemoryScope::Global,
            )
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

    #[tokio::test]
    async fn end_session_promotes_only_high_salience() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        handler
            .handle_remember(Some(json!({
                "content": "KEEP critical fact",
                "salience": 0.9
            })))
            .await
            .expect("remember high-salience should succeed");
        handler
            .handle_remember(Some(json!({
                "content": "DROP trivial note",
                "salience": 0.1
            })))
            .await
            .expect("remember low-salience should succeed");

        let end_result = handler
            .handle_end_session(Some(json!({ "promotion_threshold": 0.7 })))
            .await;
        assert!(end_result.is_ok(), "end_session should succeed");

        // High-salience memory must survive, promoted into Cortex.
        let keep_recall = handler
            .handle_recall(Some(json!({ "query": "KEEP critical fact", "limit": 10 })))
            .await
            .expect("recall should succeed");
        let keep_text = match &keep_recall.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text content from recall"),
        };
        let keep_parsed: Value =
            serde_json::from_str(&keep_text).expect("recall payload should be JSON");
        let keep_results = keep_parsed["results"]
            .as_array()
            .expect("results should be an array");
        let kept = keep_results
            .iter()
            .find(|e| e["content"] == "KEEP critical fact")
            .expect("high-salience memory should survive end_session");
        assert_eq!(
            kept["tier"], "Cortex",
            "surviving memory should have been promoted to Cortex"
        );

        // Low-salience memory must be gone from all tiers.
        let drop_recall = handler
            .handle_recall(Some(json!({ "query": "DROP trivial note", "limit": 10 })))
            .await
            .expect("recall should succeed");
        let drop_text = match &drop_recall.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text content from recall"),
        };
        let drop_parsed: Value =
            serde_json::from_str(&drop_text).expect("recall payload should be JSON");
        let drop_results = drop_parsed["results"]
            .as_array()
            .expect("results should be an array");
        assert!(
            !drop_results
                .iter()
                .any(|e| e["content"] == "DROP trivial note"),
            "low-salience memory should have been dropped by end_session"
        );
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
        // Verify get_info returns a valid ServerInfo that advertises tools.
        assert!(!info.server_info.name.is_empty());
        assert!(
            info.capabilities.tools.is_some(),
            "tools capability must be advertised"
        );
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
            .remember(
                "Test memory".to_string(),
                HashMap::new(),
                cerebrum_core::MemoryScope::Global,
            )
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
    async fn remember_persists_provenance_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let embedder: Arc<dyn cerebrum_core::Embedder> =
            Arc::new(cerebrum_core::embedder::MockEmbedder::new());
        let orchestrator = Arc::new(
            MemoryOrchestrator::new(embedder, dir.path(), "memories", 384)
                .await
                .expect("Failed to create orchestrator"),
        );
        let handler = CerebrumHandler::new(orchestrator);

        // Store a memory with provenance fields.
        let store_result = handler
            .handle_remember(Some(json!({
                "content": "provenance test memory",
                "type": "decision",
                "status": "active",
                "project": ["cerebrum", "atlas"]
            })))
            .await;
        assert!(store_result.is_ok(), "remember should succeed");

        // Recall the memory back.
        let recall_result = handler
            .handle_recall(Some(json!({
                "query": "provenance test memory",
                "limit": 5
            })))
            .await
            .expect("recall should succeed");

        // Extract the JSON payload from the first content item.
        let text = match &recall_result.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text content from recall"),
        };
        let parsed: Value = serde_json::from_str(&text).expect("recall payload should be JSON");
        let results = parsed["results"]
            .as_array()
            .expect("results should be an array");

        // Find the entry we just stored.
        let entry = results
            .iter()
            .find(|e| e["content"] == "provenance test memory")
            .expect("stored memory should be returned by recall");

        let metadata = &entry["metadata"];
        assert!(metadata.is_object(), "metadata should be a JSON object");

        // status must be "active" (lowercased by the handler).
        assert_eq!(
            metadata["status"], "active",
            "expected status 'active', got {:?}",
            metadata["status"]
        );

        // type must be "decision" (lowercased by the handler).
        assert_eq!(
            metadata["type"], "decision",
            "expected type 'decision', got {:?}",
            metadata["type"]
        );

        // project must be a JSON array string containing both project names.
        let project_raw = metadata["project"]
            .as_str()
            .expect("project metadata should be a string");
        let project_parsed: Value =
            serde_json::from_str(project_raw).expect("project metadata should be valid JSON");
        let project_arr = project_parsed
            .as_array()
            .expect("project metadata should be a JSON array");
        let project_names: Vec<&str> = project_arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            project_names.contains(&"cerebrum"),
            "project array should contain 'cerebrum', got {:?}",
            project_names
        );
        assert!(
            project_names.contains(&"atlas"),
            "project array should contain 'atlas', got {:?}",
            project_names
        );
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
