# Phase 4: MCP Tool Handler

## Overview
Implement an MCP (Model Context Protocol) server that exposes the Cerebrum memory tools via the MCP protocol. This allows Claude and other AI agents to interact with the memory system through a standardized interface.

## Architecture
- Single MCP server using `rmcp` crate
- Exposes 5 tools: `remember`, `recall`, `memorize`, `forget`, `end_session`
- Tool definitions follow MCP spec with proper input schemas
- Server runs as a subprocess, communicates via stdio

## Constraints
- Use `rmcp` crate for MCP protocol handling
- All tools must be properly typed with JSON schemas
- Error handling via MCP error responses
- Thread-safe access to MemoryOrchestrator
- No breaking changes to Phase 3 code

## Deliverables
1. MCP server implementation (`src/mcp_server.rs`)
2. Tool definitions and handlers
3. Integration tests with MCP client simulation
4. Updated documentation
5. Code quality gates (90%+ coverage, 0 clippy warnings)

## Steps

### Step 1: Add rmcp Dependency
- Add `rmcp` to `Cargo.toml` in `cerebrum` crate
- Verify dependency resolves without conflicts

### Step 2: Implement MCP Server Struct
- Create `src/mcp_server.rs` with `McpServer` struct
- Initialize with MemoryOrchestrator instance
- Implement tool registration

### Step 3: Implement Tool Handlers
- `remember_handler`: Store memory with auto-embedding
- `recall_handler`: Search both tiers with blended results
- `memorize_handler`: Promote Synapse→Cortex
- `forget_handler`: Delete from both tiers
- `end_session_handler`: Clear Synapse with auto-promotion

### Step 4: Add Tool Definitions
- Define JSON schemas for each tool's input
- Include descriptions, parameter types, required fields
- Validate inputs before processing

### Step 5: Implement Server Lifecycle
- `start()`: Initialize MCP server, register tools
- `run()`: Main event loop handling tool calls
- `shutdown()`: Graceful cleanup

### Step 6: Add Integration Tests
- Test each tool via MCP protocol simulation
- Test error handling and validation
- Test concurrent tool calls
- Verify response formats

### Step 7: Update Documentation
- Add MCP server section to `docs/architecture.md`
- Document tool definitions and schemas
- Add usage examples

### Step 8: Verify Code Quality
- Run `cargo test` — all tests passing
- Run `cargo clippy -- -D warnings` — 0 warnings
- Run `cargo fmt` — properly formatted
- Run `cargo tarpaulin` — ≥90% coverage

## Success Criteria
- ✅ All 5 tools exposed via MCP protocol
- ✅ Tool definitions with proper JSON schemas
- ✅ Integration tests covering all tools
- ✅ 90%+ code coverage
- ✅ 0 clippy warnings
- ✅ All tests passing
- ✅ Documentation updated

## Timeline
Estimated: 8-10 hours of focused development

## Notes
- rmcp provides high-level MCP abstractions
- Tool handlers should be async-ready for future phases
- Error responses must follow MCP spec
- Consider logging for debugging tool calls
