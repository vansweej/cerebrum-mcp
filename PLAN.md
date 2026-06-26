# Feature: Semantic Two-Tier Agent Memory (Ollama + LanceDB)

## Overview
Implement real semantic memory for cerebrum-mcp: short-term RAM-based Synapse (session-scoped), long-term LanceDB-based Cortex (persistent), both searchable via Ollama nomic-embed-text (768-dim) embeddings. Embedder lives only on the orchestrator; stores operate on precomputed query vectors. One embed per operation (remember/recall). Ollama is a hard dependency.

## Phase 1: Flip MemoryStore Trait to Vector Seam

Commit message: `refactor: centralize embedding in orchestrator, stores take precomputed vectors`

### Step 1: Update MemoryStore trait signature
- Change `remember(&mut self, entry: MemoryEntry) -> Result<()>` to `remember(&mut self, entry: MemoryEntry, embedding: &[f32]) -> Result<()>`.
- Change `recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>>` to `recall(&self, query_vector: &[f32], limit: usize) -> Result<Vec<MemoryEntry>>`.
- Change `recall_by_scope(&self, scope: &str, query: &str, limit: usize) -> Result<Vec<MemoryEntry>>` to `recall_by_scope(&self, scope: &str, query_vector: &[f32], limit: usize) -> Result<Vec<MemoryEntry>>`.
- Change `promote(&mut self, entry_id: &str) -> Result<()>` to `promote(&mut self, entry_id: &str, embedding: &[f32]) -> Result<()>`.
- Remove any embedder field or embedding logic from trait definition.

### Step 2: Refactor SynapseMemory
- Remove `embedder: Arc<dyn Embedder>` field.
- Update `remember()` to accept `embedding: &[f32]` parameter; store it in `MemoryEntry.embedding`.
- Update `recall()` to accept `query_vector: &[f32]`; use it directly for similarity search (no embedding step).
- Update `recall_by_scope()` similarly.
- Update `promote()` to accept `embedding: &[f32]`; use it when updating the entry.
- Keep the `0.7*sim + 0.3*salience` blend logic intact.
- Update constructor: `pub fn new() -> Self` (no embedder parameter).

### Step 3: Refactor LanceDBCortex
- Remove `embedder: Arc<dyn Embedder>` field.
- Remove the dimension guard that checked `embedder.dim() == config.embedding_dim` (this will be validated in orchestrator warmup).
- Update `remember()` to accept `embedding: &[f32]` parameter; store it in `MemoryEntry.embedding`.
- Update `recall()` to accept `query_vector: &[f32]`; use it directly for LanceDB vector search.
- Update `recall_by_scope()` similarly.
- Update `promote()` to accept `embedding: &[f32]`.
- Update constructor: `pub fn new(path: &Path, table_name: &str, embedding_dim: usize) -> Result<Self>` (no embedder parameter).

### Step 4: Update MemoryOrchestrator (partial)
- Add `embedder: Arc<dyn Embedder>` field to orchestrator.
- Update `remember()` to:
  1. Embed the document text using `self.embedder.embed(&text)`.
  2. Call `self.synapse.remember(entry.clone(), &embedding)`.
  3. Call `self.cortex.remember(entry, &embedding)`.
- Update `recall()` to:
  1. Embed the query text using `self.embedder.embed(&query)`.
  2. Call `self.synapse.recall(&embedding, limit)`.
  3. Call `self.cortex.recall(&embedding, limit)`.
  4. Merge and deduplicate results.
- Update `recall_by_scope()` similarly.
- Update `promote()` to:
  1. Retrieve the entry from one of the stores.
  2. Use its stored `embedding` field (or re-embed if not present).
  3. Call both stores' `promote()` with the vector.
- Keep constructor as-is for now (will be replaced in Phase 4).

### Step 5: Update all call sites
- In `main.rs`: update any direct calls to `synapse.remember()`, `cortex.remember()`, etc. to pass dummy vectors (e.g., `&[]`) for now (will be fixed in Phase 4).
- In tests: update mock calls to pass vectors.
- Verify no compilation errors; all tests should still pass (using MockEmbedder at 384 dims for now).

### Verification
- `cargo fmt && cargo clippy -D warnings && cargo test --workspace` — all green.
- No live Ollama required yet; MockEmbedder still in use.

---

## Phase 2: Add Ollama Configuration

Commit message: `feat: add Ollama config (url, model, prefixes, dim 768)`

### Step 1: Extend Config struct
- Add `ollama_url: String` (default: `"http://localhost:11434"`).
- Add `embed_model: String` (default: `"nomic-embed-text"`).
- Add `embedding_dim: usize` (change from 384 to `768`).
- Add `query_prefix: String` (default: `"search_query: "`).
- Add `document_prefix: String` (default: `"search_document: "`).
- Add `ollama_timeout_secs: u64` (default: `30`).
- Add `ollama_warmup_timeout_secs: u64` (default: `60`).

### Step 2: Update Config deserialization
- Ensure all new fields have sensible defaults.
- Add validation: `embedding_dim` must be > 0.

### Step 3: Update MemoryEntry
- Verify `embedding: Option<Vec<f32>>` field exists (it should already).
- No changes needed; this field will hold 768-dim vectors going forward.

### Step 4: Update test fixtures
- Update any hardcoded `embedding_dim: 384` to `768` in tests.
- Update MockEmbedder to emit 384-dim vectors (test-only; production will use real Ollama at 768).

### Verification
- `cargo fmt && cargo clippy -D warnings && cargo test --workspace` — all green.
- Config can be serialized/deserialized with new fields.

---

## Phase 3: Fix FastEmbedEmbedder API

Commit message: `fix: correct FastEmbedEmbedder to use batch /api/embed endpoint`

### Step 1: Fix request/response shape
- Change request from `{model, prompt}` to `{model, input: [text]}` (batch format).
- Change response parsing from `{embedding: [...]}` to `{embeddings: [[...]]}` (batch format).
- Extract the first embedding from the batch: `embeddings[0]`.

### Step 2: Add per-instance HTTP client
- Add `client: reqwest::Client` field to `FastEmbedEmbedder`.
- Initialize it in `new()` with sensible timeouts (use `config.ollama_timeout_secs`).

### Step 3: Add configurable dimension
- Add `dim: usize` field to `FastEmbedEmbedder`.
- Update `new()` to accept `dim: usize` parameter.
- Update `dim()` method to return `self.dim`.

### Step 4: Update tests
- Fix mock responses to use batch shape: `{embeddings: [[0.1f32; 384]]}`.
- Add test for correct request shape: `{model: "nomic-embed-text", input: ["test"]}`.

### Verification
- `cargo fmt && cargo clippy -D warnings && cargo test --workspace` — all green.
- Tests pass with mocked batch responses.

---

## Phase 4: Wire Ollama via from_config with Warmup Probe

Commit message: `feat: add MemoryOrchestrator::from_config with Ollama warmup probe and prefixes`

### Step 1: Implement from_config constructor
- Create `pub async fn from_config(config: &Config) -> Result<Self>`.
- Instantiate `FastEmbedEmbedder::new(config.ollama_url.clone(), config.embed_model.clone(), config.embedding_dim)`.
- Instantiate `SynapseMemory::new()`.
- Instantiate `LanceDBCortex::new(&data_dir, "memories", config.embedding_dim)`.
- Return `MemoryOrchestrator { embedder, synapse, cortex }`.

### Step 2: Add warmup probe
- In `from_config`, after instantiating the embedder, call a warmup probe:
  1. Embed a test string: `"test"`.
  2. Verify the returned vector has length `config.embedding_dim` (should be 768).
  3. If mismatch or error, return `Err` with descriptive message.
  4. This pre-loads the Ollama model and validates configuration.

### Step 3: Apply prefixes in orchestrator methods
- In `remember()`: prepend `config.document_prefix` to the document text before embedding.
  - Example: `let prefixed = format!("{}{}", config.document_prefix, text);`
  - Embed `prefixed`, not `text`.
- In `recall()`: prepend `config.query_prefix` to the query text before embedding.
  - Example: `let prefixed = format!("{}{}", config.query_prefix, query);`
  - Embed `prefixed`, not `query`.
- In `recall_by_scope()`: apply query prefix similarly.
- Store the original (unprefixed) text in `MemoryEntry.text`.

### Step 4: Update MemoryOrchestrator constructor
- Keep the old `new()` constructor for backward compatibility (used in tests with MockEmbedder).
- Make `from_config()` the primary production path.

### Verification
- `cargo fmt && cargo clippy -D warnings && cargo test --workspace` — all green.
- Warmup probe validates Ollama connection and dimension.
- Prefixes are applied correctly (can be verified in logs or by inspecting embedded text in tests).

---

## Phase 5: Switch main.rs to from_config and Add E2E Test

Commit message: `feat: switch main.rs to MemoryOrchestrator::from_config, add wiremock E2E test`

### Step 1: Update main.rs
- Replace `MemoryOrchestrator::new()` with `MemoryOrchestrator::from_config(&Config::default()).await?`.
- Ensure error handling is in place (from_config is async and can fail).
- Log the warmup probe result (e.g., "Ollama warmup successful, embedding_dim=768").

### Step 2: Create end-to-end test with wiremock
- Create `crates/cerebrum-core/tests/ollama_integration_tests.rs`.
- Use `wiremock` to mock Ollama `/api/embed` endpoint.
- Test scenario:
  1. Mock Ollama to return 768-dim vectors.
  2. Call `MemoryOrchestrator::from_config()` with mocked Ollama URL.
  3. Verify warmup probe succeeds.
  4. Call `remember()` with a document; verify it's stored in both Synapse and Cortex.
  5. Call `recall()` with a query; verify results are returned from both stores.
  6. Verify prefixes were applied (by inspecting the request body sent to mocked Ollama).
  7. Verify Synapse is offline (no Ollama call for Synapse-only recall).

### Step 3: Ensure test data cleanup
- Use temporary directories for LanceDB in tests (not `~/.local/share/cerebrum`).
- Clean up after each test.

### Verification
- `cargo fmt && cargo clippy -D warnings && cargo test --workspace` — all green.
- E2E test passes without live Ollama (wiremock only).
- `cargo test -- --ignored` (if any ignored tests exist) can be run against live Ollama for 768-dim validation.

---

## Phase 6: Documentation

Commit message: `docs: add Ollama runtime dependency, prefixes, schema-wipe caveat`

### Step 1: Update README.md
- Add section: "Runtime Dependencies"
  - Ollama must be running at `http://localhost:11434` (configurable via `Config.ollama_url`).
  - Model `nomic-embed-text` must be pulled: `ollama pull nomic-embed-text`.
  - Embedding dimension is 768; older schemas (384-dim) are incompatible.
- Add section: "Semantic Search Prefixes"
  - Explain `search_query:` and `search_document:` prefixes (nomic best practice).
  - Configurable via `Config.query_prefix` and `document_prefix`.
- Add section: "Schema Migration"
  - Warn that changing `embedding_dim` requires wiping `~/.local/share/cerebrum/data/cerebrum/memories.lance`.
  - Provide command: `rm -rf ~/.local/share/cerebrum/data/cerebrum/`.

### Step 2: Update CONTEXT.md or architecture docs (if present)
- Document the two-tier memory architecture: Synapse (RAM, fast, session-scoped) + Cortex (LanceDB, persistent, semantic).
- Explain the embedder seam: orchestrator embeds once per operation, stores take vectors.
- Explain the warmup probe: validates Ollama connection and dimension at startup.

### Step 3: Add inline code comments
- In `orchestrator.rs`: document the prefix application and warmup probe.
- In `fastembed_embedder.rs`: document the batch API shape.
- In `synapse.rs` and `lancedb_cortex.rs`: document that they accept precomputed vectors.

### Verification
- README is clear and complete.
- All new features are documented.
- No typos or broken links.

---

## Post-Implementation Checklist

After all phases are complete:

1. **Wipe old schema:**
   ```bash
   rm -rf ~/.local/share/cerebrum/data/cerebrum/
   ```

2. **Verify build:**
   ```bash
   nix develop . --command cargo fmt && cargo clippy -D warnings && cargo test --workspace
   ```

3. **Test with live Ollama (optional):**
   ```bash
   # Ensure Ollama is running and nomic-embed-text is pulled
   ollama pull nomic-embed-text
   nix develop . --command cargo test -- --ignored
   ```

4. **Build release:**
   ```bash
   nix develop . --command cargo build --release
   ```

5. **Commit strategy:**
   - One commit per phase (6 commits total).
   - Use conventional commit messages as specified above.
   - Do not push; user will integrate into home-manager.

6. **Verification points:**
   - Phase 1: Stores take vectors; orchestrator embeds; all tests pass.
   - Phase 2: Config has Ollama fields; dim is 768.
   - Phase 3: FastEmbedEmbedder uses batch API; tests pass.
   - Phase 4: from_config works; warmup probe validates; prefixes applied.
   - Phase 5: main.rs uses from_config; E2E test passes.
   - Phase 6: Docs complete; no typos.

---

## Key Design Decisions (Locked)

- **Embedder seam:** Embedder lives only on `MemoryOrchestrator`. Both `SynapseMemory` and `LanceDBCortex` take precomputed `&[f32]` vectors.
- **One embed per operation:** `remember()` embeds document once; `recall()` embeds query once; both fan vector to Synapse and Cortex.
- **Prefixes:** `search_query: ` and `search_document: ` applied in orchestrator before embedding (nomic best practice).
- **Warmup probe:** Validates Ollama connection and dimension at startup; pre-loads model to avoid cold-start hang.
- **Ollama mandatory:** No fallback to MockEmbedder in production; fail-fast on connection/model errors.
- **Dimension:** 768 (nomic-embed-text truth); old 384-dim schemas must be wiped.
- **Synapse offline:** No Ollama calls for Synapse-only recall; fast, session-scoped.
- **Cortex persistent:** LanceDB stores vectors; survives session restart; semantically searchable.

---

## Reference Implementation

See `/Users/Shared/PhilipsDev/athenaeum-mcp/crates/core/src/{embed.rs, engine.rs, store.rs}` for seam shape and patterns. Cerebrum's two-tier memory is different (Synapse + Cortex vs. athenaeum's flat passage library), but the embedder seam and orchestrator wiring follow the same design.
