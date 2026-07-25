# Feature: Provenance metadata on stored memories

## Overview
Add structured provenance metadata to every stored memory and use it at recall time to keep cross-session, cross-repo context clean. Four tags — `project` (multi-valued JSON array), `type`, `status`, `confidence` — all live inside the existing `metadata_json` blob (no new Arrow column, no DB migration). Recall deprioritizes (never excludes) memories via score multipliers for `status` and optional `project` affinity, falling back to the `CEREBRUM_PROJECT` env default when no explicit `prefer_project` is supplied. This is Plan 1 (cerebrum-mcp capability); Plan 2 (agora agent wiring) follows separately and is out of scope here.

## Phase 1: Provenance vocabulary and weighting primitives

Commit message: `feat: add provenance metadata module with status and project weighting`

Coverage: skip

### Step 1: Create the provenance module

Create a new file `crates/cerebrum-core/src/provenance.rs`.

Define the metadata-key constants and pure helper functions used to tag and rank memories. All four provenance tags live inside the existing `MemoryEntry.metadata` (`HashMap<String, String>`); this module never adds database columns.

Add, with a `//!` module doc comment and `///` doc comments on every public item:

- `use std::collections::HashMap;`
- `pub const KEY_PROJECT: &str = "project";`
- `pub const KEY_TYPE: &str = "type";`
- `pub const KEY_STATUS: &str = "status";`
- `pub const KEY_CONFIDENCE: &str = "confidence";`
- `pub fn project_array_json(projects: &[String]) -> String` — return a JSON array string of the de-duplicated, order-preserving, non-empty entries (trim each; drop empties). An empty input returns `"[]"`. Use `serde_json`.
- `pub fn parse_project_array(value: &str) -> Vec<String>` — parse `value` as a JSON array of strings; if JSON parsing fails, return a single-element vec containing the trimmed `value` (unless it is empty, then an empty vec). This makes the function robust to hand-written tags.
- `pub fn status_weight(metadata: &HashMap<String, String>) -> f32` — look up `KEY_STATUS`, trim + lowercase it, then return `0.4` for `"parked"`, `0.3` for `"done"`, and `1.0` for everything else including `"active"`, unknown values, and a missing key. Add `const PARKED_WEIGHT: f32 = 0.4;` and `const DONE_WEIGHT: f32 = 0.3;` above the function.
- `pub fn project_weight(metadata: &HashMap<String, String>, prefer_project: Option<&str>) -> f32` — return `1.0` when `prefer_project` is `None`. Otherwise parse `metadata.get(KEY_PROJECT)` with `parse_project_array`; return `1.0` when the parsed list is empty (untagged / back-catalogue stays neutral) OR contains the `prefer_project` value (case-sensitive exact match); otherwise return `NON_MEMBER_WEIGHT`. Add `const NON_MEMBER_WEIGHT: f32 = 0.7;` above the function. Never return 0.0 — this is a deprioritization, never an exclusion.

Add a `#[cfg(test)] mod tests` covering: status weights for parked/done/active/unknown/missing; project weight for None-prefer, member, non-member, and untagged-memory cases; and `parse_project_array` for a JSON array input, a bare non-JSON string input, and an empty string.

### Step 2: Register the provenance module and re-exports

Edit `crates/cerebrum-core/src/lib.rs`. Add `pub mod provenance;` alongside the other `pub mod` declarations. Following the existing re-export style already present in that file, also re-export the provenance key constants and the two weight functions (`status_weight`, `project_weight`) at the crate root so other crates can reference them as `cerebrum_core::status_weight` etc. Do not remove or reorder existing exports.

### Step 3: Add the ScoredMemory type

Edit `crates/cerebrum-core/src/models.rs`. Add a new public struct near the `MemoryEntry` definition:

```rust
/// A memory paired with its blended-and-weighted rank score.
///
/// Carries the store-computed score (`sim*0.7 + salience*0.3`, multiplied by
/// the provenance status and project weights) across the tier boundary so the
/// orchestrator can merge results from both tiers by a single comparable score
/// instead of re-ranking by salience alone.
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    /// The underlying memory entry.
    pub entry: MemoryEntry,
    /// Blended similarity/salience score after provenance weighting.
    pub score: f32,
}
```

Keep all existing items in the file unchanged.

## Phase 2: Score-carrying retrieval across both tiers

Commit message: `refactor: carry provenance-weighted scores out of memory stores`

Coverage: skip

### Step 1: Add scored retrieval methods to the MemoryStore trait

Edit `crates/cerebrum-core/src/traits.rs`.

Import `ScoredMemory` from the models module (the trait already imports `MemoryEntry, MemoryId, MemoryScope` — add `ScoredMemory` to that import).

Add two new REQUIRED methods to the `MemoryStore` trait, each with a `///` doc comment:

```rust
async fn retrieve_scored(
    &self,
    query_vec: &[f32],
    limit: usize,
    prefer_project: Option<&str>,
) -> Result<Vec<ScoredMemory>>;

async fn retrieve_by_scope_scored(
    &self,
    query_vec: &[f32],
    scope: &MemoryScope,
    limit: usize,
    prefer_project: Option<&str>,
) -> Result<Vec<ScoredMemory>>;
```

Convert the existing `retrieve` and `retrieve_by_scope` methods into DEFAULT methods (provide a body in the trait) that delegate to the scored variants with `prefer_project = None` and strip the score:

```rust
async fn retrieve(&self, query_vec: &[f32], limit: usize) -> Result<Vec<MemoryEntry>> {
    Ok(self
        .retrieve_scored(query_vec, limit, None)
        .await?
        .into_iter()
        .map(|s| s.entry)
        .collect())
}
```

and the analogous default for `retrieve_by_scope`. Update the doc comments to note these are convenience wrappers over the scored variants.

In the `#[cfg(test)] mod tests` `DefaultStore` impl in the same file, replace its `retrieve` / `retrieve_by_scope` implementations with `retrieve_scored` (return the two fixed entries each wrapped in `ScoredMemory { entry, score: 1.0 }`) and `retrieve_by_scope_scored` (return an empty vec), matching the previous returned data. Keep the existing `store`/`delete`/`list`/`len`/`is_empty` impls and the three test functions unchanged.

### Step 2: Apply provenance weighting inside the Synapse store

Edit `crates/cerebrum-core/src/synapse.rs`.

Replace the `MemoryStore::retrieve` and `MemoryStore::retrieve_by_scope` implementations with `retrieve_scored` and `retrieve_by_scope_scored` carrying the new signatures (add `limit`/`scope` as before, plus `prefer_project: Option<&str>`). Reuse the existing scan logic; where the code currently computes `let score = (similarity * 0.7) + (entry.salience * 0.3);`, multiply that by the provenance weights:

```rust
let base = (similarity * 0.7) + (entry.salience * 0.3);
let score = base
    * crate::provenance::status_weight(&entry.metadata)
    * crate::provenance::project_weight(&entry.metadata, prefer_project);
```

Collect `crate::models::ScoredMemory { entry: entry.clone(), score }`, sort by `score` descending, take `limit`, and return `Vec<ScoredMemory>`. Keep the `retrieve_by_scope_scored` scope filter (`entry.scope.matches(scope)`) unchanged. Import `ScoredMemory` as needed. Do NOT re-add the old `retrieve`/`retrieve_by_scope` methods — they now come from the trait defaults. Leave the inherent `list`/`len`/`is_empty`/`clear`/`cosine_similarity` methods and all existing tests unchanged; the tests call `retrieve(&vec, n)` and must still compile and pass via the trait default.

### Step 3: Apply provenance weighting inside the LanceDB Cortex store

Edit `crates/cerebrum-core/src/lancedb_cortex.rs`.

Replace the `MemoryStore::retrieve` and `MemoryStore::retrieve_by_scope` implementations with `retrieve_scored` and `retrieve_by_scope_scored` carrying the new signatures (add `prefer_project: Option<&str>`). Keep the existing table scan, the empty-table early return, and the scope SQL pushdown in `retrieve_by_scope_scored` unchanged. Where the code currently computes `let score = sim * 0.7 + record.salience * 0.3;`, first convert the record to an entry (in `retrieve_scored`, call `record.to_entry()?`; in `retrieve_by_scope_scored`, the entry is already produced by the existing `record.to_entry().ok()?` step), then:

```rust
let base = sim * 0.7 + entry.salience * 0.3;
let score = base
    * crate::provenance::status_weight(&entry.metadata)
    * crate::provenance::project_weight(&entry.metadata, prefer_project);
```

Return `Vec<crate::models::ScoredMemory>` built as `ScoredMemory { entry, score }`, sorted by `score` descending and truncated to `limit`. Import `ScoredMemory`. Do NOT re-add the old `retrieve`/`retrieve_by_scope` methods — they come from the trait defaults. Leave `store`, `delete`, `list`, `len`, `is_empty`, `search_by_salience`, the `schema()` function, `LanceDBMemoryRecord`, and all existing tests unchanged; existing tests call `retrieve(&qvec(), n)` and must pass via the trait default.

## Phase 3: Merge tiers by carried score in the orchestrator

Commit message: `feat: rank recall by carried provenance score and add prefer_project`

Coverage: skip

### Step 1: Add project-aware recall methods to the orchestrator

Edit `crates/cerebrum-core/src/orchestrator.rs`.

Add two new public async methods and make the existing `recall` and `recall_by_scope` delegate to them (mirroring the existing `remember` / `remember_with_salience` wrapper idiom already in this file):

```rust
pub async fn recall(&self, query: String, limit: usize) -> Result<Vec<MemoryEntry>> {
    self.recall_with_project(query, limit, None).await
}

pub async fn recall_with_project(
    &self,
    query: String,
    limit: usize,
    prefer_project: Option<&str>,
) -> Result<Vec<MemoryEntry>> { ... }
```

and analogously `recall_by_scope(query, scope, limit)` delegating to `recall_by_scope_with_project(query, scope, limit, prefer_project)`.

In each `_with_project` body: embed the query once with `self.query_prefix` (unchanged), call `retrieve_scored` (or `retrieve_by_scope_scored`) on both `self.synapse` and `self.cortex` passing `limit` and `prefer_project`, extend both `Vec<ScoredMemory>` into one vector, sort it by `score` descending (`partial_cmp`, `unwrap_or(Ordering::Equal)`), then dedup by `entry.id` keeping the first (now-highest-scored) occurrence using a `HashSet<MemoryId>`, take `limit`, and map to `entry`. This replaces the previous salience-only sort. Import `ScoredMemory` from `crate::models`. Update the `recall` doc comment paragraph that currently says "Ranks by salience (descending)" to describe score-based ranking with provenance weighting. Leave `remember`, `memorize`, `forget`, `end_session`, and the accessors unchanged. Existing tests call `recall`/`recall_by_scope` with the old two/three-argument signatures and must still compile and pass.

## Phase 4: Ingest provenance tags via the remember tool

Commit message: `feat: accept provenance tags and env-default project on remember`

Coverage: skip

### Step 1: Add an env-default project field to the handler

Edit `crates/cerebrum/src/mcp_server.rs` and `crates/cerebrum/src/main.rs`.

In `mcp_server.rs`, add a field `default_project: Vec<String>` to the `CerebrumHandler` struct. Keep the existing `pub fn new(orchestrator: Arc<MemoryOrchestrator>) -> Self` constructor working by defaulting `default_project` to an empty `Vec` (so all existing tests that call `CerebrumHandler::new(...)` are unaffected). Add a second constructor `pub fn with_default_project(orchestrator: Arc<MemoryOrchestrator>, default_project: Vec<String>) -> Self` with a `///` doc comment.

In `main.rs`, read the `CEREBRUM_PROJECT` environment variable (`std::env::var("CEREBRUM_PROJECT")`), split it on commas into a `Vec<String>` of trimmed non-empty entries (empty/unset → empty vec), and construct the handler via `CerebrumHandler::with_default_project(orchestrator, projects)` instead of `CerebrumHandler::new(orchestrator)`.

### Step 2: Extend the remember tool schema with provenance fields

Edit `crates/cerebrum/src/mcp_server.rs`, function `remember_tool()`. Add four optional properties to the JSON schema, alongside the existing `content`, `salience`, `scope`:

- `type`: string — "Provenance type. Recommended (not enforced): decision, finding, idea, done, plan, gotcha, convention, context."
- `status`: string — "Lifecycle status. Recommended: active, parked, done. Defaults to 'active'."
- `confidence`: string — "Optional confidence: proposed, confirmed, verified."
- `project`: "Project tag(s): a string or an array of strings. Merged with the server's CEREBRUM_PROJECT default." Express this in JSON schema as `"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]`.

Keep `"required": ["content"]`. Do not change other tool definitions in this step.

### Step 3: Populate provenance metadata in handle_remember

Edit `crates/cerebrum/src/mcp_server.rs`, function `handle_remember`. Replace the line `let metadata = HashMap::new();` with construction of a populated `HashMap<String, String>` using the provenance key constants from `cerebrum_core::provenance`:

- `status`: read `args["status"]` as a string, trim + lowercase; if absent use `"active"`. Insert under `provenance::KEY_STATUS`.
- `type`: if `args["type"]` is a present string, trim + lowercase and insert under `provenance::KEY_TYPE`. Omit the key when absent.
- `confidence`: same handling as `type`, under `provenance::KEY_CONFIDENCE`.
- `project`: collect provided project values — accept either a JSON string (one value) or a JSON array of strings from `args["project"]`. Build a `Vec<String>` = `self.default_project` followed by the provided values, trimmed and de-duplicated preserving order. If the resulting vec is non-empty, insert `provenance::project_array_json(&vec)` under `provenance::KEY_PROJECT`; if empty, omit the key.

Pass this populated metadata map into the existing `remember_with_salience(content, metadata, scope, salience)` call. Keep the rest of the function (content/salience/scope parsing, success/error responses) unchanged. Ensure `use std::collections::HashMap;` remains.

## Phase 5: Surface provenance on recall

Commit message: `feat: add prefer_project recall arg and return metadata in results`

Coverage: skip

### Step 1: Add prefer_project to the recall tool schemas

Edit `crates/cerebrum/src/mcp_server.rs`, functions `recall_tool()` and `recall_by_scope_tool()`. Add one optional property `prefer_project` (string) to each schema with the description: "Optional project name to gently boost matching memories in ranking (never excludes others). Defaults to the server's CEREBRUM_PROJECT." Do not change the existing required fields.

### Step 2: Thread prefer_project through the recall handlers

Edit `crates/cerebrum/src/mcp_server.rs`, functions `handle_recall` and `handle_recall_by_scope`. In each, resolve the preferred project: read `args["prefer_project"]` as a trimmed string if present; otherwise fall back to the first element of `self.default_project` (the env-default). The result is an `Option<String>`. Call the new orchestrator methods `recall_with_project(query, limit, prefer.as_deref())` and `recall_by_scope_with_project(query, scope, limit, prefer.as_deref())` respectively instead of `recall` / `recall_by_scope`. Keep query/scope/limit parsing and error handling unchanged.

### Step 3: Include and normalize metadata in recall responses

Edit `crates/cerebrum/src/mcp_server.rs`, functions `handle_recall` and `handle_recall_by_scope`. Normalize both result JSON shapes to emit the SAME fields for each entry: `id`, `content`, `salience`, `scope` (use `entry.scope.as_str()`), `tier` (`format!("{:?}", entry.tier)`), `timestamp`, and `metadata` (serialize `entry.metadata`, a `HashMap<String, String>`, as a JSON object via `json!(entry.metadata)`). This means adding `scope` to the `handle_recall` mapping and adding `metadata` to BOTH mappings. Keep the surrounding `success`/`count`/`results` envelope and error handling unchanged.

## Phase 6: Provenance integration tests

Commit message: `test: cover provenance ingest, weighting, and recall surface`

Coverage: skip

### Step 1: Add store-level provenance tests

Create `crates/cerebrum-core/tests/provenance_tests.rs`. Using `MockEmbedder` (dimension 384) and `tempfile::tempdir()` for any LanceDB path, add async tests:

- Synapse round-trip: store a `MemoryEntry` whose metadata carries `status=active` and `project=["repoA"]` (JSON array string), then call `retrieve_scored(&query_vec, 10, None)` and assert the metadata survives on the returned entry.
- Backward compatibility: store an entry with an empty metadata map, retrieve it, and assert it is returned (neutral weighting, never dropped).
- Invalid enum accepted: store an entry with `status="bogus"`; assert it is returned by `retrieve_scored` and that `cerebrum_core::provenance::status_weight` of its metadata is `1.0` (unknown treated as active, never rejected).
- Status deprioritization: store two entries with identical embeddings and salience but `status=active` vs `status=parked`; assert the active one ranks first in `retrieve_scored` output ordering.
- Project membership boost: with two equally-similar entries tagged `project=["repoA"]` and `project=["repoB"]`, assert that `retrieve_scored(&qv, 10, Some("repoA"))` ranks the repoA entry first, and that an untagged entry is not penalized relative to a non-member entry.

Use the crate's public API (`cerebrum_core::provenance`, `SynapseMemory`, `LanceDBCortex`, `MemoryEntry::builder`) and follow the Result-returning style; avoid `unwrap()` in non-test-assertion paths where a `?`/expect message is clearer.

### Step 2: Add MCP round-trip provenance test

Add a test to the existing `#[cfg(test)] mod tests` in `crates/cerebrum/src/mcp_server.rs`. Build a handler with `MockEmbedder` and a `tempdir` orchestrator, call `handle_remember` with content plus `type`, `status`, and `project` arguments, then call `handle_recall` for that content and parse the JSON response. Assert the matching result entry contains a `metadata` object carrying the expected `status`, `type`, and `project` values. Follow the existing test helpers and JSON-extraction pattern already used in `remember_persists_caller_salience`.

## Phase 7: Documentation

Commit message: `docs: document provenance metadata tags and recall weighting`

### Step 1: Document provenance in the README

Edit `README.md`. Add a "Provenance metadata" section documenting: the four tags (`project`, `type`, `status`, `confidence`) and their recommended vocabularies; that values are soft-validated (unknown values are accepted, never rejected, and never cause a memory to be dropped); the `CEREBRUM_PROJECT` environment variable as the default project source; the optional `prefer_project` recall argument and that it gently boosts (never excludes) matching memories; and that all tags are stored inside the existing `metadata_json` blob with no schema migration and full backward compatibility for existing rows. Match the existing README tone and formatting; do not remove unrelated sections.
