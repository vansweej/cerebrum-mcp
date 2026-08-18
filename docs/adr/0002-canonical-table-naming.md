# ADR 0002: Canonical Table Naming

## Status

Accepted

## Context

Prior to this decision, `Config::default()` in
`crates/cerebrum-core/src/config.rs` paired `table_name = "memories"` (the
old 768-dimension table) with `embedding_dim = 1024`. This was an internally
contradictory default: the only on-disk table named `memories` is the dead
768-dim table retained for rollback, while the live 1024-dim qwen3 store is
named `memories_qwen3`. As a result, any launch path that ran on the bare
`Config::default()` — such as the raw pipeline CLI resolving `--plan-ref`,
or choragos's plan lookup — opened the wrong table. The MCP server wrapper
masked this by injecting the environment variable
`CEREBRUM_TABLE_NAME=memories_qwen3`, but that override only reached the MCP
stdio launch path, not direct CLI invocations. This is an "env set in some
launch paths but not others" fragility.

## Decision

The compiled `Config::default().table_name` is now `memories_qwen3`. This
makes the default self-consistent with the 1024-dimension qwen3 embedding
model (`qwen3-embedding:0.6b`) and the on-disk table produced by the qwen3
migration (see ADR 0001). Every launch path now agrees by construction.

## Consequences

Every launch path (MCP wrapper, raw pipeline CLI `--plan-ref` resolution,
choragos plan lookup) now resolves to the correct 1024-dim `memories_qwen3`
table by construction, with no reliance on environment injection.
`CEREBRUM_TABLE_NAME` remains supported as an optional override and is still
set explicitly in home-manager for clarity.

## Deferred Question

The canonical table name now carries a model suffix (`_qwen3`). An
alternative convention was considered and deliberately deferred: a
model-agnostic canonical name like `memories` that is reembed-migrated in
place on every model change. Rationale for deferral: every
embedding-dimension change already forces a full `cerebrum-reembed`
migration, so a model-suffixed name costs nothing today; the convention can
be revisited at the next model change without penalty. See ADR 0001 (qwen3
migration).
