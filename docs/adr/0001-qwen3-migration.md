# ADR 0001: Migration to qwen3-embedding (768 → 1024)

## Status

Accepted

## Context

cerebrum stores memories as vector embeddings in a LanceDB table. The system
originally used the `nomic-embed-text` model producing 768-dimensional
vectors, with an 8192-token context. Large memory bodies (notably full plan
bodies stored as `plan:` scopes) exceeded practical embedding limits and
repeatedly failed to store, forcing plans to be kept as temporary files
instead of durable `plan:` references. A larger-context, higher-dimensional
embedding model was required.

## Decision

The default embedding model is now `qwen3-embedding:0.6b`, producing
1024-dimensional vectors with a 32768-token context. The migration is
lossless: raw `content` is stored verbatim in LanceDB and re-embedding is
deterministic. The old 768-dim table (`memories`) is converted into a new
1024-dim table via the `cerebrum-reembed` binary; see
`docs/runbooks/reembed-migration.md` for the operational procedure.

## Assumptions and Invariants

- **A1** — The embedding dimension is config-asserted but probe-verified:
  `FastEmbedEmbedder::embed` rejects any vector whose width differs before
  any write.
- **A2** — A premature model/dimension flip causes no data loss (raw
  `content` is stored in full and re-embedding is lossless) but is a hard
  break: a post-flip startup on an un-converted 768 table fails closed.
- **A5** — A no-manifest state over an existing table must not trust
  config: the physical schema width is authoritative. `ensure_manifest`
  stamps the probed width, never the configured one, when a table already
  exists.
- **A6** — The converter's embedding input must always equal `remember()`'s
  embedding input exactly: `format!("{document_prefix}{content}")`. For
  qwen3, `document_prefix` is the empty string; this empty document prefix
  is the correctness-critical half of the switchover and is load-bearing
  for the symmetry invariant. The query-side instruction (`query_prefix`)
  is a tunable quality knob, opt-in at runtime via `CEREBRUM_QUERY_PREFIX`,
  and does not affect stored vectors.

## Consequences

All new embeddings are 1024-dim. Existing 768-dim stores must be migrated
with `cerebrum-reembed` before the flipped default is activated, or the
server fails closed at construction with a dimension-mismatch error naming
the remediation. The input-bounding guard (default 96,000 characters,
tunable via `CEREBRUM_MAX_INPUT_CHARS`) bounds both `remember()` and the
converter identically, preserving the A6 symmetry invariant.
