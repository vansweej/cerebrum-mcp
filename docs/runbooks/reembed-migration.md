# Runbook: Migrating to `qwen3-embedding:0.6b` (768 → 1024 dim)

This runbook documents the **manual** steps required to migrate an existing
Cerebrum LanceDB table from the `nomic-embed-text` model (768-dim) to the
`qwen3-embedding:0.6b` model (1024-dim) using the `cerebrum-reembed` binary.

These steps are performed **outside** the normal server startup/runtime
pipeline. They do not require any changes to the Nix flake.

---

## 1. Prerequisite: pull the new model

```bash
ollama pull qwen3-embedding:0.6b
```

(If this has already been done in your environment, skip this step. The flake
itself is untouched by this migration — no Nix changes are required.)

---

## 2. First startup after upgrading the binary

After upgrading to a Cerebrum build that includes the manifest/schema-probe
guard (`ensure_manifest`), the **first startup** against your existing table
will write a truthful manifest reflecting the table's actual probed
dimension (768) and the model that was configured at the time
(`nomic-embed-text`). This is a **no-op** with respect to behavior — no data
is re-embedded, no schema is changed. It simply ensures the on-disk manifest
accurately reflects reality before any migration begins (see ADR A5: never
trust config over a real table).

---

## 3. Run the reembed binary

Set the following environment variables and run the migration binary:

```bash
export CEREBRUM_EMBED_MODEL=qwen3-embedding:0.6b
export CEREBRUM_EMBEDDING_DIM=1024
export CEREBRUM_SOURCE_TABLE=memories
export CEREBRUM_TABLE_NAME=memories_qwen3

cargo run --bin cerebrum-reembed
```

This reads every row from `memories` (the source table, dim=768), re-embeds
each row's content with `qwen3-embedding:0.6b` (applying the same
`document_prefix` used at write time), and writes the result into a new
table `memories_qwen3` (dim=1024). The source table is never modified or
deleted.

On success, the binary prints a summary report including `total`, `written`,
`truncated`, and `verified` counts.

---

## 4. No-op migration refusal

The binary refuses to run a migration where the source and target
dimension **and** model are identical (a no-op), unless you explicitly
override this safety check:

```bash
export CEREBRUM_ALLOW_SAME_DIM=1
```

This prevents accidentally re-running a migration that would do nothing
useful (or worse, silently succeed while masking a misconfiguration).

---

## 5. Switchover

Once the migration completes and reports `verified: true`, switch the
server to use the new table by setting, for the server's environment:

```bash
export CEREBRUM_TABLE_NAME=memories_qwen3
```

Restart the server. It will now read/write against `memories_qwen3`
(dim=1024, model=`qwen3-embedding:0.6b`).

---

## 6. Note on long content and head-only truncation (A9 / F12)

Entries whose content exceeds the configured input bound (default: 96,000
characters, tunable via `CEREBRUM_MAX_INPUT_CHARS`) are **indexed head-only**:
only the first `max_input_chars` characters are sent to the embedder. The
full `content` field is always preserved **verbatim** in storage and remains
fully retrievable by exact scope (e.g. `recall_by_scope` with
`exact_scope: true`) — only the *searchable vector* is affected, never the
stored text.

The migration report's `truncated` count tells you how many entries were
affected by this head-only bounding during the reembed pass, so you can
audit whether any long-form content (e.g. large plan bodies) lost some
semantic search precision on their tail content.

---

## 7. Two-startup fail-closed sequence (F9)

For a pre-existing deployment upgrading through this feature, the sequence
is:

1. **First startup after upgrade** (config still points at the old table,
   old dimension): the manifest is absent, so `ensure_manifest` probes the
   live table, finds its actual width (768), and writes a manifest
   reflecting that reality. This succeeds silently — no behavior change.
2. **Second startup, after flipping `CEREBRUM_TABLE_NAME` (or
   `CEREBRUM_EMBEDDING_DIM`) without running the migration first**: this
   **fails closed** with the pinned mismatch error (see `ensure_manifest.rs`),
   because the manifest now on disk (dim 768) does not match the newly
   configured dimension (1024). The server will refuse to start until you
   either (a) run `cerebrum-reembed` to produce a correctly-dimensioned
   table and manifest, or (b) revert the config back to the original
   table/dimension.

This fail-closed behavior is intentional: it prevents the server from ever
silently reading/writing vectors of the wrong dimension against an
incompatible table.
