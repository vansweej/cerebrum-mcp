//! Input-bounding guard for embedding inputs.
//!
//! Ollama's embedding endpoint rejects inputs that exceed the model's context
//! window. This module provides a character-count-based head-only bounding
//! function that is applied before every embed call to prevent `400` errors.
//!
//! The `MAX_EMBED_INPUT_CHARS` constant is the compile-time default; the actual
//! runtime bound is sourced from `Config::max_input_chars` and is tunable via
//! the `CEREBRUM_MAX_INPUT_CHARS` environment variable.

/// Default maximum number of characters to send to the embedder.
///
/// This default (96,000 characters ≈ 24k tokens) sits well inside
/// qwen3-embedding's 32,768-token context while being large enough to embed
/// a full plan body. The runtime bound is read from `Config::max_input_chars`.
pub const MAX_EMBED_INPUT_CHARS: usize = 96_000;

/// Bound `input` to at most `max_chars` characters, returning the head slice.
///
/// Counts **characters**, not bytes: the returned slice contains at most
/// `max_chars` Unicode scalar values and always ends on a valid char boundary.
///
/// Returns `(head, truncated)` where:
/// - `head` is `&input[..end]` with `end` the byte offset of the
///   `max_chars`-th char boundary (or `input.len()` when the input is shorter).
/// - `truncated` is `true` when the input had more than `max_chars` characters.
pub fn bound_input(input: &str, max_chars: usize) -> (&str, bool) {
    match input.char_indices().nth(max_chars) {
        Some((byte_offset, _)) => (&input[..byte_offset], true),
        None => (input, false),
    }
}
