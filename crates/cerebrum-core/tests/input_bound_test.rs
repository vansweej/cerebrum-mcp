//! Tests for the input-bounding helper.

use cerebrum_core::input_bound::bound_input;

/// 200 ASCII characters with bound of 100 → truncated head of exactly 100 chars.
#[test]
fn ascii_input_longer_than_bound_is_truncated() {
    let input = "a".repeat(200);
    let (head, truncated) = bound_input(&input, 100);
    assert_eq!(head.chars().count(), 100);
    assert!(truncated);
}

/// 10-character input with bound of 100 → returned as-is, not truncated.
#[test]
fn short_input_returned_unchanged() {
    let input = "hello worl";
    let (head, truncated) = bound_input(input, 100);
    assert_eq!(head, input);
    assert!(!truncated);
}

/// Input of exactly 100 multibyte characters (2 bytes each) with bound 100 →
/// head ends on a char boundary, contains ≤ 100 chars, truncated is false.
#[test]
fn multibyte_exactly_at_bound_ends_on_char_boundary() {
    // '£' is U+00A3, 2 bytes in UTF-8. 100 of them = 200 bytes, 100 chars.
    let input: String = std::iter::repeat('£').take(100).collect();
    let (head, truncated) = bound_input(&input, 100);
    // Must end on a valid char boundary (would panic on slice if not).
    assert!(input.is_char_boundary(head.len()));
    assert!(head.chars().count() <= 100);
    assert!(!truncated);
}
