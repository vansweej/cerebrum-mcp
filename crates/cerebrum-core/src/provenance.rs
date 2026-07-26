//! Provenance metadata helpers for tagging and ranking [`crate::models::MemoryEntry`] instances.
//!
//! All four provenance tags live inside the existing `MemoryEntry.metadata`
//! (`HashMap<String, String>`); this module never adds database columns.
//!
//! # Metadata keys
//! | Constant | Key string | Example value |
//! |---|---|---|
//! | [`KEY_PROJECT`] | `"project"` | `["cerebrum","atlas"]` (JSON array) |
//! | [`KEY_TYPE`] | `"type"` | `"decision"` |
//! | [`KEY_STATUS`] | `"status"` | `"active"` / `"parked"` / `"done"` |
//! | [`KEY_CONFIDENCE`] | `"confidence"` | `"0.85"` |

use std::collections::HashMap;

/// Metadata key for the project tag (JSON array of project names).
pub const KEY_PROJECT: &str = "project";

/// Metadata key for the memory type tag (e.g. `"decision"`, `"note"`).
pub const KEY_TYPE: &str = "type";

/// Metadata key for the lifecycle status tag (`"active"`, `"parked"`, `"done"`).
pub const KEY_STATUS: &str = "status";

/// Metadata key for the confidence score tag (stringified `f32`, e.g. `"0.85"`).
pub const KEY_CONFIDENCE: &str = "confidence";

/// Serialise a slice of project names into a compact JSON array string.
///
/// De-duplicates while preserving the first-seen order. Empty strings and
/// strings that are blank after trimming are silently dropped.
///
/// # Examples
/// ```
/// # use cerebrum_core::provenance::project_array_json;
/// assert_eq!(project_array_json(&[]), "[]");
/// assert_eq!(project_array_json(&["a".into(), "b".into()]), r#"["a","b"]"#);
/// assert_eq!(project_array_json(&["a".into(), "a".into()]), r#"["a"]"#);
/// ```
pub fn project_array_json(projects: &[String]) -> String {
    let mut seen = std::collections::HashSet::new();
    let deduped: Vec<&str> = projects
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter(|s| seen.insert(*s))
        .collect();

    serde_json::to_string(&deduped).unwrap_or_else(|_| "[]".to_string())
}

/// Parse a project-array metadata value back into a `Vec<String>`.
///
/// If `value` is a valid JSON array of strings, the trimmed, non-empty
/// elements are returned. Otherwise the function falls back to treating
/// `value` as a single bare project name: it returns a one-element vec
/// containing the trimmed value, or an empty vec when the trimmed value is
/// empty.
///
/// This makes the function robust to hand-written tags that were never
/// serialised through [`project_array_json`].
///
/// # Examples
/// ```
/// # use cerebrum_core::provenance::parse_project_array;
/// assert_eq!(parse_project_array(r#"["a","b"]"#), vec!["a", "b"]);
/// assert_eq!(parse_project_array("cerebrum"), vec!["cerebrum"]);
/// assert_eq!(parse_project_array(""), Vec::<String>::new());
/// ```
pub fn parse_project_array(value: &str) -> Vec<String> {
    // Attempt JSON parse first.
    if let Ok(serde_json::Value::Array(arr)) = serde_json::from_str(value) {
        return arr
            .into_iter()
            .filter_map(|v| {
                if let serde_json::Value::String(s) = v {
                    let t = s.trim().to_string();
                    if t.is_empty() {
                        None
                    } else {
                        Some(t)
                    }
                } else {
                    None
                }
            })
            .collect();
    }

    // Fallback: treat the whole value as a bare project name.
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        vec![]
    } else {
        vec![trimmed]
    }
}

/// Ranking weight applied to memories whose status is `"parked"`.
const PARKED_WEIGHT: f32 = 0.4;

/// Ranking weight applied to memories whose status is `"done"`.
const DONE_WEIGHT: f32 = 0.3;

/// Return a salience multiplier based on the `"status"` metadata tag.
///
/// | Status value | Weight |
/// |---|---|
/// | `"parked"` | `0.4` |
/// | `"done"` | `0.3` |
/// | `"active"`, unknown, or missing | `1.0` |
///
/// The comparison is case-insensitive and trims surrounding whitespace.
pub fn status_weight(metadata: &HashMap<String, String>) -> f32 {
    match metadata.get(KEY_STATUS) {
        None => 1.0,
        Some(raw) => match raw.trim().to_lowercase().as_str() {
            "parked" => PARKED_WEIGHT,
            "done" => DONE_WEIGHT,
            _ => 1.0,
        },
    }
}

/// Ranking weight applied to memories that do **not** belong to the
/// preferred project.  Never `0.0` — this is a deprioritisation, not an
/// exclusion.
const NON_MEMBER_WEIGHT: f32 = 0.7;

/// Return a salience multiplier based on project membership.
///
/// | Condition | Weight |
/// |---|---|
/// | `prefer_project` is `None` | `1.0` |
/// | Memory has no project tag (untagged) | `1.0` |
/// | Memory's project list contains `prefer_project` (exact, case-sensitive) | `1.0` |
/// | Memory's project list does **not** contain `prefer_project` | `0.7` |
///
/// The function never returns `0.0`; non-member memories are deprioritised,
/// not excluded.
pub fn project_weight(metadata: &HashMap<String, String>, prefer_project: Option<&str>) -> f32 {
    let preferred = match prefer_project {
        None => return 1.0,
        Some(p) => p,
    };

    let projects = match metadata.get(KEY_PROJECT) {
        None => return 1.0, // untagged — stays neutral
        Some(raw) => parse_project_array(raw),
    };

    if projects.is_empty() {
        return 1.0; // back-catalogue / untagged stays neutral
    }

    if projects.iter().any(|p| p == preferred) {
        1.0
    } else {
        NON_MEMBER_WEIGHT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── status_weight ────────────────────────────────────────────────────────

    #[test]
    fn status_weight_parked() {
        let mut m = HashMap::new();
        m.insert(KEY_STATUS.to_string(), "parked".to_string());
        assert_eq!(status_weight(&m), PARKED_WEIGHT);
    }

    #[test]
    fn status_weight_parked_uppercase() {
        let mut m = HashMap::new();
        m.insert(KEY_STATUS.to_string(), "PARKED".to_string());
        assert_eq!(status_weight(&m), PARKED_WEIGHT);
    }

    #[test]
    fn status_weight_done() {
        let mut m = HashMap::new();
        m.insert(KEY_STATUS.to_string(), "done".to_string());
        assert_eq!(status_weight(&m), DONE_WEIGHT);
    }

    #[test]
    fn status_weight_done_mixed_case() {
        let mut m = HashMap::new();
        m.insert(KEY_STATUS.to_string(), "  Done  ".to_string());
        assert_eq!(status_weight(&m), DONE_WEIGHT);
    }

    #[test]
    fn status_weight_active() {
        let mut m = HashMap::new();
        m.insert(KEY_STATUS.to_string(), "active".to_string());
        assert_eq!(status_weight(&m), 1.0);
    }

    #[test]
    fn status_weight_unknown_value() {
        let mut m = HashMap::new();
        m.insert(KEY_STATUS.to_string(), "in-progress".to_string());
        assert_eq!(status_weight(&m), 1.0);
    }

    #[test]
    fn status_weight_missing_key() {
        let m = HashMap::new();
        assert_eq!(status_weight(&m), 1.0);
    }

    // ── project_weight ───────────────────────────────────────────────────────

    #[test]
    fn project_weight_none_prefer() {
        let m = HashMap::new();
        assert_eq!(project_weight(&m, None), 1.0);
    }

    #[test]
    fn project_weight_member() {
        let mut m = HashMap::new();
        m.insert(
            KEY_PROJECT.to_string(),
            project_array_json(&["cerebrum".to_string(), "atlas".to_string()]),
        );
        assert_eq!(project_weight(&m, Some("cerebrum")), 1.0);
    }

    #[test]
    fn project_weight_non_member() {
        let mut m = HashMap::new();
        m.insert(
            KEY_PROJECT.to_string(),
            project_array_json(&["atlas".to_string()]),
        );
        assert_eq!(project_weight(&m, Some("cerebrum")), NON_MEMBER_WEIGHT);
    }

    #[test]
    fn project_weight_untagged_memory() {
        // No KEY_PROJECT entry at all — back-catalogue stays neutral.
        let m = HashMap::new();
        assert_eq!(project_weight(&m, Some("cerebrum")), 1.0);
    }

    #[test]
    fn project_weight_empty_array_tag() {
        let mut m = HashMap::new();
        m.insert(KEY_PROJECT.to_string(), "[]".to_string());
        // Empty list → neutral (back-catalogue).
        assert_eq!(project_weight(&m, Some("cerebrum")), 1.0);
    }

    #[test]
    fn project_weight_never_zero() {
        let mut m = HashMap::new();
        m.insert(
            KEY_PROJECT.to_string(),
            project_array_json(&["other".to_string()]),
        );
        let w = project_weight(&m, Some("cerebrum"));
        assert!(w > 0.0, "project_weight must never return 0.0");
    }

    // ── parse_project_array ──────────────────────────────────────────────────

    #[test]
    fn parse_project_array_json_input() {
        let result = parse_project_array(r#"["cerebrum","atlas"]"#);
        assert_eq!(result, vec!["cerebrum", "atlas"]);
    }

    #[test]
    fn parse_project_array_bare_string() {
        let result = parse_project_array("cerebrum");
        assert_eq!(result, vec!["cerebrum"]);
    }

    #[test]
    fn parse_project_array_empty_string() {
        let result = parse_project_array("");
        assert!(result.is_empty());
    }

    #[test]
    fn parse_project_array_whitespace_only() {
        let result = parse_project_array("   ");
        assert!(result.is_empty());
    }

    #[test]
    fn parse_project_array_json_with_blanks() {
        let result = parse_project_array(r#"["a","","b"]"#);
        assert_eq!(result, vec!["a", "b"]);
    }

    // ── project_array_json ───────────────────────────────────────────────────

    #[test]
    fn project_array_json_empty_input() {
        assert_eq!(project_array_json(&[]), "[]");
    }

    #[test]
    fn project_array_json_deduplicates() {
        let result = project_array_json(&["a".into(), "b".into(), "a".into()]);
        assert_eq!(result, r#"["a","b"]"#);
    }

    #[test]
    fn project_array_json_drops_blanks() {
        let result = project_array_json(&["a".into(), "  ".into(), "b".into()]);
        assert_eq!(result, r#"["a","b"]"#);
    }

    #[test]
    fn project_array_json_roundtrip() {
        let projects = vec!["cerebrum".to_string(), "atlas".to_string()];
        let json = project_array_json(&projects);
        let parsed = parse_project_array(&json);
        assert_eq!(parsed, projects);
    }
}
