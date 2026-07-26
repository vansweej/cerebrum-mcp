//! Provenance metadata round-trip and ranking tests.
//!
//! Covers:
//! - Metadata survival through Synapse and LanceDB round-trips
//! - Backward compatibility with empty metadata
//! - Unknown/invalid status values treated as neutral (weight 1.0)
//! - `status=parked` deprioritises entries relative to `status=active`
//! - `prefer_project` boosts matching entries; untagged entries are not penalised

use cerebrum_core::{
    embedder::MockEmbedder,
    lancedb_cortex::LanceDBCortex,
    models::{MemoryEntry, MemoryId, MemoryScope, MemoryTier},
    provenance::{project_array_json, status_weight, KEY_PROJECT, KEY_STATUS},
    synapse::SynapseMemory,
    traits::{Embedder, MemoryStore},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Build a normalised query vector of dimension 384 (matches MockEmbedder).
fn qvec() -> Vec<f32> {
    let raw = vec![1.0_f32; 384];
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.into_iter().map(|x| x / norm).collect()
}

/// Embed a text string using `MockEmbedder` (dimension 384).
async fn embed(text: &str) -> Vec<f32> {
    let embedder = MockEmbedder::new();
    embedder
        .embed(text)
        .await
        .expect("MockEmbedder must not fail on non-empty text")
}

// ============================================================================
// 1. Synapse round-trip: metadata survives store → retrieve_scored
// ============================================================================

#[tokio::test]
async fn test_synapse_metadata_round_trip() {
    let synapse = SynapseMemory::new();
    let embedding = embed("synapse round-trip").await;

    let mut metadata = HashMap::new();
    metadata.insert(KEY_STATUS.to_string(), "active".to_string());
    metadata.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoA".to_string()]),
    );

    let id = MemoryId::new();
    let mut entry = MemoryEntry::builder(id, "synapse round-trip".to_string())
        .embedding(embedding)
        .salience(0.7)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    entry.metadata = metadata.clone();

    synapse.store(entry).await.expect("store must succeed");

    let results = synapse
        .retrieve_scored(&qvec(), 10, None)
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 1, "exactly one result expected");
    let returned_meta = &results[0].entry.metadata;
    assert_eq!(
        returned_meta.get(KEY_STATUS).map(String::as_str),
        Some("active"),
        "status metadata must survive round-trip"
    );
    assert_eq!(
        returned_meta.get(KEY_PROJECT),
        metadata.get(KEY_PROJECT),
        "project metadata must survive round-trip"
    );
}

// ============================================================================
// 2. LanceDB round-trip: metadata survives store → retrieve_scored
// ============================================================================

#[tokio::test]
async fn test_lancedb_metadata_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
        .await
        .expect("LanceDBCortex::new must succeed");

    let embedding = embed("lancedb round-trip").await;

    let mut metadata = HashMap::new();
    metadata.insert(KEY_STATUS.to_string(), "active".to_string());
    metadata.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoA".to_string()]),
    );

    let id = MemoryId::new();
    let mut entry = MemoryEntry::builder(id, "lancedb round-trip".to_string())
        .embedding(embedding)
        .salience(0.7)
        .tier(MemoryTier::Cortex)
        .scope(MemoryScope::Global)
        .build();
    entry.metadata = metadata.clone();

    cortex.store(entry).await.expect("store must succeed");

    let results = cortex
        .retrieve_scored(&qvec(), 10, None)
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 1, "exactly one result expected");
    let returned_meta = &results[0].entry.metadata;
    assert_eq!(
        returned_meta.get(KEY_STATUS).map(String::as_str),
        Some("active"),
        "status metadata must survive LanceDB round-trip"
    );
    assert_eq!(
        returned_meta.get(KEY_PROJECT),
        metadata.get(KEY_PROJECT),
        "project metadata must survive LanceDB round-trip"
    );
}

// ============================================================================
// 3. Backward compatibility: empty metadata → entry is returned, not dropped
// ============================================================================

#[tokio::test]
async fn test_synapse_empty_metadata_backward_compat() {
    let synapse = SynapseMemory::new();
    let embedding = embed("empty metadata entry").await;

    let entry = MemoryEntry::builder(MemoryId::new(), "empty metadata entry".to_string())
        .embedding(embedding)
        .salience(0.5)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    // entry.metadata is empty by default

    synapse.store(entry).await.expect("store must succeed");

    let results = synapse
        .retrieve_scored(&qvec(), 10, None)
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(
        results.len(),
        1,
        "entry with empty metadata must not be dropped"
    );
    assert!(
        results[0].entry.metadata.is_empty(),
        "metadata must remain empty"
    );
}

#[tokio::test]
async fn test_lancedb_empty_metadata_backward_compat() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
        .await
        .expect("LanceDBCortex::new must succeed");

    let embedding = embed("empty metadata lancedb").await;

    let entry = MemoryEntry::builder(MemoryId::new(), "empty metadata lancedb".to_string())
        .embedding(embedding)
        .salience(0.5)
        .tier(MemoryTier::Cortex)
        .scope(MemoryScope::Global)
        .build();

    cortex.store(entry).await.expect("store must succeed");

    let results = cortex
        .retrieve_scored(&qvec(), 10, None)
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(
        results.len(),
        1,
        "entry with empty metadata must not be dropped from LanceDB"
    );
}

// ============================================================================
// 4. Invalid / unknown status value is accepted and treated as weight 1.0
// ============================================================================

#[tokio::test]
async fn test_synapse_invalid_status_accepted_as_neutral() {
    let synapse = SynapseMemory::new();
    let embedding = embed("bogus status entry").await;

    let mut metadata = HashMap::new();
    metadata.insert(KEY_STATUS.to_string(), "bogus".to_string());

    let id = MemoryId::new();
    let mut entry = MemoryEntry::builder(id, "bogus status entry".to_string())
        .embedding(embedding)
        .salience(0.5)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    entry.metadata = metadata;

    synapse.store(entry).await.expect("store must succeed");

    let results = synapse
        .retrieve_scored(&qvec(), 10, None)
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 1, "bogus status must not drop the entry");

    let returned_meta = &results[0].entry.metadata;
    let weight = status_weight(returned_meta);
    assert!(
        (weight - 1.0_f32).abs() < f32::EPSILON,
        "unknown status must yield weight 1.0, got {weight}"
    );
}

// ============================================================================
// 5. Status deprioritisation: active ranks above parked (same embedding + salience)
// ============================================================================

#[tokio::test]
async fn test_synapse_active_ranks_above_parked() {
    let synapse = SynapseMemory::new();
    // Use identical embeddings so similarity is equal; salience is also equal.
    let embedding = embed("status ranking test").await;

    let mut active_meta = HashMap::new();
    active_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let mut parked_meta = HashMap::new();
    parked_meta.insert(KEY_STATUS.to_string(), "parked".to_string());

    let active_id = MemoryId::new();
    let mut active_entry = MemoryEntry::builder(active_id, "active memory".to_string())
        .embedding(embedding.clone())
        .salience(0.6)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    active_entry.metadata = active_meta;

    let parked_id = MemoryId::new();
    let mut parked_entry = MemoryEntry::builder(parked_id, "parked memory".to_string())
        .embedding(embedding)
        .salience(0.6)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    parked_entry.metadata = parked_meta;

    synapse
        .store(active_entry)
        .await
        .expect("store active must succeed");
    synapse
        .store(parked_entry)
        .await
        .expect("store parked must succeed");

    let results = synapse
        .retrieve_scored(&qvec(), 10, None)
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 2, "both entries must be returned");

    let first_status = results[0]
        .entry
        .metadata
        .get(KEY_STATUS)
        .map(String::as_str)
        .unwrap_or("");
    assert_eq!(
        first_status, "active",
        "active entry must rank above parked entry; got first={first_status}"
    );
}

#[tokio::test]
async fn test_lancedb_active_ranks_above_parked() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
        .await
        .expect("LanceDBCortex::new must succeed");

    let embedding = embed("lancedb status ranking").await;

    let mut active_meta = HashMap::new();
    active_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let mut parked_meta = HashMap::new();
    parked_meta.insert(KEY_STATUS.to_string(), "parked".to_string());

    let active_id = MemoryId::new();
    let mut active_entry = MemoryEntry::builder(active_id, "active lancedb memory".to_string())
        .embedding(embedding.clone())
        .salience(0.6)
        .tier(MemoryTier::Cortex)
        .scope(MemoryScope::Global)
        .build();
    active_entry.metadata = active_meta;

    let parked_id = MemoryId::new();
    let mut parked_entry = MemoryEntry::builder(parked_id, "parked lancedb memory".to_string())
        .embedding(embedding)
        .salience(0.6)
        .tier(MemoryTier::Cortex)
        .scope(MemoryScope::Global)
        .build();
    parked_entry.metadata = parked_meta;

    cortex
        .store(active_entry)
        .await
        .expect("store active must succeed");
    cortex
        .store(parked_entry)
        .await
        .expect("store parked must succeed");

    let results = cortex
        .retrieve_scored(&qvec(), 10, None)
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 2, "both entries must be returned");

    let first_status = results[0]
        .entry
        .metadata
        .get(KEY_STATUS)
        .map(String::as_str)
        .unwrap_or("");
    assert_eq!(
        first_status, "active",
        "active entry must rank above parked in LanceDB; got first={first_status}"
    );
}

// ============================================================================
// 6. Project membership boost: repoA entry ranks above repoB when prefer_project=repoA
// ============================================================================

#[tokio::test]
async fn test_synapse_project_membership_boost() {
    let synapse = SynapseMemory::new();
    let embedding = embed("project boost test").await;

    let mut repo_a_meta = HashMap::new();
    repo_a_meta.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoA".to_string()]),
    );
    repo_a_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let mut repo_b_meta = HashMap::new();
    repo_b_meta.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoB".to_string()]),
    );
    repo_b_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let id_a = MemoryId::new();
    let mut entry_a = MemoryEntry::builder(id_a, "repoA memory".to_string())
        .embedding(embedding.clone())
        .salience(0.5)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    entry_a.metadata = repo_a_meta;

    let id_b = MemoryId::new();
    let mut entry_b = MemoryEntry::builder(id_b, "repoB memory".to_string())
        .embedding(embedding)
        .salience(0.5)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    entry_b.metadata = repo_b_meta;

    synapse
        .store(entry_a)
        .await
        .expect("store repoA must succeed");
    synapse
        .store(entry_b)
        .await
        .expect("store repoB must succeed");

    let results = synapse
        .retrieve_scored(&qvec(), 10, Some("repoA"))
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 2, "both entries must be returned");

    let first_project = results[0]
        .entry
        .metadata
        .get(KEY_PROJECT)
        .cloned()
        .unwrap_or_default();
    assert!(
        first_project.contains("repoA"),
        "repoA entry must rank first when prefer_project=repoA; got project={first_project}"
    );
}

#[tokio::test]
async fn test_lancedb_project_membership_boost() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
        .await
        .expect("LanceDBCortex::new must succeed");

    let embedding = embed("lancedb project boost").await;

    let mut repo_a_meta = HashMap::new();
    repo_a_meta.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoA".to_string()]),
    );
    repo_a_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let mut repo_b_meta = HashMap::new();
    repo_b_meta.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoB".to_string()]),
    );
    repo_b_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let id_a = MemoryId::new();
    let mut entry_a = MemoryEntry::builder(id_a, "repoA lancedb memory".to_string())
        .embedding(embedding.clone())
        .salience(0.5)
        .tier(MemoryTier::Cortex)
        .scope(MemoryScope::Global)
        .build();
    entry_a.metadata = repo_a_meta;

    let id_b = MemoryId::new();
    let mut entry_b = MemoryEntry::builder(id_b, "repoB lancedb memory".to_string())
        .embedding(embedding)
        .salience(0.5)
        .tier(MemoryTier::Cortex)
        .scope(MemoryScope::Global)
        .build();
    entry_b.metadata = repo_b_meta;

    cortex
        .store(entry_a)
        .await
        .expect("store repoA must succeed");
    cortex
        .store(entry_b)
        .await
        .expect("store repoB must succeed");

    let results = cortex
        .retrieve_scored(&qvec(), 10, Some("repoA"))
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 2, "both entries must be returned");

    let first_project = results[0]
        .entry
        .metadata
        .get(KEY_PROJECT)
        .cloned()
        .unwrap_or_default();
    assert!(
        first_project.contains("repoA"),
        "repoA entry must rank first in LanceDB when prefer_project=repoA; got project={first_project}"
    );
}

// ============================================================================
// 7. Untagged entry is not penalised relative to a non-member entry
// ============================================================================

#[tokio::test]
async fn test_synapse_untagged_not_penalised_vs_non_member() {
    let synapse = SynapseMemory::new();
    let embedding = embed("untagged vs non-member").await;

    // Non-member: tagged with repoB (not the preferred repoA)
    let mut non_member_meta = HashMap::new();
    non_member_meta.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoB".to_string()]),
    );
    non_member_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    // Untagged: no project key at all
    let mut untagged_meta = HashMap::new();
    untagged_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let id_non_member = MemoryId::new();
    let mut non_member_entry = MemoryEntry::builder(id_non_member, "non-member memory".to_string())
        .embedding(embedding.clone())
        .salience(0.5)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    non_member_entry.metadata = non_member_meta;

    let id_untagged = MemoryId::new();
    let mut untagged_entry = MemoryEntry::builder(id_untagged, "untagged memory".to_string())
        .embedding(embedding)
        .salience(0.5)
        .tier(MemoryTier::Synapse)
        .scope(MemoryScope::Global)
        .build();
    untagged_entry.metadata = untagged_meta;

    synapse
        .store(non_member_entry)
        .await
        .expect("store non-member must succeed");
    synapse
        .store(untagged_entry)
        .await
        .expect("store untagged must succeed");

    let results = synapse
        .retrieve_scored(&qvec(), 10, Some("repoA"))
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 2, "both entries must be returned");

    // Untagged should rank >= non-member (untagged weight=1.0, non-member weight=0.7)
    let untagged_pos = results
        .iter()
        .position(|r| r.entry.id == id_untagged)
        .expect("untagged entry must be in results");
    let non_member_pos = results
        .iter()
        .position(|r| r.entry.id == id_non_member)
        .expect("non-member entry must be in results");

    assert!(
        untagged_pos <= non_member_pos,
        "untagged entry (neutral weight 1.0) must rank at least as high as non-member entry (weight 0.7); \
         untagged_pos={untagged_pos}, non_member_pos={non_member_pos}"
    );
}

#[tokio::test]
async fn test_lancedb_untagged_not_penalised_vs_non_member() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let cortex = LanceDBCortex::new(dir.path(), "memories", 384)
        .await
        .expect("LanceDBCortex::new must succeed");

    let embedding = embed("lancedb untagged vs non-member").await;

    let mut non_member_meta = HashMap::new();
    non_member_meta.insert(
        KEY_PROJECT.to_string(),
        project_array_json(&["repoB".to_string()]),
    );
    non_member_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let mut untagged_meta = HashMap::new();
    untagged_meta.insert(KEY_STATUS.to_string(), "active".to_string());

    let id_non_member = MemoryId::new();
    let mut non_member_entry =
        MemoryEntry::builder(id_non_member, "lancedb non-member memory".to_string())
            .embedding(embedding.clone())
            .salience(0.5)
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::Global)
            .build();
    non_member_entry.metadata = non_member_meta;

    let id_untagged = MemoryId::new();
    let mut untagged_entry =
        MemoryEntry::builder(id_untagged, "lancedb untagged memory".to_string())
            .embedding(embedding)
            .salience(0.5)
            .tier(MemoryTier::Cortex)
            .scope(MemoryScope::Global)
            .build();
    untagged_entry.metadata = untagged_meta;

    cortex
        .store(non_member_entry)
        .await
        .expect("store non-member must succeed");
    cortex
        .store(untagged_entry)
        .await
        .expect("store untagged must succeed");

    let results = cortex
        .retrieve_scored(&qvec(), 10, Some("repoA"))
        .await
        .expect("retrieve_scored must succeed");

    assert_eq!(results.len(), 2, "both entries must be returned");

    let untagged_pos = results
        .iter()
        .position(|r| r.entry.id == id_untagged)
        .expect("untagged entry must be in results");
    let non_member_pos = results
        .iter()
        .position(|r| r.entry.id == id_non_member)
        .expect("non-member entry must be in results");

    assert!(
        untagged_pos <= non_member_pos,
        "untagged entry must rank at least as high as non-member in LanceDB; \
         untagged_pos={untagged_pos}, non_member_pos={non_member_pos}"
    );
}
