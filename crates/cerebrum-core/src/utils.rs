use crate::error::Result;
use crate::models::MemoryId;

/// Generate a new random MemoryId.
pub fn generate_memory_id() -> MemoryId {
    MemoryId::new()
}

/// Validate that an embedding has the correct dimension (384 for BGE-small).
pub fn validate_embedding_dimension(embedding: &[f32]) -> Result<()> {
    const EXPECTED_DIM: usize = 384;
    if embedding.len() != EXPECTED_DIM {
        return Err(crate::error::CerebrumError::Embedding(format!(
            "Invalid embedding dimension: expected {}, got {}",
            EXPECTED_DIM,
            embedding.len()
        )));
    }
    Ok(())
}

/// Get the default salience score for new memories.
pub fn default_salience() -> f32 {
    0.5
}

/// Get the current UTC timestamp.
pub fn current_timestamp() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_memory_id() {
        let id1 = generate_memory_id();
        let id2 = generate_memory_id();

        // IDs should be unique
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_validate_embedding_dimension_valid() {
        let embedding = vec![0.1; 384];
        assert!(validate_embedding_dimension(&embedding).is_ok());
    }

    #[test]
    fn test_validate_embedding_dimension_invalid() {
        let embedding = vec![0.1; 256];
        assert!(validate_embedding_dimension(&embedding).is_err());
    }

    #[test]
    fn test_default_salience() {
        assert_eq!(default_salience(), 0.5);
    }

    #[test]
    fn test_current_timestamp() {
        let before = chrono::Utc::now();
        let timestamp = current_timestamp();
        let after = chrono::Utc::now();

        assert!(timestamp >= before);
        assert!(timestamp <= after);
    }
}
