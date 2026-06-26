use crate::error::{CerebrumError, Result};
use async_trait::async_trait;

// Re-export the Embedder trait from traits module
pub use crate::traits::Embedder;

/// Mock embedder for development and testing.
///
/// This is a placeholder implementation that generates deterministic embeddings
/// based on text hashing. In production, this should be replaced with a real
/// embedding model like fastembed (BGE-small, 384-dimensional).
pub struct MockEmbedder;

impl MockEmbedder {
    /// Create a new MockEmbedder.
    pub fn new() -> Self {
        Self
    }

    /// Get the embedding dimension (384 for BGE-small compatibility).
    pub fn dimension() -> usize {
        384
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(CerebrumError::Embedding(
                "Cannot embed empty text".to_string(),
            ));
        }

        // Generate a deterministic embedding based on text hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Create a 384-dimensional vector from the hash
        let mut embedding = Vec::with_capacity(Self::dimension());
        let mut seed = hash;

        for _ in 0..Self::dimension() {
            // Linear congruential generator for pseudo-random values
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((seed / 65536) % 32768) as f32 / 32768.0;
            embedding.push(value);
        }

        // Normalize the embedding to unit length
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        384
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedder_dimension() {
        assert_eq!(MockEmbedder::dimension(), 384);
    }

    #[tokio::test]
    async fn test_mock_embed_simple_text() {
        let embedder = MockEmbedder::new();
        let embedding = embedder
            .embed("Hello, world!")
            .await
            .expect("Failed to embed text");

        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_mock_embed_empty_text() {
        let embedder = MockEmbedder::new();
        let result = embedder.embed("").await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_embed_consistency() {
        let embedder = MockEmbedder::new();
        let text = "The quick brown fox jumps over the lazy dog";

        let embedding1 = embedder.embed(text).await.expect("Failed to embed text");
        let embedding2 = embedder.embed(text).await.expect("Failed to embed text");

        // Embeddings should be identical for the same text
        assert_eq!(embedding1, embedding2);
    }

    #[tokio::test]
    async fn test_mock_embed_different_texts() {
        let embedder = MockEmbedder::new();

        let embedding1 = embedder
            .embed("Hello, world!")
            .await
            .expect("Failed to embed");
        let embedding2 = embedder
            .embed("Goodbye, world!")
            .await
            .expect("Failed to embed");

        // Different texts should produce different embeddings
        assert_ne!(embedding1, embedding2);
    }

    #[tokio::test]
    async fn test_mock_embed_normalized() {
        let embedder = MockEmbedder::new();
        let embedding = embedder.embed("Test text").await.expect("Failed to embed");

        // Check that embedding is approximately unit length
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "Embedding should be normalized");
    }
}
