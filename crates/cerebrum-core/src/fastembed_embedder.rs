use crate::config::{DEFAULT_EMBED_CONNECT_TIMEOUT, DEFAULT_EMBED_TIMEOUT};
use crate::error::{CerebrumError, Result};
use crate::observability::OperationMetrics;
use crate::resilience::{CircuitBreaker, CircuitBreakerConfig};
use crate::traits::Embedder;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Request body for the Ollama batch embedding API (`POST /api/embed`).
///
/// The Ollama `/api/embed` endpoint accepts a batch of texts and returns
/// a batch of embeddings. This struct represents a single request.
///
/// # Example
/// ```json
/// {
///   "model": "nomic-embed-text",
///   "input": ["search_document: user preferences", "search_document: system settings"]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

/// Response body from the Ollama batch embedding API.
///
/// The Ollama `/api/embed` endpoint returns a batch of embeddings.
/// Each embedding is a 768-dimensional vector for nomic-embed-text.
///
/// # Example
/// ```json
/// {
///   "embeddings": [
///     [0.1, 0.2, ..., 0.3],  // 768 dimensions
///     [0.4, 0.5, ..., 0.6]   // 768 dimensions
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// Ollama-based embedder using the nomic-embed-text model (768-dimensional).
///
/// Provides real semantic embeddings for accurate similarity search via the
/// Ollama batch `POST /api/embed` endpoint
/// (`{model, input:[...]}` -> `{embeddings:[[...]]}`).
/// Requires Ollama to be running at the configured endpoint.
///
/// # Batch API
/// This embedder uses the batch API (`/api/embed`) which accepts multiple texts
/// in a single request. Currently, we embed one text at a time for simplicity,
/// but the infrastructure supports batch processing for future optimization.
///
/// # Prefix Handling
/// This embedder is single-text and model-agnostic: it applies NO query or
/// document prefixes. The orchestrator is responsible for prefixing before
/// calling `embed()`. This separation of concerns keeps the embedder simple
/// and the prefix logic centralized in the orchestrator.
///
/// # Observability & Resilience
/// Includes observability metrics and resilience patterns:
/// - Tracks latency, success rate, and error counts via `OperationMetrics`
/// - Circuit breaker for handling Ollama endpoint failures
/// - Automatic failure detection and recovery
///
/// # Dimension Validation
/// The embedder validates that returned embeddings match the expected dimension.
/// For nomic-embed-text, this is always 768. If Ollama returns a different
/// dimension, the request fails with a validation error.
pub struct FastEmbedEmbedder {
    /// Ollama endpoint URL (e.g., "http://localhost:11434").
    endpoint: String,
    /// Model name (default: "nomic-embed-text").
    model: String,
    /// Expected embedding dimension produced by `model` (768 for nomic-embed-text).
    dim: usize,
    /// Per-instance HTTP client (carries connect/request timeouts).
    /// Configured with `embed_timeout` and `embed_connect_timeout` from Config.
    client: reqwest::Client,
    /// Metrics for tracking operation performance (latency, success rate, error counts).
    metrics: Arc<OperationMetrics>,
    /// Circuit breaker for handling transient failures.
    /// Opens after 5 consecutive failures, transitions to HalfOpen after 60 seconds.
    circuit_breaker: Arc<CircuitBreaker>,
}

impl FastEmbedEmbedder {
    /// Create a new embedder pointing at the local Ollama instance using
    /// `nomic-embed-text` (768-dimensional) and the crate default timeouts.
    pub fn new() -> Self {
        Self::with_timeouts(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            768,
            DEFAULT_EMBED_TIMEOUT,
            DEFAULT_EMBED_CONNECT_TIMEOUT,
        )
    }

    /// Create a new embedder with a custom endpoint, defaulting to
    /// `nomic-embed-text` (768-dimensional).
    pub fn with_endpoint(endpoint: String) -> Self {
        Self::with_config(endpoint, "nomic-embed-text".to_string(), 768)
    }

    /// Create a new embedder with a custom endpoint, model, and dimension,
    /// using the crate default timeouts.
    pub fn with_config(endpoint: String, model: String, dim: usize) -> Self {
        Self::with_timeouts(
            endpoint,
            model,
            dim,
            DEFAULT_EMBED_TIMEOUT,
            DEFAULT_EMBED_CONNECT_TIMEOUT,
        )
    }

    /// Create a new embedder with explicit endpoint, model, dimension, and
    /// request/connect timeouts.
    pub fn with_timeouts(
        endpoint: String,
        model: String,
        dim: usize,
        timeout: Duration,
        connect_timeout: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            endpoint,
            model,
            dim,
            client,
            metrics: Arc::new(OperationMetrics::new()),
            circuit_breaker: Arc::new(CircuitBreaker::new(CircuitBreakerConfig::new())),
        }
    }

    /// Get the embedding dimension for this model.
    pub fn embedding_dim(&self) -> usize {
        self.dim
    }

    /// Get the metrics for this embedder.
    pub fn metrics(&self) -> Arc<OperationMetrics> {
        self.metrics.clone()
    }

    /// Get the circuit breaker for this embedder.
    pub fn circuit_breaker(&self) -> Arc<CircuitBreaker> {
        self.circuit_breaker.clone()
    }

    /// Check if Ollama is available at the configured endpoint.
    pub async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.endpoint);
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.client.get(&url).send(),
        )
        .await
        {
            Ok(Ok(response)) => response.status().is_success(),
            _ => false,
        }
    }
}

impl Default for FastEmbedEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for FastEmbedEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let start_time = Instant::now();

        // Check circuit breaker before making request
        self.circuit_breaker.allow_request()?;

        let url = format!("{}/api/embed", self.endpoint);

        let request = OllamaEmbedRequest {
            model: self.model.clone(),
            input: vec![text.to_string()],
        };

        let result = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                CerebrumError::Embedding(format!(
                    "Failed to connect to Ollama at {}: {}",
                    self.endpoint, e
                ))
            })
            .and_then(|response| {
                if !response.status().is_success() {
                    return Err(CerebrumError::Embedding(format!(
                        "Ollama API error: {}",
                        response.status()
                    )));
                }
                Ok(response)
            });

        let response = match result {
            Ok(resp) => resp,
            Err(e) => {
                // Record failure and update circuit breaker
                let duration_ms = start_time.elapsed().as_millis() as u64;
                self.metrics.record_failure(duration_ms);
                self.circuit_breaker.record_failure();
                return Err(e);
            }
        };

        let embed_response: OllamaEmbedResponse = response.json().await.map_err(|e| {
            let duration_ms = start_time.elapsed().as_millis() as u64;
            self.metrics.record_failure(duration_ms);
            self.circuit_breaker.record_failure();
            CerebrumError::Embedding(format!("Failed to parse Ollama response: {}", e))
        })?;

        // Take the first embedding from the batch response.
        let embedding = match embed_response.embeddings.into_iter().next() {
            Some(e) => e,
            None => {
                let duration_ms = start_time.elapsed().as_millis() as u64;
                self.metrics.record_failure(duration_ms);
                self.circuit_breaker.record_failure();
                return Err(CerebrumError::Embedding(
                    "Ollama returned no embeddings".to_string(),
                ));
            }
        };

        // Verify dimensions
        if embedding.len() != self.dim {
            let duration_ms = start_time.elapsed().as_millis() as u64;
            self.metrics.record_failure(duration_ms);
            self.circuit_breaker.record_failure();
            return Err(CerebrumError::Validation(format!(
                "Invalid embedding dimension from Ollama: expected {}, got {}",
                self.dim,
                embedding.len()
            )));
        }

        // Record success
        let duration_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.record_success(duration_ms);
        self.circuit_breaker.record_success();

        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_fastembed_embedder_new() {
        let embedder = FastEmbedEmbedder::new();
        assert_eq!(embedder.endpoint, "http://localhost:11434");
        assert_eq!(embedder.model, "nomic-embed-text");
    }

    #[tokio::test]
    async fn test_fastembed_embedder_default() {
        let embedder = FastEmbedEmbedder::default();
        assert_eq!(embedder.endpoint, "http://localhost:11434");
        assert_eq!(embedder.model, "nomic-embed-text");
    }

    #[tokio::test]
    async fn test_fastembed_embedder_with_endpoint() {
        let embedder = FastEmbedEmbedder::with_endpoint("http://custom:11434".to_string());
        assert_eq!(embedder.endpoint, "http://custom:11434");
        assert_eq!(embedder.model, "nomic-embed-text");
    }

    #[tokio::test]
    async fn test_fastembed_embedder_with_config() {
        let embedder = FastEmbedEmbedder::with_config(
            "http://custom:11434".to_string(),
            "custom-model".to_string(),
            768,
        );
        assert_eq!(embedder.endpoint, "http://custom:11434");
        assert_eq!(embedder.model, "custom-model");
        assert_eq!(embedder.dimension(), 768);
    }

    #[tokio::test]
    async fn test_fastembed_embedder_embedding_dim() {
        let embedder = FastEmbedEmbedder::new();
        assert_eq!(embedder.embedding_dim(), 768);
    }

    #[tokio::test]
    async fn test_fastembed_embedder_embed_requires_ollama() {
        let embedder = FastEmbedEmbedder::new();
        let result = embedder.embed("test text").await;

        // This test will fail if Ollama is not running
        // In CI/CD, this should be skipped or Ollama should be available
        match result {
            Ok(embedding) => {
                // Ollama is available
                assert_eq!(embedding.len(), 768);
            }
            Err(CerebrumError::Embedding(msg)) => {
                // Ollama is not available - this is expected in some environments
                assert!(msg.contains("Failed to connect") || msg.contains("Ollama"));
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_fastembed_embedder_consistency_requires_ollama() {
        let embedder = FastEmbedEmbedder::new();

        // Skip if Ollama is not available
        if !embedder.is_available().await {
            return;
        }

        let embedding1 = embedder.embed("hello world").await.unwrap();
        let embedding2 = embedder.embed("hello world").await.unwrap();

        // Same text should produce same embedding
        assert_eq!(embedding1, embedding2);
    }

    #[tokio::test]
    #[ignore]
    async fn test_fastembed_embedder_different_texts_requires_ollama() {
        let embedder = FastEmbedEmbedder::new();

        // Skip if Ollama is not available
        if !embedder.is_available().await {
            return;
        }

        let embedding1 = embedder.embed("hello world").await.unwrap();
        let embedding2 = embedder.embed("goodbye world").await.unwrap();

        // Different texts should produce different embeddings
        assert_ne!(embedding1, embedding2);
    }

    #[tokio::test]
    #[ignore]
    async fn test_fastembed_embedder_empty_text_requires_ollama() {
        let embedder = FastEmbedEmbedder::new();

        // Skip if Ollama is not available
        if !embedder.is_available().await {
            return;
        }

        let embedding = embedder.embed("").await;

        // Empty text should still produce an embedding
        assert!(embedding.is_ok());
        let vec = embedding.unwrap();
        assert_eq!(vec.len(), 768);
    }

    #[tokio::test]
    #[ignore]
    async fn test_fastembed_embedder_concurrent_access_requires_ollama() {
        let embedder = Arc::new(FastEmbedEmbedder::new());

        // Skip if Ollama is not available
        if !embedder.is_available().await {
            return;
        }

        // Create multiple concurrent embedding tasks
        let mut handles = vec![];
        for i in 0..3 {
            let embedder_clone = Arc::clone(&embedder);
            let handle = tokio::spawn(async move {
                let text = format!("text {}", i);
                embedder_clone.embed(&text).await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await;
            assert!(result.is_ok());
            let embedding_result = result.unwrap();
            assert!(embedding_result.is_ok());
            let vec = embedding_result.unwrap();
            assert_eq!(vec.len(), 768);
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_fastembed_embedder_normalized_requires_ollama() {
        let embedder = FastEmbedEmbedder::new();

        // Skip if Ollama is not available
        if !embedder.is_available().await {
            return;
        }

        let embedding = embedder.embed("test").await.unwrap();

        // Embedding should be normalized (magnitude close to 1)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01);
    }

    // ============================================================================
    // Phase 3: Observability & Resilience Tests
    // ============================================================================

    #[tokio::test]
    async fn test_fastembed_embedder_metrics_initialization() {
        let embedder = FastEmbedEmbedder::new();
        let metrics = embedder.metrics();

        // Verify metrics are initialized to zero
        assert_eq!(metrics.total_operations(), 0);
        assert_eq!(metrics.successful_operations(), 0);
        assert_eq!(metrics.failed_operations(), 0);
        assert_eq!(metrics.total_time_ms(), 0);
        assert_eq!(metrics.success_rate(), 100.0); // 0/0 = 100%
    }

    #[tokio::test]
    async fn test_fastembed_embedder_circuit_breaker_initialization() {
        let embedder = FastEmbedEmbedder::new();
        let cb = embedder.circuit_breaker();

        // Verify circuit breaker starts in Closed state
        use crate::resilience::CircuitState;
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_fastembed_embedder_circuit_breaker_allow_request_when_closed() {
        let embedder = FastEmbedEmbedder::new();
        let cb = embedder.circuit_breaker();

        // Circuit breaker should allow requests when Closed
        assert!(cb.allow_request().is_ok());
    }

    #[tokio::test]
    async fn test_fastembed_embedder_circuit_breaker_records_success() {
        let embedder = FastEmbedEmbedder::new();
        let cb = embedder.circuit_breaker();

        // Record a success
        cb.record_success();

        // Circuit breaker should still be Closed
        use crate::resilience::CircuitState;
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_fastembed_embedder_circuit_breaker_opens_on_failures() {
        let embedder = FastEmbedEmbedder::new();
        let cb = embedder.circuit_breaker();

        // Record multiple failures to trigger Open state
        for _ in 0..5 {
            cb.record_failure();
        }

        // Circuit breaker should be Open
        use crate::resilience::CircuitState;
        assert_eq!(cb.state(), CircuitState::Open);

        // Circuit breaker should deny requests when Open
        assert!(cb.allow_request().is_err());
    }

    #[tokio::test]
    async fn test_fastembed_embedder_metrics_track_operations() {
        let embedder = FastEmbedEmbedder::new();
        let metrics = embedder.metrics();

        // Record some operations
        metrics.record_success(100);
        metrics.record_success(200);
        metrics.record_failure(150);

        // Verify metrics are updated
        assert_eq!(metrics.total_operations(), 3);
        assert_eq!(metrics.successful_operations(), 2);
        assert_eq!(metrics.failed_operations(), 1);
        assert_eq!(metrics.total_time_ms(), 450);

        // Verify success rate
        let success_rate = metrics.success_rate();
        assert!((success_rate - 66.66).abs() < 0.1); // 2/3 ≈ 66.66%

        // Verify average time
        let avg_time = metrics.average_time_ms();
        assert!((avg_time - 150.0).abs() < 0.1); // 450/3 = 150
    }

    // ============================================================================
    // Phase 3: Wiremock HTTP Tests
    // ============================================================================

    #[tokio::test]
    async fn test_is_available_true_against_mock() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let embedder = FastEmbedEmbedder::with_endpoint(mock_server.uri());
        assert_eq!(embedder.is_available().await, true);
    }

    #[tokio::test]
    async fn test_is_available_false_when_unreachable() {
        let embedder = FastEmbedEmbedder::with_endpoint("http://127.0.0.1:1".to_string());
        assert_eq!(embedder.is_available().await, false);
    }

    #[tokio::test]
    async fn test_embed_success_against_mock() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };

        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vec![vec![0.1f32; 768]] })),
            )
            .mount(&mock_server)
            .await;

        let embedder = FastEmbedEmbedder::with_endpoint(mock_server.uri());
        let result = embedder.embed("hello").await;
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 768);
        assert_eq!(embedder.metrics().successful_operations(), 1);
    }

    #[tokio::test]
    async fn test_embed_http_error_records_failure() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };

        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&mock_server)
            .await;

        let embedder = FastEmbedEmbedder::with_endpoint(mock_server.uri());
        let result = embedder.embed("hello").await;
        assert!(result.is_err());
        match result {
            Err(CerebrumError::Embedding(_)) => {}
            _ => panic!("Expected Embedding error"),
        }
        assert_eq!(embedder.metrics().failed_operations(), 1);
    }

    #[tokio::test]
    async fn test_embed_parse_error_records_failure() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };

        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&mock_server)
            .await;

        let embedder = FastEmbedEmbedder::with_endpoint(mock_server.uri());
        let result = embedder.embed("hello").await;
        assert!(result.is_err());
        match result {
            Err(CerebrumError::Embedding(_)) => {}
            _ => panic!("Expected Embedding error"),
        }
        assert_eq!(embedder.metrics().failed_operations(), 1);
    }

    #[tokio::test]
    async fn test_embed_dimension_mismatch_records_failure() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };

        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": [[0.1f32, 0.2, 0.3]] })),
            )
            .mount(&mock_server)
            .await;

        let embedder = FastEmbedEmbedder::with_endpoint(mock_server.uri());
        let result = embedder.embed("hello").await;
        assert!(result.is_err());
        match result {
            Err(CerebrumError::Validation(_)) => {}
            _ => panic!("Expected Validation error"),
        }
        assert_eq!(embedder.metrics().failed_operations(), 1);
    }
}
