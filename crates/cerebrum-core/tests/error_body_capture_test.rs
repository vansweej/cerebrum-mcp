//! Integration test verifying that Ollama error response bodies are captured
//! and surfaced in the resulting `CerebrumError::Embedding` message.

use cerebrum_core::fastembed_embedder::FastEmbedEmbedder;
use cerebrum_core::traits::Embedder;
use cerebrum_core::CerebrumError;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

/// Verifies that a 400 response with a JSON error body results in an
/// `Embedding` error whose message contains both the status code and the
/// body text.
#[tokio::test]
async fn test_embed_error_body_is_captured() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(400).set_body_string(r#"model "x" not found"#))
        .mount(&mock_server)
        .await;

    let embedder = FastEmbedEmbedder::with_config(mock_server.uri(), "test-model".to_string(), 768);

    let result = embedder.embed("hello world").await;

    match result {
        Err(CerebrumError::Embedding(msg)) => {
            assert!(
                msg.contains("400"),
                "error message must contain status code 400: {msg}"
            );
            assert!(
                msg.contains(r#"model "x" not found"#),
                "error message must contain the response body: {msg}"
            );
        }
        other => panic!("expected CerebrumError::Embedding, got {other:?}"),
    }
}
