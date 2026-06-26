use crate::error::{CerebrumError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Configuration for retry logic with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial backoff duration in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds.
    pub max_backoff_ms: u64,
    /// Backoff multiplier for exponential growth.
    pub backoff_multiplier: f64,
    /// Whether to add jitter to backoff times.
    pub use_jitter: bool,
}

impl RetryConfig {
    /// Create a new RetryConfig with default values.
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 10000,
            backoff_multiplier: 2.0,
            use_jitter: true,
        }
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the initial backoff duration.
    pub fn with_initial_backoff_ms(mut self, ms: u64) -> Self {
        self.initial_backoff_ms = ms;
        self
    }

    /// Set the maximum backoff duration.
    pub fn with_max_backoff_ms(mut self, ms: u64) -> Self {
        self.max_backoff_ms = ms;
        self
    }

    /// Set the backoff multiplier.
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set whether to use jitter.
    pub fn with_jitter(mut self, use_jitter: bool) -> Self {
        self.use_jitter = use_jitter;
        self
    }

    /// Calculate the backoff duration for a given attempt number.
    pub fn calculate_backoff(&self, attempt: u32) -> Duration {
        let backoff_ms =
            (self.initial_backoff_ms as f64 * self.backoff_multiplier.powi(attempt as i32)) as u64;
        let backoff_ms = backoff_ms.min(self.max_backoff_ms);

        if self.use_jitter {
            // Add jitter: random value between 0 and backoff_ms
            let jitter = (backoff_ms as f64 * 0.1) as u64; // 10% jitter
            Duration::from_millis(backoff_ms + jitter)
        } else {
            Duration::from_millis(backoff_ms)
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker for handling transient failures.
///
/// The circuit breaker has three states:
/// - Closed: Normal operation, requests pass through
/// - Open: Too many failures, requests fail immediately
/// - Half-Open: Testing if the service has recovered
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation, requests pass through.
    Closed,
    /// Too many failures, requests fail immediately.
    Open,
    /// Testing if the service has recovered.
    HalfOpen,
}

/// Configuration for circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to trigger open state.
    pub failure_threshold: u64,
    /// Duration to wait before transitioning from Open to HalfOpen.
    pub timeout_ms: u64,
}

impl CircuitBreakerConfig {
    /// Create a new CircuitBreakerConfig with default values.
    pub fn new() -> Self {
        Self {
            failure_threshold: 5,
            timeout_ms: 60000, // 1 minute
        }
    }

    /// Set the failure threshold.
    pub fn with_failure_threshold(mut self, threshold: u64) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set the timeout duration.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker for handling transient failures.
pub struct CircuitBreaker {
    state: Arc<std::sync::Mutex<CircuitState>>,
    failure_count: Arc<AtomicU64>,
    last_failure_time: Arc<std::sync::Mutex<Option<Instant>>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(std::sync::Mutex::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicU64::new(0)),
            last_failure_time: Arc::new(std::sync::Mutex::new(None)),
            config,
        }
    }

    /// Get the current state of the circuit breaker.
    pub fn state(&self) -> CircuitState {
        *self.state.lock().unwrap()
    }

    /// Record a successful operation.
    pub fn record_success(&self) {
        let mut state = self.state.lock().unwrap();
        self.failure_count.store(0, Ordering::Relaxed);
        *state = CircuitState::Closed;
        debug!("Circuit breaker: success, state=Closed");
    }

    /// Record a failed operation.
    pub fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure_time.lock().unwrap() = Some(Instant::now());

        if failures >= self.config.failure_threshold {
            let mut state = self.state.lock().unwrap();
            *state = CircuitState::Open;
            warn!(
                "Circuit breaker: failure threshold reached ({}), state=Open",
                failures
            );
        }
    }

    /// Check if a request should be allowed.
    pub fn allow_request(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();

        match *state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open => {
                // Check if timeout has elapsed
                if let Some(last_failure) = *self.last_failure_time.lock().unwrap() {
                    if last_failure.elapsed() > Duration::from_millis(self.config.timeout_ms) {
                        *state = CircuitState::HalfOpen;
                        debug!("Circuit breaker: timeout elapsed, state=HalfOpen");
                        Ok(())
                    } else {
                        Err(CerebrumError::Unavailable(
                            "Circuit breaker is open".to_string(),
                        ))
                    }
                } else {
                    Err(CerebrumError::Unavailable(
                        "Circuit breaker is open".to_string(),
                    ))
                }
            }
            CircuitState::HalfOpen => Ok(()),
        }
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        *self.last_failure_time.lock().unwrap() = None;
        debug!("Circuit breaker: reset, state=Closed");
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            failure_count: Arc::clone(&self.failure_count),
            last_failure_time: Arc::clone(&self.last_failure_time),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_new() {
        let config = RetryConfig::new();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 100);
        assert_eq!(config.max_backoff_ms, 10000);
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(config.use_jitter);
    }

    #[test]
    fn test_retry_config_with_max_retries() {
        let config = RetryConfig::new().with_max_retries(5);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_retry_config_with_initial_backoff() {
        let config = RetryConfig::new().with_initial_backoff_ms(200);
        assert_eq!(config.initial_backoff_ms, 200);
    }

    #[test]
    fn test_retry_config_calculate_backoff() {
        let config = RetryConfig::new()
            .with_initial_backoff_ms(100)
            .with_jitter(false);

        let backoff0 = config.calculate_backoff(0);
        let backoff1 = config.calculate_backoff(1);
        let backoff2 = config.calculate_backoff(2);

        assert_eq!(backoff0.as_millis(), 100);
        assert_eq!(backoff1.as_millis(), 200);
        assert_eq!(backoff2.as_millis(), 400);
    }

    #[test]
    fn test_retry_config_calculate_backoff_max() {
        let config = RetryConfig::new()
            .with_initial_backoff_ms(100)
            .with_max_backoff_ms(500)
            .with_jitter(false);

        let backoff5 = config.calculate_backoff(5);
        assert!(backoff5.as_millis() <= 500);
    }

    #[test]
    fn test_circuit_breaker_config_new() {
        let config = CircuitBreakerConfig::new();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.timeout_ms, 60000);
    }

    #[test]
    fn test_circuit_breaker_new() {
        let config = CircuitBreakerConfig::new();
        let breaker = CircuitBreaker::new(config);
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_allow_request_closed() {
        let config = CircuitBreakerConfig::new();
        let breaker = CircuitBreaker::new(config);
        assert!(breaker.allow_request().is_ok());
    }

    #[test]
    fn test_circuit_breaker_record_success() {
        let config = CircuitBreakerConfig::new();
        let breaker = CircuitBreaker::new(config);
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_record_failure() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(3);
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_open_denies_requests() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(1);
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(breaker.allow_request().is_err());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let config = CircuitBreakerConfig::new().with_failure_threshold(1);
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        breaker.reset();
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.allow_request().is_ok());
    }

    #[test]
    fn test_circuit_breaker_clone() {
        let config = CircuitBreakerConfig::new();
        let breaker = CircuitBreaker::new(config);
        let cloned = breaker.clone();

        breaker.record_failure();
        // Both should share the same state
        assert_eq!(cloned.state(), breaker.state());
    }

    #[test]
    fn test_retry_config_default() {
        let default_config = RetryConfig::default();
        let new_config = RetryConfig::new();
        assert_eq!(
            default_config.calculate_backoff(0).as_millis(),
            new_config.calculate_backoff(0).as_millis()
        );
    }

    #[test]
    fn test_circuit_breaker_config_default() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.timeout_ms, 60000);
    }

    #[test]
    fn test_retry_config_with_backoff_multiplier() {
        let config = RetryConfig::new()
            .with_initial_backoff_ms(100)
            .with_backoff_multiplier(3.0)
            .with_jitter(false);
        let backoff0 = config.calculate_backoff(0);
        let backoff1 = config.calculate_backoff(1);
        assert!(backoff1.as_millis() > backoff0.as_millis());
    }
}
