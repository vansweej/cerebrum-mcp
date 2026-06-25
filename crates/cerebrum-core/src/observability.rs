use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Metrics for tracking operation performance and counts.
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Total number of operations performed.
    pub total_operations: Arc<AtomicU64>,
    /// Total number of successful operations.
    pub successful_operations: Arc<AtomicU64>,
    /// Total number of failed operations.
    pub failed_operations: Arc<AtomicU64>,
    /// Total time spent in operations (in milliseconds).
    pub total_time_ms: Arc<AtomicU64>,
}

impl OperationMetrics {
    /// Create a new OperationMetrics instance.
    pub fn new() -> Self {
        Self {
            total_operations: Arc::new(AtomicU64::new(0)),
            successful_operations: Arc::new(AtomicU64::new(0)),
            failed_operations: Arc::new(AtomicU64::new(0)),
            total_time_ms: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record a successful operation with the given duration in milliseconds.
    pub fn record_success(&self, duration_ms: u64) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.successful_operations.fetch_add(1, Ordering::Relaxed);
        self.total_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Record a failed operation with the given duration in milliseconds.
    pub fn record_failure(&self, duration_ms: u64) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
        self.total_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Get the total number of operations.
    pub fn total_operations(&self) -> u64 {
        self.total_operations.load(Ordering::Relaxed)
    }

    /// Get the number of successful operations.
    pub fn successful_operations(&self) -> u64 {
        self.successful_operations.load(Ordering::Relaxed)
    }

    /// Get the number of failed operations.
    pub fn failed_operations(&self) -> u64 {
        self.failed_operations.load(Ordering::Relaxed)
    }

    /// Get the total time spent in operations (in milliseconds).
    pub fn total_time_ms(&self) -> u64 {
        self.total_time_ms.load(Ordering::Relaxed)
    }

    /// Get the average time per operation (in milliseconds).
    pub fn average_time_ms(&self) -> f64 {
        let total = self.total_operations();
        if total == 0 {
            0.0
        } else {
            self.total_time_ms() as f64 / total as f64
        }
    }

    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.total_operations();
        if total == 0 {
            100.0
        } else {
            (self.successful_operations() as f64 / total as f64) * 100.0
        }
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.total_operations.store(0, Ordering::Relaxed);
        self.successful_operations.store(0, Ordering::Relaxed);
        self.failed_operations.store(0, Ordering::Relaxed);
        self.total_time_ms.store(0, Ordering::Relaxed);
    }
}

impl Default for OperationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Observability context for tracking operations.
#[derive(Clone)]
pub struct ObservabilityContext {
    /// Metrics for remember operations.
    pub remember_metrics: OperationMetrics,
    /// Metrics for recall operations.
    pub recall_metrics: OperationMetrics,
    /// Metrics for memorize operations.
    pub memorize_metrics: OperationMetrics,
    /// Metrics for forget operations.
    pub forget_metrics: OperationMetrics,
    /// Metrics for promote operations.
    pub promote_metrics: OperationMetrics,
    /// Metrics for decay operations.
    pub decay_metrics: OperationMetrics,
}

impl ObservabilityContext {
    /// Create a new ObservabilityContext.
    pub fn new() -> Self {
        Self {
            remember_metrics: OperationMetrics::new(),
            recall_metrics: OperationMetrics::new(),
            memorize_metrics: OperationMetrics::new(),
            forget_metrics: OperationMetrics::new(),
            promote_metrics: OperationMetrics::new(),
            decay_metrics: OperationMetrics::new(),
        }
    }

    /// Log a summary of all metrics.
    pub fn log_summary(&self) {
        info!(
            "Memory Operations Summary: remember={} recall={} memorize={} forget={} promote={} decay={}",
            self.remember_metrics.total_operations(),
            self.recall_metrics.total_operations(),
            self.memorize_metrics.total_operations(),
            self.forget_metrics.total_operations(),
            self.promote_metrics.total_operations(),
            self.decay_metrics.total_operations()
        );

        debug!(
            "Success Rates: remember={:.1}% recall={:.1}% memorize={:.1}% forget={:.1}% promote={:.1}% decay={:.1}%",
            self.remember_metrics.success_rate(),
            self.recall_metrics.success_rate(),
            self.memorize_metrics.success_rate(),
            self.forget_metrics.success_rate(),
            self.promote_metrics.success_rate(),
            self.decay_metrics.success_rate()
        );

        debug!(
            "Average Latencies (ms): remember={:.2} recall={:.2} memorize={:.2} forget={:.2} promote={:.2} decay={:.2}",
            self.remember_metrics.average_time_ms(),
            self.recall_metrics.average_time_ms(),
            self.memorize_metrics.average_time_ms(),
            self.forget_metrics.average_time_ms(),
            self.promote_metrics.average_time_ms(),
            self.decay_metrics.average_time_ms()
        );
    }

    /// Reset all metrics.
    pub fn reset_all(&self) {
        self.remember_metrics.reset();
        self.recall_metrics.reset();
        self.memorize_metrics.reset();
        self.forget_metrics.reset();
        self.promote_metrics.reset();
        self.decay_metrics.reset();
    }
}

impl Default for ObservabilityContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring operation duration.
pub struct OperationTimer {
    start: Instant,
}

impl OperationTimer {
    /// Create a new timer.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get the elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Record the operation as successful in the given metrics.
    pub fn record_success(self, metrics: &OperationMetrics) {
        let duration_ms = self.elapsed_ms();
        metrics.record_success(duration_ms);
        debug!("Operation completed successfully in {}ms", duration_ms);
    }

    /// Record the operation as failed in the given metrics.
    pub fn record_failure(self, metrics: &OperationMetrics, error: &str) {
        let duration_ms = self.elapsed_ms();
        metrics.record_failure(duration_ms);
        warn!("Operation failed after {}ms: {}", duration_ms, error);
    }
}

impl Default for OperationTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_metrics_new() {
        let metrics = OperationMetrics::new();
        assert_eq!(metrics.total_operations(), 0);
        assert_eq!(metrics.successful_operations(), 0);
        assert_eq!(metrics.failed_operations(), 0);
        assert_eq!(metrics.total_time_ms(), 0);
    }

    #[test]
    fn test_operation_metrics_record_success() {
        let metrics = OperationMetrics::new();
        metrics.record_success(100);
        assert_eq!(metrics.total_operations(), 1);
        assert_eq!(metrics.successful_operations(), 1);
        assert_eq!(metrics.failed_operations(), 0);
        assert_eq!(metrics.total_time_ms(), 100);
    }

    #[test]
    fn test_operation_metrics_record_failure() {
        let metrics = OperationMetrics::new();
        metrics.record_failure(50);
        assert_eq!(metrics.total_operations(), 1);
        assert_eq!(metrics.successful_operations(), 0);
        assert_eq!(metrics.failed_operations(), 1);
        assert_eq!(metrics.total_time_ms(), 50);
    }

    #[test]
    fn test_operation_metrics_success_rate() {
        let metrics = OperationMetrics::new();
        metrics.record_success(100);
        metrics.record_success(100);
        metrics.record_failure(50);
        assert_eq!(metrics.total_operations(), 3);
        assert_eq!(metrics.successful_operations(), 2);
        assert_eq!(metrics.failed_operations(), 1);
        assert!((metrics.success_rate() - 66.666).abs() < 0.1);
    }

    #[test]
    fn test_operation_metrics_average_time() {
        let metrics = OperationMetrics::new();
        metrics.record_success(100);
        metrics.record_success(200);
        metrics.record_success(300);
        assert_eq!(metrics.total_operations(), 3);
        assert_eq!(metrics.total_time_ms(), 600);
        assert!((metrics.average_time_ms() - 200.0).abs() < 0.1);
    }

    #[test]
    fn test_operation_metrics_success_rate_empty() {
        let metrics = OperationMetrics::new();
        assert_eq!(metrics.success_rate(), 100.0);
    }

    #[test]
    fn test_operation_metrics_average_time_empty() {
        let metrics = OperationMetrics::new();
        assert_eq!(metrics.average_time_ms(), 0.0);
    }

    #[test]
    fn test_operation_metrics_reset() {
        let metrics = OperationMetrics::new();
        metrics.record_success(100);
        metrics.record_failure(50);
        assert_eq!(metrics.total_operations(), 2);

        metrics.reset();
        assert_eq!(metrics.total_operations(), 0);
        assert_eq!(metrics.successful_operations(), 0);
        assert_eq!(metrics.failed_operations(), 0);
        assert_eq!(metrics.total_time_ms(), 0);
    }

    #[test]
    fn test_observability_context_new() {
        let ctx = ObservabilityContext::new();
        assert_eq!(ctx.remember_metrics.total_operations(), 0);
        assert_eq!(ctx.recall_metrics.total_operations(), 0);
        assert_eq!(ctx.memorize_metrics.total_operations(), 0);
        assert_eq!(ctx.forget_metrics.total_operations(), 0);
        assert_eq!(ctx.promote_metrics.total_operations(), 0);
        assert_eq!(ctx.decay_metrics.total_operations(), 0);
    }

    #[test]
    fn test_observability_context_reset_all() {
        let ctx = ObservabilityContext::new();
        ctx.remember_metrics.record_success(100);
        ctx.recall_metrics.record_failure(50);
        ctx.memorize_metrics.record_success(75);

        assert!(ctx.remember_metrics.total_operations() > 0);
        assert!(ctx.recall_metrics.total_operations() > 0);
        assert!(ctx.memorize_metrics.total_operations() > 0);

        ctx.reset_all();

        assert_eq!(ctx.remember_metrics.total_operations(), 0);
        assert_eq!(ctx.recall_metrics.total_operations(), 0);
        assert_eq!(ctx.memorize_metrics.total_operations(), 0);
    }

    #[test]
    fn test_operation_timer_new() {
        let timer = OperationTimer::new();
        let elapsed = timer.elapsed_ms();
        assert!(elapsed < 100); // Should be very fast
    }

    #[test]
    fn test_operation_timer_default() {
        let timer = OperationTimer::default();
        let elapsed = timer.elapsed_ms();
        assert!(elapsed < 100);
    }

    #[test]
    fn test_observability_context_log_summary() {
        let ctx = ObservabilityContext::new();
        ctx.remember_metrics.record_success(100);
        ctx.recall_metrics.record_failure(50);
        ctx.memorize_metrics.record_success(75);
        
        // This should not panic
        ctx.log_summary();
    }

    #[test]
    fn test_operation_metrics_clone() {
        let metrics = OperationMetrics::new();
        metrics.record_success(100);
        
        let cloned = metrics.clone();
        assert_eq!(cloned.total_operations(), 1);
        assert_eq!(cloned.successful_operations(), 1);
        assert_eq!(cloned.total_time_ms(), 100);
    }

    #[test]
    fn test_observability_context_clone() {
        let ctx = ObservabilityContext::new();
        ctx.remember_metrics.record_success(100);
        
        let cloned = ctx.clone();
        assert_eq!(cloned.remember_metrics.total_operations(), 1);
    }
}
