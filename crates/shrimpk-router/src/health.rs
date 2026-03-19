//! Circuit breaker for provider health monitoring.
//!
//! Tracks consecutive failures per provider and trips open when a threshold
//! is exceeded, preventing cascading failures. After a recovery timeout the
//! breaker moves to half-open, allowing a single probe request.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The three states of a circuit breaker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation — requests are allowed.
    Closed,
    /// Breaker tripped — requests are blocked until recovery timeout.
    Open,
    /// Recovery window — one request is allowed to probe health.
    HalfOpen,
}

/// A circuit breaker that tracks provider failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    failure_threshold: u32,
    last_failure: Option<DateTime<Utc>>,
    recovery_timeout_secs: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// - `threshold`: number of consecutive failures before the breaker opens.
    /// - `timeout_secs`: seconds to wait before transitioning from Open to HalfOpen.
    pub fn new(threshold: u32, timeout_secs: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold: threshold,
            last_failure: None,
            recovery_timeout_secs: timeout_secs,
        }
    }

    /// Record a successful request. Resets the breaker to Closed.
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.state = CircuitState::Closed;
        self.last_failure = None;
    }

    /// Record a failed request. Increments failure count and may trip the breaker.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Some(Utc::now());

        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
        }
    }

    /// Check whether this provider should be considered available.
    ///
    /// - `Closed` => available
    /// - `HalfOpen` => available (probe request)
    /// - `Open` => check if recovery timeout has elapsed; if so, transition to HalfOpen
    pub fn is_available(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if enough time has passed to attempt recovery.
                if let Some(last) = self.last_failure {
                    let elapsed = Utc::now()
                        .signed_duration_since(last)
                        .num_seconds();
                    if elapsed >= self.recovery_timeout_secs as i64 {
                        self.state = CircuitState::HalfOpen;
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Return a reference to the current circuit state.
    pub fn state(&self) -> &CircuitState {
        &self.state
    }

    /// Return the current failure count.
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_breaker_is_closed() {
        let cb = CircuitBreaker::new(5, 60);
        assert_eq!(*cb.state(), CircuitState::Closed);
        assert!(cb.failure_count() == 0);
    }

    #[test]
    fn opens_after_threshold_failures() {
        let mut cb = CircuitBreaker::new(3, 60);
        cb.record_failure();
        assert_eq!(*cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(*cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(*cb.state(), CircuitState::Open);
        assert!(!cb.is_available());
    }

    #[test]
    fn success_resets_breaker() {
        let mut cb = CircuitBreaker::new(3, 60);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);
        cb.record_success();
        assert_eq!(*cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn half_open_after_timeout() {
        let mut cb = CircuitBreaker::new(2, 0); // 0-second timeout for testing
        cb.record_failure();
        cb.record_failure();
        assert_eq!(*cb.state(), CircuitState::Open);
        // With 0-second timeout, should immediately transition to HalfOpen.
        assert!(cb.is_available());
        assert_eq!(*cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn half_open_success_closes() {
        let mut cb = CircuitBreaker::new(2, 0);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(*cb.state(), CircuitState::Open);
        // Transition to HalfOpen.
        cb.is_available();
        assert_eq!(*cb.state(), CircuitState::HalfOpen);
        // Success in HalfOpen resets to Closed.
        cb.record_success();
        assert_eq!(*cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_failure_reopens() {
        let mut cb = CircuitBreaker::new(2, 0);
        cb.record_failure();
        cb.record_failure();
        // Transition to HalfOpen.
        cb.is_available();
        assert_eq!(*cb.state(), CircuitState::HalfOpen);
        // Failure in HalfOpen (count still >= threshold) reopens.
        cb.record_failure();
        assert_eq!(*cb.state(), CircuitState::Open);
    }

    #[test]
    fn default_breaker() {
        let cb = CircuitBreaker::default();
        assert_eq!(*cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }
}
