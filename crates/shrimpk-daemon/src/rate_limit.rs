//! Simple fixed-window rate limiter middleware for the daemon API.
//!
//! Prevents localhost DoS by limiting requests per second.
//! Default: 100 req/s, configurable via `daemon_rate_limit` in config.toml.

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Shared rate limiter state.
#[derive(Clone)]
pub struct RateLimiter {
    inner: Arc<RateLimiterInner>,
}

struct RateLimiterInner {
    /// Maximum requests allowed per window.
    max_requests: u64,
    /// Window duration in seconds.
    window_secs: u64,
    /// Request count in the current window.
    count: AtomicU64,
    /// Start of the current window (epoch seconds from a fixed instant).
    window_start: AtomicU64,
    /// Reference instant for computing elapsed time.
    epoch: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// - `max_requests_per_second`: how many requests allowed per 1-second window.
    pub fn new(max_requests_per_second: u64) -> Self {
        Self {
            inner: Arc::new(RateLimiterInner {
                max_requests: max_requests_per_second,
                window_secs: 1,
                count: AtomicU64::new(0),
                window_start: AtomicU64::new(0),
                epoch: Instant::now(),
            }),
        }
    }

    /// Check if a request is allowed. Returns `true` if within the limit.
    pub fn check(&self) -> bool {
        let now_secs = self.inner.epoch.elapsed().as_secs();
        let window = self.inner.window_start.load(Ordering::Relaxed);

        // If we've moved to a new window, reset the counter.
        if now_secs >= window + self.inner.window_secs {
            // Try to claim the window reset. If another thread already did it,
            // that's fine — we'll just increment the new window's counter.
            let _ = self.inner.window_start.compare_exchange(
                window,
                now_secs,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            self.inner.count.store(1, Ordering::Relaxed);
            return true;
        }

        // Same window — increment and check.
        let prev = self.inner.count.fetch_add(1, Ordering::Relaxed);
        prev < self.inner.max_requests
    }
}

/// Axum middleware that enforces rate limiting.
/// Returns HTTP 429 Too Many Requests when the limit is exceeded.
pub async fn rate_limit_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    // Extract the rate limiter from request extensions.
    let limiter = req.extensions().get::<RateLimiter>().cloned();

    if let Some(limiter) = limiter {
        if !limiter.check() {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
    }

    Ok(next.run(req).await)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_requests_within_limit() {
        let limiter = RateLimiter::new(10);
        for _ in 0..10 {
            assert!(limiter.check(), "should allow requests within limit");
        }
    }

    #[test]
    fn rejects_requests_over_limit() {
        let limiter = RateLimiter::new(5);
        for _ in 0..5 {
            assert!(limiter.check());
        }
        // 6th request should be rejected
        assert!(!limiter.check(), "should reject request over limit");
    }

    #[test]
    fn limit_of_one() {
        let limiter = RateLimiter::new(1);
        assert!(limiter.check());
        assert!(!limiter.check());
    }

    #[test]
    fn clone_shares_state() {
        let limiter = RateLimiter::new(3);
        let clone = limiter.clone();
        assert!(limiter.check());
        assert!(clone.check());
        assert!(limiter.check());
        // 4th total request should be rejected on either handle
        assert!(!clone.check());
    }
}
