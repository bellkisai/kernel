//! # shrimpk-router
//!
//! Intelligent AI model routing with cascade fallback.
//! Routes queries to the optimal model based on cost, quality, latency, and capabilities.
//!
//! ## Features
//! - **Cascade routing** — filters providers by capabilities, locality, budget, then ranks
//!   by cost/quality score to pick the best match.
//! - **Cost tracking** — per-provider token usage tracking with daily/monthly budget enforcement.
//! - **Circuit breaker** — health monitoring that trips open after consecutive failures and
//!   recovers via a half-open probe window.

pub mod cascade;
pub mod config;
pub mod cost;
pub mod health;

// Re-export the public API at crate root.
pub use cascade::CascadeRouter;
pub use config::{ModelConfig, ProviderConfig, RouteDecision, RouteRequest};
pub use cost::CostTracker;
pub use health::CircuitBreaker;

// Re-export identity types from core for convenience.
pub use shrimpk_core::traits::{ModelId, ProviderId};
