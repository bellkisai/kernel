//! Shared application state for the daemon.

use shrimpk_core::EchoConfig;
use shrimpk_memory::{EchoEngine, PiiFilter};
use std::sync::Arc;
use std::time::Instant;

/// Shared state passed to all route handlers via Axum's State extractor.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<EchoEngine>,
    pub config: EchoConfig,
    pub started_at: Instant,
    pub auth_token: Option<String>,
    pub pii_filter: Arc<PiiFilter>,
    /// Shared async HTTP client for proxy forwarding (connection pooling).
    pub http_client: reqwest::Client,
}
