//! ShrimPK Daemon — localhost HTTP server for Echo Memory.
//!
//! Like Ollama: loads the model once, runs forever, serves all clients.
//!
//! Usage:
//!   shrimpk-daemon                     # serve on localhost:11435
//!   shrimpk-daemon --port 8080         # custom port

mod routes;
mod state;

use axum::Router;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{delete, get, post, put};
use shrimpk_core::config;
use shrimpk_memory::EchoEngine;
use state::AppState;
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};

/// Default port (one above Ollama's 11434).
const DEFAULT_PORT: u16 = 11435;

/// Auth middleware: check Bearer token if SHRIMPK_AUTH_TOKEN is set.
async fn auth_middleware(
    state: axum::extract::State<AppState>,
    req: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(expected) = &state.auth_token {
        let auth_header = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth_header.strip_prefix("Bearer ").unwrap_or("");
        if token != expected {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    Ok(next.run(req).await)
}

/// Write PID file for daemon discovery.
fn write_pid_file(port: u16) -> std::io::Result<()> {
    let pid_path = config::config_dir().join("daemon.pid");
    std::fs::create_dir_all(config::config_dir())?;
    let content = format!(
        "port={}\npid={}\nstarted={}\n",
        port,
        std::process::id(),
        chrono::Utc::now().to_rfc3339()
    );
    std::fs::write(pid_path, content)
}

/// Remove PID file on shutdown.
fn remove_pid_file() {
    let pid_path = config::config_dir().join("daemon.pid");
    let _ = std::fs::remove_file(pid_path);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse port from args
    let args: Vec<String> = std::env::args().collect();
    let port = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    eprintln!(
        "[shrimpk] Starting daemon v{}...",
        env!("CARGO_PKG_VERSION")
    );

    // Resolve config
    let echo_config = config::resolve_config().map_err(|e| {
        eprintln!("[shrimpk] Config error: {e}");
        e
    })?;
    std::fs::create_dir_all(&echo_config.data_dir)?;

    // Load engine (model + memories — the ONE time this happens)
    eprintln!("[shrimpk] Loading Echo Memory engine...");
    let engine = Arc::new(EchoEngine::load(echo_config.clone())?);
    let total = engine.stats().await.total_memories;
    eprintln!("[shrimpk] Loaded {total} memories.");

    // Start background consolidation
    engine.start_consolidation(300);
    eprintln!("[shrimpk] Consolidation started (every 5 min).");

    // Auth token (optional)
    let auth_token = std::env::var("SHRIMPK_AUTH_TOKEN").ok();
    if auth_token.is_some() {
        eprintln!("[shrimpk] Auth token required (SHRIMPK_AUTH_TOKEN set).");
    }

    // Build app state
    let state = AppState {
        engine,
        config: echo_config,
        started_at: Instant::now(),
        auth_token,
    };

    // Build router
    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/api/store", post(routes::store))
        .route("/api/echo", post(routes::echo))
        .route("/api/stats", get(routes::stats))
        .route("/api/memories", get(routes::list_memories))
        .route("/api/memories/{id}", delete(routes::forget))
        .route("/api/config", get(routes::config_show))
        .route("/api/config", put(routes::config_set))
        .route("/api/persist", post(routes::persist))
        .route("/api/consolidate", post(routes::consolidate))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    // Write PID file
    write_pid_file(port)?;

    let addr = format!("127.0.0.1:{port}");
    eprintln!("[shrimpk] Serving on http://{addr}");
    eprintln!("[shrimpk] Press Ctrl+C to stop.");

    // Serve with graceful shutdown
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Cleanup on shutdown
    eprintln!("\n[shrimpk] Shutting down...");
    remove_pid_file();
    eprintln!("[shrimpk] Goodbye.");

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
}
