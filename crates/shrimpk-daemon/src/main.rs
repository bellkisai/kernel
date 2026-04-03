//! ShrimPK Daemon — localhost HTTP server for Echo Memory.
//!
//! Like Ollama: loads the model once, runs forever, serves all clients.
//!
//! Usage:
//!   shrimpk-daemon                     # serve on localhost:11435
//!   shrimpk-daemon --port 8080         # custom port

mod detect;
mod proxy;
mod rate_limit;
mod routes;
mod state;

use axum::Router;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{delete, get, post, put};
use shrimpk_context::{ContextAssembler, ContextConfig};
use shrimpk_core::config;
use shrimpk_memory::EchoEngine;
use state::AppState;
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};

/// Default port (one above Ollama's 11434).
const DEFAULT_PORT: u16 = 11435;

/// Resolve a `--proxy-to` shorthand (e.g. "ollama") to a full base URL.
fn resolve_proxy_target(raw: &str) -> String {
    match raw {
        "ollama" => "http://127.0.0.1:11434".into(),
        "lmstudio" | "lm-studio" => "http://127.0.0.1:1234".into(),
        "vllm" => "http://127.0.0.1:8000".into(),
        "jan" => "http://127.0.0.1:1337".into(),
        "localai" => "http://127.0.0.1:8080".into(),
        "gpt4all" => "http://127.0.0.1:4891".into(),
        other => other.into(),
    }
}

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
/// Check if a stale PID file exists and clean it up (F-03 fix).
fn validate_pid_file() {
    let pid_path = config::config_dir().join("daemon.pid");
    if !pid_path.exists() {
        return;
    }
    if let Ok(content) = std::fs::read_to_string(&pid_path) {
        // Parse PID from file
        let pid: Option<u32> = content
            .lines()
            .find(|l| l.starts_with("pid="))
            .and_then(|l| l.strip_prefix("pid="))
            .and_then(|s| s.parse().ok());

        if let Some(pid) = pid {
            // Check if process is still alive
            let alive = sysinfo::System::new_all()
                .process(sysinfo::Pid::from_u32(pid))
                .is_some();
            if !alive {
                eprintln!("[shrimpk] Removing stale PID file (pid {pid} not running).");
                let _ = std::fs::remove_file(&pid_path);
            } else {
                eprintln!("[shrimpk] WARNING: Daemon already running (pid {pid}). Exiting.");
                std::process::exit(1);
            }
        }
    }
}

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
    let args: Vec<String> = std::env::args().collect();

    // Handle --install / --uninstall before starting the server
    if args.iter().any(|a| a == "--install") {
        return install_autostart();
    }
    if args.iter().any(|a| a == "--uninstall") {
        return uninstall_autostart();
    }

    let port = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let proxy_to = args
        .iter()
        .position(|a| a == "--proxy-to")
        .and_then(|i| args.get(i + 1))
        .cloned();

    eprintln!(
        "[shrimpk] Starting daemon v{}...",
        env!("CARGO_PKG_VERSION")
    );

    // Check for stale PID file or already-running daemon (F-03)
    validate_pid_file();

    // Resolve config
    let mut echo_config = config::resolve_config().map_err(|e| {
        eprintln!("[shrimpk] Config error: {e}");
        e
    })?;
    std::fs::create_dir_all(&echo_config.data_dir)?;

    // Override proxy target if --proxy-to was given
    if let Some(ref target) = proxy_to {
        echo_config.proxy_target = resolve_proxy_target(target);
        tracing::info!("Proxy target override: {}", echo_config.proxy_target);
    }

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
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .pool_max_idle_per_host(4)
        .build()
        .expect("Failed to build HTTP client");

    // Auto-detect local LLM providers and build model routing table
    let providers = detect::detect_providers(&http_client).await;
    let model_routes = detect::build_model_routes(&providers);
    let total_models: usize = providers.iter().map(|p| p.models.len()).sum();
    for p in &providers {
        eprintln!(
            "[shrimpk] Detected: {} ({} models) at {}",
            p.name,
            p.models.len(),
            p.url
        );
    }
    if providers.is_empty() {
        eprintln!("[shrimpk] No local LLM providers detected.");
    } else {
        eprintln!(
            "[shrimpk] Total: {} provider(s), {} model(s) routable.",
            providers.len(),
            total_models
        );
    }

    // Build context assembler for token-budgeted proxy injection
    let context_config = ContextConfig {
        max_echo_results: echo_config.proxy_max_echo_results,
        max_conversation_turns: echo_config.proxy_max_conversation_turns,
        ..ContextConfig::default()
    };
    let context_assembler = Arc::new(ContextAssembler::new(context_config));

    let state = AppState {
        engine,
        config: echo_config,
        started_at: Instant::now(),
        auth_token,
        pii_filter: Arc::new(shrimpk_memory::PiiFilter::new()),
        http_client,
        model_routes: Arc::new(tokio::sync::RwLock::new(model_routes)),
        context_assembler,
    };

    // Keep engine ref for shutdown persist (must clone before state moves into router)
    let engine_for_shutdown = state.engine.clone();
    let proxy_target_display = state.config.proxy_target.clone();

    // Rate limiter (default 100 req/s, configurable via daemon_rate_limit)
    let rate_limit = state.config.daemon_rate_limit;
    let limiter = rate_limit::RateLimiter::new(rate_limit);
    eprintln!("[shrimpk] Rate limit: {rate_limit} req/s.");

    // Build router
    #[allow(unused_mut)]
    let mut app = Router::new()
        .route("/health", get(routes::health))
        .route("/debug", get(routes::debug))
        .route("/api/store", post(routes::store))
        .route("/api/echo", post(routes::echo))
        .route("/api/stats", get(routes::stats))
        .route("/api/memories", get(routes::list_memories))
        .route("/api/memories/{id}", delete(routes::forget))
        .route("/api/config", get(routes::config_show))
        .route("/api/config", put(routes::config_set))
        .route("/api/persist", post(routes::persist))
        .route("/api/consolidate", post(routes::consolidate))
        .route("/api/detect", get(routes::detect_providers))
        .route("/api/memory_graph", post(routes::memory_graph))
        .route("/api/memory_related", post(routes::memory_related))
        .route("/api/memory_get", post(routes::memory_get));

    // Multimodal routes — conditionally compiled
    #[cfg(feature = "vision")]
    {
        app = app.route("/api/store_image", post(routes::store_image));
    }
    #[cfg(feature = "speech")]
    {
        app = app.route("/api/store_audio", post(routes::store_audio));
    }

    let app = app
        // OpenAI-compatible proxy routes
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .route("/v1/models", get(proxy::models))
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(axum::Extension(limiter))
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
    eprintln!("[shrimpk] Proxy -> {proxy_target_display}");
    eprintln!("[shrimpk] Press Ctrl+C to stop.");

    // Serve with graceful shutdown
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Cleanup on shutdown — persist memories before exit (F-01 fix)
    eprintln!("\n[shrimpk] Shutting down — persisting memories...");
    if let Err(e) = engine_for_shutdown.persist().await {
        eprintln!("[shrimpk] WARNING: Failed to persist on shutdown: {e}");
    }
    remove_pid_file();
    eprintln!("[shrimpk] Goodbye.");

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
}

// ---------------------------------------------------------------------------
// Auto-start installation (Windows: Startup folder, others: launchd/systemd)
// ---------------------------------------------------------------------------

/// Get the path to the current executable.
fn daemon_exe_path() -> anyhow::Result<std::path::PathBuf> {
    std::env::current_exe().map_err(|e| anyhow::anyhow!("Cannot find daemon executable: {e}"))
}

/// Install auto-start: creates a shortcut/script in the platform's startup directory.
fn install_autostart() -> anyhow::Result<()> {
    let exe = daemon_exe_path()?;

    #[cfg(target_os = "windows")]
    {
        // Windows: create a .vbs script in the Startup folder (runs hidden, no console window)
        let startup_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?
            .join("AppData")
            .join("Roaming")
            .join("Microsoft")
            .join("Windows")
            .join("Start Menu")
            .join("Programs")
            .join("Startup");

        std::fs::create_dir_all(&startup_dir)?;

        let vbs_path = startup_dir.join("shrimpk-daemon.vbs");
        let vbs_content = format!(
            "Set WshShell = CreateObject(\"WScript.Shell\")\r\n\
             WshShell.Run \"\"\"{}\"\"\", 0, False\r\n",
            exe.display()
        );
        std::fs::write(&vbs_path, vbs_content)?;
        println!("[shrimpk] Installed auto-start: {}", vbs_path.display());
        println!("[shrimpk] Daemon will start automatically on login (hidden, no console).");
    }

    #[cfg(target_os = "macos")]
    {
        let plist_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?
            .join("Library")
            .join("LaunchAgents");
        std::fs::create_dir_all(&plist_dir)?;

        let plist_path = plist_dir.join("ai.bellkis.shrimpk-daemon.plist");
        let plist_content = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.bellkis.shrimpk-daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>{}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>"#,
            exe.display()
        );
        std::fs::write(&plist_path, plist_content)?;
        println!("[shrimpk] Installed: {}", plist_path.display());
        println!("[shrimpk] Run: launchctl load {}", plist_path.display());
    }

    #[cfg(target_os = "linux")]
    {
        let systemd_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?
            .join(".config")
            .join("systemd")
            .join("user");
        std::fs::create_dir_all(&systemd_dir)?;

        let service_path = systemd_dir.join("shrimpk-daemon.service");
        let service_content = format!(
            "[Unit]\n\
             Description=ShrimPK Echo Memory Daemon\n\
             After=network.target\n\n\
             [Service]\n\
             ExecStart={}\n\
             Restart=on-failure\n\n\
             [Install]\n\
             WantedBy=default.target\n",
            exe.display()
        );
        std::fs::write(&service_path, service_content)?;
        println!("[shrimpk] Installed: {}", service_path.display());
        println!("[shrimpk] Run: systemctl --user enable --now shrimpk-daemon");
    }

    Ok(())
}

/// Remove auto-start configuration.
fn uninstall_autostart() -> anyhow::Result<()> {
    // Stop running daemon if detected
    let pid_path = config::config_dir().join("daemon.pid");
    if pid_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&pid_path) {
            let pid: Option<u32> = content
                .lines()
                .find(|l| l.starts_with("pid="))
                .and_then(|l| l.strip_prefix("pid="))
                .and_then(|s| s.parse().ok());
            if let Some(pid) = pid {
                println!("[shrimpk] Stopping running daemon (pid {pid})...");
                #[cfg(target_os = "windows")]
                {
                    let _ = std::process::Command::new("taskkill")
                        .args(["/F", "/PID", &pid.to_string()])
                        .output();
                }
                #[cfg(not(target_os = "windows"))]
                {
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .output();
                }
            }
        }
        let _ = std::fs::remove_file(&pid_path);
    }
    #[cfg(target_os = "windows")]
    {
        let vbs_path = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?
            .join(
                "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/shrimpk-daemon.vbs",
            );
        if vbs_path.exists() {
            std::fs::remove_file(&vbs_path)?;
            println!("[shrimpk] Removed auto-start: {}", vbs_path.display());
        } else {
            println!("[shrimpk] No auto-start found.");
        }
    }

    #[cfg(target_os = "macos")]
    {
        let plist_path = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?
            .join("Library/LaunchAgents/ai.bellkis.shrimpk-daemon.plist");
        if plist_path.exists() {
            std::fs::remove_file(&plist_path)?;
            println!("[shrimpk] Removed: {}", plist_path.display());
        } else {
            println!("[shrimpk] No auto-start found.");
        }
    }

    #[cfg(target_os = "linux")]
    {
        let service_path = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?
            .join(".config/systemd/user/shrimpk-daemon.service");
        if service_path.exists() {
            std::fs::remove_file(&service_path)?;
            println!("[shrimpk] Removed: {}", service_path.display());
        } else {
            println!("[shrimpk] No auto-start found.");
        }
    }

    Ok(())
}
