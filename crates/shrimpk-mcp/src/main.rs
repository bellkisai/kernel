//! ShrimPK MCP Server — expose Echo Memory as tools for Claude Code.
//!
//! Protocol: JSON-RPC 2.0 over stdio (line-delimited).
//! stdout is the protocol channel. All logs go to stderr.

mod format;
mod handler;
mod protocol;
mod tools;

use protocol::{error_response, initialize_result, parse_request, success_response};
use shrimpk_core::{EchoConfig, config};
use shrimpk_memory::EchoEngine;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Lazily initialized engine state.
struct ServerState {
    config: EchoConfig,
    engine: Option<Arc<EchoEngine>>,
}

impl ServerState {
    fn new(config: EchoConfig) -> Self {
        Self {
            config,
            engine: None,
        }
    }

    /// Get or initialize the EchoEngine (lazy — first call loads fastembed model).
    async fn engine(&mut self) -> Result<&Arc<EchoEngine>, String> {
        if self.engine.is_none() {
            eprintln!("[shrimpk-mcp] Loading Echo Memory engine...");
            std::fs::create_dir_all(&self.config.data_dir).map_err(|e| e.to_string())?;
            let engine = EchoEngine::load(self.config.clone()).map_err(|e| e.to_string())?;
            let total = engine.stats().await.total_memories;
            eprintln!("[shrimpk-mcp] Loaded {total} memories.");
            let engine = Arc::new(engine);
            // Start background consolidation (every 5 minutes)
            engine.start_consolidation(300);
            self.engine = Some(engine);
        }
        Ok(self.engine.as_ref().unwrap())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    eprintln!("[shrimpk-mcp] Starting v{}...", env!("CARGO_PKG_VERSION"));

    let config = config::resolve_config().map_err(|e| {
        eprintln!("[shrimpk-mcp] Config error: {e}");
        e
    })?;

    eprintln!(
        "[shrimpk-mcp] Config: {} tier, data_dir={}",
        format::tier_name(&config),
        config.data_dir.display()
    );

    let mut state = ServerState::new(config);

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let response = match parse_request(&line) {
            Ok(req) => {
                eprintln!("[shrimpk-mcp] <- {}", req.method);

                match req.method.as_str() {
                    // Initialize — return immediately (no engine needed)
                    "initialize" => success_response(req.id, initialize_result()),

                    // Notification — no response needed
                    "notifications/initialized" => continue,

                    // List tools — no engine needed
                    "tools/list" => {
                        let tool_defs = tools::all_tools();
                        let tools_json: Vec<serde_json::Value> = tool_defs
                            .into_iter()
                            .map(|t| serde_json::to_value(t).unwrap())
                            .collect();
                        success_response(req.id, serde_json::json!({ "tools": tools_json }))
                    }

                    // Tool call — try daemon HTTP first, fallback to in-process
                    "tools/call" => {
                        let tool_name = req.params["name"].as_str().unwrap_or("");
                        let args = req.params["arguments"].clone();

                        // Try daemon proxy (instant, no model load)
                        if let Some(result) = proxy_to_daemon(tool_name, &args).await {
                            let result_json = serde_json::to_value(&result).unwrap();
                            success_response(req.id, result_json)
                        } else {
                            // Fallback: in-process engine (lazy init)
                            let engine = match state.engine().await {
                                Ok(e) => e.clone(),
                                Err(e) => {
                                    let resp = error_response(
                                        req.id,
                                        -32603,
                                        &format!("Engine init failed: {e}"),
                                    );
                                    let json = serde_json::to_string(&resp).unwrap();
                                    let _ = stdout.write_all(json.as_bytes()).await;
                                    let _ = stdout.write_all(b"\n").await;
                                    let _ = stdout.flush().await;
                                    continue;
                                }
                            };
                            let config = &state.config;

                            let result = handler::dispatch(&engine, config, tool_name, &args).await;
                            let result_json = serde_json::to_value(&result).unwrap();
                            success_response(req.id, result_json)
                        }
                    }

                    // Unknown method
                    _ => error_response(req.id, -32601, &format!("Unknown method: {}", req.method)),
                }
            }
            Err(e) => error_response(None, -32700, &e),
        };

        let json = serde_json::to_string(&response)?;
        stdout.write_all(json.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    eprintln!("[shrimpk-mcp] Stdin closed, shutting down.");
    Ok(())
}

/// Map MCP tool names to daemon HTTP routes and proxy the call.
/// Returns None if daemon is not running.
async fn proxy_to_daemon(
    tool_name: &str,
    args: &serde_json::Value,
) -> Option<protocol::ToolCallResult> {
    let port: u16 = std::env::var("SHRIMPK_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11435);
    let base = format!("http://127.0.0.1:{port}");
    let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));
    // Forward auth token if set (F-02 fix)
    if let Ok(token) = std::env::var("SHRIMPK_AUTH_TOKEN") {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")) {
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
        builder = builder.default_headers(headers);
    }
    let client = builder.build().ok()?;

    // Check health first (fast fail if daemon not running)
    client
        .get(format!("{base}/health"))
        .timeout(std::time::Duration::from_millis(200))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?;

    eprintln!("[shrimpk-mcp] Proxying {tool_name} to daemon at {base}");

    let resp = match tool_name {
        "store" => {
            client
                .post(format!("{base}/api/store"))
                .json(args)
                .send()
                .await
        }
        "echo" => {
            client
                .post(format!("{base}/api/echo"))
                .json(args)
                .send()
                .await
        }
        "stats" => client.get(format!("{base}/api/stats")).send().await,
        "forget" => {
            let id = args["id"].as_str().unwrap_or("");
            client
                .delete(format!("{base}/api/memories/{id}"))
                .send()
                .await
        }
        "dump" => {
            client
                .get(format!("{base}/api/memories?limit=50"))
                .send()
                .await
        }
        "config_show" => client.get(format!("{base}/api/config")).send().await,
        "config_set" => {
            client
                .put(format!("{base}/api/config"))
                .json(args)
                .send()
                .await
        }
        "persist" => client.post(format!("{base}/api/persist")).send().await,
        "status" => client.get(format!("{base}/health")).send().await,
        "memory_graph" => {
            client
                .post(format!("{base}/api/memory_graph"))
                .json(args)
                .send()
                .await
        }
        "memory_related" => {
            client
                .post(format!("{base}/api/memory_related"))
                .json(args)
                .send()
                .await
        }
        _ => {
            return Some(protocol::ToolCallResult::error(format!(
                "Unknown tool: {tool_name}"
            )));
        }
    };

    match resp {
        Ok(r) => {
            let text = r
                .text()
                .await
                .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
            Some(protocol::ToolCallResult::success(text))
        }
        Err(e) => {
            eprintln!("[shrimpk-mcp] Daemon proxy error: {e}");
            None // Fall back to in-process
        }
    }
}
