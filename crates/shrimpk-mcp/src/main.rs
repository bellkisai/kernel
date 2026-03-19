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
            self.engine = Some(Arc::new(engine));
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

                    // Tool call — needs engine (lazy init)
                    "tools/call" => {
                        let tool_name = req.params["name"].as_str().unwrap_or("");
                        let args = req.params["arguments"].clone();

                        // Clone engine Arc and config ref to avoid borrow conflict
                        let engine = match state.engine().await {
                            Ok(e) => e.clone(),
                            Err(e) => {
                                error_response(req.id, -32603, &format!("Engine init failed: {e}"));
                                continue;
                            }
                        };
                        let config = &state.config;

                        let result = handler::dispatch(&engine, config, tool_name, &args).await;
                        let result_json = serde_json::to_value(&result).unwrap();
                        success_response(req.id, result_json)
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
