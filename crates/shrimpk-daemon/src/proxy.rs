//! OpenAI-compatible proxy with transparent memory injection.
//!
//! Sits between any client and any OpenAI-compatible backend (Ollama, etc.).
//! Uses the `shrimpk-context` ContextAssembler for token-budgeted memory
//! injection. Falls back to naive string injection if assembly fails.
//! Stores user messages for future recall.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Json, Response};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use shrimpk_context::ContextSegment;

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    /// Pass through any extra vendor-specific fields (e.g. num_ctx, top_p).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the last user message in the conversation (the one we echo against).
fn extract_last_user_message(messages: &[ChatMessage]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
}

/// Format echo results into a block suitable for injection into the system prompt.
fn build_memory_block(results: &[shrimpk_core::EchoResult]) -> String {
    if results.is_empty() {
        return String::new();
    }
    let mut block =
        String::from("\n\n[Echo Memory] Relevant context from previous conversations:\n");
    for (i, r) in results.iter().enumerate() {
        block.push_str(&format!("{}. {}\n", i + 1, r.content));
    }
    block.push_str("Use these memories naturally if relevant.\n");
    block
}

/// Inject a memory block into the message list. Appends to the existing system
/// message if one exists, otherwise inserts a new system message at position 0.
fn inject_memories(messages: &mut Vec<ChatMessage>, memory_block: &str) {
    if memory_block.is_empty() {
        return;
    }
    if let Some(first) = messages.first_mut()
        && first.role == "system"
    {
        first.content = format!("{}{}", first.content, memory_block);
        return;
    }
    messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: memory_block.to_string(),
        },
    );
}

/// Convert ContextAssembler segments back into ChatMessages for the backend.
///
/// Maps segment types to OpenAI-compatible roles:
/// - `SystemPrompt` + `EchoMemory` + `RagDocument` → merged into one `system` message
/// - `ConversationMessage` → preserved role + content
/// - `UserQuery` → `user` role
fn segments_to_messages(segments: &[ContextSegment]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // Merge system prompt + echo memories + RAG into a single system message
    let mut system_parts: Vec<String> = Vec::new();
    for seg in segments {
        match seg {
            ContextSegment::SystemPrompt(text) => {
                if !text.is_empty() {
                    system_parts.push(text.clone());
                }
            }
            ContextSegment::EchoMemory {
                content,
                similarity,
                ..
            } => {
                system_parts.push(format!(
                    "[Echo Memory (relevance: {:.0}%)] {}",
                    similarity * 100.0,
                    content,
                ));
            }
            ContextSegment::RagDocument { content, source } => {
                system_parts.push(format!("[Document: {}] {}", source, content));
            }
            _ => {}
        }
    }
    if !system_parts.is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system_parts.join("\n"),
        });
    }

    // Add conversation messages and user query in order
    for seg in segments {
        match seg {
            ContextSegment::ConversationMessage { role, content } => {
                messages.push(ChatMessage {
                    role: role.clone(),
                    content: content.clone(),
                });
            }
            ContextSegment::UserQuery(text) => {
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: text.clone(),
                });
            }
            _ => {} // system/echo/rag already handled above
        }
    }

    messages
}

// ---------------------------------------------------------------------------
// Main handler — POST /v1/chat/completions
// ---------------------------------------------------------------------------

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    // Reject oversized payloads (1MB max)
    let body_size = serde_json::to_vec(&req).map(|v| v.len()).unwrap_or(0);
    if body_size > 1_048_576 {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(
                json!({"error": {"message": "Request body too large (max 1MB)", "type": "proxy_error"}}),
            ),
        ));
    }

    let is_streaming = req.stream.unwrap_or(false);

    // 1. Extract last user message
    let user_msg = extract_last_user_message(&req.messages)
        .unwrap_or("")
        .to_string();

    // 2. Echo — find relevant memories
    if !user_msg.is_empty() && state.config.proxy_enabled {
        let echo_results = state
            .engine
            .echo(&user_msg, state.config.proxy_max_echo_results)
            .await
            .unwrap_or_default();

        // 3. Assemble token-budgeted context (fallback to naive injection on failure)
        if !echo_results.is_empty() {
            // Extract system prompt and conversation from the original messages
            let system_prompt: Option<String> = req
                .messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone());

            let conversation: Vec<(String, String)> = req
                .messages
                .iter()
                .filter(|m| m.role != "system")
                // Skip the last user message — the assembler receives it separately
                .rev()
                .skip(1)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();

            let assembled = state.context_assembler.assemble(
                state.config.proxy_context_window,
                system_prompt.as_deref(),
                &echo_results,
                &[], // no RAG chunks in proxy
                &conversation,
                &user_msg,
            );

            // Convert assembled segments back to ChatMessages
            let new_messages = segments_to_messages(&assembled.segments);
            if !new_messages.is_empty() {
                req.messages = new_messages;
            } else {
                // Fallback: assembler produced empty output — use naive injection
                let memory_block = build_memory_block(&echo_results);
                inject_memories(&mut req.messages, &memory_block);
            }

            tracing::info!("Memories injected: {}", echo_results.len());
        }

        // 4. Store user message for future recall (fire-and-forget)
        let engine = state.engine.clone();
        let msg = user_msg.clone();
        tokio::spawn(async move {
            let _ = engine.store(&msg, "proxy").await;
        });
    }

    // 5. Build backend URL — route by model name if known, else default
    let backend_base = {
        let routes = state.model_routes.read().await;
        routes
            .get(&req.model)
            .cloned()
            .unwrap_or_else(|| state.config.proxy_target.clone())
    };
    let backend_url = format!("{}/v1/chat/completions", backend_base.trim_end_matches('/'));

    // 6. Forward to backend
    if is_streaming {
        forward_streaming(&state, &backend_url, &req).await
    } else {
        forward_non_streaming(&state, &backend_url, &req).await
    }
}

// ---------------------------------------------------------------------------
// Non-streaming forward
// ---------------------------------------------------------------------------

async fn forward_non_streaming(
    state: &AppState,
    url: &str,
    req: &ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let data_bytes = serde_json::to_vec(req).map(|v| v.len()).unwrap_or(0);
    tracing::info!(
        target: "shrimpk::audit",
        endpoint = %url,
        data_bytes = data_bytes,
        direction = "outbound",
        component = "proxy",
        "External data transmission"
    );

    let resp = state
        .http_client
        .post(url)
        .json(req)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("Backend unreachable: {e}"), "type": "proxy_error"}})),
            )
        })?;

    let status =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = resp.bytes().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": {"message": format!("Backend read error: {e}"), "type": "proxy_error"}})),
        )
    })?;

    let mut response = Response::new(axum::body::Body::from(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("content-type", "application/json".parse().unwrap());
    Ok(response)
}

// ---------------------------------------------------------------------------
// Streaming forward (raw byte passthrough)
// ---------------------------------------------------------------------------

async fn forward_streaming(
    state: &AppState,
    url: &str,
    req: &ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let data_bytes = serde_json::to_vec(req).map(|v| v.len()).unwrap_or(0);
    tracing::info!(
        target: "shrimpk::audit",
        endpoint = %url,
        data_bytes = data_bytes,
        direction = "outbound",
        component = "proxy-stream",
        "External data transmission"
    );

    let resp = state
        .http_client
        .post(url)
        .json(req)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("Backend unreachable: {e}"), "type": "proxy_error"}})),
            )
        })?;

    let status =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Raw byte stream passthrough — no SSE parsing needed
    let byte_stream = resp
        .bytes_stream()
        .map(|result| result.map_err(std::io::Error::other));

    let body = axum::body::Body::from_stream(byte_stream);

    let mut response = Response::new(body);
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("content-type", "text/event-stream".parse().unwrap());
    response
        .headers_mut()
        .insert("cache-control", "no-cache".parse().unwrap());
    Ok(response)
}

// ---------------------------------------------------------------------------
// Models passthrough — GET /v1/models
// ---------------------------------------------------------------------------

pub async fn models(
    State(state): State<AppState>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    // Aggregate models from the routing table — each model entry includes its
    // provider URL so clients know where it lives.
    let routes = state.model_routes.read().await;

    if routes.is_empty() {
        // No detected providers — fall back to proxying the default target
        drop(routes);
        let url = format!(
            "{}/v1/models",
            state.config.proxy_target.trim_end_matches('/')
        );

        tracing::info!(
            target: "shrimpk::audit",
            endpoint = %url,
            data_bytes = 0_usize,
            direction = "outbound",
            component = "proxy-models",
            "External data transmission"
        );

        let resp = state
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": {"message": format!("Backend unreachable: {e}"), "type": "proxy_error"}})),
                )
            })?;

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.bytes().await.map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("Backend read error: {e}"), "type": "proxy_error"}})),
            )
        })?;

        let mut response = Response::new(axum::body::Body::from(body));
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert("content-type", "application/json".parse().unwrap());
        return Ok(response);
    }

    // Build an OpenAI-compatible model list from all detected providers
    let model_entries: Vec<serde_json::Value> = routes
        .iter()
        .map(|(model_name, provider_url)| {
            json!({
                "id": model_name,
                "object": "model",
                "owned_by": provider_url
            })
        })
        .collect();
    drop(routes);

    let body = json!({
        "object": "list",
        "data": model_entries
    });

    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    let mut response = Response::new(axum::body::Body::from(body_bytes));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert("content-type", "application/json".parse().unwrap());
    Ok(response)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_last_user_message_finds_last() {
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "first".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "reply".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "second".into(),
            },
        ];
        assert_eq!(extract_last_user_message(&msgs), Some("second"));
    }

    #[test]
    fn extract_last_user_message_empty() {
        let msgs: Vec<ChatMessage> = vec![];
        assert_eq!(extract_last_user_message(&msgs), None);
    }

    #[test]
    fn inject_memories_appends_to_system() {
        let mut msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
        ];
        inject_memories(&mut msgs, "\n[Memory] fact\n");
        assert!(msgs[0].content.contains("You are helpful."));
        assert!(msgs[0].content.contains("[Memory] fact"));
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn inject_memories_creates_system_when_none() {
        let mut msgs = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        inject_memories(&mut msgs, "\n[Memory] fact\n");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
    }

    #[test]
    fn inject_memories_noop_on_empty() {
        let mut msgs = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        inject_memories(&mut msgs, "");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn chat_request_preserves_extra_fields() {
        let json_str =
            r#"{"model":"llama3.2","messages":[],"custom_field":"value","num_ctx":4096}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        assert_eq!(req.extra.get("custom_field").unwrap(), "value");
        // Re-serialize should include extra fields
        let serialized = serde_json::to_string(&req).unwrap();
        assert!(serialized.contains("custom_field"));
    }

    #[test]
    fn chat_request_minimal() {
        let json_str = r#"{"model":"test","messages":[{"role":"user","content":"hello"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.stream, None);
    }

    #[test]
    fn build_memory_block_empty() {
        assert!(build_memory_block(&[]).is_empty());
    }

    #[test]
    fn segments_to_messages_merges_system_and_echo() {
        let segments = vec![
            ContextSegment::SystemPrompt("You are helpful.".into()),
            ContextSegment::EchoMemory {
                content: "user likes Rust".into(),
                similarity: 0.92,
                source: "proxy".into(),
            },
            ContextSegment::ConversationMessage {
                role: "user".into(),
                content: "hello".into(),
            },
            ContextSegment::ConversationMessage {
                role: "assistant".into(),
                content: "hi there".into(),
            },
            ContextSegment::UserQuery("what is Rust?".into()),
        ];

        let msgs = segments_to_messages(&segments);

        // First message should be system with both system prompt and echo memory
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("You are helpful."));
        assert!(msgs[0].content.contains("user likes Rust"));
        assert!(msgs[0].content.contains("92%"));

        // Conversation messages should follow
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "hello");
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[2].content, "hi there");

        // User query should be last
        assert_eq!(msgs[3].role, "user");
        assert_eq!(msgs[3].content, "what is Rust?");
    }

    #[test]
    fn segments_to_messages_no_system_prompt() {
        let segments = vec![
            ContextSegment::SystemPrompt(String::new()),
            ContextSegment::EchoMemory {
                content: "a fact".into(),
                similarity: 0.80,
                source: "test".into(),
            },
            ContextSegment::UserQuery("q".into()),
        ];

        let msgs = segments_to_messages(&segments);

        // System message should exist with echo memory only (empty system prompt filtered)
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("a fact"));
        // User query last
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "q");
    }

    #[test]
    fn segments_to_messages_query_only() {
        let segments = vec![
            ContextSegment::SystemPrompt(String::new()),
            ContextSegment::UserQuery("hello".into()),
        ];

        let msgs = segments_to_messages(&segments);
        // No system parts with content -> only user query
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "hello");
    }

    #[test]
    fn segments_to_messages_with_rag() {
        let segments = vec![
            ContextSegment::SystemPrompt("sys".into()),
            ContextSegment::RagDocument {
                content: "doc content".into(),
                source: "chunk_0".into(),
            },
            ContextSegment::UserQuery("q".into()),
        ];

        let msgs = segments_to_messages(&segments);
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("sys"));
        assert!(msgs[0].content.contains("[Document: chunk_0] doc content"));
    }
}
