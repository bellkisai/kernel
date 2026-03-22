//! OpenAI-compatible proxy with transparent memory injection.
//!
//! Sits between any client and any OpenAI-compatible backend (Ollama, etc.).
//! Injects echo memories into the system prompt before forwarding.
//! Stores user messages for future recall.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Json, Response};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

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
    if let Some(first) = messages.first_mut() {
        if first.role == "system" {
            first.content = format!("{}{}", first.content, memory_block);
            return;
        }
    }
    messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: memory_block.to_string(),
        },
    );
}

// ---------------------------------------------------------------------------
// Main handler — POST /v1/chat/completions
// ---------------------------------------------------------------------------

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
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

        // 3. Inject memories into system prompt
        let memory_block = build_memory_block(&echo_results);
        inject_memories(&mut req.messages, &memory_block);

        // 4. Store user message for future recall (fire-and-forget)
        let engine = state.engine.clone();
        let msg = user_msg.clone();
        tokio::spawn(async move {
            let _ = engine.store(&msg, "proxy").await;
        });
    }

    // 5. Build backend URL
    let backend_url = format!(
        "{}/v1/chat/completions",
        state.config.proxy_target.trim_end_matches('/')
    );

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
        .map(|result| result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));

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
    let url = format!(
        "{}/v1/models",
        state.config.proxy_target.trim_end_matches('/')
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
}
