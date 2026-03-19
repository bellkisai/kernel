//! JSON-RPC 2.0 protocol types for the MCP server.
//!
//! The MCP protocol uses line-delimited JSON-RPC over stdio.
//! This module handles parsing requests and serializing responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A JSON-RPC 2.0 request.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

/// A single content block in an MCP tool response.
#[derive(Debug, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// The result of an MCP tool call.
#[derive(Debug, Serialize)]
pub struct ToolCallResult {
    pub content: Vec<ContentBlock>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolCallResult {
    pub fn success(text: String) -> Self {
        Self {
            content: vec![ContentBlock {
                content_type: "text".into(),
                text,
            }],
            is_error: None,
        }
    }

    pub fn error(text: String) -> Self {
        Self {
            content: vec![ContentBlock {
                content_type: "text".into(),
                text,
            }],
            is_error: Some(true),
        }
    }
}

/// An MCP tool definition.
#[derive(Debug, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Parse a JSON-RPC request from a line of text.
pub fn parse_request(line: &str) -> Result<JsonRpcRequest, String> {
    serde_json::from_str(line).map_err(|e| format!("Invalid JSON: {e}"))
}

/// Build a successful JSON-RPC response.
pub fn success_response(id: Option<Value>, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }
}

/// Build a JSON-RPC error response.
pub fn error_response(id: Option<Value>, code: i32, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.into(),
        }),
    }
}

/// Build the MCP `initialize` result.
pub fn initialize_result() -> Value {
    serde_json::json!({
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "shrimpk-mcp",
            "version": env!("CARGO_PKG_VERSION")
        },
        "capabilities": {
            "tools": {}
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_request() {
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req = parse_request(line).unwrap();
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(Value::from(1)));
    }

    #[test]
    fn parse_request_with_null_id() {
        let line = r#"{"jsonrpc":"2.0","id":null,"method":"tools/list","params":{}}"#;
        let req = parse_request(line).unwrap();
        assert_eq!(req.method, "tools/list");
        // serde deserializes "id": null as None for Option<Value>
        assert!(req.id.is_none());
    }

    #[test]
    fn parse_notification_no_id() {
        let line = r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#;
        let req = parse_request(line).unwrap();
        assert_eq!(req.method, "notifications/initialized");
        assert!(req.id.is_none());
    }

    #[test]
    fn parse_invalid_json_returns_error() {
        let result = parse_request("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn success_response_serializes() {
        let resp = success_response(Some(Value::from(1)), serde_json::json!({"ok": true}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"ok\":true"));
        assert!(!json.contains("error"));
    }

    #[test]
    fn error_response_serializes() {
        let resp = error_response(Some(Value::from(1)), -32601, "Method not found");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("-32601"));
        assert!(json.contains("Method not found"));
        assert!(!json.contains("result"));
    }

    #[test]
    fn tool_call_result_success() {
        let result = ToolCallResult::success("hello".into());
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "hello");
        assert!(json.get("isError").is_none());
    }

    #[test]
    fn tool_call_result_error() {
        let result = ToolCallResult::error("bad input".into());
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["isError"], true);
        assert_eq!(json["content"][0]["text"], "bad input");
    }

    #[test]
    fn initialize_result_has_required_fields() {
        let result = initialize_result();
        assert!(result.get("protocolVersion").is_some());
        assert!(result.get("serverInfo").is_some());
        assert!(result.get("capabilities").is_some());
        assert_eq!(result["serverInfo"]["name"], "shrimpk-mcp");
    }
}
