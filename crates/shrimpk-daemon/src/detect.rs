//! Provider auto-detection — scans for installed LLM providers.
//!
//! Probes default ports for known OpenAI-compatible and Ollama-compatible
//! backends, fetches their model lists, and builds a model-name -> provider URL
//! routing table so the daemon can forward requests to the right backend.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// A provider discovered on localhost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedProvider {
    pub name: String,
    pub url: String,
    pub port: u16,
    pub models: Vec<String>,
    pub is_running: bool,
}

/// How to parse the model-list response from a provider.
#[derive(Debug, Clone, Copy)]
enum ProbeType {
    /// Ollama's `/api/tags` → `{"models": [{"name": "..."}]}`
    OllamaTags,
    /// OpenAI-compatible `/v1/models` → `{"data": [{"id": "..."}]}`
    OpenAiModels,
}

/// Known providers with their default ports and probe endpoints.
const PROVIDERS: &[(&str, u16, &str, ProbeType)] = &[
    ("Ollama", 11434, "/api/tags", ProbeType::OllamaTags),
    ("LM Studio", 1234, "/v1/models", ProbeType::OpenAiModels),
    ("Jan.ai", 1337, "/v1/models", ProbeType::OpenAiModels),
    ("vLLM", 8000, "/v1/models", ProbeType::OpenAiModels),
    ("LocalAI", 8080, "/v1/models", ProbeType::OpenAiModels),
    ("GPT4All", 4891, "/v1/models", ProbeType::OpenAiModels),
];

/// ShrimPK's own daemon port — never probe this.
const SHRIMPK_PORT: u16 = 11435;

/// Timeout for each provider probe.
const PROBE_TIMEOUT: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// Probe helpers
// ---------------------------------------------------------------------------

/// Probe a single provider: hit its model-list endpoint and parse model names.
async fn probe_one(
    client: &reqwest::Client,
    name: &str,
    port: u16,
    path: &str,
    probe_type: ProbeType,
) -> Option<DetectedProvider> {
    if port == SHRIMPK_PORT {
        return None;
    }

    let url = format!("http://127.0.0.1:{port}{path}");
    let resp = tokio::time::timeout(PROBE_TIMEOUT, client.get(&url).send())
        .await
        .ok()?  // timeout
        .ok()?; // request error

    if !resp.status().is_success() {
        return None;
    }

    let body = tokio::time::timeout(PROBE_TIMEOUT, resp.text())
        .await
        .ok()?  // timeout
        .ok()?; // read error

    let models = parse_models(&body, probe_type);

    Some(DetectedProvider {
        name: name.to_string(),
        url: format!("http://127.0.0.1:{port}"),
        port,
        models,
        is_running: true,
    })
}

/// Parse model names from a JSON response body.
fn parse_models(body: &str, probe_type: ProbeType) -> Vec<String> {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
        return Vec::new();
    };

    match probe_type {
        ProbeType::OllamaTags => {
            // {"models": [{"name": "gemma3:1b", ...}, ...]}
            json.get("models")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        }
        ProbeType::OpenAiModels => {
            // {"data": [{"id": "model-name", ...}, ...]}
            json.get("data")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| m.get("id").and_then(|n| n.as_str()).map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Probe all known provider ports in parallel. Returns only providers that
/// responded successfully. Each probe has a 500ms timeout — total wall time
/// is ~500ms regardless of how many providers are checked.
pub async fn detect_providers(client: &reqwest::Client) -> Vec<DetectedProvider> {
    let futures: Vec<_> = PROVIDERS
        .iter()
        .map(|(name, port, path, probe_type)| probe_one(client, name, *port, path, *probe_type))
        .collect();

    let results = futures_util::future::join_all(futures).await;

    results.into_iter().flatten().collect()
}

/// Build a model_name -> provider_url routing map from detected providers.
/// When a model appears in multiple providers, the first provider wins
/// (ordered by the PROVIDERS constant above).
pub fn build_model_routes(providers: &[DetectedProvider]) -> HashMap<String, String> {
    let mut routes = HashMap::new();
    for provider in providers {
        for model in &provider.models {
            // First provider to claim a model name wins
            routes.entry(model.clone()).or_insert_with(|| provider.url.clone());
        }
    }
    routes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ollama_tags() {
        let body = r#"{"models":[{"name":"gemma3:1b","size":123},{"name":"llama3.2:latest","size":456}]}"#;
        let models = parse_models(body, ProbeType::OllamaTags);
        assert_eq!(models, vec!["gemma3:1b", "llama3.2:latest"]);
    }

    #[test]
    fn parse_openai_models() {
        let body = r#"{"data":[{"id":"gpt-4","object":"model"},{"id":"mistral-7b","object":"model"}]}"#;
        let models = parse_models(body, ProbeType::OpenAiModels);
        assert_eq!(models, vec!["gpt-4", "mistral-7b"]);
    }

    #[test]
    fn parse_empty_response() {
        assert!(parse_models("{}", ProbeType::OllamaTags).is_empty());
        assert!(parse_models("{}", ProbeType::OpenAiModels).is_empty());
        assert!(parse_models("not json", ProbeType::OllamaTags).is_empty());
    }

    #[test]
    fn parse_missing_fields() {
        let body = r#"{"models":[]}"#;
        assert!(parse_models(body, ProbeType::OllamaTags).is_empty());

        let body = r#"{"data":[]}"#;
        assert!(parse_models(body, ProbeType::OpenAiModels).is_empty());
    }

    #[test]
    fn build_routes_first_provider_wins() {
        let providers = vec![
            DetectedProvider {
                name: "Ollama".into(),
                url: "http://127.0.0.1:11434".into(),
                port: 11434,
                models: vec!["shared-model".into(), "ollama-only".into()],
                is_running: true,
            },
            DetectedProvider {
                name: "LM Studio".into(),
                url: "http://127.0.0.1:1234".into(),
                port: 1234,
                models: vec!["shared-model".into(), "lmstudio-only".into()],
                is_running: true,
            },
        ];

        let routes = build_model_routes(&providers);
        assert_eq!(routes.get("shared-model").unwrap(), "http://127.0.0.1:11434");
        assert_eq!(routes.get("ollama-only").unwrap(), "http://127.0.0.1:11434");
        assert_eq!(routes.get("lmstudio-only").unwrap(), "http://127.0.0.1:1234");
    }

    #[test]
    fn build_routes_empty() {
        let routes = build_model_routes(&[]);
        assert!(routes.is_empty());
    }

    #[test]
    fn shrimpk_port_skipped() {
        // Verify the constant matches expectations
        assert_eq!(SHRIMPK_PORT, 11435);
    }
}
