//! Consolidator implementations — swappable fact extraction backends.
//!
//! During sleep consolidation, the consolidator extracts atomic facts from
//! stored memories so they can be embedded and retrieved more precisely.
//!
//! # Providers
//! - [`OllamaConsolidator`] — local Ollama API (default, offline)
//! - [`HttpConsolidator`] — any OpenAI-compatible API (cloud)
//! - [`NoopConsolidator`] — disabled, returns empty vec
//!
//! Use [`from_config`] to construct the right provider from [`EchoConfig`].

use shrimpk_core::{Consolidator, EchoConfig};

// ---- Shared prompt + parser ----

fn fact_extraction_prompt(max_facts: usize) -> String {
    format!(
        "You are extracting personal facts from a conversation snippet. \
         Output ONLY the facts, one per line. Be specific \
         - include names, dates, places, numbers, preferences. \
         Max {max_facts} facts. If no personal facts found, output NONE."
    )
}

/// Parse facts from LLM response. One fact per line, skip empties and "NONE".
fn parse_facts(response: &str, max_facts: usize) -> Vec<String> {
    response
        .lines()
        .map(|l| l.trim().trim_start_matches("- ").trim().to_string())
        .filter(|l| !l.is_empty())
        .filter(|l| !l.eq_ignore_ascii_case("none"))
        .filter(|l| l.len() > 5)
        .take(max_facts)
        .collect()
}

// ===========================================================================
// NoopConsolidator
// ===========================================================================

/// No-op consolidator. Returns empty vec. Used when enrichment is disabled.
pub struct NoopConsolidator;

impl Consolidator for NoopConsolidator {
    fn extract_facts(&self, _text: &str, _max_facts: usize) -> Vec<String> {
        Vec::new()
    }
    fn name(&self) -> &str {
        "noop"
    }
}

// ===========================================================================
// OllamaConsolidator
// ===========================================================================

/// Calls Ollama's native chat API for fact extraction.
pub struct OllamaConsolidator {
    url: String,
    model: String,
    client: reqwest::blocking::Client,
}

impl OllamaConsolidator {
    pub fn new(url: String, model: String) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self { url, model, client }
    }
}

impl Consolidator for OllamaConsolidator {
    fn extract_facts(&self, text: &str, max_facts: usize) -> Vec<String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": fact_extraction_prompt(max_facts)},
                {"role": "user", "content": text}
            ],
            "stream": false,
            "options": {"temperature": 0.0, "num_predict": 256}
        });

        let endpoint = format!("{}/api/chat", self.url.trim_end_matches('/'));

        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        tracing::info!(
            target: "shrimpk::audit",
            endpoint = %endpoint,
            data_bytes = body_bytes.len(),
            direction = "outbound",
            component = "consolidator",
            "External data transmission"
        );

        let resp = match self
            .client
            .post(&endpoint)
            .json(&body)
            .send()
        {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(provider = "ollama", error = %e, "Consolidator: Ollama unreachable");
                return Vec::new();
            }
        };

        let json: serde_json::Value = match resp.json() {
            Ok(j) => j,
            Err(e) => {
                tracing::debug!(provider = "ollama", error = %e, "Consolidator: parse error");
                return Vec::new();
            }
        };

        let content = json["message"]["content"].as_str().unwrap_or("");
        parse_facts(content, max_facts)
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

// ===========================================================================
// HttpConsolidator
// ===========================================================================

/// Calls any OpenAI-compatible chat completions API for fact extraction.
/// Works with OpenAI, xAI (Grok), Groq, Together, etc.
pub struct HttpConsolidator {
    url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl HttpConsolidator {
    pub fn new(url: String, model: String, api_key: Option<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self {
            url,
            model,
            api_key,
            client,
        }
    }
}

impl Consolidator for HttpConsolidator {
    fn extract_facts(&self, text: &str, max_facts: usize) -> Vec<String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": fact_extraction_prompt(max_facts)},
                {"role": "user", "content": text}
            ],
            "max_tokens": 512,
            "temperature": 0.0
        });

        let endpoint = format!(
            "{}/v1/chat/completions",
            self.url.trim_end_matches('/')
        );

        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        tracing::info!(
            target: "shrimpk::audit",
            endpoint = %endpoint,
            data_bytes = body_bytes.len(),
            direction = "outbound",
            component = "consolidator",
            "External data transmission"
        );

        let mut req = self
            .client
            .post(&endpoint)
            .json(&body);

        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = match req.send() {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(provider = "http", error = %e, "Consolidator: API unreachable");
                return Vec::new();
            }
        };

        let json: serde_json::Value = match resp.json() {
            Ok(j) => j,
            Err(e) => {
                tracing::debug!(provider = "http", error = %e, "Consolidator: parse error");
                return Vec::new();
            }
        };

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        parse_facts(content, max_facts)
    }

    fn name(&self) -> &str {
        "http"
    }
}

// ===========================================================================
// Factory
// ===========================================================================

/// Create a consolidator from the resolved config.
///
/// Maps `config.consolidation_provider` to a concrete implementation:
/// - `"ollama"` -> [`OllamaConsolidator`] (default)
/// - `"http"` / `"openai"` -> [`HttpConsolidator`]
/// - `"none"` / `"noop"` -> [`NoopConsolidator`]
pub fn from_config(config: &EchoConfig) -> Box<dyn Consolidator> {
    match config.consolidation_provider.to_lowercase().as_str() {
        "ollama" => {
            tracing::info!(
                url = %config.ollama_url,
                model = %config.enrichment_model,
                "Consolidator: Ollama"
            );
            Box::new(OllamaConsolidator::new(
                config.ollama_url.clone(),
                config.enrichment_model.clone(),
            ))
        }
        "http" | "openai" => {
            if !config.consolidation_consent_given {
                tracing::warn!(
                    "External consolidation endpoint configured but consent not given. \
                     Set consolidation_consent_given = true in config.toml to enable."
                );
                return Box::new(NoopConsolidator);
            }
            let api_key = std::env::var("SHRIMPK_CONSOLIDATION_API_KEY").ok();
            tracing::info!(
                url = %config.ollama_url,
                model = %config.enrichment_model,
                has_key = api_key.is_some(),
                "Consolidator: HTTP (OpenAI-compatible)"
            );
            Box::new(HttpConsolidator::new(
                config.ollama_url.clone(),
                config.enrichment_model.clone(),
                api_key,
            ))
        }
        "none" | "noop" => {
            tracing::info!("Consolidator: disabled");
            Box::new(NoopConsolidator)
        }
        other => {
            tracing::warn!(provider = other, "Unknown consolidation provider, falling back to noop");
            Box::new(NoopConsolidator)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrimpk_core::Consolidator;

    #[test]
    fn noop_returns_empty() {
        let c = NoopConsolidator;
        assert!(c.extract_facts("any text", 5).is_empty());
        assert_eq!(c.name(), "noop");
    }

    #[test]
    fn parse_facts_multiline() {
        let input = "User graduated with Business Administration\n\
                     User lives in New York\n\
                     NONE\n\
                     \n\
                     User prefers coffee over tea";
        let facts = parse_facts(input, 5);
        assert_eq!(facts.len(), 3);
        assert!(facts[0].contains("Business Administration"));
    }

    #[test]
    fn parse_facts_with_bullet_points() {
        let input = "- User's name is Alice\n- User works at Acme Corp\n- User has two cats";
        let facts = parse_facts(input, 5);
        assert_eq!(facts.len(), 3);
        assert!(facts[0].contains("Alice"));
    }

    #[test]
    fn parse_facts_respects_max() {
        let input = "Fact A\nFact B\nFact C\nFact D\nFact E\nFact F";
        let facts = parse_facts(input, 3);
        assert_eq!(facts.len(), 3);
    }

    #[test]
    fn parse_facts_filters_short() {
        let input = "Good fact about something\nhi\nok\nAnother good fact here";
        let facts = parse_facts(input, 5);
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn from_config_none_returns_noop() {
        let mut config = EchoConfig::default();
        config.consolidation_provider = "none".to_string();
        let c = from_config(&config);
        assert_eq!(c.name(), "noop");
    }

    #[test]
    fn from_config_unknown_returns_noop() {
        let mut config = EchoConfig::default();
        config.consolidation_provider = "banana".to_string();
        let c = from_config(&config);
        assert_eq!(c.name(), "noop");
    }

    #[test]
    fn from_config_default_is_ollama() {
        let config = EchoConfig::default();
        let c = from_config(&config);
        assert_eq!(c.name(), "ollama");
    }

    #[test]
    fn from_config_http_with_consent() {
        let mut config = EchoConfig::default();
        config.consolidation_provider = "http".to_string();
        config.consolidation_consent_given = true;
        let c = from_config(&config);
        assert_eq!(c.name(), "http");
    }

    #[test]
    fn from_config_http_without_consent_falls_back_to_noop() {
        let mut config = EchoConfig::default();
        config.consolidation_provider = "http".to_string();
        config.consolidation_consent_given = false;
        let c = from_config(&config);
        assert_eq!(c.name(), "noop");
    }

    #[test]
    fn from_config_openai_without_consent_falls_back_to_noop() {
        let mut config = EchoConfig::default();
        config.consolidation_provider = "openai".to_string();
        config.consolidation_consent_given = false;
        let c = from_config(&config);
        assert_eq!(c.name(), "noop");
    }

    #[test]
    fn ollama_handles_unreachable() {
        let c = OllamaConsolidator::new("http://127.0.0.1:99999".into(), "test".into());
        let facts = c.extract_facts("test text", 5);
        assert!(facts.is_empty());
    }
}
