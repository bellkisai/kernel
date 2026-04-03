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

use shrimpk_core::{ConsolidationOutput, Consolidator, EchoConfig, LabelSet};

// ---- Shared prompt + parser ----

/// Default fact extraction prompt (C1 verb whitelist — KS22 winner).
/// Forces "The user [verb] [object]" format for regex-compatible Supersedes detection.
/// Produces ~26 clean facts vs ~73 noisy label:value pairs with the original prompt.
fn default_fact_prompt(max_facts: usize) -> String {
    format!(
        "Extract personal facts from the text. Rules:\n\
         1. One fact per line, starting with \"The user\"\n\
         2. Use ONLY these verbs: works at, works for, joined, lives in, moved to, \
         based in, uses, prefers, switched to, likes, chose, belongs to, member of, part of\n\
         3. No colons, labels, or key-value pairs\n\n\
         Example:\n  The user uses Neovim\n  The user lives in Berlin\n  \
         The user switched to Python from Java\n\n\
         Max {max_facts} facts. If none found, output NONE."
    )
}

/// Resolve the fact extraction prompt: use custom if configured, else default.
pub fn fact_extraction_prompt(config: &shrimpk_core::EchoConfig, max_facts: usize) -> String {
    match &config.fact_extraction_prompt {
        Some(custom) => custom.replace("{max_facts}", &max_facts.to_string()),
        None => default_fact_prompt(max_facts),
    }
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
    agent: ureq::Agent,
    system_prompt: Option<String>,
}

impl OllamaConsolidator {
    pub fn new(url: String, model: String, system_prompt: Option<String>) -> Self {
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(60)))
                .build(),
        );
        Self {
            url,
            model,
            agent,
            system_prompt,
        }
    }
}

/// Combined prompt for fact extraction + label classification (ADR-015 D7).
fn combined_enrichment_prompt(max_facts: usize) -> String {
    format!(
        "Analyze this memory and return a JSON object with two keys.\n\n\
         1. \"facts\": Extract up to {max_facts} atomic facts. Each starts with \"The user\".\n\
         2. \"labels\": Classify based on what this memory IS ABOUT, not just the words it contains.\n\
            A memory about attending a class implies learning/education.\n\
            A memory about running implies exercise/health.\n\n\
         Label categories:\n\
         - topic: [career, language, education, health, fitness, housing, food, music, technology, finance, travel, relationships, hobby, entertainment, pets]\n\
         - domain: [work, life, social, health, creative]\n\
         - action: [learning, building, planning, moving, exercising, leading, deciding]\n\
         - memtype: one of [fact, preference, goal, habit, event, opinion, plan]\n\
         - sentiment: one of [positive, negative, neutral, mixed]\n\n\
         Return JSON:\n\
         {{\"facts\": [\"The user ...\"], \"labels\": {{\"topic\": [...], \"domain\": [...], \"action\": [...], \"memtype\": \"...\", \"sentiment\": \"...\"}}}}"
    )
}

/// Parse the combined JSON response from the LLM.
fn parse_combined_response(content: &str, max_facts: usize) -> ConsolidationOutput {
    // Try to parse as JSON first
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
        let facts = json["facts"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| s.len() > 5)
                    .take(max_facts)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let labels = if let Some(labels_obj) = json.get("labels") {
            match serde_json::from_value::<LabelSet>(labels_obj.clone()) {
                Ok(ls) => Some(ls),
                Err(e) => {
                    tracing::debug!(error = %e, "Failed to parse LabelSet from combined response");
                    None
                }
            }
        } else {
            None
        };

        ConsolidationOutput { facts, labels }
    } else {
        // Fallback: parse as plain text facts (backward compat with non-JSON responses)
        ConsolidationOutput {
            facts: parse_facts(content, max_facts),
            labels: None,
        }
    }
}

impl Consolidator for OllamaConsolidator {
    fn extract_facts(&self, text: &str, max_facts: usize) -> Vec<String> {
        let prompt = match &self.system_prompt {
            Some(p) => p.replace("{max_facts}", &max_facts.to_string()),
            None => default_fact_prompt(max_facts),
        };
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
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

        let mut resp = match self.agent.post(&endpoint).send_json(&body) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(provider = "ollama", error = %e, "Consolidator: Ollama unreachable");
                return Vec::new();
            }
        };

        let json: serde_json::Value = match resp.body_mut().read_json() {
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

    fn extract_facts_and_labels(&self, text: &str, max_facts: usize) -> ConsolidationOutput {
        let prompt = combined_enrichment_prompt(max_facts);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "stream": false,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 512}
        });

        let endpoint = format!("{}/api/chat", self.url.trim_end_matches('/'));

        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        tracing::info!(
            target: "shrimpk::audit",
            endpoint = %endpoint,
            data_bytes = body_bytes.len(),
            direction = "outbound",
            component = "consolidator-combined",
            "External data transmission (facts + labels)"
        );

        let mut resp = match self.agent.post(&endpoint).send_json(&body) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(provider = "ollama", error = %e, "Combined consolidator: Ollama unreachable");
                return ConsolidationOutput::default();
            }
        };

        let json: serde_json::Value = match resp.body_mut().read_json() {
            Ok(j) => j,
            Err(e) => {
                tracing::debug!(provider = "ollama", error = %e, "Combined consolidator: parse error");
                return ConsolidationOutput::default();
            }
        };

        let content = json["message"]["content"].as_str().unwrap_or("");
        parse_combined_response(content, max_facts)
    }

    fn summarize_cluster(&self, memories: &[&str], label: &str) -> Option<String> {
        let joined = memories
            .iter()
            .enumerate()
            .map(|(i, m)| format!("{}. {}", i + 1, m))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are summarizing a cluster of related memories labeled \"{label}\".\n\
             Write a concise 1-3 sentence summary capturing the key themes and facts.\n\
             Output ONLY the summary text, nothing else."
        );

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": joined}
            ],
            "stream": false,
            "options": {"temperature": 0.0, "num_predict": 200}
        });

        let endpoint = format!("{}/api/chat", self.url.trim_end_matches('/'));
        let mut resp = self.agent.post(&endpoint).send_json(&body).ok()?;
        let json: serde_json::Value = resp.body_mut().read_json().ok()?;
        let content = json["message"]["content"].as_str()?;
        let trimmed = content.trim();
        if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
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
    agent: ureq::Agent,
    system_prompt: Option<String>,
}

impl HttpConsolidator {
    pub fn new(
        url: String,
        model: String,
        api_key: Option<String>,
        system_prompt: Option<String>,
    ) -> Self {
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(60)))
                .build(),
        );
        Self {
            url,
            model,
            api_key,
            agent,
            system_prompt,
        }
    }
}

impl Consolidator for HttpConsolidator {
    fn extract_facts(&self, text: &str, max_facts: usize) -> Vec<String> {
        let prompt = match &self.system_prompt {
            Some(p) => p.replace("{max_facts}", &max_facts.to_string()),
            None => default_fact_prompt(max_facts),
        };
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "max_tokens": 512,
            "temperature": 0.0
        });

        let endpoint = format!("{}/v1/chat/completions", self.url.trim_end_matches('/'));

        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        tracing::info!(
            target: "shrimpk::audit",
            endpoint = %endpoint,
            data_bytes = body_bytes.len(),
            direction = "outbound",
            component = "consolidator",
            "External data transmission"
        );

        let mut req = self.agent.post(&endpoint);

        if let Some(key) = &self.api_key {
            req = req.header("Authorization", &format!("Bearer {key}"));
        }

        let mut resp = match req.send_json(&body) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(provider = "http", error = %e, "Consolidator: API unreachable");
                return Vec::new();
            }
        };

        let json: serde_json::Value = match resp.body_mut().read_json() {
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
                config.fact_extraction_prompt.clone(),
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
                config.fact_extraction_prompt.clone(),
            ))
        }
        "none" | "noop" => {
            tracing::info!("Consolidator: disabled");
            Box::new(NoopConsolidator)
        }
        other => {
            tracing::warn!(
                provider = other,
                "Unknown consolidation provider, falling back to noop"
            );
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
        let c = OllamaConsolidator::new("http://127.0.0.1:99999".into(), "test".into(), None);
        let facts = c.extract_facts("test text", 5);
        assert!(facts.is_empty());
    }

    // --- KS46: ConsolidationOutput + combined response parsing ---

    #[test]
    fn noop_extract_facts_and_labels_returns_none_labels() {
        let c = NoopConsolidator;
        let output = c.extract_facts_and_labels("any text", 5);
        assert!(output.facts.is_empty());
        assert!(output.labels.is_none(), "Noop should return labels: None");
    }

    #[test]
    fn parse_combined_valid_json() {
        let json = r#"{"facts": ["The user uses Neovim", "The user lives in Berlin"], "labels": {"topic": ["technology"], "domain": ["work"], "action": ["building"], "memtype": "preference", "sentiment": "positive"}}"#;
        let output = parse_combined_response(json, 10);
        assert_eq!(output.facts.len(), 2);
        assert!(output.facts[0].contains("Neovim"));

        let labels = output.labels.expect("Should have labels");
        assert_eq!(labels.topic, vec!["technology"]);
        assert_eq!(labels.domain, vec!["work"]);
        assert_eq!(labels.action, vec!["building"]);
        assert_eq!(labels.memtype, Some("preference".into()));
        assert_eq!(labels.sentiment, Some("positive".into()));
    }

    #[test]
    fn parse_combined_facts_only_fallback() {
        // If LLM returns plain text instead of JSON, fall back to fact parsing
        let text = "The user uses Neovim\nThe user lives in Berlin";
        let output = parse_combined_response(text, 10);
        assert_eq!(output.facts.len(), 2);
        assert!(output.labels.is_none(), "Plain text should have no labels");
    }

    #[test]
    fn parse_combined_partial_labels() {
        // JSON with facts but malformed labels — facts should still work
        let json = r#"{"facts": ["The user prefers dark mode"], "labels": "invalid"}"#;
        let output = parse_combined_response(json, 10);
        assert_eq!(output.facts.len(), 1);
        assert!(output.labels.is_none(), "Malformed labels should be None");
    }

    #[test]
    fn parse_combined_empty_labels() {
        let json = r#"{"facts": ["The user runs daily"], "labels": {"topic": [], "domain": [], "action": [], "memtype": null, "sentiment": null}}"#;
        let output = parse_combined_response(json, 10);
        assert_eq!(output.facts.len(), 1);
        let labels = output.labels.expect("Should parse empty labels");
        assert!(labels.topic.is_empty());
        assert!(labels.memtype.is_none());
    }

    #[test]
    fn parse_combined_respects_max_facts() {
        let json = r#"{"facts": ["Fact A here", "Fact B here", "Fact C here", "Fact D here"], "labels": {}}"#;
        let output = parse_combined_response(json, 2);
        assert_eq!(output.facts.len(), 2);
    }

    #[test]
    fn combined_prompt_contains_label_categories() {
        let prompt = combined_enrichment_prompt(5);
        assert!(prompt.contains("topic:"), "Prompt should mention topic");
        assert!(prompt.contains("memtype:"), "Prompt should mention memtype");
        assert!(
            prompt.contains("sentiment:"),
            "Prompt should mention sentiment"
        );
        assert!(
            prompt.contains("IS ABOUT"),
            "Prompt should emphasize implicit inference"
        );
    }

    #[test]
    fn ollama_combined_handles_unreachable() {
        let c = OllamaConsolidator::new("http://127.0.0.1:99999".into(), "test".into(), None);
        let output = c.extract_facts_and_labels("test text", 5);
        assert!(output.facts.is_empty());
        assert!(output.labels.is_none());
    }
}
