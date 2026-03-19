//! Provider and model configuration types for the router.
//!
//! These structs define how providers and models are described,
//! how routing requests are formed, and what decisions the router returns.

use serde::{Deserialize, Serialize};
use shrimpk_core::traits::{ModelId, ProviderId};

/// Configuration for an AI provider (e.g., Ollama, OpenAI, Anthropic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Unique provider identifier.
    pub id: ProviderId,
    /// Human-readable name (e.g., "OpenAI", "Local Ollama").
    pub name: String,
    /// Base API URL for this provider.
    pub api_url: String,
    /// Models available from this provider.
    pub models: Vec<ModelConfig>,
    /// Whether this provider runs locally (no data leaves the machine).
    pub is_local: bool,
    /// Whether this provider is currently enabled for routing.
    pub enabled: bool,
}

/// Configuration for a single model within a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Unique model identifier.
    pub id: ModelId,
    /// Cost in USD per input token.
    pub cost_per_input_token: f64,
    /// Cost in USD per output token.
    pub cost_per_output_token: f64,
    /// Maximum context window in tokens.
    pub max_context_tokens: usize,
    /// Whether this model supports vision/image input.
    pub supports_vision: bool,
    /// Whether this model supports function calling / tool use.
    pub supports_function_calling: bool,
    /// Estimated quality score (0.0 = lowest, 1.0 = highest).
    pub quality_score: f64,
}

/// A request to the router asking it to pick a provider + model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRequest {
    /// The user's query text (used for cost estimation via token count heuristic).
    pub query: String,
    /// Required capabilities (e.g., "vision", "function_calling").
    pub required_capabilities: Vec<String>,
    /// Maximum USD budget for this single request.
    pub budget_limit: Option<f64>,
    /// If set, try this provider first before falling back.
    pub preferred_provider: Option<ProviderId>,
    /// If true, only route to local providers (no data leaves machine).
    pub is_sensitive: bool,
}

/// The router's decision: which provider + model to use and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    /// Selected provider.
    pub provider_id: ProviderId,
    /// Selected model within that provider.
    pub model_id: ModelId,
    /// Human-readable explanation of why this route was chosen.
    pub reason: String,
    /// Estimated cost in USD for this request.
    pub estimated_cost: f64,
    /// Whether the selected provider is local.
    pub is_local: bool,
}

impl ModelConfig {
    /// Check whether this model satisfies all required capabilities.
    pub fn satisfies_capabilities(&self, required: &[String]) -> bool {
        for cap in required {
            match cap.as_str() {
                "vision" => {
                    if !self.supports_vision {
                        return false;
                    }
                }
                "function_calling" => {
                    if !self.supports_function_calling {
                        return false;
                    }
                }
                _ => {
                    // Unknown capability — cannot satisfy.
                    return false;
                }
            }
        }
        true
    }

    /// Estimate the cost of a request given an approximate token count.
    /// Uses a simple heuristic: ~4 chars per token for input,
    /// and assumes output is roughly equal to input length.
    pub fn estimate_cost(&self, query: &str) -> f64 {
        let estimated_input_tokens = (query.len() as f64 / 4.0).ceil() as u64;
        // Assume output is roughly the same size as input for estimation.
        let estimated_output_tokens = estimated_input_tokens;
        (estimated_input_tokens as f64 * self.cost_per_input_token)
            + (estimated_output_tokens as f64 * self.cost_per_output_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(vision: bool, fc: bool, quality: f64) -> ModelConfig {
        ModelConfig {
            id: ModelId::new("test-model"),
            cost_per_input_token: 0.001,
            cost_per_output_token: 0.002,
            max_context_tokens: 4096,
            supports_vision: vision,
            supports_function_calling: fc,
            quality_score: quality,
        }
    }

    #[test]
    fn capability_check_vision() {
        let m = make_model(true, false, 0.8);
        assert!(m.satisfies_capabilities(&["vision".into()]));
        assert!(!m.satisfies_capabilities(&["function_calling".into()]));
    }

    #[test]
    fn capability_check_function_calling() {
        let m = make_model(false, true, 0.8);
        assert!(m.satisfies_capabilities(&["function_calling".into()]));
        assert!(!m.satisfies_capabilities(&["vision".into()]));
    }

    #[test]
    fn capability_check_both() {
        let m = make_model(true, true, 0.9);
        assert!(m.satisfies_capabilities(&["vision".into(), "function_calling".into()]));
    }

    #[test]
    fn capability_check_unknown_rejected() {
        let m = make_model(true, true, 0.9);
        assert!(!m.satisfies_capabilities(&["teleportation".into()]));
    }

    #[test]
    fn cost_estimation() {
        let m = ModelConfig {
            id: ModelId::new("test"),
            cost_per_input_token: 0.01,
            cost_per_output_token: 0.02,
            max_context_tokens: 4096,
            supports_vision: false,
            supports_function_calling: false,
            quality_score: 0.5,
        };
        // "hello world" = 11 chars => ceil(11/4) = 3 tokens input, 3 tokens output
        let cost = m.estimate_cost("hello world");
        let expected = 3.0 * 0.01 + 3.0 * 0.02;
        assert!((cost - expected).abs() < 1e-10);
    }
}
