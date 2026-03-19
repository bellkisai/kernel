//! Cascade routing logic.
//!
//! The `CascadeRouter` selects the optimal provider + model for a given request
//! by filtering on capabilities, locality, budget, and then ranking by a
//! cost-adjusted quality score.

use shrimpk_core::error::ShrimPKError;
use shrimpk_core::traits::ProviderId;

use crate::config::{ModelConfig, ProviderConfig, RouteDecision, RouteRequest};

/// Cascade router that selects the best provider + model for each request.
///
/// Routing algorithm:
/// 1. Filter providers: enabled, has models satisfying capabilities, within budget.
/// 2. If `is_sensitive` — only local providers.
/// 3. If `preferred_provider` — try it first.
/// 4. Sort remaining by: cheapest cost * inverse quality_score (lower = better).
/// 5. Return the best match.
/// 6. If none found — return error.
pub struct CascadeRouter {
    providers: Vec<ProviderConfig>,
    fallback_chain: Vec<ProviderId>,
}

/// A candidate (provider, model) pair with its routing score.
struct Candidate<'a> {
    provider: &'a ProviderConfig,
    model: &'a ModelConfig,
    estimated_cost: f64,
    /// Lower score = better. Computed as cost / quality_score.
    score: f64,
}

impl CascadeRouter {
    /// Create a new router with the given provider configs.
    pub fn new(providers: Vec<ProviderConfig>) -> Self {
        let fallback_chain = providers.iter().map(|p| p.id.clone()).collect();
        Self {
            providers,
            fallback_chain,
        }
    }

    /// Route a request to the best available provider + model.
    pub fn route(&self, request: &RouteRequest) -> Result<RouteDecision, ShrimPKError> {
        let mut candidates: Vec<Candidate> = Vec::new();

        for provider in &self.providers {
            // Skip disabled providers.
            if !provider.enabled {
                continue;
            }

            // If sensitive, only allow local providers.
            if request.is_sensitive && !provider.is_local {
                continue;
            }

            for model in &provider.models {
                // Check capabilities.
                if !model.satisfies_capabilities(&request.required_capabilities) {
                    continue;
                }

                let estimated_cost = model.estimate_cost(&request.query);

                // Check per-request budget limit.
                if let Some(budget) = request.budget_limit
                    && estimated_cost > budget
                {
                    continue;
                }

                // Score: lower is better. Divide cost by quality so higher quality
                // reduces the score (making the candidate more attractive).
                // Guard against zero quality_score.
                let quality = if model.quality_score > 0.0 {
                    model.quality_score
                } else {
                    0.01 // floor to avoid division by zero
                };
                let score = estimated_cost / quality;

                candidates.push(Candidate {
                    provider,
                    model,
                    estimated_cost,
                    score,
                });
            }
        }

        if candidates.is_empty() {
            return Err(ShrimPKError::Router(
                "No suitable provider found for the given request".into(),
            ));
        }

        // If preferred_provider is set, check if any candidate matches.
        if let Some(ref preferred) = request.preferred_provider {
            let preferred_candidates: Vec<&Candidate> = candidates
                .iter()
                .filter(|c| c.provider.id == *preferred)
                .collect();

            if let Some(best) = preferred_candidates.iter().min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                return Ok(RouteDecision {
                    provider_id: best.provider.id.clone(),
                    model_id: best.model.id.clone(),
                    reason: format!(
                        "Preferred provider '{}' selected (model '{}', quality {:.2})",
                        best.provider.name, best.model.id, best.model.quality_score
                    ),
                    estimated_cost: best.estimated_cost,
                    is_local: best.provider.is_local,
                });
            }
            // Preferred provider not available — fall through to best overall.
        }

        // Sort by score (lowest = best).
        candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = &candidates[0];
        let reason = if request.is_sensitive {
            format!(
                "Local provider '{}' selected for sensitive data (model '{}', quality {:.2})",
                best.provider.name, best.model.id, best.model.quality_score
            )
        } else {
            format!(
                "Best cost/quality: '{}' model '{}' (quality {:.2}, est. ${:.6})",
                best.provider.name, best.model.id, best.model.quality_score, best.estimated_cost
            )
        };

        Ok(RouteDecision {
            provider_id: best.provider.id.clone(),
            model_id: best.model.id.clone(),
            reason,
            estimated_cost: best.estimated_cost,
            is_local: best.provider.is_local,
        })
    }

    /// Add a new provider to the router.
    pub fn add_provider(&mut self, config: ProviderConfig) {
        self.fallback_chain.push(config.id.clone());
        self.providers.push(config);
    }

    /// Remove a provider by ID. Returns true if found and removed.
    pub fn remove_provider(&mut self, id: &ProviderId) -> bool {
        let before = self.providers.len();
        self.providers.retain(|p| p.id != *id);
        self.fallback_chain.retain(|pid| pid != id);
        self.providers.len() < before
    }

    /// Get the number of registered providers.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Get the fallback chain order.
    pub fn fallback_chain(&self) -> &[ProviderId] {
        &self.fallback_chain
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────

#[cfg(test)]
mod test_helpers {
    use super::*;
    use shrimpk_core::traits::ModelId;

    pub fn local_provider() -> ProviderConfig {
        ProviderConfig {
            id: ProviderId::new("ollama"),
            name: "Local Ollama".into(),
            api_url: "http://localhost:11434".into(),
            models: vec![ModelConfig {
                id: ModelId::new("llama-3.1-8b"),
                cost_per_input_token: 0.0,
                cost_per_output_token: 0.0,
                max_context_tokens: 8192,
                supports_vision: false,
                supports_function_calling: false,
                quality_score: 0.6,
            }],
            is_local: true,
            enabled: true,
        }
    }

    pub fn cheap_cloud_provider() -> ProviderConfig {
        ProviderConfig {
            id: ProviderId::new("openai"),
            name: "OpenAI".into(),
            api_url: "https://api.openai.com".into(),
            models: vec![ModelConfig {
                id: ModelId::new("gpt-4o-mini"),
                cost_per_input_token: 0.00015,
                cost_per_output_token: 0.0006,
                max_context_tokens: 128_000,
                supports_vision: true,
                supports_function_calling: true,
                quality_score: 0.85,
            }],
            is_local: false,
            enabled: true,
        }
    }

    pub fn expensive_cloud_provider() -> ProviderConfig {
        ProviderConfig {
            id: ProviderId::new("anthropic"),
            name: "Anthropic".into(),
            api_url: "https://api.anthropic.com".into(),
            models: vec![ModelConfig {
                id: ModelId::new("claude-opus-4"),
                cost_per_input_token: 0.015,
                cost_per_output_token: 0.075,
                max_context_tokens: 200_000,
                supports_vision: true,
                supports_function_calling: true,
                quality_score: 0.98,
            }],
            is_local: false,
            enabled: true,
        }
    }

    pub fn disabled_provider() -> ProviderConfig {
        ProviderConfig {
            id: ProviderId::new("disabled-provider"),
            name: "Disabled".into(),
            api_url: "https://disabled.example.com".into(),
            models: vec![ModelConfig {
                id: ModelId::new("disabled-model"),
                cost_per_input_token: 0.0,
                cost_per_output_token: 0.0,
                max_context_tokens: 4096,
                supports_vision: true,
                supports_function_calling: true,
                quality_score: 1.0,
            }],
            is_local: false,
            enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::*;
    use super::*;

    #[test]
    fn route_to_cheapest_provider() {
        let router = CascadeRouter::new(vec![cheap_cloud_provider(), expensive_cloud_provider()]);
        let request = RouteRequest {
            query: "Hello world".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: false,
        };
        let decision = router.route(&request).unwrap();
        // The local provider has cost 0, but we didn't include it.
        // Between OpenAI and Anthropic, OpenAI is cheaper per-quality.
        assert_eq!(decision.provider_id, ProviderId::new("openai"));
    }

    #[test]
    fn route_to_local_when_sensitive() {
        let router = CascadeRouter::new(vec![
            local_provider(),
            cheap_cloud_provider(),
            expensive_cloud_provider(),
        ]);
        let request = RouteRequest {
            query: "My secret medical data".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: true,
        };
        let decision = router.route(&request).unwrap();
        assert_eq!(decision.provider_id, ProviderId::new("ollama"));
        assert!(decision.is_local);
    }

    #[test]
    fn preferred_provider_honored() {
        let router = CascadeRouter::new(vec![
            local_provider(),
            cheap_cloud_provider(),
            expensive_cloud_provider(),
        ]);
        let request = RouteRequest {
            query: "Tell me a joke".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: Some(ProviderId::new("anthropic")),
            is_sensitive: false,
        };
        let decision = router.route(&request).unwrap();
        assert_eq!(decision.provider_id, ProviderId::new("anthropic"));
    }

    #[test]
    fn no_provider_available_returns_error() {
        let router = CascadeRouter::new(vec![disabled_provider()]);
        let request = RouteRequest {
            query: "Hello".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: false,
        };
        let result = router.route(&request);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No suitable provider"));
    }

    #[test]
    fn budget_enforcement_blocks_over_limit() {
        let router = CascadeRouter::new(vec![
            expensive_cloud_provider(), // very expensive
        ]);
        let request = RouteRequest {
            query: "Write me a novel".into(),
            required_capabilities: vec![],
            budget_limit: Some(0.000001), // extremely low budget
            preferred_provider: None,
            is_sensitive: false,
        };
        let result = router.route(&request);
        assert!(result.is_err());
    }

    #[test]
    fn capability_filtering_vision() {
        let router = CascadeRouter::new(vec![
            local_provider(),       // no vision
            cheap_cloud_provider(), // has vision
        ]);
        let request = RouteRequest {
            query: "Describe this image".into(),
            required_capabilities: vec!["vision".into()],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: false,
        };
        let decision = router.route(&request).unwrap();
        // Only OpenAI supports vision in this set.
        assert_eq!(decision.provider_id, ProviderId::new("openai"));
    }

    #[test]
    fn capability_filtering_function_calling() {
        let router = CascadeRouter::new(vec![
            local_provider(),       // no function calling
            cheap_cloud_provider(), // has function calling
        ]);
        let request = RouteRequest {
            query: "Call the weather API".into(),
            required_capabilities: vec!["function_calling".into()],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: false,
        };
        let decision = router.route(&request).unwrap();
        assert_eq!(decision.provider_id, ProviderId::new("openai"));
    }

    #[test]
    fn sensitive_with_no_local_provider_fails() {
        let router = CascadeRouter::new(vec![cheap_cloud_provider(), expensive_cloud_provider()]);
        let request = RouteRequest {
            query: "Private data".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: true,
        };
        let result = router.route(&request);
        assert!(result.is_err());
    }

    #[test]
    fn add_and_remove_provider() {
        let mut router = CascadeRouter::new(vec![local_provider()]);
        assert_eq!(router.provider_count(), 1);

        router.add_provider(cheap_cloud_provider());
        assert_eq!(router.provider_count(), 2);

        router.remove_provider(&ProviderId::new("ollama"));
        assert_eq!(router.provider_count(), 1);
        assert_eq!(router.fallback_chain()[0], ProviderId::new("openai"));
    }

    #[test]
    fn free_local_model_wins_over_paid_cloud() {
        let router = CascadeRouter::new(vec![
            local_provider(),       // cost 0, quality 0.6
            cheap_cloud_provider(), // cost > 0, quality 0.85
        ]);
        let request = RouteRequest {
            query: "Simple question".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: false,
        };
        let decision = router.route(&request).unwrap();
        // Free local model has score 0/0.6 = 0, which beats any positive cost.
        assert_eq!(decision.provider_id, ProviderId::new("ollama"));
    }

    #[test]
    fn preferred_provider_fallback_when_unavailable() {
        // Preferred provider is disabled, should fall back to best available.
        let router = CascadeRouter::new(vec![cheap_cloud_provider(), disabled_provider()]);
        let request = RouteRequest {
            query: "Hello".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: Some(ProviderId::new("disabled-provider")),
            is_sensitive: false,
        };
        let decision = router.route(&request).unwrap();
        // Disabled provider skipped, falls through to OpenAI.
        assert_eq!(decision.provider_id, ProviderId::new("openai"));
    }

    #[test]
    fn empty_router_returns_error() {
        let router = CascadeRouter::new(vec![]);
        let request = RouteRequest {
            query: "Hello".into(),
            required_capabilities: vec![],
            budget_limit: None,
            preferred_provider: None,
            is_sensitive: false,
        };
        assert!(router.route(&request).is_err());
    }
}
