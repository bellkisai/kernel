//! Trait definitions for kernel subsystems.
//!
//! These traits define the contracts between kernel crates.
//! Concrete implementations live in their respective crates.
//! Stubs for KS1 — full implementation in KS2+.

use crate::error::Result;

/// Unique identifier for an AI provider.
#[derive(Debug, Clone, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProviderId(pub String);

impl ProviderId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for ProviderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a model.
#[derive(Debug, Clone, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An AI provider that can serve models.
///
/// Implementations: Ollama, OpenAI, Anthropic, Google, etc.
/// Full implementation in shrimpk-router (KS2).
pub trait Provider: Send + Sync {
    /// Provider identifier (e.g., "ollama", "openai").
    fn id(&self) -> &ProviderId;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// List available models from this provider.
    fn models(&self) -> Vec<ModelId>;

    /// Check if the provider is reachable and healthy.
    fn is_healthy(&self) -> bool;
}

/// A model backend that can generate text and/or embeddings.
///
/// Implementations: Ollama backend, cloud API backend, direct llama.cpp, MLX.
/// Full implementation in shrimpk-router (KS2).
pub trait ModelBackend: Send + Sync {
    /// Generate a text completion.
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;

    /// Generate embeddings for the given text.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// What this model can do.
    fn capabilities(&self) -> ModelCapabilities;
}

/// Capabilities of a model.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilities {
    /// Maximum context window in tokens.
    pub max_context_tokens: usize,
    /// Supports vision/image input.
    pub supports_vision: bool,
    /// Supports function calling / tool use.
    pub supports_function_calling: bool,
    /// Supports streaming output.
    pub supports_streaming: bool,
    /// Whether this is a local model (no data leaves the machine).
    pub is_local: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_id_equality() {
        let a = ProviderId::new("ollama");
        let b = ProviderId::new("ollama");
        let c = ProviderId::new("openai");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn model_id_display() {
        let id = ModelId::new("llama-3.1-8b");
        assert_eq!(id.to_string(), "llama-3.1-8b");
    }
}
