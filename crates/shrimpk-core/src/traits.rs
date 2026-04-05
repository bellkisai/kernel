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

/// Type classification for extracted facts (KS67 — schema-driven extraction).
///
/// Maps to distinct retrieval patterns and supersession thresholds.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactType {
    Personal,
    Project,
    Preference,
    Goal,
    Status,
    Event,
    Relationship,
}

/// A structured fact extracted from memory content (KS67 — schema-driven extraction).
///
/// Replaces the flat `Vec<String>` fact representation with typed, entity-aware facts.
/// The LLM outputs these as JSON objects; the regex fallback produces legacy `String`
/// facts wrapped in `ExtractedFact { text, subject: None, .. }`.
#[derive(Debug, Clone, Default)]
pub struct ExtractedFact {
    /// The fact as a complete sentence (e.g., "Sam works at Anthropic").
    pub text: String,
    /// The primary entity this fact is about (e.g., "Sam", "MLTK").
    pub subject: Option<String>,
    /// Classification of the fact type.
    pub fact_type: Option<FactType>,
    /// LLM confidence in this fact (0.0–1.0).
    pub confidence: Option<f32>,
}

/// Structured output from the combined fact extraction + label classification call (ADR-015 D7).
#[derive(Debug, Clone, Default)]
pub struct ConsolidationOutput {
    /// Extracted atomic facts as plain strings (legacy path, backward compat).
    pub facts: Vec<String>,
    /// Semantic labels from LLM classification (Tier 2). None for legacy consolidators.
    pub labels: Option<LabelSet>,
    /// Structured facts with subject, type, and confidence (KS67 v2 path).
    /// When non-empty, these take precedence over `facts` for child creation.
    pub structured_facts: Vec<ExtractedFact>,
}

/// Semantic labels extracted by the LLM during consolidation (ADR-015).
/// Used for Tier 2 label enrichment (upgrades label_version from 1 to 2).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LabelSet {
    /// Topic labels (e.g., ["career", "language"]).
    #[serde(default)]
    pub topic: Vec<String>,
    /// Domain labels (e.g., ["work", "life"]).
    #[serde(default)]
    pub domain: Vec<String>,
    /// Action labels (e.g., ["learning", "building"]).
    #[serde(default)]
    pub action: Vec<String>,
    /// Memory type (e.g., "fact", "preference", "goal").
    #[serde(default)]
    pub memtype: Option<String>,
    /// Sentiment (e.g., "positive", "negative", "neutral").
    #[serde(default)]
    pub sentiment: Option<String>,
}

/// A backend that extracts atomic facts from memory text during consolidation.
///
/// Implementations call LLMs (local or cloud) to decompose a paragraph-length
/// memory into standalone facts that can be embedded and retrieved individually.
///
/// # Contract
/// - `extract_facts` must NEVER panic. On any error, return an empty `Vec`.
/// - Implementations must be `Send + Sync` for use across tokio tasks.
///
/// # Providers
/// - `OllamaConsolidator` — local Ollama (default, offline)
/// - `HttpConsolidator` — any OpenAI-compatible API (cloud)
/// - `NoopConsolidator` — disabled (returns empty)
/// - Future: `BuiltinConsolidator` — bundled ShrimPK model
pub trait Consolidator: Send + Sync {
    /// Extract up to `max_facts` atomic facts from the given text.
    ///
    /// Each returned string should be a standalone, self-contained statement.
    /// Returns an empty `Vec` on any error (network, parse, etc.).
    fn extract_facts(&self, text: &str, max_facts: usize) -> Vec<String>;

    /// Human-readable name for this consolidator backend.
    fn name(&self) -> &str;

    /// Combined fact extraction + label classification (ADR-015 D7).
    ///
    /// Returns both facts and semantic labels in a single call. The default
    /// implementation calls `extract_facts()` and returns labels: None,
    /// preserving backward compatibility with existing consolidators.
    ///
    /// Override this in consolidators that support the combined prompt
    /// (e.g., OllamaConsolidator with JSON schema output).
    fn extract_facts_and_labels(&self, text: &str, max_facts: usize) -> ConsolidationOutput {
        ConsolidationOutput {
            facts: self.extract_facts(text, max_facts),
            labels: None,
            ..Default::default()
        }
    }

    /// Summarize a cluster of memories for a given label (KS64 — GraphRAG P4).
    ///
    /// Called during consolidation for label clusters with enough members.
    /// The summary is stored as a `CommunitySummary` and used as a fallback
    /// when echo queries return weak results.
    ///
    /// Default: returns `None` (no-op for consolidators that don't support summarization).
    fn summarize_cluster(&self, _memories: &[&str], _label: &str) -> Option<String> {
        None
    }
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

    // --- KS46: ConsolidationOutput + LabelSet ---

    #[test]
    fn consolidation_output_default() {
        let output = ConsolidationOutput::default();
        assert!(output.facts.is_empty());
        assert!(output.labels.is_none());
        assert!(output.structured_facts.is_empty());
    }

    // --- KS67: ExtractedFact + FactType ---

    #[test]
    fn fact_type_serde_roundtrip() {
        let types = vec![
            FactType::Personal,
            FactType::Project,
            FactType::Preference,
            FactType::Goal,
            FactType::Status,
            FactType::Event,
            FactType::Relationship,
        ];
        for ft in &types {
            let json = serde_json::to_string(ft).unwrap();
            let deserialized: FactType = serde_json::from_str(&json).unwrap();
            assert_eq!(&deserialized, ft);
        }
        // Verify lowercase serialization
        assert_eq!(
            serde_json::to_string(&FactType::Personal).unwrap(),
            "\"personal\""
        );
        assert_eq!(
            serde_json::to_string(&FactType::Project).unwrap(),
            "\"project\""
        );
    }

    #[test]
    fn extracted_fact_default() {
        let fact = ExtractedFact::default();
        assert!(fact.text.is_empty());
        assert!(fact.subject.is_none());
        assert!(fact.fact_type.is_none());
        assert!(fact.confidence.is_none());
    }

    #[test]
    fn consolidation_output_with_structured_facts() {
        let output = ConsolidationOutput {
            facts: vec!["Sam works at Anthropic".into()],
            structured_facts: vec![ExtractedFact {
                text: "Sam works at Anthropic".into(),
                subject: Some("Sam".into()),
                fact_type: Some(FactType::Personal),
                confidence: Some(0.95),
            }],
            ..Default::default()
        };
        assert_eq!(output.structured_facts.len(), 1);
        assert_eq!(output.structured_facts[0].subject, Some("Sam".into()));
        assert_eq!(
            output.structured_facts[0].fact_type,
            Some(FactType::Personal)
        );
    }

    #[test]
    fn label_set_serde_roundtrip() {
        let ls = LabelSet {
            topic: vec!["career".into(), "technology".into()],
            domain: vec!["work".into()],
            action: vec!["building".into()],
            memtype: Some("preference".into()),
            sentiment: Some("positive".into()),
        };
        let json = serde_json::to_string(&ls).unwrap();
        let deserialized: LabelSet = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.topic, vec!["career", "technology"]);
        assert_eq!(deserialized.memtype, Some("preference".into()));
    }

    #[test]
    fn label_set_deserialize_with_missing_fields() {
        let json = r#"{"topic": ["language"]}"#;
        let ls: LabelSet = serde_json::from_str(json).unwrap();
        assert_eq!(ls.topic, vec!["language"]);
        assert!(ls.domain.is_empty());
        assert!(ls.memtype.is_none());
    }
}
