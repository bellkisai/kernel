//! Memory types — the core data structures for Echo Memory.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a stored memory.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryId(pub uuid::Uuid);

impl MemoryId {
    /// Create a new random MemoryId.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }

    /// Create from an existing UUID.
    pub fn from_uuid(id: uuid::Uuid) -> Self {
        Self(id)
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Category classification for adaptive memory decay.
///
/// Different types of memories should persist for different durations.
/// Project context stays relevant for weeks; preferences last months;
/// one-off conversations fade in days.  This is Echo Memory's
/// differentiator over MuninnDB's uniform ACT-R decay (d=0.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryCategory {
    /// Current work, active context — half-life: 14 days.
    ActiveProject,
    /// Tool choices, coding style, workflow preferences — half-life: 60 days.
    Preference,
    /// One-time discussions, quick questions — half-life: 3 days.
    Conversation,
    /// Name, location, language, personal facts — half-life: 365 days.
    Identity,
    /// Learned information, technical knowledge — half-life: 30 days.
    Fact,
    /// Uncategorized memories — half-life: 7 days.
    #[default]
    Default,
}

impl MemoryCategory {
    /// Half-life in seconds for this category's decay rate.
    pub fn half_life_secs(&self) -> f64 {
        match self {
            Self::ActiveProject => 14.0 * 86400.0, // 14 days
            Self::Preference => 60.0 * 86400.0,    // 60 days
            Self::Conversation => 3.0 * 86400.0,   // 3 days
            Self::Identity => 365.0 * 86400.0,     // 1 year
            Self::Fact => 30.0 * 86400.0,          // 30 days
            Self::Default => 7.0 * 86400.0,        // 7 days
        }
    }

    /// Importance weight for this category (levels-of-processing theory).
    pub fn importance_weight(&self) -> f32 {
        match self {
            Self::Identity => 1.0,
            Self::Preference => 0.8,
            Self::Fact => 0.6,
            Self::ActiveProject => 0.5,
            Self::Default => 0.3,
            Self::Conversation => 0.1,
        }
    }

    /// ACT-R decay parameter d for this category.
    pub fn actr_decay_d(&self) -> f64 {
        match self {
            Self::Identity => 0.3,
            Self::Preference => 0.4,
            Self::Fact | Self::ActiveProject | Self::Default => 0.5,
            Self::Conversation => 0.7,
        }
    }

    /// FSRS stability in days (S = half_life / 1.73).
    pub fn stability_days(&self) -> f64 {
        self.half_life_secs() / 86400.0 / 1.73
    }
}

/// Source weight for importance scoring (encoding depth proxy).
pub fn source_weight(source: &str) -> f32 {
    match source {
        "document" | "file" | "code" => 1.0,
        "claude-code" | "mcp" => 0.7,
        "conversation" | "cli" => 0.5,
        "enrichment" | "child" => 0.0,
        _ => 0.3,
    }
}

/// Modality of a stored memory — which sensory channel produced it.
///
/// Text is the default and original modality. Vision (CLIP) and Speech
/// (audio embeddings preserving paralinguistic features) are added for
/// multimodal robotics and social AI use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Modality {
    /// Text content embedded with all-MiniLM-L6-v2 (384-dim).
    #[default]
    Text,
    /// Visual content embedded with CLIP (512-dim). Images and text share the same space.
    Vision,
    /// Audio content embedded with a speech encoder (preserves tone, volume, emotion, pace).
    /// NOT speech-to-text — captures paralinguistic features for social AI.
    Speech,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Vision => write!(f, "vision"),
            Self::Speech => write!(f, "speech"),
        }
    }
}

/// Query mode for echo — which channels to search.
///
/// Controls how `echo_with_mode` routes a query through the embedding pipeline.
/// `Text` (default) preserves backward compatibility. `Vision` enables cross-modal
/// text-to-image retrieval via CLIP. `Auto` merges results from all enabled channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QueryMode {
    /// Search text channel only (default, backward compatible).
    /// Uses all-MiniLM-L6-v2 384-dim embeddings.
    #[default]
    Text,
    /// Search vision channel only (CLIP text-to-image cross-modal).
    /// Embeds query with CLIP text encoder, matches against CLIP image embeddings.
    Vision,
    /// Search all enabled channels, merge results by final_score.
    /// Deduplicates by memory_id, returns top-N across all channels.
    Auto,
}

impl fmt::Display for QueryMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Vision => write!(f, "vision"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Sensitivity classification for stored memories.
///
/// Controls where a memory can be pushed during echo activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SensitivityLevel {
    /// Safe to push to any model (local or cloud).
    #[default]
    Public,
    /// Push only to local models, never to cloud providers.
    Private,
    /// Stored but never pushed to any model. User can view manually.
    Restricted,
    /// Not stored at all — filtered pre-storage (secrets, credentials).
    Blocked,
}

/// A stored memory entry in the Echo Memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: MemoryId,
    /// The original text content of this memory.
    pub content: String,
    /// PII-masked version of the content (if PII was detected).
    pub masked_content: Option<String>,
    /// Reformulated version of the content for better embedding recall.
    /// Structured patterns like "Preference: X for Y" score ~9% higher
    /// on cosine similarity than natural text.
    /// When present, this text is used for embedding instead of the original.
    /// Echo returns the original content to the user (natural text).
    #[serde(default)]
    pub reformulated: Option<String>,
    /// Embedding vector capturing semantic meaning.
    /// Dimension depends on modality: Text=384, Vision=512, Speech=TBD.
    pub embedding: Vec<f32>,
    /// Which sensory channel produced this memory.
    #[serde(default)]
    pub modality: Modality,
    /// CLIP vision embedding (512-dim). Present when image data was stored.
    #[serde(default)]
    pub vision_embedding: Option<Vec<f32>>,
    /// Speech audio embedding (640-dim). Present when audio data was stored.
    #[serde(default)]
    pub speech_embedding: Option<Vec<f32>>,
    /// Where this memory came from (e.g., "conversation", "document", "manual").
    pub source: String,
    /// Sensitivity classification controlling push behavior.
    pub sensitivity: SensitivityLevel,
    /// Category for adaptive decay — different memory types fade at different rates.
    #[serde(default)]
    pub category: MemoryCategory,
    /// When this memory was created.
    pub created_at: DateTime<Utc>,
    /// When this memory was last activated by an echo cycle.
    pub last_echoed: Option<DateTime<Utc>>,
    /// How many times this memory has self-activated.
    pub echo_count: u32,
    /// Whether this memory has been enriched by LLM fact extraction during consolidation.
    /// Enriched memories have child memories (linked via `parent_id`) with extracted facts.
    #[serde(default)]
    pub enriched: bool,
    /// Number of LLM extraction attempts (KS69). Capped at 3 to avoid infinite retries.
    #[serde(default)]
    pub enrichment_attempts: u8,
    /// LLM extraction confidence (KS69). 0.0-1.0, default 0.0 for non-extracted entries.
    #[serde(default)]
    pub confidence: f32,
    /// Primary entity this memory is about (KS69). Populated during LLM extraction.
    #[serde(default)]
    pub subject: Option<String>,
    /// Link to parent memory. Child memories are LLM-extracted facts created during
    /// consolidation. When a child matches during echo, the parent's content is returned.
    #[serde(default)]
    pub parent_id: Option<MemoryId>,
    /// Semantic labels for pre-filtered retrieval (ADR-015).
    /// Each label is a prefixed string (e.g., "topic:language", "entity:rust").
    /// Populated incrementally: Tier 1 at store time, Tier 2 during consolidation.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Label enrichment version.
    /// 0 = unlabeled (legacy or new, pre-classification).
    /// 1 = keyword-only (Tier 1: prototype matching + rules).
    /// 2 = LLM-enriched (Tier 2: NER + Ollama classification).
    #[serde(default)]
    pub label_version: u8,
    /// Knowledge graph triples extracted during consolidation.
    /// Each triple is (subject, predicate, object) — e.g., ("Lior", WorksAt, "Bellkis").
    #[serde(default)]
    pub triples: Vec<Triple>,
    /// Novelty score at store time: 1.0 - max cosine similarity to existing memories.
    /// Higher = more novel/unique. Used for consolidation priority.
    #[serde(default)]
    pub novelty_score: f32,
    /// Multi-signal importance score (0.0-1.0). Recomputed during consolidation.
    #[serde(default)]
    pub importance: f32,
    /// Cached ACT-R base-level activation (OL approximation).
    #[serde(default)]
    pub activation_cache: f32,
    /// When importance was last computed (for staleness detection).
    #[serde(default)]
    pub importance_computed_at: Option<DateTime<Utc>>,
    /// Retrieval timestamps as seconds since UNIX epoch (ring buffer, cap 16). Future: full BLA.
    #[serde(default)]
    pub retrieval_history_secs: Vec<u32>,
}

impl MemoryEntry {
    /// Create a new memory entry with the given content and embedding.
    pub fn new(content: String, embedding: Vec<f32>, source: String) -> Self {
        Self {
            id: MemoryId::new(),
            content,
            masked_content: None,
            reformulated: None,
            embedding,
            modality: Modality::Text,
            vision_embedding: None,
            speech_embedding: None,
            source,
            sensitivity: SensitivityLevel::Public,
            category: MemoryCategory::Default,
            created_at: Utc::now(),
            last_echoed: None,
            echo_count: 0,
            enriched: false,
            enrichment_attempts: 0,
            confidence: 0.0,
            subject: None,
            parent_id: None,
            labels: Vec::new(),
            label_version: 0,
            triples: Vec::new(),
            novelty_score: 0.0,
            importance: 0.0,
            activation_cache: 0.0,
            importance_computed_at: None,
            retrieval_history_secs: Vec::new(),
        }
    }

    /// Create a new memory entry with a specific modality.
    pub fn new_with_modality(
        content: String,
        embedding: Vec<f32>,
        source: String,
        modality: Modality,
    ) -> Self {
        let mut entry = Self::new(content, embedding, source);
        entry.modality = modality;
        entry
    }

    /// Get the content to display — masked if available, original otherwise.
    pub fn display_content(&self) -> &str {
        self.masked_content.as_deref().unwrap_or(&self.content)
    }

    /// Get the content appropriate for a given context (local vs cloud).
    pub fn content_for_provider(&self, is_local: bool) -> Option<&str> {
        match self.sensitivity {
            SensitivityLevel::Public => Some(self.display_content()),
            SensitivityLevel::Private if is_local => Some(&self.content),
            SensitivityLevel::Private => Some(self.display_content()),
            SensitivityLevel::Restricted | SensitivityLevel::Blocked => None,
        }
    }
}

/// Result of an echo query — a memory that self-activated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoResult {
    /// The memory that activated.
    pub memory_id: MemoryId,
    /// Content to display (masked if needed).
    pub content: String,
    /// Raw cosine similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Final composite score after all boosts.
    pub final_score: f64,
    /// Where the memory came from.
    pub source: String,
    /// When this echo activation happened.
    pub echoed_at: DateTime<Utc>,
    /// Which sensory channel matched this result.
    #[serde(default)]
    pub modality: Modality,
    /// Labels on this memory (ADR-015). Exposed for graph navigation.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Content of the child memory that matched the query (KS69 Tier 1).
    /// Present when a child rescued a parent (Pipe B) or appeared directly.
    /// Gives downstream consumers the focused fact text that triggered retrieval.
    #[serde(default)]
    pub matched_child_content: Option<String>,
}

/// A label connection in the memory graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelConnection {
    /// The label string (e.g. "topic:language").
    pub label: String,
    /// How many memories share this label.
    pub count: usize,
    /// Top memory IDs in this label group, ranked by cosine similarity to the source.
    pub top_ids: Vec<MemoryId>,
}

/// Result of a memory graph query — connections from a single memory via labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphResult {
    /// The source memory ID.
    pub memory_id: MemoryId,
    /// Preview of the source memory content.
    pub content_preview: String,
    /// Labels on the source memory.
    pub labels: Vec<String>,
    /// Connections grouped by label.
    pub connections: Vec<LabelConnection>,
    /// Total number of connected memories (deduplicated across labels).
    pub total_connected: usize,
    /// Total unique connected memories.
    pub unique_connected: usize,
}

/// Predicate type for knowledge graph triples.
/// Lightweight enum for core — maps from `hebbian::RelationshipType` at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TriplePredicate {
    WorksAt,
    LivesIn,
    PrefersTool,
    PartOf,
    Custom(String),
}

/// A structured knowledge triple: (subject, predicate, object).
/// Extracted from facts during consolidation for entity-anchored graph queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: TriplePredicate,
    pub object: String,
}

/// A community summary for a label cluster (KS64 — GraphRAG P4).
///
/// Generated during consolidation for label clusters with enough members.
/// Used as a fallback in echo queries when no direct results score well,
/// providing a "global view" answer synthesized from the cluster's contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunitySummary {
    /// The label this summary represents.
    pub label: String,
    /// LLM-generated summary of the cluster's contents.
    pub summary: String,
    /// Embedding of the summary text (for cosine matching during echo fallback).
    pub embedding: Vec<f32>,
    /// Number of memories in this cluster at summarization time.
    pub member_count: usize,
    /// When this summary was last generated/updated.
    pub updated_at: DateTime<Utc>,
}

/// Statistics about the Echo Memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of stored memories.
    pub total_memories: usize,
    /// Approximate size of the in-memory index in bytes.
    pub index_size_bytes: u64,
    /// Approximate total RAM usage in bytes.
    pub ram_usage_bytes: u64,
    /// Maximum capacity based on current config.
    pub max_capacity: usize,
    /// Average echo latency in milliseconds (from recent queries).
    pub avg_echo_latency_ms: f64,
    /// Total echo queries processed.
    pub total_echo_queries: u64,
    /// Current disk usage in bytes (data directory total).
    #[serde(default)]
    pub disk_usage_bytes: u64,
    /// Maximum disk usage allowed in bytes.
    #[serde(default)]
    pub max_disk_bytes: u64,
    /// Number of text-modality memories.
    #[serde(default)]
    pub text_count: usize,
    /// Number of vision-modality memories.
    #[serde(default)]
    pub vision_count: usize,
    /// Number of speech-modality memories.
    #[serde(default)]
    pub speech_count: usize,
}

/// Summary of a memory entry for dump/listing (no embedding data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntrySummary {
    pub id: MemoryId,
    pub content: String,
    pub source: String,
    pub echo_count: u32,
    pub sensitivity: SensitivityLevel,
    pub category: MemoryCategory,
    /// Novelty score at store time (0.0 to 1.0).
    #[serde(default)]
    pub novelty_score: f32,
    /// Multi-signal importance score (0.0-1.0).
    #[serde(default)]
    pub importance: f32,
}

// ---------------------------------------------------------------------------
// Graph visualization types (KS65)
// ---------------------------------------------------------------------------

/// A node in the graph visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: MemoryId,
    pub content_preview: String,
    pub labels: Vec<String>,
    pub importance: f32,
    pub category: String,
    pub novelty: f32,
}

/// Minimal node preview for cluster listings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNodePreview {
    pub id: MemoryId,
    pub content_preview: String,
}

/// A Hebbian neighbor with edge metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNeighbor {
    pub id: MemoryId,
    pub content_preview: String,
    pub labels: Vec<String>,
    pub weight: f64,
    pub relationship: Option<String>,
    pub cosine_similarity: f32,
}

/// Result of graph/neighbors query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNeighborsResult {
    pub node: GraphNode,
    pub neighbors: Vec<GraphNeighbor>,
}

/// An edge in the subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: MemoryId,
    pub target: MemoryId,
    pub weight: f64,
    pub relationship: Option<String>,
}

/// Result of graph/subgraph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSubgraphResult {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// A label cluster for the Galaxy view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCluster {
    pub label: String,
    pub member_count: usize,
    pub summary: Option<String>,
    pub top_members: Vec<GraphNodePreview>,
}

/// An edge between two label clusters (shared members).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphInterEdge {
    pub source_label: String,
    pub target_label: String,
    pub shared_count: usize,
}

/// Result of graph/overview query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOverviewResult {
    pub clusters: Vec<GraphCluster>,
    pub inter_edges: Vec<GraphInterEdge>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_id_unique() {
        let a = MemoryId::new();
        let b = MemoryId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn memory_entry_display_content() {
        let mut entry = MemoryEntry::new(
            "My API key is sk-abc123".into(),
            vec![0.0; 384],
            "test".into(),
        );
        // Without masking — shows original
        assert_eq!(entry.display_content(), "My API key is sk-abc123");

        // With masking — shows masked
        entry.masked_content = Some("My API key is [MASKED:api_key]".into());
        assert_eq!(entry.display_content(), "My API key is [MASKED:api_key]");
    }

    #[test]
    fn content_for_provider_respects_sensitivity() {
        let mut entry = MemoryEntry::new("secret data".into(), vec![], "test".into());

        // Public — visible everywhere
        entry.sensitivity = SensitivityLevel::Public;
        assert!(entry.content_for_provider(true).is_some());
        assert!(entry.content_for_provider(false).is_some());

        // Private — visible local, masked for cloud
        entry.sensitivity = SensitivityLevel::Private;
        assert!(entry.content_for_provider(true).is_some());
        assert!(entry.content_for_provider(false).is_some());

        // Restricted — never pushed
        entry.sensitivity = SensitivityLevel::Restricted;
        assert!(entry.content_for_provider(true).is_none());
        assert!(entry.content_for_provider(false).is_none());
    }

    #[test]
    fn memory_entry_serializes() {
        let entry = MemoryEntry::new("test".into(), vec![1.0, 2.0, 3.0], "test".into());
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content, "test");
        assert_eq!(deserialized.embedding.len(), 3);
    }

    #[test]
    fn memory_entry_default_category() {
        let entry = MemoryEntry::new("test".into(), vec![], "test".into());
        assert_eq!(entry.category, MemoryCategory::Default);
    }

    // --- MemoryCategory half-life tests ---

    #[test]
    fn category_half_life_active_project() {
        assert_eq!(
            MemoryCategory::ActiveProject.half_life_secs(),
            14.0 * 86400.0
        );
    }

    #[test]
    fn category_half_life_preference() {
        assert_eq!(MemoryCategory::Preference.half_life_secs(), 60.0 * 86400.0);
    }

    #[test]
    fn category_half_life_conversation() {
        assert_eq!(MemoryCategory::Conversation.half_life_secs(), 3.0 * 86400.0);
    }

    #[test]
    fn category_half_life_identity() {
        assert_eq!(MemoryCategory::Identity.half_life_secs(), 365.0 * 86400.0);
    }

    #[test]
    fn category_half_life_fact() {
        assert_eq!(MemoryCategory::Fact.half_life_secs(), 30.0 * 86400.0);
    }

    #[test]
    fn category_half_life_default() {
        assert_eq!(MemoryCategory::Default.half_life_secs(), 7.0 * 86400.0);
    }

    #[test]
    fn category_default_is_default() {
        assert_eq!(MemoryCategory::default(), MemoryCategory::Default);
    }

    // --- MemoryCategory serialization roundtrip ---

    #[test]
    fn category_serialization_roundtrip() {
        let categories = [
            MemoryCategory::ActiveProject,
            MemoryCategory::Preference,
            MemoryCategory::Conversation,
            MemoryCategory::Identity,
            MemoryCategory::Fact,
            MemoryCategory::Default,
        ];
        for cat in &categories {
            let json = serde_json::to_string(cat).unwrap();
            let deserialized: MemoryCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(*cat, deserialized, "Roundtrip failed for {cat:?}");
        }
    }

    #[test]
    fn category_preserved_in_memory_entry_serialization() {
        let mut entry = MemoryEntry::new("test".into(), vec![1.0], "test".into());
        entry.category = MemoryCategory::Identity;

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.category, MemoryCategory::Identity);
    }

    #[test]
    fn category_defaults_on_missing_field() {
        // Simulates deserializing a legacy MemoryEntry without the category field.
        // The #[serde(default)] attribute should fill in MemoryCategory::Default.
        let json = r#"{
            "id":"00000000-0000-0000-0000-000000000000",
            "content":"legacy","masked_content":null,"reformulated":null,
            "embedding":[],"source":"test",
            "sensitivity":"Public",
            "created_at":"2025-01-01T00:00:00Z",
            "last_echoed":null,"echo_count":0
        }"#;
        let entry: MemoryEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.category, MemoryCategory::Default);
    }

    // --- QueryMode tests (KS35) ---

    #[test]
    fn query_mode_default_is_text() {
        assert_eq!(QueryMode::default(), QueryMode::Text);
    }

    #[test]
    fn query_mode_serde_roundtrip() {
        let modes = [QueryMode::Text, QueryMode::Vision, QueryMode::Auto];
        for mode in &modes {
            let json = serde_json::to_string(mode).unwrap();
            let deserialized: QueryMode = serde_json::from_str(&json).unwrap();
            assert_eq!(*mode, deserialized, "Roundtrip failed for {mode:?}");
        }
    }

    #[test]
    fn query_mode_display() {
        assert_eq!(QueryMode::Text.to_string(), "text");
        assert_eq!(QueryMode::Vision.to_string(), "vision");
        assert_eq!(QueryMode::Auto.to_string(), "auto");
    }

    // --- Label fields (KS42, ADR-015) ---

    #[test]
    fn label_fields_default_on_new() {
        let entry = MemoryEntry::new("test".into(), vec![1.0], "test".into());
        assert!(entry.labels.is_empty());
        assert_eq!(entry.label_version, 0);
    }

    #[test]
    fn label_fields_serde_roundtrip() {
        let mut entry = MemoryEntry::new("test".into(), vec![1.0], "test".into());
        entry.labels = vec![
            "topic:language".into(),
            "entity:rust".into(),
            "domain:work".into(),
        ];
        entry.label_version = 2;

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.labels,
            vec!["topic:language", "entity:rust", "domain:work"]
        );
        assert_eq!(deserialized.label_version, 2);
    }

    #[test]
    fn label_fields_default_on_legacy_json() {
        // Simulates a pre-label MemoryEntry without labels/label_version fields
        let json = r#"{
            "id":"00000000-0000-0000-0000-000000000000",
            "content":"legacy","masked_content":null,"reformulated":null,
            "embedding":[],"source":"test",
            "sensitivity":"Public",
            "created_at":"2025-01-01T00:00:00Z",
            "last_echoed":null,"echo_count":0
        }"#;
        let entry: MemoryEntry = serde_json::from_str(json).unwrap();
        assert!(
            entry.labels.is_empty(),
            "Legacy JSON should deserialize to empty labels"
        );
        assert_eq!(
            entry.label_version, 0,
            "Legacy JSON should deserialize to label_version 0"
        );
    }

    #[test]
    fn label_fields_empty_labels_roundtrip() {
        let entry = MemoryEntry::new("test".into(), vec![], "test".into());
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert!(deserialized.labels.is_empty());
        assert_eq!(deserialized.label_version, 0);
    }

    #[test]
    fn label_fields_max_labels() {
        let mut entry = MemoryEntry::new("test".into(), vec![], "test".into());
        entry.labels = (0..10).map(|i| format!("topic:label{i}")).collect();
        entry.label_version = 1;

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.labels.len(), 10);
    }

    // --- Triple / entity graph tests (KS61) ---

    #[test]
    fn triple_serde_roundtrip() {
        let triple = Triple {
            subject: "Lior".into(),
            predicate: TriplePredicate::WorksAt,
            object: "Bellkis AI".into(),
        };
        let json = serde_json::to_string(&triple).unwrap();
        let back: Triple = serde_json::from_str(&json).unwrap();
        assert_eq!(back.subject, "Lior");
        assert_eq!(back.predicate, TriplePredicate::WorksAt);
        assert_eq!(back.object, "Bellkis AI");
    }

    #[test]
    fn triple_defaults_on_legacy_json() {
        // MemoryEntry JSON without triples field should deserialize with empty triples vec
        let json = r#"{"id":"00000000-0000-0000-0000-000000000000","content":"test","masked_content":null,"embedding":[],"source":"test","sensitivity":"Public","created_at":"2024-01-01T00:00:00Z","last_echoed":null,"echo_count":0}"#;
        let entry: MemoryEntry = serde_json::from_str(json).unwrap();
        assert!(entry.triples.is_empty());
    }

    #[test]
    fn memory_entry_with_vision_embedding_roundtrip() {
        let mut entry = MemoryEntry::new_with_modality(
            "[image]".into(),
            Vec::new(),
            "test".into(),
            Modality::Vision,
        );
        entry.vision_embedding = Some(vec![0.1, 0.2, 0.3, 0.4]);

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.modality, Modality::Vision);
        assert_eq!(deserialized.content, "[image]");
        assert!(deserialized.embedding.is_empty());
        assert_eq!(
            deserialized.vision_embedding,
            Some(vec![0.1, 0.2, 0.3, 0.4])
        );
    }
}
