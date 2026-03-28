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
    /// Speech audio embedding (579-dim). Present when audio data was stored.
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
    /// Link to parent memory. Child memories are LLM-extracted facts created during
    /// consolidation. When a child matches during echo, the parent's content is returned.
    #[serde(default)]
    pub parent_id: Option<MemoryId>,
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
            parent_id: None,
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
}
