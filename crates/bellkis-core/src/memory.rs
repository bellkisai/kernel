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

/// Sensitivity classification for stored memories.
///
/// Controls where a memory can be pushed during echo activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityLevel {
    /// Safe to push to any model (local or cloud).
    Public,
    /// Push only to local models, never to cloud providers.
    Private,
    /// Stored but never pushed to any model. User can view manually.
    Restricted,
    /// Not stored at all — filtered pre-storage (secrets, credentials).
    Blocked,
}

impl Default for SensitivityLevel {
    fn default() -> Self {
        Self::Public
    }
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
    /// 384-dimensional embedding vector capturing semantic meaning.
    pub embedding: Vec<f32>,
    /// Where this memory came from (e.g., "conversation", "document", "manual").
    pub source: String,
    /// Sensitivity classification controlling push behavior.
    pub sensitivity: SensitivityLevel,
    /// When this memory was created.
    pub created_at: DateTime<Utc>,
    /// When this memory was last activated by an echo cycle.
    pub last_echoed: Option<DateTime<Utc>>,
    /// How many times this memory has self-activated.
    pub echo_count: u32,
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
            source,
            sensitivity: SensitivityLevel::Public,
            created_at: Utc::now(),
            last_echoed: None,
            echo_count: 0,
        }
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
}
