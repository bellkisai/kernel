//! Entity types for entity-unified memory (KS73).
//!
//! An `EntityFrame` is a persistent identity that links multiple memories
//! about the same real-world entity (person, project, tool, etc.).
//! `EntityId` is deterministic from the canonical name via UUID v5.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::MemoryId;

/// Namespace UUID for deterministic entity IDs.
const ENTITY_NS: Uuid = Uuid::from_bytes([
    0x53, 0x68, 0x72, 0x69, 0x6d, 0x50, 0x4b, 0x45, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x4e, 0x53, 0x31,
]); // "ShrimPKEntityNS1"

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct EntityId(pub Uuid);

impl EntityId {
    /// Deterministic ID from canonical name — same name always produces same ID.
    pub fn from_name(name: &str) -> Self {
        Self(Uuid::new_v5(
            &ENTITY_NS,
            name.to_lowercase().trim().as_bytes(),
        ))
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFrame {
    pub id: EntityId,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub kind: EntityKind,
    pub attributes: HashMap<String, String>,
    pub memory_refs: Vec<MemoryId>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl EntityFrame {
    pub fn new(name: String, kind: EntityKind) -> Self {
        let id = EntityId::from_name(&name);
        Self {
            id,
            aliases: vec![name.clone()],
            canonical_name: name,
            kind,
            attributes: HashMap::new(),
            memory_refs: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EntityKind {
    Person,
    Organization,
    Project,
    Tool,
    Place,
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_deterministic() {
        let a = EntityId::from_name("Lior");
        let b = EntityId::from_name("Lior");
        assert_eq!(a, b);
    }

    #[test]
    fn entity_id_case_insensitive() {
        let a = EntityId::from_name("Lior");
        let b = EntityId::from_name("lior");
        assert_eq!(a, b);
    }

    #[test]
    fn entity_id_trims_whitespace() {
        let a = EntityId::from_name("Lior");
        let b = EntityId::from_name("  Lior  ");
        assert_eq!(a, b);
    }

    #[test]
    fn entity_id_different_names_differ() {
        let a = EntityId::from_name("Lior");
        let b = EntityId::from_name("Alice");
        assert_ne!(a, b);
    }

    #[test]
    fn entity_frame_new_seeds_aliases() {
        let frame = EntityFrame::new("Rust".to_string(), EntityKind::Tool);
        assert_eq!(frame.canonical_name, "Rust");
        assert_eq!(frame.aliases, vec!["Rust"]);
        assert_eq!(frame.kind, EntityKind::Tool);
        assert!(frame.memory_refs.is_empty());
        assert_eq!(frame.id, EntityId::from_name("Rust"));
    }

    #[test]
    fn entity_frame_serde_roundtrip() {
        let frame = EntityFrame::new("Bellkis AI".to_string(), EntityKind::Organization);
        let json = serde_json::to_string(&frame).unwrap();
        let back: EntityFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back.canonical_name, "Bellkis AI");
        assert_eq!(back.kind, EntityKind::Organization);
        assert_eq!(back.id, frame.id);
    }

    #[test]
    fn entity_kind_serde_roundtrip() {
        let kinds = [
            EntityKind::Person,
            EntityKind::Organization,
            EntityKind::Project,
            EntityKind::Tool,
            EntityKind::Place,
            EntityKind::Other,
        ];
        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let back: EntityKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }
}
