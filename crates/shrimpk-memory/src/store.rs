//! In-memory vector store with binary + JSON persistence.
//!
//! Primary format: binary (`.shrm`) via `crate::persistence` — structured header,
//! flat embedding array, CRC32 checksum. Falls back to JSON for migration.

use shrimpk_core::{MemoryEntry, MemoryId, Result, ShrimPKError};
use std::collections::HashMap;
use std::path::Path;
use tracing::instrument;

use crate::persistence;

/// In-memory vector store for Echo Memory.
///
/// Maintains a parallel array of embeddings alongside memory entries
/// for efficient brute-force similarity search. The `id_to_index` map
/// provides O(1) lookup by MemoryId.
///
/// # Invariants
/// - `entries.len() == embeddings.len()` always
/// - Every entry's ID exists in `id_to_index`
/// - `id_to_index[id] < entries.len()` for all stored IDs
#[derive(Debug)]
pub struct EchoStore {
    /// All stored memory entries.
    entries: Vec<MemoryEntry>,
    /// Parallel array of embedding vectors (same index as entries).
    embeddings: Vec<Vec<f32>>,
    /// Quick lookup: MemoryId -> index in entries/embeddings.
    id_to_index: HashMap<MemoryId, usize>,
}

impl EchoStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            embeddings: Vec::new(),
            id_to_index: HashMap::new(),
        }
    }

    /// Add a memory entry to the store.
    ///
    /// The embedding is extracted from the entry and stored in the parallel array.
    /// Returns the index where the entry was stored.
    pub fn add(&mut self, entry: MemoryEntry) -> usize {
        let index = self.entries.len();
        let embedding = entry.embedding.clone();
        self.id_to_index.insert(entry.id.clone(), index);
        self.entries.push(entry);
        self.embeddings.push(embedding);
        index
    }

    /// Remove a memory entry by its ID.
    ///
    /// Uses swap-remove for O(1) removal, then fixes up the index map.
    /// Returns the removed entry if found.
    pub fn remove(&mut self, id: &MemoryId) -> Option<MemoryEntry> {
        let index = self.id_to_index.remove(id)?;
        let last_index = self.entries.len() - 1;

        if index != last_index {
            // Swap the last entry into this slot
            let moved_id = self.entries[last_index].id.clone();
            self.id_to_index.insert(moved_id, index);
        }

        self.embeddings.swap_remove(index);
        Some(self.entries.swap_remove(index))
    }

    /// Get a reference to a memory entry by its ID.
    pub fn get(&self, id: &MemoryId) -> Option<&MemoryEntry> {
        let &index = self.id_to_index.get(id)?;
        self.entries.get(index)
    }

    /// Get a mutable reference to a memory entry by its ID.
    pub fn get_mut(&mut self, id: &MemoryId) -> Option<&mut MemoryEntry> {
        let &index = self.id_to_index.get(id)?;
        self.entries.get_mut(index)
    }

    /// Get the internal index for a MemoryId.
    ///
    /// Used by the LSH index to track swap-remove index changes.
    pub fn index_of(&self, id: &MemoryId) -> Option<usize> {
        self.id_to_index.get(id).copied()
    }

    /// Number of stored memories.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the parallel embedding array for similarity search.
    pub fn all_embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Get all entries (for serialization or iteration).
    pub fn all_entries(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// Get the entry at a specific index (used after similarity search).
    pub fn entry_at(&self, index: usize) -> Option<&MemoryEntry> {
        self.entries.get(index)
    }

    /// Get a mutable entry at a specific index (for updating echo_count, last_echoed).
    pub fn entry_at_mut(&mut self, index: usize) -> Option<&mut MemoryEntry> {
        self.entries.get_mut(index)
    }

    /// Save the store to a binary file.
    ///
    /// Uses the SHRM binary format: 64-byte header + flat f32 embedding array
    /// + CRC32 checksum + JSON metadata. See `crate::persistence` for format details.
    #[instrument(skip(self), fields(entries = self.entries.len()))]
    pub fn save(&self, path: &Path) -> Result<()> {
        persistence::save_binary(self, path)
    }

    /// Load a store from disk.
    ///
    /// Tries binary format first. If the binary file does not exist, falls back
    /// to JSON for migration from the old format. Returns empty store if neither exists.
    #[instrument(fields(path = %path.display()))]
    pub fn load(path: &Path) -> Result<Self> {
        // Try binary first
        if path.exists() {
            // Check if it's a valid binary file by looking at magic bytes
            if let Ok(data) = std::fs::read(path)
                && data.len() >= 4
                && &data[0..4] == b"SHRM"
            {
                return persistence::load_binary(path);
            }
            // Not binary — try JSON fallback
            tracing::info!(
                path = %path.display(),
                "Binary magic not found, attempting JSON fallback"
            );
            return Self::load_json(path);
        }

        // Check for a sibling .json file (migration scenario: path is .shrm but .json exists)
        let json_path = path.with_extension("json");
        if json_path.exists() {
            tracing::info!(
                json_path = %json_path.display(),
                "Found legacy JSON store, migrating"
            );
            return Self::load_json(&json_path);
        }

        tracing::info!(path = %path.display(), "No store file found, starting empty");
        Ok(Self::new())
    }

    /// Save the store to a JSON file (legacy format, for backward compatibility).
    #[instrument(skip(self), fields(entries = self.entries.len()))]
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let start = std::time::Instant::now();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string(&self.entries)?;
        std::fs::write(path, json)?;

        let elapsed = start.elapsed();
        tracing::info!(
            entries = self.entries.len(),
            path = %path.display(),
            elapsed_ms = elapsed.as_millis(),
            "Store saved to JSON (legacy)"
        );

        Ok(())
    }

    /// Load a store from a JSON file (legacy format, for backward compatibility).
    #[instrument(fields(path = %path.display()))]
    pub fn load_json(path: &Path) -> Result<Self> {
        let start = std::time::Instant::now();

        if !path.exists() {
            tracing::info!(path = %path.display(), "No JSON store file found, starting empty");
            return Ok(Self::new());
        }

        let json = std::fs::read_to_string(path)
            .map_err(|e| ShrimPKError::Persistence(format!("Failed to read store: {e}")))?;

        let entries: Vec<MemoryEntry> = serde_json::from_str(&json)
            .map_err(|e| ShrimPKError::Persistence(format!("Failed to parse store: {e}")))?;

        let mut store = Self::new();
        for entry in entries {
            store.add(entry);
        }

        let elapsed = start.elapsed();
        tracing::info!(
            entries = store.len(),
            path = %path.display(),
            elapsed_ms = elapsed.as_millis(),
            "Store loaded from JSON (legacy)"
        );

        Ok(store)
    }
}

impl Default for EchoStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(content: &str) -> MemoryEntry {
        MemoryEntry::new(content.to_string(), vec![1.0, 2.0, 3.0], "test".to_string())
    }

    #[test]
    fn add_and_get() {
        let mut store = EchoStore::new();
        let entry = make_entry("hello world");
        let id = entry.id.clone();
        let idx = store.add(entry);

        assert_eq!(idx, 0);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        let retrieved = store.get(&id).expect("should find entry");
        assert_eq!(retrieved.content, "hello world");
    }

    #[test]
    fn remove_entry() {
        let mut store = EchoStore::new();
        let e1 = make_entry("first");
        let e2 = make_entry("second");
        let e3 = make_entry("third");
        let id1 = e1.id.clone();
        let id2 = e2.id.clone();
        let id3 = e3.id.clone();

        store.add(e1);
        store.add(e2);
        store.add(e3);
        assert_eq!(store.len(), 3);

        // Remove middle entry
        let removed = store.remove(&id2);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().content, "second");
        assert_eq!(store.len(), 2);

        // Other entries still accessible
        assert!(store.get(&id1).is_some());
        assert!(store.get(&id3).is_some());
        assert!(store.get(&id2).is_none());
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut store = EchoStore::new();
        let result = store.remove(&MemoryId::new());
        assert!(result.is_none());
    }

    #[test]
    fn embeddings_parallel_array() {
        let mut store = EchoStore::new();
        store.add(MemoryEntry::new("a".into(), vec![1.0, 0.0], "test".into()));
        store.add(MemoryEntry::new("b".into(), vec![0.0, 1.0], "test".into()));

        let embeddings = store.all_embeddings();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], vec![1.0, 0.0]);
        assert_eq!(embeddings[1], vec![0.0, 1.0]);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_store.shrm");

        // Create and save (now uses binary format)
        let mut store = EchoStore::new();
        let entry = make_entry("persistent memory");
        let id = entry.id.clone();
        store.add(entry);
        store.save(&path).expect("Should save");

        // Load
        let loaded = EchoStore::load(&path).expect("Should load");
        assert_eq!(loaded.len(), 1);
        let retrieved = loaded.get(&id).expect("should find entry");
        assert_eq!(retrieved.content, "persistent memory");
        assert_eq!(retrieved.embedding, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn save_json_and_load_json_roundtrip() {
        let dir = tempfile::tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_store.json");

        let mut store = EchoStore::new();
        let entry = make_entry("json legacy");
        let id = entry.id.clone();
        store.add(entry);
        store.save_json(&path).expect("Should save JSON");

        let loaded = EchoStore::load_json(&path).expect("Should load JSON");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get(&id).unwrap().content, "json legacy");
    }

    #[test]
    fn load_falls_back_to_json() {
        // When a file exists but has no SHRM magic, load() should fall back to JSON
        let dir = tempfile::tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_store.shrm");

        let mut store = EchoStore::new();
        let entry = make_entry("json fallback test");
        let id = entry.id.clone();
        store.add(entry);

        // Save as JSON to the .shrm path (simulating legacy file)
        store.save_json(&path).expect("Should save JSON");

        // load() should detect non-binary and fall back to JSON
        let loaded = EchoStore::load(&path).expect("Should load via fallback");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get(&id).unwrap().content, "json fallback test");
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let path = Path::new("/tmp/nonexistent_store_12345.shrm");
        let store = EchoStore::load(path).expect("Should return empty store");
        assert!(store.is_empty());
    }

    #[test]
    fn entry_at_index() {
        let mut store = EchoStore::new();
        store.add(make_entry("zero"));
        store.add(make_entry("one"));

        assert_eq!(store.entry_at(0).unwrap().content, "zero");
        assert_eq!(store.entry_at(1).unwrap().content, "one");
        assert!(store.entry_at(99).is_none());
    }

    #[test]
    fn swap_remove_fixes_indices() {
        let mut store = EchoStore::new();
        let e1 = make_entry("first");
        let e2 = make_entry("second");
        let e3 = make_entry("third");
        let id1 = e1.id.clone();
        let id3 = e3.id.clone();

        store.add(e1);
        store.add(e2);
        store.add(e3);

        // Remove first entry — "third" gets swapped to index 0
        store.remove(&id1);
        assert_eq!(store.len(), 2);

        // "third" should now be at index 0
        let third = store.get(&id3).expect("should find third");
        assert_eq!(third.content, "third");

        // Verify embeddings are consistent
        assert_eq!(store.all_embeddings().len(), 2);
    }
}
