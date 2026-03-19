//! In-memory vector store with JSON persistence.
//!
//! Phase 1: Vec-based storage with parallel embedding array for fast similarity search.
//! Phase 2 will add mmap-backed binary persistence for zero-copy reads.

use bellkis_core::{BellkisError, MemoryEntry, MemoryId, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::instrument;

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

    /// Serialize the store to a JSON file.
    ///
    /// Phase 1 uses JSON for simplicity and debuggability.
    /// Phase 2 will use a binary format (mmap + rkyv) for performance.
    #[instrument(skip(self), fields(entries = self.entries.len()))]
    pub fn save(&self, path: &Path) -> Result<()> {
        let start = std::time::Instant::now();

        // Ensure parent directory exists
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
            "Store saved to disk"
        );

        Ok(())
    }

    /// Load a store from a JSON file.
    ///
    /// Rebuilds the parallel embedding array and index map from deserialized entries.
    #[instrument(fields(path = %path.display()))]
    pub fn load(path: &Path) -> Result<Self> {
        let start = std::time::Instant::now();

        if !path.exists() {
            tracing::info!(path = %path.display(), "No store file found, starting empty");
            return Ok(Self::new());
        }

        let json = std::fs::read_to_string(path)
            .map_err(|e| BellkisError::Persistence(format!("Failed to read store: {e}")))?;

        let entries: Vec<MemoryEntry> = serde_json::from_str(&json)
            .map_err(|e| BellkisError::Persistence(format!("Failed to parse store: {e}")))?;

        let mut store = Self::new();
        for entry in entries {
            store.add(entry);
        }

        let elapsed = start.elapsed();
        tracing::info!(
            entries = store.len(),
            path = %path.display(),
            elapsed_ms = elapsed.as_millis(),
            "Store loaded from disk"
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
        let path = dir.path().join("test_store.json");

        // Create and save
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
    fn load_nonexistent_returns_empty() {
        let path = Path::new("/tmp/nonexistent_store_12345.json");
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
