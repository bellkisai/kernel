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
    /// Reverse index: parent MemoryId -> child entry indices.
    /// Maintained incrementally on add/remove. Enables O(1) child lookup for Pipe B.
    parent_children: HashMap<MemoryId, Vec<usize>>,
    /// Inverted index: label string -> sorted store indices (ADR-015 D3).
    /// Maintained incrementally on add/remove. Enables O(1) label-based pre-filtering.
    label_index: HashMap<String, Vec<u32>>,
    /// Entity name -> store indices. Populated from MemoryEntry.triples during add/load.
    /// Normalized to lowercase for case-insensitive lookup.
    entity_index: HashMap<String, Vec<u32>>,
}

impl EchoStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            embeddings: Vec::new(),
            id_to_index: HashMap::new(),
            parent_children: HashMap::new(),
            label_index: HashMap::new(),
            entity_index: HashMap::new(),
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
        // Maintain parent-children index
        if let Some(ref pid) = entry.parent_id {
            self.parent_children
                .entry(pid.clone())
                .or_default()
                .push(index);
        }
        // Maintain label inverted index (ADR-015 D3)
        for label in &entry.labels {
            self.label_index
                .entry(label.clone())
                .or_default()
                .push(index as u32);
        }
        // Maintain entity index from triples (KS61)
        for triple in &entry.triples {
            let subj = triple.subject.to_lowercase();
            let obj = triple.object.to_lowercase();
            self.entity_index.entry(subj).or_default().push(index as u32);
            self.entity_index.entry(obj).or_default().push(index as u32);
        }
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

        // Clean parent-children index before swap_remove
        if let Some(ref pid) = self.entries[index].parent_id {
            if let Some(children) = self.parent_children.get_mut(pid) {
                children.retain(|&i| i != index);
                if children.is_empty() {
                    self.parent_children.remove(pid);
                }
            }
        }
        self.parent_children.remove(&self.entries[index].id);

        // Clean label index for the removed entry
        for label in &self.entries[index].labels {
            if let Some(posting) = self.label_index.get_mut(label) {
                posting.retain(|&i| i != index as u32);
                if posting.is_empty() {
                    self.label_index.remove(label);
                }
            }
        }
        // Clean entity index for the removed entry (KS61)
        for triple in &self.entries[index].triples {
            let subj = triple.subject.to_lowercase();
            let obj = triple.object.to_lowercase();
            for key in [subj, obj] {
                if let Some(posting) = self.entity_index.get_mut(&key) {
                    posting.retain(|&i| i != index as u32);
                    if posting.is_empty() {
                        self.entity_index.remove(&key);
                    }
                }
            }
        }

        if index != last_index {
            // Swap the last entry into this slot
            let moved_id = self.entries[last_index].id.clone();
            self.id_to_index.insert(moved_id, index);
            // Fix parent-children references from last_index to index
            for children in self.parent_children.values_mut() {
                for idx in children.iter_mut() {
                    if *idx == last_index {
                        *idx = index;
                    }
                }
            }
            // Fix label index references from last_index to index
            for label in &self.entries[last_index].labels {
                if let Some(posting) = self.label_index.get_mut(label) {
                    for idx in posting.iter_mut() {
                        if *idx == last_index as u32 {
                            *idx = index as u32;
                        }
                    }
                }
            }
            // Fix entity index references from last_index to index (KS61)
            for triple in &self.entries[last_index].triples {
                let subj = triple.subject.to_lowercase();
                let obj = triple.object.to_lowercase();
                for key in [subj, obj] {
                    if let Some(posting) = self.entity_index.get_mut(&key) {
                        for idx in posting.iter_mut() {
                            if *idx == last_index as u32 {
                                *idx = index as u32;
                            }
                        }
                    }
                }
            }
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

    /// Get the indices of all child memories for a given parent.
    ///
    /// Used by Pipe B in echo() to check if near-miss parents have
    /// enriched children that score better.
    pub fn children_of(&self, parent_id: &MemoryId) -> &[usize] {
        self.parent_children
            .get(parent_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Whether any memories have been enriched (have children).
    ///
    /// Used to short-circuit Pipe B entirely when no enrichment has occurred.
    pub fn has_enriched_memories(&self) -> bool {
        !self.parent_children.is_empty()
    }

    // --- Label index API (ADR-015 D3) ---

    /// Query the label index with OR semantics.
    ///
    /// Returns a deduplicated, sorted vector of store indices matching ANY of the
    /// provided labels. If labels is empty, returns empty.
    pub fn query_labels(&self, labels: &[String]) -> Vec<u32> {
        if labels.is_empty() {
            return Vec::new();
        }
        let mut result: Vec<u32> = Vec::new();
        for label in labels {
            if let Some(posting) = self.label_index.get(label) {
                result.extend_from_slice(posting);
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Rebuild the label index from scratch by scanning all entries.
    ///
    /// Called during load() to reconstruct the derived index.
    /// O(N * L) where N = entries, L = average labels per entry.
    pub fn rebuild_label_index(&mut self) {
        self.label_index.clear();
        for (idx, entry) in self.entries.iter().enumerate() {
            for label in &entry.labels {
                self.label_index
                    .entry(label.clone())
                    .or_default()
                    .push(idx as u32);
            }
        }
    }

    /// Number of unique labels in the index.
    pub fn label_count(&self) -> usize {
        self.label_index.len()
    }

    /// Get the labels for a specific entry by index.
    pub fn labels_for(&self, index: usize) -> &[String] {
        self.entries
            .get(index)
            .map(|e| e.labels.as_slice())
            .unwrap_or(&[])
    }

    /// Get mutable access to the label index (for consolidation Pass 4).
    pub fn label_index_mut(&mut self) -> &mut HashMap<String, Vec<u32>> {
        &mut self.label_index
    }

    /// Get the posting list size for a specific label.
    pub fn label_posting_len(&self, label: &str) -> usize {
        self.label_index.get(label).map_or(0, |v| v.len())
    }

    // --- Entity index API (KS61) ---

    /// Look up memories mentioning a specific entity (case-insensitive).
    pub fn query_entities(&self, entity: &str) -> Vec<u32> {
        let key = entity.to_lowercase();
        self.entity_index.get(&key).cloned().unwrap_or_default()
    }

    /// Rebuild entity index from all entries' triples (called after load).
    pub fn rebuild_entity_index(&mut self) {
        self.entity_index.clear();
        for (idx, entry) in self.entries.iter().enumerate() {
            for triple in &entry.triples {
                let subj = triple.subject.to_lowercase();
                let obj = triple.object.to_lowercase();
                self.entity_index.entry(subj).or_default().push(idx as u32);
                self.entity_index.entry(obj).or_default().push(idx as u32);
            }
        }
    }

    /// Read-only reference to the entity index (for query-time entity detection).
    pub fn entity_index_ref(&self) -> &HashMap<String, Vec<u32>> {
        &self.entity_index
    }

    /// Get the embedding vector at a specific index.
    pub fn embedding_at(&self, index: usize) -> Option<&[f32]> {
        self.embeddings.get(index).map(|v| v.as_slice())
    }

    /// Return all store indices that share at least one label with the given entry,
    /// grouped by label. Excludes the source entry itself.
    ///
    /// Used by `memory_graph` to show connections per label dimension.
    pub fn connected_by_labels(&self, index: usize) -> HashMap<String, Vec<u32>> {
        let mut result: HashMap<String, Vec<u32>> = HashMap::new();
        let labels = self.labels_for(index);
        let self_idx = index as u32;
        for label in labels {
            if let Some(posting) = self.label_index.get(label) {
                let filtered: Vec<u32> = posting
                    .iter()
                    .copied()
                    .filter(|&idx| idx != self_idx)
                    .collect();
                if !filtered.is_empty() {
                    result.insert(label.clone(), filtered);
                }
            }
        }
        result
    }

    /// Return all unique store indices connected to the given entry via labels,
    /// optionally filtered to a specific label.
    ///
    /// Used by `memory_related` for the cosine-only fast path.
    pub fn connected_indices(&self, index: usize, label_filter: Option<&str>) -> Vec<u32> {
        let self_idx = index as u32;
        let mut result: Vec<u32> = Vec::new();

        match label_filter {
            Some(lbl) => {
                // Single label lookup
                if let Some(posting) = self.label_index.get(lbl) {
                    result.extend(posting.iter().copied().filter(|&idx| idx != self_idx));
                }
            }
            None => {
                // All labels on this entry
                for label in self.labels_for(index) {
                    if let Some(posting) = self.label_index.get(label) {
                        result.extend(posting.iter().copied().filter(|&idx| idx != self_idx));
                    }
                }
                result.sort_unstable();
                result.dedup();
            }
        }
        result
    }

    /// One-time bootstrap: apply Tier 1 labels to all unlabeled entries (ADR-015 D7).
    ///
    /// Called after load() when entries with label_version == 0 are detected.
    /// Uses prototype cosine matching + rule-based classification.
    /// Pure Rust, no LLM — processes 100K entries in seconds.
    pub fn bootstrap_tier1_labels(&mut self, prototypes: &crate::labels::LabelPrototypes) -> usize {
        let mut updated = 0;
        for idx in 0..self.entries.len() {
            if self.entries[idx].label_version == 0 {
                let labels = crate::labels::generate_tier1_labels(
                    &self.entries[idx].content,
                    &self.embeddings[idx],
                    prototypes,
                );
                if !labels.is_empty() {
                    // Update label index
                    for label in &labels {
                        self.label_index
                            .entry(label.clone())
                            .or_default()
                            .push(idx as u32);
                    }
                    self.entries[idx].labels = labels;
                    self.entries[idx].label_version = 1;
                    updated += 1;
                }
            }
        }
        tracing::info!(
            updated,
            total = self.entries.len(),
            "Tier 1 label bootstrap complete"
        );
        updated
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

    // --- Label index tests (KS43, ADR-015 D3) ---

    fn make_labeled_entry(content: &str, labels: &[&str]) -> MemoryEntry {
        let mut entry = make_entry(content);
        entry.labels = labels.iter().map(|s| s.to_string()).collect();
        entry.label_version = 1;
        entry
    }

    #[test]
    fn label_index_add_indexes_labels() {
        let mut store = EchoStore::new();
        store.add(make_labeled_entry(
            "learning japanese",
            &["topic:language", "action:learning"],
        ));

        let results = store.query_labels(&["topic:language".into()]);
        assert_eq!(
            results,
            vec![0],
            "Entry 0 should be in topic:language posting list"
        );

        let results = store.query_labels(&["action:learning".into()]);
        assert_eq!(
            results,
            vec![0],
            "Entry 0 should be in action:learning posting list"
        );

        assert_eq!(store.label_count(), 2);
    }

    #[test]
    fn label_index_remove_cleans_index() {
        let mut store = EchoStore::new();
        let entry = make_labeled_entry("temp", &["topic:temp"]);
        let id = entry.id.clone();
        store.add(entry);

        assert_eq!(store.label_posting_len("topic:temp"), 1);
        store.remove(&id);
        assert_eq!(store.label_posting_len("topic:temp"), 0);
        assert_eq!(
            store.label_count(),
            0,
            "Empty posting list should be removed"
        );
    }

    #[test]
    fn label_index_swap_remove_fixup() {
        let mut store = EchoStore::new();
        let e0 = make_labeled_entry("first", &["shared:label", "only:first"]);
        let e1 = make_labeled_entry("second", &["shared:label", "only:second"]);
        let e2 = make_labeled_entry("third", &["shared:label", "only:third"]);
        let id0 = e0.id.clone();
        let id2 = e2.id.clone();
        store.add(e0);
        store.add(e1);
        store.add(e2);

        // Remove entry 0 — entry 2 (third) gets swapped to index 0
        store.remove(&id0);
        assert_eq!(store.len(), 2);

        // "third" is now at index 0
        assert_eq!(store.get(&id2).unwrap().content, "third");

        // Label index for "only:third" should point to 0 (not 2)
        let results = store.query_labels(&["only:third".into()]);
        assert_eq!(results, vec![0], "Swapped entry should be at index 0");

        // "shared:label" should contain indices 0 and 1 (not 0 and 2)
        let shared = store.query_labels(&["shared:label".into()]);
        assert_eq!(
            shared,
            vec![0, 1],
            "shared:label should have updated indices"
        );

        // "only:first" should be gone
        assert_eq!(store.label_posting_len("only:first"), 0);
    }

    #[test]
    fn label_index_remove_last_no_swap() {
        let mut store = EchoStore::new();
        let e0 = make_labeled_entry("first", &["topic:a"]);
        let e1 = make_labeled_entry("second", &["topic:b"]);
        let id1 = e1.id.clone();
        store.add(e0);
        store.add(e1);

        // Remove last entry — no swap needed
        store.remove(&id1);
        assert_eq!(store.len(), 1);
        assert_eq!(store.label_posting_len("topic:b"), 0);
        assert_eq!(store.query_labels(&["topic:a".into()]), vec![0]);
    }

    #[test]
    fn label_index_rebuild_matches_incremental() {
        let mut store = EchoStore::new();
        store.add(make_labeled_entry("a", &["x:1", "y:2"]));
        store.add(make_labeled_entry("b", &["x:1", "z:3"]));
        store.add(make_labeled_entry("c", &["y:2", "z:3"]));

        // Capture incremental state
        let before = store.label_index.clone();

        // Rebuild from scratch
        store.rebuild_label_index();

        // Sort posting lists for comparison (order may differ)
        let normalize = |m: &HashMap<String, Vec<u32>>| -> HashMap<String, Vec<u32>> {
            m.iter()
                .map(|(k, v)| {
                    let mut sorted = v.clone();
                    sorted.sort();
                    (k.clone(), sorted)
                })
                .collect()
        };

        assert_eq!(
            normalize(&before),
            normalize(&store.label_index),
            "Rebuilt index should match incrementally-built index"
        );
    }

    #[test]
    fn query_labels_or_semantics() {
        let mut store = EchoStore::new();
        store.add(make_labeled_entry("a", &["topic:language"]));
        store.add(make_labeled_entry("b", &["topic:career"]));
        store.add(make_labeled_entry("c", &["topic:language", "topic:career"]));

        // OR: entries matching either label
        let results = store.query_labels(&["topic:language".into(), "topic:career".into()]);
        assert_eq!(
            results,
            vec![0, 1, 2],
            "OR should return union, deduplicated and sorted"
        );
    }

    #[test]
    fn query_labels_empty_input() {
        let mut store = EchoStore::new();
        store.add(make_labeled_entry("a", &["topic:x"]));
        let results = store.query_labels(&[]);
        assert!(results.is_empty(), "Empty label query should return empty");
    }

    #[test]
    fn label_count_tracks_unique_labels() {
        let mut store = EchoStore::new();
        assert_eq!(store.label_count(), 0);

        store.add(make_labeled_entry("a", &["x:1", "y:2"]));
        assert_eq!(store.label_count(), 2);

        store.add(make_labeled_entry("b", &["x:1", "z:3"]));
        assert_eq!(store.label_count(), 3); // x:1, y:2, z:3
    }

    #[test]
    fn labels_for_returns_entry_labels() {
        let mut store = EchoStore::new();
        store.add(make_labeled_entry("a", &["topic:language", "domain:life"]));
        assert_eq!(store.labels_for(0), &["topic:language", "domain:life"]);
        assert!(store.labels_for(99).is_empty());
    }

    #[test]
    fn label_index_stress_add_remove_sequence() {
        // Deterministic 100-operation add/remove sequence
        let mut store = EchoStore::new();
        let mut ids: Vec<MemoryId> = Vec::new();

        // Add 50 entries
        for i in 0..50 {
            let labels: Vec<&str> = match i % 5 {
                0 => vec!["topic:a", "domain:work"],
                1 => vec!["topic:b", "domain:life"],
                2 => vec!["topic:a", "topic:b"],
                3 => vec!["domain:work"],
                _ => vec!["topic:a", "domain:life", "action:learning"],
            };
            let entry = make_labeled_entry(&format!("entry_{i}"), &labels);
            ids.push(entry.id.clone());
            store.add(entry);
        }

        // Remove 25 entries (every other one)
        for i in (0..50).step_by(2) {
            store.remove(&ids[i]);
        }

        assert_eq!(store.len(), 25);

        // Verify consistency: rebuild should match
        let before = store.label_index.clone();
        store.rebuild_label_index();

        let normalize = |m: &HashMap<String, Vec<u32>>| -> HashMap<String, Vec<u32>> {
            m.iter()
                .map(|(k, v)| {
                    let mut sorted = v.clone();
                    sorted.sort();
                    (k.clone(), sorted)
                })
                .collect()
        };

        assert_eq!(
            normalize(&before),
            normalize(&store.label_index),
            "After 50 adds + 25 removes, rebuild should match incremental index"
        );
    }

    // --- Graph navigation tests (KS57) ---

    #[test]
    fn connected_by_labels_returns_grouped() {
        let mut store = EchoStore::new();
        // Entry 0: shares "topic:lang" with entry 1 and entry 2, "domain:work" with entry 2
        store.add(make_labeled_entry("main", &["topic:lang", "domain:work"]));
        // Entry 1: shares "topic:lang" with entry 0
        store.add(make_labeled_entry("peer-a", &["topic:lang"]));
        // Entry 2: shares both labels with entry 0
        store.add(make_labeled_entry("peer-b", &["topic:lang", "domain:work"]));

        let grouped = store.connected_by_labels(0);

        // "topic:lang" should contain entries 1, 2 (not 0)
        let lang = grouped.get("topic:lang").expect("should have topic:lang");
        assert!(lang.contains(&1), "peer-a should be in topic:lang");
        assert!(lang.contains(&2), "peer-b should be in topic:lang");
        assert!(!lang.contains(&0), "source entry should be excluded");

        // "domain:work" should contain entry 2 only (not 0)
        let work = grouped.get("domain:work").expect("should have domain:work");
        assert_eq!(work, &vec![2], "only peer-b shares domain:work");
    }

    #[test]
    fn connected_indices_with_filter() {
        let mut store = EchoStore::new();
        store.add(make_labeled_entry("main", &["topic:lang", "domain:work"]));
        store.add(make_labeled_entry("lang-only", &["topic:lang"]));
        store.add(make_labeled_entry("work-only", &["domain:work"]));
        store.add(make_labeled_entry("both", &["topic:lang", "domain:work"]));

        // Filter to "topic:lang" only — should return entries 1, 3 (not 0, not 2)
        let result = store.connected_indices(0, Some("topic:lang"));
        assert!(result.contains(&1), "lang-only should match");
        assert!(result.contains(&3), "both should match");
        assert!(!result.contains(&0), "source excluded");
        assert!(
            !result.contains(&2),
            "work-only should not match lang filter"
        );
    }

    #[test]
    fn connected_indices_all_labels() {
        let mut store = EchoStore::new();
        store.add(make_labeled_entry("main", &["topic:lang", "domain:work"]));
        store.add(make_labeled_entry("lang-only", &["topic:lang"]));
        store.add(make_labeled_entry("work-only", &["domain:work"]));
        store.add(make_labeled_entry("neither", &["topic:other"]));

        // No filter — union across all labels on entry 0
        let result = store.connected_indices(0, None);
        assert!(result.contains(&1), "lang-only should match via topic:lang");
        assert!(
            result.contains(&2),
            "work-only should match via domain:work"
        );
        assert!(!result.contains(&3), "neither should not match");
        assert!(!result.contains(&0), "source excluded");
        // Should be deduped
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(result.len(), sorted.len(), "result should be deduplicated");
    }

    #[test]
    fn embedding_at_valid_index() {
        let mut store = EchoStore::new();
        store.add(MemoryEntry::new(
            "a".into(),
            vec![1.0, 2.0, 3.0],
            "test".into(),
        ));
        store.add(MemoryEntry::new(
            "b".into(),
            vec![4.0, 5.0, 6.0],
            "test".into(),
        ));

        let emb0 = store.embedding_at(0).expect("should have embedding at 0");
        assert_eq!(emb0, &[1.0, 2.0, 3.0]);

        let emb1 = store.embedding_at(1).expect("should have embedding at 1");
        assert_eq!(emb1, &[4.0, 5.0, 6.0]);

        assert!(
            store.embedding_at(99).is_none(),
            "out-of-bounds should return None"
        );
    }

    // --- Entity index tests (KS61) ---

    #[test]
    fn entity_index_query_case_insensitive() {
        use shrimpk_core::{Triple, TriplePredicate};

        let mut store = EchoStore::new();
        let mut entry = MemoryEntry::new("test".into(), vec![0.0; 384], "test".into());
        entry.triples.push(Triple {
            subject: "Lior".into(),
            predicate: TriplePredicate::WorksAt,
            object: "Bellkis".into(),
        });
        store.add(entry);

        assert!(!store.query_entities("lior").is_empty());
        assert!(!store.query_entities("LIOR").is_empty());
        assert!(!store.query_entities("bellkis").is_empty());
        assert!(store.query_entities("unknown").is_empty());
    }

    #[test]
    fn entity_index_rebuild_matches_incremental() {
        use shrimpk_core::{Triple, TriplePredicate};

        let mut store = EchoStore::new();
        let mut entry = MemoryEntry::new("test".into(), vec![0.0; 384], "test".into());
        entry.triples.push(Triple {
            subject: "Alice".into(),
            predicate: TriplePredicate::LivesIn,
            object: "NYC".into(),
        });
        store.add(entry);

        let before = store.query_entities("alice").len();
        store.rebuild_entity_index();
        let after = store.query_entities("alice").len();
        assert_eq!(before, after);
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
