//! The Echo cycle — the core activation engine.
//!
//! Implements the "push-based memory" concept: instead of explicitly searching,
//! memories self-activate based on contextual similarity to the current input.
//!
//! Phase 1: brute-force cosine similarity against all stored embeddings.
//! Phase 2: LSH + Bloom filter for sub-linear candidate retrieval, Hebbian learning.

use bellkis_core::{
    BellkisError, EchoConfig, EchoResult, MemoryEntry, MemoryId, MemoryStats, Result,
    SensitivityLevel,
};
use chrono::Utc;
use std::path::PathBuf;
use std::sync::Mutex;
use tokio::sync::RwLock;
use tracing::instrument;

use crate::embedder::Embedder;
use crate::pii::PiiFilter;
use crate::similarity;
use crate::store::EchoStore;

/// Running statistics for the Echo engine.
#[derive(Debug, Default)]
struct EchoStats {
    /// Total number of echo queries processed.
    query_count: u64,
    /// Sum of all query latencies in microseconds (for averaging).
    total_latency_us: u64,
}

/// The Echo Memory engine.
///
/// Owns the embedder, store, PII filter, and config.
/// Thread-safe via `RwLock<EchoStore>` for concurrent reads during echo queries
/// and exclusive writes during store operations.
/// The embedder is behind a `Mutex` because fastembed requires `&mut self`.
pub struct EchoEngine {
    /// Sentence embedder (fastembed all-MiniLM-L6-v2), behind Mutex for mut access.
    embedder: Mutex<Embedder>,
    /// The in-memory vector store (behind RwLock for concurrent access).
    store: RwLock<EchoStore>,
    /// PII/secret detection and masking.
    pii_filter: PiiFilter,
    /// Engine configuration.
    config: EchoConfig,
    /// Running statistics.
    stats: Mutex<EchoStats>,
}

impl EchoEngine {
    /// Initialize a new EchoEngine with an empty store.
    ///
    /// Downloads/loads the embedding model on first call.
    ///
    /// # Errors
    /// Returns `BellkisError::Embedding` if the model fails to initialize.
    #[instrument(skip(config), fields(max_memories = config.max_memories, threshold = config.similarity_threshold))]
    pub fn new(config: EchoConfig) -> Result<Self> {
        let embedder = Embedder::new()?;
        let pii_filter = PiiFilter::new();
        let store = RwLock::new(EchoStore::new());

        tracing::info!(
            max_memories = config.max_memories,
            threshold = config.similarity_threshold,
            dim = config.embedding_dim,
            "EchoEngine initialized (empty store)"
        );

        Ok(Self {
            embedder: Mutex::new(embedder),
            store,
            pii_filter,
            config,
            stats: Mutex::new(EchoStats::default()),
        })
    }

    /// Store a new memory.
    ///
    /// Pipeline:
    /// 1. Run PII filter on the text -> get masked text + sensitivity level
    /// 2. Generate embedding for the **original** text (semantic meaning preserved)
    /// 3. Create a MemoryEntry with masked content and sensitivity classification
    /// 4. Add to the store
    /// 5. Return the MemoryId
    ///
    /// # Arguments
    /// * `text` - The raw text to store
    /// * `source` - Where this memory came from (e.g., "conversation", "document")
    ///
    /// # Errors
    /// Returns `BellkisError::Embedding` if embedding generation fails.
    /// Returns `BellkisError::Memory` if the store is at capacity.
    #[instrument(skip(self, text), fields(text_len = text.len(), source = source))]
    pub async fn store(&self, text: &str, source: &str) -> Result<MemoryId> {
        // Check capacity
        {
            let store = self.store.read().await;
            if store.len() >= self.config.max_memories {
                return Err(BellkisError::Memory(format!(
                    "Store at capacity ({} memories). Remove memories or increase max_memories.",
                    self.config.max_memories
                )));
            }
        }

        // 1. PII filtering
        let (masked_text, pii_matches) = self.pii_filter.mask(text);
        let sensitivity = self.pii_filter.classify(text);

        // If the text is entirely blocked-level sensitive, don't store
        if sensitivity == SensitivityLevel::Blocked {
            return Err(BellkisError::Memory(
                "Content classified as Blocked — not stored".into(),
            ));
        }

        // 2. Generate embedding from ORIGINAL text (not masked)
        //    The semantic meaning is in the original text.
        let embedding = {
            let mut embedder = self.embedder.lock()
                .map_err(|e| BellkisError::Embedding(format!("Embedder lock poisoned: {e}")))?;
            embedder.embed(text)?
        };

        // 3. Build entry
        let mut entry = MemoryEntry::new(text.to_string(), embedding, source.to_string());
        entry.sensitivity = sensitivity;
        if !pii_matches.is_empty() {
            entry.masked_content = Some(masked_text);
        }
        let id = entry.id.clone();

        // 4. Add to store
        {
            let mut store = self.store.write().await;
            store.add(entry);
        }

        tracing::info!(
            memory_id = %id,
            sensitivity = ?sensitivity,
            pii_matches = pii_matches.len(),
            "Memory stored"
        );

        Ok(id)
    }

    /// Perform an echo query — find memories that resonate with the query.
    ///
    /// Pipeline:
    /// 1. Embed the query text
    /// 2. Read-lock the store
    /// 3. Brute-force cosine similarity against ALL embeddings (Phase 1)
    /// 4. Filter by threshold, sort by score, take top `max_results`
    /// 5. Update echo_count and last_echoed on matched entries
    /// 6. Build and return EchoResult vec
    ///
    /// # Arguments
    /// * `query` - The text to find resonating memories for
    /// * `max_results` - Maximum number of results to return
    ///
    /// # Errors
    /// Returns `BellkisError::Embedding` if query embedding fails.
    #[instrument(skip(self, query), fields(query_len = query.len(), max_results))]
    pub async fn echo(&self, query: &str, max_results: usize) -> Result<Vec<EchoResult>> {
        let start = std::time::Instant::now();

        // 1. Embed the query
        let query_embedding = {
            let mut embedder = self.embedder.lock()
                .map_err(|e| BellkisError::Embedding(format!("Embedder lock poisoned: {e}")))?;
            embedder.embed(query)?
        };

        // 2. Read-lock the store
        let store = self.store.read().await;

        // 3. Handle empty store gracefully
        if store.is_empty() {
            tracing::debug!("Empty store, returning no results");
            self.record_latency(start.elapsed().as_micros() as u64);
            return Ok(Vec::new());
        }

        // 4. Brute-force similarity against all embeddings
        let embeddings = store.all_embeddings();
        let candidates: Vec<(usize, &[f32])> = embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.as_slice()))
            .collect();

        let ranked = similarity::rank_candidates(
            &query_embedding,
            &candidates,
            self.config.similarity_threshold,
        );

        // 5. Take top results
        let top: Vec<(usize, f32)> = ranked.into_iter().take(max_results).collect();

        // 6. Build EchoResult vec
        let results: Vec<EchoResult> = top
            .iter()
            .filter_map(|&(idx, score)| {
                let entry = store.entry_at(idx)?;
                Some(EchoResult {
                    memory_id: entry.id.clone(),
                    content: entry.display_content().to_string(),
                    similarity: score,
                    final_score: score as f64, // Phase 1: no Hebbian boost yet
                    source: entry.source.clone(),
                    echoed_at: Utc::now(),
                })
            })
            .collect();

        // Release read lock before acquiring write lock
        let matched_ids: Vec<(MemoryId, usize)> = top
            .iter()
            .filter_map(|&(idx, _)| {
                store.entry_at(idx).map(|e| (e.id.clone(), idx))
            })
            .collect();
        drop(store);

        // 7. Update echo_count and last_echoed (requires write lock)
        if !matched_ids.is_empty() {
            let mut store = self.store.write().await;
            let now = Utc::now();
            for (_, idx) in &matched_ids {
                if let Some(entry) = store.entry_at_mut(*idx) {
                    entry.echo_count += 1;
                    entry.last_echoed = Some(now);
                }
            }
        }

        // 8. Record latency
        let elapsed = start.elapsed();
        self.record_latency(elapsed.as_micros() as u64);

        tracing::info!(
            results = results.len(),
            elapsed_ms = elapsed.as_millis(),
            "Echo query complete"
        );

        Ok(results)
    }

    /// Forget (remove) a memory by its ID.
    ///
    /// # Errors
    /// Returns `BellkisError::Memory` if the ID is not found.
    #[instrument(skip(self), fields(memory_id = %id))]
    pub async fn forget(&self, id: MemoryId) -> Result<()> {
        let mut store = self.store.write().await;
        match store.remove(&id) {
            Some(_) => {
                tracing::info!(memory_id = %id, "Memory forgotten");
                Ok(())
            }
            None => Err(BellkisError::Memory(format!(
                "Memory not found: {id}"
            ))),
        }
    }

    /// Get current engine statistics.
    pub async fn stats(&self) -> MemoryStats {
        let store = self.store.read().await;
        let total = store.len();
        let stats = self.stats.lock().unwrap();

        let avg_latency_ms = if stats.query_count > 0 {
            (stats.total_latency_us as f64 / stats.query_count as f64) / 1000.0
        } else {
            0.0
        };

        // Estimate memory usage:
        // Each entry: embedding (384 * 4 bytes) + content (~200 bytes avg) + metadata (~100 bytes)
        let bytes_per_entry = (self.config.embedding_dim * 4 + 300) as u64;
        let ram_usage = total as u64 * bytes_per_entry;

        MemoryStats {
            total_memories: total,
            index_size_bytes: (total * self.config.embedding_dim * 4) as u64,
            ram_usage_bytes: ram_usage,
            max_capacity: self.config.max_memories,
            avg_echo_latency_ms: avg_latency_ms,
            total_echo_queries: stats.query_count,
        }
    }

    /// Persist the store to disk.
    ///
    /// Saves to `config.data_dir/echo_store.json`.
    #[instrument(skip(self))]
    pub async fn persist(&self) -> Result<()> {
        let store = self.store.read().await;
        let path = self.store_path();
        store.save(&path)
    }

    /// Load an EchoEngine from disk.
    ///
    /// If the store file exists, loads it. Otherwise starts empty.
    ///
    /// # Errors
    /// Returns `BellkisError::Embedding` if model init fails.
    /// Returns `BellkisError::Persistence` if store file is corrupted.
    #[instrument(skip(config), fields(data_dir = %config.data_dir.display()))]
    pub fn load(config: EchoConfig) -> Result<Self> {
        let embedder = Embedder::new()?;
        let pii_filter = PiiFilter::new();

        let store_path = config.data_dir.join("echo_store.json");
        let loaded_store = EchoStore::load(&store_path)?;

        tracing::info!(
            entries = loaded_store.len(),
            path = %store_path.display(),
            "EchoEngine loaded from disk"
        );

        Ok(Self {
            embedder: Mutex::new(embedder),
            store: RwLock::new(loaded_store),
            pii_filter,
            config,
            stats: Mutex::new(EchoStats::default()),
        })
    }

    /// Get the path to the store file.
    fn store_path(&self) -> PathBuf {
        self.config.data_dir.join("echo_store.json")
    }

    /// Record a query latency measurement.
    fn record_latency(&self, latency_us: u64) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.query_count += 1;
            stats.total_latency_us += latency_us;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellkis_core::EchoConfig;
    use std::path::PathBuf;

    fn test_config() -> EchoConfig {
        EchoConfig {
            max_memories: 1000,
            similarity_threshold: 0.3,
            max_echo_results: 10,
            ram_budget_bytes: 100_000_000,
            data_dir: PathBuf::from("/tmp/bellkis-test"),
            embedding_dim: 384,
            ..Default::default()
        }
    }

    // Integration tests require the fastembed model — marked as ignored.
    // Run with: cargo test -p bellkis-memory -- --ignored

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn engine_store_and_echo() {
        let engine = EchoEngine::new(test_config()).expect("Should init");

        // Store a memory
        let id = engine
            .store("Rust is a systems programming language", "test")
            .await
            .expect("Should store");

        // Echo with a related query
        let results = engine.echo("systems programming", 5).await.expect("Should echo");
        assert!(!results.is_empty(), "Should find related memory");
        assert_eq!(results[0].memory_id, id);
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn echo_empty_store_returns_empty() {
        let engine = EchoEngine::new(test_config()).expect("Should init");
        let results = engine.echo("any query", 5).await.expect("Should succeed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn forget_removes_memory() {
        let engine = EchoEngine::new(test_config()).expect("Should init");
        let id = engine
            .store("remember this", "test")
            .await
            .expect("Should store");

        engine.forget(id.clone()).await.expect("Should forget");

        // Echo should not find it
        let results = engine.echo("remember this", 5).await.expect("Should echo");
        assert!(
            results.iter().all(|r| r.memory_id != id),
            "Forgotten memory should not appear in results"
        );
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn pii_masking_on_store() {
        let engine = EchoEngine::new(test_config()).expect("Should init");
        let id = engine
            .store("Contact admin@example.com for help", "test")
            .await
            .expect("Should store");

        let results = engine
            .echo("contact email help", 5)
            .await
            .expect("Should echo");

        if let Some(result) = results.iter().find(|r| r.memory_id == id) {
            assert!(
                result.content.contains("[MASKED:email]"),
                "Email should be masked in echo results, got: {}",
                result.content
            );
            assert!(
                !result.content.contains("admin@example.com"),
                "Original email should not be in echo results"
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn stats_track_queries() {
        let engine = EchoEngine::new(test_config()).expect("Should init");
        engine.store("test memory", "test").await.unwrap();

        let stats_before = engine.stats().await;
        assert_eq!(stats_before.total_echo_queries, 0);

        engine.echo("test", 5).await.unwrap();
        engine.echo("test", 5).await.unwrap();

        let stats_after = engine.stats().await;
        assert_eq!(stats_after.total_echo_queries, 2);
        assert!(stats_after.avg_echo_latency_ms > 0.0);
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn persist_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut config = test_config();
        config.data_dir = dir.path().to_path_buf();

        let engine = EchoEngine::new(config.clone()).expect("Should init");
        let id = engine
            .store("persistent memory content", "test")
            .await
            .expect("Should store");
        engine.persist().await.expect("Should persist");

        // Load into a new engine
        let engine2 = EchoEngine::load(config).expect("Should load");
        let stats = engine2.stats().await;
        assert_eq!(stats.total_memories, 1);

        // The memory should be retrievable
        let results = engine2
            .echo("persistent memory", 5)
            .await
            .expect("Should echo");
        assert!(!results.is_empty());
        assert_eq!(results[0].memory_id, id);
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn capacity_limit_enforced() {
        let mut config = test_config();
        config.max_memories = 2; // very small limit for testing

        let engine = EchoEngine::new(config).expect("Should init");
        engine.store("first", "test").await.expect("Should store 1");
        engine.store("second", "test").await.expect("Should store 2");

        let result = engine.store("third", "test").await;
        assert!(result.is_err(), "Should reject when at capacity");
    }
}
