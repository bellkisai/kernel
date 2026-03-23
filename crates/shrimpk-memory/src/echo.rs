//! The Echo cycle — the core activation engine.
//!
//! Implements the "push-based memory" concept: instead of explicitly searching,
//! memories self-activate based on contextual similarity to the current input.
//!
//! Phase 1: brute-force cosine similarity against all stored embeddings.
//! Phase 2: LSH for sub-linear candidate retrieval, with brute-force fallback.

use chrono::Utc;
use shrimpk_core::{
    EchoConfig, EchoResult, MemoryEntry, MemoryEntrySummary, MemoryId, MemoryStats, Result,
    SensitivityLevel, ShrimPKError,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::instrument;

use crate::bloom::TopicFilter;
use crate::consolidation::{self, ConsolidationResult};
use crate::consolidator;
use crate::embedder::Embedder;
use crate::hebbian::HebbianGraph;
use crate::lsh::CosineHash;
use crate::pii::PiiFilter;
use crate::reformulator::MemoryReformulator;
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
    /// LSH index for sub-linear candidate retrieval (behind Mutex for mut access).
    lsh: Mutex<CosineHash>,
    /// Bloom filter for O(1) topic pre-screening (behind RwLock for concurrent reads).
    bloom: RwLock<TopicFilter>,
    /// Whether the Bloom filter needs rebuilding (set after deletions).
    bloom_dirty: Mutex<bool>,
    /// PII/secret detection and masking.
    pii_filter: PiiFilter,
    /// Memory reformulator for improved echo recall (~9% higher similarity).
    reformulator: MemoryReformulator,
    /// Engine configuration.
    config: EchoConfig,
    /// Hebbian co-activation graph for associative memory boosting.
    hebbian: RwLock<HebbianGraph>,
    /// Running statistics.
    stats: Mutex<EchoStats>,
    /// Handle to the background consolidation task (if started).
    consolidation_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Consolidator for LLM-based fact extraction during sleep consolidation.
    consolidator: Box<dyn shrimpk_core::Consolidator>,
}

impl EchoEngine {
    /// Initialize a new EchoEngine with an empty store.
    ///
    /// Downloads/loads the embedding model on first call.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if the model fails to initialize.
    #[instrument(skip(config), fields(max_memories = config.max_memories, threshold = config.similarity_threshold))]
    pub fn new(config: EchoConfig) -> Result<Self> {
        let embedder = Embedder::new()?;
        let pii_filter = PiiFilter::new();
        let reformulator = MemoryReformulator::new();
        let store = RwLock::new(EchoStore::new());
        let lsh = CosineHash::new(config.embedding_dim, 16, 10);
        let bloom = TopicFilter::new(config.max_memories, 0.01);

        tracing::info!(
            max_memories = config.max_memories,
            threshold = config.similarity_threshold,
            dim = config.embedding_dim,
            use_lsh = config.use_lsh,
            use_bloom = config.use_bloom,
            "EchoEngine initialized (empty store)"
        );

        let consolidator_impl = consolidator::from_config(&config);

        Ok(Self {
            embedder: Mutex::new(embedder),
            store,
            lsh: Mutex::new(lsh),
            bloom: RwLock::new(bloom),
            bloom_dirty: Mutex::new(false),
            pii_filter,
            reformulator,
            hebbian: RwLock::new(HebbianGraph::new(604_800.0, 0.01)),
            config,
            stats: Mutex::new(EchoStats::default()),
            consolidation_handle: Mutex::new(None),
            consolidator: consolidator_impl,
        })
    }

    /// Explicitly shut down the engine, dropping it in a blocking context.
    ///
    /// This avoids the Tokio runtime nesting panic caused by `ort`'s internal
    /// runtime when `EchoEngine` is implicitly dropped inside an async context.
    /// Call this instead of relying on implicit drop in async tests/code.
    pub async fn shutdown(self) {
        tokio::task::spawn_blocking(move || drop(self))
            .await
            .expect("shutdown task panicked");
    }

    /// Arc variant of shutdown for use with shared engine references.
    pub async fn shutdown_arc(self: Arc<Self>) {
        tokio::task::spawn_blocking(move || drop(self))
            .await
            .expect("shutdown task panicked");
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
    /// Returns `ShrimPKError::Embedding` if embedding generation fails.
    /// Returns `ShrimPKError::Memory` if the store is at capacity.
    #[instrument(skip(self, text), fields(text_len = text.len(), source = source))]
    pub async fn store(&self, text: &str, source: &str) -> Result<MemoryId> {
        // Check capacity
        {
            let store = self.store.read().await;
            if store.len() >= self.config.max_memories {
                return Err(ShrimPKError::Memory(format!(
                    "Store at capacity ({} memories). Remove memories or increase max_memories.",
                    self.config.max_memories
                )));
            }
        }

        // Check disk limit
        if self.config.max_disk_bytes > 0 {
            let usage = shrimpk_core::config::disk_usage(&self.config.data_dir).unwrap_or(0);
            if usage >= self.config.max_disk_bytes {
                return Err(ShrimPKError::DiskLimit(format!(
                    "Disk limit reached ({} / {} bytes). Free space or increase max_disk_bytes.",
                    usage, self.config.max_disk_bytes
                )));
            }
            // Warn at 80%
            let threshold = self.config.max_disk_bytes * 80 / 100;
            if usage >= threshold {
                tracing::warn!(
                    usage_bytes = usage,
                    limit_bytes = self.config.max_disk_bytes,
                    "Disk usage at {}% of limit",
                    usage * 100 / self.config.max_disk_bytes
                );
            }
        }

        // 1. PII filtering
        let (masked_text, pii_matches) = self.pii_filter.mask(text);
        let sensitivity = self.pii_filter.classify(text);

        // If the text is entirely blocked-level sensitive, don't store
        if sensitivity == SensitivityLevel::Blocked {
            return Err(ShrimPKError::Memory(
                "Content classified as Blocked — not stored".into(),
            ));
        }

        // 2. Try reformulation for better echo recall (~9% higher similarity)
        //    Reformulate the PII-masked text (or original if no PII found)
        let text_for_reformulation = if !pii_matches.is_empty() {
            &masked_text
        } else {
            text
        };
        let reformulated = self.reformulator.reformulate(text_for_reformulation);

        // 3. Generate embedding from the BEST text for recall:
        //    - Reformulated text if available (structured form embeds better)
        //    - Otherwise original text (semantic meaning preserved)
        let embed_text = reformulated.as_deref().unwrap_or(text);
        let embedding = {
            let mut embedder = self
                .embedder
                .lock()
                .map_err(|e| ShrimPKError::Embedding(format!("Embedder lock poisoned: {e}")))?;
            embedder.embed(embed_text)?
        };

        // 4. Build entry with auto-categorization for adaptive decay
        let category = self.reformulator.categorize(text);
        let mut entry = MemoryEntry::new(text.to_string(), embedding.clone(), source.to_string());
        entry.sensitivity = sensitivity;
        entry.category = category;
        if !pii_matches.is_empty() {
            entry.masked_content = Some(masked_text);
        }
        entry.reformulated = reformulated.clone();
        let id = entry.id.clone();

        tracing::debug!(
            reformulated = reformulated.is_some(),
            category = ?category,
            "Memory reformulation + categorization step"
        );

        // 4. Add to store, LSH index, and Bloom filter
        {
            let mut store = self.store.write().await;
            let index = store.add(entry);

            // Insert into LSH index for sub-linear retrieval
            if self.config.use_lsh
                && let Ok(mut lsh) = self.lsh.lock()
            {
                lsh.insert(index as u32, &embedding);
            }

            // Insert into Bloom filter for topic pre-screening
            if self.config.use_bloom {
                let mut bloom = self.bloom.write().await;
                bloom.insert_memory(text);
            }
        }

        tracing::info!(
            memory_id = %id,
            sensitivity = ?sensitivity,
            category = ?category,
            pii_matches = pii_matches.len(),
            "Memory stored"
        );

        Ok(id)
    }

    /// Perform an echo query — find memories that resonate with the query.
    ///
    /// Pipeline:
    /// 1. Embed the query text
    /// 2. Bloom pre-check — if no fingerprints match, return empty immediately (O(1))
    /// 3. Read-lock the store
    /// 4. Handle empty store
    /// 5. Use LSH for sub-linear candidate retrieval (if enabled and sufficient candidates)
    ///    Falls back to brute-force if LSH returns < 10 candidates
    /// 6. Filter by threshold, sort by score, take top `max_results`
    /// 7. Build EchoResult vec
    /// 8. Update echo_count and last_echoed on matched entries
    /// 9. Record latency
    ///
    /// # Arguments
    /// * `query` - The text to find resonating memories for
    /// * `max_results` - Maximum number of results to return
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if query embedding fails.
    #[instrument(skip(self, query), fields(query_len = query.len(), max_results))]
    pub async fn echo(&self, query: &str, max_results: usize) -> Result<Vec<EchoResult>> {
        let start = std::time::Instant::now();

        // 1. Embed the query
        let query_embedding = {
            let mut embedder = self
                .embedder
                .lock()
                .map_err(|e| ShrimPKError::Embedding(format!("Embedder lock poisoned: {e}")))?;
            embedder.embed(query)?
        };

        // 2. Bloom filter pre-check — skip everything if no fingerprints match.
        //    Bypass for small stores (< 50 entries) where Bloom adds risk without benefit.
        if self.config.use_bloom {
            let bloom = self.bloom.read().await;
            if bloom.len() >= 50 && !bloom.is_empty() && !bloom.might_match(query) {
                tracing::debug!("Bloom filter rejected query — no matching fingerprints");
                self.record_latency(start.elapsed().as_micros() as u64);
                return Ok(Vec::new());
            }
        }

        // 3. Read-lock the store
        let store = self.store.read().await;

        // 4. Handle empty store gracefully
        if store.is_empty() {
            tracing::debug!("Empty store, returning no results");
            self.record_latency(start.elapsed().as_micros() as u64);
            return Ok(Vec::new());
        }

        // 5. Candidate retrieval: LSH (sub-linear) or brute-force fallback
        let embeddings = store.all_embeddings();

        let candidates: Vec<(usize, &[f32])> = if self.config.use_lsh {
            // Query LSH for candidate indices
            let lsh_candidates = self
                .lsh
                .lock()
                .map_err(|e| ShrimPKError::Memory(format!("LSH lock poisoned: {e}")))?
                .query(&query_embedding);

            if lsh_candidates.len() >= 10 {
                // LSH returned enough candidates — compute similarity only on these
                tracing::debug!(
                    lsh_candidates = lsh_candidates.len(),
                    total = embeddings.len(),
                    "LSH candidate retrieval (sub-linear)"
                );
                lsh_candidates
                    .iter()
                    .filter_map(|&idx| {
                        let i = idx as usize;
                        embeddings.get(i).map(|e| (i, e.as_slice()))
                    })
                    .collect()
            } else {
                // LSH returned too few candidates — fall back to brute-force
                tracing::debug!(
                    lsh_candidates = lsh_candidates.len(),
                    total = embeddings.len(),
                    "LSH returned < 10 candidates, falling back to brute-force"
                );
                embeddings
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (i, e.as_slice()))
                    .collect()
            }
        } else {
            // LSH disabled — brute-force
            embeddings
                .iter()
                .enumerate()
                .map(|(i, e)| (i, e.as_slice()))
                .collect()
        };

        // 5b. Split pipeline: Pipe A (above threshold) + Pipe B (near-miss child rescue)
        //     Use half-threshold to capture near-misses for potential child rescue.
        let near_miss_threshold = self.config.similarity_threshold * 0.5;
        let all_ranked = similarity::rank_candidates(
            &query_embedding,
            &candidates,
            near_miss_threshold,
        );

        let threshold = self.config.similarity_threshold;
        let (pipe_a, pipe_b): (Vec<(usize, f32)>, Vec<(usize, f32)>) = all_ranked
            .into_iter()
            .partition(|&(_, score)| score >= threshold);

        // 6. Pipe B: check if near-miss parents have enriched children that score better
        let promoted: Vec<(usize, f32)> = if store.has_enriched_memories() && !pipe_b.is_empty() {
            let mut promotions: Vec<(usize, f32)> = Vec::new();
            for &(idx, _parent_score) in &pipe_b {
                if let Some(entry) = store.entry_at(idx) {
                    if !entry.enriched {
                        continue;
                    }
                    let child_indices = store.children_of(&entry.id);
                    if child_indices.is_empty() {
                        continue;
                    }
                    let embeddings = store.all_embeddings();
                    let mut best_child_score: f32 = 0.0;
                    for &child_idx in child_indices {
                        if let Some(child_emb) = embeddings.get(child_idx) {
                            let child_sim =
                                similarity::cosine_similarity(&query_embedding, child_emb);
                            if child_sim > best_child_score {
                                best_child_score = child_sim;
                            }
                        }
                    }
                    if best_child_score >= threshold {
                        tracing::debug!(
                            parent_idx = idx,
                            child_score = best_child_score,
                            "Pipe B: child rescued parent memory"
                        );
                        promotions.push((idx, best_child_score));
                    }
                }
            }
            promotions
        } else {
            Vec::new()
        };

        // Merge Pipe A + promoted, deduplicate
        let mut combined = pipe_a;
        if !promoted.is_empty() {
            let existing: std::collections::HashSet<usize> =
                combined.iter().map(|&(idx, _)| idx).collect();
            for (idx, score) in promoted {
                if !existing.contains(&idx) {
                    combined.push((idx, score));
                }
            }
            combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Take top results
        let top: Vec<(usize, f32)> = combined.into_iter().take(max_results).collect();

        // 7. Hebbian two-pass ranking:
        //    a) Co-activate all pairs of returned memories (strengthens associations)
        //    b) Boost results that have Hebbian associations with OTHER results
        let top_indices: Vec<u32> = top.iter().map(|&(idx, _)| idx as u32).collect();

        // 7a. Co-activate all pairs of returned results
        {
            let mut hebbian = self.hebbian.write().await;
            for i in 0..top_indices.len() {
                for j in (i + 1)..top_indices.len() {
                    // Strength proportional to geometric mean of both similarities
                    let sim_i = top[i].1 as f64;
                    let sim_j = top[j].1 as f64;
                    let strength = (sim_i * sim_j).sqrt() * 0.1; // modest reinforcement
                    hebbian.co_activate(top_indices[i], top_indices[j], strength);
                }
            }
        }

        // 7b. Second pass — compute Hebbian boost for each result.
        //     Enhanced with typed relationship awareness (KS18 Track 4):
        //     - Supersedes edges: the NEWER memory gets an extra boost (knowledge updates)
        //     - Typed relationships: small extra boost when relationship type is relevant
        let hebbian_boosts: Vec<f64> = {
            let hebbian = self.hebbian.read().await;
            top.iter()
                .map(|&(idx, _)| {
                    let idx = idx as u32;
                    let mut boost: f64 = 0.0;

                    for &other in top_indices.iter().filter(|&&o| o != idx) {
                        let weight = hebbian.get_weight(idx, other);
                        if weight <= 0.0 {
                            continue;
                        }
                        boost += weight;

                        // Typed relationship bonus
                        if let Some(rel) = hebbian.get_relationship(idx, other) {
                            match rel {
                                // Supersedes: the node with the HIGHER index is newer
                                // (added later to the store). Give it an extra boost.
                                crate::hebbian::RelationshipType::Supersedes => {
                                    if idx > other {
                                        // This node is the newer memory — boost it
                                        boost += 0.1;
                                    }
                                    // If idx < other, the OTHER node is newer — no boost for us
                                }
                                // Any typed (non-CoActivation) relationship gets a small
                                // relevance bonus — these edges carry semantic meaning
                                // beyond mere co-occurrence.
                                crate::hebbian::RelationshipType::CoActivation => {}
                                _ => {
                                    boost += 0.05;
                                }
                            }
                        }
                    }

                    // Cap Hebbian boost at 0.4 (raised from 0.3 to accommodate
                    // relationship bonuses without squeezing co-activation boost)
                    boost.min(0.4)
                })
                .collect()
        };

        // 7c. Build EchoResult vec with final_score = similarity + hebbian + recency, scaled by decay
        let now = Utc::now();
        let recency_weight = self.config.recency_weight as f64;
        let mut results: Vec<EchoResult> = top
            .iter()
            .zip(hebbian_boosts.iter())
            .filter_map(|(&(idx, score), &boost)| {
                let entry = store.entry_at(idx)?;

                // Apply category-aware decay: older memories score lower (F-02 fix)
                let age_secs = (now - entry.created_at).num_seconds().max(0) as f64;
                let half_life = entry.category.half_life_secs();
                let decay = (-age_secs * std::f64::consts::LN_2 / half_life).exp();

                // Recency boost (KS18 Track 3, tuned KS19 Track 2): sqrt-based decay.
                // Formula: recency_weight / (1.0 + sqrt(days_since_stored))
                // At 0 days: +0.15. At 1 day: +0.075. At 7 days: +0.050. At 30 days: +0.026.
                // Stronger differentiation for recent memories while still fading gracefully.
                let days_since_stored = age_secs / 86400.0;
                let recency_boost = recency_weight / (1.0 + days_since_stored.sqrt());

                Some(EchoResult {
                    memory_id: entry.id.clone(),
                    content: entry.display_content().to_string(),
                    similarity: score,
                    final_score: (score as f64 + boost + recency_boost) * decay,
                    source: entry.source.clone(),
                    echoed_at: now,
                })
            })
            .collect();

        // 7d. Re-sort by final_score (similarity + hebbian boost)
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Release read lock before acquiring write lock
        let matched_ids: Vec<(MemoryId, usize)> = top
            .iter()
            .filter_map(|&(idx, _)| store.entry_at(idx).map(|e| (e.id.clone(), idx)))
            .collect();
        drop(store);

        // 8. Update echo_count and last_echoed (requires write lock)
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

        // 9. Record latency
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
    /// Returns `ShrimPKError::Memory` if the ID is not found.
    #[instrument(skip(self), fields(memory_id = %id))]
    pub async fn forget(&self, id: MemoryId) -> Result<()> {
        let mut store = self.store.write().await;

        // Capture index and length before removal for LSH swap-remove tracking
        let removed_index = store.index_of(&id);
        let len_before = store.len();

        match store.remove(&id) {
            Some(_) => {
                // Remove from Hebbian graph
                if let Some(removed_idx) = removed_index {
                    let mut hebbian = self.hebbian.write().await;
                    hebbian.remove_node(removed_idx as u32);
                }

                // Update LSH index to reflect the swap-remove
                if self.config.use_lsh
                    && let Ok(mut lsh) = self.lsh.lock()
                    && let Some(removed_idx) = removed_index
                {
                    // Remove the deleted entry from LSH
                    lsh.remove(removed_idx as u32);

                    // If swap-remove moved the last entry to the removed position,
                    // update its LSH entry: remove old index, re-insert at new index
                    let last_index = len_before - 1;
                    if removed_idx != last_index {
                        lsh.remove(last_index as u32);
                        if let Some(embedding) = store.all_embeddings().get(removed_idx) {
                            lsh.insert(removed_idx as u32, embedding);
                        }
                    }
                }

                // Mark Bloom filter as dirty — cannot remove individual items,
                // rebuild will happen on next persist()
                if self.config.use_bloom
                    && let Ok(mut dirty) = self.bloom_dirty.lock()
                {
                    *dirty = true;
                }

                tracing::info!(memory_id = %id, "Memory forgotten");
                Ok(())
            }
            None => Err(ShrimPKError::Memory(format!("Memory not found: {id}"))),
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

        let disk_usage = shrimpk_core::config::disk_usage(&self.config.data_dir).unwrap_or(0);

        MemoryStats {
            total_memories: total,
            index_size_bytes: (total * self.config.embedding_dim * 4) as u64,
            ram_usage_bytes: ram_usage,
            max_capacity: self.config.max_memories,
            avg_echo_latency_ms: avg_latency_ms,
            total_echo_queries: stats.query_count,
            disk_usage_bytes: disk_usage,
            max_disk_bytes: self.config.max_disk_bytes,
        }
    }

    /// Return summaries of all stored memories (for dump/listing).
    pub async fn all_entry_summaries(&self) -> Vec<MemoryEntrySummary> {
        let store = self.store.read().await;
        store
            .all_entries()
            .iter()
            .map(|e| MemoryEntrySummary {
                id: e.id.clone(),
                content: e
                    .masked_content
                    .as_deref()
                    .unwrap_or(&e.content)
                    .to_string(),
                source: e.source.clone(),
                echo_count: e.echo_count,
                sensitivity: e.sensitivity,
                category: e.category,
            })
            .collect()
    }

    /// Persist the store to disk.
    ///
    /// Saves to `config.data_dir/echo_store.shrm` in binary format.
    #[instrument(skip(self))]
    pub async fn persist(&self) -> Result<()> {
        // Rebuild Bloom filter if dirty (deletions occurred since last rebuild)
        if self.config.use_bloom {
            let needs_rebuild = self.bloom_dirty.lock().map(|d| *d).unwrap_or(false);

            if needs_rebuild {
                let store = self.store.read().await;
                let texts: Vec<String> = store
                    .all_entries()
                    .iter()
                    .map(|e| e.content.clone())
                    .collect();
                drop(store);

                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let mut bloom = self.bloom.write().await;
                bloom.rebuild(&text_refs);

                if let Ok(mut dirty) = self.bloom_dirty.lock() {
                    *dirty = false;
                }

                tracing::info!(
                    entries = texts.len(),
                    bloom_size_bytes = bloom.size_bytes(),
                    "Bloom filter rebuilt after deletions"
                );
            }
        }

        let store = self.store.read().await;
        let path = self.store_path();
        store.save(&path)?;

        // Persist Hebbian graph
        let hebbian_path = self.config.data_dir.join("hebbian.json");
        let hebbian = self.hebbian.read().await;
        hebbian
            .save(&hebbian_path)
            .map_err(|e| ShrimPKError::Persistence(format!("Failed to save Hebbian graph: {e}")))?;

        tracing::info!(
            hebbian_edges = hebbian.len(),
            hebbian_activations = hebbian.total_activations(),
            "Hebbian graph persisted"
        );

        Ok(())
    }

    /// Load an EchoEngine from disk.
    ///
    /// Tries binary format (`.shrm`) first. If not found, falls back to legacy
    /// JSON format (`.json`) for migration. Starts empty if neither exists.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if model init fails.
    /// Returns `ShrimPKError::Persistence` if store file is corrupted.
    #[instrument(skip(config), fields(data_dir = %config.data_dir.display()))]
    pub fn load(config: EchoConfig) -> Result<Self> {
        let embedder = Embedder::new()?;
        let pii_filter = PiiFilter::new();
        let reformulator = MemoryReformulator::new();

        let store_path = config.data_dir.join("echo_store.shrm");
        let loaded_store = EchoStore::load(&store_path)?;

        // Rebuild LSH index from loaded embeddings
        let mut lsh = CosineHash::new(config.embedding_dim, 16, 10);
        if config.use_lsh {
            for (i, embedding) in loaded_store.all_embeddings().iter().enumerate() {
                lsh.insert(i as u32, embedding);
            }
        }

        // Rebuild Bloom filter from loaded memory texts
        let mut bloom = TopicFilter::new(config.max_memories, 0.01);
        if config.use_bloom {
            for entry in loaded_store.all_entries() {
                bloom.insert_memory(&entry.content);
            }
        }

        // Load Hebbian graph from disk
        let hebbian_path = config.data_dir.join("hebbian.json");
        let hebbian = HebbianGraph::load(&hebbian_path, 604_800.0, 0.01).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "Failed to load Hebbian graph, starting fresh");
            HebbianGraph::new(604_800.0, 0.01)
        });

        tracing::info!(
            entries = loaded_store.len(),
            lsh_entries = lsh.len(),
            bloom_entries = bloom.len(),
            bloom_size_bytes = bloom.size_bytes(),
            hebbian_edges = hebbian.len(),
            hebbian_activations = hebbian.total_activations(),
            path = %store_path.display(),
            "EchoEngine loaded from disk"
        );

        let consolidator_impl = consolidator::from_config(&config);

        Ok(Self {
            embedder: Mutex::new(embedder),
            store: RwLock::new(loaded_store),
            lsh: Mutex::new(lsh),
            bloom: RwLock::new(bloom),
            bloom_dirty: Mutex::new(false),
            pii_filter,
            reformulator,
            hebbian: RwLock::new(hebbian),
            config,
            stats: Mutex::new(EchoStats::default()),
            consolidation_handle: Mutex::new(None),
            consolidator: consolidator_impl,
        })
    }

    /// Run a consolidation pass immediately, acquiring all necessary locks.
    ///
    /// This is the manual trigger for maintenance. It acquires write locks on
    /// the store, Hebbian graph, and Bloom filter, then delegates to
    /// [`consolidation::consolidate`].
    pub async fn consolidate_now(&self) -> ConsolidationResult {
        let mut store = self.store.write().await;
        let mut hebbian = self.hebbian.write().await;
        let mut bloom = self.bloom.write().await;
        let mut bloom_dirty = self.bloom_dirty.lock().unwrap_or_else(|e| e.into_inner());
        let mut lsh = self.lsh.lock().unwrap_or_else(|e| e.into_inner());

        consolidation::consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &self.config,
            &*self.consolidator,
            Some(&self.embedder),
            &mut lsh,
        )
    }

    /// Spawn a background consolidation task that runs every `interval_secs` seconds.
    ///
    /// The task loops indefinitely, sleeping for the interval then running a
    /// full consolidation pass. Results are logged via `tracing::info!`.
    ///
    /// The engine must be wrapped in an `Arc` for the background task to hold
    /// a reference. Only one consolidation task can be active at a time; calling
    /// this while a task is already running replaces the previous handle.
    pub fn start_consolidation(self: &Arc<Self>, interval_secs: u64) {
        let engine = Arc::clone(self);
        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                let result = engine.consolidate_now().await;
                tracing::info!(
                    pruned = result.hebbian_edges_pruned,
                    merged = result.duplicates_merged,
                    bloom = result.bloom_rebuilt,
                    decayed = result.echo_counts_decayed,
                    relationships = result.relationships_created,
                    supersedes = result.supersedes_created,
                    ms = result.duration_ms,
                    "Consolidation complete"
                );
            }
        });

        if let Ok(mut guard) = self.consolidation_handle.lock() {
            // Abort previous task if one is running
            if let Some(prev) = guard.take() {
                prev.abort();
            }
            *guard = Some(handle);
        }
    }

    /// Get the path to the store file.
    fn store_path(&self) -> PathBuf {
        self.config.data_dir.join("echo_store.shrm")
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
    use shrimpk_core::EchoConfig;
    use std::path::PathBuf;

    fn test_config() -> EchoConfig {
        EchoConfig {
            max_memories: 1000,
            similarity_threshold: 0.3,
            max_echo_results: 10,
            ram_budget_bytes: 100_000_000,
            data_dir: PathBuf::from("/tmp/shrimpk-test"),
            embedding_dim: 384,
            ..Default::default()
        }
    }

    // Integration tests require the fastembed model — marked as ignored.
    // Run with: cargo test -p shrimpk-memory -- --ignored

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
        let results = engine
            .echo("systems programming", 5)
            .await
            .expect("Should echo");
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
        engine
            .store("second", "test")
            .await
            .expect("Should store 2");

        let result = engine.store("third", "test").await;
        assert!(result.is_err(), "Should reject when at capacity");
    }

    // --- Recency boost tests (KS18 Track 3) ---

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn recency_boost_newer_memory_ranks_higher() {
        // Two semantically similar memories about the same topic.
        // The newer one (stored later) should rank higher due to recency boost.
        let mut config = test_config();
        config.recency_weight = 0.05;

        let engine = EchoEngine::new(config).expect("Should init");

        // Store old fact
        engine
            .store("I work as an engineer at Google on the Cloud team", "old_session")
            .await
            .expect("Should store old");

        // Manually backdate the old memory's created_at
        {
            let mut store = engine.store.write().await;
            if let Some(entry) = store.entry_at_mut(0) {
                entry.created_at = Utc::now() - chrono::Duration::days(30);
            }
        }

        // Store new correction (created_at = now, so recency boost is maximal)
        engine
            .store("I left Google. I now work at Meta on the infrastructure team", "new_session")
            .await
            .expect("Should store new");

        let results = engine
            .echo("Where do I currently work?", 5)
            .await
            .expect("Should echo");

        assert!(results.len() >= 2, "Should have at least 2 results");

        // Find both memories in results
        let meta_result = results.iter().find(|r| r.content.contains("Meta"));
        let google_result = results.iter().find(|r| r.content.contains("Google"));

        assert!(meta_result.is_some(), "Meta memory should surface");
        assert!(google_result.is_some(), "Google memory should surface");

        // Newer memory (Meta) should have a higher final_score than older (Google)
        let meta_score = meta_result.unwrap().final_score;
        let google_score = google_result.unwrap().final_score;
        assert!(
            meta_score > google_score,
            "Newer memory (Meta, score={meta_score:.6}) should rank higher than older (Google, score={google_score:.6})"
        );
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn recency_boost_disabled_when_zero() {
        // When recency_weight is 0, no recency boost should be applied.
        let mut config = test_config();
        config.recency_weight = 0.0;

        let engine = EchoEngine::new(config).expect("Should init");
        engine
            .store("Test memory for recency", "test")
            .await
            .expect("Should store");

        let results = engine
            .echo("test memory recency", 5)
            .await
            .expect("Should echo");

        assert!(!results.is_empty(), "Should have results");
        // With recency_weight=0 and a fresh memory, score should just be
        // similarity * decay (decay ~1.0 for fresh memory) + hebbian (0 with single result)
        let result = &results[0];
        // Just verify it doesn't crash and returns a sane score
        assert!(result.final_score > 0.0);
        assert!(result.final_score <= 2.0, "Score should be reasonable");
    }
}
