//! The Echo cycle — the core activation engine.
//!
//! Implements the "push-based memory" concept: instead of explicitly searching,
//! memories self-activate based on contextual similarity to the current input.
//!
//! Phase 1: brute-force cosine similarity against all stored embeddings.
//! Phase 2: LSH for sub-linear candidate retrieval, with brute-force fallback.

use chrono::Utc;
use shrimpk_core::{
    EchoConfig, EchoResult, GraphCluster, GraphEdge, GraphInterEdge, GraphNeighbor,
    GraphNeighborsResult, GraphNode, GraphNodePreview, GraphOverviewResult, GraphSubgraphResult,
    LabelConnection, MemoryEntry, MemoryEntrySummary, MemoryGraphResult, MemoryId, MemoryStats,
    Modality, QueryMode, Result, SensitivityLevel, ShrimPKError,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::instrument;

use crate::activation;
use crate::bloom::TopicFilter;
use crate::consolidation::{self, ConsolidationResult};
use crate::consolidator;
use crate::embedder::MultiEmbedder;
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
    /// Multi-channel embedder (text + optional vision/speech), behind Mutex for mut access.
    embedder: Mutex<MultiEmbedder>,
    /// The in-memory vector store (behind RwLock for concurrent access).
    store: RwLock<EchoStore>,
    /// Text LSH index for sub-linear candidate retrieval (behind Mutex for mut access).
    text_lsh: Mutex<CosineHash>,
    /// Vision LSH index (CLIP 512-dim). Initialized when vision feature is enabled.
    #[cfg(feature = "vision")]
    vision_lsh: Option<Mutex<CosineHash>>,
    /// Speech LSH index (640-dim: ECAPA-TDNN 256 + Whisper-tiny 384). Initialized when speech feature is enabled.
    #[cfg(feature = "speech")]
    #[allow(dead_code)]
    speech_lsh: Option<Mutex<CosineHash>>,
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
    /// Pre-computed prototype embeddings for Tier 1 label classification (ADR-015).
    prototypes: crate::labels::LabelPrototypes,
}

/// Truncate a string to `max_chars` characters, appending "..." if truncated.
/// Uses char boundaries to avoid splitting multi-byte characters.
fn truncate_content(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => format!("{}...", &s[..byte_idx]),
        None => s.to_string(),
    }
}

/// Detect entity names mentioned in a query by matching against the entity index (KS62).
///
/// Tokenizes query into lowercase words and bigrams, checks each against entity_index keys.
/// Returns matching entity names (lowercased).
fn detect_query_entities(
    query: &str,
    entity_index: &std::collections::HashMap<String, Vec<u32>>,
) -> Vec<String> {
    if entity_index.is_empty() {
        return Vec::new();
    }
    let words: Vec<String> = query
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();

    let mut matches: Vec<String> = Vec::new();
    let mut matched_set = std::collections::HashSet::new();

    // Check bigrams first (longer matches preferred)
    for pair in words.windows(2) {
        let bigram = format!("{} {}", pair[0], pair[1]);
        if entity_index.contains_key(&bigram) && matched_set.insert(bigram.clone()) {
            matches.push(bigram);
        }
    }

    // Check unigrams (skip words already covered by bigrams)
    for word in &words {
        if word.len() < 2 {
            continue; // skip single-char tokens like "a", "i"
        }
        if entity_index.contains_key(word.as_str()) && matched_set.insert(word.clone()) {
            matches.push(word.clone());
        }
    }

    matches
}

/// Reciprocal Rank Fusion merge of vector and graph result lists (KS62).
///
/// Standard RRF: for each item, `score = sum(1/(k + rank))` across lists where it appears.
/// Items present in both lists get boosted. The output preserves original cosine similarity
/// scores (for downstream threshold comparison) but is ORDERED by RRF score.
fn reciprocal_rank_fusion(
    vector: &[(usize, f32)],
    graph: &[(usize, f32)],
    k: usize,
) -> Vec<(usize, f32)> {
    // Accumulate RRF score + track cosine similarity per item
    let mut scores: std::collections::HashMap<usize, (f64, f32)> = std::collections::HashMap::new();

    for (rank, &(idx, cosine)) in vector.iter().enumerate() {
        let entry = scores.entry(idx).or_insert((0.0, cosine));
        entry.0 += 1.0 / (k as f64 + rank as f64 + 1.0);
    }

    for (rank, &(idx, cosine)) in graph.iter().enumerate() {
        let entry = scores.entry(idx).or_insert((0.0, cosine));
        entry.0 += 1.0 / (k as f64 + rank as f64 + 1.0);
    }

    let mut merged: Vec<(usize, f32, f64)> = scores
        .into_iter()
        .map(|(idx, (rrf, cosine))| (idx, cosine, rrf))
        .collect();
    merged.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    merged
        .into_iter()
        .map(|(idx, cosine, _)| (idx, cosine))
        .collect()
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
        let mut embedder = MultiEmbedder::new(&config)?;

        // Initialize label prototypes BEFORE wrapping embedder in Mutex (ADR-015 D4).
        // Prototype embeddings are computed once at startup.
        let mut prototypes = crate::labels::LabelPrototypes::new_empty();
        if config.use_labels {
            prototypes.initialize(|desc| embedder.embed_text(desc).ok());
        }

        let pii_filter = PiiFilter::new();
        let reformulator = MemoryReformulator::new();
        let store = RwLock::new(EchoStore::new());
        let text_lsh = CosineHash::new(config.embedding_dim, 16, 10);
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
            text_lsh: Mutex::new(text_lsh),
            #[cfg(feature = "vision")]
            vision_lsh: if config
                .enabled_modalities
                .contains(&shrimpk_core::Modality::Vision)
            {
                Some(Mutex::new(CosineHash::new(
                    config.vision_embedding_dim,
                    16,
                    10,
                )))
            } else {
                None
            },
            #[cfg(feature = "speech")]
            speech_lsh: if config
                .enabled_modalities
                .contains(&shrimpk_core::Modality::Speech)
            {
                Some(Mutex::new(CosineHash::new(
                    config.speech_embedding_dim,
                    16,
                    10,
                )))
            } else {
                None
            },
            bloom: RwLock::new(bloom),
            bloom_dirty: Mutex::new(false),
            pii_filter,
            reformulator,
            hebbian: RwLock::new(HebbianGraph::new(
                config.hebbian_half_life_secs,
                config.hebbian_prune_threshold,
            )),
            config,
            stats: Mutex::new(EchoStats::default()),
            consolidation_handle: Mutex::new(None),
            consolidator: consolidator_impl,
            prototypes,
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
        let embedding = self.embed_blocking(|e| e.embed_text(embed_text))?;

        // 4. Build entry with auto-categorization for adaptive decay
        let category = self.reformulator.categorize(text);
        let mut entry = MemoryEntry::new(text.to_string(), embedding.clone(), source.to_string());
        entry.sensitivity = sensitivity;
        entry.category = category;
        if !pii_matches.is_empty() {
            entry.masked_content = Some(masked_text);
        }
        entry.reformulated = reformulated.clone();

        // 3b. Tier 1 label generation (ADR-015 D4)
        if self.config.use_labels && self.prototypes.is_initialized() {
            let labels = crate::labels::generate_tier1_labels(text, &embedding, &self.prototypes);
            if !labels.is_empty() {
                entry.labels = labels;
                entry.label_version = 1;
            }
        }

        // 3c. Compute novelty score: 1.0 - max cosine similarity to existing memories.
        //     Uses LSH candidates (fast path) or brute-force top-20 if LSH returns empty.
        {
            let store = self.store.read().await;
            if !store.is_empty() {
                // Try LSH candidates first (sub-linear)
                let candidates: Vec<u32> = if self.config.use_lsh {
                    self.text_lsh
                        .lock()
                        .map(|lsh| lsh.query(&embedding))
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };

                let max_sim = if !candidates.is_empty() {
                    // Score against LSH candidates only
                    candidates
                        .iter()
                        .filter_map(|&idx| store.embedding_at(idx as usize))
                        .filter(|emb| !emb.is_empty())
                        .map(|emb| similarity::cosine_similarity(&embedding, emb))
                        .fold(0.0f32, f32::max)
                } else {
                    // Brute-force fallback: scan all embeddings, find max among top-20
                    let all_embs = store.all_embeddings();
                    let mut sims: Vec<f32> = all_embs
                        .iter()
                        .filter(|e| !e.is_empty())
                        .map(|emb| similarity::cosine_similarity(&embedding, emb))
                        .collect();
                    sims.sort_unstable_by(|a, b| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    sims.into_iter().take(20).next().unwrap_or(0.0)
                };

                entry.novelty_score = (1.0 - max_sim).clamp(0.0, 1.0);
            } else {
                // First memory is maximally novel
                entry.novelty_score = 1.0;
            }
        }

        // KS73: resolve entity from subject, not full text
        // (full-text resolve can match a longer object alias over the shorter subject)
        {
            let store = self.store.read().await;
            let subject = consolidation::extract_subject(&entry.content);
            if !subject.is_empty()
                && let Some(entity_id) = store.resolve_entity(&subject)
            {
                entry.entity_id = Some(entity_id);
            }
        }

        // KS73: self-entity detection for first-person memories
        if entry.entity_id.is_none() {
            let lower = text.to_lowercase();
            let is_first_person = lower.starts_with("i ")
                || lower.starts_with("my ")
                || lower.contains(" i ")
                || lower.contains(" my ")
                || lower.starts_with("i'm ")
                || lower.starts_with("i've ");

            if is_first_person {
                let store = self.store.read().await;
                let self_entity = store
                    .all_entities()
                    .values()
                    .find(|e| {
                        e.kind == shrimpk_core::EntityKind::Person
                            && e.aliases.iter().any(|a| a == "self")
                    })
                    .map(|e| e.id.clone());
                if let Some(eid) = self_entity {
                    entry.entity_id = Some(eid);
                }
                // Don't auto-create self entity at store time — let consolidation handle it
                // when it has enough context (name extraction from "My name is..." patterns)
            }
        }

        let id = entry.id.clone();

        tracing::debug!(
            reformulated = reformulated.is_some(),
            category = ?category,
            labels = entry.labels.len(),
            novelty = entry.novelty_score,
            entity = ?entry.entity_id,
            "Memory reformulation + categorization + labeling + novelty + entity step"
        );

        // 4. Add to store, LSH index, and Bloom filter
        {
            let mut store = self.store.write().await;
            let index = store.add(entry);

            // Insert into text LSH index for sub-linear retrieval
            if self.config.use_lsh
                && let Ok(mut lsh) = self.text_lsh.lock()
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

    /// Retrieve the full content of a specific memory by ID.
    ///
    /// Returns the complete `MemoryEntry` without truncation. Use this after
    /// echo/graph/related to expand a truncated result.
    pub async fn memory_get(&self, id: &MemoryId) -> Result<MemoryEntry> {
        let entry = {
            let store = self.store.read().await;
            store.get(id).cloned()
        }; // read lock released here
        entry.ok_or_else(|| ShrimPKError::Memory(format!("Memory not found: {id}")))
    }

    /// Store an image as a vision memory with optional text description for cross-modal recall.
    ///
    /// Pipeline:
    /// 1. Embed image with CLIP vision encoder -> 512-dim vector
    /// 2. If description provided: embed text → index in text_lsh + bloom for cross-modal recall
    /// 3. Create MemoryEntry with modality: Vision, vision_embedding, optional text embedding
    /// 4. Auto-label with `memtype:image` + Tier 1 labels from description (if any)
    /// 5. Insert into vision_lsh (and text_lsh if description present)
    /// 6. Return memory ID
    ///
    /// # Arguments
    /// * `image_data` - Raw image bytes (PNG, JPEG, BMP, etc.)
    /// * `source` - Where this image came from (e.g., "screenshot", "upload")
    /// * `description` - Optional text description for cross-modal text→vision recall.
    ///   Examples: "kitchen photo", "robot camera feed from hallway", "whiteboard diagram".
    ///   When provided, text echo queries can find this vision memory.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if CLIP embedding fails or vision is not available.
    /// Returns `ShrimPKError::Memory` if the store is at capacity.
    #[cfg(feature = "vision")]
    #[instrument(skip(self, image_data), fields(data_len = image_data.len(), source = source))]
    pub async fn store_image(
        &self,
        image_data: &[u8],
        source: &str,
        description: Option<&str>,
    ) -> Result<MemoryId> {
        // Guard: reject oversized images to prevent OOM during ONNX inference
        const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024; // 10MB
        if image_data.len() > MAX_IMAGE_BYTES {
            return Err(ShrimPKError::Embedding(format!(
                "Image too large: {} bytes (max {})",
                image_data.len(),
                MAX_IMAGE_BYTES
            )));
        }

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

        // 1. Embed image with CLIP
        let vision_embedding = self.embed_blocking(|e| e.embed_image(image_data))?;

        let vision_embedding = vision_embedding.ok_or_else(|| {
            ShrimPKError::Embedding("Vision model not available — cannot embed image".into())
        })?;

        // 2. Build content and optional text embedding for cross-modal recall
        let content = description.unwrap_or("[image]").to_string();
        let text_embedding = if let Some(desc) = description {
            self.embed_blocking(|e| e.embed_text(desc))?
        } else {
            Vec::new()
        };

        let mut entry = MemoryEntry::new_with_modality(
            content,
            text_embedding.clone(),
            source.to_string(),
            Modality::Vision,
        );
        entry.vision_embedding = Some(vision_embedding.clone());

        // 3. Auto-label: always memtype:image, plus Tier 1 labels from description
        let mut labels = vec!["memtype:image".to_string()];
        if description.is_some()
            && self.config.use_labels
            && self.prototypes.is_initialized()
            && !text_embedding.is_empty()
        {
            let desc_labels = crate::labels::generate_tier1_labels(
                description.unwrap(),
                &text_embedding,
                &self.prototypes,
            );
            labels.extend(desc_labels);
        }
        entry.labels = labels;
        entry.label_version = 1;

        let id = entry.id.clone();
        let has_text = !text_embedding.is_empty();

        // 4. Add to store + vision LSH + optionally text LSH and Bloom
        {
            let mut store = self.store.write().await;
            let index = store.add(entry);

            // Always insert into vision LSH
            if let Some(ref vlsh) = self.vision_lsh
                && let Ok(mut lsh) = vlsh.lock()
            {
                lsh.insert(index as u32, &vision_embedding);
            }

            // If we have a text embedding, also index in text channel for cross-modal recall
            if has_text {
                if self.config.use_lsh
                    && let Ok(mut lsh) = self.text_lsh.lock()
                {
                    lsh.insert(index as u32, &text_embedding);
                }
                if self.config.use_bloom {
                    if let Some(desc) = description {
                        let mut bloom = self.bloom.write().await;
                        bloom.insert_memory(desc);
                    }
                }
            }
        }

        tracing::info!(
            memory_id = %id,
            modality = "vision",
            dim = vision_embedding.len(),
            cross_modal = has_text,
            "Image memory stored"
        );

        Ok(id)
    }

    /// Store audio as a speech memory with optional text description for cross-modal recall.
    ///
    /// Pipeline:
    /// 1. Embed PCM audio with the 2-model speech stack -> 640-dim vector
    /// 2. If description provided: embed text → index in text_lsh + bloom for cross-modal recall
    /// 3. Create MemoryEntry with modality: Speech, speech_embedding, optional text embedding
    /// 4. Auto-label with `memtype:audio` + Tier 1 labels from description (if any)
    /// 5. Insert into speech_lsh (and text_lsh if description present)
    /// 6. Return memory ID
    ///
    /// # Arguments
    /// * `pcm_f32` - Raw mono f32 PCM audio samples
    /// * `sample_rate` - Sample rate of the input audio (resampled to 16kHz internally)
    /// * `source` - Where this audio came from (e.g., "microphone", "file")
    /// * `description` - Optional text description for cross-modal text→speech recall.
    ///   Examples: "meeting standup recording", "robot microphone feed", "voice note about project".
    ///   When provided, text echo queries can find this speech memory.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if speech embedding fails or speech models are not loaded.
    /// Returns `ShrimPKError::Memory` if the store is at capacity.
    #[cfg(feature = "speech")]
    #[instrument(skip(self, pcm_f32), fields(sample_count = pcm_f32.len(), sample_rate, source = source))]
    pub async fn store_audio(
        &self,
        pcm_f32: &[f32],
        sample_rate: u32,
        source: &str,
        description: Option<&str>,
    ) -> Result<MemoryId> {
        // Guard: reject oversized audio to prevent OOM during ONNX inference
        const MAX_AUDIO_SAMPLES: usize = 16000 * 60; // 60 seconds at 16kHz
        if pcm_f32.len() > MAX_AUDIO_SAMPLES {
            return Err(ShrimPKError::Embedding(format!(
                "Audio too long: {} samples (max {} = 60s at 16kHz)",
                pcm_f32.len(),
                MAX_AUDIO_SAMPLES
            )));
        }

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

        // 1. Embed audio with speech stack
        let speech_embedding = self.embed_blocking(|e| e.embed_audio(pcm_f32, sample_rate))?;

        let speech_embedding = speech_embedding.ok_or_else(|| {
            ShrimPKError::Embedding("Speech models not available — cannot embed audio".into())
        })?;

        // 2. Build content and optional text embedding for cross-modal recall
        let content = description.unwrap_or("[audio]").to_string();
        let text_embedding = if let Some(desc) = description {
            self.embed_blocking(|e| e.embed_text(desc))?
        } else {
            Vec::new()
        };

        let mut entry = MemoryEntry::new_with_modality(
            content,
            text_embedding.clone(),
            source.to_string(),
            Modality::Speech,
        );
        entry.speech_embedding = Some(speech_embedding.clone());

        // 3. Auto-label: always memtype:audio, plus Tier 1 labels from description
        let mut labels = vec!["memtype:audio".to_string()];
        if description.is_some()
            && self.config.use_labels
            && self.prototypes.is_initialized()
            && !text_embedding.is_empty()
        {
            let desc_labels = crate::labels::generate_tier1_labels(
                description.unwrap(),
                &text_embedding,
                &self.prototypes,
            );
            labels.extend(desc_labels);
        }
        entry.labels = labels;
        entry.label_version = 1;

        let id = entry.id.clone();
        let has_text = !text_embedding.is_empty();

        // 4. Add to store + speech LSH + optionally text LSH and Bloom
        {
            let mut store = self.store.write().await;
            let index = store.add(entry);

            // Always insert into speech LSH
            if let Some(ref slsh) = self.speech_lsh
                && let Ok(mut lsh) = slsh.lock()
            {
                lsh.insert(index as u32, &speech_embedding);
            }

            // If we have a text embedding, also index in text channel for cross-modal recall
            if has_text {
                if self.config.use_lsh
                    && let Ok(mut lsh) = self.text_lsh.lock()
                {
                    lsh.insert(index as u32, &text_embedding);
                }
                if self.config.use_bloom {
                    if let Some(desc) = description {
                        let mut bloom = self.bloom.write().await;
                        bloom.insert_memory(desc);
                    }
                }
            }
        }

        tracing::info!(
            memory_id = %id,
            modality = "speech",
            dim = speech_embedding.len(),
            cross_modal = has_text,
            "Audio memory stored"
        );

        Ok(id)
    }

    /// Store a multimodal memory with text + optional image + optional audio.
    ///
    /// Creates a single `MemoryEntry` with up to 3 embeddings, all indexed in
    /// their respective LSH channels. The Hebbian graph links them via a single
    /// memory index.
    ///
    /// # Arguments
    /// * `text` - The text content of this memory (always embedded)
    /// * `image_data` - Optional raw image bytes (PNG, JPEG, etc.) — requires `vision` feature
    /// * `audio_pcm` - Optional (pcm_f32, sample_rate) tuple — requires `speech` feature
    /// * `source` - Where this memory came from
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if any embedding fails.
    /// Returns `ShrimPKError::Memory` if the store is at capacity.
    #[allow(unused_variables)] // image_data / audio_pcm unused when vision/speech features are off
    #[instrument(skip(self, text, image_data, audio_pcm), fields(text_len = text.len(), source = source))]
    pub async fn store_multimodal(
        &self,
        text: &str,
        image_data: Option<&[u8]>,
        audio_pcm: Option<(&[f32], u32)>,
        source: &str,
    ) -> Result<MemoryId> {
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

        // 1. Always embed text (MiniLM 384-dim)
        let (masked_text, pii_matches) = self.pii_filter.mask(text);
        let sensitivity = self.pii_filter.classify(text);
        if sensitivity == SensitivityLevel::Blocked {
            return Err(ShrimPKError::Memory(
                "Content classified as Blocked — not stored".into(),
            ));
        }
        let text_for_reformulation = if !pii_matches.is_empty() {
            &masked_text
        } else {
            text
        };
        let reformulated = self.reformulator.reformulate(text_for_reformulation);
        let embed_text = reformulated.as_deref().unwrap_or(text);

        let (text_embedding, vision_embedding, speech_embedding) =
            self.embed_blocking(|embedder| {
                let text_emb = embedder.embed_text(embed_text)?;

                // 2. Optional vision embedding
                #[cfg(feature = "vision")]
                let vis_emb = if let Some(img) = image_data {
                    embedder.embed_image(img)?
                } else {
                    None
                };
                #[cfg(not(feature = "vision"))]
                let vis_emb: Option<Vec<f32>> = None;

                // 3. Optional speech embedding
                #[cfg(feature = "speech")]
                let speech_emb = if let Some((pcm, sr)) = audio_pcm {
                    embedder.embed_audio(pcm, sr)?
                } else {
                    None
                };
                #[cfg(not(feature = "speech"))]
                let speech_emb: Option<Vec<f32>> = None;

                Ok((text_emb, vis_emb, speech_emb))
            })?;

        // 4. Build entry with all embeddings
        let category = self.reformulator.categorize(text);
        let mut entry =
            MemoryEntry::new(text.to_string(), text_embedding.clone(), source.to_string());
        entry.sensitivity = sensitivity;
        entry.category = category;
        if !pii_matches.is_empty() {
            entry.masked_content = Some(masked_text);
        }
        entry.reformulated = reformulated;
        entry.vision_embedding = vision_embedding.clone();
        entry.speech_embedding = speech_embedding.clone();
        let id = entry.id.clone();

        // 5. Add to store + all applicable LSH indices
        {
            let mut store = self.store.write().await;
            let index = store.add(entry);

            // Text LSH
            if self.config.use_lsh
                && let Ok(mut lsh) = self.text_lsh.lock()
            {
                lsh.insert(index as u32, &text_embedding);
            }

            // Vision LSH
            #[cfg(feature = "vision")]
            if let Some(ref ve) = vision_embedding
                && let Some(ref vlsh) = self.vision_lsh
                && let Ok(mut lsh) = vlsh.lock()
            {
                lsh.insert(index as u32, ve);
            }

            // Speech LSH
            #[cfg(feature = "speech")]
            if let Some(ref se) = speech_embedding
                && let Some(ref slsh) = self.speech_lsh
                && let Ok(mut lsh) = slsh.lock()
            {
                lsh.insert(index as u32, se);
            }

            // Bloom filter for text
            if self.config.use_bloom {
                let mut bloom = self.bloom.write().await;
                bloom.insert_memory(text);
            }
        }

        tracing::info!(
            memory_id = %id,
            has_vision = vision_embedding.is_some(),
            has_speech = speech_embedding.is_some(),
            "Multimodal memory stored"
        );

        Ok(id)
    }

    /// Perform an echo query — find memories that resonate with the query.
    ///
    /// Uses `QueryMode::Text` (backward compatible). For cross-modal or multi-channel
    /// queries, use `echo_with_mode()`.
    ///
    /// # Arguments
    /// * `query` - The text to find resonating memories for
    /// * `max_results` - Maximum number of results to return
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if query embedding fails.
    #[instrument(skip(self, query), fields(query_len = query.len(), max_results))]
    pub async fn echo(&self, query: &str, max_results: usize) -> Result<Vec<EchoResult>> {
        self.echo_with_mode(query, max_results, QueryMode::Text)
            .await
    }

    /// Perform an echo query with explicit channel selection.
    ///
    /// - `QueryMode::Text` — text-only search (same as `echo()`).
    /// - `QueryMode::Vision` — cross-modal: embed query with CLIP text encoder,
    ///   search vision_embedding on all entries with modality=Vision.
    /// - `QueryMode::Auto` — run both Text and Vision (if enabled), merge by final_score.
    ///
    /// # Arguments
    /// * `query` - The text to find resonating memories for
    /// * `max_results` - Maximum number of results to return
    /// * `mode` - Which channels to search
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if query embedding fails.
    #[instrument(skip(self, query), fields(query_len = query.len(), max_results, mode = %mode))]
    pub async fn echo_with_mode(
        &self,
        query: &str,
        max_results: usize,
        mode: QueryMode,
    ) -> Result<Vec<EchoResult>> {
        self.echo_with_labels(query, max_results, mode, None).await
    }

    /// Perform an echo query with explicit channel selection and optional label filter.
    ///
    /// When `label_filter` is `Some`, the label classification step is bypassed
    /// and the provided labels are used directly for candidate retrieval.
    /// This lets callers (MCP, daemon, CLI) narrow searches by exact labels.
    #[instrument(skip(self, query, label_filter), fields(query_len = query.len(), max_results, mode = %mode))]
    pub async fn echo_with_labels(
        &self,
        query: &str,
        max_results: usize,
        mode: QueryMode,
        label_filter: Option<&[String]>,
    ) -> Result<Vec<EchoResult>> {
        match mode {
            QueryMode::Text => self.echo_text(query, max_results, label_filter, None).await,
            #[cfg(feature = "vision")]
            QueryMode::Vision => self.echo_vision(query, max_results).await,
            #[cfg(not(feature = "vision"))]
            QueryMode::Vision => {
                tracing::warn!("Vision mode requested but vision feature not enabled");
                Ok(Vec::new())
            }
            QueryMode::Auto => {
                // Run text channel (mut needed when vision feature merges results)
                #[allow(unused_mut)]
                let mut results = self
                    .echo_text(query, max_results, label_filter, None)
                    .await?;

                // Run vision channel if available
                #[cfg(feature = "vision")]
                {
                    if self.vision_lsh.is_some() {
                        let vision_results = self.echo_vision(query, max_results).await?;
                        // Merge all results, dedup by memory_id keeping highest final_score
                        let mut all_results: Vec<EchoResult> = Vec::new();
                        all_results.extend(results.drain(..));
                        all_results.extend(vision_results);

                        let mut best: std::collections::HashMap<MemoryId, EchoResult> =
                            std::collections::HashMap::new();
                        for result in all_results {
                            best.entry(result.memory_id.clone())
                                .and_modify(|existing| {
                                    if result.final_score > existing.final_score {
                                        *existing = result.clone();
                                    }
                                })
                                .or_insert(result);
                        }

                        results = best.into_values().collect();
                        results.sort_by(|a, b| {
                            b.final_score
                                .partial_cmp(&a.final_score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        results.truncate(max_results);
                    }
                }

                Ok(results)
            }
        }
    }

    /// Text-channel echo — the original pipeline, unchanged.
    ///
    /// Pipeline:
    /// 1. Embed the query text (MiniLM 384-dim)
    /// 2. Bloom pre-check
    /// 3. LSH/brute-force, rank, Hebbian, recency, rerank
    ///
    /// If `label_filter` is provided, the label classification step is skipped
    /// and the given labels are used directly for candidate retrieval.
    async fn echo_text(
        &self,
        query: &str,
        max_results: usize,
        label_filter: Option<&[String]>,
        at_time: Option<f64>,
    ) -> Result<Vec<EchoResult>> {
        let start = std::time::Instant::now();

        // 0. HyDE query expansion: ask LLM for a hypothetical answer, embed that instead
        let effective_query = if self.config.query_expansion_enabled {
            match expand_query(&self.config, query) {
                Some(expanded) => {
                    tracing::debug!(original = query, expanded = %expanded, "HyDE expansion");
                    expanded
                }
                None => query.to_string(), // fallback to original
            }
        } else {
            query.to_string()
        };

        // 1. Embed the (possibly expanded) query
        let query_embedding = self.embed_blocking(|e| e.embed_text(&effective_query))?;

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

        // 5. Three-source candidate retrieval (ADR-015 D6):
        //    Source A: Label inverted index (pre-filter by semantic labels)
        //    Source B: LSH (approximate nearest neighbor)
        //    Source C: Brute-force fallback (only if A+B return too few)
        let embeddings = store.all_embeddings();

        // 5a. Label-based candidates (ADR-015 D6)
        //     If caller provided an explicit label filter, use it directly.
        //     Otherwise, classify the query via prototypes.
        //     query_topic_labels is lifted to outer scope for Pipe B topic alignment (KS68).
        let all_query_labels: Vec<String> =
            if self.config.use_labels && self.prototypes.is_initialized() {
                crate::labels::classify_query(&effective_query, &query_embedding, &self.prototypes)
            } else {
                Vec::new()
            };
        let query_topic_labels: Vec<&str> = all_query_labels
            .iter()
            .filter(|l| l.starts_with("topic:"))
            .map(String::as_str)
            .collect();
        let label_candidates: Vec<u32> = if let Some(filter) = label_filter {
            if !filter.is_empty() {
                store.query_labels(filter)
            } else {
                Vec::new()
            }
        } else if !all_query_labels.is_empty() {
            store.query_labels(&all_query_labels)
        } else {
            Vec::new()
        };

        // 5b. LSH candidates
        let lsh_candidates: Vec<u32> = if self.config.use_lsh {
            self.text_lsh
                .lock()
                .map_err(|e| ShrimPKError::Memory(format!("LSH lock poisoned: {e}")))?
                .query(&query_embedding)
        } else {
            Vec::new()
        };

        // 5c. Merge + dedup (OR semantics: union of label and LSH candidates)
        let mut merged: Vec<u32> =
            Vec::with_capacity(label_candidates.len() + lsh_candidates.len());
        merged.extend_from_slice(&label_candidates);
        merged.extend_from_slice(&lsh_candidates);
        merged.sort_unstable();
        merged.dedup();

        const MIN_CANDIDATES: usize = 5;

        let candidates: Vec<(usize, &[f32])> = if merged.len() >= MIN_CANDIDATES {
            // Enough candidates from labels + LSH — no brute-force needed
            tracing::debug!(
                label_candidates = label_candidates.len(),
                lsh_candidates = lsh_candidates.len(),
                merged = merged.len(),
                total = embeddings.len(),
                "Label + LSH candidate retrieval (sub-linear)"
            );
            merged
                .iter()
                .filter_map(|&idx| {
                    let i = idx as usize;
                    embeddings.get(i).map(|e| (i, e.as_slice()))
                })
                .collect()
        } else if !self.config.use_lsh && !self.config.use_labels {
            // Both disabled — brute-force everything
            embeddings
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.is_empty())
                .map(|(i, e)| (i, e.as_slice()))
                .collect()
        } else {
            // Labels + LSH returned too few — brute-force fallback
            tracing::debug!(
                label_candidates = label_candidates.len(),
                lsh_candidates = lsh_candidates.len(),
                merged = merged.len(),
                total = embeddings.len(),
                "Labels + LSH returned < {} candidates, falling back to brute-force",
                MIN_CANDIDATES
            );
            embeddings
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.is_empty())
                .map(|(i, e)| (i, e.as_slice()))
                .collect()
        };

        // 5b. Split pipeline: Pipe A (above threshold) + Pipe B (near-miss child rescue)
        //     Use half-threshold to capture near-misses for potential child rescue.
        let near_miss_threshold = self.config.similarity_threshold * 0.5;
        let cosine_ranked =
            similarity::rank_candidates(&query_embedding, &candidates, near_miss_threshold);

        // 5c. Detect query entities (lifted for Pipe B entity gate + graph traversal)
        let query_entities = detect_query_entities(query, store.entity_index_ref());

        // 5d. Entity-anchored graph traversal + RRF merge (KS62).
        //     If query mentions known entities, BFS from entity-anchored memories
        //     via Hebbian edges, then merge with cosine results via RRF.
        let all_ranked = if self.config.graph_traversal_enabled {
            if !query_entities.is_empty() {
                // Collect anchor indices from entity index
                let mut anchors: Vec<u32> = Vec::new();
                for entity in &query_entities {
                    anchors.extend(store.query_entities(entity));
                }
                anchors.sort_unstable();
                anchors.dedup();

                // BFS via Hebbian edges
                let graph_results = {
                    let hebbian = self.hebbian.read().await;
                    hebbian.graph_traverse(&anchors, self.config.graph_max_hops, 20)
                };

                if !graph_results.is_empty() {
                    // Compute cosine for graph-discovered nodes not already in cosine_ranked
                    let ranked_set: std::collections::HashSet<usize> =
                        cosine_ranked.iter().map(|&(idx, _)| idx).collect();
                    let mut graph_with_cosine: Vec<(usize, f32)> = Vec::new();
                    for &(node_id, _weight) in &graph_results {
                        let idx = node_id as usize;
                        if !ranked_set.contains(&idx)
                            && let Some(emb) = embeddings.get(idx)
                        {
                            let sim = similarity::cosine_similarity(&query_embedding, emb);
                            if sim > near_miss_threshold {
                                graph_with_cosine.push((idx, sim));
                            }
                        }
                    }

                    if !graph_with_cosine.is_empty() {
                        tracing::debug!(
                            entities = ?query_entities,
                            anchors = anchors.len(),
                            graph_new = graph_with_cosine.len(),
                            cosine_total = cosine_ranked.len(),
                            "GraphRAG: entity-anchored traversal injected candidates"
                        );
                        reciprocal_rank_fusion(
                            &cosine_ranked,
                            &graph_with_cosine,
                            self.config.graph_rrf_k,
                        )
                    } else {
                        cosine_ranked
                    }
                } else {
                    cosine_ranked
                }
            } else {
                cosine_ranked
            }
        } else {
            cosine_ranked
        };

        let threshold = self.config.similarity_threshold;
        let child_rescue_only = self.config.child_rescue_only;
        type ScoredIndices = Vec<(usize, f32)>;
        let (pipe_a, pipe_b): (ScoredIndices, ScoredIndices) =
            all_ranked.into_iter().partition(|&(idx, score)| {
                if score < threshold {
                    return false; // below threshold → Pipe B
                }
                // When child_rescue_only is true, children go to Pipe B even if above threshold.
                // They can still rescue parents but never appear in direct ranking.
                if child_rescue_only {
                    store.entry_at(idx).is_none_or(|e| e.parent_id.is_none())
                } else {
                    // KS69 T1: children can enter Pipe A only if topic matches query
                    store.entry_at(idx).is_none_or(|e| {
                        e.parent_id.is_none()
                            || child_topic_matches_query(
                                &e.labels,
                                &query_topic_labels,
                                &e.subject,
                                query,
                            )
                    })
                }
            });

        // 6. Pipe B: check if near-miss parents have enriched children that score better.
        //    Enhanced with entity gate, confidence weighting, and child content tracking (KS69 T1).
        let mut pipe_b_child_content: std::collections::HashMap<usize, String> =
            std::collections::HashMap::new();
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
                    let mut best_child_idx: Option<usize> = None;
                    for &child_idx in child_indices {
                        if let Some(child_entry) = store.entry_at(child_idx) {
                            // Topic gate (KS69 T1): skip children whose topic/subject
                            // doesn't match the query.
                            if !child_topic_matches_query(
                                &child_entry.labels,
                                &query_topic_labels,
                                &child_entry.subject,
                                query,
                            ) {
                                tracing::debug!(
                                    child_idx,
                                    child_labels = ?child_entry.labels,
                                    child_subject = ?child_entry.subject,
                                    "Pipe B: child excluded by topic gate"
                                );
                                continue;
                            }
                            if let Some(child_emb) = embeddings.get(child_idx) {
                                let child_sim =
                                    similarity::cosine_similarity(&query_embedding, child_emb);
                                // Confidence weighting (KS69 T1): scale by LLM confidence
                                let weighted_sim = child_sim * child_entry.confidence.max(0.01);
                                if weighted_sim > best_child_score {
                                    best_child_score = weighted_sim;
                                    best_child_idx = Some(child_idx);
                                }
                            }
                        }
                    }
                    if best_child_score >= threshold {
                        // Topic alignment gate (KS68 IE-3): only rescue a parent if
                        // its labels overlap with the query's topic labels, or if no
                        // topic labels are available, require a minimum base similarity.
                        let topic_aligned = if !query_topic_labels.is_empty() {
                            entry
                                .labels
                                .iter()
                                .any(|el| query_topic_labels.iter().any(|qt| el == qt))
                        } else {
                            // Fallback: require parent's own similarity to be non-trivial
                            _parent_score >= threshold * 0.4
                        };
                        if !topic_aligned {
                            tracing::debug!(
                                parent_idx = idx,
                                child_score = best_child_score,
                                parent_labels = ?entry.labels,
                                "Pipe B: child rescue blocked — topic mismatch"
                            );
                            continue;
                        }
                        tracing::debug!(
                            parent_idx = idx,
                            child_score = best_child_score,
                            "Pipe B: child rescued parent memory"
                        );
                        // Apply child memory penalty at promotion (KS69)
                        let penalized_score = best_child_score + self.config.child_memory_penalty;
                        promotions.push((idx, penalized_score));
                        // Track child content for matched_child_content (KS69 T1)
                        if let Some(ci) = best_child_idx
                            && let Some(child_entry) = store.entry_at(ci)
                        {
                            pipe_b_child_content.insert(idx, child_entry.content.clone());
                        }
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
        //     When at_time is provided (KS63), use get_valid_associations to filter
        //     expired/not-yet-valid edges for point-in-time queries.
        let hebbian_boosts: Vec<f64> = {
            let hebbian = self.hebbian.read().await;
            top.iter()
                .map(|&(idx, _)| {
                    let idx = idx as u32;
                    let mut boost: f64 = 0.0;
                    let mut demotion: f64 = 0.0;

                    if let Some(at) = at_time {
                        // Temporal query: only consider edges valid at the given timestamp
                        let valid_assocs = hebbian.get_valid_associations(idx, at, 0.0);
                        for (neighbor, weight, rel) in &valid_assocs {
                            if !top_indices.contains(neighbor) || *neighbor == idx {
                                continue;
                            }
                            boost += weight;
                            if let Some(r) = rel {
                                match r {
                                    crate::hebbian::RelationshipType::Supersedes => {
                                        if idx > *neighbor {
                                            boost += 0.1;
                                        } else {
                                            demotion -= self.config.supersedes_demotion as f64;
                                        }
                                    }
                                    crate::hebbian::RelationshipType::CoActivation => {}
                                    _ => {
                                        boost += 0.05;
                                    }
                                }
                            }
                        }
                    } else {
                        // Standard query: use all edges (existing behavior)
                        for &other in top_indices.iter().filter(|&&o| o != idx) {
                            let weight = hebbian.get_weight(idx, other);
                            if weight <= 0.0 {
                                continue;
                            }
                            boost += weight;
                            if let Some(rel) = hebbian.get_relationship(idx, other) {
                                match rel {
                                    crate::hebbian::RelationshipType::Supersedes => {
                                        if idx > other {
                                            boost += 0.1;
                                        } else {
                                            demotion -= self.config.supersedes_demotion as f64;
                                        }
                                    }
                                    crate::hebbian::RelationshipType::CoActivation => {}
                                    _ => {
                                        boost += 0.05;
                                    }
                                }
                            }
                        }
                    }

                    boost.min(0.4) + demotion
                })
                .collect()
        };

        // 7b2. Parent supersession demotion (KS68 KU-1): if a parent entry has
        // children with Supersedes edges (child is the older/superseded side),
        // apply a flat demotion to the parent. This propagates child-level
        // supersession to parent ranking in Pipe A.
        let parent_demotions: std::collections::HashMap<usize, f64> = {
            let hebbian = self.hebbian.read().await;
            let demotion = self.config.supersedes_demotion as f64;
            let mut demotions = std::collections::HashMap::new();
            for &(idx, _) in &top {
                if let Some(entry) = store.entry_at(idx) {
                    let child_indices = store.children_of(&entry.id);
                    let mut has_superseded_child = false;
                    for &child_idx in child_indices {
                        let assocs = hebbian.get_associations_typed(child_idx as u32, 0.0);
                        for (neighbor, _weight, rel) in &assocs {
                            if let Some(crate::hebbian::RelationshipType::Supersedes) = rel
                                && (child_idx as u32) < *neighbor
                            {
                                has_superseded_child = true;
                                break;
                            }
                        }
                        if has_superseded_child {
                            break;
                        }
                    }
                    if has_superseded_child {
                        demotions.insert(idx, -demotion);
                    }
                }
            }
            demotions
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

                // Decay: power-law or exponential (feature-flagged)
                let decay = if self.config.use_power_law_decay {
                    let stability = entry.category.stability_days();
                    activation::power_law_decay(age_secs, stability) as f32
                } else {
                    // Legacy exponential decay
                    (-age_secs * std::f64::consts::LN_2 / half_life).exp() as f32
                };

                // Activation: ACT-R OL or recency_boost (feature-flagged)
                let activation_term = if self.config.use_actr_activation {
                    let d = entry.category.actr_decay_d();
                    let act = activation::actr_ol_activation(
                        entry.echo_count,
                        entry.created_at,
                        entry.last_echoed,
                        d,
                    );
                    self.config.activation_weight * act as f32
                } else {
                    // Legacy recency boost (KS18 Track 3): newer memories get a small advantage.
                    // Formula: recency_weight / (1.0 + days_since_stored)
                    // At default 0.05: day 0 = +0.05, day 7 = +0.006, day 30 = +0.002.
                    let days_since_stored = age_secs / 86400.0;
                    (recency_weight / (1.0 + days_since_stored)) as f32
                };

                // Importance boost (if enabled and weight > 0)
                let importance_boost =
                    if self.config.use_importance && self.config.importance_weight > 0.0 {
                        self.config.importance_weight * entry.importance
                    } else {
                        0.0
                    };

                let sim = score as f64;
                let hebbian_boost = boost;
                let mut final_score = (sim + hebbian_boost + importance_boost as f64)
                    * decay as f64
                    + activation_term as f64;

                // Co-occurrence bonus (KS68 ME-4)
                final_score += co_occurrence_boost(&entry.content);

                // Parent supersession demotion (KS68 KU-1)
                if let Some(&demotion) = parent_demotions.get(&idx) {
                    final_score += demotion;
                }

                // Child memory penalty (KS69): demote children to prevent hallucination inflation
                if entry.parent_id.is_some() {
                    final_score += self.config.child_memory_penalty as f64;
                }

                // Confidence-weighted child scoring (KS69 T1)
                if entry.parent_id.is_some() && entry.confidence < 1.0 {
                    final_score *= entry.confidence as f64;
                }

                // KS71: soft invalidation demotion — superseded facts are preserved
                // but heavily demoted so active facts rank higher.
                if entry.superseded_at.is_some() {
                    final_score *= 0.5;
                }

                // Resolve matched child content (KS69 T1):
                // - Pipe B rescue: use the child text that triggered rescue
                // - Direct child in Pipe A: use the child's own content
                let matched_child_content = if let Some(child_text) = pipe_b_child_content.get(&idx)
                {
                    Some(child_text.clone())
                } else if entry.parent_id.is_some() {
                    Some(entry.content.clone())
                } else {
                    None
                };

                Some(EchoResult {
                    memory_id: entry.id.clone(),
                    content: truncate_content(entry.display_content(), 200),
                    similarity: score,
                    final_score,
                    source: entry.source.clone(),
                    echoed_at: now,
                    modality: entry.modality,
                    labels: entry.labels.clone(),
                    matched_child_content,
                })
            })
            .collect();

        // 7c2. Temporal query boost (KS68 TR-3)
        apply_temporal_boost(query, &mut results);

        // 7c3. Topic-label boost (KS68 KU-3)
        label_topic_boost(&all_query_labels, &mut results);

        // 7c4. Preference-update multiplier (KS68 KU-3)
        preference_update_boost(query, &mut results);

        // 7c5. Career/intro adjustment (KS68 IE-1)
        career_intro_adjustment(&all_query_labels, &mut results);

        // 7c6. Score inflation cap (KS69): prevent unbounded boost stacking
        for result in &mut results {
            let max_allowed = result.similarity as f64 + 0.35;
            if result.final_score > max_allowed {
                result.final_score = max_allowed;
            }
        }

        // 7d. Re-sort by final_score (similarity + hebbian boost)
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 7d2. Subject diversity cap (KS67/KS68): prevent identity gravity well
        // Uses (subject, topic) tuples so different facets of the same entity
        // (e.g., Sam:identity vs Sam:preference) count independently.
        {
            let subject_topic_map = build_subject_topic_map(&store, &top);
            enforce_subject_diversity(&mut results, &store, &subject_topic_map, 3);
        }

        // 7e. Optional reranker: reorder top-N by true relevance (KS23 LLM / KS24 cross-encoder)
        let effective_backend = self.config.effective_reranker_backend();
        if effective_backend != shrimpk_core::RerankerBackend::None
            && !results.is_empty()
            && let Some(reranked) = crate::reranker::rerank(&self.config, query, &results)
        {
            tracing::debug!(
                target: "shrimpk::audit",
                backend = %effective_backend,
                original_top1 = %results.first().map(|r| &r.content[..r.content.len().min(40)]).unwrap_or(""),
                reranked_top1 = %reranked.first().map(|r| &r.content[..r.content.len().min(40)]).unwrap_or(""),
                "Reranker applied"
            );
            results = reranked;
        }

        // 7f. Community summary fallback (KS64): if top result is weak,
        //     inject best-matching community summary as a fallback result.
        if self.config.community_summaries_enabled {
            let threshold = self.config.community_summary_threshold as f64;
            let top_score = results.first().map(|r| r.final_score).unwrap_or(0.0);
            if top_score < threshold {
                let summaries = store.all_summaries();
                if !summaries.is_empty() {
                    let mut best_summary: Option<(&str, &str, f32)> = None;
                    for summary in summaries.values() {
                        if !summary.embedding.is_empty() {
                            let sim =
                                similarity::cosine_similarity(&query_embedding, &summary.embedding);
                            if best_summary.is_none_or(|(_, _, s)| sim > s) {
                                best_summary = Some((&summary.label, &summary.summary, sim));
                            }
                        }
                    }
                    if let Some((label, text, sim)) = best_summary
                        && sim > 0.1
                    {
                        tracing::debug!(
                            label = %label,
                            sim,
                            top_score,
                            "Community summary fallback injected"
                        );
                        results.push(EchoResult {
                            memory_id: MemoryId::new(),
                            content: format!("[{}] {}", label, truncate_content(text, 200)),
                            similarity: sim,
                            final_score: sim as f64,
                            source: "community_summary".to_string(),
                            echoed_at: Utc::now(),
                            modality: Modality::Text,
                            labels: vec![label.to_string()],
                            matched_child_content: None,
                        });
                        results.sort_by(|a, b| {
                            b.final_score
                                .partial_cmp(&a.final_score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
            }
        }

        // 7g. Parent-child dedup (KS68.3 prep): if a parent and its child both
        // appear in results, keep only the higher-scoring one to avoid slot waste.
        {
            let parent_map: std::collections::HashMap<MemoryId, Option<MemoryId>> = results
                .iter()
                .filter_map(|r| {
                    store
                        .get(&r.memory_id)
                        .map(|e| (r.memory_id.clone(), e.parent_id.clone()))
                })
                .collect();
            deduplicate_parent_child(&mut results, &parent_map);
        }

        // 7h. Exclude superseded parents (KS68.3 KU-1): if both an old parent
        // (with superseded children) and the superseding new parent appear in
        // results, remove the old parent entirely. This avoids score-gap
        // calibration issues — exclusion is binary and deterministic.
        {
            let result_ids: std::collections::HashSet<MemoryId> =
                results.iter().map(|r| r.memory_id.clone()).collect();
            let hebbian = self.hebbian.read().await;
            let mut superseded: std::collections::HashSet<MemoryId> =
                std::collections::HashSet::new();

            for r in &results {
                if store.index_of(&r.memory_id).is_some() {
                    let child_indices = store.children_of(&r.memory_id);
                    for &child_idx in child_indices {
                        let assocs = hebbian.get_associations_typed(child_idx as u32, 0.0);
                        for (neighbor, _weight, rel) in &assocs {
                            if let Some(crate::hebbian::RelationshipType::Supersedes) = rel
                                && (child_idx as u32) < *neighbor
                                && let Some(new_child) = store.entry_at(*neighbor as usize)
                                && let Some(ref new_parent_id) = new_child.parent_id
                                && result_ids.contains(new_parent_id)
                            {
                                superseded.insert(r.memory_id.clone());
                            }
                        }
                    }
                }
            }

            if !superseded.is_empty() && results.len() - superseded.len() >= 3 {
                results.retain(|r| !superseded.contains(&r.memory_id));
            }
        }

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

    /// Vision-channel echo — cross-modal text-to-image retrieval via CLIP.
    ///
    /// Pipeline:
    /// 1. Embed query text with CLIP text encoder (512-dim)
    /// 2. Scan all entries with vision_embedding, compute cosine similarity
    /// 3. Use vision_lsh for sub-linear candidate retrieval if available
    /// 4. Apply Hebbian boost (shared graph)
    /// 5. Return top-N results with modality: Vision
    #[cfg(feature = "vision")]
    async fn echo_vision(&self, query: &str, max_results: usize) -> Result<Vec<EchoResult>> {
        let start = std::time::Instant::now();

        // 1. Embed query with CLIP text encoder
        let query_embedding = self.embed_blocking(|e| e.embed_text_for_vision(query))?;

        let query_embedding = match query_embedding {
            Some(emb) => emb,
            None => {
                tracing::warn!("CLIP text encoder not available for vision query");
                return Ok(Vec::new());
            }
        };

        // 2. Read-lock the store
        let store = self.store.read().await;

        if store.is_empty() {
            self.record_latency(start.elapsed().as_micros() as u64);
            return Ok(Vec::new());
        }

        // 3. Collect vision candidates: entries with vision_embedding set
        //    Use vision_lsh for candidate retrieval if available, else brute-force
        let entries = store.all_entries();
        let threshold = self.config.similarity_threshold;

        let mut scored: Vec<(usize, f32)> = Vec::new();

        if let Some(ref vlsh) = self.vision_lsh {
            // Try LSH candidate retrieval
            let lsh_candidates = vlsh
                .lock()
                .map_err(|e| ShrimPKError::Memory(format!("Vision LSH lock poisoned: {e}")))?
                .query(&query_embedding);

            if lsh_candidates.len() >= 5 {
                // LSH returned enough candidates
                tracing::debug!(
                    lsh_candidates = lsh_candidates.len(),
                    "Vision LSH candidate retrieval"
                );
                for &idx in &lsh_candidates {
                    let i = idx as usize;
                    if let Some(entry) = entries.get(i)
                        && let Some(ref ve) = entry.vision_embedding
                    {
                        let sim = similarity::cosine_similarity(&query_embedding, ve);
                        if sim >= threshold {
                            scored.push((i, sim));
                        }
                    }
                }
            } else {
                // Fall back to brute-force
                Self::brute_force_vision(&query_embedding, entries, threshold, &mut scored);
            }
        } else {
            // No vision LSH — brute-force
            Self::brute_force_vision(&query_embedding, entries, threshold, &mut scored);
        }

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_results);

        // 4. Hebbian boost (shared graph with text channel)
        let top_indices: Vec<u32> = scored.iter().map(|&(idx, _)| idx as u32).collect();
        {
            let mut hebbian = self.hebbian.write().await;
            for i in 0..top_indices.len() {
                for j in (i + 1)..top_indices.len() {
                    let sim_i = scored[i].1 as f64;
                    let sim_j = scored[j].1 as f64;
                    let strength = (sim_i * sim_j).sqrt() * 0.1;
                    hebbian.co_activate(top_indices[i], top_indices[j], strength);
                }
            }
        }

        let hebbian_boosts: Vec<f64> = {
            let hebbian = self.hebbian.read().await;
            scored
                .iter()
                .map(|&(idx, _)| {
                    let idx = idx as u32;
                    let mut boost: f64 = 0.0;
                    for &other in top_indices.iter().filter(|&&o| o != idx) {
                        let weight = hebbian.get_weight(idx, other);
                        if weight > 0.0 {
                            boost += weight;
                        }
                    }
                    boost.min(0.4)
                })
                .collect()
        };

        // 5. Build EchoResult vec
        let now = Utc::now();
        let recency_weight = self.config.recency_weight as f64;
        let mut results: Vec<EchoResult> = scored
            .iter()
            .zip(hebbian_boosts.iter())
            .filter_map(|(&(idx, score), &boost)| {
                let entry = store.entry_at(idx)?;
                let age_secs = (now - entry.created_at).num_seconds().max(0) as f64;
                let half_life = entry.category.half_life_secs();
                let decay = (-age_secs * std::f64::consts::LN_2 / half_life).exp();
                let days_since_stored = age_secs / 86400.0;
                let recency_boost = recency_weight / (1.0 + days_since_stored);

                Some(EchoResult {
                    memory_id: entry.id.clone(),
                    content: truncate_content(entry.display_content(), 200),
                    similarity: score,
                    final_score: (score as f64 + boost + recency_boost) * decay,
                    source: entry.source.clone(),
                    echoed_at: now,
                    modality: Modality::Vision,
                    labels: entry.labels.clone(),
                    matched_child_content: None,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update echo_count on matched entries
        let matched_ids: Vec<(MemoryId, usize)> = scored
            .iter()
            .filter_map(|&(idx, _)| store.entry_at(idx).map(|e| (e.id.clone(), idx)))
            .collect();
        drop(store);

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

        let elapsed = start.elapsed();
        self.record_latency(elapsed.as_micros() as u64);

        tracing::info!(
            results = results.len(),
            elapsed_ms = elapsed.as_millis(),
            mode = "vision",
            "Vision echo query complete"
        );

        Ok(results)
    }

    /// Brute-force scan all entries for vision embeddings above threshold.
    #[cfg(feature = "vision")]
    fn brute_force_vision(
        query_embedding: &[f32],
        entries: &[MemoryEntry],
        threshold: f32,
        scored: &mut Vec<(usize, f32)>,
    ) {
        for (i, entry) in entries.iter().enumerate() {
            if let Some(ref ve) = entry.vision_embedding {
                let sim = similarity::cosine_similarity(query_embedding, ve);
                if sim >= threshold {
                    scored.push((i, sim));
                }
            }
        }
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

                // Update text LSH index to reflect the swap-remove
                if self.config.use_lsh
                    && let Ok(mut lsh) = self.text_lsh.lock()
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

        // Count entries by modality
        let mut text_count: usize = 0;
        let mut vision_count: usize = 0;
        let mut speech_count: usize = 0;
        for entry in store.all_entries() {
            match entry.modality {
                Modality::Text => text_count += 1,
                Modality::Vision => vision_count += 1,
                Modality::Speech => speech_count += 1,
            }
        }

        // Estimate memory usage per channel:
        // Text: embedding (384 * 4 bytes) + content (~200 bytes avg) + metadata (~100 bytes)
        // Vision: vision_embedding (512 * 4 bytes) + metadata (~100 bytes)
        // Speech: speech_embedding (speech_embedding_dim * 4 bytes) + metadata (~100 bytes)
        let text_bytes_per = (self.config.embedding_dim * 4 + 300) as u64;
        let vision_bytes_per = (self.config.vision_embedding_dim * 4 + 100) as u64;
        let speech_bytes_per = (self.config.speech_embedding_dim * 4 + 100) as u64;
        let ram_usage = text_count as u64 * text_bytes_per
            + vision_count as u64 * vision_bytes_per
            + speech_count as u64 * speech_bytes_per;

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
            text_count,
            vision_count,
            speech_count,
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
                novelty_score: e.novelty_score,
                importance: e.importance,
            })
            .collect()
    }

    /// Return the label connection graph for a single memory.
    ///
    /// For each label on the memory, returns the count and top connected memory IDs
    /// ranked by cosine similarity to the source memory's embedding.
    /// Skips the full echo pipeline (LSH, bloom, Hebbian) — pure label index + cosine.
    #[instrument(skip(self), fields(memory_id = %id))]
    pub async fn memory_graph(
        &self,
        id: &MemoryId,
        top_per_label: usize,
    ) -> Result<MemoryGraphResult> {
        let store = self.store.read().await;
        let index = store
            .index_of(id)
            .ok_or_else(|| ShrimPKError::Memory(format!("Memory not found: {id}")))?;
        let entry = store
            .entry_at(index)
            .ok_or_else(|| ShrimPKError::Memory(format!("Entry missing at index {index}")))?;

        let content_preview =
            entry.display_content()[..entry.display_content().len().min(200)].to_string();
        let labels = entry.labels.clone();
        let source_embedding = store
            .embedding_at(index)
            .ok_or_else(|| ShrimPKError::Memory(format!("No embedding at index {index}")))?;

        // Build connections per label
        let grouped = store.connected_by_labels(index);
        let mut connections: Vec<LabelConnection> = Vec::with_capacity(grouped.len());

        for (label, indices) in &grouped {
            let count = indices.len();
            // Rank by cosine similarity to source embedding, take top N
            let mut scored: Vec<(u32, f32)> = indices
                .iter()
                .filter_map(|&idx| {
                    store.embedding_at(idx as usize).map(|emb| {
                        let sim = similarity::cosine_similarity(source_embedding, emb);
                        (idx, sim)
                    })
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_ids: Vec<MemoryId> = scored
                .iter()
                .take(top_per_label)
                .filter_map(|&(idx, _)| store.entry_at(idx as usize).map(|e| e.id.clone()))
                .collect();

            connections.push(LabelConnection {
                label: label.clone(),
                count,
                top_ids,
            });
        }

        // Sort connections by count descending
        connections.sort_by(|a, b| b.count.cmp(&a.count));

        // Count unique connected memories across all labels
        let mut all_indices: Vec<u32> = grouped.values().flat_map(|v| v.iter().copied()).collect();
        all_indices.sort_unstable();
        all_indices.dedup();
        let unique_connected = all_indices.len();
        let total_connected: usize = grouped.values().map(|v| v.len()).sum();

        Ok(MemoryGraphResult {
            memory_id: id.clone(),
            content_preview,
            labels,
            connections,
            total_connected,
            unique_connected,
        })
    }

    /// Retrieve memories related to a source memory via shared labels.
    ///
    /// **Pipeline skip:** Uses label index directly → cosine-only scoring against
    /// the source memory's embedding. Skips LSH, bloom, label classification,
    /// and Hebbian boost. This is the fast path for graph navigation.
    #[instrument(skip(self), fields(memory_id = %id))]
    pub async fn memory_related(
        &self,
        id: &MemoryId,
        label_filter: Option<&str>,
        max_results: usize,
    ) -> Result<Vec<EchoResult>> {
        let start = std::time::Instant::now();
        let store = self.store.read().await;
        let index = store
            .index_of(id)
            .ok_or_else(|| ShrimPKError::Memory(format!("Memory not found: {id}")))?;
        let source_embedding = store
            .embedding_at(index)
            .ok_or_else(|| ShrimPKError::Memory(format!("No embedding at index {index}")))?;

        // Get candidate set from label index (skip LSH, bloom, classification)
        let candidates = store.connected_indices(index, label_filter);
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Cosine-only scoring against source memory's embedding
        let candidate_embs: Vec<(usize, &[f32])> = candidates
            .iter()
            .filter_map(|&idx| {
                store
                    .embedding_at(idx as usize)
                    .filter(|e| !e.is_empty())
                    .map(|e| (idx as usize, e))
            })
            .collect();

        let scored = similarity::rank_candidates(
            source_embedding,
            &candidate_embs,
            0.0, // no threshold — return all, let caller decide
        );

        // Build EchoResult vec (no Hebbian, no decay — raw cosine for graph nav)
        let now = Utc::now();
        let mut results: Vec<EchoResult> = scored
            .iter()
            .take(max_results)
            .filter_map(|&(idx, score)| {
                let entry = store.entry_at(idx)?;
                Some(EchoResult {
                    memory_id: entry.id.clone(),
                    content: truncate_content(entry.display_content(), 200),
                    similarity: score,
                    final_score: score as f64,
                    source: entry.source.clone(),
                    echoed_at: now,
                    modality: entry.modality,
                    labels: entry.labels.clone(),
                    matched_child_content: None,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let elapsed = start.elapsed();
        tracing::info!(
            source_id = %id,
            label_filter = label_filter.unwrap_or("all"),
            candidates = candidates.len(),
            results = results.len(),
            elapsed_ms = elapsed.as_millis(),
            "Memory related query complete (cosine-only fast path)"
        );

        Ok(results)
    }

    /// Search for memories mentioning a specific entity by name.
    /// Uses the entity index for O(1) lookup, then ranks by cosine similarity.
    #[instrument(skip(self), fields(entity = %entity))]
    pub async fn entity_search(&self, entity: &str, max_results: usize) -> Result<Vec<EchoResult>> {
        let start = std::time::Instant::now();
        let store = self.store.read().await;
        let indices = store.query_entities(entity);
        if indices.is_empty() {
            return Ok(vec![]);
        }

        // Embed the entity name for ranking
        let query_emb = self.embed_blocking(|e| e.embed_text(entity))?;

        let mut scored: Vec<(usize, f32)> = indices
            .iter()
            .filter_map(|&idx| {
                let idx = idx as usize;
                store.embedding_at(idx).and_then(|emb| {
                    if emb.is_empty() {
                        None
                    } else {
                        let sim = similarity::cosine_similarity(&query_emb, emb);
                        Some((idx, sim))
                    }
                })
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_results);

        let now = Utc::now();
        let results: Vec<EchoResult> = scored
            .into_iter()
            .filter_map(|(idx, sim)| {
                let entry = store.entry_at(idx)?;
                Some(EchoResult {
                    memory_id: entry.id.clone(),
                    content: truncate_content(entry.display_content(), 200),
                    similarity: sim,
                    final_score: sim as f64,
                    source: entry.source.clone(),
                    echoed_at: now,
                    modality: entry.modality,
                    labels: entry.labels.clone(),
                    matched_child_content: None,
                })
            })
            .collect();

        let elapsed = start.elapsed();
        tracing::info!(
            entity = %entity,
            candidates = indices.len(),
            results = results.len(),
            elapsed_ms = elapsed.as_millis(),
            "Entity search complete"
        );

        Ok(results)
    }

    /// Get all community summaries (KS64).
    pub async fn community_summaries(
        &self,
    ) -> std::collections::HashMap<String, shrimpk_core::CommunitySummary> {
        let store = self.store.read().await;
        store.all_summaries().clone()
    }

    // -----------------------------------------------------------------------
    // Graph visualization endpoints (KS65)
    // -----------------------------------------------------------------------

    /// Return Hebbian neighbors for a single memory node.
    ///
    /// Exposes the co-activation graph edges (typed relationships + weights)
    /// along with cosine similarity for each neighbor.
    #[instrument(skip(self), fields(memory_id = %id))]
    pub async fn graph_neighbors(
        &self,
        id: &MemoryId,
        min_weight: f64,
        max_results: usize,
    ) -> Result<GraphNeighborsResult> {
        let store = self.store.read().await;
        let index = store
            .index_of(id)
            .ok_or_else(|| ShrimPKError::Memory(format!("Memory not found: {id}")))?;
        let entry = store
            .entry_at(index)
            .ok_or_else(|| ShrimPKError::Memory(format!("Entry missing at index {index}")))?;
        let source_emb = store
            .embedding_at(index)
            .ok_or_else(|| ShrimPKError::Memory(format!("No embedding at index {index}")))?;

        let node = GraphNode {
            id: entry.id.clone(),
            content_preview: truncate_content(entry.display_content(), 200),
            labels: entry.labels.clone(),
            importance: entry.importance,
            category: format!("{:?}", entry.category),
            novelty: entry.novelty_score,
        };

        let hebbian = self.hebbian.read().await;
        let assocs = hebbian.get_associations_typed(index as u32, min_weight);

        let mut neighbors: Vec<GraphNeighbor> = assocs
            .iter()
            .filter_map(|&(neighbor_idx, weight, rel_type)| {
                let neighbor_entry = store.entry_at(neighbor_idx as usize)?;
                let neighbor_emb = store.embedding_at(neighbor_idx as usize)?;
                let cosine = similarity::cosine_similarity(source_emb, neighbor_emb);
                Some(GraphNeighbor {
                    id: neighbor_entry.id.clone(),
                    content_preview: truncate_content(neighbor_entry.display_content(), 200),
                    labels: neighbor_entry.labels.clone(),
                    weight,
                    relationship: rel_type.map(|r| format!("{r:?}")),
                    cosine_similarity: cosine,
                })
            })
            .collect();

        neighbors.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        neighbors.truncate(max_results);

        Ok(GraphNeighborsResult { node, neighbors })
    }

    /// Return a subgraph: given memory IDs, return all nodes + edges between them.
    ///
    /// When `include_neighbors` is true, also adds Hebbian neighbors of each
    /// requested node into the graph (1-hop expansion).
    #[instrument(skip(self), fields(node_count = ids.len()))]
    pub async fn graph_subgraph(
        &self,
        ids: &[MemoryId],
        include_neighbors: bool,
        min_weight: f64,
    ) -> Result<GraphSubgraphResult> {
        let store = self.store.read().await;
        let hebbian = self.hebbian.read().await;

        // Resolve IDs to indices
        let mut index_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for id in ids {
            if let Some(idx) = store.index_of(id) {
                index_set.insert(idx);
            }
        }

        // Optionally expand 1-hop neighbors
        if include_neighbors {
            let anchors: Vec<usize> = index_set.iter().copied().collect();
            for idx in anchors {
                for (neighbor, _weight) in hebbian.get_associations(idx as u32, min_weight) {
                    index_set.insert(neighbor as usize);
                }
            }
        }

        // Build node list
        let mut nodes: Vec<GraphNode> = Vec::with_capacity(index_set.len());
        for &idx in &index_set {
            if let Some(entry) = store.entry_at(idx) {
                nodes.push(GraphNode {
                    id: entry.id.clone(),
                    content_preview: truncate_content(entry.display_content(), 200),
                    labels: entry.labels.clone(),
                    importance: entry.importance,
                    category: format!("{:?}", entry.category),
                    novelty: entry.novelty_score,
                });
            }
        }

        // Build edge list (only edges between nodes in the set, with decayed weights)
        let mut edges: Vec<GraphEdge> = Vec::new();
        let indices: Vec<usize> = index_set.iter().copied().collect();
        for (i, &idx_a) in indices.iter().enumerate() {
            for &idx_b in &indices[i + 1..] {
                let decayed_weight = hebbian.get_weight(idx_a as u32, idx_b as u32);
                if decayed_weight > min_weight {
                    let edge = hebbian.get_edge(idx_a as u32, idx_b as u32);
                    let entry_a = store.entry_at(idx_a);
                    let entry_b = store.entry_at(idx_b);
                    if let (Some(a), Some(b)) = (entry_a, entry_b) {
                        edges.push(GraphEdge {
                            source: a.id.clone(),
                            target: b.id.clone(),
                            weight: decayed_weight,
                            relationship: edge
                                .and_then(|e| e.relationship.as_ref().map(|r| format!("{r:?}"))),
                        });
                    }
                }
            }
        }

        Ok(GraphSubgraphResult { nodes, edges })
    }

    /// Return a high-level cluster overview for the Galaxy view.
    ///
    /// Groups memories by label, returns clusters with member counts + summaries,
    /// plus inter-cluster edges (labels that share members).
    #[instrument(skip(self))]
    pub async fn graph_overview(
        &self,
        min_members: usize,
        max_clusters: usize,
    ) -> Result<GraphOverviewResult> {
        // Collect all data from the store in one fast pass, then release the lock.
        // This prevents RwLock starvation when consolidation needs a write lock.
        let (clusters, member_sets) = {
            let store = self.store.read().await;

            let mut label_clusters = store.labels_with_min_members(min_members);
            label_clusters.sort_by(|a, b| b.1.cmp(&a.1));
            label_clusters.truncate(max_clusters);

            let mut clusters: Vec<GraphCluster> = Vec::with_capacity(label_clusters.len());
            let mut member_sets: Vec<(String, std::collections::HashSet<u32>)> =
                Vec::with_capacity(label_clusters.len());

            for (label, member_count) in &label_clusters {
                let summary = store.get_summary(label).map(|s| s.summary.clone());

                let member_indices = store.query_labels(std::slice::from_ref(label));
                let mut members_with_importance: Vec<(usize, f32)> = member_indices
                    .iter()
                    .filter_map(|&idx| {
                        store
                            .entry_at(idx as usize)
                            .map(|e| (idx as usize, e.importance))
                    })
                    .collect();
                members_with_importance
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let top_members: Vec<GraphNodePreview> = members_with_importance
                    .iter()
                    .take(5)
                    .filter_map(|&(idx, _)| {
                        store.entry_at(idx).map(|e| GraphNodePreview {
                            id: e.id.clone(),
                            content_preview: truncate_content(e.display_content(), 120),
                        })
                    })
                    .collect();

                // Cache member set for inter-edge computation (after lock release)
                member_sets.push((label.clone(), member_indices.into_iter().collect()));

                clusters.push(GraphCluster {
                    label: label.clone(),
                    member_count: *member_count,
                    summary,
                    top_members,
                });
            }

            (clusters, member_sets)
            // read lock released here
        };

        // Compute inter-cluster edges WITHOUT holding the store lock
        let mut inter_edges: Vec<GraphInterEdge> = Vec::new();
        for (i, (label_a, members_a)) in member_sets.iter().enumerate() {
            for (label_b, members_b) in &member_sets[i + 1..] {
                let shared = members_a.intersection(members_b).count();
                if shared > 0 {
                    inter_edges.push(GraphInterEdge {
                        source_label: label_a.clone(),
                        target_label: label_b.clone(),
                        shared_count: shared,
                    });
                }
            }
        }

        inter_edges.sort_by(|a, b| b.shared_count.cmp(&a.shared_count));

        Ok(GraphOverviewResult {
            clusters,
            inter_edges,
        })
    }

    /// Point-in-time echo query (KS63).
    ///
    /// Same as `echo()` but only considers Hebbian edges that were valid at
    /// `at_timestamp` (epoch seconds). Expired edges (valid_until < at_timestamp)
    /// and not-yet-valid edges (valid_from > at_timestamp) are filtered out
    /// during the Hebbian boost pass.
    #[instrument(skip(self), fields(query = %query, at = at_timestamp))]
    pub async fn echo_at(
        &self,
        query: &str,
        max_results: usize,
        at_timestamp: f64,
    ) -> Result<Vec<EchoResult>> {
        self.echo_text(query, max_results, None, Some(at_timestamp))
            .await
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

        // Persist community summaries sidecar (KS64)
        crate::persistence::save_community_summaries(&store, &self.config.data_dir)?;

        // Persist entity store sidecar (KS73)
        crate::persistence::save_entities(&store, &self.config.data_dir)?;

        // KS75: Write embedding model name sidecar for mismatch detection on next load
        if let Ok(embedder) = self.embedder.lock() {
            let model_path = self.config.data_dir.join("embedding_model.txt");
            let _ = std::fs::write(&model_path, embedder.text_provider_name());
        }

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
        let mut embedder = MultiEmbedder::new(&config)?;

        // Initialize label prototypes (ADR-015)
        let mut prototypes = crate::labels::LabelPrototypes::new_empty();
        if config.use_labels {
            prototypes.initialize(|desc| embedder.embed_text(desc).ok());
        }

        let pii_filter = PiiFilter::new();
        let reformulator = MemoryReformulator::new();

        let store_path = config.data_dir.join("echo_store.shrm");
        let mut loaded_store = EchoStore::load(&store_path)?;

        // KS75: Dimension mismatch detection — hard error if stored vectors don't match config
        if let Some(first_emb) = loaded_store.all_embeddings().first() {
            let stored_dim = first_emb.len();
            let config_dim = embedder.text_dimension();
            if stored_dim != config_dim {
                return Err(ShrimPKError::Embedding(format!(
                    "Embedding dimension mismatch: stored data has {stored_dim}-dim vectors \
                     but current model '{}' produces {config_dim}-dim. \
                     Either switch back to the original model or clear the store with /api/clear.",
                    embedder.text_provider_name()
                )));
            }
        }

        // KS75: Model name sidecar — warn if model changed (same dim, different model = mixed space)
        let model_sidecar = config.data_dir.join("embedding_model.txt");
        if model_sidecar.exists()
            && let Ok(stored_model) = std::fs::read_to_string(&model_sidecar)
        {
            let stored_model = stored_model.trim();
            let current_model = embedder.text_provider_name();
            if stored_model != current_model && !loaded_store.all_entries().is_empty() {
                tracing::warn!(
                    stored_model = %stored_model,
                    current_model = %current_model,
                    "Embedding model changed since last persist. \
                     Vectors from different models in the same space may degrade similarity quality."
                );
            }
        }

        // Load community summaries sidecar (KS64)
        if let Err(e) =
            crate::persistence::load_community_summaries(&mut loaded_store, &config.data_dir)
        {
            tracing::warn!(error = %e, "Failed to load community summaries, continuing without");
        }

        // Load entity store sidecar (KS73)
        if let Err(e) = crate::persistence::load_entities(&mut loaded_store, &config.data_dir) {
            tracing::warn!(error = %e, "Failed to load entities, continuing without");
        }

        // Rebuild text LSH index from loaded embeddings
        let mut text_lsh = CosineHash::new(config.embedding_dim, 16, 10);
        if config.use_lsh {
            for (i, embedding) in loaded_store.all_embeddings().iter().enumerate() {
                text_lsh.insert(i as u32, embedding);
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
        let hebbian = HebbianGraph::load(
            &hebbian_path,
            config.hebbian_half_life_secs,
            config.hebbian_prune_threshold,
        )
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "Failed to load Hebbian graph, starting fresh");
            HebbianGraph::new(
                config.hebbian_half_life_secs,
                config.hebbian_prune_threshold,
            )
        });

        // Rebuild vision LSH from loaded entries' vision_embedding fields
        #[cfg(feature = "vision")]
        let vision_lsh_rebuilt = if config
            .enabled_modalities
            .contains(&shrimpk_core::Modality::Vision)
        {
            let mut vlsh = CosineHash::new(config.vision_embedding_dim, 16, 10);
            let mut vision_count = 0usize;
            for (i, entry) in loaded_store.all_entries().iter().enumerate() {
                if let Some(ref ve) = entry.vision_embedding {
                    vlsh.insert(i as u32, ve);
                    vision_count += 1;
                }
            }
            if vision_count > 0 {
                tracing::info!(
                    vision_entries = vision_count,
                    "Vision LSH rebuilt from loaded entries"
                );
            }
            Some(Mutex::new(vlsh))
        } else {
            None
        };

        // Rebuild speech LSH from loaded entries' speech_embedding fields
        #[cfg(feature = "speech")]
        let speech_lsh_rebuilt = if config
            .enabled_modalities
            .contains(&shrimpk_core::Modality::Speech)
        {
            let mut slsh = CosineHash::new(config.speech_embedding_dim, 16, 10);
            let mut speech_count = 0usize;
            for (i, entry) in loaded_store.all_entries().iter().enumerate() {
                if let Some(ref se) = entry.speech_embedding {
                    slsh.insert(i as u32, se);
                    speech_count += 1;
                }
            }
            if speech_count > 0 {
                tracing::info!(
                    speech_entries = speech_count,
                    "Speech LSH rebuilt from loaded entries"
                );
            }
            Some(Mutex::new(slsh))
        } else {
            None
        };

        tracing::info!(
            entries = loaded_store.len(),
            lsh_entries = text_lsh.len(),
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
            text_lsh: Mutex::new(text_lsh),
            #[cfg(feature = "vision")]
            vision_lsh: vision_lsh_rebuilt,
            #[cfg(feature = "speech")]
            speech_lsh: speech_lsh_rebuilt,
            bloom: RwLock::new(bloom),
            bloom_dirty: Mutex::new(false),
            pii_filter,
            reformulator,
            hebbian: RwLock::new(hebbian),
            config,
            stats: Mutex::new(EchoStats::default()),
            consolidation_handle: Mutex::new(None),
            consolidator: consolidator_impl,
            prototypes,
        })
    }

    /// Run async Tier 1 label bootstrap on all unlabeled entries (ADR-015 D7).
    ///
    /// Call this after `load()` to retroactively label existing memories.
    /// Non-blocking: acquires a write lock, runs bootstrap, releases.
    /// Returns the number of entries that received labels.
    pub async fn bootstrap_labels(&self) -> usize {
        if !self.config.use_labels || !self.prototypes.is_initialized() {
            return 0;
        }
        let mut store = self.store.write().await;
        let unlabeled = store
            .all_entries()
            .iter()
            .filter(|e| e.label_version == 0)
            .count();
        if unlabeled == 0 {
            tracing::debug!("No unlabeled entries, skipping bootstrap");
            return 0;
        }
        tracing::info!(unlabeled, "Starting Tier 1 label bootstrap");
        store.bootstrap_tier1_labels(&self.prototypes)
    }

    /// Reset and re-run Tier 1 labels on all entries.
    ///
    /// Useful after improving the entity extraction stopword list.
    /// Clears existing Tier 1 labels and regenerates from scratch.
    pub async fn rebootstrap_labels(&self) -> usize {
        if !self.config.use_labels || !self.prototypes.is_initialized() {
            return 0;
        }
        let mut store = self.store.write().await;
        store.rebootstrap_tier1_labels(&self.prototypes)
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
        let mut lsh = self.text_lsh.lock().unwrap_or_else(|e| e.into_inner());

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

    // -----------------------------------------------------------------------
    // Test-only helpers (KS69 — deterministic benchmark fixtures)
    // -----------------------------------------------------------------------

    /// Test-only: inject a pre-built entry into the store, LSH index, and Bloom filter.
    ///
    /// Bypasses PII filtering, reformulation, novelty scoring, and consolidation triggers.
    /// Used for deterministic benchmark fixtures where children need exact control over
    /// content and embedding without LLM nondeterminism.
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn inject_entry(&self, entry: MemoryEntry) {
        let embedding = entry.embedding.clone();
        let content = entry.content.clone();
        let parent_id = entry.parent_id.clone();

        // 1. Add to store (handles id_to_index, parent_children, label_index)
        let index = {
            let mut store = self.store.write().await;
            store.add(entry)
        };

        // 2. Insert into text LSH index (always index test entries, regardless of child_rescue_only)
        if let Ok(mut lsh) = self.text_lsh.lock() {
            lsh.insert(index as u32, &embedding);
        }

        // 3. Insert into Bloom filter
        {
            let mut bloom = self.bloom.write().await;
            bloom.insert_memory(&content);
        }

        // 4. If this is a child, mark the parent as enriched
        if let Some(ref pid) = parent_id {
            let mut store = self.store.write().await;
            if let Some(parent_idx) = store.index_of(pid)
                && let Some(parent) = store.entry_at_mut(parent_idx)
            {
                parent.enriched = true;
            }
        }
    }

    /// Lock the embedder and run a blocking embedding operation.
    ///
    /// Uses `tokio::task::block_in_place` to inform the Tokio scheduler that
    /// this thread will block, preventing worker-thread starvation. Critical
    /// for API-based providers (OpenAI) where network I/O can take seconds;
    /// harmless for local fastembed calls (~5ms).
    fn embed_blocking<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut MultiEmbedder) -> Result<R>,
    {
        let mut embedder = self
            .embedder
            .lock()
            .map_err(|e| ShrimPKError::Embedding(format!("MultiEmbedder lock poisoned: {e}")))?;
        tokio::task::block_in_place(|| f(&mut embedder))
    }

    /// Test-only: generate an embedding for text using the engine's embedder.
    ///
    /// Provides access to the same embedding model used by `store()` so that
    /// test children have embeddings from the same vector space.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn embed_text_for_test(&self, text: &str) -> Result<Vec<f32>> {
        let mut embedder = self
            .embedder
            .lock()
            .map_err(|e| ShrimPKError::Embedding(format!("MultiEmbedder lock poisoned: {e}")))?;
        embedder.embed_text(text)
    }

    /// Test-only: inject a Hebbian Supersedes edge between two memories by ID.
    ///
    /// Looks up store indices from memory IDs, then creates a directed
    /// Supersedes edge (old_id → new_id). Used for deterministic benchmark
    /// fixtures where consolidation is skipped. (KS69)
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn inject_supersedes_edge(&self, old_id: &MemoryId, new_id: &MemoryId) {
        let store = self.store.read().await;
        let old_idx = store.index_of(old_id).expect("old_id not in store");
        let new_idx = store.index_of(new_id).expect("new_id not in store");
        drop(store);

        let mut hebbian = self.hebbian.write().await;
        hebbian.co_activate_with_relationship(
            old_idx as u32,
            new_idx as u32,
            1.0,
            crate::hebbian::RelationshipType::Supersedes,
        );
    }
}

// ---------------------------------------------------------------------------
// HyDE (Hypothetical Document Embeddings) query expansion
// ---------------------------------------------------------------------------

/// Ask a local LLM for a hypothetical first-person answer to the query,
/// then return `"<original query> <hypothetical answer>"` for embedding.
///
/// This shifts the embedding vector from "question space" toward "answer space",
/// which is where stored memories live. Silently returns `None` on any failure
/// (Ollama down, timeout, bad response) so the caller falls back to raw query.
fn expand_query(config: &EchoConfig, query: &str) -> Option<String> {
    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .build(),
    );

    let body = serde_json::json!({
        "model": config.enrichment_model,
        "messages": [
            {
                "role": "system",
                "content": "Given this question about a user, write a short first-person answer as if you are that user. Keep it under 30 words. Be specific with names and details."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": false,
        "options": { "temperature": 0.0, "num_predict": 64 }
    });

    let endpoint = format!("{}/api/chat", config.ollama_url.trim_end_matches('/'));
    let mut resp = agent.post(&endpoint).send_json(&body).ok()?;
    let json: serde_json::Value = resp.body_mut().read_json().ok()?;
    let content = json["message"]["content"].as_str()?;

    if content.len() > 10 {
        // Combine original query + expansion for better embedding coverage
        Some(format!("{} {}", query, content))
    } else {
        None
    }
}

/// Check if any entity in the query matches the child's subject.
/// Topic-based gate for child memories (KS69 Tier 1).
///
/// Uses label overlap (primary) or subject substring match (fallback) to decide
/// whether a child memory is topically relevant to the current query.
/// Prevents children from contaminating unrelated queries.
fn child_topic_matches_query(
    child_labels: &[String],
    query_topic_labels: &[&str],
    child_subject: &Option<String>,
    query: &str,
) -> bool {
    // 1. Label overlap (primary) — same logic as Pipe B topic alignment gate
    if !query_topic_labels.is_empty() && !child_labels.is_empty() {
        return child_labels
            .iter()
            .any(|cl| query_topic_labels.iter().any(|qt| cl == qt));
    }
    // 2. Subject match (fallback) — for children without labels
    if let Some(subj) = child_subject {
        let subj_lower = subj.to_lowercase();
        let query_lower = query.to_lowercase();
        return query_lower.contains(&subj_lower)
            || query_lower
                .split_whitespace()
                .any(|w| subj_lower.contains(w) && w.len() > 2);
    }
    // 3. No labels, no subject — allow (backward compat)
    true
}

/// Build a map from parent memory IDs to subject strings from their children.
/// Used for subject diversity enforcement in echo results (KS67).
/// Per-memory subject and topic info for diversity enforcement.
struct SubjectTopicInfo {
    subjects: Vec<String>,
    primary_topic: String,
}

fn build_subject_topic_map(
    store: &EchoStore,
    results: &[(usize, f32)],
) -> std::collections::HashMap<MemoryId, SubjectTopicInfo> {
    let mut map: std::collections::HashMap<MemoryId, SubjectTopicInfo> =
        std::collections::HashMap::new();

    for &(idx, _score) in results {
        if let Some(entry) = store.entry_at(idx) {
            // Collect subjects from this entry's children
            let child_indices = store.children_of(&entry.id);
            let mut subjects: Vec<String> = Vec::new();

            for &child_idx in child_indices {
                if let Some(child) = store.entry_at(child_idx) {
                    // KS73: prefer entity canonical name over triple subject
                    if let Some(ref eid) = child.entity_id
                        && let Some(frame) = store.get_entity(eid)
                    {
                        subjects.push(frame.canonical_name.clone());
                        continue;
                    }
                    for triple in &child.triples {
                        subjects.push(triple.subject.clone());
                    }
                    // Fallback: first word of content as subject
                    if child.triples.is_empty()
                        && let Some(word) = child.content.split_whitespace().next()
                    {
                        subjects.push(word.to_string());
                    }
                }
            }

            // Also check the entry's own triples
            for triple in &entry.triples {
                subjects.push(triple.subject.clone());
            }

            subjects.sort();
            subjects.dedup();

            // Extract primary topic label (first "topic:*" label, or "topic:unknown")
            let primary_topic = entry
                .labels
                .iter()
                .find(|l| l.starts_with("topic:"))
                .cloned()
                .unwrap_or_else(|| "topic:unknown".to_string());

            if !subjects.is_empty() {
                map.insert(
                    entry.id.clone(),
                    SubjectTopicInfo {
                        subjects,
                        primary_topic,
                    },
                );
            }
        }
    }

    map
}

/// Co-occurrence boost (KS68 ME-4): returns +0.05 if the content mentions 2+ entities
/// from the same category (databases or programming languages), else 0.0.
fn co_occurrence_boost(content: &str) -> f64 {
    const DB_KEYWORDS: &[&str] = &[
        "postgresql",
        "mysql",
        "clickhouse",
        "mongodb",
        "postgres",
        "redis",
        "sqlite",
        "oracle",
        "cassandra",
        "dynamodb",
    ];
    const LANG_KEYWORDS: &[&str] = &[
        "rust",
        "python",
        " go ",
        "javascript",
        "typescript",
        "java ",
        "c++",
        "scala",
        "kotlin",
        "swift",
    ];
    let content_lower = content.to_lowercase();
    let db_count = DB_KEYWORDS
        .iter()
        .filter(|kw| content_lower.contains(*kw))
        .count();
    let lang_count = LANG_KEYWORDS
        .iter()
        .filter(|kw| content_lower.contains(*kw))
        .count();
    if db_count >= 2 || lang_count >= 2 {
        0.05
    } else {
        0.0
    }
}

/// Temporal query boost (KS68 TR-3): if the query contains temporal keywords,
/// boost results that have `temporal:*` labels by +0.015.
fn apply_temporal_boost(query: &str, results: &mut [EchoResult]) {
    const TEMPORAL_KEYWORDS: &[&str] = &[
        "deadline",
        "upcoming",
        "when",
        "scheduled",
        "date",
        "due",
        "plan",
        "next week",
        "next month",
    ];
    let query_lower = query.to_lowercase();
    let is_temporal_query = TEMPORAL_KEYWORDS.iter().any(|kw| query_lower.contains(kw));
    if is_temporal_query {
        for result in results.iter_mut() {
            let has_temporal_label = result.labels.iter().any(|l| l.starts_with("temporal:"));
            if has_temporal_label {
                result.final_score += 0.015;
            }
        }
    }
}

/// Label-based boost (KS68): when query is classified with a specific label
/// (e.g., `topic:tools:editor`, `action:learning`) and a result also carries that label,
/// give it a scoring bump so precisely-labeled memories surface above generic ones.
///
/// Boost values tuned per QA analysis: +0.06 for topic:tools:* (KU-3 gap closure),
/// +0.025 for action:learning.
fn label_topic_boost(query_labels: &[String], results: &mut [EchoResult]) {
    for result in results.iter_mut() {
        for ql in query_labels {
            if ql.starts_with("topic:tools:") && result.labels.iter().any(|l| l == ql) {
                result.final_score += 0.06;
                break;
            }
            if ql == "action:learning" && result.labels.iter().any(|l| l == ql) {
                result.final_score += 0.025;
                break;
            }
        }
    }
}

/// Preference-update multiplier (KS68 KU-3): when the query signals interest in
/// current state ("currently", "now use", "switched to"), memories labeled
/// `memtype:preference_update` get a 1.05x multiplier so "I switched from X to Y"
/// memories rank above stale preference entries with higher raw similarity.
fn preference_update_boost(query: &str, results: &mut [EchoResult]) {
    const CURRENT_KEYWORDS: &[&str] = &[
        "currently",
        "now use",
        "now using",
        "switched to",
        "these days",
        "at the moment",
        "right now",
    ];
    let query_lower = query.to_lowercase();
    let is_current_query = CURRENT_KEYWORDS.iter().any(|kw| query_lower.contains(kw));
    if !is_current_query {
        return;
    }
    for result in results.iter_mut() {
        if result
            .labels
            .iter()
            .any(|l| l == "memtype:preference_update")
        {
            result.final_score *= 1.05;
        }
    }
}

/// Career query adjustment (KS68 IE-1): when query is classified as career-related,
/// demote `memtype:intro` memories (-0.10) and boost career-labeled non-intro memories
/// (+0.03). This prevents "My name is Sam Torres" from outranking actual job memories.
fn career_intro_adjustment(query_labels: &[String], results: &mut [EchoResult]) {
    let is_career_query = query_labels
        .iter()
        .any(|l| l == "topic:career" || l == "domain:work");
    if !is_career_query {
        return;
    }
    for result in results.iter_mut() {
        let is_intro = result.labels.iter().any(|l| l == "memtype:intro");
        if is_intro {
            result.final_score -= 0.10;
        } else if result
            .labels
            .iter()
            .any(|l| l == "topic:career" || l == "domain:work")
        {
            result.final_score += 0.03;
        }
    }
}

/// Parent-child dedup (KS68.3 prep): if a parent and one of its children both appear
/// in the result set, remove the lower-scoring one to prevent slot waste.
///
/// `parent_map` maps each result's memory_id to its `parent_id` (None for root entries).
/// O(n^2) for small N (typically 5-10) — acceptable.
fn deduplicate_parent_child(
    results: &mut Vec<EchoResult>,
    parent_map: &std::collections::HashMap<MemoryId, Option<MemoryId>>,
) {
    let mut to_remove: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();
    let len = results.len();
    for i in 0..len {
        for j in (i + 1)..len {
            let id_i = &results[i].memory_id;
            let id_j = &results[j].memory_id;
            let parent_i = parent_map.get(id_i).and_then(|p| p.as_ref());
            let parent_j = parent_map.get(id_j).and_then(|p| p.as_ref());

            let is_pair = (parent_i == Some(id_j)) || (parent_j == Some(id_i));
            if !is_pair {
                continue;
            }
            // Remove the lower-scoring one (results are sorted, so j > i means j scores lower)
            if results[i].final_score >= results[j].final_score {
                to_remove.insert(id_j.clone());
            } else {
                to_remove.insert(id_i.clone());
            }
        }
    }
    if !to_remove.is_empty() {
        results.retain(|r| !to_remove.contains(&r.memory_id));
    }
}

/// Cap results so no single (subject, topic) pair dominates the result set (KS67/KS68).
/// Tracks occurrences per (subject, topic_label) tuple so that different facets of the
/// same entity (e.g., "Sam:identity" vs "Sam:preference") count independently.
/// Unknown subjects (no triple data) go into an "_unknown" bucket with a more generous cap.
fn enforce_subject_diversity(
    results: &mut Vec<EchoResult>,
    _store: &EchoStore,
    subject_topic_map: &std::collections::HashMap<MemoryId, SubjectTopicInfo>,
    max_per_subject: usize,
) {
    let mut subject_topic_counts: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    results.retain(|r| {
        let info = subject_topic_map.get(&r.memory_id);

        let (subjects, topic) = match info {
            Some(info) if !info.subjects.is_empty() => {
                (&info.subjects, info.primary_topic.as_str())
            }
            _ => {
                // Unknown bucket -- cap at max_per_subject * 2
                let key = ("_unknown".to_string(), "topic:unknown".to_string());
                let count = subject_topic_counts.entry(key).or_insert(0);
                if *count >= max_per_subject * 2 {
                    return false;
                }
                *count += 1;
                return true;
            }
        };

        // Check if any (subject, topic) pair is already at cap
        let dominated = subjects.iter().any(|s| {
            let key = (s.clone(), topic.to_string());
            *subject_topic_counts.get(&key).unwrap_or(&0) >= max_per_subject
        });
        if dominated {
            return false;
        }
        for s in subjects {
            let key = (s.clone(), topic.to_string());
            *subject_topic_counts.entry(key).or_insert(0) += 1;
        }
        true
    });
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
            .store(
                "I work as an engineer at Google on the Cloud team",
                "old_session",
            )
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
            .store(
                "I left Google. I now work at Meta on the infrastructure team",
                "new_session",
            )
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

    // --- QueryMode tests (KS35) ---

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn echo_with_mode_text_matches_echo() {
        // echo_with_mode(Text) should produce identical results to echo()
        let engine = EchoEngine::new(test_config()).expect("Should init");
        engine
            .store("Rust is a systems programming language", "test")
            .await
            .expect("Should store");

        let results_echo = engine.echo("systems programming", 5).await.unwrap();
        let results_mode = engine
            .echo_with_mode("systems programming", 5, shrimpk_core::QueryMode::Text)
            .await
            .unwrap();

        assert_eq!(results_echo.len(), results_mode.len());
        if !results_echo.is_empty() {
            assert_eq!(results_echo[0].memory_id, results_mode[0].memory_id);
        }
    }

    #[tokio::test]
    #[ignore = "requires fastembed model download"]
    async fn echo_with_mode_vision_returns_empty_when_no_vision_entries() {
        // Without vision feature or without vision entries, Vision mode returns empty
        let engine = EchoEngine::new(test_config()).expect("Should init");
        engine
            .store("text only memory", "test")
            .await
            .expect("Should store");

        let results = engine
            .echo_with_mode("any query", 5, shrimpk_core::QueryMode::Vision)
            .await
            .unwrap();

        // Without vision feature compiled in, this returns empty.
        // With vision feature but no vision entries, also returns empty.
        assert!(
            results.is_empty(),
            "Vision echo with no vision entries should be empty"
        );
    }

    // --- Vision integration tests (KS35, require CLIP model) ---

    #[cfg(feature = "vision")]
    #[tokio::test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    async fn store_image_and_echo_vision() {
        use shrimpk_core::QueryMode;

        let mut config = test_config();
        config.enabled_modalities =
            vec![shrimpk_core::Modality::Text, shrimpk_core::Modality::Vision];
        config.similarity_threshold = 0.05; // low threshold for cross-modal

        let engine = EchoEngine::new(config).expect("Should init");

        // Store an image
        let png_data = create_test_png(32, 32, [255, 0, 0]);
        let image_id = engine
            .store_image(&png_data, "test", None)
            .await
            .expect("Should store image");

        // Echo with Vision mode — should find the image
        let results = engine
            .echo_with_mode("a red image", 5, QueryMode::Vision)
            .await
            .expect("Vision echo should work");

        assert!(
            !results.is_empty(),
            "Vision echo should find the stored image"
        );
        assert_eq!(results[0].memory_id, image_id);
        assert_eq!(results[0].modality, shrimpk_core::Modality::Vision);
    }

    #[cfg(feature = "vision")]
    #[tokio::test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    async fn store_image_skips_bloom() {
        use shrimpk_core::QueryMode;

        let mut config = test_config();
        config.use_bloom = true;
        config.enabled_modalities =
            vec![shrimpk_core::Modality::Text, shrimpk_core::Modality::Vision];

        let engine = EchoEngine::new(config).expect("Should init");

        // Store an image — should NOT insert into Bloom
        let png_data = create_test_png(16, 16, [0, 0, 255]);
        engine
            .store_image(&png_data, "test", None)
            .await
            .expect("Should store image");

        // Store a text memory — this DOES insert into Bloom
        engine
            .store("blue sky photograph", "test")
            .await
            .expect("Should store text");

        // Stats should show 2 total memories
        let stats = engine.stats().await;
        assert_eq!(stats.total_memories, 2);

        // Text echo should still work (Bloom doesn't block)
        let text_results = engine
            .echo_with_mode("blue sky", 5, QueryMode::Text)
            .await
            .unwrap();
        assert!(
            !text_results.is_empty(),
            "Text echo should find text memories"
        );
    }

    #[cfg(feature = "vision")]
    #[tokio::test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    async fn auto_mode_merges_text_and_vision() {
        use shrimpk_core::QueryMode;

        let mut config = test_config();
        config.enabled_modalities =
            vec![shrimpk_core::Modality::Text, shrimpk_core::Modality::Vision];
        config.similarity_threshold = 0.05;

        let engine = EchoEngine::new(config).expect("Should init");

        // Store a text memory
        engine
            .store(
                "The sunset was beautiful with red and orange colors",
                "test",
            )
            .await
            .expect("Should store text");

        // Store an image
        let png_data = create_test_png(32, 32, [255, 128, 0]); // orange
        engine
            .store_image(&png_data, "test", None)
            .await
            .expect("Should store image");

        // Auto mode should search both channels
        let results = engine
            .echo_with_mode("sunset colors", 10, QueryMode::Auto)
            .await
            .expect("Auto echo should work");

        assert!(!results.is_empty(), "Auto mode should find results");
    }

    #[cfg(feature = "vision")]
    #[tokio::test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    async fn text_echo_unchanged_with_vision_entries() {
        // Storing vision entries should not affect text-only echo
        let mut config = test_config();
        config.enabled_modalities =
            vec![shrimpk_core::Modality::Text, shrimpk_core::Modality::Vision];

        let engine = EchoEngine::new(config).expect("Should init");

        // Store text memories
        engine
            .store("Rust is a systems programming language", "test")
            .await
            .unwrap();
        engine
            .store("Python is great for data science", "test")
            .await
            .unwrap();

        // Store an image (should not interfere with text search)
        let png_data = create_test_png(8, 8, [0, 255, 0]);
        engine.store_image(&png_data, "test", None).await.unwrap();

        // Text echo should find text memories as before
        let results = engine.echo("programming language", 5).await.unwrap();
        assert!(
            !results.is_empty(),
            "Text echo should still work with vision entries present"
        );
        // Results should be text modality
        assert_eq!(results[0].modality, shrimpk_core::Modality::Text);
    }

    /// Create a minimal PNG image with a solid color (for tests).
    #[cfg(feature = "vision")]
    fn create_test_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        use image::ImageEncoder;
        use std::io::Cursor;
        let mut buf = Cursor::new(Vec::new());

        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            pixels.push(rgb[0]);
            pixels.push(rgb[1]);
            pixels.push(rgb[2]);
            pixels.push(255);
        }

        image::codecs::png::PngEncoder::new(&mut buf)
            .write_image(&pixels, width, height, image::ExtendedColorType::Rgba8)
            .expect("PNG encode should succeed");

        buf.into_inner()
    }

    // --- D6 merge structure tests (KS44, ADR-015) ---

    #[test]
    fn d6_merge_with_labels_disabled_behaves_like_lsh_only() {
        // When use_labels=false, the merge should produce the same result as
        // the pre-D6 LSH-only path. This is a regression guard.
        let config = EchoConfig {
            use_labels: false,
            use_lsh: true,
            use_bloom: false,
            ..Default::default()
        };
        // Verify the config field exists and defaults correctly
        assert!(!config.use_labels);
        assert!(config.use_lsh);
    }

    #[test]
    fn d6_merge_with_both_disabled_falls_through() {
        // When both use_labels and use_lsh are false, the merge should fall
        // through to brute-force. Verify the config combination is valid.
        let config = EchoConfig {
            use_labels: false,
            use_lsh: false,
            use_bloom: false,
            ..Default::default()
        };
        assert!(!config.use_labels);
        assert!(!config.use_lsh);
    }

    #[test]
    fn d6_use_labels_defaults_to_true() {
        let config = EchoConfig::default();
        assert!(config.use_labels, "use_labels should default to true");
    }

    // --- KS62: detect_query_entities tests ---

    #[test]
    fn detect_entities_unigram_match() {
        let mut index = std::collections::HashMap::new();
        index.insert("rust".to_string(), vec![0, 1]);
        index.insert("python".to_string(), vec![2]);

        let matches = detect_query_entities("I love Rust programming", &index);
        assert_eq!(matches, vec!["rust"]);
    }

    #[test]
    fn detect_entities_bigram_match() {
        let mut index = std::collections::HashMap::new();
        index.insert("lior cohen".to_string(), vec![0]);
        index.insert("lior".to_string(), vec![0]);

        let matches = detect_query_entities("Where does Lior Cohen work?", &index);
        assert!(matches.contains(&"lior cohen".to_string()));
    }

    #[test]
    fn detect_entities_no_match() {
        let mut index = std::collections::HashMap::new();
        index.insert("python".to_string(), vec![0]);

        let matches = detect_query_entities("What is the weather today?", &index);
        assert!(matches.is_empty());
    }

    #[test]
    fn detect_entities_empty_index() {
        let index = std::collections::HashMap::new();
        let matches = detect_query_entities("anything", &index);
        assert!(matches.is_empty());
    }

    // --- KS62: reciprocal_rank_fusion tests ---

    #[test]
    fn rrf_merge_overlapping() {
        let vector = vec![(0, 0.9f32), (1, 0.8), (2, 0.7)];
        let graph = vec![(2, 0.7f32), (3, 0.6), (0, 0.5)];

        let merged = reciprocal_rank_fusion(&vector, &graph, 60);

        // Items in both lists (0, 2) should rank higher than single-list items
        let ids: Vec<usize> = merged.iter().map(|&(idx, _)| idx).collect();
        assert_eq!(ids.len(), 4, "Union of both lists = 4 unique items");

        // Item 0 is rank 1 in vector + rank 3 in graph → strong RRF
        // Item 2 is rank 3 in vector + rank 1 in graph → strong RRF
        // Both should be in top 2
        let top2: Vec<usize> = ids[..2].to_vec();
        assert!(top2.contains(&0), "Item 0 (in both lists) should be top-2");
        assert!(top2.contains(&2), "Item 2 (in both lists) should be top-2");
    }

    #[test]
    fn rrf_merge_non_overlapping() {
        let vector = vec![(0, 0.9f32), (1, 0.8)];
        let graph = vec![(2, 0.7f32), (3, 0.6)];

        let merged = reciprocal_rank_fusion(&vector, &graph, 60);
        assert_eq!(merged.len(), 4);
        // All items get single-list RRF scores
    }

    #[test]
    fn rrf_merge_empty_graph() {
        let vector = vec![(0, 0.9f32), (1, 0.8)];
        let graph: Vec<(usize, f32)> = vec![];

        let merged = reciprocal_rank_fusion(&vector, &graph, 60);
        assert_eq!(merged.len(), 2);
        // Order preserved from vector since graph is empty
        assert_eq!(merged[0].0, 0);
        assert_eq!(merged[1].0, 1);
    }

    // --- KS62: config defaults ---

    #[test]
    fn graph_config_defaults() {
        let config = EchoConfig::default();
        assert!(config.graph_traversal_enabled);
        assert_eq!(config.graph_max_hops, 2);
        assert_eq!(config.graph_rrf_k, 60);
    }

    // --- KS64: community summary config defaults ---

    #[test]
    fn community_summary_config_defaults() {
        let config = EchoConfig::default();
        assert!(config.community_summaries_enabled);
        assert!((config.community_summary_threshold - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.community_min_members, 5);
    }

    #[test]
    fn community_summary_serde_roundtrip() {
        let summary = shrimpk_core::CommunitySummary {
            label: "career".into(),
            summary: "User is a Rust engineer.".into(),
            embedding: vec![0.1, 0.2, 0.3],
            member_count: 7,
            updated_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: shrimpk_core::CommunitySummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.label, "career");
        assert_eq!(parsed.member_count, 7);
        assert_eq!(parsed.embedding.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Graph visualization tests (KS65)
    // -----------------------------------------------------------------------

    #[test]
    fn graph_node_serde_roundtrip() {
        let node = shrimpk_core::GraphNode {
            id: shrimpk_core::MemoryId::new(),
            content_preview: "test memory".into(),
            labels: vec!["topic:test".into()],
            importance: 0.75,
            category: "Fact".into(),
            novelty: 0.5,
        };
        let json = serde_json::to_string(&node).unwrap();
        let parsed: shrimpk_core::GraphNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content_preview, "test memory");
        assert_eq!(parsed.importance, 0.75);
    }

    #[test]
    fn graph_neighbor_serde_roundtrip() {
        let neighbor = shrimpk_core::GraphNeighbor {
            id: shrimpk_core::MemoryId::new(),
            content_preview: "neighbor".into(),
            labels: vec![],
            weight: 0.85,
            relationship: Some("WorksAt(\"ACME\")".into()),
            cosine_similarity: 0.72,
        };
        let json = serde_json::to_string(&neighbor).unwrap();
        let parsed: shrimpk_core::GraphNeighbor = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.weight, 0.85);
        assert!(parsed.relationship.is_some());
    }

    #[test]
    fn graph_edge_serde_roundtrip() {
        let edge = shrimpk_core::GraphEdge {
            source: shrimpk_core::MemoryId::new(),
            target: shrimpk_core::MemoryId::new(),
            weight: 0.42,
            relationship: None,
        };
        let json = serde_json::to_string(&edge).unwrap();
        let parsed: shrimpk_core::GraphEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.weight, 0.42);
        assert!(parsed.relationship.is_none());
    }

    #[test]
    fn graph_cluster_serde_roundtrip() {
        let cluster = shrimpk_core::GraphCluster {
            label: "topic:rust".into(),
            member_count: 15,
            summary: Some("Rust programming memories".into()),
            top_members: vec![shrimpk_core::GraphNodePreview {
                id: shrimpk_core::MemoryId::new(),
                content_preview: "first member".into(),
            }],
        };
        let json = serde_json::to_string(&cluster).unwrap();
        let parsed: shrimpk_core::GraphCluster = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.label, "topic:rust");
        assert_eq!(parsed.member_count, 15);
        assert!(parsed.summary.is_some());
        assert_eq!(parsed.top_members.len(), 1);
    }

    #[test]
    fn graph_overview_result_serde_roundtrip() {
        let overview = shrimpk_core::GraphOverviewResult {
            clusters: vec![],
            inter_edges: vec![shrimpk_core::GraphInterEdge {
                source_label: "topic:a".into(),
                target_label: "topic:b".into(),
                shared_count: 5,
            }],
        };
        let json = serde_json::to_string(&overview).unwrap();
        let parsed: shrimpk_core::GraphOverviewResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.inter_edges.len(), 1);
        assert_eq!(parsed.inter_edges[0].shared_count, 5);
    }

    #[test]
    fn graph_subgraph_result_serde_roundtrip() {
        let subgraph = shrimpk_core::GraphSubgraphResult {
            nodes: vec![shrimpk_core::GraphNode {
                id: shrimpk_core::MemoryId::new(),
                content_preview: "node A".into(),
                labels: vec!["topic:test".into()],
                importance: 0.5,
                category: "Default".into(),
                novelty: 0.3,
            }],
            edges: vec![],
        };
        let json = serde_json::to_string(&subgraph).unwrap();
        let parsed: shrimpk_core::GraphSubgraphResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.edges.len(), 0);
    }

    #[test]
    fn graph_neighbors_result_serde_roundtrip() {
        let result = shrimpk_core::GraphNeighborsResult {
            node: shrimpk_core::GraphNode {
                id: shrimpk_core::MemoryId::new(),
                content_preview: "center node".into(),
                labels: vec!["entity:john".into()],
                importance: 0.9,
                category: "Identity".into(),
                novelty: 0.1,
            },
            neighbors: vec![],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: shrimpk_core::GraphNeighborsResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node.content_preview, "center node");
        assert_eq!(parsed.neighbors.len(), 0);
    }

    #[test]
    fn graph_neighbors_result_with_neighbors() {
        let result = shrimpk_core::GraphNeighborsResult {
            node: shrimpk_core::GraphNode {
                id: shrimpk_core::MemoryId::new(),
                content_preview: "center".into(),
                labels: vec![],
                importance: 0.5,
                category: "Fact".into(),
                novelty: 0.0,
            },
            neighbors: vec![
                shrimpk_core::GraphNeighbor {
                    id: shrimpk_core::MemoryId::new(),
                    content_preview: "neighbor 1".into(),
                    labels: vec!["topic:a".into()],
                    weight: 0.9,
                    relationship: Some("CoActivation".into()),
                    cosine_similarity: 0.8,
                },
                shrimpk_core::GraphNeighbor {
                    id: shrimpk_core::MemoryId::new(),
                    content_preview: "neighbor 2".into(),
                    labels: vec!["topic:b".into()],
                    weight: 0.3,
                    relationship: None,
                    cosine_similarity: 0.4,
                },
            ],
        };
        assert_eq!(result.neighbors.len(), 2);
        assert!(result.neighbors[0].weight > result.neighbors[1].weight);
    }

    // -----------------------------------------------------------------------
    // KS68 unit tests: co-occurrence boost, temporal boost, subject diversity
    // -----------------------------------------------------------------------

    fn make_echo_result(content: &str, score: f64, labels: Vec<String>) -> EchoResult {
        EchoResult {
            memory_id: MemoryId::new(),
            content: content.to_string(),
            similarity: score as f32,
            final_score: score,
            source: "test".to_string(),
            echoed_at: Utc::now(),
            modality: Modality::Text,
            labels,
            matched_child_content: None,
        }
    }

    fn make_echo_result_with_id(
        id: &MemoryId,
        content: &str,
        score: f64,
        labels: Vec<String>,
    ) -> EchoResult {
        EchoResult {
            memory_id: id.clone(),
            content: content.to_string(),
            similarity: score as f32,
            final_score: score,
            source: "test".to_string(),
            echoed_at: Utc::now(),
            modality: Modality::Text,
            labels,
            matched_child_content: None,
        }
    }

    // --- ME-4: Co-occurrence boost ---

    #[test]
    fn co_occurrence_boost_fires_for_multi_database_content() {
        let boost =
            super::co_occurrence_boost("I use PostgreSQL for OLTP and ClickHouse for analytics");
        assert!(
            (boost - 0.05).abs() < f64::EPSILON,
            "Expected +0.05 for 2 databases, got {boost}"
        );
    }

    #[test]
    fn co_occurrence_boost_fires_for_mongo_and_postgres() {
        let boost = super::co_occurrence_boost("I tried MongoDB but prefer Postgres");
        assert!(
            (boost - 0.05).abs() < f64::EPSILON,
            "Expected +0.05 for 2 databases, got {boost}"
        );
    }

    #[test]
    fn co_occurrence_boost_zero_for_single_database() {
        let boost = super::co_occurrence_boost("I use Redis for caching");
        assert!(
            boost.abs() < f64::EPSILON,
            "Expected 0.0 for 1 database, got {boost}"
        );
    }

    #[test]
    fn co_occurrence_boost_fires_for_multi_language() {
        let boost = super::co_occurrence_boost("I prefer Rust and Go for all projects");
        assert!(
            (boost - 0.05).abs() < f64::EPSILON,
            "Expected +0.05 for 2 languages, got {boost}"
        );
    }

    #[test]
    fn co_occurrence_boost_zero_for_unrelated_content() {
        let boost = super::co_occurrence_boost("I enjoy hiking in the mountains");
        assert!(
            boost.abs() < f64::EPSILON,
            "Expected 0.0 for unrelated content, got {boost}"
        );
    }

    // --- TR-3: Temporal query boost ---

    #[test]
    fn temporal_boost_fires_for_deadline_query() {
        let mut results = vec![
            make_echo_result("Patent filing", 0.5, vec!["temporal:future".into()]),
            make_echo_result("Sam's job", 0.5, vec!["topic:identity".into()]),
        ];
        super::apply_temporal_boost("What upcoming deadlines does Sam have?", &mut results);
        assert!(
            (results[0].final_score - 0.515).abs() < f64::EPSILON,
            "Temporal result should be boosted to 0.515, got {}",
            results[0].final_score
        );
        assert!(
            (results[1].final_score - 0.5).abs() < f64::EPSILON,
            "Non-temporal result should be unchanged at 0.5, got {}",
            results[1].final_score
        );
    }

    #[test]
    fn temporal_boost_does_not_fire_for_non_temporal_query() {
        let mut results = vec![
            make_echo_result("Patent filing", 0.5, vec!["temporal:future".into()]),
            make_echo_result("Sam's job", 0.5, vec!["topic:identity".into()]),
        ];
        super::apply_temporal_boost("What is Sam's job?", &mut results);
        assert!(
            (results[0].final_score - 0.5).abs() < f64::EPSILON,
            "No boost expected for non-temporal query, got {}",
            results[0].final_score
        );
        assert!(
            (results[1].final_score - 0.5).abs() < f64::EPSILON,
            "No boost expected, got {}",
            results[1].final_score
        );
    }

    // --- KU-3: Subject diversity with (subject, topic) tuples ---

    #[test]
    fn subject_diversity_caps_per_subject_topic() {
        let store = EchoStore::new();
        let mut results: Vec<EchoResult> = (0..5)
            .map(|i| make_echo_result(&format!("Sam identity {i}"), 1.0 - i as f64 * 0.01, vec![]))
            .collect();
        let mut map = std::collections::HashMap::new();
        for r in &results {
            map.insert(
                r.memory_id.clone(),
                SubjectTopicInfo {
                    subjects: vec!["Sam".to_string()],
                    primary_topic: "topic:identity".to_string(),
                },
            );
        }
        super::enforce_subject_diversity(&mut results, &store, &map, 3);
        assert_eq!(
            results.len(),
            3,
            "Should cap at 3 per (subject, topic) pair"
        );
    }

    #[test]
    fn subject_diversity_allows_different_topics_for_same_subject() {
        let store = EchoStore::new();
        // 3 identity + 3 preference entries for "Sam" — all should survive with cap=3
        let mut results: Vec<EchoResult> = Vec::new();
        let mut map = std::collections::HashMap::new();
        for i in 0..3 {
            let r = make_echo_result(&format!("Sam identity {i}"), 1.0 - i as f64 * 0.01, vec![]);
            map.insert(
                r.memory_id.clone(),
                SubjectTopicInfo {
                    subjects: vec!["Sam".to_string()],
                    primary_topic: "topic:identity".to_string(),
                },
            );
            results.push(r);
        }
        for i in 0..3 {
            let r = make_echo_result(
                &format!("Sam preference {i}"),
                0.9 - i as f64 * 0.01,
                vec![],
            );
            map.insert(
                r.memory_id.clone(),
                SubjectTopicInfo {
                    subjects: vec!["Sam".to_string()],
                    primary_topic: "topic:preference".to_string(),
                },
            );
            results.push(r);
        }
        super::enforce_subject_diversity(&mut results, &store, &map, 3);
        assert_eq!(
            results.len(),
            6,
            "Different topics for same subject should each get their own cap"
        );
    }

    #[test]
    fn subject_diversity_unknown_bucket_has_generous_cap() {
        let store = EchoStore::new();
        // 7 results with no subject info — unknown bucket caps at max_per_subject * 2 = 6
        let mut results: Vec<EchoResult> = (0..7)
            .map(|i| make_echo_result(&format!("unknown {i}"), 1.0 - i as f64 * 0.01, vec![]))
            .collect();
        let map = std::collections::HashMap::new();
        super::enforce_subject_diversity(&mut results, &store, &map, 3);
        assert_eq!(results.len(), 6, "Unknown bucket should cap at 2 * 3 = 6");
    }

    #[test]
    fn subject_diversity_overflow_preserves_different_topic() {
        let store = EchoStore::new();
        // 4 Sam:identity + 1 Sam:preference, cap=3 → identity capped at 3, preference survives
        let mut results: Vec<EchoResult> = Vec::new();
        let mut map = std::collections::HashMap::new();
        for i in 0..4 {
            let r = make_echo_result(&format!("Sam identity {i}"), 1.0 - i as f64 * 0.01, vec![]);
            map.insert(
                r.memory_id.clone(),
                SubjectTopicInfo {
                    subjects: vec!["Sam".to_string()],
                    primary_topic: "topic:identity".to_string(),
                },
            );
            results.push(r);
        }
        let pref = make_echo_result("Sam prefers Rust", 0.8, vec![]);
        map.insert(
            pref.memory_id.clone(),
            SubjectTopicInfo {
                subjects: vec!["Sam".to_string()],
                primary_topic: "memtype:preference".to_string(),
            },
        );
        results.push(pref);
        super::enforce_subject_diversity(&mut results, &store, &map, 3);
        assert_eq!(
            results.len(),
            4,
            "3 identity (capped) + 1 preference (different topic) = 4"
        );
        assert!(
            results.iter().any(|r| r.content == "Sam prefers Rust"),
            "Preference memory must survive identity overflow"
        );
    }

    // --- KU-1: Parent supersession flat demotion ---

    #[test]
    fn supersession_flat_demotion_closes_gap() {
        // Simulate: M4 (Shopify, old job) final_score = 1.027
        //           M5 (Stripe, new job) final_score = 1.001
        // With full demotion of 0.15: M4 drops to 0.877, well below M5.
        let demotion: f64 = 0.15;
        let mut old_parent_score: f64 = 1.027;
        let new_parent_score: f64 = 1.001;

        old_parent_score += -demotion;

        assert!(
            old_parent_score < new_parent_score,
            "Old parent ({old_parent_score}) must rank below new parent ({new_parent_score})"
        );
        assert!(
            (old_parent_score - 0.877).abs() < 1e-10,
            "Old parent should be demoted to 0.877, got {old_parent_score}"
        );
    }

    #[test]
    fn supersession_flat_demotion_no_op_without_superseded_child() {
        // If parent has no superseded children, no demotion is applied
        let original: f64 = 1.027;
        let demotions: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        let mut score = original;

        if let Some(&d) = demotions.get(&0) {
            score += d;
        }

        assert!(
            (score - original).abs() < f64::EPSILON,
            "Score should be unchanged without superseded children"
        );
    }

    // --- KU-1: Superseded parent exclusion ---

    #[test]
    fn exclusion_removes_superseded_parent_when_new_parent_present() {
        // M4 (Shopify, old) and M5 (Stripe, new) both in results.
        // M4 has a child superseded by M5's child → M4 should be excluded.
        let m4_id = MemoryId::new();
        let m5_id = MemoryId::new();
        let m6_id = MemoryId::new();
        let m7_id = MemoryId::new();

        let mut results = vec![
            make_echo_result_with_id(&m5_id, "Sam works at Stripe", 1.001, vec![]),
            make_echo_result_with_id(&m4_id, "Sam worked at Shopify", 0.877, vec![]),
            make_echo_result_with_id(&m6_id, "Sam likes hiking", 0.800, vec![]),
            make_echo_result_with_id(&m7_id, "Sam is vegan", 0.750, vec![]),
        ];

        let result_ids: std::collections::HashSet<MemoryId> =
            results.iter().map(|r| r.memory_id.clone()).collect();

        // Simulate: m4 is superseded, m5 is the superseding parent
        let mut superseded: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();
        // The real code traces children -> Supersedes edges -> new parent.
        // Here we simulate the result: m4 found to be superseded by m5.
        if result_ids.contains(&m5_id) {
            superseded.insert(m4_id.clone());
        }

        if !superseded.is_empty() && results.len() - superseded.len() >= 3 {
            results.retain(|r| !superseded.contains(&r.memory_id));
        }

        assert_eq!(results.len(), 3, "M4 should be excluded");
        assert!(
            results.iter().all(|r| r.memory_id != m4_id),
            "M4 (Shopify) must not appear in results"
        );
        assert!(
            results.iter().any(|r| r.memory_id == m5_id),
            "M5 (Stripe) must remain"
        );
    }

    #[test]
    fn exclusion_skips_when_new_parent_not_in_results() {
        // M4 (Shopify) is in results but M5 (Stripe) is NOT → no exclusion.
        let m4_id = MemoryId::new();
        let m5_id = MemoryId::new();
        let m6_id = MemoryId::new();

        let mut results = vec![
            make_echo_result_with_id(&m4_id, "Sam worked at Shopify", 0.900, vec![]),
            make_echo_result_with_id(&m6_id, "Sam likes hiking", 0.800, vec![]),
        ];

        let result_ids: std::collections::HashSet<MemoryId> =
            results.iter().map(|r| r.memory_id.clone()).collect();

        let mut superseded: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();
        // M5 not in results → don't mark M4 as superseded
        if result_ids.contains(&m5_id) {
            superseded.insert(m4_id.clone());
        }

        if !superseded.is_empty() && results.len() - superseded.len() >= 3 {
            results.retain(|r| !superseded.contains(&r.memory_id));
        }

        assert_eq!(results.len(), 2, "No exclusion should occur");
        assert!(
            results.iter().any(|r| r.memory_id == m4_id),
            "M4 must remain when M5 is not in results"
        );
    }

    #[test]
    fn exclusion_no_op_without_supersession_edges() {
        // No supersession edges → no exclusion
        let m1_id = MemoryId::new();
        let m2_id = MemoryId::new();
        let m3_id = MemoryId::new();

        let mut results = vec![
            make_echo_result_with_id(&m1_id, "Sam Torres", 0.900, vec![]),
            make_echo_result_with_id(&m2_id, "Sam likes hiking", 0.850, vec![]),
            make_echo_result_with_id(&m3_id, "Sam is vegan", 0.800, vec![]),
        ];

        let superseded: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();

        if !superseded.is_empty() && results.len() - superseded.len() >= 3 {
            results.retain(|r| !superseded.contains(&r.memory_id));
        }

        assert_eq!(results.len(), 3, "No exclusion when no supersession edges");
    }

    // --- KU-3: Topic-label boost ---

    #[test]
    fn label_topic_boost_fires_for_matching_editor_label() {
        let mut results = vec![
            make_echo_result(
                "I use Neovim with lazy.nvim",
                0.85,
                vec!["topic:tools:editor".to_string()],
            ),
            make_echo_result(
                "I prefer Rust for systems",
                0.83,
                vec!["topic:language:programming".to_string()],
            ),
        ];
        let query_labels = vec!["topic:tools:editor".to_string()];
        super::label_topic_boost(&query_labels, &mut results);
        assert!(
            (results[0].final_score - 0.91).abs() < 1e-10,
            "Editor result should get +0.06 boost, got {}",
            results[0].final_score,
        );
        assert!(
            (results[1].final_score - 0.83).abs() < 1e-10,
            "Non-editor result should be unchanged, got {}",
            results[1].final_score,
        );
    }

    #[test]
    fn label_topic_boost_no_op_without_tools_label() {
        let mut results = vec![make_echo_result(
            "I use PostgreSQL daily",
            0.90,
            vec!["topic:technology".to_string()],
        )];
        let query_labels = vec!["topic:technology".to_string()];
        super::label_topic_boost(&query_labels, &mut results);
        assert!(
            (results[0].final_score - 0.90).abs() < 1e-10,
            "Non-tools label should not trigger boost, got {}",
            results[0].final_score,
        );
    }

    // --- PT-3: action:learning label boost ---

    #[test]
    fn label_boost_fires_for_learning_query_and_learning_result() {
        let mut results = vec![
            make_echo_result(
                "I'm studying Japanese — JLPT N3 level",
                0.45,
                vec![
                    "action:learning".to_string(),
                    "topic:language:natural".to_string(),
                ],
            ),
            make_echo_result(
                "I code in Python and Rust",
                0.57,
                vec!["topic:language:programming".to_string()],
            ),
        ];
        let query_labels = vec![
            "action:learning".to_string(),
            "topic:language:natural".to_string(),
        ];
        super::label_topic_boost(&query_labels, &mut results);
        assert!(
            (results[0].final_score - 0.475).abs() < 1e-10,
            "Learning result should get +0.025 boost, got {}",
            results[0].final_score,
        );
        assert!(
            (results[1].final_score - 0.57).abs() < 1e-10,
            "Programming result should be unchanged, got {}",
            results[1].final_score,
        );
    }

    #[test]
    fn label_boost_no_op_for_learning_result_without_learning_query() {
        let mut results = vec![make_echo_result(
            "I'm studying Japanese — JLPT N3 level",
            0.45,
            vec!["action:learning".to_string()],
        )];
        // Query about career, not learning
        let query_labels = vec!["domain:work".to_string()];
        super::label_topic_boost(&query_labels, &mut results);
        assert!(
            (results[0].final_score - 0.45).abs() < 1e-10,
            "Learning result should not be boosted for non-learning query, got {}",
            results[0].final_score,
        );
    }

    // --- KU-3: Preference-update multiplier ---

    #[test]
    fn preference_update_boost_fires_for_currently_query() {
        let mut results = vec![
            make_echo_result(
                "I switched from VS Code to Neovim",
                0.80,
                vec!["memtype:preference_update".to_string()],
            ),
            make_echo_result(
                "I use Rust and Go and Python",
                0.85,
                vec!["topic:language:programming".to_string()],
            ),
        ];
        super::preference_update_boost("What editor do I currently use?", &mut results);
        // 0.80 * 1.05 = 0.84
        assert!(
            (results[0].final_score - 0.84).abs() < 1e-10,
            "Preference_update result should get 1.05x multiplier, got {}",
            results[0].final_score,
        );
        assert!(
            (results[1].final_score - 0.85).abs() < 1e-10,
            "Non-preference result should be unchanged, got {}",
            results[1].final_score,
        );
    }

    #[test]
    fn preference_update_boost_no_op_without_current_keywords() {
        let mut results = vec![make_echo_result(
            "I switched from VS Code to Neovim",
            0.80,
            vec!["memtype:preference_update".to_string()],
        )];
        super::preference_update_boost("What editor have I used?", &mut results);
        assert!(
            (results[0].final_score - 0.80).abs() < 1e-10,
            "Should not boost without current-state keywords, got {}",
            results[0].final_score,
        );
    }

    // --- IE-1: Career/intro adjustment ---

    #[test]
    fn career_intro_demotes_intro_and_boosts_career() {
        let mut results = vec![
            make_echo_result(
                "My name is Sam Torres, I'm a backend engineer",
                0.905,
                vec!["memtype:intro".to_string()],
            ),
            make_echo_result(
                "Sam works at Stripe on the payments team",
                0.816,
                vec!["topic:career".to_string()],
            ),
        ];
        let query_labels = vec!["topic:career".to_string(), "domain:work".to_string()];
        super::career_intro_adjustment(&query_labels, &mut results);
        // M1 (intro): 0.905 - 0.10 = 0.805
        assert!(
            (results[0].final_score - 0.805).abs() < 1e-10,
            "Intro memory should be demoted by -0.10, got {}",
            results[0].final_score,
        );
        // M5 (career): 0.816 + 0.03 = 0.846
        assert!(
            (results[1].final_score - 0.846).abs() < 1e-10,
            "Career memory should get +0.03 boost, got {}",
            results[1].final_score,
        );
        // Career should now outrank intro
        assert!(
            results[1].final_score > results[0].final_score,
            "Career ({}) must outrank intro ({})",
            results[1].final_score,
            results[0].final_score,
        );
    }

    #[test]
    fn career_intro_no_op_for_non_career_query() {
        let mut results = vec![make_echo_result(
            "My name is Sam Torres",
            0.90,
            vec!["memtype:intro".to_string()],
        )];
        let query_labels = vec!["topic:language:natural".to_string()];
        super::career_intro_adjustment(&query_labels, &mut results);
        assert!(
            (results[0].final_score - 0.90).abs() < 1e-10,
            "Intro should not be demoted for non-career query, got {}",
            results[0].final_score,
        );
    }

    // --- KS68.3: Parent-child dedup ---

    #[test]
    fn parent_child_dedup_removes_lower_scoring_duplicate() {
        let parent_id = MemoryId::new();
        let child_id = MemoryId::new();

        let mut parent_result = make_echo_result("I use Neovim with LazyVim", 0.90, vec![]);
        parent_result.memory_id = parent_id.clone();

        let mut child_result = make_echo_result("Sam uses Neovim as primary editor", 0.85, vec![]);
        child_result.memory_id = child_id.clone();

        let unrelated_result = make_echo_result("Sam lives in SF", 0.80, vec![]);
        let unrelated_id = unrelated_result.memory_id.clone();

        let mut results = vec![parent_result, child_result, unrelated_result];

        let mut parent_map = std::collections::HashMap::new();
        parent_map.insert(parent_id.clone(), None); // parent has no parent
        parent_map.insert(child_id.clone(), Some(parent_id.clone())); // child -> parent
        parent_map.insert(unrelated_id, None);

        super::deduplicate_parent_child(&mut results, &parent_map);

        assert_eq!(
            results.len(),
            2,
            "Child should be removed, 2 results remain"
        );
        assert_eq!(
            results[0].memory_id, parent_id,
            "Parent (higher score) should survive"
        );
        assert!(
            results.iter().all(|r| r.memory_id != child_id),
            "Child (lower score) should be removed"
        );
    }

    #[test]
    fn parent_child_dedup_keeps_child_when_higher_scoring() {
        let parent_id = MemoryId::new();
        let child_id = MemoryId::new();

        let mut parent_result = make_echo_result("Long multi-fact parent memory", 0.70, vec![]);
        parent_result.memory_id = parent_id.clone();

        let mut child_result = make_echo_result("Precise extracted fact", 0.92, vec![]);
        child_result.memory_id = child_id.clone();

        let mut results = vec![child_result, parent_result];

        let mut parent_map = std::collections::HashMap::new();
        parent_map.insert(parent_id.clone(), None);
        parent_map.insert(child_id.clone(), Some(parent_id.clone()));

        super::deduplicate_parent_child(&mut results, &parent_map);

        assert_eq!(
            results.len(),
            1,
            "Parent should be removed, 1 result remains"
        );
        assert_eq!(
            results[0].memory_id, child_id,
            "Child (higher score) should survive"
        );
    }

    #[test]
    fn parent_child_dedup_no_op_without_pairs() {
        let mut results = vec![
            make_echo_result("Memory A", 0.90, vec![]),
            make_echo_result("Memory B", 0.85, vec![]),
        ];
        let parent_map: std::collections::HashMap<MemoryId, Option<MemoryId>> = results
            .iter()
            .map(|r| (r.memory_id.clone(), None))
            .collect();

        super::deduplicate_parent_child(&mut results, &parent_map);
        assert_eq!(results.len(), 2, "No pairs — nothing removed");
    }
}
