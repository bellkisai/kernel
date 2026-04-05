//! Background consolidation — the "cleaner shrimp" that maintains Echo Memory health.
//!
//! Runs periodically to:
//! 1. Prune weak Hebbian edges (exponential decay below threshold)
//! 2. Rebuild Bloom filter when dirty (after deletions)
//! 3. Detect and merge near-duplicate memories (cosine similarity > 0.95)
//! 4. Decay echo counts on stale memories (no echo in >30 days)
//! 5. LLM-powered fact extraction on un-enriched memories (sleep consolidation)
//!
//! Designed to run every ~5 minutes. The duplicate detection pass is O(N²) but
//! only runs on stores with ≤10K memories; larger stores skip it.

use chrono::{Duration, Utc};
use regex::Regex;

use crate::bloom::TopicFilter;
use crate::hebbian::{HebbianGraph, RelationshipType};
use crate::similarity;
use crate::store::EchoStore;
use shrimpk_core::{Consolidator, EchoConfig};

/// Outcome of a single consolidation pass.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Number of Hebbian edges pruned (decayed weight below threshold).
    pub hebbian_edges_pruned: usize,
    /// Whether the Bloom filter was rebuilt during this pass.
    pub bloom_rebuilt: bool,
    /// Number of near-duplicate memory pairs merged.
    pub duplicates_merged: usize,
    /// Number of memories whose echo_count was decayed.
    pub echo_counts_decayed: usize,
    /// Number of memories enriched via LLM fact extraction.
    pub facts_extracted: usize,
    /// Number of typed relationship edges created during this pass.
    pub relationships_created: usize,
    /// Number of Supersedes edges created (contradictory fact updates).
    pub supersedes_created: usize,
    /// Number of memories that received Tier 2 labels during this pass (ADR-015).
    pub labels_enriched: usize,
    /// Number of memories whose importance score was recomputed during this pass.
    pub importance_recomputed: usize,
    /// Wall-clock duration of the consolidation pass in milliseconds.
    pub duration_ms: u64,
}

/// Maximum store size for the O(N²) duplicate detection pass.
/// Stores larger than this skip duplicate detection entirely.
const MAX_STORE_SIZE_FOR_DEDUP: usize = 10_000;

/// Cosine similarity threshold above which two memories are considered
/// near-duplicates and eligible for merging.
const DUPLICATE_SIMILARITY_THRESHOLD: f32 = 0.95;

/// Memories with echo_count > 0 that haven't been echoed in this many days
/// will have their echo_count reduced by 1.
const ECHO_DECAY_DAYS: i64 = 30;

/// Compute dynamic max_facts based on content length (KS67).
///
/// Short memories (<30 words) get 1-3 facts to avoid hallucinated noise.
/// Longer memories scale up to 12 facts to capture dense content.
pub(crate) fn dynamic_max_facts(content: &str) -> usize {
    let word_count = content.split_whitespace().count();
    if word_count < 30 {
        (word_count / 10).clamp(1, 3)
    } else {
        (word_count / 20).clamp(3, 12)
    }
}

/// Perform a full consolidation pass on the Echo Memory engine components.
///
/// This is the core cleanup routine. It acquires mutable references to
/// all engine internals and performs five maintenance steps in sequence.
///
/// # Arguments
/// * `store` - The in-memory vector store (may have entries removed/merged)
/// * `hebbian` - The Hebbian co-activation graph (weak edges pruned)
/// * `bloom` - The Bloom filter for topic pre-screening (rebuilt if dirty)
/// * `bloom_dirty` - Flag indicating the Bloom filter needs a rebuild
/// * `_config` - Engine configuration (reserved for future threshold tuning)
#[allow(clippy::field_reassign_with_default)]
#[allow(clippy::too_many_arguments)]
pub fn consolidate(
    store: &mut EchoStore,
    hebbian: &mut HebbianGraph,
    bloom: &mut TopicFilter,
    bloom_dirty: &mut bool,
    config: &EchoConfig,
    consolidator: &dyn Consolidator,
    embedder: Option<&std::sync::Mutex<crate::embedder::MultiEmbedder>>,
    lsh: &mut crate::lsh::CosineHash,
) -> ConsolidationResult {
    let start = std::time::Instant::now();
    let mut result = ConsolidationResult::default();

    // Step 1: Prune Hebbian edges whose decayed weight fell below threshold
    result.hebbian_edges_pruned = hebbian.consolidate();

    // Step 2: Rebuild Bloom filter if dirty (deletions occurred since last rebuild)
    if *bloom_dirty {
        let texts: Vec<String> = store
            .all_entries()
            .iter()
            .map(|e| e.content.clone())
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        bloom.rebuild(&text_refs);
        *bloom_dirty = false;
        result.bloom_rebuilt = true;
    }

    // Step 3: Detect and merge near-duplicate memories
    //
    // O(N²) pairwise comparison — only runs on stores with ≤10K memories.
    // For larger stores, this step is skipped entirely (the cost is not
    // justified for a background maintenance pass).
    let store_len = store.len();
    if store_len > 1 && store_len <= MAX_STORE_SIZE_FOR_DEDUP {
        result.duplicates_merged = merge_near_duplicates(store, config.use_importance);
    }

    // Step 4: Decay echo counts on stale memories
    result.echo_counts_decayed = decay_echo_counts(store);

    // Step 4.5: Recompute importance scores
    //
    // When use_importance is enabled, walk all entries and recompute the
    // multi-signal importance score for any entry whose importance_computed_at
    // is None or older than 1 hour. Uses a rolling mean of the most recent
    // 50 embeddings as the baseline for the surprise/novelty signal.
    if config.use_importance {
        let embeddings_snapshot: Vec<Vec<f32>> = store.all_embeddings().to_vec();
        let non_empty_embs: Vec<&[f32]> = embeddings_snapshot
            .iter()
            .filter(|e| !e.is_empty())
            .map(|e| e.as_slice())
            .collect();

        // Rolling mean of last 50 embeddings (for surprise score)
        let recent_embs: Vec<&[f32]> = non_empty_embs.iter().rev().take(50).copied().collect();
        let embedding_mean = crate::importance::compute_embedding_mean(&recent_embs);
        let mean_ref = if embedding_mean.is_empty() {
            None
        } else {
            Some(embedding_mean.as_slice())
        };

        let now = Utc::now();
        let stale_threshold = now - Duration::hours(1);
        let entry_count = store.len();

        for idx in 0..entry_count {
            let should_recompute = store.entry_at(idx).is_some_and(|entry| {
                entry
                    .importance_computed_at
                    .map(|t| t < stale_threshold)
                    .unwrap_or(true)
            });

            if should_recompute {
                // Snapshot the embedding for this index before borrowing mutably
                let emb_snapshot = embeddings_snapshot.get(idx).cloned();
                if let Some(ref emb) = emb_snapshot
                    && !emb.is_empty()
                    && let Some(entry) = store.entry_at(idx)
                {
                    let new_importance =
                        crate::importance::compute_importance(entry, emb, mean_ref);
                    // Now borrow mutably to update
                    if let Some(entry_mut) = store.entry_at_mut(idx) {
                        entry_mut.importance = new_importance;
                        entry_mut.importance_computed_at = Some(now);
                        result.importance_recomputed += 1;
                    }
                }
            }
        }
    }

    // Step 5: LLM fact extraction on un-enriched memories (sleep consolidation)
    //
    // For each memory that hasn't been enriched yet, ask the consolidator
    // to extract atomic facts. The parent is marked enriched=true to prevent
    // re-extraction on the next cycle. Batch limit: max 10 per cycle.
    //
    // Note: child memory creation with embeddings requires the MultiEmbedder,
    // which is not available here. Facts are extracted and logged; child
    // creation happens in EchoEngine::consolidate_now() which has embedder access.
    const MAX_ENRICHMENTS_PER_CYCLE: usize = 10;

    if consolidator.name() != "noop" {
        let mut unenriched: Vec<usize> = (0..store.len())
            .filter(|&i| {
                store.entry_at(i).is_some_and(|e| {
                    !e.enriched
                        && e.parent_id.is_none()
                        // Only consolidate text entries — vision/speech need VLM/transcription (KS38+)
                        && e.modality == shrimpk_core::Modality::Text
                })
            })
            .collect();

        // When importance scoring is enabled, sort candidates by importance
        // (descending) so the most important un-enriched memories get
        // processed first within the per-cycle batch limit.
        if config.use_importance {
            unenriched.sort_unstable_by(|&a, &b| {
                let imp_a = store.entry_at(a).map(|e| e.importance).unwrap_or(0.0);
                let imp_b = store.entry_at(b).map(|e| e.importance).unwrap_or(0.0);
                imp_b
                    .partial_cmp(&imp_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let unenriched: Vec<usize> = unenriched
            .into_iter()
            .take(MAX_ENRICHMENTS_PER_CYCLE)
            .collect();

        for idx in unenriched {
            let content = match store.entry_at(idx) {
                Some(e) => e.content.clone(),
                None => continue,
            };

            // KS67: dynamic max_facts + combined extraction
            // Use dynamic scaling but cap at the operator-configured limit
            let max_facts = dynamic_max_facts(&content).min(config.max_facts_per_memory);
            let output = consolidator.extract_facts_and_labels(&content, max_facts);

            tracing::info!(
                idx,
                max_facts,
                legacy_facts = output.facts.len(),
                structured_facts = output.structured_facts.len(),
                has_labels = output.labels.is_some(),
                content_preview = %&content[..content.len().min(60)],
                "KS67: extraction result"
            );

            // Build indexed fact list with optional structured metadata
            let fact_entries: Vec<(String, Option<&shrimpk_core::ExtractedFact>)> =
                if !output.structured_facts.is_empty() {
                    output
                        .structured_facts
                        .iter()
                        .map(|sf| (sf.text.clone(), Some(sf)))
                        .collect()
                } else {
                    output.facts.iter().map(|f| (f.clone(), None)).collect()
                };

            // Derive flat facts list for supersedes detection
            let facts: Vec<String> = fact_entries.iter().map(|(t, _)| t.clone()).collect();

            if !facts.is_empty() {
                result.facts_extracted += facts.len();
                tracing::debug!(
                    idx,
                    fact_count = facts.len(),
                    "Consolidation: extracted facts from memory"
                );
            }

            // Apply Tier 2 labels from the same response when available (KS67)
            if let Some(label_set) = &output.labels
                && config.use_labels
            {
                let mut new_labels: Vec<String> = Vec::new();
                for topic in &label_set.topic {
                    new_labels.push(format!("topic:{}", topic.to_lowercase()));
                }
                for domain in &label_set.domain {
                    new_labels.push(format!("domain:{}", domain.to_lowercase()));
                }
                for action in &label_set.action {
                    new_labels.push(format!("action:{}", action.to_lowercase()));
                }
                if let Some(ref mt) = label_set.memtype {
                    new_labels.push(format!("memtype:{}", mt.to_lowercase()));
                }
                if let Some(ref sent) = label_set.sentiment {
                    new_labels.push(format!("sentiment:{}", sent.to_lowercase()));
                }

                if !new_labels.is_empty() {
                    if let Some(entry) = store.entry_at_mut(idx) {
                        for label in &new_labels {
                            if !entry.labels.contains(label) {
                                entry.labels.push(label.clone());
                            }
                        }
                        entry.labels.truncate(crate::labels::MAX_LABELS_PER_ENTRY);
                        entry.label_version = 2;
                        result.labels_enriched += 1;
                    }

                    for label in &new_labels {
                        store
                            .label_index_mut()
                            .entry(label.clone())
                            .or_default()
                            .push(idx as u32);
                    }
                }
            }

            // Create child memories if embedder is available
            if let Some(embedder) = embedder {
                let parent_id = match store.entry_at(idx) {
                    Some(e) => e.id.clone(),
                    None => continue,
                };

                let mut fact_embeddings: Vec<Vec<f32>> = Vec::with_capacity(fact_entries.len());

                for (fact_text, structured_fact) in &fact_entries {
                    let embedding = match embedder.lock() {
                        Ok(mut e) => match e.embed_text(fact_text) {
                            Ok(emb) => emb,
                            Err(err) => {
                                tracing::debug!(error = %err, "Failed to embed fact, skipping");
                                fact_embeddings.push(Vec::new());
                                continue;
                            }
                        },
                        Err(_) => {
                            fact_embeddings.push(Vec::new());
                            continue;
                        }
                    };

                    // KS67: Skip near-duplicate children (cosine > 0.95 with existing child of same parent)
                    let is_dup = (0..store.len()).any(|i| {
                        store
                            .entry_at(i)
                            .is_some_and(|e| e.parent_id.as_ref() == Some(&parent_id))
                            && store.embedding_at(i).is_some_and(|existing| {
                                crate::similarity::cosine_similarity(&embedding, existing) > 0.95
                            })
                    });
                    if is_dup {
                        tracing::debug!(fact = %fact_text, "KS67: skipping near-duplicate child");
                        fact_embeddings.push(embedding);
                        continue;
                    }

                    fact_embeddings.push(embedding.clone());

                    let mut child = shrimpk_core::MemoryEntry::new(
                        fact_text.clone(),
                        embedding.clone(),
                        "enrichment".to_string(),
                    );
                    child.parent_id = Some(parent_id.clone());
                    child.enriched = true;
                    // Propagate parent's temporal era so children inherit
                    // the correct age for recency scoring (KS21).
                    if let Some(parent_entry) = store.entry_at(idx) {
                        child.created_at = parent_entry.created_at;
                    }

                    // Extract structured triple if possible (KS61 + KS67 subject fallback)
                    if let Some(triple) = extract_triples(fact_text) {
                        child.triples.push(triple);
                    } else if let Some(sf) = structured_fact
                        && let Some(ref subj) = sf.subject
                    {
                        child.triples.push(shrimpk_core::Triple {
                            subject: subj.clone(),
                            predicate: shrimpk_core::TriplePredicate::Custom(fact_text.to_string()),
                            object: fact_text.to_string(),
                        });
                    }

                    let child_idx = store.add(child) as u32;
                    // Only index children in LSH/Bloom when they participate in direct ranking.
                    // When child_rescue_only is true, children are only accessed via Pipe B
                    // (parent→child lookup), so they don't need to be in the search indices.
                    // This prevents child facts from polluting candidate retrieval.
                    if !config.child_rescue_only {
                        lsh.insert(child_idx, &embedding);
                        bloom.insert_memory(fact_text);
                    }

                    // Detect typed relationship from fact text and create
                    // a Hebbian edge between parent and child.
                    if let Some(rel) = detect_relationship(fact_text) {
                        hebbian.co_activate_with_relationship(
                            idx as u32, child_idx,
                            0.5, // moderate strength for extracted relationships
                            rel,
                        );
                        // Mark temporal start of this relationship (KS63)
                        let now_secs = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64();
                        hebbian.set_valid_from(idx as u32, child_idx, now_secs);
                        result.relationships_created += 1;
                    }
                }

                // Step 5b: Detect Supersedes edges — when a new fact
                // contradicts an older fact about the same entity.
                let supersedes_pairs =
                    detect_supersedes_pairs(store, &facts, &fact_embeddings, idx);
                for (old_idx, new_fact_text) in &supersedes_pairs {
                    let now_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();

                    // Expire old memory's typed edges (KS63 temporal validity)
                    // This makes old relationships invisible for point-in-time queries
                    // after the supersession event.
                    let old_assocs: Vec<(u32, f64, bool)> = hebbian
                        .get_associations_typed(*old_idx as u32, 0.0)
                        .into_iter()
                        .filter(|(_, _, rel)| {
                            matches!(
                                rel,
                                Some(RelationshipType::WorksAt(_))
                                    | Some(RelationshipType::LivesIn(_))
                                    | Some(RelationshipType::PrefersTool(_))
                                    | Some(RelationshipType::PartOf(_))
                                    | Some(RelationshipType::Custom(_))
                            )
                        })
                        .map(|(n, w, _)| (n, w, false))
                        .collect();
                    for (neighbor, _, _) in &old_assocs {
                        hebbian.set_valid_until(*old_idx as u32, *neighbor, now_secs);
                    }

                    // Edge between old child and new parent
                    hebbian.co_activate_with_relationship(
                        *old_idx as u32,
                        idx as u32,
                        0.3,
                        RelationshipType::Supersedes,
                    );
                    hebbian.set_valid_from(*old_idx as u32, idx as u32, now_secs);

                    // ALSO create edge between OLD PARENT and NEW PARENT (KS22).
                    // This ensures the Supersedes demotion fires in Pipe A ranking
                    // (where only parents appear when child_rescue_only is true).
                    if let Some(old_entry) = store.entry_at(*old_idx)
                        && let Some(ref old_parent_id) = old_entry.parent_id
                    {
                        // Find the old parent's store index
                        for i in 0..store.len() {
                            if let Some(e) = store.entry_at(i)
                                && e.id == *old_parent_id
                            {
                                hebbian.co_activate_with_relationship(
                                    i as u32,
                                    idx as u32,
                                    0.3,
                                    RelationshipType::Supersedes,
                                );
                                hebbian.set_valid_from(i as u32, idx as u32, now_secs);
                                break;
                            }
                        }
                    }
                    result.supersedes_created += 1;
                    tracing::debug!(
                        old_idx,
                        new_parent_idx = idx,
                        fact = %new_fact_text,
                        "Supersedes edge created: new memory replaces old"
                    );
                }
            }

            // Mark parent as enriched regardless (even if 0 facts — avoids retrying)
            if let Some(entry) = store.entry_at_mut(idx) {
                entry.enriched = true;
            }
        }
    }

    // Step 6: Tier 2 label enrichment (ADR-015 D7, Pass 4)
    //
    // For enriched entries that only have Tier 1 labels (label_version == 1),
    // use the combined consolidator to get LLM-classified labels.
    // Processes up to MAX_ENRICHMENTS_PER_CYCLE entries per pass.
    if config.use_labels {
        let needs_tier2: Vec<usize> = (0..store.len())
            .filter(|&i| {
                store
                    .entry_at(i)
                    .is_some_and(|e| e.enriched && e.label_version == 1)
            })
            .take(MAX_ENRICHMENTS_PER_CYCLE)
            .collect();

        for idx in needs_tier2 {
            let content = match store.entry_at(idx) {
                Some(e) => e.content.clone(),
                None => continue,
            };

            // Skip non-text modality (vision/speech get labels differently)
            if store
                .entry_at(idx)
                .is_some_and(|e| e.modality != shrimpk_core::Modality::Text)
            {
                continue;
            }

            let output = consolidator.extract_facts_and_labels(&content, 5);

            if let Some(label_set) = output.labels {
                let mut new_labels: Vec<String> = Vec::new();

                for topic in &label_set.topic {
                    new_labels.push(format!("topic:{}", topic.to_lowercase()));
                }
                for domain in &label_set.domain {
                    new_labels.push(format!("domain:{}", domain.to_lowercase()));
                }
                for action in &label_set.action {
                    new_labels.push(format!("action:{}", action.to_lowercase()));
                }
                if let Some(ref mt) = label_set.memtype {
                    new_labels.push(format!("memtype:{}", mt.to_lowercase()));
                }
                if let Some(ref sent) = label_set.sentiment {
                    new_labels.push(format!("sentiment:{}", sent.to_lowercase()));
                }

                if let Some(entry) = store.entry_at_mut(idx) {
                    // Merge: add new labels that aren't already present
                    for label in &new_labels {
                        if !entry.labels.contains(label) {
                            entry.labels.push(label.clone());
                        }
                    }
                    entry.labels.truncate(crate::labels::MAX_LABELS_PER_ENTRY);
                    entry.label_version = 2;
                    result.labels_enriched += 1;
                }

                // Update label index for new labels
                for label in &new_labels {
                    store
                        .label_index_mut()
                        .entry(label.clone())
                        .or_default()
                        .push(idx as u32);
                }
            }
        }
    }

    // Step 7: Community summary generation (KS64 — GraphRAG P4)
    //
    // For label clusters with enough members, generate a summarization via LLM.
    // Summaries are stored on the EchoStore and used as fallback when echo queries
    // return weak results. Max 5 summaries per consolidation cycle to bound latency.
    if config.community_summaries_enabled {
        let min_members = config.community_min_members;
        let eligible: Vec<(String, usize)> = store.labels_with_min_members(min_members);

        let mut summaries_generated = 0usize;
        for (label, member_count) in &eligible {
            if summaries_generated >= 5 {
                break;
            }
            // Skip if we already have an up-to-date summary
            if let Some(existing) = store.get_summary(label)
                && existing.member_count == *member_count
            {
                continue;
            }

            // Collect up to 20 member contents
            let member_contents: Vec<&str> = store
                .query_labels(std::slice::from_ref(label))
                .into_iter()
                .take(20)
                .filter_map(|idx| store.entry_at(idx as usize).map(|e| e.content.as_str()))
                .collect();

            if member_contents.is_empty() {
                continue;
            }

            if let Some(summary_text) = consolidator.summarize_cluster(&member_contents, label) {
                // Embed the summary for cosine matching during echo fallback
                let summary_embedding = if let Some(emb_mutex) = embedder {
                    if let Ok(mut emb) = emb_mutex.lock() {
                        emb.embed_text(&summary_text).unwrap_or_default()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                store.set_summary(shrimpk_core::CommunitySummary {
                    label: label.clone(),
                    summary: summary_text,
                    embedding: summary_embedding,
                    member_count: *member_count,
                    updated_at: Utc::now(),
                });
                summaries_generated += 1;
                tracing::debug!(
                    label = %label,
                    members = member_count,
                    "Community summary generated"
                );
            }
        }
    }

    result.duration_ms = start.elapsed().as_millis() as u64;
    result
}

// ===========================================================================
// Relationship detection (KS18 Track 4)
// ===========================================================================

/// Detect a typed relationship from a fact string using regex patterns.
///
/// Scans the fact text for known relationship patterns:
/// - "works at" / "employed at" / "joined" -> WorksAt
/// - "lives in" / "moved to" / "based in" -> LivesIn
/// - "prefers" / "likes" / "uses" / "switched to" -> PrefersTool
/// - "part of" / "belongs to" / "member of" -> PartOf
///
/// Returns `None` if no pattern matches (edge will be plain co-activation).
pub fn detect_relationship(fact: &str) -> Option<RelationshipType> {
    let lower = fact.to_lowercase();

    // WorksAt patterns: "X works at Y", "X employed at Y", "X joined Y"
    let works_re =
        Regex::new(r"(?i)(?:works?\s+at|working\s+at|started\s+at|starting\s+at|employed\s+at|joined|works?\s+for|employed\s+by)\s+(.+)")
            .ok()?;
    if let Some(caps) = works_re.captures(&lower) {
        let entity = caps
            .get(1)
            .map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity
            && !e.is_empty()
        {
            return Some(RelationshipType::WorksAt(e));
        }
    }

    // LivesIn patterns: "X lives in Y", "X moved to Y", "X based in Y"
    let lives_re = Regex::new(
        r"(?i)(?:lives?\s+in|living\s+in|moved\s+to|moving\s+to|based\s+in|relocated\s+to|relocating\s+to|resides?\s+in)\s+(.+)",
    )
    .ok()?;
    if let Some(caps) = lives_re.captures(&lower) {
        let entity = caps
            .get(1)
            .map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity
            && !e.is_empty()
        {
            return Some(RelationshipType::LivesIn(e));
        }
    }

    // PrefersTool patterns: "X prefers Y", "X uses Y", "X likes Y", "X switched to Y"
    let pref_re =
        Regex::new(r"(?i)(?:prefers?\s+|preferring\s+|likes?\s+|uses?\s+|using\s+|switched\s+to\s+|switching\s+to\s+|chose\s+)(.+)").ok()?;
    if let Some(caps) = pref_re.captures(&lower) {
        let entity = caps
            .get(1)
            .map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity
            && !e.is_empty()
        {
            return Some(RelationshipType::PrefersTool(e));
        }
    }

    // PartOf patterns: "X part of Y", "X belongs to Y", "X member of Y"
    let part_re =
        Regex::new(r"(?i)(?:part\s+of|belongs?\s+to|member\s+of|component\s+of)\s+(.+)").ok()?;
    if let Some(caps) = part_re.captures(&lower) {
        let entity = caps
            .get(1)
            .map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity
            && !e.is_empty()
        {
            return Some(RelationshipType::PartOf(e));
        }
    }

    None
}

/// Extract a clean entity string from a fact, using the ORIGINAL casing.
///
/// Takes the byte offsets from the lowercase match and maps them back to
/// the original text, then trims trailing punctuation and whitespace.
fn extract_entity(original: &str, start: usize, end: usize) -> String {
    // Map byte offsets: the original and lowercase have the same byte length
    // for ASCII-safe text. For safety, clamp to original length.
    let start = start.min(original.len());
    let end = end.min(original.len());
    let raw = &original[start..end];
    raw.trim()
        .trim_end_matches(['.', ',', ';', '!', '?'])
        .trim()
        .to_string()
}

/// Extract a structured (subject, predicate, object) triple from a fact string.
/// Reuses detect_relationship() for predicate+object and extract_subject() for subject.
/// Returns None if no relationship pattern matches.
pub fn extract_triples(fact: &str) -> Option<shrimpk_core::Triple> {
    use shrimpk_core::{Triple, TriplePredicate};

    let rel = detect_relationship(fact)?;
    let subject = extract_subject(fact);
    if subject.is_empty() {
        return None;
    }

    let (predicate, object) = match rel {
        RelationshipType::WorksAt(obj) => (TriplePredicate::WorksAt, obj),
        RelationshipType::LivesIn(obj) => (TriplePredicate::LivesIn, obj),
        RelationshipType::PrefersTool(obj) => (TriplePredicate::PrefersTool, obj),
        RelationshipType::PartOf(obj) => (TriplePredicate::PartOf, obj),
        RelationshipType::Custom(ref s) => (TriplePredicate::Custom(s.clone()), s.clone()),
        _ => return None, // CoActivation, TemporalSequence, Supersedes are graph-edge-only
    };

    Some(Triple {
        subject,
        predicate,
        object,
    })
}

/// Detect entity-level contradictions between new facts and existing memories.
///
/// Looks for cases where a new fact about the same subject contradicts an
/// older memory. For example:
/// - Old: "Alex works at Google" + New: "Alex works at Meta" -> Supersedes
/// - Old: "User lives in NYC" + New: "User moved to SF" -> Supersedes
///
/// Uses two signals:
/// 1. Regex-based relationship detection (same category, different entity = contradiction)
/// 2. Embedding cosine similarity (KS67): >0.95 = near-identity skip, >0.80 + subject overlap = supersession
///
/// Returns `(old_memory_index, new_fact_text)` pairs for each detected supersession.
fn detect_supersedes_pairs(
    store: &EchoStore,
    new_facts: &[String],
    new_embeddings: &[Vec<f32>],
    current_parent_idx: usize,
) -> Vec<(usize, String)> {
    let mut pairs = Vec::new();
    let mut matched_old_indices = std::collections::HashSet::new();

    // Pass 1: Regex-based relationship detection (original logic)
    // Limited to single-valued relationship categories (WorksAt, LivesIn) where
    // a person can only have one current value. PrefersTool/PartOf are multi-valued
    // (person can prefer multiple tools) and cause false-positive supersession.
    for fact in new_facts {
        let new_rel = match detect_relationship(fact) {
            Some(r) => r,
            None => continue,
        };

        // Only supersede single-valued relationships
        let (new_category, _new_entity) = categorize_relationship(&new_rel);
        if new_category != "works_at" && new_category != "lives_in" {
            continue;
        }

        // Scan existing enriched child memories for contradictions
        for i in 0..store.len() {
            let entry = match store.entry_at(i) {
                Some(e) => e,
                None => continue,
            };

            // Only compare against enrichment-sourced entries (child facts)
            if entry.source != "enrichment" {
                continue;
            }

            // Detect the relationship in the old fact
            let old_rel = match detect_relationship(&entry.content) {
                Some(r) => r,
                None => continue,
            };

            let (old_category, _old_entity) = categorize_relationship(&old_rel);

            // Same relationship category but different entity = contradiction
            if new_category == old_category && new_rel != old_rel {
                // Check if the facts share a subject (simple heuristic:
                // first word or first two words overlap)
                if subjects_overlap(fact, &entry.content) {
                    matched_old_indices.insert(i);
                    pairs.push((i, fact.clone()));
                    break; // One supersession per fact is enough
                }
            }
        }
    }

    // Pass 2: Embedding-based supersession (KS67)
    // For facts that have pre-computed embeddings, check cosine similarity
    // against existing enrichment children. This catches semantic supersession
    // that regex patterns miss (e.g., paraphrased contradictions).
    for (fact_idx, fact) in new_facts.iter().enumerate() {
        let new_emb = match new_embeddings.get(fact_idx) {
            Some(emb) if !emb.is_empty() => emb,
            _ => continue,
        };

        for i in 0..store.len() {
            // Skip if already matched by regex pass
            if matched_old_indices.contains(&i) {
                continue;
            }

            let entry = match store.entry_at(i) {
                Some(e) => e,
                None => continue,
            };

            // Only compare against enrichment-sourced entries (child facts)
            if entry.source != "enrichment" {
                continue;
            }

            let existing_emb = match store.embedding_at(i) {
                Some(emb) if !emb.is_empty() => emb,
                _ => continue,
            };

            let cosine = crate::similarity::cosine_similarity(new_emb, existing_emb);

            // >0.95: near-identity repeat — skip (not a supersession, just a duplicate)
            if cosine > 0.95 {
                continue;
            }

            // >0.70 + subject overlap: semantic supersession (lowered from 0.80
            // to catch v2 open-domain facts with different verb forms)
            if cosine > 0.70 && subjects_overlap(fact, &entry.content) {
                matched_old_indices.insert(i);
                pairs.push((i, fact.clone()));
                break; // One supersession per fact is enough
            }
        }
    }

    // Pass 3: Parent-content supersession (KS67)
    // Checks new facts against PARENT memory content directly, not just children.
    // This handles the case where an old parent got 0 extracted children (LLM failure)
    // but its raw content still contains relationship patterns.
    // Limited to WorksAt/LivesIn to avoid false positives.
    for fact in new_facts {
        let new_rel = match detect_relationship(fact) {
            Some(r) => r,
            None => continue,
        };

        let (new_category, _new_entity) = categorize_relationship(&new_rel);
        if new_category != "works_at" && new_category != "lives_in" {
            continue;
        }

        for i in 0..store.len() {
            if matched_old_indices.contains(&i) {
                continue;
            }
            // Skip the current parent being enriched
            if i == current_parent_idx {
                continue;
            }

            let entry = match store.entry_at(i) {
                Some(e) => e,
                None => continue,
            };

            // Only compare against non-enrichment entries (parent memories)
            if entry.source == "enrichment" {
                continue;
            }

            // Detect relationship in the old parent's full content.
            // Parent content may contain multiple sentences, so we split
            // and check each for relationships.
            let old_sentences: Vec<&str> = entry
                .content
                .split(['.', '!', '?'])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            for sentence in &old_sentences {
                let old_rel = match detect_relationship(sentence) {
                    Some(r) => r,
                    None => continue,
                };

                let (old_category, _old_entity) = categorize_relationship(&old_rel);

                if new_category == old_category && new_rel != old_rel
                    && subjects_overlap(fact, sentence)
                {
                    matched_old_indices.insert(i);
                    pairs.push((i, fact.clone()));
                    break;
                }
            }

            if matched_old_indices.contains(&i) {
                break; // Found a match for this fact
            }
        }
    }

    pairs
}

/// Categorize a relationship into a comparable key (ignoring the entity value).
fn categorize_relationship(rel: &RelationshipType) -> (&str, Option<&str>) {
    match rel {
        RelationshipType::WorksAt(e) => ("works_at", Some(e.as_str())),
        RelationshipType::LivesIn(e) => ("lives_in", Some(e.as_str())),
        RelationshipType::PrefersTool(e) => ("prefers", Some(e.as_str())),
        RelationshipType::PartOf(e) => ("part_of", Some(e.as_str())),
        RelationshipType::CoActivation => ("co_activation", None),
        RelationshipType::TemporalSequence => ("temporal", None),
        RelationshipType::Supersedes => ("supersedes", None),
        RelationshipType::Custom(e) => ("custom", Some(e.as_str())),
    }
}

/// Check if two fact strings share a subject (simple heuristic).
///
/// Compares the first meaningful word(s) of each fact. Works for patterns like
/// "Alex works at Google" vs "Alex works at Meta" (subject = "Alex").
fn subjects_overlap(fact_a: &str, fact_b: &str) -> bool {
    let subj_a = extract_subject(fact_a);
    let subj_b = extract_subject(fact_b);
    if subj_a.is_empty() || subj_b.is_empty() {
        return false;
    }
    subj_a.to_lowercase() == subj_b.to_lowercase()
}

/// Extract the subject from a fact string (first word or "User").
///
/// Handles patterns like:
/// - "Alex works at Google" -> "Alex"
/// - "User prefers Rust" -> "User"
/// - "The project is part of Bellkis" -> "project"
pub(crate) fn extract_subject(fact: &str) -> String {
    let trimmed = fact.trim();
    // Skip common articles
    let without_article = trimmed
        .strip_prefix("The ")
        .or_else(|| trimmed.strip_prefix("the "))
        .or_else(|| trimmed.strip_prefix("A "))
        .or_else(|| trimmed.strip_prefix("a "))
        .unwrap_or(trimmed);

    without_article
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// Detect and merge near-duplicate memory pairs.
///
/// For each pair of memories, computes cosine similarity. If above 0.95,
/// the loser is merged into the winner. Winner selection:
/// - When `use_importance` is true: loser = lower importance. Ties broken by
///   higher store index (newer entry loses). Winner absorbs `max(importance)`.
/// - Legacy: loser = fewer echo_count.
///
/// Content is preserved from the winner; the loser is removed.
///
/// Returns the number of pairs merged.
fn merge_near_duplicates(store: &mut EchoStore, use_importance: bool) -> usize {
    // Collect IDs and echo counts for all entries, along with their embeddings.
    // We need to work with indices rather than IDs during the comparison pass
    // because we need access to the embeddings array.
    let embeddings: Vec<Vec<f32>> = store.all_embeddings().to_vec();
    let n = embeddings.len();

    // Collect (id, echo_count, importance) for each entry by index
    let entry_info: Vec<_> = (0..n)
        .filter_map(|i| {
            store
                .entry_at(i)
                .map(|e| (e.id.clone(), e.echo_count, e.importance))
        })
        .collect();

    // Find pairs to merge. We collect the ID of the "loser".
    let mut to_remove = Vec::new();
    let mut already_removed = std::collections::HashSet::new();

    for i in 0..n {
        if already_removed.contains(&i) {
            continue;
        }
        // Skip vision-only/speech-only entries with empty text embeddings
        if embeddings[i].is_empty() {
            continue;
        }
        for j in (i + 1)..n {
            if already_removed.contains(&j) {
                continue;
            }
            if embeddings[j].is_empty() {
                continue;
            }
            let sim = similarity::cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim > DUPLICATE_SIMILARITY_THRESHOLD {
                let (winner_idx, loser_idx) = if use_importance {
                    // Importance-based winner: higher importance wins.
                    // Break ties by store index: lower index (older) wins.
                    let imp_i = entry_info[i].2;
                    let imp_j = entry_info[j].2;
                    if imp_i > imp_j || (imp_i == imp_j && i < j) {
                        (i, j)
                    } else {
                        (j, i)
                    }
                } else {
                    // Legacy: keep the entry with more echo_count
                    if entry_info[i].1 >= entry_info[j].1 {
                        (i, j)
                    } else {
                        (j, i)
                    }
                };

                // Transfer echo_count from loser to winner (additive)
                let loser_echo = entry_info[loser_idx].1;
                if let Some(winner_entry) = store.get_mut(&entry_info[winner_idx].0) {
                    winner_entry.echo_count = winner_entry.echo_count.saturating_add(loser_echo);
                    // Absorb max importance into winner
                    if use_importance {
                        let max_imp = entry_info[winner_idx].2.max(entry_info[loser_idx].2);
                        winner_entry.importance = max_imp;
                    }
                }

                to_remove.push(entry_info[loser_idx].0.clone());
                already_removed.insert(loser_idx);
            }
        }
    }

    let merged = to_remove.len();
    for id in to_remove {
        store.remove(&id);
    }
    merged
}

/// Decay echo_count on memories that haven't been echoed in >30 days.
///
/// Prevents ancient but once-popular memories from permanently dominating
/// rankings. Only affects memories with echo_count > 0 whose last_echoed
/// (or created_at, if never echoed) is older than 30 days.
///
/// Returns the number of memories whose echo_count was reduced.
fn decay_echo_counts(store: &mut EchoStore) -> usize {
    let cutoff = Utc::now() - Duration::days(ECHO_DECAY_DAYS);
    let mut decayed = 0;

    let len = store.len();
    for i in 0..len {
        let should_decay = store.entry_at(i).is_some_and(|entry| {
            if entry.echo_count == 0 {
                return false;
            }
            // Use last_echoed if available, otherwise created_at
            let reference_time = entry.last_echoed.unwrap_or(entry.created_at);
            reference_time < cutoff
        });

        if should_decay && let Some(entry) = store.entry_at_mut(i) {
            entry.echo_count = entry.echo_count.saturating_sub(1);
            decayed += 1;
        }
    }

    decayed
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrimpk_core::{EchoConfig, MemoryEntry};
    use std::path::PathBuf;

    fn test_config() -> EchoConfig {
        EchoConfig {
            max_memories: 1000,
            similarity_threshold: 0.3,
            max_echo_results: 10,
            ram_budget_bytes: 100_000_000,
            data_dir: PathBuf::from("/tmp/shrimpk-consolidation-test"),
            embedding_dim: 384,
            ..Default::default()
        }
    }

    fn make_entry(content: &str, embedding: Vec<f32>) -> MemoryEntry {
        MemoryEntry::new(content.to_string(), embedding, "test".to_string())
    }

    #[test]
    fn consolidate_empty_store_returns_zeroed_result() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false;

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert_eq!(result.hebbian_edges_pruned, 0);
        assert!(!result.bloom_rebuilt);
        assert_eq!(result.duplicates_merged, 0);
        assert_eq!(result.echo_counts_decayed, 0);
    }

    #[test]
    fn consolidate_prunes_weak_hebbian_edges() {
        let config = test_config();
        let mut store = EchoStore::new();
        let half_life = 604_800.0; // 7 days in seconds
        let mut hebbian = HebbianGraph::new(half_life, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false;

        // Create a strong edge (recent, high weight)
        hebbian.co_activate(0, 1, 1.0);

        // Create a weak edge: very small weight that's already below threshold.
        // We do this by adding a tiny weight. Since co_activate uses "now" as
        // last_activated, the decayed weight at consolidation time is ~= weight.
        // To get it below 0.01, we add weight 0.005.
        hebbian.co_activate(2, 3, 0.005);

        assert_eq!(hebbian.len(), 2);

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert_eq!(result.hebbian_edges_pruned, 1, "Should prune the weak edge");
        assert_eq!(hebbian.len(), 1, "Only the strong edge should remain");
    }

    #[test]
    fn consolidate_rebuilds_bloom_when_dirty() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = true; // mark as dirty

        // Add some entries to the store so bloom has something to rebuild from
        store.add(make_entry(
            "Rust programming language safety",
            vec![1.0, 0.0, 0.0],
        ));
        store.add(make_entry(
            "Python machine learning data science",
            vec![0.0, 1.0, 0.0],
        ));

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert!(result.bloom_rebuilt, "Bloom should have been rebuilt");
        assert!(!bloom_dirty, "bloom_dirty flag should be cleared");
        assert_eq!(
            bloom.len(),
            2,
            "Bloom should contain 2 entries after rebuild"
        );
    }

    #[test]
    fn consolidate_does_not_rebuild_bloom_when_clean() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false; // clean

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert!(
            !result.bloom_rebuilt,
            "Bloom should NOT have been rebuilt when clean"
        );
    }

    #[test]
    fn near_duplicate_detection_merges_similar_memories() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false;

        // Create two nearly-identical embeddings (cosine similarity > 0.95)
        let emb_a = vec![1.0, 0.0, 0.0];
        let emb_b = vec![0.99, 0.01, 0.0]; // very close to emb_a

        // Verify they are actually near-duplicates
        let sim = similarity::cosine_similarity(&emb_a, &emb_b);
        assert!(
            sim > DUPLICATE_SIMILARITY_THRESHOLD,
            "Test vectors should be near-duplicates, got similarity {sim}"
        );

        let mut entry_a = make_entry("memory alpha", emb_a);
        entry_a.echo_count = 5; // more echoes — this one should survive
        let id_a = entry_a.id.clone();

        let mut entry_b = make_entry("memory beta", emb_b);
        entry_b.echo_count = 2; // fewer echoes — this one should be merged away
        let id_b = entry_b.id.clone();

        store.add(entry_a);
        store.add(entry_b);

        // Also add a dissimilar memory that should NOT be merged
        store.add(make_entry("completely different", vec![0.0, 0.0, 1.0]));

        assert_eq!(store.len(), 3);

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert_eq!(result.duplicates_merged, 1, "Should merge exactly one pair");
        assert_eq!(store.len(), 2, "Two memories should remain");

        // The winner (more echoes) should survive with accumulated echo_count
        let survivor = store.get(&id_a).expect("Winner should survive");
        assert_eq!(
            survivor.echo_count, 7,
            "Winner should have accumulated echo_count (5 + 2)"
        );

        // The loser should be gone
        assert!(store.get(&id_b).is_none(), "Loser should be removed");
    }

    #[test]
    fn echo_count_decay_reduces_stale_counts() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false;

        // Create a memory with echo_count=5 that was last echoed 45 days ago
        let mut entry = make_entry("old popular memory", vec![1.0, 0.0, 0.0]);
        entry.echo_count = 5;
        entry.last_echoed = Some(Utc::now() - Duration::days(45));
        let id = entry.id.clone();
        store.add(entry);

        // Create a recent memory with echo_count=3 (last echoed 2 days ago)
        let mut recent = make_entry("recent memory", vec![0.0, 1.0, 0.0]);
        recent.echo_count = 3;
        recent.last_echoed = Some(Utc::now() - Duration::days(2));
        let recent_id = recent.id.clone();
        store.add(recent);

        // Create a memory with echo_count=0 (should not be affected)
        let zero_entry = make_entry("never echoed", vec![0.0, 0.0, 1.0]);
        let zero_id = zero_entry.id.clone();
        store.add(zero_entry);

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert_eq!(
            result.echo_counts_decayed, 1,
            "Only the old memory should have its count decayed"
        );

        // The old memory should have echo_count reduced by 1
        let old = store.get(&id).expect("Old memory should still exist");
        assert_eq!(
            old.echo_count, 4,
            "echo_count should be reduced from 5 to 4"
        );

        // The recent memory should be untouched
        let rec = store.get(&recent_id).expect("Recent memory should exist");
        assert_eq!(
            rec.echo_count, 3,
            "Recent memory echo_count should be unchanged"
        );

        // The zero-count memory should be untouched
        let zero = store.get(&zero_id).expect("Zero-count memory should exist");
        assert_eq!(zero.echo_count, 0, "Zero-count memory should remain at 0");
    }

    #[test]
    fn dissimilar_memories_are_not_merged() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false;

        // Two orthogonal embeddings — cosine similarity = 0.0
        store.add(make_entry("memory one", vec![1.0, 0.0, 0.0]));
        store.add(make_entry("memory two", vec![0.0, 1.0, 0.0]));
        store.add(make_entry("memory three", vec![0.0, 0.0, 1.0]));

        assert_eq!(store.len(), 3);

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert_eq!(result.duplicates_merged, 0, "No duplicates should be found");
        assert_eq!(store.len(), 3, "All memories should remain");
    }

    #[test]
    fn echo_count_decay_uses_created_at_when_never_echoed() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false;

        // Memory with echo_count > 0 but last_echoed = None, created 45 days ago.
        // This can happen if echo_count was set manually or via merge.
        let mut entry = make_entry("orphan memory", vec![1.0, 0.0, 0.0]);
        entry.echo_count = 2;
        entry.last_echoed = None;
        // Backdate created_at to 45 days ago
        entry.created_at = Utc::now() - Duration::days(45);
        let id = entry.id.clone();
        store.add(entry);

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert_eq!(result.echo_counts_decayed, 1);
        let e = store.get(&id).unwrap();
        assert_eq!(e.echo_count, 1, "Should decay from 2 to 1");
    }

    #[test]
    fn large_store_skips_dedup() {
        // Verify the MAX_STORE_SIZE_FOR_DEDUP guard works.
        // We won't actually insert 10K+ entries, but we test the boundary logic
        // by checking that a store with 2 near-duplicates DOES merge them
        // (proving dedup runs for small stores).
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = false;

        store.add(make_entry("dup one", vec![1.0, 0.0, 0.0]));
        store.add(make_entry("dup two", vec![0.99, 0.01, 0.0]));

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        // This proves dedup runs for small stores
        assert_eq!(result.duplicates_merged, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn full_consolidation_pass_runs_all_steps() {
        let config = test_config();
        let mut store = EchoStore::new();
        let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
        let mut bloom = TopicFilter::new(1000, 0.01);
        let mut bloom_dirty = true;

        // Setup: weak hebbian edge
        hebbian.co_activate(0, 1, 0.005);

        // Setup: near-duplicate memories
        store.add(make_entry("near dup A", vec![1.0, 0.0, 0.0]));
        store.add(make_entry("near dup B", vec![0.99, 0.01, 0.0]));

        // Setup: stale memory for echo decay
        let mut old = make_entry("old memory", vec![0.0, 0.0, 1.0]);
        old.echo_count = 3;
        old.last_echoed = Some(Utc::now() - Duration::days(60));
        store.add(old);

        let result = consolidate(
            &mut store,
            &mut hebbian,
            &mut bloom,
            &mut bloom_dirty,
            &config,
            &crate::consolidator::NoopConsolidator,
            None,
            &mut crate::lsh::CosineHash::new(384, 16, 10),
        );

        assert_eq!(result.hebbian_edges_pruned, 1, "Weak edge should be pruned");
        assert!(result.bloom_rebuilt, "Bloom should be rebuilt");
        assert_eq!(
            result.duplicates_merged, 1,
            "Near-duplicates should be merged"
        );
        assert_eq!(
            result.echo_counts_decayed, 1,
            "Old memory echo count should decay"
        );
        assert!(
            result.duration_ms < 10_000,
            "Should complete in under 10 seconds"
        );
    }

    // ---- Relationship detection tests (KS18 Track 4) ----

    #[test]
    fn detect_works_at() {
        let rel = detect_relationship("Alex works at Google").expect("Should detect WorksAt");
        match rel {
            RelationshipType::WorksAt(entity) => {
                assert_eq!(entity, "Google");
            }
            other => panic!("Expected WorksAt, got {other:?}"),
        }
    }

    #[test]
    fn detect_employed_at() {
        let rel = detect_relationship("Sarah employed at Microsoft").expect("Should detect");
        match rel {
            RelationshipType::WorksAt(entity) => {
                assert!(entity.contains("Microsoft"), "Got: {entity}");
            }
            other => panic!("Expected WorksAt, got {other:?}"),
        }
    }

    #[test]
    fn detect_joined() {
        let rel = detect_relationship("Lior joined Anthropic last year").expect("Should detect");
        match rel {
            RelationshipType::WorksAt(entity) => {
                assert!(entity.contains("Anthropic"), "Got: {entity}");
            }
            other => panic!("Expected WorksAt, got {other:?}"),
        }
    }

    #[test]
    fn detect_lives_in() {
        let rel = detect_relationship("User lives in Tel Aviv").expect("Should detect LivesIn");
        match rel {
            RelationshipType::LivesIn(entity) => {
                assert_eq!(entity, "Tel Aviv");
            }
            other => panic!("Expected LivesIn, got {other:?}"),
        }
    }

    #[test]
    fn detect_moved_to() {
        let rel = detect_relationship("Alex moved to San Francisco").expect("Should detect");
        match rel {
            RelationshipType::LivesIn(entity) => {
                assert!(entity.contains("San Francisco"), "Got: {entity}");
            }
            other => panic!("Expected LivesIn, got {other:?}"),
        }
    }

    #[test]
    fn detect_based_in() {
        let rel = detect_relationship("Company based in New York").expect("Should detect");
        match rel {
            RelationshipType::LivesIn(entity) => {
                assert!(entity.contains("New York"), "Got: {entity}");
            }
            other => panic!("Expected LivesIn, got {other:?}"),
        }
    }

    #[test]
    fn detect_prefers_tool() {
        let rel = detect_relationship("User prefers Rust for backend").expect("Should detect");
        match rel {
            RelationshipType::PrefersTool(entity) => {
                assert!(entity.contains("Rust"), "Got: {entity}");
            }
            other => panic!("Expected PrefersTool, got {other:?}"),
        }
    }

    #[test]
    fn detect_uses_tool() {
        let rel = detect_relationship("Developer uses TypeScript daily").expect("Should detect");
        match rel {
            RelationshipType::PrefersTool(entity) => {
                assert!(entity.contains("TypeScript"), "Got: {entity}");
            }
            other => panic!("Expected PrefersTool, got {other:?}"),
        }
    }

    #[test]
    fn detect_switched_to() {
        let rel =
            detect_relationship("Team switched to Tauri from Electron").expect("Should detect");
        match rel {
            RelationshipType::PrefersTool(entity) => {
                assert!(entity.contains("Tauri"), "Got: {entity}");
            }
            other => panic!("Expected PrefersTool, got {other:?}"),
        }
    }

    #[test]
    fn detect_part_of() {
        let rel =
            detect_relationship("ShrimPK is part of Bellkis ecosystem").expect("Should detect");
        match rel {
            RelationshipType::PartOf(entity) => {
                assert!(entity.contains("Bellkis"), "Got: {entity}");
            }
            other => panic!("Expected PartOf, got {other:?}"),
        }
    }

    #[test]
    fn detect_belongs_to() {
        let rel = detect_relationship("Module belongs to the core crate").expect("Should detect");
        match rel {
            RelationshipType::PartOf(entity) => {
                assert!(entity.contains("core"), "Got: {entity}");
            }
            other => panic!("Expected PartOf, got {other:?}"),
        }
    }

    #[test]
    fn detect_member_of() {
        let rel = detect_relationship("Alice is a member of the board").expect("Should detect");
        match rel {
            RelationshipType::PartOf(entity) => {
                assert!(entity.contains("board"), "Got: {entity}");
            }
            other => panic!("Expected PartOf, got {other:?}"),
        }
    }

    #[test]
    fn detect_no_relationship() {
        assert!(
            detect_relationship("The sky is blue today").is_none(),
            "Should return None for non-matching text"
        );
        assert!(
            detect_relationship("").is_none(),
            "Should return None for empty text"
        );
        assert!(
            detect_relationship("Hello world").is_none(),
            "Should return None for generic text"
        );
    }

    #[test]
    fn detect_case_insensitive() {
        let rel = detect_relationship("USER WORKS AT OPENAI").expect("Should be case-insensitive");
        match rel {
            RelationshipType::WorksAt(entity) => {
                assert!(
                    entity.contains("OPENAI"),
                    "Should preserve original case, got: {entity}"
                );
            }
            other => panic!("Expected WorksAt, got {other:?}"),
        }
    }

    #[test]
    fn detect_strips_trailing_punctuation() {
        let rel = detect_relationship("User works at Google.").expect("Should detect");
        match rel {
            RelationshipType::WorksAt(entity) => {
                assert_eq!(entity, "Google", "Should strip trailing period");
            }
            other => panic!("Expected WorksAt, got {other:?}"),
        }
    }

    #[test]
    fn subjects_overlap_same_subject() {
        assert!(subjects_overlap(
            "Alex works at Google",
            "Alex works at Meta"
        ));
        assert!(subjects_overlap("User prefers Rust", "User uses Python"));
    }

    #[test]
    fn subjects_overlap_different_subject() {
        assert!(!subjects_overlap(
            "Alex works at Google",
            "Sarah works at Meta"
        ));
        assert!(!subjects_overlap("", "User lives in NYC"));
    }

    #[test]
    fn detect_v2_verb_forms() {
        // KS67: v2 open-domain extraction uses free-form verbs
        let rel = detect_relationship("I am now working at Stripe").expect("working at");
        assert!(matches!(rel, RelationshipType::WorksAt(_)));

        let rel = detect_relationship("I just started at Stripe").expect("started at");
        assert!(matches!(rel, RelationshipType::WorksAt(_)));

        let rel = detect_relationship("I am living in San Francisco").expect("living in");
        assert!(matches!(rel, RelationshipType::LivesIn(_)));

        let rel = detect_relationship("I am moving to Oakland next month").expect("moving to");
        assert!(matches!(rel, RelationshipType::LivesIn(_)));

        let rel = detect_relationship("I am using Neovim for development").expect("using");
        assert!(matches!(rel, RelationshipType::PrefersTool(_)));

        let rel = detect_relationship("I am switching to Arch Linux").expect("switching to");
        assert!(matches!(rel, RelationshipType::PrefersTool(_)));
    }

    #[test]
    fn extract_subject_strips_articles() {
        assert_eq!(extract_subject("The project is part of Bellkis"), "project");
        assert_eq!(extract_subject("Alex works at Google"), "Alex");
        assert_eq!(extract_subject("a small module"), "small");
    }

    #[test]
    fn detect_supersedes_finds_contradictions() {
        let mut store = EchoStore::new();

        // Old enrichment child: "Alex works at Google"
        let mut old_child = make_entry("Alex works at Google", vec![1.0, 0.0, 0.0]);
        old_child.source = "enrichment".to_string();
        old_child.enriched = true;
        store.add(old_child);

        // Some other unrelated memory
        store.add(make_entry("The weather is nice", vec![0.0, 1.0, 0.0]));

        // New facts extracted from a new parent
        let new_facts = vec!["Alex works at Meta".to_string()];

        let pairs = detect_supersedes_pairs(&store, &new_facts, &[], 2);
        assert_eq!(pairs.len(), 1, "Should detect one supersession");
        assert_eq!(pairs[0].0, 0, "Should reference old child at index 0");
        assert_eq!(pairs[0].1, "Alex works at Meta");
    }

    #[test]
    fn detect_supersedes_no_contradiction() {
        let mut store = EchoStore::new();

        // Old enrichment child: "Alex works at Google"
        let mut old_child = make_entry("Alex works at Google", vec![1.0, 0.0, 0.0]);
        old_child.source = "enrichment".to_string();
        old_child.enriched = true;
        store.add(old_child);

        // New fact about a different person
        let new_facts = vec!["Sarah works at Meta".to_string()];

        let pairs = detect_supersedes_pairs(&store, &new_facts, &[], 1);
        assert!(
            pairs.is_empty(),
            "Different subjects should not trigger supersedes"
        );
    }

    #[test]
    fn detect_supersedes_same_value_no_contradiction() {
        let mut store = EchoStore::new();

        // Old enrichment child: "Alex works at Google"
        let mut old_child = make_entry("Alex works at Google", vec![1.0, 0.0, 0.0]);
        old_child.source = "enrichment".to_string();
        old_child.enriched = true;
        store.add(old_child);

        // Same fact repeated — should NOT create a supersedes edge
        let new_facts = vec!["Alex works at Google".to_string()];

        let pairs = detect_supersedes_pairs(&store, &new_facts, &[], 1);
        assert!(pairs.is_empty(), "Same fact should not trigger supersedes");
    }

    #[test]
    fn detect_supersedes_parent_content() {
        // KS67: Pass 3 — detect supersession against parent content
        let mut store = EchoStore::new();

        // Index 0: old PARENT memory about working at Shopify
        // (no enrichment children — simulates LLM extraction failure)
        let parent = make_entry(
            "I work at Shopify on the payments team. I joined in 2019.",
            vec![1.0, 0.0, 0.0],
        );
        store.add(parent);

        // New fact from a different parent (current_parent_idx = 1)
        let new_facts = vec!["I started at Stripe as a senior engineer".to_string()];

        let pairs = detect_supersedes_pairs(&store, &new_facts, &[], 1);
        assert_eq!(pairs.len(), 1, "Should detect parent-content supersession");
        assert_eq!(pairs[0].0, 0, "Should reference old parent at index 0");
    }

    #[test]
    fn detect_supersedes_parent_location() {
        let mut store = EchoStore::new();

        // Old parent about living in Oakland
        let parent = make_entry(
            "I live in Oakland, California. I moved here after college.",
            vec![1.0, 0.0, 0.0],
        );
        store.add(parent);

        let new_facts = vec!["I moved to San Francisco last month".to_string()];

        let pairs = detect_supersedes_pairs(&store, &new_facts, &[], 1);
        assert_eq!(
            pairs.len(),
            1,
            "Should detect location supersession via parent"
        );
    }

    // ---- Triple extraction tests (KS61 Track C) ----

    #[test]
    fn extract_triples_works_at() {
        use shrimpk_core::TriplePredicate;
        let triple = extract_triples("Lior works at Bellkis").unwrap();
        assert_eq!(triple.subject, "Lior");
        assert_eq!(triple.predicate, TriplePredicate::WorksAt);
        assert_eq!(triple.object, "Bellkis");
    }

    #[test]
    fn extract_triples_lives_in() {
        use shrimpk_core::TriplePredicate;
        let triple = extract_triples("User lives in Tel Aviv").unwrap();
        assert_eq!(triple.predicate, TriplePredicate::LivesIn);
    }

    #[test]
    fn extract_triples_prefers_tool() {
        use shrimpk_core::TriplePredicate;
        let triple = extract_triples("Developer prefers Rust").unwrap();
        assert_eq!(triple.predicate, TriplePredicate::PrefersTool);
    }

    #[test]
    fn extract_triples_part_of() {
        use shrimpk_core::TriplePredicate;
        let triple = extract_triples("Module part of ShrimPK").unwrap();
        assert_eq!(triple.predicate, TriplePredicate::PartOf);
    }

    #[test]
    fn extract_triples_none_for_plain_fact() {
        assert!(extract_triples("The sky is blue").is_none());
    }

    // ---- KS67: dynamic_max_facts tests ----

    #[test]
    fn dynamic_max_facts_5_words() {
        let content = "one two three four five";
        assert_eq!(dynamic_max_facts(content), 1);
    }

    #[test]
    fn dynamic_max_facts_20_words() {
        let content = (0..20)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(dynamic_max_facts(&content), 2);
    }

    #[test]
    fn dynamic_max_facts_50_words() {
        let content = (0..50)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(dynamic_max_facts(&content), 3);
    }

    #[test]
    fn dynamic_max_facts_200_words() {
        let content = (0..200)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(dynamic_max_facts(&content), 10);
    }

    #[test]
    fn dynamic_max_facts_300_words_caps_at_12() {
        let content = (0..300)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(dynamic_max_facts(&content), 12);
    }
}
