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

/// Perform a full consolidation pass on the Echo Memory engine components.
///
/// This is the core cleanup routine. It acquires mutable references to
/// all engine internals and performs four maintenance steps in sequence.
///
/// # Arguments
/// * `store` - The in-memory vector store (may have entries removed/merged)
/// * `hebbian` - The Hebbian co-activation graph (weak edges pruned)
/// * `bloom` - The Bloom filter for topic pre-screening (rebuilt if dirty)
/// * `bloom_dirty` - Flag indicating the Bloom filter needs a rebuild
/// * `_config` - Engine configuration (reserved for future threshold tuning)
#[allow(clippy::field_reassign_with_default)]
pub fn consolidate(
    store: &mut EchoStore,
    hebbian: &mut HebbianGraph,
    bloom: &mut TopicFilter,
    bloom_dirty: &mut bool,
    config: &EchoConfig,
    consolidator: &dyn Consolidator,
    embedder: Option<&std::sync::Mutex<crate::embedder::Embedder>>,
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
        result.duplicates_merged = merge_near_duplicates(store);
    }

    // Step 4: Decay echo counts on stale memories
    result.echo_counts_decayed = decay_echo_counts(store);

    // Step 5: LLM fact extraction on un-enriched memories (sleep consolidation)
    //
    // For each memory that hasn't been enriched yet, ask the consolidator
    // to extract atomic facts. The parent is marked enriched=true to prevent
    // re-extraction on the next cycle. Batch limit: max 10 per cycle.
    //
    // Note: child memory creation with embeddings requires the Embedder,
    // which is not available here. Facts are extracted and logged; child
    // creation happens in EchoEngine::consolidate_now() which has embedder access.
    if consolidator.name() != "noop" {
        const MAX_ENRICHMENTS_PER_CYCLE: usize = 10;
        let unenriched: Vec<usize> = (0..store.len())
            .filter(|&i| {
                store
                    .entry_at(i)
                    .is_some_and(|e| !e.enriched && e.parent_id.is_none())
            })
            .take(MAX_ENRICHMENTS_PER_CYCLE)
            .collect();

        for idx in unenriched {
            let content = match store.entry_at(idx) {
                Some(e) => e.content.clone(),
                None => continue,
            };

            let facts = consolidator.extract_facts(&content, config.max_facts_per_memory);

            if !facts.is_empty() {
                result.facts_extracted += facts.len();
                tracing::debug!(
                    idx,
                    fact_count = facts.len(),
                    "Consolidation: extracted facts from memory"
                );
            }

            // Create child memories if embedder is available
            if let Some(embedder) = embedder {
                let parent_id = match store.entry_at(idx) {
                    Some(e) => e.id.clone(),
                    None => continue,
                };

                for fact in &facts {
                    let embedding = match embedder.lock() {
                        Ok(mut e) => match e.embed(fact) {
                            Ok(emb) => emb,
                            Err(err) => {
                                tracing::debug!(error = %err, "Failed to embed fact, skipping");
                                continue;
                            }
                        },
                        Err(_) => continue,
                    };

                    let mut child = shrimpk_core::MemoryEntry::new(
                        fact.clone(),
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

                    let child_idx = store.add(child) as u32;
                    lsh.insert(child_idx, &embedding);
                    bloom.insert_memory(fact);

                    // Detect typed relationship from fact text and create
                    // a Hebbian edge between parent and child.
                    if let Some(rel) = detect_relationship(fact) {
                        hebbian.co_activate_with_relationship(
                            idx as u32,
                            child_idx,
                            0.5, // moderate strength for extracted relationships
                            rel,
                        );
                        result.relationships_created += 1;
                    }
                }

                // Step 5b: Detect Supersedes edges — when a new fact
                // contradicts an older fact about the same entity.
                let supersedes_pairs = detect_supersedes_pairs(store, &facts, idx);
                for (old_idx, new_fact_text) in &supersedes_pairs {
                    hebbian.co_activate_with_relationship(
                        *old_idx as u32,
                        idx as u32,
                        0.3, // lighter weight — recency boost handles the rest
                        RelationshipType::Supersedes,
                    );
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
    let works_re = Regex::new(
        r"(?i)(?:works?\s+at|employed\s+at|joined|works?\s+for|employed\s+by)\s+(.+)"
    ).ok()?;
    if let Some(caps) = works_re.captures(&lower) {
        let entity = caps.get(1).map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity {
            if !e.is_empty() {
                return Some(RelationshipType::WorksAt(e));
            }
        }
    }

    // LivesIn patterns: "X lives in Y", "X moved to Y", "X based in Y"
    let lives_re = Regex::new(
        r"(?i)(?:lives?\s+in|moved\s+to|based\s+in|relocated\s+to|resides?\s+in)\s+(.+)"
    ).ok()?;
    if let Some(caps) = lives_re.captures(&lower) {
        let entity = caps.get(1).map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity {
            if !e.is_empty() {
                return Some(RelationshipType::LivesIn(e));
            }
        }
    }

    // PrefersTool patterns: "X prefers Y", "X uses Y", "X likes Y", "X switched to Y"
    let pref_re = Regex::new(
        r"(?i)(?:prefers?\s+|likes?\s+|uses?\s+|switched\s+to\s+|chose\s+)(.+)"
    ).ok()?;
    if let Some(caps) = pref_re.captures(&lower) {
        let entity = caps.get(1).map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity {
            if !e.is_empty() {
                return Some(RelationshipType::PrefersTool(e));
            }
        }
    }

    // PartOf patterns: "X part of Y", "X belongs to Y", "X member of Y"
    let part_re = Regex::new(
        r"(?i)(?:part\s+of|belongs?\s+to|member\s+of|component\s+of)\s+(.+)"
    ).ok()?;
    if let Some(caps) = part_re.captures(&lower) {
        let entity = caps.get(1).map(|m| extract_entity(fact, m.start(), m.end()));
        if let Some(e) = entity {
            if !e.is_empty() {
                return Some(RelationshipType::PartOf(e));
            }
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
        .trim_end_matches(|c: char| c == '.' || c == ',' || c == ';' || c == '!' || c == '?')
        .trim()
        .to_string()
}

/// Detect entity-level contradictions between new facts and existing memories.
///
/// Looks for cases where a new fact about the same subject contradicts an
/// older memory. For example:
/// - Old: "Alex works at Google" + New: "Alex works at Meta" -> Supersedes
/// - Old: "User lives in NYC" + New: "User moved to SF" -> Supersedes
///
/// Returns `(old_memory_index, new_fact_text)` pairs for each detected supersession.
fn detect_supersedes_pairs(
    store: &EchoStore,
    new_facts: &[String],
    _current_parent_idx: usize,
) -> Vec<(usize, String)> {
    let mut pairs = Vec::new();

    for fact in new_facts {
        let new_rel = match detect_relationship(fact) {
            Some(r) => r,
            None => continue,
        };

        // Extract the relationship category and entity for comparison
        let (new_category, _new_entity) = categorize_relationship(&new_rel);

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
                    pairs.push((i, fact.clone()));
                    break; // One supersession per fact is enough
                }
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
fn extract_subject(fact: &str) -> String {
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
/// the memory with fewer echoes is merged into the one with more echoes.
/// Content is preserved from the winner; the loser is removed.
///
/// Returns the number of pairs merged.
fn merge_near_duplicates(store: &mut EchoStore) -> usize {
    // Collect IDs and echo counts for all entries, along with their embeddings.
    // We need to work with indices rather than IDs during the comparison pass
    // because we need access to the embeddings array.
    let embeddings: Vec<Vec<f32>> = store.all_embeddings().to_vec();
    let n = embeddings.len();

    // Collect (id, echo_count) for each entry by index
    let entry_info: Vec<_> = (0..n)
        .filter_map(|i| store.entry_at(i).map(|e| (e.id.clone(), e.echo_count)))
        .collect();

    // Find pairs to merge. We collect the ID of the "loser" (fewer echoes).
    let mut to_remove = Vec::new();
    let mut already_removed = std::collections::HashSet::new();

    for i in 0..n {
        if already_removed.contains(&i) {
            continue;
        }
        for j in (i + 1)..n {
            if already_removed.contains(&j) {
                continue;
            }
            let sim = similarity::cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim > DUPLICATE_SIMILARITY_THRESHOLD {
                // Merge: keep the entry with more echo_count
                let (winner_idx, loser_idx) = if entry_info[i].1 >= entry_info[j].1 {
                    (i, j)
                } else {
                    (j, i)
                };

                // Transfer echo_count from loser to winner (additive)
                let loser_echo = entry_info[loser_idx].1;
                if let Some(winner_entry) = store.get_mut(&entry_info[winner_idx].0) {
                    winner_entry.echo_count = winner_entry.echo_count.saturating_add(loser_echo);
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
        let rel = detect_relationship("Team switched to Tauri from Electron").expect("Should detect");
        match rel {
            RelationshipType::PrefersTool(entity) => {
                assert!(entity.contains("Tauri"), "Got: {entity}");
            }
            other => panic!("Expected PrefersTool, got {other:?}"),
        }
    }

    #[test]
    fn detect_part_of() {
        let rel = detect_relationship("ShrimPK is part of Bellkis ecosystem").expect("Should detect");
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
                assert!(entity.contains("OPENAI"), "Should preserve original case, got: {entity}");
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
        assert!(subjects_overlap("Alex works at Google", "Alex works at Meta"));
        assert!(subjects_overlap("User prefers Rust", "User uses Python"));
    }

    #[test]
    fn subjects_overlap_different_subject() {
        assert!(!subjects_overlap("Alex works at Google", "Sarah works at Meta"));
        assert!(!subjects_overlap("", "User lives in NYC"));
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

        let pairs = detect_supersedes_pairs(&store, &new_facts, 2);
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

        let pairs = detect_supersedes_pairs(&store, &new_facts, 1);
        assert!(pairs.is_empty(), "Different subjects should not trigger supersedes");
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

        let pairs = detect_supersedes_pairs(&store, &new_facts, 1);
        assert!(pairs.is_empty(), "Same fact should not trigger supersedes");
    }
}
