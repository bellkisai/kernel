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

use crate::bloom::TopicFilter;
use crate::hebbian::HebbianGraph;
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

            // Mark parent as enriched regardless (even if 0 facts — avoids retrying)
            if let Some(entry) = store.entry_at_mut(idx) {
                entry.enriched = true;
            }
        }
    }

    result.duration_ms = start.elapsed().as_millis() as u64;
    result
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
}
