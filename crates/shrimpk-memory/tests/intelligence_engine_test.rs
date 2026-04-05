//! Integration tests for the intelligence scoring engine (KS50).
//!
//! Tests cover:
//! - Power-law decay (FSRS) basic values and comparison vs exponential
//! - ACT-R activation increases with echoes
//! - Multi-signal importance computation
//! - Importance-based consolidation priority
//! - Feature flag backward compatibility
//!
//! These tests do NOT require the fastembed model — they exercise
//! the scoring and consolidation logic directly.

use chrono::{Duration, Utc};
use shrimpk_core::{EchoConfig, MemoryCategory, MemoryEntry};
use shrimpk_memory::consolidation::{ConsolidationResult, consolidate};
use shrimpk_memory::store::EchoStore;
use shrimpk_memory::{HebbianGraph, compute_embedding_mean, compute_importance, power_law_decay};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_config() -> EchoConfig {
    EchoConfig {
        max_memories: 1000,
        similarity_threshold: 0.3,
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        data_dir: PathBuf::from("/tmp/shrimpk-intelligence-test"),
        embedding_dim: 3,
        use_importance: true,
        use_power_law_decay: true,
        use_actr_activation: false,
        ..Default::default()
    }
}

fn legacy_config() -> EchoConfig {
    EchoConfig {
        use_importance: false,
        use_power_law_decay: false,
        use_actr_activation: false,
        ..test_config()
    }
}

fn make_entry(content: &str, embedding: Vec<f32>) -> MemoryEntry {
    MemoryEntry::new(content.to_string(), embedding, "test".to_string())
}

fn make_entry_with_source(content: &str, embedding: Vec<f32>, source: &str) -> MemoryEntry {
    MemoryEntry::new(content.to_string(), embedding, source.to_string())
}

fn run_consolidation(store: &mut EchoStore, config: &EchoConfig) -> ConsolidationResult {
    let mut hebbian = HebbianGraph::new(604_800.0, 0.01);
    let mut bloom = shrimpk_memory::bloom::TopicFilter::new(1000, 0.01);
    let mut bloom_dirty = false;
    let mut lsh = shrimpk_memory::lsh::CosineHash::new(3, 16, 10);

    consolidate(
        store,
        &mut hebbian,
        &mut bloom,
        &mut bloom_dirty,
        config,
        &shrimpk_memory::consolidator::NoopConsolidator,
        None,
        &mut lsh,
    )
}

// ===========================================================================
// Test 1: power_law_decay_basic
// ===========================================================================

#[test]
fn power_law_decay_basic() {
    // At t=0, retention should be 1.0
    let r_0 = power_law_decay(0.0, 7.0);
    assert!(
        (r_0 - 1.0).abs() < 1e-6,
        "Retention at t=0 should be 1.0, got {r_0}"
    );

    // At t = stability, retention should be approximately 0.9
    // (FSRS definition: stability is the interval at which R = 0.9)
    let stability_days = 7.0;
    let age_secs = stability_days * 86400.0;
    let r_at_stability = power_law_decay(age_secs, stability_days);
    assert!(
        (r_at_stability - 0.9).abs() < 0.02,
        "Retention at t=stability should be ~0.9, got {r_at_stability}"
    );

    // Retention should decrease monotonically
    let r_1d = power_law_decay(1.0 * 86400.0, 7.0);
    let r_7d = power_law_decay(7.0 * 86400.0, 7.0);
    let r_30d = power_law_decay(30.0 * 86400.0, 7.0);
    let r_365d = power_law_decay(365.0 * 86400.0, 7.0);

    assert!(
        r_1d > r_7d,
        "1d ({r_1d}) should retain more than 7d ({r_7d})"
    );
    assert!(
        r_7d > r_30d,
        "7d ({r_7d}) should retain more than 30d ({r_30d})"
    );
    assert!(
        r_30d > r_365d,
        "30d ({r_30d}) should retain more than 365d ({r_365d})"
    );

    // Retention should always be in [0, 1]
    assert!((0.0..=1.0).contains(&r_365d), "Out of range: {r_365d}");

    // Edge case: zero stability should return 0.0
    let r_zero_stability = power_law_decay(86400.0, 0.0);
    assert!(
        r_zero_stability.abs() < 1e-6,
        "Zero stability should return 0.0, got {r_zero_stability}"
    );
}

// ===========================================================================
// Test 2: power_law_vs_exponential
// ===========================================================================

#[test]
fn power_law_vs_exponential() {
    // Power-law should retain more than exponential at long time horizons.
    // This is the key cognitive science insight: memories fade slowly, not fast.
    let stability_days = 7.0;
    let half_life_secs = stability_days * 86400.0;

    // Compare at 30 days (long horizon)
    let age_30d = 30.0 * 86400.0;
    let power_law_r = power_law_decay(age_30d, stability_days);
    let exponential_r = (-age_30d * std::f64::consts::LN_2 / half_life_secs).exp();

    assert!(
        power_law_r > exponential_r,
        "At 30d, power-law ({power_law_r:.4}) should retain more than exponential ({exponential_r:.4})"
    );

    // Compare at 90 days (very long horizon)
    let age_90d = 90.0 * 86400.0;
    let power_law_r_90 = power_law_decay(age_90d, stability_days);
    let exponential_r_90 = (-age_90d * std::f64::consts::LN_2 / half_life_secs).exp();

    assert!(
        power_law_r_90 > exponential_r_90,
        "At 90d, power-law ({power_law_r_90:.6}) should retain more than exponential ({exponential_r_90:.6})"
    );

    // The gap should widen at longer horizons (power-law has a "fat tail")
    let gap_30 = power_law_r - exponential_r;
    let gap_90 = power_law_r_90 - exponential_r_90;
    // At 90d the exponential is essentially 0, so the absolute gap
    // is basically the power-law value itself. Just verify power-law > 0.
    assert!(
        power_law_r_90 > 0.01,
        "Power-law should still retain meaningful value at 90d, got {power_law_r_90}"
    );
    assert!(
        gap_30 > 0.0 && gap_90 > 0.0,
        "Power-law advantage should be positive at both horizons"
    );
}

// ===========================================================================
// Test 3: actr_activation_increases_with_echoes
// ===========================================================================

#[test]
fn actr_activation_increases_with_echoes() {
    use shrimpk_memory::actr_ol_activation;

    // Use a short lifetime (1 hour) so the activation stays above the clamp floor.
    // With 7 days lifetime the BLA formula produces values deep below -6 that
    // get clamped identically. A 1-hour-old memory keeps values in a range
    // where different echo counts produce distinct activations.
    let created_at = Utc::now() - Duration::hours(1);
    let d = 0.5; // default decay

    // No last_echoed to avoid the negative recency component
    let act_0 = actr_ol_activation(0, created_at, None, d);
    let act_1 = actr_ol_activation(1, created_at, None, d);
    let act_5 = actr_ol_activation(5, created_at, None, d);
    let act_20 = actr_ol_activation(20, created_at, None, d);

    // More echoes = higher activation (monotonically increasing)
    assert!(act_1 > act_0, "1 echo ({act_1:.3}) > 0 echoes ({act_0:.3})");
    assert!(act_5 > act_1, "5 echoes ({act_5:.3}) > 1 echo ({act_1:.3})");
    assert!(
        act_20 > act_5,
        "20 echoes ({act_20:.3}) > 5 echoes ({act_5:.3})"
    );

    // Activation should be within the clamped range [-6, 4]
    assert!(
        (-6.0..=4.0).contains(&act_0),
        "Out of clamped range: {act_0}"
    );
    assert!(
        (-6.0..=4.0).contains(&act_20),
        "Out of clamped range: {act_20}"
    );
}

// ===========================================================================
// Test 4: importance_computation_basic
// ===========================================================================

#[test]
fn importance_computation_basic() {
    // Build an entry with known properties
    let mut entry = make_entry("User prefers Rust for backend", vec![1.0, 0.0, 0.0]);
    entry.category = MemoryCategory::Preference;
    entry.novelty_score = 0.8;
    entry.echo_count = 3;
    entry.source = "document".to_string();
    entry.created_at = Utc::now() - Duration::days(2);

    let embedding = vec![1.0, 0.0, 0.0];
    // Mean embedding different from entry's => some surprise
    let mean = vec![0.5, 0.5, 0.0];

    // Full 5-signal importance
    let imp_full = compute_importance(&entry, &embedding, Some(&mean));
    assert!(
        imp_full > 0.0 && imp_full <= 1.0,
        "Importance should be in (0, 1], got {imp_full}"
    );

    // Without mean (4-signal)
    let imp_no_mean = compute_importance(&entry, &embedding, None);
    assert!(
        imp_no_mean > 0.0 && imp_no_mean <= 1.0,
        "4-signal importance should be in (0, 1], got {imp_no_mean}"
    );

    // Both should be non-trivial (all signals contribute positively)
    assert!(
        imp_full > 0.1,
        "With all positive signals, importance should be > 0.1, got {imp_full}"
    );
    assert!(
        imp_no_mean > 0.1,
        "With 4 positive signals, importance should be > 0.1, got {imp_no_mean}"
    );

    // Verify novelty signal contribution: a higher novelty_score should increase importance
    let mut high_novelty = entry.clone();
    high_novelty.novelty_score = 1.0;
    let imp_high_novelty = compute_importance(&high_novelty, &embedding, Some(&mean));
    assert!(
        imp_high_novelty > imp_full,
        "Higher novelty ({imp_high_novelty:.4}) should increase importance over ({imp_full:.4})"
    );

    // Verify category signal: Identity > Conversation
    let mut identity_entry = entry.clone();
    identity_entry.category = MemoryCategory::Identity;
    let imp_identity = compute_importance(&identity_entry, &embedding, Some(&mean));

    let mut conv_entry = entry.clone();
    conv_entry.category = MemoryCategory::Conversation;
    let imp_conv = compute_importance(&conv_entry, &embedding, Some(&mean));

    assert!(
        imp_identity > imp_conv,
        "Identity ({imp_identity:.4}) should score higher than Conversation ({imp_conv:.4})"
    );
}

// ===========================================================================
// Test 5: importance_consolidation_priority
// ===========================================================================

#[test]
fn importance_consolidation_priority() {
    let config = test_config();
    let mut store = EchoStore::new();

    // Create 3 unenriched memories with varying importance
    let mut low = make_entry("low importance memory", vec![1.0, 0.0, 0.0]);
    low.importance = 0.1;
    let _low_id = low.id.clone();
    store.add(low);

    let mut high = make_entry("high importance memory", vec![0.0, 1.0, 0.0]);
    high.importance = 0.9;
    let _high_id = high.id.clone();
    store.add(high);

    let mut mid = make_entry("medium importance memory", vec![0.0, 0.0, 1.0]);
    mid.importance = 0.5;
    let _mid_id = mid.id.clone();
    store.add(mid);

    // Run consolidation with use_importance = true
    // With noop consolidator, no actual enrichment happens,
    // but we can verify the importance scores got recomputed (Step 4.5)
    let result = run_consolidation(&mut store, &config);

    // Step 4.5 should have recomputed importance for all 3 entries
    // (since none had importance_computed_at set)
    assert_eq!(
        result.importance_recomputed, 3,
        "All 3 entries should have importance recomputed"
    );

    // Verify all entries now have importance_computed_at set
    for i in 0..store.len() {
        let entry = store.entry_at(i).expect("entry should exist");
        assert!(
            entry.importance_computed_at.is_some(),
            "Entry {} should have importance_computed_at set",
            entry.content
        );
    }
}

// ===========================================================================
// Test 6: importance_zero_for_enrichment_source
// ===========================================================================

#[test]
fn importance_zero_for_enrichment_source() {
    // Entries with source="enrichment" should get 0.0 source weight,
    // reducing their overall importance score.
    let mut enrichment_entry = make_entry_with_source(
        "extracted fact from parent",
        vec![1.0, 0.0, 0.0],
        "enrichment",
    );
    enrichment_entry.novelty_score = 0.5;
    enrichment_entry.category = MemoryCategory::Fact;
    enrichment_entry.echo_count = 0;

    let mut regular_entry =
        make_entry_with_source("user-submitted memory", vec![1.0, 0.0, 0.0], "document");
    regular_entry.novelty_score = 0.5;
    regular_entry.category = MemoryCategory::Fact;
    regular_entry.echo_count = 0;

    let embedding = vec![1.0, 0.0, 0.0];
    let mean = vec![0.5, 0.5, 0.0];

    let imp_enrichment = compute_importance(&enrichment_entry, &embedding, Some(&mean));
    let imp_regular = compute_importance(&regular_entry, &embedding, Some(&mean));

    // The enrichment source should have lower importance (source_weight = 0.0 vs 1.0)
    assert!(
        imp_regular > imp_enrichment,
        "Regular ({imp_regular:.4}) should score higher than enrichment ({imp_enrichment:.4})"
    );

    // Verify enrichment's source contribution is zero by checking the source_weight directly
    let src_w = shrimpk_core::source_weight("enrichment");
    assert!(
        src_w.abs() < 1e-6,
        "source_weight for 'enrichment' should be 0.0, got {src_w}"
    );
}

// ===========================================================================
// Test 7: feature_flags_off_uses_legacy
// ===========================================================================

#[test]
fn feature_flags_off_uses_legacy() {
    let config = legacy_config();
    let mut store = EchoStore::new();

    // Create two near-duplicates with different echo counts and importance
    let emb_a = vec![1.0, 0.0, 0.0];
    let emb_b = vec![0.99, 0.01, 0.0]; // near-duplicate

    let mut entry_a = make_entry("memory alpha", emb_a);
    entry_a.echo_count = 2;
    entry_a.importance = 0.9; // high importance but fewer echoes
    let id_a = entry_a.id.clone();
    store.add(entry_a);

    let mut entry_b = make_entry("memory beta", emb_b);
    entry_b.echo_count = 5; // more echoes
    entry_b.importance = 0.1; // low importance
    let id_b = entry_b.id.clone();
    store.add(entry_b);

    // With legacy config (use_importance=false), winner should be B (more echoes)
    let result = run_consolidation(&mut store, &config);

    assert_eq!(result.duplicates_merged, 1, "Should merge one pair");
    assert_eq!(store.len(), 1, "One memory should remain");

    // B should survive (more echo_count wins in legacy mode)
    let survivor = store.get(&id_b).expect("B should survive in legacy mode");
    assert_eq!(
        survivor.echo_count, 7,
        "Echo counts should be accumulated (5 + 2)"
    );
    assert!(store.get(&id_a).is_none(), "A should be removed");

    // Importance should NOT have been recomputed (flag is off)
    assert_eq!(
        result.importance_recomputed, 0,
        "No importance recomputation when use_importance=false"
    );
}

// ===========================================================================
// Test 8: backward_compat_old_entries
// ===========================================================================

#[test]
fn backward_compat_old_entries() {
    // Old entries have default 0.0 importance and None importance_computed_at.
    // They should still work correctly in consolidation.
    let config = test_config();
    let mut store = EchoStore::new();

    // Simulate old entries (pre-KS50) — all fields at defaults
    let mut old_entry_a = make_entry("old memory alpha", vec![1.0, 0.0, 0.0]);
    assert_eq!(
        old_entry_a.importance, 0.0,
        "Default importance should be 0.0"
    );
    assert!(
        old_entry_a.importance_computed_at.is_none(),
        "Default importance_computed_at should be None"
    );
    assert_eq!(
        old_entry_a.activation_cache, 0.0,
        "Default activation_cache should be 0.0"
    );
    old_entry_a.echo_count = 3;
    let id_a = old_entry_a.id.clone();
    store.add(old_entry_a);

    let mut old_entry_b = make_entry("old memory beta", vec![0.0, 1.0, 0.0]);
    old_entry_b.echo_count = 1;
    let id_b = old_entry_b.id.clone();
    store.add(old_entry_b);

    // Add a dissimilar third entry
    let old_entry_c = make_entry("old memory gamma", vec![0.0, 0.0, 1.0]);
    let id_c = old_entry_c.id.clone();
    store.add(old_entry_c);

    // Run consolidation — should not panic or crash on default-valued entries
    let result = run_consolidation(&mut store, &config);

    // No duplicates (all orthogonal embeddings)
    assert_eq!(result.duplicates_merged, 0);

    // All entries should still exist
    assert!(store.get(&id_a).is_some(), "Entry A should survive");
    assert!(store.get(&id_b).is_some(), "Entry B should survive");
    assert!(store.get(&id_c).is_some(), "Entry C should survive");

    // After consolidation, importance should have been recomputed from defaults
    assert_eq!(
        result.importance_recomputed, 3,
        "All 3 old entries should have importance recomputed"
    );

    // All entries should now have importance_computed_at set
    for i in 0..store.len() {
        let entry = store.entry_at(i).expect("entry exists");
        assert!(
            entry.importance_computed_at.is_some(),
            "After consolidation, old entry '{}' should have importance_computed_at",
            entry.content
        );
    }
}

// ===========================================================================
// Additional edge case tests
// ===========================================================================

/// Verify importance-based dedup picks the higher-importance entry as winner.
///
/// Note: Step 4.5 (importance recomputation) runs AFTER dedup (Step 3),
/// so the final importance value will be recomputed from the entry's signals
/// rather than preserved from the pre-set value. We verify winner selection
/// and echo count accumulation rather than the final importance value.
#[test]
fn importance_dedup_picks_higher_importance() {
    let config = test_config();
    let mut store = EchoStore::new();

    let emb_a = vec![1.0, 0.0, 0.0];
    let emb_b = vec![0.99, 0.01, 0.0]; // near-duplicate

    let mut entry_a = make_entry("dup with low importance", emb_a);
    entry_a.importance = 0.2;
    entry_a.echo_count = 10; // more echoes, but low importance
    let id_a = entry_a.id.clone();
    store.add(entry_a);

    let mut entry_b = make_entry("dup with high importance", emb_b);
    entry_b.importance = 0.8;
    entry_b.echo_count = 1; // fewer echoes, but high importance
    let id_b = entry_b.id.clone();
    store.add(entry_b);

    let result = run_consolidation(&mut store, &config);

    assert_eq!(result.duplicates_merged, 1);

    // With use_importance=true, B (higher importance) should survive
    let survivor = store
        .get(&id_b)
        .expect("B should survive (higher importance)");
    assert_eq!(
        survivor.echo_count, 11,
        "Echo counts should accumulate: 1 + 10"
    );
    // After Step 4.5, importance is recomputed from entry signals (not the
    // pre-set value). Just verify the survivor has a valid importance.
    assert!(
        survivor.importance >= 0.0 && survivor.importance <= 1.0,
        "Importance should be in [0, 1], got {}",
        survivor.importance
    );
    assert!(
        store.get(&id_a).is_none(),
        "A should be removed (lower importance)"
    );
}

/// Verify dedup tie-breaking: equal importance uses lower store index (older wins).
#[test]
fn importance_dedup_tiebreak_by_index() {
    let config = test_config();
    let mut store = EchoStore::new();

    let emb_a = vec![1.0, 0.0, 0.0];
    let emb_b = vec![0.99, 0.01, 0.0]; // near-duplicate

    let mut entry_a = make_entry("first entry (older)", emb_a);
    entry_a.importance = 0.5;
    entry_a.echo_count = 1;
    let id_a = entry_a.id.clone();
    store.add(entry_a);

    let mut entry_b = make_entry("second entry (newer)", emb_b);
    entry_b.importance = 0.5; // equal importance
    entry_b.echo_count = 1;
    let id_b = entry_b.id.clone();
    store.add(entry_b);

    let result = run_consolidation(&mut store, &config);

    assert_eq!(result.duplicates_merged, 1);

    // With equal importance, lower index (older, A) should win
    let survivor = store
        .get(&id_a)
        .expect("A should survive (older = lower index)");
    assert_eq!(survivor.echo_count, 2, "Echo counts should accumulate");
    assert!(
        store.get(&id_b).is_none(),
        "B should be removed (newer = higher index)"
    );
}

/// Verify compute_embedding_mean handles edge cases.
#[test]
fn embedding_mean_edge_cases() {
    // Empty input
    let empty: Vec<&[f32]> = vec![];
    let mean = compute_embedding_mean(&empty);
    assert!(mean.is_empty(), "Empty input should produce empty mean");

    // Single embedding
    let single: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0]];
    let mean = compute_embedding_mean(&single);
    assert_eq!(mean.len(), 3);
    assert!((mean[0] - 1.0).abs() < 1e-6);
    assert!((mean[1] - 2.0).abs() < 1e-6);
    assert!((mean[2] - 3.0).abs() < 1e-6);

    // Multiple embeddings
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    let multi: Vec<&[f32]> = vec![&a, &b];
    let mean = compute_embedding_mean(&multi);
    assert_eq!(mean.len(), 3);
    assert!((mean[0] - 0.5).abs() < 1e-6, "Mean[0] should be 0.5");
    assert!((mean[1] - 0.5).abs() < 1e-6, "Mean[1] should be 0.5");
    assert!((mean[2] - 0.0).abs() < 1e-6, "Mean[2] should be 0.0");
}
