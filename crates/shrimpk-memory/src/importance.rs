//! Importance scoring — multi-signal importance for memory prioritization.
//!
//! Signals (cognitive science grounding):
//! - Novelty (Von Restorff distinctiveness)
//! - Category weight (Levels of processing)
//! - Echo velocity (Testing effect)
//! - Surprise (SWR prediction-error replay)
//! - Source weight (Encoding depth)

use chrono::{DateTime, Utc};
use shrimpk_core::{MemoryEntry, source_weight};

/// Compute the importance score for a memory entry.
///
/// Returns a value in [0.0, 1.0].
///
/// `embedding_mean` is the rolling mean of recent embeddings (for surprise).
/// Pass `None` to skip surprise (score uses 4 signals instead of 5).
pub fn compute_importance(
    entry: &MemoryEntry,
    embedding: &[f32],
    embedding_mean: Option<&[f32]>,
) -> f32 {
    let novelty = entry.novelty_score;
    let cat_weight = entry.category.importance_weight();
    let velocity = echo_velocity(entry.echo_count, entry.created_at);
    let surprise = embedding_mean
        .map(|mean| surprise_score(embedding, mean))
        .unwrap_or(0.0);
    let src_weight = source_weight(&entry.source);

    let score = if embedding_mean.is_some() {
        // Full 5-signal formula
        0.35 * novelty + 0.25 * cat_weight + 0.20 * velocity + 0.15 * surprise + 0.05 * src_weight
    } else {
        // 4-signal (no surprise) — redistribute 0.15 proportionally
        0.41 * novelty + 0.29 * cat_weight + 0.24 * velocity + 0.06 * src_weight
    };

    score.clamp(0.0, 1.0)
}

/// Echo velocity: how frequently this memory is accessed relative to its age.
/// Uses logistic squash to [0, 1).
fn echo_velocity(echo_count: u32, created_at: DateTime<Utc>) -> f32 {
    let days = (Utc::now() - created_at).num_seconds().max(1) as f64 / 86400.0;
    let rate = echo_count as f64 / days;
    // Logistic squash: 2 / (1 + exp(-rate)) - 1, maps [0, inf) -> [0, 1)
    let logistic = 2.0 / (1.0 + (-rate).exp()) - 1.0;
    logistic as f32
}

/// Surprise score: how different this embedding is from the rolling mean.
/// Uses euclidean distance, capped at 1.0.
fn surprise_score(embedding: &[f32], mean: &[f32]) -> f32 {
    if embedding.len() != mean.len() || embedding.is_empty() {
        return 0.0;
    }
    let dist_sq: f32 = embedding
        .iter()
        .zip(mean.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    (dist_sq.sqrt() / 2.0).min(1.0)
}

/// Compute the rolling mean of a set of embeddings.
pub fn compute_embedding_mean(embeddings: &[&[f32]]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let mut mean = vec![0.0f32; dim];
    let n = embeddings.len() as f32;
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            if i < dim {
                mean[i] += v / n;
            }
        }
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_velocity_zero_for_new_memory() {
        let created = Utc::now();
        let v = echo_velocity(0, created);
        assert!(v < 0.01, "velocity should be ~0 for new memory, got {v}");
    }

    #[test]
    fn echo_velocity_increases_with_count() {
        let created = Utc::now() - chrono::Duration::days(1);
        let v1 = echo_velocity(1, created);
        let v5 = echo_velocity(5, created);
        assert!(v5 > v1, "more echoes = higher velocity");
    }

    #[test]
    fn echo_velocity_bounded() {
        let created = Utc::now() - chrono::Duration::seconds(1);
        let v = echo_velocity(1000, created);
        // Logistic squash is mathematically < 1.0 but f32 precision
        // truncates exp(-large) to 0.0 for extreme rates, giving exactly 1.0.
        assert!(v <= 1.0, "velocity must be <= 1.0, got {v}");
    }

    #[test]
    fn surprise_score_zero_for_identical() {
        let emb = vec![1.0, 0.0, 0.0];
        let mean = vec![1.0, 0.0, 0.0];
        assert_eq!(surprise_score(&emb, &mean), 0.0);
    }

    #[test]
    fn surprise_score_increases_with_distance() {
        let mean = vec![0.0; 384];
        let close = vec![0.1; 384];
        let far = vec![1.0; 384];
        let s_close = surprise_score(&close, &mean);
        let s_far = surprise_score(&far, &mean);
        assert!(s_far > s_close);
    }

    #[test]
    fn surprise_score_capped_at_one() {
        let emb = vec![100.0; 384];
        let mean = vec![0.0; 384];
        assert!(surprise_score(&emb, &mean) <= 1.0);
    }

    #[test]
    fn compute_mean_of_two() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let mean = compute_embedding_mean(&[&a, &b]);
        assert_eq!(mean, vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn compute_mean_empty() {
        let mean = compute_embedding_mean(&[]);
        assert!(mean.is_empty());
    }

    #[test]
    fn surprise_score_empty_embedding() {
        assert_eq!(surprise_score(&[], &[]), 0.0);
    }

    #[test]
    fn surprise_score_mismatched_lengths() {
        let emb = vec![1.0, 0.0];
        let mean = vec![1.0, 0.0, 0.0];
        assert_eq!(surprise_score(&emb, &mean), 0.0);
    }

    #[test]
    fn compute_importance_without_surprise() {
        let entry = MemoryEntry::new("test".into(), vec![1.0; 384], "document".into());
        let emb = vec![1.0; 384];
        let score = compute_importance(&entry, &emb, None);
        assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
    }

    #[test]
    fn compute_importance_with_surprise() {
        let entry = MemoryEntry::new("test".into(), vec![1.0; 384], "document".into());
        let emb = vec![1.0; 384];
        let mean = vec![0.0; 384];
        let score = compute_importance(&entry, &emb, Some(&mean));
        assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
    }

    #[test]
    fn compute_importance_clamped() {
        // Even with maximum inputs, score stays in [0, 1]
        let mut entry = MemoryEntry::new("test".into(), vec![100.0; 384], "document".into());
        entry.novelty_score = 1.0;
        entry.echo_count = 10000;
        let emb = vec![100.0; 384];
        let mean = vec![0.0; 384];
        let score = compute_importance(&entry, &emb, Some(&mean));
        assert!(score <= 1.0, "score must be clamped to 1.0, got {score}");
    }
}
