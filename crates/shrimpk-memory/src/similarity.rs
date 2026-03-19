//! SIMD-accelerated cosine similarity via simsimd.
//!
//! Phase 1: brute-force similarity against all embeddings.
//! Phase 2 will add LSH for sub-linear candidate retrieval.

use tracing::instrument;

/// Compute cosine similarity between two embedding vectors.
///
/// Uses simsimd for SIMD-accelerated computation when available,
/// falls back to scalar implementation otherwise.
///
/// Returns a value in [-1.0, 1.0] where 1.0 = identical direction.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have equal dimensions");

    // Try simsimd first (auto-detects AVX2/NEON/etc.)
    simd_cosine(a, b).unwrap_or_else(|| scalar_cosine(a, b))
}

/// SIMD-accelerated cosine similarity via simsimd.
///
/// Returns None if simsimd fails (e.g., unsupported platform).
/// simsimd returns cosine **distance** (1 - similarity), so we convert.
#[inline]
fn simd_cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    use simsimd::SpatialSimilarity;
    let distance = f32::cosine(a, b)?;
    Some(1.0 - distance as f32)
}

/// Scalar fallback for cosine similarity.
///
/// Used when simsimd is unavailable or returns an error.
#[inline]
fn scalar_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let ai = ai as f64;
        let bi = bi as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (dot / denom) as f32
}

/// Score and filter candidates against a query embedding.
///
/// Returns candidates with similarity >= threshold, sorted by score descending.
///
/// # Arguments
/// * `query` - The query embedding vector
/// * `candidates` - Slice of (index, embedding) pairs to score
/// * `threshold` - Minimum cosine similarity to include in results
#[instrument(skip(query, candidates), fields(num_candidates = candidates.len()))]
pub fn rank_candidates(
    query: &[f32],
    candidates: &[(usize, &[f32])],
    threshold: f32,
) -> Vec<(usize, f32)> {
    let start = std::time::Instant::now();

    let mut scored: Vec<(usize, f32)> = candidates
        .iter()
        .filter_map(|&(idx, emb)| {
            let sim = cosine_similarity(query, emb);
            if sim >= threshold {
                Some((idx, sim))
            } else {
                None
            }
        })
        .collect();

    // Sort by score descending (highest similarity first)
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let elapsed = start.elapsed();
    tracing::debug!(
        candidates = candidates.len(),
        results = scored.len(),
        elapsed_us = elapsed.as_micros(),
        "rank_candidates complete"
    );

    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5, "identical vectors should have similarity ~1.0, got {sim}");
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "orthogonal vectors should have similarity ~0.0, got {sim}");
    }

    #[test]
    fn opposite_vectors_have_negative_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5, "opposite vectors should have similarity ~-1.0, got {sim}");
    }

    #[test]
    fn scalar_fallback_matches_simd() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let scalar = scalar_cosine(&a, &b);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (scalar - sim).abs() < 1e-4,
            "scalar ({scalar}) and cosine_similarity ({sim}) should be close"
        );
    }

    #[test]
    fn zero_vector_returns_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "zero vector should give similarity 0.0, got {sim}");
    }

    #[test]
    fn rank_candidates_filters_and_sorts() {
        let query = vec![1.0, 0.0, 0.0];
        let emb_a = vec![1.0, 0.0, 0.0]; // similarity = 1.0
        let emb_b = vec![0.0, 1.0, 0.0]; // similarity = 0.0
        let emb_c = vec![0.7, 0.7, 0.0]; // similarity ~0.707

        let candidates: Vec<(usize, &[f32])> = vec![
            (0, &emb_a),
            (1, &emb_b),
            (2, &emb_c),
        ];

        let results = rank_candidates(&query, &candidates, 0.5);

        // Should include emb_a (1.0) and emb_c (~0.707), exclude emb_b (0.0)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // highest score first
        assert_eq!(results[1].0, 2);
    }

    #[test]
    fn rank_candidates_empty_input() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates: Vec<(usize, &[f32])> = vec![];
        let results = rank_candidates(&query, &candidates, 0.3);
        assert!(results.is_empty());
    }
}
