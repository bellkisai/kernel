//! Locality-Sensitive Hashing (LSH) for sub-linear Echo Memory search.
//!
//! Uses random hyperplane hashing (SimHash variant) for cosine similarity.
//! Multiple hash tables with independent hyperplane sets provide high recall
//! through the union of candidate sets.
//!
//! Complexity: O(L * K * D) per query where L = tables, K = bits, D = dim,
//! compared to O(N * D) brute-force. Sub-linear in N when L*K*D << N*D.

use rand_distr::{Distribution, Normal};
use std::collections::{HashMap, HashSet};

/// A single hash table with K random hyperplanes.
#[derive(Debug, Clone)]
struct HashTable {
    /// K random unit vectors (Gaussian distributed, L2-normalized).
    /// Each inner Vec has length `dim`.
    hyperplanes: Vec<Vec<f32>>,
    /// hash_code -> list of memory indices in that bucket.
    buckets: HashMap<u64, Vec<u32>>,
}

/// Cosine-similarity LSH using random hyperplane hashing.
///
/// Maintains L independent hash tables, each with K hyperplanes.
/// A query hashes into each table and collects the union of all matching
/// bucket contents as candidates for exact similarity computation.
#[derive(Debug, Clone)]
pub struct CosineHash {
    /// L independent hash tables.
    tables: Vec<HashTable>,
    /// Embedding dimension (e.g. 384 for all-MiniLM-L6-v2).
    dim: usize,
    /// Reverse index: id -> [(table_idx, bucket_hash)] for O(L) removal.
    reverse_index: HashMap<u32, Vec<(usize, u64)>>,
}

/// Compute the hash code for a vector against a single table's hyperplanes.
///
/// For each hyperplane h_i, set bit i = 1 if dot(vec, h_i) >= 0, else 0.
/// Returns the K-bit hash packed into a u64.
fn hash_vector(table: &HashTable, vec: &[f32]) -> u64 {
    let mut hash: u64 = 0;
    for (i, hyperplane) in table.hyperplanes.iter().enumerate() {
        let dot: f32 = vec.iter().zip(hyperplane.iter()).map(|(a, b)| a * b).sum();
        if dot >= 0.0 {
            hash |= 1u64 << i;
        }
    }
    hash
}

impl CosineHash {
    /// Create a new CosineHash index.
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension (384 for all-MiniLM-L6-v2)
    /// * `num_tables` - Number of independent hash tables (L). More tables = higher recall.
    /// * `bits_per_table` - Number of hyperplane bits per table (K). More bits = fewer candidates per bucket.
    ///
    /// Default recommendation: 16 tables, 10 bits per table.
    pub fn new(dim: usize, num_tables: usize, bits_per_table: usize) -> Self {
        assert!(
            bits_per_table <= 64,
            "bits_per_table must be <= 64 (packed into u64)"
        );
        assert!(dim > 0, "dimension must be > 0");

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0f32, 1.0f32).expect("valid normal distribution params");

        let tables = (0..num_tables)
            .map(|_| {
                let hyperplanes = (0..bits_per_table)
                    .map(|_| {
                        // Generate random vector from Normal(0,1)
                        let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
                        // L2-normalize
                        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for x in &mut v {
                                *x /= norm;
                            }
                        }
                        v
                    })
                    .collect();
                HashTable {
                    hyperplanes,
                    buckets: HashMap::new(),
                }
            })
            .collect();

        Self {
            tables,
            dim,
            reverse_index: HashMap::new(),
        }
    }

    /// Insert an entry into the LSH index.
    ///
    /// Hashes the embedding into each table and adds the id to the matching bucket.
    pub fn insert(&mut self, id: u32, embedding: &[f32]) {
        debug_assert_eq!(
            embedding.len(),
            self.dim,
            "embedding dimension mismatch: expected {}, got {}",
            self.dim,
            embedding.len()
        );

        let mut locations = Vec::with_capacity(self.tables.len());

        for (table_idx, table) in self.tables.iter_mut().enumerate() {
            let hash = hash_vector(table, embedding);
            table.buckets.entry(hash).or_default().push(id);
            locations.push((table_idx, hash));
        }

        self.reverse_index.insert(id, locations);
    }

    /// Query the LSH index for candidate nearest neighbors.
    ///
    /// Hashes the query into each table and returns the deduplicated union
    /// of all matching bucket contents.
    pub fn query(&self, embedding: &[f32]) -> Vec<u32> {
        debug_assert_eq!(
            embedding.len(),
            self.dim,
            "embedding dimension mismatch: expected {}, got {}",
            self.dim,
            embedding.len()
        );

        let mut seen = HashSet::new();

        for table in &self.tables {
            let hash = hash_vector(table, embedding);
            if let Some(bucket) = table.buckets.get(&hash) {
                for &id in bucket {
                    seen.insert(id);
                }
            }
        }

        seen.into_iter().collect()
    }

    /// Remove an entry from the LSH index.
    ///
    /// Uses the reverse index for O(L) removal across all tables.
    pub fn remove(&mut self, id: u32) {
        if let Some(locations) = self.reverse_index.remove(&id) {
            for (table_idx, bucket_hash) in locations {
                if let Some(bucket) = self.tables[table_idx].buckets.get_mut(&bucket_hash) {
                    bucket.retain(|&x| x != id);
                    // Clean up empty buckets to save memory
                    if bucket.is_empty() {
                        self.tables[table_idx].buckets.remove(&bucket_hash);
                    }
                }
            }
        }
    }

    /// Total number of unique entries in the index.
    pub fn len(&self) -> usize {
        self.reverse_index.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.reverse_index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a random 384-dim vector (not normalized, like real embeddings).
    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        (0..dim).map(|_| normal.sample(&mut rng)).collect()
    }

    #[test]
    fn insert_and_query_finds_candidates() {
        let dim = 384;
        let mut lsh = CosineHash::new(dim, 16, 10);

        // Insert 1000 random vectors
        let vectors: Vec<Vec<f32>> = (0..1000).map(|i| random_vec(dim, i)).collect();
        for (i, v) in vectors.iter().enumerate() {
            lsh.insert(i as u32, v);
        }

        assert_eq!(lsh.len(), 1000);
        assert!(!lsh.is_empty());

        // Query with one of the stored vectors — should find candidates
        let candidates = lsh.query(&vectors[42]);
        assert!(
            !candidates.is_empty(),
            "query should return at least some candidates"
        );
        // The query vector itself should always be found (same hash)
        assert!(
            candidates.contains(&42),
            "query vector should find itself among candidates"
        );
    }

    #[test]
    fn identical_vector_always_found() {
        let dim = 384;
        let mut lsh = CosineHash::new(dim, 16, 10);

        let v = random_vec(dim, 99);
        lsh.insert(0, &v);

        // Query with the exact same vector
        let candidates = lsh.query(&v);
        assert!(
            candidates.contains(&0),
            "identical vector must always be found"
        );
    }

    #[test]
    fn remove_excludes_from_results() {
        let dim = 384;
        let mut lsh = CosineHash::new(dim, 16, 10);

        let v = random_vec(dim, 7);
        lsh.insert(7, &v);
        assert_eq!(lsh.len(), 1);

        // Verify it's found
        let before = lsh.query(&v);
        assert!(before.contains(&7));

        // Remove and verify it's gone
        lsh.remove(7);
        assert_eq!(lsh.len(), 0);
        assert!(lsh.is_empty());

        let after = lsh.query(&v);
        assert!(
            !after.contains(&7),
            "removed entry should not appear in results"
        );
    }

    #[test]
    fn empty_index_returns_empty() {
        let lsh = CosineHash::new(384, 16, 10);
        assert!(lsh.is_empty());
        assert_eq!(lsh.len(), 0);

        let v = random_vec(384, 0);
        let results = lsh.query(&v);
        assert!(results.is_empty(), "empty index should return no results");
    }

    #[test]
    fn candidate_count_is_reasonable() {
        let dim = 384;
        let n = 1000;
        let mut lsh = CosineHash::new(dim, 16, 10);

        let vectors: Vec<Vec<f32>> = (0..n).map(|i| random_vec(dim, i)).collect();
        for (i, v) in vectors.iter().enumerate() {
            lsh.insert(i as u32, v);
        }

        // Query with a random vector not in the set
        let query = random_vec(dim, 999_999);
        let candidates = lsh.query(&query);

        // Should not return all N (that would mean LSH is not filtering)
        // With 10 bits per table, expected bucket size is N/2^10 = ~1 per table
        // With 16 tables, expect roughly 16 candidates on average (with some overlap)
        // Allow generous upper bound: should be well under N/2
        assert!(
            candidates.len() < (n / 2) as usize,
            "LSH should filter candidates, got {}/{n}",
            candidates.len()
        );
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let mut lsh = CosineHash::new(384, 16, 10);
        let v = random_vec(384, 1);
        lsh.insert(0, &v);

        // Removing a non-existent id should not panic or corrupt state
        lsh.remove(999);
        assert_eq!(lsh.len(), 1);

        let candidates = lsh.query(&v);
        assert!(candidates.contains(&0));
    }

    #[test]
    fn multiple_inserts_and_removes() {
        let dim = 384;
        let mut lsh = CosineHash::new(dim, 8, 8);

        let vectors: Vec<Vec<f32>> = (0..100).map(|i| random_vec(dim, i)).collect();
        for (i, v) in vectors.iter().enumerate() {
            lsh.insert(i as u32, v);
        }
        assert_eq!(lsh.len(), 100);

        // Remove every other entry
        for i in (0..100).step_by(2) {
            lsh.remove(i as u32);
        }
        assert_eq!(lsh.len(), 50);

        // Odd entries should still be findable
        for i in (1..100).step_by(2) {
            let candidates = lsh.query(&vectors[i]);
            assert!(
                candidates.contains(&(i as u32)),
                "entry {i} should still be found after removing others"
            );
        }
    }
}
