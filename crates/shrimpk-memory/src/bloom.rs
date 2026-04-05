//! Bloom filter pre-screening for the Echo Memory pipeline.
//!
//! Provides an O(1) topic-level pre-check before LSH candidate retrieval.
//! If a query's fingerprints are absent from the Bloom filter, no memories
//! can possibly match, and the entire embedding + LSH + similarity pipeline
//! is skipped. This eliminates wasted computation on clearly irrelevant queries.
//!
//! Fingerprints are unigrams (individual words) and bigrams (adjacent word pairs)
//! extracted from both stored memories and incoming queries. The filter requires
//! at least 2 matching fingerprints to report a potential match, reducing the
//! base false-positive rate significantly.

use bloomfilter::Bloom;

/// Topic-level Bloom filter for fast echo pre-screening.
///
/// Stores fingerprints (unigrams + bigrams) from all memory texts.
/// Cannot remove individual items — call [`TopicFilter::rebuild`] after deletions.
pub struct TopicFilter {
    /// The underlying Bloom filter.
    filter: Bloom<String>,
    /// Number of memory texts inserted (not fingerprint count).
    entry_count: usize,
    /// Expected items capacity (for rebuild).
    expected_items: usize,
    /// Target false-positive rate (for rebuild).
    fpr: f64,
}

impl TopicFilter {
    /// Create a new topic filter.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of fingerprints. Default: 1M items.
    /// * `fpr` - Target false-positive rate. Default: 1% (~1.14 MB at 1M items).
    pub fn new(expected_items: usize, fpr: f64) -> Self {
        Self {
            filter: Bloom::new_for_fp_rate(expected_items, fpr),
            entry_count: 0,
            expected_items,
            fpr,
        }
    }

    /// Extract text fingerprints: lowercased unigrams (3+ chars) and bigrams.
    ///
    /// Returns a combined vec of individual words and adjacent word pairs
    /// joined with `_`. Short words (< 3 chars) are filtered to avoid
    /// noise from articles, prepositions, etc.
    fn extract_fingerprints(text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().filter(|w| w.len() >= 3).collect();

        let mut fingerprints = Vec::with_capacity(words.len() * 2);

        // Unigrams
        for word in &words {
            fingerprints.push((*word).to_string());
        }

        // Bigrams
        for pair in words.windows(2) {
            fingerprints.push(format!("{}_{}", pair[0], pair[1]));
        }

        fingerprints
    }

    /// Insert a memory's text fingerprints into the filter.
    pub fn insert_memory(&mut self, text: &str) {
        let fingerprints = Self::extract_fingerprints(text);
        for fp in &fingerprints {
            self.filter.set(fp);
        }
        self.entry_count += 1;
    }

    /// Check if a query might have matching memories.
    ///
    /// Extracts fingerprints from the query and counts how many are present
    /// in the Bloom filter. Returns `true` if at least 2 fingerprints match,
    /// which reduces false positives compared to a single-match check.
    ///
    /// A `false` return is **definitive** — no memories can match this query.
    /// A `true` return means memories *might* match (subject to Bloom FPR).
    ///
    /// Uses a single-fingerprint threshold to avoid rejecting semantically
    /// valid but lexically distant queries (e.g., "What degree?" vs
    /// "Business Administration").
    pub fn might_match(&self, query: &str) -> bool {
        let fingerprints = Self::extract_fingerprints(query);
        if fingerprints.is_empty() {
            return false;
        }

        let hit_count = fingerprints
            .iter()
            .filter(|fp| self.filter.check(fp))
            .count();

        hit_count >= 1
    }

    /// Rebuild the filter from scratch with the given texts.
    ///
    /// Required after deletions since Bloom filters cannot remove items.
    /// Creates a fresh filter and reinserts all texts.
    pub fn rebuild(&mut self, texts: &[&str]) {
        self.filter.clear();
        self.entry_count = 0;
        for text in texts {
            self.insert_memory(text);
        }
    }

    /// Number of memory texts inserted.
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Whether the filter has no entries.
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Approximate memory usage in bytes.
    ///
    /// The Bloom filter bit-array size is approximately `-(n * ln(p)) / (ln(2)^2)`
    /// bits, where n = expected items and p = FPR. We convert to bytes.
    pub fn size_bytes(&self) -> usize {
        let bits = -((self.expected_items as f64) * self.fpr.ln()) / (2.0_f64.ln().powi(2));
        (bits / 8.0).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programming_texts_match_related_query() {
        let mut filter = TopicFilter::new(10_000, 0.01);

        let texts = [
            "Rust is a systems programming language focused on safety",
            "Python is great for machine learning and data science",
            "JavaScript powers the modern web with React and Node",
            "Go provides excellent concurrency with goroutines",
            "TypeScript adds static types to JavaScript development",
            "Kotlin is the preferred language for Android development",
            "Swift is used for iOS and macOS application development",
            "C++ offers low-level memory control for performance",
            "Java remains popular for enterprise backend systems",
            "Ruby on Rails is a productive web framework",
        ];

        for text in &texts {
            filter.insert_memory(text);
        }

        // Query needs at least 2 fingerprint hits. "Python machine learning" matches
        // unigrams "python", "machine", "learning" — all present in the stored texts.
        assert!(
            filter.might_match("Python machine learning"),
            "Should match: 'python', 'machine', 'learning' are all in the stored texts"
        );
    }

    #[test]
    fn unrelated_query_does_not_match() {
        let mut filter = TopicFilter::new(10_000, 0.01);

        let texts = [
            "Rust is a systems programming language focused on safety",
            "Python is great for machine learning and data science",
            "JavaScript powers the modern web with React and Node",
            "Go provides excellent concurrency with goroutines",
            "TypeScript adds static types to JavaScript development",
            "Kotlin is the preferred language for Android development",
            "Swift is used for iOS and macOS application development",
            "C++ offers low-level memory control for performance",
            "Java remains popular for enterprise backend systems",
            "Ruby on Rails is a productive web framework",
        ];

        for text in &texts {
            filter.insert_memory(text);
        }

        assert!(
            !filter.might_match("chocolate cake recipe"),
            "Should not match: no programming texts contain chocolate/cake/recipe"
        );
    }

    #[test]
    fn empty_filter_returns_false() {
        let filter = TopicFilter::new(10_000, 0.01);
        assert!(!filter.might_match("any query at all"));
    }

    #[test]
    fn rebuild_preserves_results() {
        let mut filter = TopicFilter::new(10_000, 0.01);

        let texts = [
            "Rust is a systems programming language",
            "Python is great for data science",
            "JavaScript powers the web",
        ];

        for text in &texts {
            filter.insert_memory(text);
        }

        let before_match = filter.might_match("Rust programming language");
        let before_no_match = filter.might_match("chocolate cake recipe");

        // Rebuild from scratch
        let text_refs: Vec<&str> = texts.to_vec();
        filter.rebuild(&text_refs);

        assert_eq!(
            filter.might_match("Rust programming language"),
            before_match,
            "Rebuild should produce same match results"
        );
        assert_eq!(
            filter.might_match("chocolate cake recipe"),
            before_no_match,
            "Rebuild should produce same non-match results"
        );
    }

    #[test]
    fn fingerprint_extraction_produces_unigrams_and_bigrams() {
        let fps = TopicFilter::extract_fingerprints("Rust is great for systems");
        // "is" is filtered (< 3 chars), "for" is filtered (< 3 chars on second thought — "for" is 3 chars, kept)
        // Words kept: "rust", "great", "for", "systems"
        // Unigrams: ["rust", "great", "for", "systems"]
        // Bigrams: ["rust_great", "great_for", "for_systems"]

        // Check unigrams present
        assert!(
            fps.contains(&"rust".to_string()),
            "Should have unigram 'rust'"
        );
        assert!(
            fps.contains(&"great".to_string()),
            "Should have unigram 'great'"
        );
        assert!(
            fps.contains(&"systems".to_string()),
            "Should have unigram 'systems'"
        );

        // Check bigrams present
        assert!(
            fps.contains(&"rust_great".to_string()),
            "Should have bigram 'rust_great'"
        );
        assert!(
            fps.contains(&"great_for".to_string()),
            "Should have bigram 'great_for'"
        );
        assert!(
            fps.contains(&"for_systems".to_string()),
            "Should have bigram 'for_systems'"
        );

        // "is" should be filtered (2 chars)
        assert!(
            !fps.contains(&"is".to_string()),
            "Should filter out 'is' (< 3 chars)"
        );
    }

    #[test]
    fn entry_count_tracks_insertions() {
        let mut filter = TopicFilter::new(1_000, 0.01);
        assert_eq!(filter.len(), 0);

        filter.insert_memory("first memory");
        assert_eq!(filter.len(), 1);

        filter.insert_memory("second memory");
        assert_eq!(filter.len(), 2);
    }

    #[test]
    fn size_bytes_returns_nonzero() {
        let filter = TopicFilter::new(1_000_000, 0.01);
        let size = filter.size_bytes();
        assert!(size > 0, "Size should be positive");
        // At 1M items, 1% FPR, expect ~1.14 MB
        assert!(
            size > 1_000_000 && size < 2_000_000,
            "Expected ~1.14 MB for 1M items at 1% FPR, got {} bytes",
            size
        );
    }
}
