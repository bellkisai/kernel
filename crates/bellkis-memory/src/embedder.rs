//! Sentence embedding via fastembed.
//!
//! Wraps `fastembed::TextEmbedding` with the all-MiniLM-L6-v2 model
//! for 384-dimensional sentence embeddings.

use bellkis_core::{BellkisError, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tracing::instrument;

/// Sentence embedder wrapping fastembed.
///
/// Initializes the all-MiniLM-L6-v2 model on construction.
/// Thread-safe: `TextEmbedding` is `Send` (but not `Sync`),
/// so share via `Mutex` or create per-thread instances.
pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Initialize the embedder with the all-MiniLM-L6-v2 model.
    ///
    /// This downloads the model on first run (~23MB ONNX) and caches it.
    /// Subsequent calls load from cache.
    ///
    /// # Errors
    /// Returns `BellkisError::Embedding` if model initialization fails
    /// (e.g., download failure, ONNX runtime error).
    #[instrument]
    pub fn new() -> Result<Self> {
        let start = std::time::Instant::now();

        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
        ).map_err(|e| BellkisError::Embedding(format!("Failed to init all-MiniLM-L6-v2: {e}")))?;

        let elapsed = start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis(),
            model = "all-MiniLM-L6-v2",
            dim = 384,
            "Embedder initialized"
        );

        Ok(Self { model })
    }

    /// Embed a single text string into a 384-dimensional vector.
    ///
    /// # Errors
    /// Returns `BellkisError::Embedding` if embedding generation fails.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        let results = self.model
            .embed(vec![text.to_string()], None)
            .map_err(|e| BellkisError::Embedding(format!("Embed failed: {e}")))?;

        let embedding = results
            .into_iter()
            .next()
            .ok_or_else(|| BellkisError::Embedding("Empty embedding result".into()))?;

        let elapsed = start.elapsed();
        tracing::debug!(
            dim = embedding.len(),
            elapsed_us = elapsed.as_micros(),
            "Single embed complete"
        );

        Ok(embedding)
    }

    /// Batch-embed multiple texts.
    ///
    /// More efficient than calling `embed()` in a loop because
    /// fastembed batches the ONNX inference.
    ///
    /// # Errors
    /// Returns `BellkisError::Embedding` if any embedding generation fails.
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    pub fn embed_batch(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();
        let count = texts.len();

        let results = self.model
            .embed(texts, None)
            .map_err(|e| BellkisError::Embedding(format!("Batch embed failed: {e}")))?;

        let elapsed = start.elapsed();
        tracing::debug!(
            count = count,
            elapsed_ms = elapsed.as_millis(),
            avg_us = if count > 0 { elapsed.as_micros() / count as u128 } else { 0 },
            "Batch embed complete"
        );

        Ok(results)
    }

    /// Get the embedding dimension (384 for all-MiniLM-L6-v2).
    pub fn dimension(&self) -> usize {
        384
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require the model to be downloaded.
    // They are ignored by default and run with `cargo test -- --ignored`
    // or in CI where the model cache is warm.

    #[test]
    #[ignore = "requires fastembed model download"]
    fn embedder_initializes() {
        let embedder = Embedder::new().expect("Embedder should init");
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn embed_single_text() {
        let mut embedder = Embedder::new().expect("Embedder should init");
        let embedding = embedder.embed("Hello world").expect("Should embed");
        assert_eq!(embedding.len(), 384, "MiniLM-L6-v2 should produce 384-dim vectors");
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn embed_batch_texts() {
        let mut embedder = Embedder::new().expect("Embedder should init");
        let texts = vec![
            "The cat sat on the mat".to_string(),
            "Dogs are loyal companions".to_string(),
            "Machine learning is fascinating".to_string(),
        ];
        let embeddings = embedder.embed_batch(texts).expect("Should batch embed");
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn similar_texts_have_higher_similarity() {
        let mut embedder = Embedder::new().expect("Embedder should init");
        let cat = embedder.embed("The cat sat on the mat").unwrap();
        let kitten = embedder.embed("A kitten rests on a rug").unwrap();
        let code = embedder.embed("fn main() { println!(\"hello\"); }").unwrap();

        // cat and kitten should be more similar than cat and code
        let sim_cat_kitten: f32 = cat.iter().zip(kitten.iter()).map(|(a, b)| a * b).sum();
        let sim_cat_code: f32 = cat.iter().zip(code.iter()).map(|(a, b)| a * b).sum();

        assert!(
            sim_cat_kitten > sim_cat_code,
            "cat-kitten ({sim_cat_kitten}) should be more similar than cat-code ({sim_cat_code})"
        );
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn embed_batch_empty_returns_empty() {
        let mut embedder = Embedder::new().expect("Embedder should init");
        let embeddings = embedder.embed_batch(Vec::new()).expect("Should handle empty");
        assert!(embeddings.is_empty());
    }
}
