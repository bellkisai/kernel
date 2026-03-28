//! Multi-channel embedding via fastembed.
//!
//! Wraps `fastembed::TextEmbedding` with the all-MiniLM-L6-v2 model
//! for 384-dimensional sentence embeddings. Vision (CLIP 512-dim) and
//! Speech (579-dim) channels are gated behind `vision` and `speech`
//! feature flags and will be loaded in KS35/KS36.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use shrimpk_core::{Result, ShrimPKError};
use tracing::instrument;

/// Multi-channel embedder for text, vision, and speech modalities.
///
/// Text channel (always available): all-MiniLM-L6-v2, 384-dim.
/// Vision channel (feature = "vision"): CLIP, 512-dim — KS35.
/// Speech channel (feature = "speech"): audio encoder, 579-dim — KS36.
///
/// Thread-safe: `TextEmbedding` is `Send` (but not `Sync`),
/// so share via `Mutex` or create per-thread instances.
pub struct MultiEmbedder {
    text: TextEmbedding,
}

impl MultiEmbedder {
    /// Initialize the multi-channel embedder.
    ///
    /// Currently loads only the text model (all-MiniLM-L6-v2).
    /// Vision and speech models will be loaded when their feature flags
    /// are enabled and the model loading code is implemented (KS35/KS36).
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if model initialization fails
    /// (e.g., download failure, ONNX runtime error).
    #[instrument]
    pub fn new() -> Result<Self> {
        let start = std::time::Instant::now();

        let text = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .map_err(|e| {
                ShrimPKError::Embedding(format!("Failed to init all-MiniLM-L6-v2: {e}"))
            })?;

        let elapsed = start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis(),
            model = "all-MiniLM-L6-v2",
            dim = 384,
            "MultiEmbedder initialized (text channel)"
        );

        Ok(Self { text })
    }

    /// Embed a single text string into a 384-dimensional vector.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if embedding generation fails.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        let results = self
            .text
            .embed(vec![text.to_string()], None)
            .map_err(|e| ShrimPKError::Embedding(format!("Embed failed: {e}")))?;

        let embedding = results
            .into_iter()
            .next()
            .ok_or_else(|| ShrimPKError::Embedding("Empty embedding result".into()))?;

        let elapsed = start.elapsed();
        tracing::debug!(
            dim = embedding.len(),
            elapsed_us = elapsed.as_micros(),
            "Single text embed complete"
        );

        Ok(embedding)
    }

    /// Batch-embed multiple texts.
    ///
    /// More efficient than calling `embed_text()` in a loop because
    /// fastembed batches the ONNX inference.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if any embedding generation fails.
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    pub fn embed_batch(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();
        let count = texts.len();

        let results = self
            .text
            .embed(texts, None)
            .map_err(|e| ShrimPKError::Embedding(format!("Batch embed failed: {e}")))?;

        let elapsed = start.elapsed();
        tracing::debug!(
            count = count,
            elapsed_ms = elapsed.as_millis(),
            avg_us = if count > 0 {
                elapsed.as_micros() / count as u128
            } else {
                0
            },
            "Batch text embed complete"
        );

        Ok(results)
    }

    /// Get the text embedding dimension (384 for all-MiniLM-L6-v2).
    pub fn text_dimension(&self) -> usize {
        384
    }

    /// Embed an image into a 512-dimensional CLIP vector.
    ///
    /// Phase 2 (KS35) — CLIP model not loaded yet. Returns `Ok(None)`.
    #[cfg(feature = "vision")]
    pub fn embed_image(&mut self, _image_data: &[u8]) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Embed text into CLIP's shared vision-text space (512-dim).
    ///
    /// This allows text queries to match against image embeddings.
    /// Phase 2 (KS35) — CLIP model not loaded yet. Returns `Ok(None)`.
    #[cfg(feature = "vision")]
    pub fn embed_text_for_vision(&mut self, _text: &str) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Embed raw PCM audio into a 579-dimensional speech vector.
    ///
    /// Captures paralinguistic features (tone, pace, emotion), NOT speech-to-text.
    /// Phase 3 (KS36) — speech model not loaded yet. Returns `Ok(None)`.
    #[cfg(feature = "speech")]
    pub fn embed_audio(&mut self, _pcm: &[f32], _sample_rate: u32) -> Result<Option<Vec<f32>>> {
        Ok(None)
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
        let embedder = MultiEmbedder::new().expect("MultiEmbedder should init");
        assert_eq!(embedder.text_dimension(), 384);
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn embed_single_text() {
        let mut embedder = MultiEmbedder::new().expect("MultiEmbedder should init");
        let embedding = embedder.embed_text("Hello world").expect("Should embed");
        assert_eq!(
            embedding.len(),
            384,
            "MiniLM-L6-v2 should produce 384-dim vectors"
        );
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn embed_batch_texts() {
        let mut embedder = MultiEmbedder::new().expect("MultiEmbedder should init");
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
        let mut embedder = MultiEmbedder::new().expect("MultiEmbedder should init");
        let cat = embedder.embed_text("The cat sat on the mat").unwrap();
        let kitten = embedder.embed_text("A kitten rests on a rug").unwrap();
        let code = embedder
            .embed_text("fn main() { println!(\"hello\"); }")
            .unwrap();

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
        let mut embedder = MultiEmbedder::new().expect("MultiEmbedder should init");
        let embeddings = embedder
            .embed_batch(Vec::new())
            .expect("Should handle empty");
        assert!(embeddings.is_empty());
    }
}
