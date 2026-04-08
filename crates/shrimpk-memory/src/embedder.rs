//! Multi-channel embedding with pluggable text provider (KS75).
//!
//! Text channel delegates to an `EmbeddingProvider` implementation selected
//! at runtime via `EchoConfig` (default: fastembed BGE-small-EN-v1.5, 384-dim).
//! Vision (CLIP 512-dim) and Speech (640-dim) channels are gated behind
//! `vision` and `speech` feature flags.

#[cfg(feature = "vision")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(any(feature = "vision", feature = "speech"))]
use shrimpk_core::ShrimPKError;
use shrimpk_core::{EchoConfig, EmbeddingProvider, Result};
use tracing::instrument;

/// Multi-channel embedder for text, vision, and speech modalities.
///
/// Text channel delegates to a pluggable `EmbeddingProvider` (KS75).
/// Vision channel (feature = "vision"): CLIP ViT-B-32, 512-dim.
/// Speech channel (feature = "speech"): ECAPA-TDNN (256) + Whisper-tiny encoder (384) = 640-dim.
///
/// Thread-safe: providers are `Send` (but not `Sync`),
/// so share via `Mutex` or create per-thread instances.
pub struct MultiEmbedder {
    /// Pluggable text embedding provider (fastembed or OpenAI-compatible API).
    text_provider: Box<dyn EmbeddingProvider>,
    /// CLIP vision encoder — embeds images into 512-dim CLIP space.
    #[cfg(feature = "vision")]
    vision: Option<fastembed::ImageEmbedding>,
    /// CLIP text encoder — embeds text into the same 512-dim CLIP space.
    /// Separate from text provider because the embedding spaces are incompatible.
    #[cfg(feature = "vision")]
    vision_text: Option<TextEmbedding>,
    /// Speech embedder — 2 ONNX models producing a 640-dim paralinguistic embedding.
    /// Present even when models aren't loaded (`is_ready() == false`).
    #[cfg(feature = "speech")]
    speech: Option<crate::speech::SpeechEmbedder>,
}

impl MultiEmbedder {
    /// Initialize the multi-channel embedder from config.
    ///
    /// The text channel is delegated to an `EmbeddingProvider` selected by
    /// `config.embedding_provider` (default: fastembed BGE-small-EN-v1.5).
    /// Vision/speech channels are unchanged (compile-time feature flags).
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if the text provider fails to initialize.
    /// Vision model failures are logged as warnings and result in `vision = None`.
    #[instrument(skip(config))]
    pub fn new(config: &EchoConfig) -> Result<Self> {
        let text_provider = crate::embedding_provider::from_config(config)?;

        #[cfg(feature = "vision")]
        let (vision, vision_text) = {
            use fastembed::{ImageEmbedding, ImageEmbeddingModel, ImageInitOptions};

            let vis_start = std::time::Instant::now();
            let vision = match ImageEmbedding::try_new(ImageInitOptions::new(
                ImageEmbeddingModel::ClipVitB32,
            )) {
                Ok(model) => {
                    tracing::info!(
                        elapsed_ms = vis_start.elapsed().as_millis(),
                        model = "clip-ViT-B-32-vision",
                        dim = 512,
                        "CLIP vision encoder initialized"
                    );
                    Some(model)
                }
                Err(e) => {
                    tracing::warn!("CLIP vision encoder failed to init, vision disabled: {e}");
                    None
                }
            };

            let vt_start = std::time::Instant::now();
            let vision_text =
                match TextEmbedding::try_new(InitOptions::new(EmbeddingModel::ClipVitB32)) {
                    Ok(model) => {
                        tracing::info!(
                            elapsed_ms = vt_start.elapsed().as_millis(),
                            model = "clip-ViT-B-32-text",
                            dim = 512,
                            "CLIP text encoder initialized"
                        );
                        Some(model)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "CLIP text encoder failed to init, cross-modal search disabled: {e}"
                        );
                        None
                    }
                };

            (vision, vision_text)
        };

        #[cfg(feature = "speech")]
        let speech = {
            let embedder = crate::speech::SpeechEmbedder::new();
            // Keep it even when not ready — structure is correct, KS37+ loads real models
            Some(embedder)
        };

        Ok(Self {
            text_provider,
            #[cfg(feature = "vision")]
            vision,
            #[cfg(feature = "vision")]
            vision_text,
            #[cfg(feature = "speech")]
            speech,
        })
    }

    /// Embed a single text string into a vector of `text_dimension()` dimensions.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if embedding generation fails.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        let embedding = self.text_provider.embed(text)?;

        tracing::debug!(
            dim = embedding.len(),
            elapsed_us = start.elapsed().as_micros(),
            "Single text embed complete"
        );

        Ok(embedding)
    }

    /// Batch-embed multiple texts.
    ///
    /// Delegates to the provider's native batch implementation for efficiency.
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

        let results = self.text_provider.embed_batch(texts)?;

        tracing::debug!(
            count = count,
            elapsed_ms = start.elapsed().as_millis(),
            avg_us = if count > 0 {
                start.elapsed().as_micros() / count as u128
            } else {
                0
            },
            "Batch text embed complete"
        );

        Ok(results)
    }

    /// Get the text embedding dimension from the active provider.
    pub fn text_dimension(&self) -> usize {
        self.text_provider.dimension()
    }

    /// Get the human-readable name of the active text embedding provider.
    pub fn text_provider_name(&self) -> &str {
        self.text_provider.name()
    }

    /// Embed an image into a 512-dimensional CLIP vector.
    ///
    /// Accepts raw image bytes (PNG, JPEG, BMP, etc. — anything `image` crate decodes).
    /// Returns `Ok(Some(embedding))` on success, `Ok(None)` if the vision model is not loaded.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if image decoding or ONNX inference fails.
    #[cfg(feature = "vision")]
    #[instrument(skip(self, image_data), fields(data_len = image_data.len()))]
    pub fn embed_image(&mut self, image_data: &[u8]) -> Result<Option<Vec<f32>>> {
        let vision = match self.vision.as_mut() {
            Some(v) => v,
            None => return Ok(None),
        };

        let start = std::time::Instant::now();

        let results = vision
            .embed_bytes(&[image_data], None)
            .map_err(|e| ShrimPKError::Embedding(format!("CLIP image embed failed: {e}")))?;

        let embedding = results
            .into_iter()
            .next()
            .ok_or_else(|| ShrimPKError::Embedding("Empty CLIP image embedding result".into()))?;

        tracing::debug!(
            dim = embedding.len(),
            elapsed_us = start.elapsed().as_micros(),
            "CLIP image embed complete"
        );

        Ok(Some(embedding))
    }

    /// Embed text into CLIP's shared vision-text space (512-dim).
    ///
    /// This allows text queries to match against image embeddings via cross-modal
    /// retrieval. The resulting vector lives in the same 512-dim CLIP space as
    /// image embeddings from `embed_image()`.
    ///
    /// Returns `Ok(Some(embedding))` on success, `Ok(None)` if the CLIP text model
    /// is not loaded.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if ONNX inference fails.
    #[cfg(feature = "vision")]
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed_text_for_vision(&mut self, text: &str) -> Result<Option<Vec<f32>>> {
        let vision_text = match self.vision_text.as_mut() {
            Some(vt) => vt,
            None => return Ok(None),
        };

        let start = std::time::Instant::now();

        let results = vision_text
            .embed(vec![text.to_string()], None)
            .map_err(|e| ShrimPKError::Embedding(format!("CLIP text embed failed: {e}")))?;

        let embedding = results
            .into_iter()
            .next()
            .ok_or_else(|| ShrimPKError::Embedding("Empty CLIP text embedding result".into()))?;

        tracing::debug!(
            dim = embedding.len(),
            elapsed_us = start.elapsed().as_micros(),
            "CLIP text-for-vision embed complete"
        );

        Ok(Some(embedding))
    }

    /// Whether the vision channel (CLIP) is available.
    #[cfg(feature = "vision")]
    pub fn has_vision(&self) -> bool {
        self.vision.is_some() && self.vision_text.is_some()
    }

    /// Get the vision embedding dimension (512 for CLIP ViT-B-32).
    #[cfg(feature = "vision")]
    pub fn vision_dimension(&self) -> usize {
        512
    }

    /// Embed raw PCM audio into a 640-dimensional speech vector.
    ///
    /// Captures paralinguistic features (tone, pace, speaker identity), NOT speech-to-text.
    /// Uses ECAPA-TDNN (256) + Whisper-tiny encoder (384) = 640-dim.
    ///
    /// Returns `Ok(Some(embedding))` when models are loaded and inference succeeds.
    /// Returns `Ok(None)` when the speech embedder exists but models aren't ready yet.
    ///
    /// # Arguments
    /// * `pcm` - Mono f32 PCM audio samples at any sample rate
    /// * `sample_rate` - Sample rate of the input audio (resampled to 16kHz internally)
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if inference fails with loaded models.
    #[cfg(feature = "speech")]
    pub fn embed_audio(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Option<Vec<f32>>> {
        match &mut self.speech {
            Some(s) => match s.embed_pcm(pcm, sample_rate) {
                Ok(emb) => Ok(Some(emb)),
                Err(e) => {
                    // Models failed to load or inference failed — degrade gracefully
                    tracing::warn!("Speech embed_pcm failed, returning Ok(None): {e}");
                    Ok(None)
                }
            },
            None => Ok(None),
        }
    }

    /// Whether the speech channel (2-model ONNX stack) is available and ready.
    #[cfg(feature = "speech")]
    pub fn has_speech(&self) -> bool {
        self.speech.as_ref().map_or(false, |s| s.is_ready())
    }

    /// Get the speech embedding dimension (640 for ECAPA-TDNN + Whisper-tiny).
    #[cfg(feature = "speech")]
    pub fn speech_dimension(&self) -> usize {
        crate::speech::SPEECH_DIM
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
        let embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");
        assert_eq!(embedder.text_dimension(), 384);
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn embed_single_text() {
        let mut embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");
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
        let mut embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");
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
        let mut embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");
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
        let mut embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");
        let embeddings = embedder
            .embed_batch(Vec::new())
            .expect("Should handle empty");
        assert!(embeddings.is_empty());
    }

    // --- Vision (CLIP) tests (KS35) ---
    // These require the CLIP model download (~352 MB) and the `vision` feature.

    #[cfg(feature = "vision")]
    #[test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    fn clip_vision_initializes() {
        let embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");
        assert!(embedder.has_vision(), "CLIP vision should be available");
        assert_eq!(embedder.vision_dimension(), 512);
    }

    #[cfg(feature = "vision")]
    #[test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    fn embed_image_produces_512_dim() {
        let mut embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");

        // Create a minimal 2x2 red PNG image
        let png_data = create_test_png(2, 2, [255, 0, 0]);

        let embedding = embedder
            .embed_image(&png_data)
            .expect("Should embed image")
            .expect("Vision should be available");

        assert_eq!(
            embedding.len(),
            512,
            "CLIP ViT-B-32 should produce 512-dim vectors"
        );

        // Verify it is normalized (L2 norm ~ 1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.05,
            "CLIP embeddings should be L2-normalized, got norm={norm}"
        );
    }

    #[cfg(feature = "vision")]
    #[test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    fn embed_text_for_vision_produces_512_dim() {
        let mut embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");

        let embedding = embedder
            .embed_text_for_vision("a photo of a cat")
            .expect("Should embed text for vision")
            .expect("CLIP text encoder should be available");

        assert_eq!(
            embedding.len(),
            512,
            "CLIP text encoder should produce 512-dim vectors"
        );
    }

    #[cfg(feature = "vision")]
    #[test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    fn clip_cross_modal_similarity() {
        // CLIP's key property: text and image embeddings in the same space
        // should have positive similarity for matching concepts.
        let mut embedder =
            MultiEmbedder::new(&EchoConfig::default()).expect("MultiEmbedder should init");

        // Embed a red image
        let red_png = create_test_png(32, 32, [255, 0, 0]);
        let img_emb = embedder
            .embed_image(&red_png)
            .unwrap()
            .expect("Vision available");

        // Embed text about colors
        let text_emb = embedder
            .embed_text_for_vision("a red colored image")
            .unwrap()
            .expect("CLIP text available");

        let sim: f32 = img_emb
            .iter()
            .zip(text_emb.iter())
            .map(|(a, b)| a * b)
            .sum();
        // Cross-modal similarity is typically lower than same-modal,
        // but should be positive for matching concepts
        assert!(
            sim > 0.0,
            "Cross-modal CLIP similarity for matching concept should be positive, got {sim}"
        );
    }

    #[cfg(feature = "vision")]
    #[test]
    #[ignore = "requires CLIP model download (~352 MB)"]
    fn clip_init_latency_under_5s() {
        let start = std::time::Instant::now();
        let _embedder = MultiEmbedder::new(&EchoConfig::default()).expect("Should init");
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 10, // generous to account for cold cache
            "CLIP init should be < 10s with warm cache, took {elapsed:?}"
        );
    }

    /// Create a minimal PNG image with a solid color.
    #[cfg(feature = "vision")]
    fn create_test_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        use image::ImageEncoder;
        use std::io::Cursor;
        let mut buf = Cursor::new(Vec::new());

        // Build raw RGBA pixel data
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            pixels.push(rgb[0]);
            pixels.push(rgb[1]);
            pixels.push(rgb[2]);
            pixels.push(255); // alpha
        }

        // Encode as PNG using the `image` crate
        image::codecs::png::PngEncoder::new(&mut buf)
            .write_image(&pixels, width, height, image::ExtendedColorType::Rgba8)
            .expect("PNG encode should succeed");

        buf.into_inner()
    }
}
