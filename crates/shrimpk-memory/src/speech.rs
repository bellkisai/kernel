//! Speech embedding via ONNX models for paralinguistic features.
//!
//! Captures HOW things were said (tone, volume, emotion, pace, speaker ID),
//! NOT speech-to-text. Three specialized models produce a concatenated
//! 899-dim embedding: ECAPA-TDNN (512) + Wav2Small (3) + Whisper-tiny (384).
//!
//! Input: 16kHz mono PCM f32 audio samples.
//! Output: concatenated `[speaker_512 | emotion_3 | prosody_384]`.
//!
//! # Architecture (ADR-014 D8)
//!
//! | Model        | Purpose                          | Dim | ONNX Size |
//! |-------------|----------------------------------|-----|-----------|
//! | ECAPA-TDNN  | Speaker identification (who)     | 512 | ~25 MB    |
//! | Wav2Small   | Emotion (arousal/dom/valence)     |   3 | ~120 KB   |
//! | Whisper-tiny| Prosody (rhythm/stress/pace)      | 384 | ~33 MB    |

use shrimpk_core::{Result, ShrimPKError};

/// Total speech embedding dimension: 512 + 3 + 384.
pub const SPEECH_DIM: usize = 899;

/// Per-model dimensions.
pub const SPEAKER_DIM: usize = 512;
pub const EMOTION_DIM: usize = 3;
pub const PROSODY_DIM: usize = 384;

/// Expected sample rate for all speech models (16 kHz mono).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Configuration for speech model paths.
#[derive(Debug, Clone, Default)]
pub struct SpeechConfig {
    /// Path to the ECAPA-TDNN ONNX model for speaker identification.
    pub speaker_model_path: Option<String>,
    /// Path to the Wav2Small ONNX model for emotion embedding.
    pub emotion_model_path: Option<String>,
    /// Path to the Whisper-tiny encoder ONNX model for prosody embedding.
    pub prosody_model_path: Option<String>,
}

/// Speech embedder combining 3 specialized ONNX models.
///
/// Produces a 899-dim embedding capturing paralinguistic features:
/// - Speaker identity (ECAPA-TDNN, 512-dim)
/// - Emotion (Wav2Small, 3-dim: arousal / dominance / valence)
/// - Prosody (Whisper-tiny encoder, 384-dim: rhythm / stress / pace)
///
/// When models are not loaded (`is_ready() == false`), `embed()` returns
/// a descriptive error. The struct still initializes — KS37+ will plug in
/// real ONNX sessions.
pub struct SpeechEmbedder {
    speaker_dim: usize,
    emotion_dim: usize,
    prosody_dim: usize,
    initialized: bool,
    // KS37+: add ort::Session fields here for the 3 models
}

impl Default for SpeechEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechEmbedder {
    /// Initialize the speech embedder.
    ///
    /// For KS36 the structure is correct but no ONNX models are loaded.
    /// `is_ready()` will return `false` until real model sessions are added.
    pub fn new() -> Self {
        tracing::info!(
            speaker_dim = SPEAKER_DIM,
            emotion_dim = EMOTION_DIM,
            prosody_dim = PROSODY_DIM,
            total_dim = SPEECH_DIM,
            "SpeechEmbedder initialized (model loading deferred to KS37+)"
        );
        Self {
            speaker_dim: SPEAKER_DIM,
            emotion_dim: EMOTION_DIM,
            prosody_dim: PROSODY_DIM,
            initialized: false, // true when real ONNX sessions are loaded
        }
    }

    /// Initialize from explicit model paths.
    ///
    /// Attempts to load ONNX sessions for all 3 models. If any model path
    /// is `None` or fails to load, marks the embedder as not ready.
    /// This is the entry point for KS37+ real model loading.
    pub fn from_config(config: &SpeechConfig) -> Self {
        let embedder = Self::new();

        // KS37+: load ort::Session from each path
        // For now, just log what we would load.
        if config.speaker_model_path.is_some()
            && config.emotion_model_path.is_some()
            && config.prosody_model_path.is_some()
        {
            tracing::info!(
                speaker = config.speaker_model_path.as_deref().unwrap_or(""),
                emotion = config.emotion_model_path.as_deref().unwrap_or(""),
                prosody = config.prosody_model_path.as_deref().unwrap_or(""),
                "Speech model paths provided — loading deferred to KS37+"
            );
            // embedder.initialized = true; // uncomment when real loading works
        }

        embedder
    }

    /// Total embedding dimension (512 + 3 + 384 = 899).
    pub fn dimension(&self) -> usize {
        self.speaker_dim + self.emotion_dim + self.prosody_dim
    }

    /// Speaker sub-embedding dimension (512).
    pub fn speaker_dimension(&self) -> usize {
        self.speaker_dim
    }

    /// Emotion sub-embedding dimension (3).
    pub fn emotion_dimension(&self) -> usize {
        self.emotion_dim
    }

    /// Prosody sub-embedding dimension (384).
    pub fn prosody_dimension(&self) -> usize {
        self.prosody_dim
    }

    /// Whether real ONNX models are loaded and ready for inference.
    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// Embed raw PCM audio into a 899-dim vector.
    ///
    /// Input: mono f32 PCM samples at any sample rate (resampled to 16 kHz internally).
    /// Output: concatenated `[speaker_512 | emotion_3 | prosody_384]`.
    ///
    /// # Errors
    /// Returns `ShrimPKError::Embedding` if models aren't loaded yet or inference fails.
    pub fn embed(&self, pcm_f32: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(ShrimPKError::Embedding(
                "Speech models not loaded. Install ONNX models or enable speech feature with model paths.".into()
            ));
        }

        if pcm_f32.is_empty() {
            return Err(ShrimPKError::Embedding(
                "Empty audio input — cannot embed zero samples".into(),
            ));
        }

        // Resample to 16 kHz if needed
        let _samples = if sample_rate != TARGET_SAMPLE_RATE {
            resample_linear(pcm_f32, sample_rate, TARGET_SAMPLE_RATE)
        } else {
            pcm_f32.to_vec()
        };

        // KS37+: Run through 3 ONNX sessions, normalize, and concatenate:
        //   let mut speaker_emb = session_speaker.run(&samples)?;  // 512-dim
        //   let mut emotion_emb = session_emotion.run(&samples)?;  // 3-dim
        //   let mut prosody_emb = session_prosody.run(&samples)?;  // 384-dim
        //
        //   // L2-normalize each sub-embedding before concat to prevent
        //   // the 384-dim prosody from dominating cosine similarity
        //   // over the 3-dim emotion vector.
        //   l2_normalize(&mut speaker_emb);
        //   l2_normalize(&mut emotion_emb);
        //   l2_normalize(&mut prosody_emb);
        //
        //   let mut combined = Vec::with_capacity(SPEECH_DIM);
        //   combined.extend_from_slice(&speaker_emb);
        //   combined.extend_from_slice(&emotion_emb);
        //   combined.extend_from_slice(&prosody_emb);
        //   Ok(combined)

        Err(ShrimPKError::Embedding(
            "Speech model inference not yet implemented (KS37+)".into(),
        ))
    }
}

/// L2-normalize a vector in-place.
///
/// Prevents high-dimensional sub-embeddings (e.g. 384-dim prosody) from
/// dominating cosine similarity over low-dimensional ones (e.g. 3-dim emotion)
/// when concatenated into the composite speech embedding.
///
/// No-op if the vector norm is near zero (< 1e-10) to avoid division by zero.
/// Used by `embed()` (KS37+) to normalize each sub-embedding before concat.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Simple linear resampling (placeholder until rubato crate is added).
///
/// Converts PCM audio from one sample rate to another using linear interpolation.
/// Suitable for prototyping; production should use `rubato` for band-limited
/// resampling that avoids aliasing.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;
        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac
        } else {
            samples[src_idx.min(samples.len() - 1)]
        };
        output.push(sample);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_embedder_dimensions() {
        let embedder = SpeechEmbedder::new();
        assert_eq!(embedder.dimension(), 899);
        assert_eq!(embedder.speaker_dimension(), 512);
        assert_eq!(embedder.emotion_dimension(), 3);
        assert_eq!(embedder.prosody_dimension(), 384);
    }

    #[test]
    fn speech_embedder_constants_match() {
        assert_eq!(SPEECH_DIM, SPEAKER_DIM + EMOTION_DIM + PROSODY_DIM);
        assert_eq!(SPEECH_DIM, 899);
        assert_eq!(TARGET_SAMPLE_RATE, 16_000);
    }

    #[test]
    fn speech_embedder_not_ready_by_default() {
        let embedder = SpeechEmbedder::new();
        assert!(!embedder.is_ready());
    }

    #[test]
    fn embed_returns_error_when_not_initialized() {
        let embedder = SpeechEmbedder::new();
        let samples = vec![0.0f32; 16000]; // 1 second of silence
        let result = embedder.embed(&samples, 16000);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Speech models not loaded"),
            "Expected 'not loaded' error, got: {err_msg}"
        );
    }

    #[test]
    fn embed_rejects_empty_audio() {
        // Even if initialized, empty input should be rejected.
        // Since the embedder is not initialized, we get the "not loaded" error first.
        // This test ensures the code path doesn't panic on empty input.
        let embedder = SpeechEmbedder::new();
        let result = embedder.embed(&[], 16000);
        assert!(result.is_err());
    }

    #[test]
    fn from_config_default() {
        let config = SpeechConfig::default();
        let embedder = SpeechEmbedder::from_config(&config);
        assert!(!embedder.is_ready());
        assert_eq!(embedder.dimension(), 899);
    }

    #[test]
    fn from_config_with_paths_still_not_ready() {
        // Paths are provided but models aren't actually loaded yet (KS37+)
        let config = SpeechConfig {
            speaker_model_path: Some("models/ecapa_tdnn.onnx".into()),
            emotion_model_path: Some("models/wav2small.onnx".into()),
            prosody_model_path: Some("models/whisper_tiny_encoder.onnx".into()),
        };
        let embedder = SpeechEmbedder::from_config(&config);
        assert!(!embedder.is_ready()); // real loading in KS37+
    }

    // --- resample_linear tests ---

    #[test]
    fn resample_identity() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let output = resample_linear(&samples, 16000, 16000);
        assert_eq!(output, samples, "Same rate should return identical samples");
    }

    #[test]
    fn resample_empty() {
        let output = resample_linear(&[], 48000, 16000);
        assert!(output.is_empty(), "Empty input should return empty output");
    }

    #[test]
    fn resample_downsample_48k_to_16k() {
        // 48kHz -> 16kHz is a 3:1 ratio, so output length ~ input_len / 3
        let input_len = 4800; // 100ms of 48kHz audio
        let samples: Vec<f32> = (0..input_len).map(|i| (i as f32).sin()).collect();
        let output = resample_linear(&samples, 48000, 16000);

        let expected_len = (input_len as f64 * 16000.0 / 48000.0) as usize;
        assert_eq!(
            output.len(),
            expected_len,
            "48kHz -> 16kHz should produce ~{expected_len} samples, got {}",
            output.len()
        );
    }

    #[test]
    fn resample_upsample_8k_to_16k() {
        // 8kHz -> 16kHz is a 1:2 ratio, so output length ~ input_len * 2
        let input_len = 800; // 100ms of 8kHz audio
        let samples: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = resample_linear(&samples, 8000, 16000);

        let expected_len = (input_len as f64 * 16000.0 / 8000.0) as usize;
        assert_eq!(
            output.len(),
            expected_len,
            "8kHz -> 16kHz should produce ~{expected_len} samples, got {}",
            output.len()
        );
    }

    #[test]
    fn resample_preserves_dc_value() {
        // A constant-value signal should stay constant after resampling
        let samples = vec![0.5f32; 1000];
        let output = resample_linear(&samples, 44100, 16000);
        for (i, &s) in output.iter().enumerate() {
            assert!(
                (s - 0.5).abs() < 1e-6,
                "DC signal should be preserved at index {i}, got {s}"
            );
        }
    }

    #[test]
    fn resample_single_sample_upsample() {
        // Upsampling a single sample should produce at least 1 output
        let samples = vec![0.42];
        let output = resample_linear(&samples, 8000, 16000);
        assert!(
            !output.is_empty(),
            "Single sample upsample should produce output"
        );
    }

    #[test]
    fn resample_single_sample_downsample() {
        // Downsampling a single sample with ratio < 1 produces 0 samples
        // (1 * 16000/48000 truncates to 0). This is correct behavior for
        // a single sample at 3:1 ratio — there's not enough data.
        let samples = vec![0.42];
        let output = resample_linear(&samples, 48000, 16000);
        assert_eq!(
            output.len(),
            0,
            "1 sample at 3:1 downsample should produce 0 samples"
        );
    }

    // --- serde roundtrip for MemoryEntry with speech_embedding ---

    #[test]
    fn memory_entry_speech_embedding_roundtrip() {
        use shrimpk_core::{MemoryEntry, Modality};

        let mut entry = MemoryEntry::new_with_modality(
            "[audio]".into(),
            Vec::new(), // no text embedding
            "test".into(),
            Modality::Speech,
        );
        entry.speech_embedding = Some(vec![0.1; 899]);

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.modality, Modality::Speech);
        assert_eq!(deserialized.content, "[audio]");
        assert!(deserialized.embedding.is_empty());
        let speech_emb = deserialized.speech_embedding.unwrap();
        assert_eq!(speech_emb.len(), 899);
        assert!((speech_emb[0] - 0.1).abs() < 1e-6);
    }
}
