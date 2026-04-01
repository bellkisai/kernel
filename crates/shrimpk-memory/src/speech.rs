//! Speech embedding via ONNX models for paralinguistic features.
//!
//! Captures HOW things were said (tone, pace, speaker identity),
//! NOT speech-to-text. Two specialized models produce a concatenated
//! 896-dim embedding: ECAPA-TDNN (512) + Whisper-tiny encoder (384).
//!
//! Input: 16kHz mono PCM f32 audio samples.
//! Output: concatenated `[speaker_512 | prosody_384]`, both L2-normalized.
//!
//! # Architecture (KS50)
//!
//! | Model        | Purpose                          | Dim | License   | ONNX Size |
//! |-------------|----------------------------------|-----|-----------|-----------|
//! | ECAPA-TDNN  | Speaker identification (who)     | 512 | Apache-2.0| ~25 MB    |
//! | Whisper-tiny| Prosody (rhythm/stress/pace)      | 384 | MIT       | ~33 MB    |
//!
//! The Wav2Small emotion channel (3-dim) was dropped (CC-BY-NC-SA incompatible).
//! See CHANGELOG v0.6.0 for the architecture decision.
//!
//! # Model Sources
//! - ECAPA-TDNN: `Wespeaker/wespeaker-cnceleb-resnet34-LM` (Apache-2.0)
//!   <https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM>
//! - Whisper-tiny encoder: `onnx-community/whisper-tiny` (MIT / OpenAI Whisper License)
//!   <https://huggingface.co/onnx-community/whisper-tiny>
//!
//! # Cache
//! Models are stored in `~/.shrimpk-kernel/models/` and downloaded on first use.

use shrimpk_core::{Result, ShrimPKError};

/// Total speech embedding dimension: 512 (ECAPA-TDNN) + 384 (Whisper-tiny) = 896.
pub const SPEECH_DIM: usize = 896;

/// Speaker sub-embedding dimension (ECAPA-TDNN).
pub const SPEAKER_DIM: usize = 512;

/// Prosody sub-embedding dimension (Whisper-tiny encoder mean-pool).
pub const PROSODY_DIM: usize = 384;

/// Expected sample rate for all speech models (16 kHz mono).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Configuration for speech model paths (overrides auto-download).
#[derive(Debug, Clone, Default)]
pub struct SpeechConfig {
    /// Path to the ECAPA-TDNN ONNX model for speaker identification.
    /// If `None`, the model is auto-downloaded from HuggingFace on first use.
    pub speaker_model_path: Option<String>,
    /// Path to the Whisper-tiny encoder ONNX model for prosody embedding.
    /// If `None`, the model is auto-downloaded from HuggingFace on first use.
    pub prosody_model_path: Option<String>,
}

// ============================================================================
// Feature-gated implementation (real ONNX inference)
// ============================================================================

#[cfg(feature = "speech")]
mod inner {
    use super::*;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;

    // Whisper mel-spectrogram constants (standard Whisper preprocessing)
    const N_MELS: usize = 80;
    const N_FFT: usize = 400;
    const HOP_LENGTH: usize = 160;
    // Whisper-tiny expects exactly 3000 frames (30s window at 16kHz / hop_length=160)
    const N_FRAMES: usize = 3000;

    // HuggingFace model references
    // ECAPA-TDNN — Apache-2.0 license
    // https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM
    const SPEAKER_REPO: &str = "Wespeaker/wespeaker-cnceleb-resnet34-LM";
    const SPEAKER_FILENAME: &str = "wespeaker-cnceleb-resnet34-LM.onnx";

    // Whisper-tiny encoder — MIT license / OpenAI Whisper License
    // https://huggingface.co/onnx-community/whisper-tiny
    const PROSODY_REPO: &str = "onnx-community/whisper-tiny";
    const PROSODY_FILENAME: &str = "encoder_model.onnx";

    // -------------------------------------------------------------------------
    // Model cache + download
    // -------------------------------------------------------------------------

    /// Download a model file from HuggingFace Hub if not already cached.
    ///
    /// Uses `hf-hub`'s built-in caching — files are stored in the HF Hub cache
    /// (typically `~/.cache/huggingface/hub/`).
    fn ensure_model(repo_id: &str, filename: &str) -> Result<std::path::PathBuf> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let api = Api::new().map_err(|e| {
            ShrimPKError::Embedding(format!("Failed to create HF Hub API client: {e}"))
        })?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        let path = repo.get(filename).map_err(|e| {
            ShrimPKError::Embedding(format!(
                "Failed to download {filename} from {repo_id}: {e}"
            ))
        })?;
        Ok(path)
    }

    /// Build an ORT `Session` from a local model file path.
    ///
    /// Uses Level3 optimization (all optimizations enabled by default in ort rc.11).
    fn build_session(path: &std::path::Path) -> Result<Session> {
        Session::builder()
            .map_err(|e| ShrimPKError::Embedding(format!("ORT session builder failed: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ShrimPKError::Embedding(format!("ORT opt level failed: {e}")))?
            .commit_from_file(path)
            .map_err(|e| {
                ShrimPKError::Embedding(format!(
                    "ORT failed to load model from {}: {e}",
                    path.display()
                ))
            })
    }

    // -------------------------------------------------------------------------
    // Mel-spectrogram computation (Whisper preprocessing)
    // -------------------------------------------------------------------------

    /// Apply a Hann window to a PCM frame.
    fn hann_window(frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        frame
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos());
                s * w
            })
            .collect()
    }

    /// Real DFT — power spectrum for first N/2+1 bins (O(N²), fine for N=400).
    fn power_spectrum(frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        let n_out = n / 2 + 1;
        let mut power = vec![0.0f32; n_out];
        for k in 0..n_out {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (j, &s) in frame.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
                re += s * angle.cos();
                im += s * angle.sin();
            }
            power[k] = re * re + im * im;
        }
        power
    }

    /// Build mel filterbank matrix [n_mels × (N_FFT/2+1)] using HTK mel scale.
    fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f32>> {
        let n_bins = n_fft / 2 + 1;
        let sr = sample_rate as f32;
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(sr / 2.0);

        let mel_points: Vec<f32> = (0..=(n_mels + 1))
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft + 1) as f32 * hz / sr).floor() as usize)
            .collect();

        let mut filters = vec![vec![0.0f32; n_bins]; n_mels];
        for m in 0..n_mels {
            let start = bin_points[m];
            let peak = bin_points[m + 1];
            let end = bin_points[m + 2];
            for k in start..peak {
                if k < n_bins && peak > start {
                    filters[m][k] = (k - start) as f32 / (peak - start) as f32;
                }
            }
            for k in peak..end {
                if k < n_bins && end > peak {
                    filters[m][k] = (end - k) as f32 / (end - peak) as f32;
                }
            }
        }
        filters
    }

    /// Compute Whisper-style log-Mel spectrogram.
    ///
    /// Returns a flat Vec of shape `[1, N_MELS, N_FRAMES]` (row-major).
    /// Audio shorter than the full window is zero-padded; longer audio is truncated.
    fn compute_log_mel_flat(pcm: &[f32]) -> Vec<f32> {
        let filters = mel_filterbank(N_MELS, N_FFT, TARGET_SAMPLE_RATE);
        let mut mel_spec = vec![0.0f32; N_MELS * N_FRAMES]; // [N_MELS, N_FRAMES] row-major

        for frame_idx in 0..N_FRAMES {
            let start = frame_idx * HOP_LENGTH;
            let end = start + N_FFT;

            // Zero-pad frame if it extends beyond the signal
            let mut frame = vec![0.0f32; N_FFT];
            if start < pcm.len() {
                let copy_end = end.min(pcm.len());
                frame[..copy_end - start].copy_from_slice(&pcm[start..copy_end]);
            }

            let windowed = hann_window(&frame);
            let power = power_spectrum(&windowed);

            for (m_idx, filter) in filters.iter().enumerate() {
                let mel_val: f32 = filter.iter().zip(power.iter()).map(|(f, p)| f * p).sum();
                mel_spec[m_idx * N_FRAMES + frame_idx] = mel_val.max(1e-10).log10();
            }
        }

        // Whisper normalization: clamp to (max - 8), then scale to [-1, 1] range
        let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in mel_spec.iter_mut() {
            *v = ((*v).max(max_val - 8.0) + 4.0) / 8.0;
        }

        // Return as [1, N_MELS, N_FRAMES] flat
        let mut batched = vec![0.0f32; 1 * N_MELS * N_FRAMES];
        batched.copy_from_slice(&mel_spec);
        batched
    }

    // -------------------------------------------------------------------------
    // L2 normalization
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // SpeechEmbedder — real ONNX implementation
    // -------------------------------------------------------------------------

    /// Speech embedder backed by two ONNX sessions.
    pub struct SpeechEmbedder {
        /// ECAPA-TDNN ORT session → 512-dim speaker embedding (Apache-2.0).
        speaker_session: Option<Session>,
        /// Whisper-tiny encoder ORT session → 384-dim prosody embedding (MIT).
        prosody_session: Option<Session>,
    }

    impl SpeechEmbedder {
        /// Create a new embedder. Models are NOT loaded — call `load_models()` first
        /// or let `embed_pcm()` auto-load on first call.
        pub fn new() -> Self {
            tracing::info!(
                speaker_dim = SPEAKER_DIM,
                prosody_dim = PROSODY_DIM,
                total_dim = SPEECH_DIM,
                "SpeechEmbedder created (speech feature enabled, models deferred)"
            );
            Self {
                speaker_session: None,
                prosody_session: None,
            }
        }

        /// Initialize from explicit local model paths (testing / offline use).
        pub fn from_config(config: &SpeechConfig) -> Self {
            let mut embedder = Self::new();

            if let Some(speaker_path) = &config.speaker_model_path {
                match build_session(std::path::Path::new(speaker_path)) {
                    Ok(session) => {
                        tracing::info!(path = speaker_path, "ECAPA-TDNN session loaded from config path");
                        embedder.speaker_session = Some(session);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load ECAPA-TDNN from {speaker_path}: {e}");
                    }
                }
            }

            if let Some(prosody_path) = &config.prosody_model_path {
                match build_session(std::path::Path::new(prosody_path)) {
                    Ok(session) => {
                        tracing::info!(path = prosody_path, "Whisper-tiny encoder session loaded");
                        embedder.prosody_session = Some(session);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load Whisper-tiny from {prosody_path}: {e}");
                    }
                }
            }

            embedder
        }

        /// Download models from HuggingFace Hub (if not cached) and load ORT sessions.
        ///
        /// Models are cached in the HF Hub cache after first download.
        /// `embed_pcm()` calls this automatically on first use.
        ///
        /// # Errors
        /// Returns `ShrimPKError::Embedding` if any model fails to load.
        pub fn load_models(&mut self) -> Result<()> {
            // ECAPA-TDNN — Apache-2.0
            // https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM
            if self.speaker_session.is_none() {
                let start = std::time::Instant::now();
                tracing::info!(
                    repo = SPEAKER_REPO,
                    file = SPEAKER_FILENAME,
                    "Downloading ECAPA-TDNN (Apache-2.0) from HuggingFace Hub"
                );
                match ensure_model(SPEAKER_REPO, SPEAKER_FILENAME) {
                    Ok(path) => match build_session(&path) {
                        Ok(session) => {
                            tracing::info!(
                                elapsed_ms = start.elapsed().as_millis(),
                                path = %path.display(),
                                "ECAPA-TDNN session ready"
                            );
                            self.speaker_session = Some(session);
                        }
                        Err(e) => tracing::warn!("ECAPA-TDNN session build failed: {e}"),
                    },
                    Err(e) => tracing::warn!("ECAPA-TDNN download failed: {e}"),
                }
            }

            // Whisper-tiny encoder — MIT license
            // https://huggingface.co/onnx-community/whisper-tiny
            if self.prosody_session.is_none() {
                let start = std::time::Instant::now();
                tracing::info!(
                    repo = PROSODY_REPO,
                    file = PROSODY_FILENAME,
                    "Downloading Whisper-tiny encoder (MIT) from HuggingFace Hub"
                );
                match ensure_model(PROSODY_REPO, PROSODY_FILENAME) {
                    Ok(path) => match build_session(&path) {
                        Ok(session) => {
                            tracing::info!(
                                elapsed_ms = start.elapsed().as_millis(),
                                path = %path.display(),
                                "Whisper-tiny encoder session ready"
                            );
                            self.prosody_session = Some(session);
                        }
                        Err(e) => tracing::warn!("Whisper-tiny session build failed: {e}"),
                    },
                    Err(e) => tracing::warn!("Whisper-tiny download failed: {e}"),
                }
            }

            if self.speaker_session.is_none() || self.prosody_session.is_none() {
                return Err(ShrimPKError::Embedding(
                    "One or more speech models failed to load. \
                     Check network connectivity or provide model paths via SpeechConfig."
                        .into(),
                ));
            }

            Ok(())
        }

        /// Whether both ONNX sessions are loaded and ready for inference.
        pub fn is_ready(&self) -> bool {
            self.speaker_session.is_some() && self.prosody_session.is_some()
        }

        fn ensure_loaded(&mut self) -> Result<()> {
            if !self.is_ready() {
                self.load_models()?;
            }
            Ok(())
        }

        /// Run ECAPA-TDNN → 512-dim speaker embedding.
        ///
        /// Input: `[1, T]` waveform tensor (f32 PCM at 16kHz).
        fn run_speaker(&mut self, pcm_16k: &[f32]) -> Result<Vec<f32>> {
            let session = self.speaker_session.as_mut().ok_or_else(|| {
                ShrimPKError::Embedding("ECAPA-TDNN session not loaded".into())
            })?;

            // Build input tensor [1, T]
            let t = pcm_16k.len();
            let input_tensor =
                Tensor::<f32>::from_array(([1usize, t], pcm_16k.to_vec().into_boxed_slice()))
                    .map_err(|e| {
                        ShrimPKError::Embedding(format!("ECAPA-TDNN input tensor failed: {e}"))
                    })?;

            // Run inference
            let outputs = session
                .run(ort::inputs!["waveform" => input_tensor])
                .map_err(|e| {
                    ShrimPKError::Embedding(format!("ECAPA-TDNN inference failed: {e}"))
                })?;

            // Extract output — expected [1, 512] or [512]
            let output_val = outputs
                .values()
                .next()
                .ok_or_else(|| ShrimPKError::Embedding("ECAPA-TDNN: no output".into()))?;

            let (_shape, flat) = output_val
                .try_extract_tensor::<f32>()
                .map_err(|e| ShrimPKError::Embedding(format!("ECAPA-TDNN extract failed: {e}")))?;

            // Take last SPEAKER_DIM values in case of batch dim
            if flat.len() < SPEAKER_DIM {
                return Err(ShrimPKError::Embedding(format!(
                    "ECAPA-TDNN output too small: expected >= {SPEAKER_DIM}, got {}",
                    flat.len()
                )));
            }
            Ok(flat[flat.len() - SPEAKER_DIM..].to_vec())
        }

        /// Run Whisper-tiny encoder → 384-dim prosody embedding (mean-pool over time).
        ///
        /// Input: log-Mel spectrogram `[1, 80, 3000]`.
        fn run_prosody(&mut self, pcm_16k: &[f32]) -> Result<Vec<f32>> {
            let session = self.prosody_session.as_mut().ok_or_else(|| {
                ShrimPKError::Embedding("Whisper-tiny encoder session not loaded".into())
            })?;

            // Compute log-Mel → flat [1, N_MELS, N_FRAMES]
            let mel_flat = compute_log_mel_flat(pcm_16k);

            let input_tensor = Tensor::<f32>::from_array((
                [1usize, N_MELS, N_FRAMES],
                mel_flat.into_boxed_slice(),
            ))
            .map_err(|e| {
                ShrimPKError::Embedding(format!("Whisper mel tensor failed: {e}"))
            })?;

            let outputs = session
                .run(ort::inputs!["input_features" => input_tensor])
                .map_err(|e| {
                    ShrimPKError::Embedding(format!("Whisper encoder inference failed: {e}"))
                })?;

            // Extract output — encoder hidden states [1, T', 384] or [T', 384]
            let output_val = outputs
                .values()
                .next()
                .ok_or_else(|| ShrimPKError::Embedding("Whisper encoder: no output".into()))?;

            let (shape, flat) = output_val
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    ShrimPKError::Embedding(format!("Whisper encoder extract failed: {e}"))
                })?;

            // Mean-pool over time dimension to produce [hidden_dim]
            // Shape is [batch, time_steps, hidden_dim] or [time_steps, hidden_dim]
            let prosody_emb = mean_pool_last_dim(flat, shape)?;

            if prosody_emb.len() != PROSODY_DIM {
                return Err(ShrimPKError::Embedding(format!(
                    "Whisper encoder dim mismatch: expected {PROSODY_DIM}, got {}",
                    prosody_emb.len()
                )));
            }

            Ok(prosody_emb)
        }

        /// Embed raw PCM audio into a 896-dim vector.
        ///
        /// Input: mono f32 PCM samples at any sample rate.
        /// Output: L2-normalized `[speaker_512 | prosody_384]`.
        ///
        /// Auto-downloads ONNX models on first call (~58 MB total).
        ///
        /// # Errors
        /// Returns `ShrimPKError::Embedding` if model loading or inference fails.
        pub fn embed_pcm(&mut self, pcm_f32: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
            if pcm_f32.is_empty() {
                return Err(ShrimPKError::Embedding(
                    "Empty audio input — cannot embed zero samples".into(),
                ));
            }

            self.ensure_loaded()?;

            // Resample to 16kHz if needed
            let pcm_16k = if sample_rate != TARGET_SAMPLE_RATE {
                resample_linear(pcm_f32, sample_rate, TARGET_SAMPLE_RATE)
            } else {
                pcm_f32.to_vec()
            };

            // ECAPA-TDNN → 512-dim (L2-normalized)
            let mut speaker_emb = self.run_speaker(&pcm_16k)?;
            super::l2_normalize(&mut speaker_emb);

            // Whisper-tiny encoder → 384-dim (L2-normalized)
            let mut prosody_emb = self.run_prosody(&pcm_16k)?;
            super::l2_normalize(&mut prosody_emb);

            // Concatenate → 896-dim
            let mut combined = Vec::with_capacity(SPEECH_DIM);
            combined.extend_from_slice(&speaker_emb);
            combined.extend_from_slice(&prosody_emb);

            debug_assert_eq!(combined.len(), SPEECH_DIM);

            tracing::debug!(
                dim = combined.len(),
                sample_rate,
                pcm_samples = pcm_f32.len(),
                "Speech embed complete"
            );

            Ok(combined)
        }
    }

    impl Default for SpeechEmbedder {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Mean-pool the last axis of a flat tensor given its shape.
    ///
    /// Handles shapes [batch, time, hidden] and [time, hidden].
    /// Returns `Vec<f32>` of length `hidden_dim`.
    fn mean_pool_last_dim(flat: &[f32], shape: &[i64]) -> Result<Vec<f32>> {
        match shape {
            // [batch, time_steps, hidden_dim] — take batch 0, pool over time
            [_batch, time_steps, hidden_dim] => {
                let t = *time_steps as usize;
                let h = *hidden_dim as usize;
                if flat.len() < t * h {
                    return Err(ShrimPKError::Embedding(format!(
                        "Tensor flat slice too short: expected {} got {}",
                        t * h,
                        flat.len()
                    )));
                }
                // Take batch 0: first t*h elements
                let batch0 = &flat[..t * h];
                let mut mean = vec![0.0f32; h];
                for frame in 0..t {
                    for dim in 0..h {
                        mean[dim] += batch0[frame * h + dim];
                    }
                }
                let t_f = t as f32;
                mean.iter_mut().for_each(|v| *v /= t_f);
                Ok(mean)
            }
            // [time_steps, hidden_dim] — pool over time
            [time_steps, hidden_dim] => {
                let t = *time_steps as usize;
                let h = *hidden_dim as usize;
                if flat.len() < t * h {
                    return Err(ShrimPKError::Embedding(format!(
                        "Tensor flat slice too short: expected {} got {}",
                        t * h,
                        flat.len()
                    )));
                }
                let mut mean = vec![0.0f32; h];
                for frame in 0..t {
                    for dim in 0..h {
                        mean[dim] += flat[frame * h + dim];
                    }
                }
                let t_f = t as f32;
                mean.iter_mut().for_each(|v| *v /= t_f);
                Ok(mean)
            }
            other => Err(ShrimPKError::Embedding(format!(
                "Unexpected Whisper encoder output shape: {other:?}"
            ))),
        }
    }
} // mod inner

// ============================================================================
// Public re-exports — present only with speech feature
// ============================================================================

#[cfg(feature = "speech")]
pub use inner::SpeechEmbedder;

// ============================================================================
// Stub when speech feature is disabled
// ============================================================================

/// Stub speech embedder compiled when `speech` feature is absent.
///
/// All operations return graceful no-ops — no panics, no runtime errors leaked upstream.
#[cfg(not(feature = "speech"))]
pub struct SpeechEmbedder {
    _priv: (),
}

#[cfg(not(feature = "speech"))]
impl Default for SpeechEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "speech"))]
impl SpeechEmbedder {
    pub fn new() -> Self {
        Self { _priv: () }
    }

    pub fn from_config(_config: &SpeechConfig) -> Self {
        Self::new()
    }

    pub fn is_ready(&self) -> bool {
        false
    }

    /// Stub — always errors since ONNX models aren't compiled in.
    pub fn embed_pcm(&self, _pcm_f32: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        Err(ShrimPKError::Embedding(
            "Speech feature not enabled — compile with --features shrimpk-memory/speech".into(),
        ))
    }
}

// ============================================================================
// Shared utilities (always compiled)
// ============================================================================

/// L2-normalize a vector in-place.
///
/// No-op if norm < 1e-10 (prevents division by zero on silent audio).
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Simple linear interpolation resampler.
///
/// Converts PCM from `from_rate` Hz to `to_rate` Hz using linear interpolation.
/// Suitable for offline use where aliasing artefacts are acceptable.
/// For production quality consider replacing with a windowed-sinc or polyphase FIR.
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Constant sanity checks (always run) ---

    #[test]
    fn speech_dim_is_896() {
        assert_eq!(SPEECH_DIM, 896, "SPEECH_DIM must be 896 (512 + 384)");
    }

    #[test]
    fn speech_dims_add_up() {
        assert_eq!(SPEAKER_DIM + PROSODY_DIM, SPEECH_DIM);
    }

    #[test]
    fn target_sample_rate_16k() {
        assert_eq!(TARGET_SAMPLE_RATE, 16_000);
    }

    // --- Stub tests (no feature needed) ---

    #[cfg(not(feature = "speech"))]
    #[test]
    fn stub_not_ready() {
        assert!(!SpeechEmbedder::new().is_ready());
    }

    #[cfg(not(feature = "speech"))]
    #[test]
    fn stub_returns_feature_error() {
        let result = SpeechEmbedder::new().embed_pcm(&vec![0.0f32; 100], 16000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Speech feature not enabled"));
    }

    // --- Feature tests ---

    #[cfg(feature = "speech")]
    #[test]
    fn speech_embedder_not_ready_before_load() {
        let emb = SpeechEmbedder::new();
        assert!(!emb.is_ready());
    }

    // --- l2_normalize ---

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![3.0f32, 4.0]; // norm = 5.0
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut v = vec![0.0f32; 10];
        l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    // --- resample_linear ---

    #[test]
    fn resample_identity() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(resample_linear(&samples, 16000, 16000), samples);
    }

    #[test]
    fn resample_empty() {
        assert!(resample_linear(&[], 48000, 16000).is_empty());
    }

    #[test]
    fn resample_downsample_48k_to_16k() {
        let input_len = 4800;
        let samples: Vec<f32> = (0..input_len).map(|i| (i as f32).sin()).collect();
        let output = resample_linear(&samples, 48000, 16000);
        let expected = (input_len as f64 * 16000.0 / 48000.0) as usize;
        assert_eq!(output.len(), expected);
    }

    #[test]
    fn resample_upsample_8k_to_16k() {
        let input_len = 800;
        let samples: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = resample_linear(&samples, 8000, 16000);
        let expected = (input_len as f64 * 16000.0 / 8000.0) as usize;
        assert_eq!(output.len(), expected);
    }

    #[test]
    fn resample_preserves_dc_value() {
        let samples = vec![0.5f32; 1000];
        let output = resample_linear(&samples, 44100, 16000);
        for (i, &s) in output.iter().enumerate() {
            assert!((s - 0.5).abs() < 1e-6, "DC at index {i}, got {s}");
        }
    }

    #[test]
    fn resample_single_sample_upsample() {
        assert!(!resample_linear(&[0.42], 8000, 16000).is_empty());
    }

    #[test]
    fn resample_single_sample_downsample() {
        assert_eq!(resample_linear(&[0.42], 48000, 16000).len(), 0);
    }

    // --- MemoryEntry roundtrip with 896-dim speech embedding ---

    #[test]
    fn memory_entry_speech_embedding_roundtrip_896() {
        use shrimpk_core::{MemoryEntry, Modality};

        let mut entry = MemoryEntry::new_with_modality(
            "[audio]".into(),
            Vec::new(),
            "test".into(),
            Modality::Speech,
        );
        entry.speech_embedding = Some(vec![0.1; SPEECH_DIM]);

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.modality, Modality::Speech);
        let speech_emb = deserialized.speech_embedding.unwrap();
        assert_eq!(speech_emb.len(), SPEECH_DIM);
        assert!((speech_emb[0] - 0.1).abs() < 1e-6);
    }
}
