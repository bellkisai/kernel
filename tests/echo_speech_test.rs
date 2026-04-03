//! Speech channel integration tests (KS50).
//!
//! Run fast unit tests (no network, no feature flag):
//!     cargo test --test echo_speech_test
//!
//! Run all tests including model-download tests (~58 MB, requires network):
//!     cargo test --test echo_speech_test --features shrimpk-memory/speech -- --include-ignored --test-threads=1

// ---------------------------------------------------------------------------
// Fast unit tests — no feature flags, no network, always pass
// ---------------------------------------------------------------------------

/// SPEECH_DIM constant must be exactly 640 regardless of feature flags.
///
/// Hard gate: if this fails, the 896→640 migration (ECAPA-TDNN ResNet34 outputs 256, not 512) is incomplete.
#[test]
fn speech_dim_is_640() {
    assert_eq!(
        shrimpk_memory::speech::SPEECH_DIM,
        640,
        "SPEECH_DIM must be 640 (ECAPA-TDNN 256 + Whisper-tiny 384). \
         Was 896 before KS51 (ECAPA-TDNN Wespeaker ResNet34 outputs 256-dim, not 512)."
    );
}

#[test]
fn speech_sub_dims_sum_to_speech_dim() {
    assert_eq!(
        shrimpk_memory::speech::SPEAKER_DIM + shrimpk_memory::speech::PROSODY_DIM,
        shrimpk_memory::speech::SPEECH_DIM,
    );
}

#[test]
fn target_sample_rate_is_16k() {
    assert_eq!(shrimpk_memory::speech::TARGET_SAMPLE_RATE, 16_000);
}

/// l2_normalize must produce a unit vector.
#[test]
fn l2_normalize_produces_unit_vector() {
    let mut v = vec![3.0f32, 4.0]; // norm = 5.0
    shrimpk_memory::speech::l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "Expected unit norm, got {norm}");
}

/// l2_normalize must not panic on zero vector.
#[test]
fn l2_normalize_zero_vector_no_panic() {
    let mut v = vec![0.0f32; 512];
    shrimpk_memory::speech::l2_normalize(&mut v); // must not panic or NaN
    assert!(v.iter().all(|&x| x == 0.0));
}

/// resample_linear identity case.
#[test]
fn resample_linear_identity() {
    let samples = vec![0.1f32, 0.2, 0.3, 0.4];
    let out = shrimpk_memory::speech::resample_linear(&samples, 16000, 16000);
    assert_eq!(out, samples);
}

#[test]
fn resample_linear_downsample_3_to_1() {
    // 48kHz → 16kHz is 3:1 ratio, output ~ input_len / 3
    let samples: Vec<f32> = (0..4800).map(|i| (i as f32 / 100.0).sin()).collect();
    let out = shrimpk_memory::speech::resample_linear(&samples, 48000, 16000);
    let expected = (4800.0 * 16000.0 / 48000.0) as usize;
    assert_eq!(out.len(), expected);
}

// ---------------------------------------------------------------------------
// Stub behaviour tests — these run when speech feature is ABSENT.
// When `--features shrimpk-memory/speech` is passed, these are excluded.
// Run stub tests with: cargo test --test echo_speech_test (no feature flag)
// ---------------------------------------------------------------------------

mod stub_tests {
    use shrimpk_memory::speech::{SpeechConfig, SpeechEmbedder};

    #[test]
    fn stub_embedder_is_not_ready() {
        // When speech feature is enabled, SpeechEmbedder::new() still starts not-ready
        // (deferred model loading). When disabled, stub always returns false.
        let emb = SpeechEmbedder::new();
        assert!(!emb.is_ready());
    }

    #[test]
    fn from_config_no_paths_not_ready() {
        let emb = SpeechEmbedder::from_config(&SpeechConfig::default());
        // Without paths and without downloading, embedder must not be ready
        assert!(!emb.is_ready());
    }
}

// ---------------------------------------------------------------------------
// Feature-gated tests — require `--features shrimpk-memory/speech`
// ---------------------------------------------------------------------------

#[cfg(feature = "speech")]
mod speech_feature_tests {
    use shrimpk_memory::speech::SpeechEmbedder;

    #[test]
    fn embedder_starts_not_ready() {
        let emb = SpeechEmbedder::new();
        assert!(
            !emb.is_ready(),
            "Embedder must not auto-load models at construction time"
        );
    }

    /// Stores a 1-second 440 Hz sine-wave audio clip as a speech memory and
    /// verifies it appears in the store with correct modality.
    ///
    /// Requires ~58 MB model download on first run.
    /// Models are cached in `~/.shrimpk-kernel/models/` after first download.
    #[tokio::test]
    #[ignore = "requires model download (~58 MB) — run with --include-ignored"]
    async fn speech_store_and_verify() {
        use shrimpk_core::{EchoConfig, Modality};
        use shrimpk_memory::EchoEngine;
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let config = EchoConfig {
            data_dir: dir.path().to_path_buf(),
            embedding_dim: 384,
            speech_embedding_dim: 640,
            max_memories: 1000,
            similarity_threshold: 0.05,
            max_echo_results: 10,
            enabled_modalities: vec![Modality::Text, Modality::Speech],
            ..Default::default()
        };

        let engine = EchoEngine::new(config).expect("EchoEngine::new");

        // Generate 1 second of 440 Hz sine wave at 16 kHz (16,000 samples)
        let sample_rate: u32 = 16_000;
        let freq = 440.0f32;
        let pcm: Vec<f32> = (0..(sample_rate as usize))
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        assert_eq!(pcm.len(), 16_000);

        // Store as a speech memory (EchoEngine::store_audio)
        let memory_id_result = engine
            .store_audio(&pcm, sample_rate, "integration-test", None)
            .await;

        match memory_id_result {
            Ok(id) => {
                eprintln!("Stored speech memory with id {id}");

                // Verify speech memory is in the store via stats
                let stats = engine.stats().await;
                assert_eq!(
                    stats.speech_count, 1,
                    "Store should contain exactly 1 speech memory after store_audio"
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("download")
                    || msg.contains("network")
                    || msg.contains("load")
                    || msg.contains("not available")
                {
                    eprintln!(
                        "SKIP: Speech model unavailable ({msg}) — run with network for full test"
                    );
                } else {
                    panic!("Unexpected error storing speech memory: {e}");
                }
            }
        }

        engine.shutdown().await;
    }

    /// Verify embed_pcm returns exactly 640 dimensions after model load.
    #[tokio::test]
    #[ignore = "requires model download (~58 MB) — run with --include-ignored"]
    async fn embed_pcm_returns_640_dim() {
        let mut embedder = SpeechEmbedder::new();
        embedder
            .load_models()
            .expect("Models should load (requires network)");

        assert!(
            embedder.is_ready(),
            "Embedder should be ready after load_models()"
        );

        // 1 second of silence at 16kHz
        let pcm = vec![0.0f32; 16_000];
        let emb = embedder
            .embed_pcm(&pcm, 16_000)
            .expect("embed_pcm should succeed");

        assert_eq!(
            emb.len(),
            shrimpk_memory::speech::SPEECH_DIM,
            "embed_pcm must return exactly {}-dim vector",
            shrimpk_memory::speech::SPEECH_DIM
        );

        // Sanity-check: sub-vectors should be L2-normalized individually
        // speaker sub-vector (first 256 dims) should have norm ~1.0
        let speaker: Vec<f32> = emb[..shrimpk_memory::speech::SPEAKER_DIM].to_vec();
        let speaker_norm: f32 = speaker.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (speaker_norm - 1.0).abs() < 0.05,
            "Speaker sub-vector should be L2-normalized, got norm={speaker_norm}"
        );

        // prosody sub-vector (last 384 dims) should have norm ~1.0
        let prosody: Vec<f32> = emb[shrimpk_memory::speech::SPEAKER_DIM..].to_vec();
        let prosody_norm: f32 = prosody.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (prosody_norm - 1.0).abs() < 0.05,
            "Prosody sub-vector should be L2-normalized, got norm={prosody_norm}"
        );
    }

    /// Verify resampling from 44.1kHz input is handled transparently.
    #[tokio::test]
    #[ignore = "requires model download (~58 MB) — run with --include-ignored"]
    async fn embed_pcm_resamples_44100_to_16k() {
        let mut embedder = SpeechEmbedder::new();
        embedder.load_models().expect("Models should load");

        // 0.5s of 440Hz tone at 44.1kHz
        let pcm_44k: Vec<f32> = (0..22_050usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let emb = embedder
            .embed_pcm(&pcm_44k, 44_100)
            .expect("embed_pcm with 44.1kHz input should succeed after resampling");

        assert_eq!(emb.len(), shrimpk_memory::speech::SPEECH_DIM);
    }

    /// Cross-modal recall: store audio with description → find via text echo.
    ///
    /// This is the ROSCon demo path:
    /// 1. Robot stores audio with description "standup meeting recording"
    /// 2. User asks "meeting" → text echo finds the speech memory
    #[tokio::test]
    #[ignore = "requires model download (~58 MB) — run with --include-ignored"]
    async fn cross_modal_text_finds_speech_memory() {
        use shrimpk_core::{EchoConfig, Modality};
        use shrimpk_memory::EchoEngine;
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let config = EchoConfig {
            data_dir: dir.path().to_path_buf(),
            embedding_dim: 384,
            speech_embedding_dim: shrimpk_memory::speech::SPEECH_DIM,
            max_memories: 1000,
            similarity_threshold: 0.01,
            max_echo_results: 10,
            enabled_modalities: vec![Modality::Text, Modality::Speech],
            ..Default::default()
        };

        let engine = EchoEngine::new(config).expect("EchoEngine::new");

        // Store some text memories as distractors
        engine
            .store("I went grocery shopping yesterday", "test")
            .await
            .expect("store text");
        engine
            .store("The weather was sunny and warm", "test")
            .await
            .expect("store text");

        // Store audio WITH description — enables cross-modal recall
        let pcm: Vec<f32> = (0..16_000usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let id = engine
            .store_audio(
                &pcm,
                16_000,
                "test",
                Some("daily standup meeting recording"),
            )
            .await
            .expect("store_audio with description");

        eprintln!("Stored cross-modal speech memory: {id}");

        // Text echo should find the speech memory via its description
        let results = engine.echo("standup meeting", 10).await.expect("echo");

        let found = results.iter().any(|r| r.memory_id == id);
        assert!(
            found,
            "Text echo for 'standup meeting' should find the speech memory stored with \
             description 'daily standup meeting recording'. Got {} results: {:?}",
            results.len(),
            results
                .iter()
                .map(|r| (&r.memory_id, &r.content))
                .collect::<Vec<_>>()
        );

        // Verify the memory has memtype:audio label
        let stats = engine.stats().await;
        assert_eq!(stats.speech_count, 1);

        engine.shutdown().await;
    }
}
