//! Speech channel latency benchmark (KS53).
//!
//! Measures embed_pcm latency, store_audio throughput, and model load time.
//! Requires `--features speech` and network access for model download.
//!
//!     cargo test --test echo_speech_bench --features speech -- --ignored --nocapture --test-threads=1

#[cfg(feature = "speech")]
mod bench {
    use shrimpk_core::{EchoConfig, Modality};
    use shrimpk_memory::speech::SpeechEmbedder;
    use shrimpk_memory::EchoEngine;
    use std::time::Instant;
    use tempfile::tempdir;

    /// Generate a sine-wave PCM buffer at the given frequency and duration.
    fn sine_wave(freq_hz: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin()
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Bench 1: Model load time
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires model download (~58 MB)"]
    fn bench_model_load_time() {
        println!("\n=== SPEECH MODEL LOAD TIME ===\n");

        let start = Instant::now();
        let mut emb = SpeechEmbedder::new();
        emb.load_models().expect("load_models");
        let load_ms = start.elapsed().as_millis();

        println!("Model load time: {load_ms}ms");
        println!("  (includes HF Hub cache check, ONNX session creation)");
        assert!(emb.is_ready());

        // Second load should be near-instant (already loaded)
        let start2 = Instant::now();
        emb.load_models().expect("reload");
        let reload_ms = start2.elapsed().as_millis();
        println!("Reload time (already loaded): {reload_ms}ms");
    }

    // -----------------------------------------------------------------------
    // Bench 2: embed_pcm latency (single inference)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires model download (~58 MB)"]
    fn bench_embed_pcm_latency() {
        println!("\n=== EMBED_PCM LATENCY ===\n");

        let mut emb = SpeechEmbedder::new();
        emb.load_models().expect("load_models");

        // Warmup
        let warmup = sine_wave(440.0, 1.0, 16_000);
        let _ = emb.embed_pcm(&warmup, 16_000).expect("warmup");

        // Test different durations
        for (label, duration_secs) in [
            ("0.5s", 0.5),
            ("1.0s", 1.0),
            ("2.0s", 2.0),
            ("5.0s", 5.0),
            ("10.0s", 10.0),
        ] {
            let pcm = sine_wave(440.0, duration_secs, 16_000);
            let samples = pcm.len();

            // Run 5 iterations, take median
            let mut times_us: Vec<u128> = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                let result = emb.embed_pcm(&pcm, 16_000).expect("embed_pcm");
                let elapsed = start.elapsed().as_micros();
                times_us.push(elapsed);
                assert_eq!(result.len(), shrimpk_memory::speech::SPEECH_DIM);
            }
            times_us.sort();
            let median_us = times_us[2];
            let median_ms = median_us as f64 / 1000.0;

            println!(
                "{label:>5} ({samples:>7} samples): {median_ms:>8.2}ms  (min={:.2}ms max={:.2}ms)",
                times_us[0] as f64 / 1000.0,
                times_us[4] as f64 / 1000.0,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bench 3: embed_pcm with resampling (44.1kHz → 16kHz)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires model download (~58 MB)"]
    fn bench_embed_pcm_resample_overhead() {
        println!("\n=== RESAMPLE OVERHEAD (44.1kHz vs 16kHz) ===\n");

        let mut emb = SpeechEmbedder::new();
        emb.load_models().expect("load_models");

        // 2 second clip
        let pcm_16k = sine_wave(440.0, 2.0, 16_000);
        let pcm_44k = sine_wave(440.0, 2.0, 44_100);

        // Warmup
        let _ = emb.embed_pcm(&pcm_16k, 16_000).expect("warmup");

        // 16kHz (no resample)
        let mut times_16k: Vec<u128> = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let _ = emb.embed_pcm(&pcm_16k, 16_000).expect("embed");
            times_16k.push(start.elapsed().as_micros());
        }
        times_16k.sort();

        // 44.1kHz (resample needed)
        let mut times_44k: Vec<u128> = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let _ = emb.embed_pcm(&pcm_44k, 44_100).expect("embed");
            times_44k.push(start.elapsed().as_micros());
        }
        times_44k.sort();

        let median_16k = times_16k[2] as f64 / 1000.0;
        let median_44k = times_44k[2] as f64 / 1000.0;
        let overhead_pct = ((median_44k - median_16k) / median_16k) * 100.0;

        println!("2s clip @ 16kHz (native):    {median_16k:.2}ms");
        println!("2s clip @ 44.1kHz (resample): {median_44k:.2}ms");
        println!("Resample overhead: {overhead_pct:+.1}%");
    }

    // -----------------------------------------------------------------------
    // Bench 4: store_audio end-to-end (EchoEngine)
    // -----------------------------------------------------------------------

    #[tokio::test]
    #[ignore = "requires model download (~58 MB)"]
    async fn bench_store_audio_throughput() {
        println!("\n=== STORE_AUDIO THROUGHPUT ===\n");

        let dir = tempdir().expect("tempdir");
        let config = EchoConfig {
            data_dir: dir.path().to_path_buf(),
            embedding_dim: 384,
            speech_embedding_dim: shrimpk_memory::speech::SPEECH_DIM,
            max_memories: 10_000,
            similarity_threshold: 0.05,
            max_echo_results: 10,
            enabled_modalities: vec![Modality::Text, Modality::Speech],
            ..Default::default()
        };

        let engine = EchoEngine::new(config).expect("engine");

        // Store 20 audio clips of varying lengths
        let clips: Vec<(String, Vec<f32>)> = vec![
            ("0.5s".into(), sine_wave(440.0, 0.5, 16_000)),
            ("1.0s".into(), sine_wave(880.0, 1.0, 16_000)),
            ("2.0s".into(), sine_wave(220.0, 2.0, 16_000)),
            ("5.0s".into(), sine_wave(660.0, 5.0, 16_000)),
        ];

        // Warmup with first clip
        let _ = engine.store_audio(&clips[0].1, 16_000, "bench-warmup", None).await;

        let n = 20;
        let start = Instant::now();
        let mut ok = 0;
        let mut fail = 0;

        for i in 0..n {
            let (_, pcm) = &clips[i % clips.len()];
            match engine.store_audio(pcm, 16_000, "bench", None).await {
                Ok(_) => ok += 1,
                Err(e) => {
                    eprintln!("store_audio failed: {e}");
                    fail += 1;
                }
            }
        }

        let total_ms = start.elapsed().as_millis();
        let per_clip_ms = total_ms as f64 / n as f64;

        println!("Stored {ok}/{n} clips in {total_ms}ms ({per_clip_ms:.1}ms/clip)");
        if fail > 0 {
            println!("  ({fail} failures)");
        }

        let stats = engine.stats().await;
        println!("Speech memories in store: {}", stats.speech_count);

        engine.shutdown().await;
    }
}

// Non-feature-gated: version check (always runs)
#[test]
fn version_is_0_7_0() {
    let version = env!("CARGO_PKG_VERSION");
    assert!(
        version.starts_with("0.7."),
        "Expected v0.7.x, got {version}. Did you forget the version bump?"
    );
}
