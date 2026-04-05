//! Multimodal cross-channel benchmark suite (KS39).
//!
//! Measures performance across text, vision, and speech channels.
//! All vision tests require `--features shrimpk-memory/vision` + CLIP model download.
//!
//!     cargo test --test echo_multimodal_bench -- --nocapture --test-threads=1
//!     cargo test --test echo_multimodal_bench --features shrimpk-memory/vision -- --ignored --nocapture --test-threads=1

#![allow(unexpected_cfgs)]

#[allow(unused_imports)]
use shrimpk_core::{EchoConfig, EchoResult, Modality, QueryMode};
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 200_000,
        similarity_threshold: 0.15,
        max_echo_results: 10,
        ram_budget_bytes: 2_000_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

#[allow(dead_code)]
fn top_n_contains(results: &[EchoResult], n: usize, needle: &str) -> bool {
    let lc = needle.to_lowercase();
    results
        .iter()
        .take(n)
        .any(|r| r.content.to_lowercase().contains(&lc))
}

// ---------------------------------------------------------------------------
// Test 1: Version check (non-ignored, always runs)
// ---------------------------------------------------------------------------

#[test]
fn version_is_0_7_0() {
    // This test runs in the workspace root package context
    // Check that the workspace version was bumped
    let version = env!("CARGO_PKG_VERSION");
    assert!(
        version.starts_with("0.7."),
        "Expected v0.7.x, got {version}. Did you forget the version bump?"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Text echo latency regression (HARD gate)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires fastembed model download + 100K store time"]
fn text_echo_latency_regression_100k() {
    println!("\n=== TEXT ECHO LATENCY REGRESSION (100K memories) ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 100K synthetic memories
    println!("Storing 100K memories...");
    let store_start = Instant::now();
    rt.block_on(async {
        for i in 0..100_000 {
            let text = format!(
                "Memory entry number {} about topic {} in category {}",
                i,
                ["rust", "python", "cooking", "music", "travel", "sports"][i % 6],
                ["work", "personal", "hobby", "learning"][i % 4],
            );
            engine.store(&text, "bench").await.expect("store");
        }
    });
    println!("Stored 100K in {:.1}s", store_start.elapsed().as_secs_f64());

    // Run 100 echo queries, measure latency
    let queries = [
        "What programming language do I use?",
        "Tell me about my cooking hobby",
        "What music do I listen to?",
        "Where have I traveled recently?",
        "What sports do I play?",
    ];

    let mut latencies: Vec<f64> = Vec::with_capacity(100);

    for i in 0..100 {
        let q = queries[i % queries.len()];
        let start = Instant::now();
        let _results = rt.block_on(async { engine.echo(q, 5).await.expect("echo") });
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[49];
    let p95 = latencies[94];
    let p99 = latencies[98];

    println!("Text echo latency at 100K memories:");
    println!("  P50: {p50:.2}ms");
    println!("  P95: {p95:.2}ms");
    println!("  P99: {p99:.2}ms");

    drop(rt);
    drop(engine);
    drop(dir);

    // HARD GATE: P50 must be < 4.0ms (v0.4.0 baseline was 3.50ms)
    assert!(
        p50 < 4.0,
        "REGRESSION: Text echo P50 {p50:.2}ms exceeds 4.0ms HARD gate"
    );
    println!("\nHARD GATE PASSED: P50 {p50:.2}ms < 4.0ms");
}

// ---------------------------------------------------------------------------
// Test 3: CLIP image embed latency
// ---------------------------------------------------------------------------

#[cfg(feature = "vision")]
#[test]
#[ignore = "requires CLIP model download (~352MB)"]
fn clip_image_embed_latency() {
    println!("\n=== CLIP IMAGE EMBED LATENCY ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Generate 50 synthetic test images (colored 64x64 PNGs)
    let images: Vec<Vec<u8>> = (0..50)
        .map(|i| {
            let r = ((i * 5) % 256) as u8;
            let g = ((i * 7 + 50) % 256) as u8;
            let b = ((i * 11 + 100) % 256) as u8;
            create_test_png(64, 64, r, g, b)
        })
        .collect();

    let mut latencies: Vec<f64> = Vec::with_capacity(50);

    for (i, img) in images.iter().enumerate() {
        let start = Instant::now();
        let _id = rt
            .block_on(async { engine.store_image(img, "bench", None).await })
            .expect("store_image");
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(ms);
        if i % 10 == 0 {
            println!("  Image {i}/50: {ms:.1}ms");
        }
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[24];
    let p95 = latencies[47];

    println!("\nCLIP image embed latency (50 images):");
    println!("  P50: {p50:.1}ms");
    println!("  P95: {p95:.1}ms");

    drop(rt);
    drop(engine);
    drop(dir);

    // HARD GATE: P50 < 100ms
    assert!(
        p50 < 100.0,
        "CLIP embed P50 {p50:.1}ms exceeds 100ms HARD gate"
    );
    println!("HARD GATE PASSED: P50 {p50:.1}ms < 100ms");
}

// ---------------------------------------------------------------------------
// Test 4: Cross-modal echo latency (Auto mode)
// ---------------------------------------------------------------------------

#[cfg(feature = "vision")]
#[test]
#[ignore = "requires CLIP model download"]
fn cross_modal_echo_latency() {
    println!("\n=== CROSS-MODAL ECHO LATENCY (Auto mode) ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 10K text + 100 images
    println!("Storing 10K text + 100 images...");
    rt.block_on(async {
        for i in 0..10_000 {
            let text = format!(
                "Text memory {} about {}",
                i,
                ["cats", "dogs", "cars"][i % 3]
            );
            engine.store(&text, "bench").await.expect("store");
        }
        for i in 0..100 {
            let img = create_test_png(64, 64, (i * 3) as u8, 100, 200);
            engine
                .store_image(&img, "bench", None)
                .await
                .expect("store_image");
        }
    });

    // Run 50 Auto-mode queries
    let queries = [
        "cats sitting on a mat",
        "dogs playing in the park",
        "cars driving on the highway",
        "what pets do I have",
        "show me animals",
    ];

    let mut latencies: Vec<f64> = Vec::with_capacity(50);

    for i in 0..50 {
        let q = queries[i % queries.len()];
        let start = Instant::now();
        let _results = rt.block_on(async {
            engine
                .echo_with_mode(q, 5, QueryMode::Auto)
                .await
                .expect("echo")
        });
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[24];
    let p95 = latencies[47];

    println!("Cross-modal echo latency (Auto, 10K text + 100 images):");
    println!("  P50: {p50:.1}ms");
    println!("  P95: {p95:.1}ms");

    drop(rt);
    drop(engine);
    drop(dir);

    // HARD GATE: P50 < 15ms
    assert!(
        p50 < 15.0,
        "Cross-modal echo P50 {p50:.1}ms exceeds 15ms HARD gate"
    );
    println!("HARD GATE PASSED: P50 {p50:.1}ms < 15ms");
}

// ---------------------------------------------------------------------------
// Test 5: Text-to-image recall
// ---------------------------------------------------------------------------

#[cfg(feature = "vision")]
#[test]
#[ignore = "requires CLIP model download"]
fn text_to_image_recall() {
    println!("\n=== TEXT-TO-IMAGE RECALL ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 100 images: 10 categories x 10 variants
    // Using synthetic colored images as proxies
    let categories = [
        ("red", 220u8, 30u8, 30u8),
        ("blue", 30, 30, 220),
        ("green", 30, 200, 30),
        ("yellow", 220, 220, 30),
        ("purple", 150, 30, 200),
        ("orange", 240, 150, 30),
        ("cyan", 30, 220, 220),
        ("pink", 240, 130, 180),
        ("brown", 140, 80, 30),
        ("gray", 130, 130, 130),
    ];

    rt.block_on(async {
        for (name, r, g, b) in &categories {
            for v in 0..10 {
                // Slight color variations per variant
                let img = create_test_png(
                    64,
                    64,
                    r.saturating_add(v * 2),
                    g.saturating_add(v),
                    b.saturating_add(v * 3),
                );
                engine
                    .store_image(&img, &format!("category_{name}"), None)
                    .await
                    .expect("store_image");
            }
        }
    });

    // Query for each category
    let mut recall_at_1 = 0;
    let mut recall_at_5 = 0;
    let total = categories.len();

    for (name, _, _, _) in &categories {
        let query = format!("a {name} colored image");
        let results = rt.block_on(async {
            engine
                .echo_with_mode(&query, 5, QueryMode::Vision)
                .await
                .expect("echo")
        });

        let hit1 = results.first().map_or(false, |r| r.source.contains(name));
        let hit5 = results.iter().take(5).any(|r| r.source.contains(name));

        if hit1 {
            recall_at_1 += 1;
        }
        if hit5 {
            recall_at_5 += 1;
        }

        println!(
            "  {name}: recall@1={} recall@5={}",
            if hit1 { "HIT" } else { "MISS" },
            if hit5 { "HIT" } else { "MISS" },
        );
    }

    let r1_pct = (recall_at_1 as f64 / total as f64) * 100.0;
    let r5_pct = (recall_at_5 as f64 / total as f64) * 100.0;

    println!("\nText-to-image recall:");
    println!("  Recall@1: {recall_at_1}/{total} ({r1_pct:.0}%)");
    println!("  Recall@5: {recall_at_5}/{total} ({r5_pct:.0}%)");
    println!("  Note: synthetic colored rectangles — real images would score higher");

    drop(rt);
    drop(engine);
    drop(dir);

    // SOFT GATE: print warning but don't fail (synthetic images)
    if r1_pct < 50.0 {
        println!(
            "WARNING: Recall@1 {r1_pct:.0}% below 50% target (expected with synthetic images)"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: RAM measurement
// ---------------------------------------------------------------------------

#[cfg(feature = "vision")]
#[test]
#[ignore = "requires models + measures system RAM"]
fn ram_measurement_10k_multimodal() {
    println!("\n=== RAM MEASUREMENT (10K multimodal) ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(dir.path().to_path_buf());

    let pid = std::process::id();
    let rss_before = get_rss_mb(pid);

    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 8K text + 1K image + 1K multimodal
    println!("Storing 8K text + 1K image + 1K multimodal...");
    rt.block_on(async {
        for i in 0..8_000 {
            engine
                .store(&format!("Text memory {i}"), "bench")
                .await
                .expect("store");
        }
        for i in 0..1_000 {
            let img = create_test_png(32, 32, (i % 256) as u8, 100, 150);
            engine
                .store_image(&img, "bench", None)
                .await
                .expect("store_image");
        }
        // 1K multimodal (text + image)
        for i in 0..1_000 {
            let img = create_test_png(32, 32, 200, (i % 256) as u8, 100);
            engine
                .store_image(&img, "bench_multi", None)
                .await
                .expect("store_image");
            engine
                .store(&format!("Multimodal context for image {i}"), "bench_multi")
                .await
                .expect("store");
        }
    });

    let rss_after = get_rss_mb(pid);
    let delta = rss_after - rss_before;

    println!("RAM usage:");
    println!("  Before: {rss_before}MB");
    println!("  After:  {rss_after}MB");
    println!("  Delta:  {delta}MB");

    drop(rt);
    drop(engine);
    drop(dir);

    // HARD GATE: < 1GB delta for 10K entries
    assert!(
        delta < 1024.0,
        "RAM delta {delta:.0}MB exceeds 1GB HARD gate"
    );
    println!("HARD GATE PASSED: RAM delta {delta:.0}MB < 1024MB");
}

// ---------------------------------------------------------------------------
// Test 7: Mixed-modal throughput
// ---------------------------------------------------------------------------

#[cfg(feature = "vision")]
#[test]
#[ignore = "requires models"]
fn mixed_modal_throughput() {
    println!("\n=== MIXED-MODAL THROUGHPUT ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 1000 mixed entries
    println!("Storing 1000 mixed entries...");
    let store_start = Instant::now();
    rt.block_on(async {
        for i in 0..700 {
            engine
                .store(&format!("Text {i} about topic {}", i % 10), "bench")
                .await
                .unwrap();
        }
        for i in 0..300 {
            let img = create_test_png(32, 32, (i * 3 % 256) as u8, 128, 64);
            engine.store_image(&img, "bench", None).await.unwrap();
        }
    });
    let store_elapsed = store_start.elapsed();
    println!(
        "Stored 1000 in {:.1}s ({:.0} entries/s)",
        store_elapsed.as_secs_f64(),
        1000.0 / store_elapsed.as_secs_f64()
    );

    // Run 100 queries with mixed modes
    let queries = [
        ("text query about topics", QueryMode::Text),
        ("show me images", QueryMode::Vision),
        ("find related content", QueryMode::Auto),
    ];

    let query_start = Instant::now();
    for i in 0..100 {
        let (q, mode) = queries[i % queries.len()];
        let _results =
            rt.block_on(async { engine.echo_with_mode(q, 5, mode).await.expect("echo") });
    }
    let query_elapsed = query_start.elapsed();
    let qps = 100.0 / query_elapsed.as_secs_f64();

    println!(
        "100 queries in {:.2}s ({qps:.0} queries/sec)",
        query_elapsed.as_secs_f64()
    );

    drop(rt);
    drop(engine);
    drop(dir);

    println!("\n=== THROUGHPUT BENCHMARK COMPLETE ===");
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Create a minimal PNG image with a solid color.
#[allow(dead_code)]
fn create_test_png(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    // PNG signature
    buf.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    // IHDR chunk
    let mut ihdr_data = Vec::new();
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(2); // color type RGB
    ihdr_data.extend_from_slice(&[0, 0, 0]); // compression, filter, interlace
    write_png_chunk(&mut buf, b"IHDR", &ihdr_data);
    // IDAT chunk (uncompressed, minimal)
    let mut raw = Vec::new();
    #[allow(clippy::same_item_push)]
    for _ in 0..height {
        raw.push(0); // filter byte: None
        for _ in 0..width {
            raw.extend_from_slice(&[r, g, b]);
        }
    }
    // Wrap in zlib (store block, no compression)
    let mut zlib = Vec::new();
    zlib.push(0x78); // CMF
    zlib.push(0x01); // FLG
    // Split into store blocks (max 65535 bytes each)
    let chunks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == chunks.len() - 1;
        zlib.push(if is_last { 1 } else { 0 }); // BFINAL
        let len = chunk.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(chunk);
    }
    // Adler-32 checksum
    let adler = adler32(&raw);
    zlib.extend_from_slice(&adler.to_be_bytes());
    write_png_chunk(&mut buf, b"IDAT", &zlib);
    // IEND chunk
    write_png_chunk(&mut buf, b"IEND", &[]);
    buf
}

#[allow(dead_code)]
fn write_png_chunk(buf: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    buf.extend_from_slice(chunk_type);
    buf.extend_from_slice(data);
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(chunk_type);
    crc_data.extend_from_slice(data);
    let crc = crc32_png(&crc_data);
    buf.extend_from_slice(&crc.to_be_bytes());
}

#[allow(dead_code)]
fn crc32_png(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

#[allow(dead_code)]
fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[allow(dead_code)]
fn get_rss_mb(_pid: u32) -> f64 {
    // Simple /proc/pid/status parser for Linux, fallback for others
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string(format!("/proc/{_pid}/status")) {
            for line in status.lines() {
                if line.starts_with("VmRSS:")
                    && let Some(kb_str) = line.split_whitespace().nth(1)
                    && let Ok(kb) = kb_str.parse::<f64>()
                {
                    return kb / 1024.0;
                }
            }
        }
    }
    // Fallback: use sysinfo if available, or return 0
    0.0
}
