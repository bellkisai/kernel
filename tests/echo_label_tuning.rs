//! Label tuning and profiling suite (KS48/KS49).
//!
//! Isolates timing of each echo pipeline stage at 10K scale and sweeps
//! prototype thresholds to find the optimal configuration.
//!
//!     cargo test --release --test echo_label_tuning -- --ignored --nocapture --test-threads=1

use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

fn generate_memory(i: usize) -> String {
    let templates = [
        "I joined a new company as a senior engineer last month",
        "I started learning Japanese on Duolingo about 30 days ago",
        "I ran 5K this morning in 24 minutes personal best",
        "I made homemade pasta from scratch for the first time",
        "I've been writing Rust for systems programming projects",
        "I moved from Oakland to San Francisco last month",
        "I play acoustic guitar in the evenings mostly folk songs",
        "I visited Tokyo last November and stayed in Shinjuku",
        "Jordan and I started dating after meeting at a Rust meetup",
        "I set up a monthly budget tracking system in a spreadsheet",
        "I have a tabby cat named Mochi who is 3 years old",
        "I enrolled in a master's degree program in computer science",
        "My doctor recommended reducing caffeine intake",
        "I'm reading Designing Data-Intensive Applications right now",
        "I started woodworking and built my first bookshelf",
    ];
    templates[i % templates.len()].to_string()
}

fn make_config(data_dir: PathBuf, use_labels: bool) -> EchoConfig {
    EchoConfig {
        max_memories: 200_000,
        similarity_threshold: 0.15,
        max_echo_results: 5,
        ram_budget_bytes: 2_000_000_000,
        data_dir,
        embedding_dim: 384,
        use_labels,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Test 1: Pipeline stage timing at 10K
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires fastembed model download"]
fn pipeline_stage_timing_10k() {
    println!("\n=== PIPELINE STAGE TIMING (10K memories) ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(dir.path().to_path_buf(), true);
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 10K memories
    println!("Storing 10K memories...");
    rt.block_on(async {
        for i in 0..10_000 {
            engine
                .store(&generate_memory(i), "bench")
                .await
                .expect("store");
        }
    });
    println!("Stored 10K memories.\n");

    // Warm up embedding model
    rt.block_on(async {
        let _ = engine.echo("warmup", 1).await;
    });

    // Time just the embedding step (no echo, just embed)
    let queries = [
        "What programming language do I use?",
        "What languages am I learning?",
        "How do I exercise?",
        "What do I like to cook?",
        "Where do I live?",
    ];

    // Full echo timing
    println!("Full echo timing (10K, labels ON):");
    let mut echo_times: Vec<f64> = Vec::new();
    for round in 0..10 {
        for q in &queries {
            let start = Instant::now();
            let _ = rt.block_on(async { engine.echo(q, 5).await.expect("echo") });
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            echo_times.push(ms);
            if round == 0 {
                println!("  {ms:6.2}ms | {q}");
            }
        }
    }
    echo_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_echo = echo_times[echo_times.len() / 2];
    let p95_echo = echo_times[echo_times.len() * 95 / 100];
    println!("\n  10K echo: P50={p50_echo:.2}ms  P95={p95_echo:.2}ms");

    // Compare: labels OFF at same scale
    drop(rt);
    drop(engine);
    drop(dir);

    let dir2 = tempdir().expect("temp dir");
    let config2 = make_config(dir2.path().to_path_buf(), false);
    let engine2 = EchoEngine::new(config2).expect("engine init");
    let rt2 = tokio::runtime::Runtime::new().unwrap();

    rt2.block_on(async {
        for i in 0..10_000 {
            engine2
                .store(&generate_memory(i), "bench")
                .await
                .expect("store");
        }
    });
    rt2.block_on(async {
        let _ = engine2.echo("warmup", 1).await;
    });

    println!("\nFull echo timing (10K, labels OFF):");
    let mut echo_off: Vec<f64> = Vec::new();
    for round in 0..10 {
        for q in &queries {
            let start = Instant::now();
            let _ = rt2.block_on(async { engine2.echo(q, 5).await.expect("echo") });
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            echo_off.push(ms);
            if round == 0 {
                println!("  {ms:6.2}ms | {q}");
            }
        }
    }
    echo_off.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_off = echo_off[echo_off.len() / 2];
    let p95_off = echo_off[echo_off.len() * 95 / 100];
    println!("\n  10K echo OFF: P50={p50_off:.2}ms  P95={p95_off:.2}ms");

    println!("\n  Delta: {:.1}x speedup with labels", p50_off / p50_echo);

    drop(rt2);
    drop(engine2);
    drop(dir2);
}

// ---------------------------------------------------------------------------
// Test 2: Threshold sweep at 10K (fast iteration)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires fastembed model download"]
fn threshold_sweep_10k() {
    println!("\n=== PROTOTYPE THRESHOLD SWEEP (10K memories) ===\n");

    let thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75];
    let queries = [
        "What programming language do I use?",
        "What languages am I learning?",
        "How do I exercise?",
        "What do I like to cook?",
        "Where have I traveled?",
    ];

    println!("Threshold | P50 (ms) | P95 (ms) | Avg Results | Notes");
    println!("----------|----------|----------|-------------|------");

    for &threshold in &thresholds {
        let dir = tempdir().expect("temp dir");
        let mut config = make_config(dir.path().to_path_buf(), true);
        config.use_labels = true;
        let engine = EchoEngine::new(config).expect("engine init");
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Manually set prototype threshold (need to access the engine internals)
        // For now, we test with the default threshold and vary content diversity

        // Store 10K
        rt.block_on(async {
            for i in 0..10_000 {
                engine
                    .store(&generate_memory(i), "bench")
                    .await
                    .expect("store");
            }
        });

        // Warm up
        rt.block_on(async {
            let _ = engine.echo("warmup", 1).await;
        });

        // Measure
        let mut latencies: Vec<f64> = Vec::new();
        let mut total_results = 0usize;
        for _ in 0..5 {
            for q in &queries {
                let start = Instant::now();
                let results = rt.block_on(async { engine.echo(q, 5).await.expect("echo") });
                latencies.push(start.elapsed().as_secs_f64() * 1000.0);
                total_results += results.len();
            }
        }
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = latencies[latencies.len() / 2];
        let p95 = latencies[latencies.len() * 95 / 100];
        let avg_results = total_results as f64 / (5.0 * queries.len() as f64);

        // Note: we can't change the threshold at runtime yet (it's in LabelPrototypes)
        // This test establishes the baseline at default threshold
        println!(
            "  {threshold:.2}     | {p50:7.2}  | {p95:7.2}  | {avg_results:10.1}  | default={}",
            if (threshold - 0.55_f64).abs() < 0.01 {
                "<-- current"
            } else {
                ""
            }
        );

        drop(rt);
        drop(engine);
        drop(dir);

        // Only run once since we can't change threshold at runtime yet
        // TODO: Add runtime threshold override for sweep testing
        break;
    }

    println!("\nNote: Full threshold sweep requires runtime threshold override.");
    println!("Current default: 0.55. Tune by modifying DEFAULT_PROTOTYPE_THRESHOLD in labels.rs.");
}
