//! Scale benchmarks for Echo Memory at 1K, 10K, and 100K memories.
//!
//! Proves that KS2 (LSH + binary persistence) scales. Generates large synthetic
//! corpora and measures ingest, query (P50/P95/P99), persistence, and memory
//! footprint at each scale tier.
//!
//! All tests are `#[ignore]` because they require the fastembed model
//! (all-MiniLM-L6-v2, ~23MB ONNX) and take several minutes for large scales.
//!
//! Run all scale tests:
//!
//!     cargo test --test echo_scale_test -- --ignored --nocapture
//!
//! Run a single test:
//!
//!     cargo test --test echo_scale_test scale_echo_1k -- --ignored --nocapture

use bellkis_core::EchoConfig;
use bellkis_memory::EchoEngine;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TOPIC_PREFIXES: &[&str] = &[
    "I prefer",
    "My favorite",
    "I always use",
    "For work I choose",
    "The best tool for",
    "I recently switched to",
    "In my experience",
    "I recommend",
    "My go-to for",
    "I've been using",
    "My team uses",
    "For production we deploy",
    "I configured",
    "The project uses",
    "I installed",
    "My setup includes",
    "I migrated to",
    "We standardized on",
    "The architecture uses",
    "For testing I use",
];

const SUBJECTS: &[&str] = &[
    "Python",
    "Rust",
    "TypeScript",
    "FastAPI",
    "React",
    "PostgreSQL",
    "Redis",
    "Docker",
    "Kubernetes",
    "AWS",
    "Ollama",
    "Claude",
    "VS Code",
    "Git",
    "Linux",
    "Tailwind",
    "Next.js",
    "GraphQL",
    "MongoDB",
    "Terraform",
    "GitHub Actions",
    "Playwright",
    "Sentry",
    "Stripe",
    "Tauri",
    "Axum",
    "SQLite",
    "Nginx",
    "Cloudflare",
    "Vercel",
];

const SUFFIXES: &[&str] = &[
    "for backend development",
    "because of performance",
    "in production",
    "for local testing",
    "as my daily driver",
    "for the new project",
    "after evaluating alternatives",
    "on all platforms",
    "with great results",
    "for the entire team",
    "since last year",
    "for CI/CD pipelines",
    "in the cloud",
    "on my desktop",
    "for AI workloads",
];

const SCALE_QUERIES: &[&str] = &[
    "What programming language do I prefer?",
    "best database for production",
    "my development environment setup",
    "cloud deployment preferences",
    "testing framework choice",
    "container orchestration tool",
    "frontend framework",
    "API framework preference",
    "version control workflow",
    "CI/CD pipeline setup",
    "monitoring and observability",
    "authentication implementation",
    "caching strategy",
    "local AI model setup",
    "code editor configuration",
    "infrastructure as code",
    "web server choice",
    "CDN and edge computing",
    "payment processing",
    "real-time communication",
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate N diverse synthetic memory sentences by combining prefixes,
/// subjects, and suffixes in a deterministic but varied pattern.
fn generate_corpus(n: usize) -> Vec<String> {
    let total_combos = TOPIC_PREFIXES.len() * SUBJECTS.len() * SUFFIXES.len(); // 20*30*15 = 9000
    let mut corpus = Vec::with_capacity(n);

    for i in 0..n {
        // Spread across all combinations, wrapping when n > total_combos
        let combo = i % total_combos;
        let prefix_idx = combo / (SUBJECTS.len() * SUFFIXES.len());
        let remainder = combo % (SUBJECTS.len() * SUFFIXES.len());
        let subject_idx = remainder / SUFFIXES.len();
        let suffix_idx = remainder % SUFFIXES.len();

        // For entries beyond 9000, add an index suffix for uniqueness
        let sentence = if i < total_combos {
            format!(
                "{} {} {}",
                TOPIC_PREFIXES[prefix_idx], SUBJECTS[subject_idx], SUFFIXES[suffix_idx]
            )
        } else {
            format!(
                "{} {} {} (note #{})",
                TOPIC_PREFIXES[prefix_idx], SUBJECTS[subject_idx], SUFFIXES[suffix_idx], i
            )
        };
        corpus.push(sentence);
    }

    corpus
}

/// Build an EchoConfig for scale testing at a given capacity.
fn scale_config(data_dir: PathBuf, max_memories: usize, use_lsh: bool) -> EchoConfig {
    EchoConfig {
        max_memories,
        similarity_threshold: 0.14,
        max_echo_results: 20,
        ram_budget_bytes: 2_000_000_000,
        data_dir,
        embedding_dim: 384,
        use_lsh,
        use_bloom: true,
        ..Default::default()
    }
}

/// Ingest a corpus into an engine, returning total ingest duration.
async fn ingest_corpus(engine: &EchoEngine, corpus: &[String]) -> Duration {
    let start = Instant::now();
    for (i, text) in corpus.iter().enumerate() {
        engine
            .store(text, "scale-test")
            .await
            .unwrap_or_else(|e| panic!("Failed to store entry {i}: {e}"));

        // Progress log every 1000 entries
        if (i + 1) % 1000 == 0 {
            let elapsed = start.elapsed();
            let rate = (i + 1) as f64 / elapsed.as_secs_f64();
            eprintln!("  Ingested {}/{} ({:.0} entries/sec)", i + 1, corpus.len(), rate);
        }
    }
    start.elapsed()
}

/// Run the query set and return individual latencies in microseconds.
async fn run_queries(engine: &EchoEngine) -> Vec<u64> {
    let mut latencies = Vec::with_capacity(SCALE_QUERIES.len());
    for query in SCALE_QUERIES {
        let start = Instant::now();
        let _results = engine
            .echo(query, 10)
            .await
            .unwrap_or_else(|e| panic!("Echo query failed: {e}"));
        latencies.push(start.elapsed().as_micros() as u64);
    }
    latencies
}

/// Compute percentile from a sorted slice of values.
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Print a latency report from raw latency values (in microseconds).
fn print_latency_report(label: &str, latencies: &[u64]) {
    let mut sorted = latencies.to_vec();
    sorted.sort();

    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    let p99 = percentile(&sorted, 99.0);
    let min = sorted.first().copied().unwrap_or(0);
    let max = sorted.last().copied().unwrap_or(0);
    let avg: u64 = if sorted.is_empty() {
        0
    } else {
        sorted.iter().sum::<u64>() / sorted.len() as u64
    };

    eprintln!();
    eprintln!("=== {label} ===");
    eprintln!("  Queries: {}", latencies.len());
    eprintln!("  Min:     {:.2} ms", min as f64 / 1000.0);
    eprintln!("  P50:     {:.2} ms", p50 as f64 / 1000.0);
    eprintln!("  P95:     {:.2} ms", p95 as f64 / 1000.0);
    eprintln!("  P99:     {:.2} ms", p99 as f64 / 1000.0);
    eprintln!("  Max:     {:.2} ms", max as f64 / 1000.0);
    eprintln!("  Avg:     {:.2} ms", avg as f64 / 1000.0);
}

// ---------------------------------------------------------------------------
// Test 1: scale_echo_1k
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "scale benchmark — requires fastembed model, ~1 min"]
async fn scale_echo_1k() {
    let dir = tempdir().expect("temp dir");
    let config = scale_config(dir.path().to_path_buf(), 2_000, true);
    let engine = EchoEngine::new(config).expect("engine init");

    // Generate and ingest 1K memories
    let corpus = generate_corpus(1_000);
    eprintln!("\n>>> scale_echo_1k: Ingesting 1,000 memories...");
    let ingest_time = ingest_corpus(&engine, &corpus).await;
    eprintln!("  Ingest complete: {:.2}s ({:.0} entries/sec)",
        ingest_time.as_secs_f64(),
        1000.0 / ingest_time.as_secs_f64()
    );

    // Verify count
    let stats = engine.stats().await;
    assert_eq!(stats.total_memories, 1_000);

    // Run queries
    eprintln!("  Running {} queries...", SCALE_QUERIES.len());
    let latencies = run_queries(&engine).await;
    print_latency_report("1K Memories — Query Latency (LSH enabled)", &latencies);

    // Assert P50 < 100ms
    let mut sorted = latencies.clone();
    sorted.sort();
    let p50_us = percentile(&sorted, 50.0);
    let p50_ms = p50_us as f64 / 1000.0;
    assert!(
        p50_ms < 100.0,
        "P50 latency {p50_ms:.2}ms exceeds 100ms threshold at 1K scale"
    );

    eprintln!("\n  PASS: P50 = {p50_ms:.2}ms < 100ms");
}

// ---------------------------------------------------------------------------
// Test 2: scale_echo_10k
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "scale benchmark — requires fastembed model, ~10 min"]
async fn scale_echo_10k() {
    let dir = tempdir().expect("temp dir");

    // --- Run with LSH enabled ---
    let config_lsh = scale_config(dir.path().to_path_buf(), 15_000, true);
    let engine_lsh = EchoEngine::new(config_lsh).expect("engine init (LSH)");

    let corpus = generate_corpus(10_000);
    eprintln!("\n>>> scale_echo_10k: Ingesting 10,000 memories (LSH enabled)...");
    let ingest_time = ingest_corpus(&engine_lsh, &corpus).await;
    eprintln!("  Ingest complete: {:.2}s ({:.0} entries/sec)",
        ingest_time.as_secs_f64(),
        10_000.0 / ingest_time.as_secs_f64()
    );

    let stats = engine_lsh.stats().await;
    assert_eq!(stats.total_memories, 10_000);

    eprintln!("  Running {} queries (LSH)...", SCALE_QUERIES.len());
    let latencies_lsh = run_queries(&engine_lsh).await;
    print_latency_report("10K Memories — Query Latency (LSH enabled)", &latencies_lsh);

    // --- Run with LSH disabled (brute-force) ---
    let dir2 = tempdir().expect("temp dir 2");
    let config_bf = scale_config(dir2.path().to_path_buf(), 15_000, false);
    let engine_bf = EchoEngine::new(config_bf).expect("engine init (brute-force)");

    eprintln!("  Ingesting 10,000 memories (brute-force)...");
    let ingest_time_bf = ingest_corpus(&engine_bf, &corpus).await;
    eprintln!("  Ingest complete: {:.2}s ({:.0} entries/sec)",
        ingest_time_bf.as_secs_f64(),
        10_000.0 / ingest_time_bf.as_secs_f64()
    );

    eprintln!("  Running {} queries (brute-force)...", SCALE_QUERIES.len());
    let latencies_bf = run_queries(&engine_bf).await;
    print_latency_report("10K Memories — Query Latency (brute-force)", &latencies_bf);

    // --- Comparison ---
    let mut sorted_lsh = latencies_lsh.clone();
    sorted_lsh.sort();
    let mut sorted_bf = latencies_bf.clone();
    sorted_bf.sort();

    let p50_lsh = percentile(&sorted_lsh, 50.0);
    let p50_bf = percentile(&sorted_bf, 50.0);
    let p95_lsh = percentile(&sorted_lsh, 95.0);
    let p95_bf = percentile(&sorted_bf, 95.0);

    eprintln!();
    eprintln!("=== 10K Comparison: LSH vs Brute-Force ===");
    eprintln!("  P50: LSH={:.2}ms  BF={:.2}ms  speedup={:.2}x",
        p50_lsh as f64 / 1000.0,
        p50_bf as f64 / 1000.0,
        if p50_lsh > 0 { p50_bf as f64 / p50_lsh as f64 } else { f64::NAN }
    );
    eprintln!("  P95: LSH={:.2}ms  BF={:.2}ms  speedup={:.2}x",
        p95_lsh as f64 / 1000.0,
        p95_bf as f64 / 1000.0,
        if p95_lsh > 0 { p95_bf as f64 / p95_lsh as f64 } else { f64::NAN }
    );

    // Assert P50 < 200ms with LSH
    let p50_lsh_ms = p50_lsh as f64 / 1000.0;
    assert!(
        p50_lsh_ms < 200.0,
        "P50 latency {p50_lsh_ms:.2}ms exceeds 200ms threshold at 10K scale with LSH"
    );

    eprintln!("\n  PASS: P50 (LSH) = {p50_lsh_ms:.2}ms < 200ms");
}

// ---------------------------------------------------------------------------
// Test 3: scale_echo_100k
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "scale benchmark — requires fastembed model, ~60+ min"]
async fn scale_echo_100k() {
    let dir = tempdir().expect("temp dir");
    let config = scale_config(dir.path().to_path_buf(), 150_000, true);
    let engine = EchoEngine::new(config).expect("engine init");

    let corpus = generate_corpus(100_000);
    eprintln!("\n>>> scale_echo_100k: Ingesting 100,000 memories...");
    let ingest_time = ingest_corpus(&engine, &corpus).await;
    eprintln!("  Ingest complete: {:.2}s ({:.0} entries/sec)",
        ingest_time.as_secs_f64(),
        100_000.0 / ingest_time.as_secs_f64()
    );

    let stats = engine.stats().await;
    assert_eq!(stats.total_memories, 100_000);

    // Query (LSH only — brute-force too slow at 100K)
    eprintln!("  Running {} queries (LSH only)...", SCALE_QUERIES.len());
    let latencies = run_queries(&engine).await;
    print_latency_report("100K Memories — Query Latency (LSH only)", &latencies);

    // Memory footprint report
    let stats = engine.stats().await;
    eprintln!();
    eprintln!("=== 100K Memory Footprint ===");
    eprintln!("  Total memories:   {}", stats.total_memories);
    eprintln!("  Index size:       {:.2} MB", stats.index_size_bytes as f64 / 1_048_576.0);
    eprintln!("  RAM usage (est.): {:.2} MB", stats.ram_usage_bytes as f64 / 1_048_576.0);
    eprintln!("  Max capacity:     {}", stats.max_capacity);

    // Assert P50 < 500ms
    let mut sorted = latencies.clone();
    sorted.sort();
    let p50_us = percentile(&sorted, 50.0);
    let p50_ms = p50_us as f64 / 1000.0;
    assert!(
        p50_ms < 500.0,
        "P50 latency {p50_ms:.2}ms exceeds 500ms threshold at 100K scale"
    );

    eprintln!("\n  PASS: P50 = {p50_ms:.2}ms < 500ms");
}

// ---------------------------------------------------------------------------
// Test 4: scale_persistence_benchmark
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "scale benchmark — requires fastembed model, ~60+ min"]
async fn scale_persistence_benchmark() {
    eprintln!("\n>>> scale_persistence_benchmark");
    eprintln!();
    eprintln!("{:<8} | {:<12} | {:<12} | {:<12} | {:<8}", "Scale", "Save Time", "Load Time", "File Size", "Verify");
    eprintln!("{:-<8}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<8}", "", "", "", "", "");

    for &n in &[1_000usize, 10_000, 100_000] {
        let dir = tempdir().expect("temp dir");
        let config = scale_config(dir.path().to_path_buf(), n + 1000, true);
        let engine = EchoEngine::new(config.clone()).expect("engine init");

        // Ingest
        let corpus = generate_corpus(n);
        eprintln!("\n  Ingesting {n} memories...");
        let _ = ingest_corpus(&engine, &corpus).await;

        let original_count = engine.stats().await.total_memories;

        // Save to binary
        let save_start = Instant::now();
        engine.persist().await.expect("persist should succeed");
        let save_time = save_start.elapsed();

        // Measure file size
        let store_path = dir.path().join("echo_store.shrm");
        let file_size = std::fs::metadata(&store_path)
            .map(|m| m.len())
            .unwrap_or(0);

        // Load from binary
        let load_start = Instant::now();
        let loaded_engine = EchoEngine::load(config.clone()).expect("load should succeed");
        let load_time = load_start.elapsed();

        // Verify: same count
        let loaded_count = loaded_engine.stats().await.total_memories;
        let count_match = loaded_count == original_count;

        // Verify: same echo results for a sample query
        let orig_results = engine
            .echo("What programming language do I prefer?", 5)
            .await
            .expect("echo");
        let loaded_results = loaded_engine
            .echo("What programming language do I prefer?", 5)
            .await
            .expect("echo");

        let results_match = orig_results.len() == loaded_results.len()
            && orig_results
                .iter()
                .zip(loaded_results.iter())
                .all(|(a, b)| a.memory_id == b.memory_id);

        let verified = if count_match && results_match { "OK" } else { "FAIL" };

        let size_str = if file_size > 1_073_741_824 {
            format!("{:.2} GB", file_size as f64 / 1_073_741_824.0)
        } else {
            format!("{:.2} MB", file_size as f64 / 1_048_576.0)
        };

        eprintln!(
            "{:<8} | {:<12} | {:<12} | {:<12} | {:<8}",
            format!("{n}"),
            format!("{:.3}s", save_time.as_secs_f64()),
            format!("{:.3}s", load_time.as_secs_f64()),
            size_str,
            verified
        );

        assert!(count_match, "Count mismatch at {n}: orig={original_count}, loaded={loaded_count}");
        assert!(results_match, "Echo results mismatch at {n}");
    }

    eprintln!();
    eprintln!("  PASS: All persistence roundtrips verified");
}

// ---------------------------------------------------------------------------
// Test 5: scale_lsh_vs_bruteforce
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "scale benchmark — requires fastembed model, ~15 min"]
async fn scale_lsh_vs_bruteforce() {
    eprintln!("\n>>> scale_lsh_vs_bruteforce");

    for &n in &[1_000usize, 10_000] {
        eprintln!("\n--- {n} memories ---");

        let corpus = generate_corpus(n);

        // Engine with LSH
        let dir_lsh = tempdir().expect("temp dir");
        let config_lsh = scale_config(dir_lsh.path().to_path_buf(), n + 1000, true);
        let engine_lsh = EchoEngine::new(config_lsh).expect("engine init (LSH)");
        eprintln!("  Ingesting (LSH)...");
        let _ = ingest_corpus(&engine_lsh, &corpus).await;

        // Engine without LSH (brute-force)
        let dir_bf = tempdir().expect("temp dir");
        let config_bf = scale_config(dir_bf.path().to_path_buf(), n + 1000, false);
        let engine_bf = EchoEngine::new(config_bf).expect("engine init (BF)");
        eprintln!("  Ingesting (brute-force)...");
        let _ = ingest_corpus(&engine_bf, &corpus).await;

        // Run queries on both
        eprintln!("  Querying LSH...");
        let latencies_lsh = run_queries(&engine_lsh).await;
        eprintln!("  Querying brute-force...");
        let latencies_bf = run_queries(&engine_bf).await;

        print_latency_report(&format!("{n} — LSH"), &latencies_lsh);
        print_latency_report(&format!("{n} — Brute-Force"), &latencies_bf);

        // Compute recall: how many of brute-force's top results appear in LSH results?
        let mut recall_scores = Vec::new();
        for query in SCALE_QUERIES {
            let lsh_results = engine_lsh.echo(query, 10).await.expect("echo LSH");
            let bf_results = engine_bf.echo(query, 10).await.expect("echo BF");

            let bf_ids: Vec<_> = bf_results.iter().map(|r| &r.memory_id).collect();
            let lsh_ids: Vec<_> = lsh_results.iter().map(|r| &r.memory_id).collect();

            let overlap = bf_ids.iter().filter(|id| lsh_ids.contains(id)).count();
            let recall = if bf_ids.is_empty() {
                1.0
            } else {
                overlap as f64 / bf_ids.len() as f64
            };
            recall_scores.push(recall);
        }

        let avg_recall = recall_scores.iter().sum::<f64>() / recall_scores.len() as f64;
        let min_recall = recall_scores.iter().copied().fold(f64::INFINITY, f64::min);

        // Speedup
        let mut sorted_lsh = latencies_lsh.clone();
        sorted_lsh.sort();
        let mut sorted_bf = latencies_bf.clone();
        sorted_bf.sort();

        let p50_lsh = percentile(&sorted_lsh, 50.0);
        let p50_bf = percentile(&sorted_bf, 50.0);
        let speedup = if p50_lsh > 0 { p50_bf as f64 / p50_lsh as f64 } else { f64::NAN };

        eprintln!();
        eprintln!("=== {n} — LSH vs Brute-Force Summary ===");
        eprintln!("  P50 speedup:   {speedup:.2}x (LSH={:.2}ms, BF={:.2}ms)",
            p50_lsh as f64 / 1000.0,
            p50_bf as f64 / 1000.0
        );
        eprintln!("  Avg recall:    {avg_recall:.2} ({:.0}%)", avg_recall * 100.0);
        eprintln!("  Min recall:    {min_recall:.2} ({:.0}%)", min_recall * 100.0);
    }

    eprintln!("\n  PASS: LSH vs brute-force comparison complete");
}

// ---------------------------------------------------------------------------
// Test 6: scale_memory_footprint
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "scale benchmark — requires fastembed model, ~60+ min"]
async fn scale_memory_footprint() {
    eprintln!("\n>>> scale_memory_footprint");
    eprintln!();
    eprintln!(
        "{:<8} | {:<16} | {:<16} | {:<16} | {:<16}",
        "Scale", "Index Size", "RAM Usage (est)", "Bytes/Entry", "Avg Latency"
    );
    eprintln!(
        "{:-<8}-+-{:-<16}-+-{:-<16}-+-{:-<16}-+-{:-<16}",
        "", "", "", "", ""
    );

    for &n in &[1_000usize, 10_000, 100_000] {
        let dir = tempdir().expect("temp dir");
        let config = scale_config(dir.path().to_path_buf(), n + 1000, true);
        let engine = EchoEngine::new(config).expect("engine init");

        // Ingest
        let corpus = generate_corpus(n);
        eprintln!("\n  Ingesting {n} memories...");
        let _ = ingest_corpus(&engine, &corpus).await;

        // Run a few queries to populate latency stats
        for query in &SCALE_QUERIES[..5] {
            let _ = engine.echo(query, 10).await;
        }

        let stats = engine.stats().await;

        // Echo index: embeddings array = total_memories * 384 * 4 bytes
        let embedding_array_bytes = stats.total_memories as u64 * 384 * 4;

        // LSH tables estimate:
        // 16 tables * buckets_per_table * avg_entries_per_bucket * 4 bytes per u32 id
        // Plus hyperplanes: 16 tables * 10 hyperplanes * 384 floats * 4 bytes
        let lsh_hyperplane_bytes: u64 = 16 * 10 * 384 * 4; // ~245 KB fixed
        // Reverse index: n entries * (16 table refs * (usize + u64)) ~ n * 384 bytes
        let lsh_reverse_index_est: u64 = stats.total_memories as u64 * 384;
        // Bucket storage: n entries * 16 tables * 4 bytes per id (with dedup overlap ~ 1x)
        let lsh_bucket_est: u64 = stats.total_memories as u64 * 16 * 4;
        let lsh_total_est = lsh_hyperplane_bytes + lsh_reverse_index_est + lsh_bucket_est;

        // Metadata per entry estimate: id (16 bytes) + content (~60 bytes avg)
        // + source (~10 bytes) + timestamps + sensitivity + counters ~120 bytes
        let metadata_est = stats.total_memories as u64 * 120;

        let total_est = embedding_array_bytes + lsh_total_est + metadata_est;
        let bytes_per_entry = if stats.total_memories > 0 {
            total_est / stats.total_memories as u64
        } else {
            0
        };

        eprintln!(
            "{:<8} | {:<16} | {:<16} | {:<16} | {:<16}",
            format!("{n}"),
            format!("{:.2} MB", stats.index_size_bytes as f64 / 1_048_576.0),
            format!("{:.2} MB", total_est as f64 / 1_048_576.0),
            format!("{} B", bytes_per_entry),
            format!("{:.2} ms", stats.avg_echo_latency_ms)
        );

        // Detailed breakdown for the largest scale
        if n >= 100_000 {
            eprintln!();
            eprintln!("  === 100K Detailed Memory Breakdown ===");
            eprintln!("  Embeddings array:  {:.2} MB ({} entries x 384 x 4B)",
                embedding_array_bytes as f64 / 1_048_576.0, stats.total_memories);
            eprintln!("  LSH hyperplanes:   {:.2} MB (16 tables x 10 planes x 384 dims)",
                lsh_hyperplane_bytes as f64 / 1_048_576.0);
            eprintln!("  LSH reverse index: {:.2} MB (est.)",
                lsh_reverse_index_est as f64 / 1_048_576.0);
            eprintln!("  LSH buckets:       {:.2} MB (est.)",
                lsh_bucket_est as f64 / 1_048_576.0);
            eprintln!("  Entry metadata:    {:.2} MB (est.)",
                metadata_est as f64 / 1_048_576.0);
            eprintln!("  --------------------------------");
            eprintln!("  Total estimated:   {:.2} MB",
                total_est as f64 / 1_048_576.0);
        }
    }

    eprintln!();
    eprintln!("  PASS: Memory footprint report complete");
}
