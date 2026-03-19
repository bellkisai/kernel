//! Echo Memory CLI — interactive test harness for the ShrimPK kernel.
//!
//! Usage:
//!   echo-cli store "I prefer FastAPI for REST APIs"
//!   echo-cli echo "What framework for APIs?"
//!   echo-cli stats
//!   echo-cli forget <uuid>
//!   echo-cli dump
//!   echo-cli bench <count>

use shrimpk_core::{EchoConfig, QuantizationMode};
use shrimpk_memory::{EchoEngine, PiiFilter};
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Instant;

const VERSION: &str = "0.1.0";

/// Sentence pool for benchmark generation.
const SENTENCE_POOL: &[&str] = &[
    "I prefer Python for backend development",
    "React is great for building user interfaces",
    "Kubernetes manages container orchestration",
    "PostgreSQL is my database of choice",
    "I use VS Code as my primary editor",
    "Rust gives me fearless concurrency and memory safety",
    "TypeScript catches bugs before runtime",
    "Docker containers simplify deployment workflows",
    "Redis is perfect for caching and session storage",
    "GraphQL provides flexible API queries",
    "Terraform automates infrastructure provisioning",
    "Git branching strategies improve team collaboration",
    "Linux is the foundation of modern cloud infrastructure",
    "WebAssembly runs near-native code in the browser",
    "Prometheus and Grafana handle monitoring and alerting",
    "CI/CD pipelines automate testing and deployment",
    "Nginx is my go-to reverse proxy and load balancer",
    "Machine learning models need clean training data",
    "SQLite is ideal for embedded and mobile databases",
    "Vim keybindings make me faster at editing code",
    "Tailwind CSS speeds up frontend styling decisions",
    "gRPC is efficient for microservice communication",
    "AWS Lambda handles serverless compute workloads",
    "Tokio powers async Rust applications",
    "Next.js combines server rendering with React",
    "Neovim with LSP is a powerful development environment",
    "ClickHouse is excellent for analytical queries",
    "Tauri builds lightweight cross-platform desktop apps",
    "FastAPI makes building REST APIs quick and simple",
    "Godot is a great open-source game engine",
];

/// Extra fragments to shuffle into generated sentences for variety.
const FRAGMENTS: &[&str] = &[
    "for production workloads",
    "in my daily workflow",
    "when building microservices",
    "for rapid prototyping",
    "on large codebases",
    "with strict type safety",
    "across distributed systems",
    "for real-time applications",
    "in containerized environments",
    "with automated testing",
];

fn print_help() {
    println!("[echo-cli] Echo Memory CLI v{VERSION}");
    println!();
    println!("Usage:");
    println!("  echo-cli store \"text here\"     Store a memory, show ID + PII scan");
    println!("  echo-cli echo \"query here\"     Run echo cycle, show results with scores");
    println!("  echo-cli stats                 Show memory count, index size, config tier");
    println!("  echo-cli forget <uuid>         Remove a memory by ID");
    println!("  echo-cli dump                  List all stored memories (first 50 chars)");
    println!("  echo-cli bench <count>         Benchmark: store N memories, run 10 echo queries");
    println!();
    println!("Data directory: ~/.shrimpk-kernel/");
    println!("Memories persist automatically after store/forget commands.");
}

/// Detect the config tier name from the quantization mode and max_memories.
fn tier_name(config: &EchoConfig) -> &'static str {
    match (config.quantization, config.max_memories) {
        (QuantizationMode::Binary, _) => "minimal",
        (QuantizationMode::F32, m) if m >= 5_000_000 => "maximum",
        (QuantizationMode::F32, m) if m >= 1_000_000 => "full",
        (QuantizationMode::F32, m) if m >= 500_000 => "standard",
        _ => "custom",
    }
}

/// Format byte count as human-readable string.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Format a number with comma separators.
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

/// Truncate a string to a maximum length, appending "..." if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = s.char_indices()
            .take_while(|&(i, _)| i < max_len.saturating_sub(3))
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        format!("{}...", &s[..end])
    }
}

/// Detect total system RAM in GB for display.
fn detect_ram_gb() -> u64 {
    use sysinfo::System;
    let sys = System::new_all();
    sys.total_memory() / 1_073_741_824
}

/// Generate a random sentence by picking from the pool and optionally appending a fragment.
fn generate_sentence(rng: &mut impl Rng) -> String {
    let base = SENTENCE_POOL.choose(rng).unwrap();
    if rng.gen_bool(0.4) {
        let fragment = FRAGMENTS.choose(rng).unwrap();
        format!("{base} {fragment}")
    } else {
        base.to_string()
    }
}

/// Calculate percentile from a sorted slice of durations (in microseconds).
fn percentile(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx] as f64 / 1000.0 // convert us to ms
}

async fn cmd_store(engine: &EchoEngine, text: &str) -> anyhow::Result<()> {
    // Run PII scan for display
    let pii_filter = PiiFilter::new();
    let pii_matches = pii_filter.scan(text);
    let sensitivity = pii_filter.classify(text);

    let id = engine.store(text, "cli").await?;
    engine.persist().await?;

    println!(
        "[echo-cli] Stored memory {}... (sensitivity: {sensitivity:?})",
        &id.to_string()[..8]
    );

    if pii_matches.is_empty() {
        println!("[echo-cli] PII scan: no sensitive data detected");
    } else {
        let types: Vec<String> = pii_matches.iter().map(|m| m.pattern_type.to_string()).collect();
        println!(
            "[echo-cli] PII scan: {} match(es) detected [{}]",
            pii_matches.len(),
            types.join(", ")
        );
    }

    Ok(())
}

async fn cmd_echo(engine: &EchoEngine, query: &str) -> anyhow::Result<()> {
    let start = Instant::now();
    let results = engine.echo(query, 10).await?;
    let elapsed = start.elapsed();

    if results.is_empty() {
        println!(
            "[echo-cli] Echo results for \"{}\" ({:.1}ms): no matches",
            truncate(query, 60),
            elapsed.as_secs_f64() * 1000.0
        );
        return Ok(());
    }

    println!(
        "[echo-cli] Echo results for \"{}\" ({:.1}ms):",
        truncate(query, 60),
        elapsed.as_secs_f64() * 1000.0
    );

    for (i, result) in results.iter().enumerate() {
        // Look up echo_count from the store via a second echo isn't ideal,
        // but EchoResult has the data we need from the echo cycle.
        println!(
            "  #{} [{:.2}] \"{}\" (source: {}, id: {}...)",
            i + 1,
            result.similarity,
            truncate(&result.content, 60),
            result.source,
            &result.memory_id.to_string()[..8]
        );
    }

    Ok(())
}

async fn cmd_stats(engine: &EchoEngine, config: &EchoConfig) -> anyhow::Result<()> {
    let stats = engine.stats().await;
    let ram_gb = detect_ram_gb();

    println!("[echo-cli] Stats:");
    println!("  Memories:          {}", format_number(stats.total_memories));
    println!("  Max capacity:      {}", format_number(stats.max_capacity));
    println!("  Index size:        {}", format_bytes(stats.index_size_bytes));
    println!("  RAM usage:         {}", format_bytes(stats.ram_usage_bytes));
    println!("  Config tier:       {} ({ram_gb}GB detected)", tier_name(config));
    println!("  Quantization:      {:?}", config.quantization);
    println!("  Embedding dim:     {}", config.embedding_dim);
    println!("  Threshold:         {}", config.similarity_threshold);
    println!("  Echo queries:      {}", stats.total_echo_queries);
    if stats.total_echo_queries > 0 {
        println!("  Avg echo latency:  {:.1}ms", stats.avg_echo_latency_ms);
    }

    Ok(())
}

async fn cmd_forget(engine: &EchoEngine, id_str: &str) -> anyhow::Result<()> {
    let uuid = uuid::Uuid::parse_str(id_str)
        .map_err(|e| anyhow::anyhow!("Invalid UUID \"{id_str}\": {e}"))?;
    let id = shrimpk_core::MemoryId::from_uuid(uuid);

    engine.forget(id).await?;
    engine.persist().await?;

    println!("[echo-cli] Forgotten memory {}", &id_str[..id_str.len().min(8)]);

    Ok(())
}

async fn cmd_dump(engine: &EchoEngine, config: &EchoConfig) -> anyhow::Result<()> {
    let stats = engine.stats().await;

    if stats.total_memories == 0 {
        println!("[echo-cli] No memories stored.");
        return Ok(());
    }

    // Persist first to ensure the JSON file matches in-memory state
    engine.persist().await?;

    println!("[echo-cli] Stored memories ({}):", format_number(stats.total_memories));

    // EchoEngine doesn't expose direct iteration, so we read the persisted JSON.
    let store_path = config.data_dir.join("echo_store.json");

    if store_path.exists() {
        let json = std::fs::read_to_string(&store_path)?;
        let entries: Vec<serde_json::Value> = serde_json::from_str(&json)?;

        for entry in &entries {
            let id = entry["id"].as_str()
                .or_else(|| entry["id"].get("0").and_then(|v| v.as_str()))
                .unwrap_or("????????");
            let content = entry["content"].as_str().unwrap_or("");
            let masked = entry["masked_content"].as_str();
            let display = masked.unwrap_or(content);
            let source = entry["source"].as_str().unwrap_or("?");
            let echo_count = entry["echo_count"].as_u64().unwrap_or(0);
            let sensitivity = entry["sensitivity"].as_str().unwrap_or("Public");

            // Format the UUID display — handle both string and object UUID serialization
            let id_short = if id.len() >= 8 { &id[..8] } else { id };

            println!(
                "  {} \"{}\" (source: {}, echoed: {}x, sensitivity: {})",
                id_short,
                truncate(display, 50),
                source,
                echo_count,
                sensitivity,
            );
        }
    } else {
        println!("[echo-cli] No store file found at {}", store_path.display());
    }

    Ok(())
}

async fn cmd_bench(engine: &EchoEngine, count: usize) -> anyhow::Result<()> {
    println!("[echo-cli] Benchmark: storing {count} memories...");

    let mut rng = rand::thread_rng();

    // Phase 1: Store N random sentences
    let store_start = Instant::now();
    for i in 0..count {
        let sentence = generate_sentence(&mut rng);
        engine.store(&sentence, "bench").await?;
        if (i + 1) % 100 == 0 || i + 1 == count {
            print!("\r  Stored {}/{}...", i + 1, count);
        }
    }
    let store_elapsed = store_start.elapsed();
    println!(
        "\r  Stored {} memories in {:.1}ms ({:.1}ms/entry)                ",
        count,
        store_elapsed.as_secs_f64() * 1000.0,
        store_elapsed.as_secs_f64() * 1000.0 / count as f64
    );

    // Phase 2: Run 10 echo queries
    let queries = [
        "What programming language for backend?",
        "How to deploy containers?",
        "Best database for analytics?",
        "Text editor recommendations",
        "Frontend framework comparison",
        "Infrastructure as code tools",
        "How to monitor microservices?",
        "Fastest way to build REST APIs",
        "Cross-platform desktop development",
        "Machine learning workflow",
    ];

    println!("[echo-cli] Running {} echo queries...", queries.len());
    let mut latencies_us: Vec<u64> = Vec::with_capacity(queries.len());

    for query in &queries {
        let start = Instant::now();
        let results = engine.echo(query, 5).await?;
        let elapsed = start.elapsed();
        let elapsed_us = elapsed.as_micros() as u64;
        latencies_us.push(elapsed_us);

        println!(
            "  [{:.1}ms] \"{}\" -> {} results (top: {:.2})",
            elapsed_us as f64 / 1000.0,
            query,
            results.len(),
            results.first().map(|r| r.similarity).unwrap_or(0.0),
        );
    }

    // Phase 3: Compute and display percentile latencies
    latencies_us.sort();
    let p50 = percentile(&latencies_us, 50.0);
    let p95 = percentile(&latencies_us, 95.0);
    let p99 = percentile(&latencies_us, 99.0);
    let avg = latencies_us.iter().sum::<u64>() as f64 / latencies_us.len() as f64 / 1000.0;

    println!();
    println!("[echo-cli] Benchmark results:");
    println!("  Memories stored:  {}", format_number(count));
    println!("  Echo queries:     {}", queries.len());
    println!("  Avg latency:      {avg:.1}ms");
    println!("  P50 latency:      {p50:.1}ms");
    println!("  P95 latency:      {p95:.1}ms");
    println!("  P99 latency:      {p99:.1}ms");

    // Persist bench data
    engine.persist().await?;
    println!("  Data persisted to disk.");

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return Ok(());
    }

    let command = args[1].as_str();

    // Handle help early (no engine needed)
    if command == "help" || command == "--help" || command == "-h" {
        print_help();
        return Ok(());
    }

    // Auto-detect config and ensure data directory exists
    let config = EchoConfig::auto_detect();
    std::fs::create_dir_all(&config.data_dir)?;

    println!(
        "[echo-cli] Initializing Echo Memory engine ({} tier)...",
        tier_name(&config)
    );

    // Load existing memories from disk if available
    let engine = EchoEngine::load(config.clone())?;

    let initial_count = engine.stats().await.total_memories;
    if initial_count > 0 {
        println!(
            "[echo-cli] Loaded {} existing memories from disk.",
            format_number(initial_count)
        );
    }

    match command {
        "store" => {
            let text = args.get(2).ok_or_else(|| {
                anyhow::anyhow!("Usage: echo-cli store \"text to remember\"")
            })?;
            cmd_store(&engine, text).await?;
        }
        "echo" => {
            let query = args.get(2).ok_or_else(|| {
                anyhow::anyhow!("Usage: echo-cli echo \"query text\"")
            })?;
            cmd_echo(&engine, query).await?;
        }
        "stats" => {
            cmd_stats(&engine, &config).await?;
        }
        "forget" => {
            let id_str = args.get(2).ok_or_else(|| {
                anyhow::anyhow!("Usage: echo-cli forget <uuid>")
            })?;
            cmd_forget(&engine, id_str).await?;
        }
        "dump" => {
            cmd_dump(&engine, &config).await?;
        }
        "bench" => {
            let count_str = args.get(2).ok_or_else(|| {
                anyhow::anyhow!("Usage: echo-cli bench <count>")
            })?;
            let count: usize = count_str.parse().map_err(|_| {
                anyhow::anyhow!("Invalid count \"{count_str}\": must be a positive integer")
            })?;
            if count == 0 {
                anyhow::bail!("Count must be at least 1");
            }
            cmd_bench(&engine, count).await?;
        }
        other => {
            eprintln!("[echo-cli] Unknown command: \"{other}\"");
            println!();
            print_help();
            std::process::exit(1);
        }
    }

    Ok(())
}
