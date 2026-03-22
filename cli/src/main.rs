//! ShrimPK CLI — the Echo Memory command-line interface.
//!
//! Usage:
//!   shrimpk store "I prefer FastAPI for REST APIs"
//!   shrimpk echo "What framework for APIs?"
//!   shrimpk stats
//!   shrimpk config show
//!   shrimpk status

use clap::{Parser, Subcommand};
use rand::Rng;
use rand::seq::SliceRandom;
use shrimpk_core::{EchoConfig, QuantizationMode, config};
use shrimpk_memory::{EchoEngine, PiiFilter};
use std::time::Instant;

/// ShrimPK — the AI memory engine.
#[derive(Parser)]
#[command(name = "shrimpk", version, about = "Push-based AI memory engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Store a memory
    Store {
        /// Text content to remember
        text: String,
        /// Source label
        #[arg(short, long, default_value = "cli")]
        source: String,
        /// Quiet mode — suppress output (for hook/script integration)
        #[arg(long)]
        quiet: bool,
    },
    /// Find memories that resonate with a query
    Echo {
        /// Query text
        query: String,
        /// Maximum results to return
        #[arg(short, long, default_value_t = 10)]
        max_results: usize,
        /// Output as JSON (for hook/script integration)
        #[arg(long)]
        json: bool,
    },
    /// Show engine statistics
    Stats,
    /// Remove a memory by UUID
    Forget {
        /// Memory UUID
        id: String,
    },
    /// List all stored memories
    Dump,
    /// Run performance benchmark
    Bench {
        /// Number of memories to store
        count: usize,
    },
    /// View and manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Show disk usage and system status
    Status,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration with source info
    Show,
    /// Set a configuration value (writes to config.toml)
    Set {
        /// Config key (e.g., max_memories, similarity_threshold, max_disk_bytes)
        key: String,
        /// Value to set
        value: String,
    },
    /// Reset configuration to auto-detect defaults
    Reset,
    /// Show config file path
    Path,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    "Machine learning models need clean training data",
    "SQLite is ideal for embedded and mobile databases",
    "Tailwind CSS speeds up frontend styling decisions",
    "gRPC is efficient for microservice communication",
    "Tokio powers async Rust applications",
    "Tauri builds lightweight cross-platform desktop apps",
    "FastAPI makes building REST APIs quick and simple",
    "Godot is a great open-source game engine",
];

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

fn tier_name(config: &EchoConfig) -> &'static str {
    match (config.quantization, config.max_memories) {
        (QuantizationMode::Binary, _) => "minimal",
        (QuantizationMode::F32, m) if m >= 5_000_000 => "maximum",
        (QuantizationMode::F32, m) if m >= 1_000_000 => "full",
        (QuantizationMode::F32, m) if m >= 500_000 => "standard",
        _ => "custom",
    }
}

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

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(c);
    }
    result
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = s
            .char_indices()
            .take_while(|&(i, _)| i < max_len.saturating_sub(3))
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        format!("{}...", &s[..end])
    }
}

fn detect_ram_gb() -> u64 {
    use sysinfo::System;
    let sys = System::new_all();
    sys.total_memory() / 1_073_741_824
}

fn generate_sentence(rng: &mut impl Rng) -> String {
    let base = SENTENCE_POOL.choose(rng).unwrap();
    if rng.gen_bool(0.4) {
        let fragment = FRAGMENTS.choose(rng).unwrap();
        format!("{base} {fragment}")
    } else {
        base.to_string()
    }
}

fn percentile(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx] as f64 / 1000.0
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async fn cmd_store(engine: &EchoEngine, text: &str, source: &str) -> anyhow::Result<()> {
    let pii_filter = PiiFilter::new();
    let pii_matches = pii_filter.scan(text);
    let sensitivity = pii_filter.classify(text);

    let id = engine.store(text, source).await?;
    engine.persist().await?;

    println!(
        "[shrimpk] Stored memory {}... (sensitivity: {sensitivity:?})",
        &id.to_string()[..8]
    );

    if pii_matches.is_empty() {
        println!("[shrimpk] PII scan: no sensitive data detected");
    } else {
        let types: Vec<String> = pii_matches
            .iter()
            .map(|m| m.pattern_type.to_string())
            .collect();
        println!(
            "[shrimpk] PII scan: {} match(es) detected [{}]",
            pii_matches.len(),
            types.join(", ")
        );
    }

    Ok(())
}

async fn cmd_echo(engine: &EchoEngine, query: &str, max_results: usize) -> anyhow::Result<()> {
    let start = Instant::now();
    let results = engine.echo(query, max_results).await?;
    let elapsed = start.elapsed();

    if results.is_empty() {
        println!(
            "[shrimpk] Echo results for \"{}\" ({:.1}ms): no matches",
            truncate(query, 60),
            elapsed.as_secs_f64() * 1000.0
        );
        return Ok(());
    }

    println!(
        "[shrimpk] Echo results for \"{}\" ({:.1}ms):",
        truncate(query, 60),
        elapsed.as_secs_f64() * 1000.0
    );

    for (i, result) in results.iter().enumerate() {
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

async fn cmd_echo_json(engine: &EchoEngine, query: &str, max_results: usize) -> anyhow::Result<()> {
    let results = engine.echo(query, max_results).await?;

    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "content": r.content,
                "similarity": (r.similarity * 100.0).round() / 100.0,
                "source": r.source
            })
        })
        .collect();

    println!("{}", serde_json::to_string(&json_results)?);
    Ok(())
}

async fn cmd_stats(engine: &EchoEngine, config: &EchoConfig) -> anyhow::Result<()> {
    let stats = engine.stats().await;
    let ram_gb = detect_ram_gb();

    println!("[shrimpk] Stats:");
    println!(
        "  Memories:          {}",
        format_number(stats.total_memories)
    );
    println!("  Max capacity:      {}", format_number(stats.max_capacity));
    println!(
        "  Index size:        {}",
        format_bytes(stats.index_size_bytes)
    );
    println!(
        "  RAM usage:         {}",
        format_bytes(stats.ram_usage_bytes)
    );
    println!(
        "  Disk usage:        {} / {}",
        format_bytes(stats.disk_usage_bytes),
        format_bytes(stats.max_disk_bytes)
    );
    println!(
        "  Config tier:       {} ({ram_gb}GB detected)",
        tier_name(config)
    );
    println!("  Quantization:      {}", config.quantization);
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

    println!(
        "[shrimpk] Forgotten memory {}",
        &id_str[..id_str.len().min(8)]
    );

    Ok(())
}

async fn cmd_dump(engine: &EchoEngine, config: &EchoConfig) -> anyhow::Result<()> {
    let stats = engine.stats().await;

    if stats.total_memories == 0 {
        println!("[shrimpk] No memories stored.");
        return Ok(());
    }

    engine.persist().await?;

    println!(
        "[shrimpk] Stored memories ({}):",
        format_number(stats.total_memories)
    );

    let store_path = config.data_dir.join("echo_store.json");

    if store_path.exists() {
        let json = std::fs::read_to_string(&store_path)?;
        let entries: Vec<serde_json::Value> = serde_json::from_str(&json)?;

        for entry in &entries {
            let id = entry["id"]
                .as_str()
                .or_else(|| entry["id"].get("0").and_then(|v| v.as_str()))
                .unwrap_or("????????");
            let content = entry["content"].as_str().unwrap_or("");
            let masked = entry["masked_content"].as_str();
            let display = masked.unwrap_or(content);
            let source = entry["source"].as_str().unwrap_or("?");
            let echo_count = entry["echo_count"].as_u64().unwrap_or(0);
            let sensitivity = entry["sensitivity"].as_str().unwrap_or("Public");

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
        println!("[shrimpk] No store file found at {}", store_path.display());
    }

    Ok(())
}

async fn cmd_bench(engine: &EchoEngine, count: usize) -> anyhow::Result<()> {
    println!("[shrimpk] Benchmark: storing {count} memories...");

    let mut rng = rand::thread_rng();

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

    println!("[shrimpk] Running {} echo queries...", queries.len());
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

    latencies_us.sort();
    let p50 = percentile(&latencies_us, 50.0);
    let p95 = percentile(&latencies_us, 95.0);
    let p99 = percentile(&latencies_us, 99.0);
    let avg = latencies_us.iter().sum::<u64>() as f64 / latencies_us.len() as f64 / 1000.0;

    println!();
    println!("[shrimpk] Benchmark results:");
    println!("  Memories stored:  {}", format_number(count));
    println!("  Echo queries:     {}", queries.len());
    println!("  Avg latency:      {avg:.1}ms");
    println!("  P50 latency:      {p50:.1}ms");
    println!("  P95 latency:      {p95:.1}ms");
    println!("  P99 latency:      {p99:.1}ms");

    engine.persist().await?;
    println!("  Data persisted to disk.");

    Ok(())
}

fn cmd_config_show(config: &EchoConfig) {
    let file_config = config::load_config_file().ok().flatten();
    let has_file = file_config.is_some();

    println!("[shrimpk] Configuration (priority: env > file > auto-detect):");
    println!(
        "  Config file: {} {}",
        config::config_path().display(),
        if has_file { "(exists)" } else { "(not found)" }
    );
    println!();

    let fc = file_config.unwrap_or_default();

    println!("  {:25} {:>15}  Source", "Key", "Value");
    println!("  {:25} {:>15}  ------", "---", "-----");

    let source = |env: &str, file_val: bool| -> &'static str {
        if std::env::var(env).is_ok() {
            "env"
        } else if file_val {
            "file"
        } else {
            "auto"
        }
    };

    println!(
        "  {:25} {:>15}  {}",
        "max_memories",
        format_number(config.max_memories),
        source("SHRIMPK_MAX_MEMORIES", fc.max_memories.is_some())
    );
    println!(
        "  {:25} {:>15}  {}",
        "similarity_threshold",
        format!("{:.2}", config.similarity_threshold),
        source(
            "SHRIMPK_SIMILARITY_THRESHOLD",
            fc.similarity_threshold.is_some()
        )
    );
    println!(
        "  {:25} {:>15}  {}",
        "max_echo_results",
        config.max_echo_results,
        if fc.max_echo_results.is_some() {
            "file"
        } else {
            "auto"
        }
    );
    println!(
        "  {:25} {:>15}  {}",
        "quantization",
        config.quantization.to_string(),
        source("SHRIMPK_QUANTIZATION", fc.quantization.is_some())
    );
    println!(
        "  {:25} {:>15}  {}",
        "max_disk_bytes",
        format_bytes(config.max_disk_bytes),
        source("SHRIMPK_MAX_DISK_BYTES", fc.max_disk_bytes.is_some())
    );
    println!(
        "  {:25} {:>15}  {}",
        "data_dir",
        truncate(&config.data_dir.to_string_lossy(), 30),
        source("SHRIMPK_DATA_DIR", fc.data_dir.is_some())
    );
    println!(
        "  {:25} {:>15}  {}",
        "use_lsh",
        config.use_lsh,
        if fc.use_lsh.is_some() { "file" } else { "auto" }
    );
    println!(
        "  {:25} {:>15}  {}",
        "use_bloom",
        config.use_bloom,
        if fc.use_bloom.is_some() {
            "file"
        } else {
            "auto"
        }
    );
    println!(
        "  {:25} {:>15}  {}",
        "consolidation_provider",
        &config.consolidation_provider,
        source("SHRIMPK_CONSOLIDATION_PROVIDER", fc.consolidation_provider.is_some())
    );
    println!(
        "  {:25} {:>15}  {}",
        "ollama_url",
        &config.ollama_url,
        source("SHRIMPK_OLLAMA_URL", fc.ollama_url.is_some())
    );
    println!(
        "  {:25} {:>15}  {}",
        "enrichment_model",
        &config.enrichment_model,
        source("SHRIMPK_ENRICHMENT_MODEL", fc.enrichment_model.is_some())
    );
    println!(
        "  {:25} {:>15}  {}",
        "max_facts_per_memory",
        config.max_facts_per_memory,
        if fc.max_facts_per_memory.is_some() { "file" } else { "auto" }
    );
}

fn cmd_config_set(key: &str, value: &str) -> anyhow::Result<()> {
    let mut fc = config::load_config_file()?.unwrap_or_default();

    match key {
        "max_memories" => fc.max_memories = Some(value.parse()?),
        "similarity_threshold" => fc.similarity_threshold = Some(value.parse()?),
        "max_echo_results" => fc.max_echo_results = Some(value.parse()?),
        "ram_budget_bytes" => fc.ram_budget_bytes = Some(value.parse()?),
        "max_disk_bytes" => fc.max_disk_bytes = Some(value.parse()?),
        "embedding_dim" => fc.embedding_dim = Some(value.parse()?),
        "use_lsh" => fc.use_lsh = Some(value.parse()?),
        "use_bloom" => fc.use_bloom = Some(value.parse()?),
        "quantization" => {
            fc.quantization = Some(value.parse().map_err(|e: String| anyhow::anyhow!(e))?)
        }
        "data_dir" => fc.data_dir = Some(std::path::PathBuf::from(value)),
        "ollama_url" => fc.ollama_url = Some(value.to_string()),
        "enrichment_model" => fc.enrichment_model = Some(value.to_string()),
        "consolidation_provider" => fc.consolidation_provider = Some(value.to_string()),
        "max_facts_per_memory" => fc.max_facts_per_memory = Some(value.parse()?),
        other => anyhow::bail!("Unknown config key: \"{other}\""),
    }

    config::save_config_file(&fc)?;
    println!(
        "[shrimpk] Set {key} = {value} in {}",
        config::config_path().display()
    );

    Ok(())
}

fn cmd_config_reset() -> anyhow::Result<()> {
    let path = config::config_path();
    if path.exists() {
        std::fs::remove_file(&path)?;
        println!("[shrimpk] Removed {}", path.display());
    } else {
        println!("[shrimpk] No config file to reset ({})", path.display());
    }
    println!("[shrimpk] Using auto-detect defaults.");
    Ok(())
}

async fn cmd_status(config: &EchoConfig) -> anyhow::Result<()> {
    let ram_gb = detect_ram_gb();
    let disk_usage = config::disk_usage(&config.data_dir).unwrap_or(0);
    let disk_pct = if config.max_disk_bytes > 0 {
        (disk_usage as f64 / config.max_disk_bytes as f64 * 100.0) as u64
    } else {
        0
    };

    let bar_width = 30;
    let filled = (disk_pct as usize * bar_width / 100).min(bar_width);
    let bar: String = format!("[{}{}]", "#".repeat(filled), "-".repeat(bar_width - filled));

    println!("[shrimpk] System Status:");
    println!(
        "  Config tier:   {} ({ram_gb}GB RAM detected)",
        tier_name(config)
    );
    println!("  Data dir:      {}", config.data_dir.display());
    println!(
        "  Disk usage:    {} / {} ({}%) {}",
        format_bytes(disk_usage),
        format_bytes(config.max_disk_bytes),
        disk_pct,
        bar
    );
    println!("  RAM budget:    {}", format_bytes(config.ram_budget_bytes));
    println!("  Quantization:  {}", config.quantization);
    println!("  Max memories:  {}", format_number(config.max_memories));

    if disk_pct >= 80 {
        println!();
        println!(
            "  WARNING: Disk usage above 80%. Consider increasing max_disk_bytes or cleaning data."
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Handle config/status subcommands that don't need the engine
    match &cli.command {
        Commands::Config { action } => {
            let config =
                config::resolve_config().map_err(|e| anyhow::anyhow!("Config error: {e}"))?;
            match action {
                ConfigAction::Show => {
                    cmd_config_show(&config);
                    return Ok(());
                }
                ConfigAction::Set { key, value } => {
                    cmd_config_set(key, value)?;
                    return Ok(());
                }
                ConfigAction::Reset => {
                    cmd_config_reset()?;
                    return Ok(());
                }
                ConfigAction::Path => {
                    println!("{}", config::config_path().display());
                    return Ok(());
                }
            }
        }
        Commands::Status => {
            let config =
                config::resolve_config().map_err(|e| anyhow::anyhow!("Config error: {e}"))?;
            cmd_status(&config).await?;
            return Ok(());
        }
        _ => {}
    }

    // Try daemon first (instant response, no model loading)
    // Fall back to in-process engine if daemon not running
    let daemon_url = daemon_base_url();
    let quiet = matches!(&cli.command, Commands::Echo { json: true, .. })
        || matches!(&cli.command, Commands::Store { quiet: true, .. });

    if let Some(base) = &daemon_url {
        if !quiet {
            eprintln!("[shrimpk] Using daemon at {base}");
        }
        match &cli.command {
            Commands::Store {
                text,
                source,
                quiet: q,
            } => {
                let resp = daemon_post(
                    base,
                    "/api/store",
                    &serde_json::json!({"text": text, "source": source}),
                )
                .await?;
                if !q {
                    println!("{resp}");
                }
            }
            Commands::Echo {
                query,
                max_results,
                json,
            } => {
                let resp = daemon_post(
                    base,
                    "/api/echo",
                    &serde_json::json!({"query": query, "max_results": max_results}),
                )
                .await?;
                if *json {
                    // Extract just the results array for clean JSON output
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&resp) {
                        let results = &parsed["results"];
                        let slim: Vec<serde_json::Value> = results
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .map(|r| {
                                        serde_json::json!({
                                            "content": r["content"],
                                            "similarity": r["similarity"],
                                            "source": r["source"]
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        println!("{}", serde_json::to_string(&slim)?);
                    } else {
                        println!("{resp}");
                    }
                } else {
                    println!("{resp}");
                }
            }
            Commands::Stats => {
                let resp = daemon_get(base, "/api/stats").await?;
                println!("{resp}");
            }
            Commands::Forget { id } => {
                let resp = daemon_delete(base, &format!("/api/memories/{id}")).await?;
                println!("{resp}");
            }
            Commands::Dump => {
                let resp = daemon_get(base, "/api/memories?limit=50").await?;
                println!("{resp}");
            }
            Commands::Bench { .. } => {
                anyhow::bail!("Bench requires in-process engine. Stop daemon and retry.");
            }
            Commands::Config { .. } | Commands::Status => unreachable!(),
        }
        return Ok(());
    }

    // Fallback: no daemon running — load engine in-process
    let config = config::resolve_config().map_err(|e| anyhow::anyhow!("Config error: {e}"))?;
    std::fs::create_dir_all(&config.data_dir)?;

    if !quiet {
        println!(
            "[shrimpk] No daemon detected. Loading engine in-process ({} tier)...",
            tier_name(&config)
        );
    }

    let engine = EchoEngine::load(config.clone())?;

    let initial_count = engine.stats().await.total_memories;
    if initial_count > 0 && !quiet {
        println!(
            "[shrimpk] Loaded {} existing memories from disk.",
            format_number(initial_count)
        );
    }

    match cli.command {
        Commands::Store {
            text,
            source,
            quiet: q,
        } => {
            if q {
                engine.store(&text, &source).await?;
                engine.persist().await?;
            } else {
                cmd_store(&engine, &text, &source).await?;
            }
        }
        Commands::Echo {
            query,
            max_results,
            json,
        } => {
            if json {
                cmd_echo_json(&engine, &query, max_results).await?
            } else {
                cmd_echo(&engine, &query, max_results).await?
            }
        }
        Commands::Stats => cmd_stats(&engine, &config).await?,
        Commands::Forget { id } => cmd_forget(&engine, &id).await?,
        Commands::Dump => cmd_dump(&engine, &config).await?,
        Commands::Bench { count } => {
            if count == 0 {
                anyhow::bail!("Count must be at least 1");
            }
            cmd_bench(&engine, count).await?;
        }
        Commands::Config { .. } | Commands::Status => unreachable!(),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Daemon client helpers
// ---------------------------------------------------------------------------

const DEFAULT_DAEMON_PORT: u16 = 11435;

/// Check if daemon is running and return its base URL.
/// Uses a fast TCP connect probe (no HTTP overhead, no runtime conflict).
fn daemon_base_url() -> Option<String> {
    let port = std::env::var("SHRIMPK_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_DAEMON_PORT);

    // Fast TCP probe — just check if something is listening on the port
    let addr = format!("127.0.0.1:{port}");
    std::net::TcpStream::connect_timeout(&addr.parse().ok()?, std::time::Duration::from_millis(100))
        .ok()
        .map(|_| format!("http://{addr}"))
}

fn daemon_client() -> reqwest::Client {
    let mut builder = reqwest::Client::builder();
    // Forward auth token if set (F-02 fix)
    if let Ok(token) = std::env::var("SHRIMPK_AUTH_TOKEN") {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")) {
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
        builder = builder.default_headers(headers);
    }
    builder.build().unwrap_or_default()
}

async fn daemon_post(base: &str, path: &str, body: &serde_json::Value) -> anyhow::Result<String> {
    let resp = daemon_client()
        .post(format!("{base}{path}"))
        .json(body)
        .send()
        .await?;
    Ok(resp.text().await?)
}

async fn daemon_get(base: &str, path: &str) -> anyhow::Result<String> {
    let resp = daemon_client().get(format!("{base}{path}")).send().await?;
    Ok(resp.text().await?)
}

async fn daemon_delete(base: &str, path: &str) -> anyhow::Result<String> {
    let resp = daemon_client()
        .delete(format!("{base}{path}"))
        .send()
        .await?;
    Ok(resp.text().await?)
}
