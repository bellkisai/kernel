//! Regression diagnostic for the 6 failing LongMemEval tests.
//! Runs each test with 6 pipeline configs to isolate HyDE vs reranker issues.
//! Configs A-D use LLM reranker (Ollama), E-F use fastembed cross-encoder.
//!
//!     cargo test --test echo_regression_diagnostic -- --ignored --nocapture --test-threads=1

use shrimpk_core::{EchoConfig, RerankerBackend};
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

fn top_n_contains(results: &[shrimpk_core::EchoResult], n: usize, needle: &str) -> bool {
    let lc = needle.to_lowercase();
    results.iter().take(n).any(|r| r.content.to_lowercase().contains(&lc))
}

fn any_needle_in_top_n(results: &[shrimpk_core::EchoResult], n: usize, needles: &[&str]) -> bool {
    needles.iter().any(|needle| top_n_contains(results, n, needle))
}

fn make_config(data_dir: PathBuf, hyde: bool, reranker: bool, backend: RerankerBackend) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15,
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        data_dir,
        embedding_dim: 384,
        consolidation_provider: "ollama".to_string(),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        enrichment_model: "llama3.2:3b".to_string(),
        consolidation_consent_given: true,
        query_expansion_enabled: hyde,
        reranker_enabled: reranker,
        reranker_backend: backend,
        ..Default::default()
    }
}

fn run_consolidation(engine: &EchoEngine) {
    std::thread::scope(|s| {
        s.spawn(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            for pass in 1..=10 {
                let r = rt.block_on(engine.consolidate_now());
                if r.facts_extracted == 0 { break; }
                println!("    consolidation pass {pass}: facts={}", r.facts_extracted);
            }
        }).join().expect("consolidation panicked");
    });
}

fn print_results(results: &[shrimpk_core::EchoResult], label: &str, needles: &[&str]) {
    let hit3 = any_needle_in_top_n(results, 3, needles);
    let hit5 = any_needle_in_top_n(results, 5, needles);
    println!("  [{label}]");
    for (i, r) in results.iter().take(5).enumerate() {
        let content = &r.content[..r.content.len().min(90)];
        println!("    #{}: sim={:.3} score={:.3} | {}", i+1, r.similarity, r.final_score, content);
    }
    println!("    Verdict: top3={} top5={}", if hit3 {"PASS"} else {"MISS"}, if hit5 {"PASS"} else {"MISS"});
    println!();
}

struct TestCase {
    name: &'static str,
    query: &'static str,
    needles: Vec<&'static str>,
    memories: Vec<(&'static str, &'static str)>, // (text, source)
}

fn test_cases() -> Vec<TestCase> {
    vec![
        TestCase {
            name: "PT-4: OS",
            query: "What operating system do I use?",
            needles: vec!["Arch", "Hyprland"],
            memories: vec![
                ("I use Windows 11 on all my machines for gaming and development", "month1"),
                ("I dual-boot Linux Mint alongside Windows now for dev work", "month3"),
                ("I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely", "month6"),
                ("I bought a new 4K monitor for my desk setup", "noise"),
            ],
        },
        TestCase {
            name: "PT-6: Commute",
            query: "How do I get to work?",
            needles: vec!["bike"],
            memories: vec![
                ("I drive my car to work every day, about a 40-minute commute on the highway", "month1"),
                ("I started taking the BART train to reduce my carbon footprint", "month4"),
                ("I now bike to work every day, it's a 25-minute ride and I love the fresh air", "month8"),
                ("Gas prices have been going up a lot this year", "noise"),
            ],
        },
        TestCase {
            name: "PT-10: Cooking",
            query: "What do I like to cook?",
            needles: vec!["Thai", "Japanese", "pad see ew", "ramen"],
            memories: vec![
                ("I mostly order takeout and eat at restaurants, I don't cook much", "month1"),
                ("I started meal prepping on Sundays, mostly simple pasta and salad recipes", "month4"),
                ("I've gotten into Thai and Japanese home cooking, I make pad see ew and ramen from scratch now", "month8"),
                ("I need to buy a new set of kitchen knives", "noise"),
            ],
        },
        TestCase {
            name: "TR-7: Career Timing",
            query: "When did I start working at my current company?",
            needles: vec!["2022", "Stripe"],
            memories: vec![
                ("In 2016 I got my first programming job at a small Vancouver startup", "career1"),
                ("In 2019 I joined Shopify as a backend engineer on their payments team", "career2"),
                ("In January 2022 I started at Stripe as a senior backend engineer", "career3"),
                ("I've been at Stripe for over 2 years now and I'm up for a promotion", "career4"),
                ("My friend just started a new job at Apple last week", "noise"),
            ],
        },
        TestCase {
            name: "IE-9: Vehicle",
            query: "What car do I drive?",
            needles: vec!["Tesla", "Model 3"],
            memories: vec![
                ("I drive a 2022 Tesla Model 3 but mostly bike to work on my Canyon Grail gravel bike", "profile"),
                ("I work as a senior backend engineer at Stripe in San Francisco", "profile"),
                ("I have a golden retriever named Pixel who is 4 years old", "profile"),
                ("I practice Brazilian jiu-jitsu three times a week at a Gracie gym", "profile"),
                ("I prefer Rust for systems programming and Go for microservices", "profile"),
                ("My morning routine is meditation, coffee, then 30 minutes of reading", "profile"),
                ("I enjoy listening to classical music while coding, especially Chopin", "profile"),
            ],
        },
        TestCase {
            name: "MSR-2: Travel+Lang",
            query: "Have I traveled anywhere related to languages I'm learning?",
            needles: vec!["Tokyo"],
            memories: vec![
                ("I'm learning Japanese and currently at JLPT N3 level", "session3"),
                ("I visited Tokyo last November and stayed in Shinjuku for two weeks", "session4"),
                ("I was born in Taipei, Taiwan but grew up in Vancouver, Canada", "session1"),
                ("I prefer Rust for systems programming and Go for microservices", "session3"),
                ("My favorite cuisine is Thai food, especially pad see ew and massaman curry", "session3"),
            ],
        },
    ]
}

const NUM_CONFIGS: usize = 6;

#[test]
#[ignore = "requires Ollama with llama3.2:3b"]
fn regression_diagnostic() {
    let configs: Vec<(&str, bool, bool, RerankerBackend)> = vec![
        ("A: Baseline",       false, false, RerankerBackend::None),
        ("B: HyDE only",      true,  false, RerankerBackend::None),
        ("C: Reranker (LLM)", false, true,  RerankerBackend::Llm),
        ("D: Combined (LLM)", true,  true,  RerankerBackend::Llm),
        ("E: CrossEncoder",   false, true,  RerankerBackend::CrossEncoder),
        ("F: CE + HyDE",      true,  true,  RerankerBackend::CrossEncoder),
    ];

    let cases = test_cases();

    // Summary table: name, [pass/fail per config], [latency_ms per config]
    let mut summary: Vec<(String, [bool; NUM_CONFIGS], [u128; NUM_CONFIGS])> = Vec::new();

    for case in &cases {
        println!("\n============================================================");
        println!("=== {} — \"{}\" ===", case.name, case.query);
        println!("  Needles: {:?}", case.needles);
        println!("  Memories: {} stored", case.memories.len());
        for (i, (text, src)) in case.memories.iter().enumerate() {
            println!("    [{}] ({}) {}", i, src, &text[..text.len().min(70)]);
        }
        println!();

        let mut row = [false; NUM_CONFIGS];
        let mut latencies = [0u128; NUM_CONFIGS];

        for (ci, (label, hyde, reranker, backend)) in configs.iter().enumerate() {
            let dir = tempdir().expect("temp dir");
            let config = make_config(dir.path().to_path_buf(), *hyde, *reranker, *backend);
            let engine = EchoEngine::new(config).expect("engine init");

            let rt = tokio::runtime::Runtime::new().unwrap();

            // Store memories
            rt.block_on(async {
                for (text, source) in &case.memories {
                    engine.store(text, source).await.expect("store");
                }
            });

            // Run consolidation
            println!("  [{label}] consolidating...");
            run_consolidation(&engine);

            // Query with latency measurement
            let start = Instant::now();
            let results = rt.block_on(async {
                engine.echo(case.query, 5).await.expect("echo")
            });
            let latency_ms = start.elapsed().as_millis();
            drop(rt);

            let hit3 = any_needle_in_top_n(&results, 3, &case.needles);
            row[ci] = hit3;
            latencies[ci] = latency_ms;
            print_results(&results, &format!("{label} ({latency_ms}ms)"), &case.needles);

            // Explicitly drop engine to free fastembed memory before next config
            drop(engine);
            drop(dir);
        }

        summary.push((case.name.to_string(), row, latencies));
    }

    // Print summary matrix
    println!("\n============================================================");
    println!("=== DIAGNOSTIC SUMMARY (6 configs) ===\n");
    println!(
        "{:<20} | {:^8} | {:^6} | {:^10} | {:^10} | {:^8} | {:^8}",
        "Test", "Baseline", "HyDE", "Rerank(LLM)", "Comb(LLM)", "CE only", "CE+HyDE"
    );
    println!("{}", "-".repeat(90));
    for (name, results, _latencies) in &summary {
        println!(
            "{:<20} | {:^8} | {:^6} | {:^10} | {:^10} | {:^8} | {:^8}",
            name,
            if results[0] {"PASS"} else {"MISS"},
            if results[1] {"PASS"} else {"MISS"},
            if results[2] {"PASS"} else {"MISS"},
            if results[3] {"PASS"} else {"MISS"},
            if results[4] {"PASS"} else {"MISS"},
            if results[5] {"PASS"} else {"MISS"},
        );
    }
    let totals: Vec<usize> = (0..NUM_CONFIGS).map(|i| summary.iter().filter(|(_, r, _)| r[i]).count()).collect();
    println!("{}", "-".repeat(90));
    println!(
        "{:<20} | {:^8} | {:^6} | {:^10} | {:^10} | {:^8} | {:^8}",
        format!("TOTAL ({}/6)", totals.iter().max().unwrap_or(&0)),
        format!("{}/6", totals[0]),
        format!("{}/6", totals[1]),
        format!("{}/6", totals[2]),
        format!("{}/6", totals[3]),
        format!("{}/6", totals[4]),
        format!("{}/6", totals[5]),
    );

    // Latency comparison
    println!("\n=== LATENCY COMPARISON (ms) ===\n");
    println!(
        "{:<20} | {:>8} | {:>6} | {:>10} | {:>10} | {:>8} | {:>8}",
        "Test", "Baseline", "HyDE", "Rerank(LLM)", "Comb(LLM)", "CE only", "CE+HyDE"
    );
    println!("{}", "-".repeat(90));
    for (name, _results, latencies) in &summary {
        println!(
            "{:<20} | {:>6}ms | {:>4}ms | {:>8}ms | {:>8}ms | {:>6}ms | {:>6}ms",
            name,
            latencies[0], latencies[1], latencies[2], latencies[3], latencies[4], latencies[5],
        );
    }
    let avg_latencies: Vec<u128> = (0..NUM_CONFIGS).map(|i| {
        let sum: u128 = summary.iter().map(|(_, _, l)| l[i]).sum();
        sum / summary.len().max(1) as u128
    }).collect();
    println!("{}", "-".repeat(90));
    println!(
        "{:<20} | {:>6}ms | {:>4}ms | {:>8}ms | {:>8}ms | {:>6}ms | {:>6}ms",
        "AVERAGE",
        avg_latencies[0], avg_latencies[1], avg_latencies[2], avg_latencies[3], avg_latencies[4], avg_latencies[5],
    );
}
