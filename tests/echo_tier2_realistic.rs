//! Tier 2: Realistic User Simulation Benchmark
//!
//! A single synthetic user ("Sam Torres") lifecycle spanning 6 sessions over
//! ~60 days. Unlike Tier 1 (isolated per-test engines with 3-5 memories),
//! Tier 2 uses ONE engine with ~40 organically accumulated memories, then
//! fires 25 queries at it. This simulates real-world usage.
//!
//! Follows KS28 Benchmark Design Principles — Principle 5 (Real User Simulation).
//!
//!     cargo test --test echo_tier2_realistic -- --ignored --nocapture --test-threads=1

use shrimpk_core::{EchoConfig, EchoResult, RerankerBackend};
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn top_n_contains(results: &[EchoResult], n: usize, needle: &str) -> bool {
    let lc = needle.to_lowercase();
    results
        .iter()
        .take(n)
        .any(|r| r.content.to_lowercase().contains(&lc))
}

fn any_needle_in_top_n(results: &[EchoResult], n: usize, needles: &[&str]) -> bool {
    needles
        .iter()
        .any(|needle| top_n_contains(results, n, needle))
}

fn make_config(
    data_dir: PathBuf,
    hyde: bool,
    reranker: bool,
    backend: RerankerBackend,
) -> EchoConfig {
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

fn run_consolidation(engine: &EchoEngine) -> usize {
    let mut total = 0;
    std::thread::scope(|s| {
        let result = s
            .spawn(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut facts = 0;
                for pass in 1..=10 {
                    let r = rt.block_on(engine.consolidate_now());
                    if r.facts_extracted == 0 {
                        break;
                    }
                    println!(
                        "    consolidation pass {pass}: facts={}, merged={}",
                        r.facts_extracted, r.duplicates_merged
                    );
                    facts += r.facts_extracted;
                }
                facts
            })
            .join()
            .expect("consolidation panicked");
        total = result;
    });
    total
}

// ---------------------------------------------------------------------------
// Sam Torres — User Persona
// ---------------------------------------------------------------------------

/// Returns ~40 memories across 6 sessions simulating a real user over 60 days.
/// Each tuple is (text, session_source_tag).
fn sam_lifecycle() -> Vec<(&'static str, &'static str)> {
    vec![
        // === Session 1 (Day 1): Personal basics ===
        ("My name is Sam Torres, I'm 29 years old", "session1_day1"),
        (
            "I live in a one-bedroom apartment in Oakland, California",
            "session1_day1",
        ),
        (
            "I work as a frontend developer at Figma on their design tools team",
            "session1_day1",
        ),
        (
            "I have a tabby cat named Mochi who is 3 years old",
            "session1_day1",
        ),
        (
            "I grew up in Sacramento and got my CS degree from UC Davis in 2019",
            "session1_day1",
        ),
        (
            "My parents still live in Sacramento, I visit them once a month",
            "session1_day1",
        ),
        // === Session 2 (Day 3): Work context ===
        (
            "My team at Figma works on the real-time collaboration features like multiplayer cursors and live comments",
            "session2_day3",
        ),
        (
            "We use TypeScript and React for the frontend with a custom WebGL renderer for the canvas",
            "session2_day3",
        ),
        (
            "I'm currently working on improving the performance of multiplayer editing for large files",
            "session2_day3",
        ),
        (
            "My manager is Lisa Chen and our team has 6 engineers including me",
            "session2_day3",
        ),
        (
            "We do two-week sprints and I'm the on-call rotation lead this month",
            "session2_day3",
        ),
        (
            "Our biggest competitor is Penpot and the team watches their releases closely",
            "session2_day3",
        ),
        (
            "I sit in the 4th floor open office area next to the coffee station",
            "session2_day3",
        ),
        // === Session 3 (Day 7): Personal preferences ===
        (
            "I use VS Code with the Dracula theme and Vim keybindings for all my coding",
            "session3_day7",
        ),
        (
            "I'm vegetarian and love cooking Indian food at home, especially daal and paneer tikka masala",
            "session3_day7",
        ),
        (
            "I run 5K three times a week, usually in the morning before work around Lake Merritt",
            "session3_day7",
        ),
        (
            "I take BART to the Figma office in San Francisco, about a 35-minute commute each way",
            "session3_day7",
        ),
        (
            "I'm reading Designing Data-Intensive Applications by Martin Kleppmann, about halfway through",
            "session3_day7",
        ),
        (
            "I drink oat milk lattes, usually grab one from Blue Bottle Coffee near the office",
            "session3_day7",
        ),
        (
            "I play acoustic guitar in the evenings, mostly folk and indie songs",
            "session3_day7",
        ),
        // === Session 4 (Day 14): Work update ===
        (
            "We shipped the new real-time collaboration engine last week and got great user feedback on performance improvements",
            "session4_day14",
        ),
        (
            "I've been learning Rust on the side, working through the Rust Book and building a small CLI tool",
            "session4_day14",
        ),
        (
            "Lisa asked me to lead the new plugin API project starting next sprint, it's a big opportunity",
            "session4_day14",
        ),
        (
            "I got a new 32-inch 4K LG monitor and a standing desk from Autonomous for my home office",
            "session4_day14",
        ),
        (
            "Our team is hiring two more engineers and I'm helping conduct technical interviews",
            "session4_day14",
        ),
        (
            "I signed up for a half marathon in June, so I'm increasing my running distance gradually",
            "session4_day14",
        ),
        // === Session 5 (Day 30): Preference changes ===
        (
            "I switched from VS Code to Neovim with a custom Lua config, took me two weeks to set up but I love it",
            "session5_day30",
        ),
        (
            "I had to stop running because of a knee injury from over-training for the half marathon",
            "session5_day30",
        ),
        (
            "I started doing yoga three times a week at CorePower instead of running, it's easier on my joints",
            "session5_day30",
        ),
        (
            "I've gotten into sourdough baking, made my first decent loaf last weekend after three failed attempts",
            "session5_day30",
        ),
        (
            "I finished DDIA and now I'm reading The Pragmatic Programmer, finding it really practical",
            "session5_day30",
        ),
        (
            "I started biking to work instead of taking BART, got a used Trek road bike for the commute",
            "session5_day30",
        ),
        (
            "I'm learning Japanese on Duolingo, about 30 days into my streak now",
            "session5_day30",
        ),
        // === Session 6 (Day 60): Major life updates ===
        (
            "I left Figma last month and joined Vercel as a senior frontend engineer on the Next.js team",
            "session6_day60",
        ),
        (
            "I moved from Oakland to San Francisco, got an apartment in the Mission District to be closer to work",
            "session6_day60",
        ),
        (
            "Jordan and I started dating three weeks ago, we met at the SF Rust meetup",
            "session6_day60",
        ),
        (
            "At Vercel I'm working on Next.js server components and the edge runtime",
            "session6_day60",
        ),
        (
            "I adopted a second cat, a black kitten named Pixel, Mochi is adjusting to having a sibling",
            "session6_day60",
        ),
        (
            "I passed my Rust skills assessment and I'm now contributing to an open source Rust project on weekends",
            "session6_day60",
        ),
        (
            "I switched to making pour-over coffee at home with a Chemex after visiting a specialty roaster",
            "session6_day60",
        ),
        (
            "My new commute is a 15-minute bike ride which is way better than the 35-minute BART from Oakland",
            "session6_day60",
        ),
    ]
}

// ---------------------------------------------------------------------------
// Query definitions
// ---------------------------------------------------------------------------

struct QueryCase {
    name: &'static str,
    category: &'static str,
    query: &'static str,
    needles: Vec<&'static str>,
}

fn tier2_queries() -> Vec<QueryCase> {
    vec![
        // === IE: Information Extraction (5) — Direct fact recall ===
        QueryCase {
            name: "IE-1: Pets",
            category: "IE",
            query: "What pets do I have?",
            needles: vec!["Mochi", "Pixel"],
        },
        QueryCase {
            name: "IE-2: Current Job",
            category: "IE",
            query: "Where do I currently work?",
            needles: vec!["Vercel"],
        },
        QueryCase {
            name: "IE-3: Education",
            category: "IE",
            query: "Where did I go to college and what did I study?",
            needles: vec!["UC Davis"],
        },
        QueryCase {
            name: "IE-4: Partner",
            category: "IE",
            query: "Am I in a relationship? Who am I dating?",
            needles: vec!["Jordan"],
        },
        QueryCase {
            name: "IE-5: Neighborhood",
            category: "IE",
            query: "What neighborhood do I live in?",
            needles: vec!["Mission"],
        },
        // === MSR: Multi-Session Reasoning (5) — Cross-session connections ===
        QueryCase {
            name: "MSR-1: Skills for Job",
            category: "MSR",
            query: "What programming skills do I have that are relevant to my current role?",
            needles: vec!["TypeScript", "React"],
        },
        QueryCase {
            name: "MSR-2: How Met Partner",
            category: "MSR",
            query: "How did I meet my partner?",
            needles: vec!["Rust meetup"],
        },
        QueryCase {
            name: "MSR-3: Career Tech Stack",
            category: "MSR",
            query: "What technologies have I worked with across my jobs?",
            needles: vec!["TypeScript", "React"],
        },
        QueryCase {
            name: "MSR-4: Hobby to Career",
            category: "MSR",
            query: "How have my side projects influenced my career?",
            needles: vec!["Rust"],
        },
        QueryCase {
            name: "MSR-5: City and Commute",
            category: "MSR",
            query: "How has my living situation affected my commute?",
            needles: vec!["Mission", "bike", "15-minute"],
        },
        // === TR: Temporal Reasoning (5) — Time-ordered recall ===
        QueryCase {
            name: "TR-1: Job Change",
            category: "TR",
            query: "What was my most recent job change?",
            needles: vec!["Vercel", "Figma"],
        },
        QueryCase {
            name: "TR-2: Exercise History",
            category: "TR",
            query: "How has my exercise routine changed over time?",
            needles: vec!["yoga"],
        },
        QueryCase {
            name: "TR-3: Reading List",
            category: "TR",
            query: "What books have I read recently, in what order?",
            needles: vec!["Pragmatic Programmer"],
        },
        QueryCase {
            name: "TR-4: Moving History",
            category: "TR",
            query: "Where have I lived and when did I move?",
            needles: vec!["San Francisco", "Oakland"],
        },
        QueryCase {
            name: "TR-5: Career Timeline",
            category: "TR",
            query: "What's the timeline of my career so far?",
            needles: vec!["Figma", "Vercel"],
        },
        // === KU: Knowledge Update (5) — Most recent info surfaces ===
        QueryCase {
            name: "KU-1: Current Editor",
            category: "KU",
            query: "What code editor do I use?",
            needles: vec!["Neovim"],
        },
        QueryCase {
            name: "KU-2: Current Commute",
            category: "KU",
            query: "How do I get to work?",
            needles: vec!["bike"],
        },
        QueryCase {
            name: "KU-3: Current Coffee",
            category: "KU",
            query: "What coffee do I drink?",
            needles: vec!["pour-over", "Chemex"],
        },
        QueryCase {
            name: "KU-4: Current Book",
            category: "KU",
            query: "What am I currently reading?",
            needles: vec!["Pragmatic Programmer"],
        },
        QueryCase {
            name: "KU-5: Current Exercise",
            category: "KU",
            query: "What exercise do I do regularly?",
            needles: vec!["yoga"],
        },
        // === PT: Preference Tracking (5) — Evolving preferences ===
        QueryCase {
            name: "PT-1: Current Tech Stack",
            category: "PT",
            query: "What framework am I working with now?",
            needles: vec!["Next.js", "server components"],
        },
        QueryCase {
            name: "PT-2: Cooking Interests",
            category: "PT",
            query: "What do I cook at home?",
            needles: vec!["Indian", "sourdough"],
        },
        QueryCase {
            name: "PT-3: Morning Routine",
            category: "PT",
            query: "What does my morning look like?",
            needles: vec!["yoga"],
        },
        QueryCase {
            name: "PT-4: Languages Learning",
            category: "PT",
            query: "Am I learning any new languages?",
            needles: vec!["Japanese"],
        },
        QueryCase {
            name: "PT-5: Music and Hobbies",
            category: "PT",
            query: "What hobbies do I have outside of work?",
            needles: vec!["guitar", "yoga", "sourdough"],
        },
    ]
}

// ---------------------------------------------------------------------------
// Result formatting
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[allow(dead_code)]
struct TestResult {
    name: &'static str,
    category: &'static str,
    top3: bool,
    top5: bool,
    latency_ms: u128,
}

fn print_result_detail(
    results: &[EchoResult],
    case: &QueryCase,
    config_label: &str,
    hit3: bool,
    hit5: bool,
    latency_ms: u128,
) {
    println!(
        "  [{config_label}] {}: top3={} top5={} ({latency_ms}ms)",
        case.name,
        if hit3 { "PASS" } else { "MISS" },
        if hit5 { "PASS" } else { "MISS" },
    );
    for (i, r) in results.iter().take(5).enumerate() {
        let content = &r.content[..r.content.len().min(80)];
        println!(
            "    #{}: sim={:.3} score={:.3} | {}",
            i + 1,
            r.similarity,
            r.final_score,
            content
        );
    }
}

fn print_scorecard(config_label: &str, results: &[TestResult]) {
    let categories = ["IE", "MSR", "TR", "KU", "PT"];

    println!("\n  [{config_label}] SCORECARD");
    println!(
        "  {:<30} | {:>5} | {:>5} | {:>6}",
        "Category", "Top-3", "Top-5", "Avg ms"
    );
    println!("  {}", "-".repeat(60));

    for cat in &categories {
        let cat_results: Vec<&TestResult> = results.iter().filter(|r| r.category == *cat).collect();
        let total = cat_results.len();
        let s3 = cat_results.iter().filter(|r| r.top3).count();
        let s5 = cat_results.iter().filter(|r| r.top5).count();
        let avg_ms = if total > 0 {
            cat_results.iter().map(|r| r.latency_ms).sum::<u128>() / total as u128
        } else {
            0
        };
        println!(
            "  {:<30} | {}/{:<3} | {}/{:<3} | {:>4}ms",
            format!("{cat} ({total} tests)"),
            s3,
            total,
            s5,
            total,
            avg_ms,
        );
    }

    let total = results.len();
    let strict = results.iter().filter(|r| r.top3).count();
    let relaxed = results.iter().filter(|r| r.top5).count();
    let avg_ms = if total > 0 {
        results.iter().map(|r| r.latency_ms).sum::<u128>() / total as u128
    } else {
        0
    };
    let strict_pct = (strict as f64 / total as f64) * 100.0;
    let relaxed_pct = (relaxed as f64 / total as f64) * 100.0;

    println!("  {}", "-".repeat(60));
    println!(
        "  {:<30} | {}/{:<3} | {}/{:<3} | {:>4}ms",
        "TOTAL", strict, total, relaxed, total, avg_ms,
    );
    println!(
        "  Top-3: {strict}/{total} ({strict_pct:.1}%)  Top-5: {relaxed}/{total} ({relaxed_pct:.1}%)"
    );
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Ollama with llama3.2:3b for consolidation + HyDE + reranker"]
fn tier2_realistic_user_simulation() {
    println!("\n================================================================");
    println!("=== TIER 2: Realistic User Simulation — Sam Torres Lifecycle ===");
    println!("================================================================\n");

    let configs: Vec<(&str, bool, bool, RerankerBackend)> = vec![
        ("A: Baseline", false, false, RerankerBackend::None),
        ("B: HyDE only", true, false, RerankerBackend::None),
        ("C: Reranker (LLM)", false, true, RerankerBackend::Llm),
        ("D: Combined (HyDE+LLM)", true, true, RerankerBackend::Llm),
    ];

    let memories = sam_lifecycle();
    let queries = tier2_queries();

    // Summary across all configs
    let mut summary: Vec<(String, usize, usize, usize, u128)> = Vec::new(); // (label, top3, top5, total, avg_ms)

    for (label, hyde, reranker, backend) in &configs {
        println!("\n============================================================");
        println!("=== Config: {label} ===");
        println!("============================================================");

        let dir = tempdir().expect("temp dir");
        let config = make_config(dir.path().to_path_buf(), *hyde, *reranker, *backend);
        let engine = EchoEngine::new(config).expect("engine init");
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Store all memories in session order
        println!("  Storing {} memories...", memories.len());
        rt.block_on(async {
            for (text, source) in &memories {
                engine.store(text, source).await.expect("store");
            }
        });

        // Run consolidation (LLM fact extraction)
        println!("  Running consolidation...");
        let facts = run_consolidation(&engine);
        println!("  Consolidation complete: {facts} facts extracted\n");

        // Run all queries
        let mut config_results: Vec<TestResult> = Vec::new();

        for case in &queries {
            let start = Instant::now();
            let results = rt.block_on(async { engine.echo(case.query, 5).await.expect("echo") });
            let latency_ms = start.elapsed().as_millis();

            let hit3 = any_needle_in_top_n(&results, 3, &case.needles);
            let hit5 = any_needle_in_top_n(&results, 5, &case.needles);

            print_result_detail(&results, case, label, hit3, hit5, latency_ms);

            config_results.push(TestResult {
                name: case.name,
                category: case.category,
                top3: hit3,
                top5: hit5,
                latency_ms,
            });
        }

        // Print per-config scorecard
        print_scorecard(label, &config_results);

        let total = config_results.len();
        let s3 = config_results.iter().filter(|r| r.top3).count();
        let s5 = config_results.iter().filter(|r| r.top5).count();
        let avg = if total > 0 {
            config_results.iter().map(|r| r.latency_ms).sum::<u128>() / total as u128
        } else {
            0
        };
        summary.push((label.to_string(), s3, s5, total, avg));

        drop(rt);
        drop(engine);
        drop(dir);
    }

    // =========================================================================
    // Final comparison matrix
    // =========================================================================
    println!("\n================================================================");
    println!("=== TIER 2 FINAL COMPARISON ===");
    println!("================================================================\n");
    println!(
        "{:<30} | {:>9} | {:>9} | {:>8}",
        "Config", "Top-3", "Top-5", "Avg(ms)"
    );
    println!("{}", "-".repeat(65));
    for (label, s3, s5, total, avg) in &summary {
        let pct3 = (*s3 as f64 / *total as f64) * 100.0;
        let pct5 = (*s5 as f64 / *total as f64) * 100.0;
        println!(
            "{:<30} | {}/{} ({:>5.1}%) | {}/{} ({:>5.1}%) | {:>5}ms",
            label, s3, total, pct3, s5, total, pct5, avg,
        );
    }
    println!();

    // Print memory profile summary
    println!("Benchmark profile:");
    println!("  User: Sam Torres (software engineer, Bay Area)");
    println!("  Sessions: 6 (over ~60 days)");
    println!("  Memories stored: {}", memories.len());
    println!("  Queries: {}", queries.len());
    println!("  Categories: IE(5) MSR(5) TR(5) KU(5) PT(5)");
    println!("  Noise level: Organic (no artificial noise, all memories are real user data)");
    println!("\n=== TIER 2 BENCHMARK COMPLETE ===");
}

// ---------------------------------------------------------------------------
// Standalone baseline test (no Ollama required)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "integration test — uses fastembed embeddings"]
fn tier2_baseline_only() {
    println!("\n=== TIER 2 BASELINE (no HyDE, no reranker, no consolidation) ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_config(
        dir.path().to_path_buf(),
        false,
        false,
        RerankerBackend::None,
    );
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    let memories = sam_lifecycle();
    println!("Storing {} memories...", memories.len());
    rt.block_on(async {
        for (text, source) in &memories {
            engine.store(text, source).await.expect("store");
        }
    });

    // NO consolidation — pure embedding pipeline
    let queries = tier2_queries();
    let mut results: Vec<TestResult> = Vec::new();

    for case in &queries {
        let start = Instant::now();
        let res = rt.block_on(async { engine.echo(case.query, 5).await.expect("echo") });
        let latency_ms = start.elapsed().as_millis();

        let hit3 = any_needle_in_top_n(&res, 3, &case.needles);
        let hit5 = any_needle_in_top_n(&res, 5, &case.needles);

        println!(
            "  {}: top3={} top5={} ({latency_ms}ms)",
            case.name,
            if hit3 { "PASS" } else { "MISS" },
            if hit5 { "PASS" } else { "MISS" },
        );

        results.push(TestResult {
            name: case.name,
            category: case.category,
            top3: hit3,
            top5: hit5,
            latency_ms,
        });
    }

    print_scorecard("Baseline (no consolidation)", &results);

    drop(rt);
    drop(engine);
    drop(dir);

    println!("\n=== TIER 2 BASELINE COMPLETE ===");
}
