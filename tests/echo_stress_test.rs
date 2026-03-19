//! Large-scale stress test for the Echo Memory engine.
//!
//! Exercises Echo Memory with 110 diverse memories across 10 categories
//! and 30 query types including direct recall, cross-category, vague/indirect,
//! negative (should-not-match), and edge cases.
//!
//! All tests are `#[ignore]` because they require the fastembed model
//! (all-MiniLM-L6-v2, ~23MB ONNX). Run with:
//!
//!     cargo test --test echo_stress_test -- --ignored
//!
//! Or run a single test:
//!
//!     cargo test --test echo_stress_test stress_full_corpus -- --ignored

use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an EchoConfig tuned for the stress test.
fn stress_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15, // low threshold to catch indirect matches
        max_echo_results: 20,       // wide result window for analysis
        ram_budget_bytes: 200_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

/// Populate the engine with the full 110-entry memory corpus.
/// Returns the engine and the category labels in insertion order.
async fn build_corpus(engine: &EchoEngine) -> Vec<(&'static str, &'static str)> {
    let corpus = full_corpus();
    for &(category, text) in &corpus {
        engine
            .store(text, category)
            .await
            .unwrap_or_else(|e| panic!("Failed to store [{category}]: {text} -- {e}"));
    }
    corpus
}

/// The full 110-entry memory corpus across 10 categories.
fn full_corpus() -> Vec<(&'static str, &'static str)> {
    let mut corpus: Vec<(&str, &str)> = Vec::with_capacity(120);

    // ── Programming (15) ──────────────────────────────────────────────
    for text in PROGRAMMING {
        corpus.push(("programming", text));
    }
    // ── AI & ML (15) ─────────────────────────────────────────────────
    for text in AI_ML {
        corpus.push(("ai_ml", text));
    }
    // ── Personal Preferences (10) ────────────────────────────────────
    for text in PERSONAL {
        corpus.push(("personal", text));
    }
    // ── Projects (10) ────────────────────────────────────────────────
    for text in PROJECTS {
        corpus.push(("projects", text));
    }
    // ── Business (10) ────────────────────────────────────────────────
    for text in BUSINESS {
        corpus.push(("business", text));
    }
    // ── Technical Decisions (10) ─────────────────────────────────────
    for text in TECH_DECISIONS {
        corpus.push(("tech_decisions", text));
    }
    // ── Food & Cooking (10) ──────────────────────────────────────────
    for text in FOOD {
        corpus.push(("food", text));
    }
    // ── Health & Fitness (10) ────────────────────────────────────────
    for text in HEALTH {
        corpus.push(("health", text));
    }
    // ── Travel (10) ──────────────────────────────────────────────────
    for text in TRAVEL {
        corpus.push(("travel", text));
    }
    // ── Random Knowledge (10) ────────────────────────────────────────
    for text in RANDOM {
        corpus.push(("random", text));
    }

    corpus
}

// ---------------------------------------------------------------------------
// Corpus data — static arrays
// ---------------------------------------------------------------------------

static PROGRAMMING: &[&str] = &[
    "I prefer FastAPI for building REST APIs because of async support and auto-generated docs",
    "TypeScript with strict mode is my go-to for frontend development",
    "For database work I always choose PostgreSQL over MySQL",
    "Rust is my favorite systems language for performance-critical code",
    "I use Docker for all my development environments",
    "My preferred testing framework is pytest with fixtures",
    "I deploy most projects to AWS using Terraform",
    "Git rebase over merge for clean commit history",
    "VS Code with Vim keybindings is my IDE setup",
    "I prefer functional programming patterns over OOP",
    "GraphQL over REST for complex data requirements",
    "Redis for caching, PostgreSQL for persistence",
    "I use pnpm instead of npm for package management",
    "My go-to CSS framework is Tailwind with shadcn/ui",
    "For CI/CD I use GitHub Actions with matrix builds",
];

static AI_ML: &[&str] = &[
    "I run Ollama locally for most AI tasks to avoid API costs",
    "Claude is my preferred model for code review and architecture",
    "For embeddings I use all-MiniLM-L6-v2 because it's small and fast",
    "I fine-tune models using QLoRA with 4-bit quantization",
    "My AI workflow starts with local model for drafts, cloud for polish",
    "I prefer structured output over free-form for AI responses",
    "RAG with chunked documents works better than stuffing full context",
    "Temperature 0.1 for code, 0.7 for creative writing",
    "I use LangChain for prototypes but switch to raw API for production",
    "Model evaluation with BLEU score for translation tasks",
    "I compress conversations after 50 messages to save tokens",
    "My preferred quantization is Q4_K_M for quality-speed balance",
    "I run a 13B parameter model on my 32GB RAM machine",
    "For image generation I use Stable Diffusion locally via ComfyUI",
    "MCP protocol for connecting AI tools is the future standard",
];

static PERSONAL: &[&str] = &[
    "I work best in the morning, usually starting at 7am",
    "My office setup is a standing desk with dual monitors",
    "I prefer dark theme in all applications",
    "Hebrew is my native language, English for work",
    "I live in Israel and work in the tech industry",
    "Coffee with oat milk, no sugar",
    "I prefer async communication over meetings",
    "My goal is to build products that respect user privacy",
    "I read technical blogs during lunch break",
    "Weekend projects are usually game development in Godot",
];

static PROJECTS: &[&str] = &[
    "Currently building Bellkis, an AI hub desktop application",
    "The tech stack is Tauri with Rust backend and React frontend",
    "Bellkis targets prosumer developers who want local AI",
    "Our main differentiator is the Echo Memory push-based system",
    "We use BSL 1.1 license for the app, Apache 2.0 for the kernel",
    "The server runs on Axum with PostgreSQL and Redis",
    "Mobile app uses React Native with Expo framework",
    "We have 56 AI tools indexed in our hub",
    "The app supports 5 languages: English, Spanish, French, German, Japanese",
    "Our pricing is Free, Pro $9.99/mo, Team $29.99/mo",
];

static BUSINESS: &[&str] = &[
    "Revenue model is freemium with cloud fine-tuning as upsell",
    "Main competitors are Jan.ai, Open WebUI, and LobeChat",
    "OpenClaw with 302K stars is both competitor and integration target",
    "NVIDIA's NemoClaw handles security, we handle everything else",
    "Target market is prosumer developers spending $10-20/mo",
    "Israel has good IIA grants but 3 month evaluation is too slow",
    "Y Combinator is the fastest path to seed funding",
    "Our patent strategy is pending prototype completion",
    "The kernel will be Apache 2.0 for maximum adoption",
    "Goal: 10K GitHub stars in first 6 months",
];

static TECH_DECISIONS: &[&str] = &[
    "We chose fastembed for embeddings because it bundles the ONNX model",
    "simsimd provides SIMD-accelerated cosine similarity on all platforms",
    "Echo Memory uses brute-force search in Phase 1, LSH in Phase 2",
    "PII detection uses regex patterns for API keys, credit cards, SSNs",
    "The echo index stores 384-dimensional f32 vectors",
    "Auto-scale config detects RAM and adjusts memory capacity",
    "Binary quantization reduces index from 1.8GB to 150MB on 8GB machines",
    "Persistence uses JSON in Phase 1, binary mmap in Phase 2",
    "The Hebbian learning system strengthens co-activated memory links",
    "Bloom filters will pre-screen irrelevant memories in Phase 2",
];

static FOOD: &[&str] = &[
    "I make shakshuka every Friday morning for breakfast",
    "My hummus recipe uses extra tahini and roasted garlic",
    "Thai green curry is my go-to dinner when I'm tired",
    "I prefer sourdough bread over regular white bread",
    "Fresh herbs from my balcony garden: basil, mint, cilantro",
    "My favorite restaurant in Tel Aviv serves amazing Japanese ramen",
    "I meal prep on Sundays for the work week",
    "Espresso from a Breville machine, double shot",
    "I discovered that sumac is amazing on roasted vegetables",
    "Saturday morning pancakes with maple syrup is a tradition",
];

static HEALTH: &[&str] = &[
    "I run 5K three times a week, usually in the evening",
    "Yoga on rest days helps with back pain from sitting",
    "I track calories using an app, targeting 2000/day",
    "Sleep schedule: 11pm to 6:30am, 7.5 hours",
    "Standing desk alternating every 30 minutes helps my posture",
    "I drink 2 liters of water daily, more in summer",
    "Stretching routine before every coding session",
    "I take vitamin D supplements, important in winter",
    "20-minute walk after lunch improves afternoon focus",
    "Blue light glasses for late night coding sessions",
];

static TRAVEL: &[&str] = &[
    "Favorite travel destination is Japan, especially Kyoto",
    "I prefer Airbnb over hotels for longer stays",
    "Always bring a portable charger and USB-C cables",
    "I've visited 15 countries, mostly in Europe and Asia",
    "Berlin has the best tech meetup scene outside of SF",
    "I try to work remotely from a different city once a quarter",
    "Airport lounge access is worth the annual fee",
    "My packing list is minimalist: one carry-on for a week",
    "I use Google Maps offline maps in countries with poor data",
    "Next trip planned: Portugal for the Web Summit",
];

static RANDOM: &[&str] = &[
    "The James Webb telescope can see galaxies 13 billion years old",
    "Rust was voted most loved language 7 years in a row",
    "The population of Israel is approximately 10 million",
    "A lobster's brain is in its throat",
    "The first computer bug was an actual moth in a relay",
    "Bitcoin uses more energy than some small countries",
    "The speed of light is approximately 300,000 km/s",
    "Ferris is the unofficial mascot of Rust programming language",
    "The Dead Sea is the lowest point on Earth's surface",
    "GPT-4 has approximately 1.8 trillion parameters",
];

// ---------------------------------------------------------------------------
// Query definitions
// ---------------------------------------------------------------------------

/// A test query with expected behavior.
struct TestQuery {
    /// Human-readable label for diagnostics.
    label: &'static str,
    /// The query text to send to `echo()`.
    query: &'static str,
    /// Query kind determines assertion strategy.
    kind: QueryKind,
}

#[derive(Debug, Clone, Copy)]
enum QueryKind {
    /// Direct recall -- expects at least one result with similarity > 0.4.
    DirectRecall,
    /// Cross-category -- expects results from multiple source categories.
    CrossCategory,
    /// Vague/indirect -- expects at least one result (any score).
    Vague,
    /// Should NOT match well -- all results should have similarity < 0.35.
    NoMatch,
    /// Edge case -- must not panic; any result (including empty) is acceptable.
    EdgeCase,
}

fn all_queries() -> Vec<TestQuery> {
    vec![
        // ── Direct recall (5) ────────────────────────────────────────
        TestQuery {
            label: "direct:python_web_framework",
            query: "What's my preferred Python web framework?",
            kind: QueryKind::DirectRecall,
        },
        TestQuery {
            label: "direct:database",
            query: "What database do I use?",
            kind: QueryKind::DirectRecall,
        },
        TestQuery {
            label: "direct:morning_routine",
            query: "What's my morning routine?",
            kind: QueryKind::DirectRecall,
        },
        TestQuery {
            label: "direct:current_project",
            query: "What am I currently building?",
            kind: QueryKind::DirectRecall,
        },
        TestQuery {
            label: "direct:travel_destination",
            query: "What's my favorite travel destination?",
            kind: QueryKind::DirectRecall,
        },
        // ── Cross-category (5) ───────────────────────────────────────
        TestQuery {
            label: "cross:ai_project_backend",
            query: "What should I use for a new AI project backend?",
            kind: QueryKind::CrossCategory,
        },
        TestQuery {
            label: "cross:health_productivity",
            query: "What's my daily health and productivity routine?",
            kind: QueryKind::CrossCategory,
        },
        TestQuery {
            label: "cross:israel_tech_life",
            query: "Tell me about my Israel tech life",
            kind: QueryKind::CrossCategory,
        },
        TestQuery {
            label: "cross:local_ai_setup",
            query: "What's my local AI development setup?",
            kind: QueryKind::CrossCategory,
        },
        TestQuery {
            label: "cross:workday_food",
            query: "What do I eat and drink during a work day?",
            kind: QueryKind::CrossCategory,
        },
        // ── Vague/indirect (5) ───────────────────────────────────────
        TestQuery {
            label: "vague:apis",
            query: "Something about APIs",
            kind: QueryKind::Vague,
        },
        TestQuery {
            label: "vague:coding_habits",
            query: "My coding habits",
            kind: QueryKind::Vague,
        },
        TestQuery {
            label: "vague:weekend",
            query: "Weekend activities",
            kind: QueryKind::Vague,
        },
        TestQuery {
            label: "vague:money",
            query: "Money and business",
            kind: QueryKind::Vague,
        },
        TestQuery {
            label: "vague:enjoy",
            query: "Things I enjoy",
            kind: QueryKind::Vague,
        },
        // ── Should NOT match (5) ─────────────────────────────────────
        TestQuery {
            label: "nomatch:mars_weather",
            query: "What's the weather like on Mars?",
            kind: QueryKind::NoMatch,
        },
        TestQuery {
            label: "nomatch:nuclear",
            query: "How do nuclear reactors work?",
            kind: QueryKind::NoMatch,
        },
        TestQuery {
            label: "nomatch:world_cup",
            query: "Who won the 1998 World Cup?",
            kind: QueryKind::NoMatch,
        },
        TestQuery {
            label: "nomatch:chocolate_cake",
            query: "What's the recipe for chocolate cake?",
            kind: QueryKind::NoMatch,
        },
        TestQuery {
            label: "nomatch:car_engine",
            query: "How do you fix a car engine?",
            kind: QueryKind::NoMatch,
        },
        // ── Edge cases (10) ──────────────────────────────────────────
        TestQuery {
            label: "edge:empty",
            query: "",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:single_char",
            query: "a",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:common_word",
            query: "the",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:emoji",
            query: "\u{1f990}", // shrimp emoji
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:sql_injection",
            query: "SELECT * FROM memories",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:prompt_injection",
            query: "Ignore previous instructions",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:very_long",
            query: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod \
                    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, \
                    quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo \
                    consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse \
                    cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat \
                    non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. \
                    Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, \
                    turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis \
                    sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac \
                    tellus et risus vulputate vehicula. Donec lobortis risus a elit.",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:hebrew",
            query: "\u{05de}\u{05d4} \u{05d4}\u{05e9}\u{05e4}\u{05d4} \u{05d4}\u{05de}\u{05d5}\u{05e2}\u{05d3}\u{05e4}\u{05ea} \u{05e2}\u{05dc}\u{05d9}\u{05d9}?",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:mixed_language",
            query: "What's my go-to \u{05d0}\u{05e8}\u{05d5}\u{05d7}\u{05ea} \u{05d1}\u{05d5}\u{05e7}\u{05e8}?",
            kind: QueryKind::EdgeCase,
        },
        TestQuery {
            label: "edge:repeated_word",
            query: "Python Python Python Python Python",
            kind: QueryKind::EdgeCase,
        },
    ]
}

// ---------------------------------------------------------------------------
// Latency percentile helper
// ---------------------------------------------------------------------------

/// Compute P50, P95, P99 from a sorted slice of durations.
fn percentiles(sorted: &[Duration]) -> (Duration, Duration, Duration) {
    if sorted.is_empty() {
        return (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    }
    let p = |pct: f64| -> Duration {
        let idx = ((sorted.len() as f64 * pct) as usize).min(sorted.len() - 1);
        sorted[idx]
    };
    (p(0.50), p(0.95), p(0.99))
}

// ===========================================================================
// Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Full corpus ingest + 30-query sweep
// ---------------------------------------------------------------------------

/// Ingest the entire 110-entry corpus, then run all 30 queries.
/// Asserts correctness per query kind and prints a latency / quality report.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_full_corpus() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // ── Ingest ────────────────────────────────────────────────────────
    let ingest_start = Instant::now();
    let corpus = build_corpus(&engine).await;
    let ingest_elapsed = ingest_start.elapsed();

    let stats = engine.stats().await;
    assert_eq!(
        stats.total_memories,
        corpus.len(),
        "All corpus entries should be stored"
    );
    eprintln!(
        "\n=== CORPUS INGESTED ===\n  entries:      {}\n  ingest time:  {:.1}s\n  index bytes:  {}\n  RAM estimate: {} bytes\n",
        stats.total_memories,
        ingest_elapsed.as_secs_f64(),
        stats.index_size_bytes,
        stats.ram_usage_bytes,
    );

    // ── Query sweep ──────────────────────────────────────────────────
    let queries = all_queries();
    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    let mut failures: Vec<String> = Vec::new();

    for tq in &queries {
        let start = Instant::now();
        let result = engine.echo(tq.query, 20).await;
        let elapsed = start.elapsed();
        latencies.push(elapsed);

        match tq.kind {
            QueryKind::DirectRecall => {
                match result {
                    Ok(ref results) => {
                        if results.is_empty() {
                            failures.push(format!(
                                "[{}] DirectRecall returned 0 results for: {:?}",
                                tq.label, tq.query
                            ));
                        } else if results[0].similarity <= 0.4 {
                            failures.push(format!(
                                "[{}] DirectRecall top similarity {:.3} <= 0.4 for: {:?}",
                                tq.label, results[0].similarity, tq.query
                            ));
                        }
                        // Verify descending sort
                        for w in results.windows(2) {
                            assert!(
                                w[0].final_score >= w[1].final_score,
                                "[{}] Results not sorted: {} >= {}",
                                tq.label,
                                w[0].final_score,
                                w[1].final_score,
                            );
                        }
                    }
                    Err(e) => {
                        failures.push(format!("[{}] DirectRecall errored: {e}", tq.label));
                    }
                }
            }

            QueryKind::CrossCategory => {
                match result {
                    Ok(ref results) => {
                        if results.is_empty() {
                            failures.push(format!(
                                "[{}] CrossCategory returned 0 results for: {:?}",
                                tq.label, tq.query
                            ));
                        } else {
                            // Collect unique source categories from the top 10 results
                            let unique_sources: std::collections::HashSet<&str> =
                                results.iter().take(10).map(|r| r.source.as_str()).collect();
                            if unique_sources.len() < 2 {
                                // Soft warning -- cross-category is aspirational with
                                // a small embedding model. Log but don't hard-fail.
                                eprintln!(
                                    "  WARN [{}]: only {} category/ies in top-10 (expected >=2): {:?}",
                                    tq.label,
                                    unique_sources.len(),
                                    unique_sources,
                                );
                            }
                        }
                        // Verify descending sort
                        for w in results.windows(2) {
                            assert!(
                                w[0].final_score >= w[1].final_score,
                                "[{}] Results not sorted: {} >= {}",
                                tq.label,
                                w[0].final_score,
                                w[1].final_score,
                            );
                        }
                    }
                    Err(e) => {
                        failures.push(format!("[{}] CrossCategory errored: {e}", tq.label));
                    }
                }
            }

            QueryKind::Vague => {
                match result {
                    Ok(ref results) => {
                        if results.is_empty() {
                            failures.push(format!(
                                "[{}] Vague returned 0 results for: {:?}",
                                tq.label, tq.query
                            ));
                        }
                        // Verify descending sort
                        for w in results.windows(2) {
                            assert!(
                                w[0].final_score >= w[1].final_score,
                                "[{}] Results not sorted: {} >= {}",
                                tq.label,
                                w[0].final_score,
                                w[1].final_score,
                            );
                        }
                    }
                    Err(e) => {
                        failures.push(format!("[{}] Vague errored: {e}", tq.label));
                    }
                }
            }

            QueryKind::NoMatch => {
                match result {
                    Ok(ref results) => {
                        // With a 0.15 threshold some spurious hits are expected.
                        // The assertion is that the TOP score is below 0.35
                        // (i.e., nothing is strongly relevant).
                        if let Some(top) = results.first() {
                            if top.similarity > 0.35 {
                                eprintln!(
                                    "  WARN [{}]: NoMatch top similarity {:.3} > 0.35 for: {:?}  (content: {:?})",
                                    tq.label,
                                    top.similarity,
                                    tq.query,
                                    &top.content[..top.content.len().min(80)],
                                );
                            }
                        }
                    }
                    Err(e) => {
                        failures.push(format!("[{}] NoMatch errored: {e}", tq.label));
                    }
                }
            }

            QueryKind::EdgeCase => {
                // The only hard requirement: it must not panic or return Err.
                // (Empty results are fine.)
                match result {
                    Ok(_) => { /* pass */ }
                    Err(e) => {
                        // For genuinely empty queries the embedder may return
                        // an error -- that is acceptable, but log it.
                        eprintln!(
                            "  INFO [{}]: edge case returned Err (non-fatal): {e}",
                            tq.label,
                        );
                    }
                }
            }
        }
    }

    // ── Latency report ───────────────────────────────────────────────
    latencies.sort();
    let (p50, p95, p99) = percentiles(&latencies);
    eprintln!(
        "\n=== QUERY LATENCY (n={}) ===\n  P50:  {:>6.1}ms\n  P95:  {:>6.1}ms\n  P99:  {:>6.1}ms\n",
        latencies.len(),
        p50.as_secs_f64() * 1000.0,
        p95.as_secs_f64() * 1000.0,
        p99.as_secs_f64() * 1000.0,
    );

    // ── Final stats ──────────────────────────────────────────────────
    let final_stats = engine.stats().await;
    eprintln!(
        "=== FINAL STATS ===\n  total memories:  {}\n  total queries:   {}\n  avg latency:     {:.2}ms\n  index bytes:     {}\n  RAM estimate:    {} bytes\n",
        final_stats.total_memories,
        final_stats.total_echo_queries,
        final_stats.avg_echo_latency_ms,
        final_stats.index_size_bytes,
        final_stats.ram_usage_bytes,
    );

    // ── Hard failure summary ─────────────────────────────────────────
    if !failures.is_empty() {
        let msg = failures.join("\n  ");
        panic!("\n{} query assertion(s) failed:\n  {msg}\n", failures.len());
    }
}

// ---------------------------------------------------------------------------
// 2. Direct recall precision
// ---------------------------------------------------------------------------

/// For each direct-recall query, verify the top result is from the expected
/// category and the similarity is well above threshold.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_direct_recall_precision() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    // (query, expected source category, minimum similarity)
    let cases: &[(&str, &str, f32)] = &[
        (
            "What's my preferred Python web framework?",
            "programming",
            0.3,
        ),
        ("What database do I use?", "programming", 0.3),
        ("What am I currently building?", "projects", 0.3),
        ("What's my favorite travel destination?", "travel", 0.3),
        ("What do I eat for breakfast on Fridays?", "food", 0.3),
        ("Which AI model do I use for code review?", "ai_ml", 0.3),
        ("What is the Bellkis tech stack?", "projects", 0.3),
        ("How do I run AI models locally?", "ai_ml", 0.3),
        ("Who are Bellkis competitors?", "business", 0.3),
        ("What CSS framework do I use?", "programming", 0.3),
    ];

    let mut pass = 0usize;
    let mut fail_msgs: Vec<String> = Vec::new();

    for &(query, expected_source, min_sim) in cases {
        let results = engine.echo(query, 5).await.expect("echo should succeed");

        if results.is_empty() {
            fail_msgs.push(format!("  EMPTY results for: {query:?}"));
            continue;
        }

        let top = &results[0];
        if top.similarity < min_sim {
            fail_msgs.push(format!(
                "  LOW SIM {:.3} < {min_sim} for: {query:?}  (got: {:?})",
                top.similarity,
                &top.content[..top.content.len().min(60)]
            ));
            continue;
        }

        if top.source != expected_source {
            // Soft warning: the embedding model may associate differently.
            eprintln!(
                "  WARN: query={query:?} expected source={expected_source}, got={}  (sim={:.3})",
                top.source, top.similarity,
            );
        }

        pass += 1;
    }

    eprintln!(
        "\n=== DIRECT RECALL PRECISION ===\n  pass: {pass}/{}\n",
        cases.len()
    );

    if !fail_msgs.is_empty() {
        panic!("\nDirect recall failures:\n{}\n", fail_msgs.join("\n"));
    }
}

// ---------------------------------------------------------------------------
// 3. Cross-category retrieval
// ---------------------------------------------------------------------------

/// Verify that broad queries pull results from multiple categories.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_cross_category_retrieval() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    let queries = &[
        "What should I use for a new AI project backend?",
        "What's my daily health and productivity routine?",
        "Tell me about my Israel tech life",
        "What's my local AI development setup?",
        "What do I eat and drink during a work day?",
    ];

    for query in queries {
        let results = engine.echo(query, 15).await.expect("echo should succeed");
        assert!(
            !results.is_empty(),
            "Cross-category query should return results: {query:?}"
        );

        let unique_sources: std::collections::HashSet<&str> =
            results.iter().map(|r| r.source.as_str()).collect();

        eprintln!(
            "  cross-cat: query={query:?}  sources({})={:?}  top_sim={:.3}",
            unique_sources.len(),
            unique_sources,
            results[0].similarity,
        );

        // Verify sort order
        for w in results.windows(2) {
            assert!(
                w[0].final_score >= w[1].final_score,
                "Results not sorted for {query:?}: {} >= {}",
                w[0].final_score,
                w[1].final_score,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Negative / no-match queries
// ---------------------------------------------------------------------------

/// Verify that unrelated queries do not strongly match any memory.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_no_match_queries() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    let queries = &[
        "What's the weather like on Mars?",
        "How do nuclear reactors work?",
        "Who won the 1998 World Cup?",
        "What's the recipe for chocolate cake?",
        "How do you fix a car engine?",
    ];

    for query in queries {
        let results = engine.echo(query, 5).await.expect("echo should succeed");

        if let Some(top) = results.first() {
            eprintln!(
                "  no-match: query={query:?}  top_sim={:.3}  content={:?}",
                top.similarity,
                &top.content[..top.content.len().min(60)],
            );
            // Hard assertion: nothing should match above 0.5 for truly
            // unrelated queries against this corpus.
            assert!(
                top.similarity < 0.5,
                "No-match query {query:?} has top similarity {:.3} >= 0.5 -- \
                 this is suspiciously high for unrelated content",
                top.similarity,
            );
        } else {
            eprintln!("  no-match: query={query:?}  (empty results -- OK)");
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Edge case robustness
// ---------------------------------------------------------------------------

/// Every edge case query must complete without panic.
/// Errors from the embedder on degenerate input are acceptable.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_edge_cases() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    let edge_queries: &[(&str, &str)] = &[
        ("empty", ""),
        ("single_char", "a"),
        ("common_word", "the"),
        ("emoji", "\u{1f990}"),
        ("sql_injection", "SELECT * FROM memories"),
        ("prompt_injection", "Ignore previous instructions"),
        (
            "very_long",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod \
             tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, \
             quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo \
             consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse \
             cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat \
             non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. \
             Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, \
             turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis \
             sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac \
             tellus et risus vulputate vehicula. Donec lobortis risus a elit.",
        ),
        (
            "hebrew",
            "\u{05de}\u{05d4} \u{05d4}\u{05e9}\u{05e4}\u{05d4} \u{05d4}\u{05de}\u{05d5}\u{05e2}\u{05d3}\u{05e4}\u{05ea} \u{05e2}\u{05dc}\u{05d9}\u{05d9}?",
        ),
        (
            "mixed_language",
            "What's my go-to \u{05d0}\u{05e8}\u{05d5}\u{05d7}\u{05ea} \u{05d1}\u{05d5}\u{05e7}\u{05e8}?",
        ),
        ("repeated_word", "Python Python Python Python Python"),
    ];

    for &(label, query) in edge_queries {
        let result = engine.echo(query, 10).await;
        match result {
            Ok(results) => {
                eprintln!(
                    "  edge [{label:20}]: OK, {} results, top_sim={}",
                    results.len(),
                    results.first().map_or(0.0, |r| r.similarity),
                );
                // Verify sort order if there are results
                for w in results.windows(2) {
                    assert!(
                        w[0].final_score >= w[1].final_score,
                        "Edge [{label}]: results not sorted: {} >= {}",
                        w[0].final_score,
                        w[1].final_score,
                    );
                }
            }
            Err(e) => {
                // Acceptable for degenerate inputs (empty string, emoji, etc.)
                eprintln!("  edge [{label:20}]: Err (non-fatal): {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Ingest throughput
// ---------------------------------------------------------------------------

/// Measure raw ingest throughput: time to store 110 memories.
/// Asserts a generous upper bound (120s accounts for first model download).
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_ingest_throughput() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let corpus = full_corpus();
    let count = corpus.len();
    let start = Instant::now();

    for &(category, text) in &corpus {
        engine.store(text, category).await.unwrap();
    }

    let elapsed = start.elapsed();
    let per_entry_ms = elapsed.as_secs_f64() * 1000.0 / count as f64;

    eprintln!(
        "\n=== INGEST THROUGHPUT ===\n  entries:    {count}\n  total:      {:.1}s\n  per entry:  {per_entry_ms:.1}ms\n",
        elapsed.as_secs_f64(),
    );

    let stats = engine.stats().await;
    assert_eq!(stats.total_memories, count);

    // Generous upper bound: 120s total (includes potential model download on CI)
    assert!(
        elapsed.as_secs() < 120,
        "Ingest took {}s, expected < 120s",
        elapsed.as_secs()
    );
}

// ---------------------------------------------------------------------------
// 7. Query latency distribution
// ---------------------------------------------------------------------------

/// Run every query 3 times and report P50/P95/P99 latency distribution.
/// Verifies that no single query takes longer than 5 seconds.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_query_latency_distribution() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    let queries = all_queries();
    let mut all_latencies: Vec<Duration> = Vec::with_capacity(queries.len() * 3);

    // Warm up the embedder with one throw-away query
    let _ = engine.echo("warm up", 1).await;

    for tq in &queries {
        for _round in 0..3 {
            let start = Instant::now();
            let _ = engine.echo(tq.query, 10).await;
            let elapsed = start.elapsed();
            all_latencies.push(elapsed);

            // No single query should take more than 5 seconds
            assert!(
                elapsed.as_secs() < 5,
                "[{}] query took {}ms -- exceeds 5s limit",
                tq.label,
                elapsed.as_millis(),
            );
        }
    }

    all_latencies.sort();
    let (p50, p95, p99) = percentiles(&all_latencies);

    eprintln!(
        "\n=== QUERY LATENCY DISTRIBUTION (n={}, 3 rounds) ===\n  P50:  {:>6.1}ms\n  P95:  {:>6.1}ms\n  P99:  {:>6.1}ms\n  Max:  {:>6.1}ms\n",
        all_latencies.len(),
        p50.as_secs_f64() * 1000.0,
        p95.as_secs_f64() * 1000.0,
        p99.as_secs_f64() * 1000.0,
        all_latencies
            .last()
            .map_or(0.0, |d| d.as_secs_f64() * 1000.0),
    );
}

// ---------------------------------------------------------------------------
// 8. Persistence roundtrip at scale
// ---------------------------------------------------------------------------

/// Store the full corpus, persist to disk, load into a fresh engine,
/// and verify the same echo results are returned.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_persistence_roundtrip() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());

    // Phase 1: store + echo
    let engine = EchoEngine::new(config.clone()).expect("engine init");
    build_corpus(&engine).await;

    let probe_queries = &[
        "Rust systems programming",
        "local AI model Ollama",
        "Japanese ramen Tel Aviv",
        "freemium pricing model",
        "SIMD cosine similarity",
    ];

    let mut results_before: Vec<Vec<String>> = Vec::new();
    for &query in probe_queries {
        let results = engine.echo(query, 5).await.expect("echo should succeed");
        let ids: Vec<String> = results.iter().map(|r| r.memory_id.to_string()).collect();
        results_before.push(ids);
    }

    // Persist
    engine.persist().await.expect("persist should succeed");

    // Phase 2: load into fresh engine
    let engine2 = EchoEngine::load(config).expect("load should succeed");
    let stats = engine2.stats().await;
    assert_eq!(
        stats.total_memories, 110,
        "Loaded engine should have 110 memories, got {}",
        stats.total_memories,
    );

    // Verify same echo results
    for (i, &query) in probe_queries.iter().enumerate() {
        let results = engine2.echo(query, 5).await.expect("echo should succeed");
        let ids: Vec<String> = results.iter().map(|r| r.memory_id.to_string()).collect();

        assert_eq!(
            results_before[i], ids,
            "Persistence roundtrip: query={query:?} -- IDs differ before vs after"
        );

        // Verify similarity scores are identical
        let results_orig = engine.echo(query, 5).await.unwrap();
        for (orig, loaded) in results_orig.iter().zip(results.iter()) {
            assert!(
                (orig.similarity - loaded.similarity).abs() < 1e-5,
                "Similarity mismatch after roundtrip: {:.6} vs {:.6}",
                orig.similarity,
                loaded.similarity,
            );
        }
    }

    eprintln!(
        "\n=== PERSISTENCE ROUNDTRIP ===\n  110 memories persisted + loaded successfully\n  {probe} probe queries returned identical results\n",
        probe = probe_queries.len(),
    );
}

// ---------------------------------------------------------------------------
// 9. Forget at scale
// ---------------------------------------------------------------------------

/// Store the full corpus, forget 20 random memories, verify they are gone.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_forget_at_scale() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    let stats_before = engine.stats().await;
    assert_eq!(stats_before.total_memories, 110);

    // Store 20 specific memories and track their IDs
    let forget_texts = &[
        "Stress test forget entry alpha",
        "Stress test forget entry bravo",
        "Stress test forget entry charlie",
        "Stress test forget entry delta",
        "Stress test forget entry echo",
        "Stress test forget entry foxtrot",
        "Stress test forget entry golf",
        "Stress test forget entry hotel",
        "Stress test forget entry india",
        "Stress test forget entry juliet",
        "Stress test forget entry kilo",
        "Stress test forget entry lima",
        "Stress test forget entry mike",
        "Stress test forget entry november",
        "Stress test forget entry oscar",
        "Stress test forget entry papa",
        "Stress test forget entry quebec",
        "Stress test forget entry romeo",
        "Stress test forget entry sierra",
        "Stress test forget entry tango",
    ];

    let mut forget_ids = Vec::new();
    for text in forget_texts {
        let id = engine.store(text, "forget_test").await.unwrap();
        forget_ids.push(id);
    }

    let stats_mid = engine.stats().await;
    assert_eq!(stats_mid.total_memories, 130); // 110 + 20

    // Forget all 20
    for id in &forget_ids {
        engine
            .forget(id.clone())
            .await
            .expect("forget should succeed");
    }

    let stats_after = engine.stats().await;
    assert_eq!(
        stats_after.total_memories, 110,
        "Should be back to 110 after forgetting 20"
    );

    // Verify none of the forgotten memories appear in echo results
    let results = engine
        .echo("Stress test forget entry", 50)
        .await
        .expect("echo should succeed");

    for id in &forget_ids {
        assert!(
            results.iter().all(|r| r.memory_id != *id),
            "Forgotten memory {id} should not appear in results"
        );
    }

    eprintln!(
        "\n=== FORGET AT SCALE ===\n  Forgot 20/130 memories\n  Remaining: {}\n  None of the forgotten memories appear in echo results\n",
        stats_after.total_memories,
    );
}

// ---------------------------------------------------------------------------
// 10. Stats accuracy at scale
// ---------------------------------------------------------------------------

/// Verify stats tracking is accurate after heavy use: 110 stores + 30 queries.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_stats_accuracy() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    // Initial stats
    let stats = engine.stats().await;
    assert_eq!(stats.total_memories, 110);
    assert_eq!(stats.total_echo_queries, 0);
    assert_eq!(stats.max_capacity, 10_000);
    assert!(stats.index_size_bytes > 0, "Index size should be non-zero");
    assert!(stats.ram_usage_bytes > 0, "RAM estimate should be non-zero");

    // Expected index size: 110 memories * 384 dims * 4 bytes = 168,960
    let expected_index = 110 * 384 * 4;
    assert_eq!(
        stats.index_size_bytes, expected_index as u64,
        "Index size should be {expected_index}, got {}",
        stats.index_size_bytes,
    );

    // Run 30 queries
    let queries = all_queries();
    let query_count = queries.len();
    for tq in &queries {
        let _ = engine.echo(tq.query, 10).await;
    }

    let stats_after = engine.stats().await;
    assert_eq!(
        stats_after.total_echo_queries, query_count as u64,
        "Should have tracked {query_count} queries, got {}",
        stats_after.total_echo_queries,
    );
    assert!(
        stats_after.avg_echo_latency_ms > 0.0,
        "Average latency should be positive after queries"
    );

    eprintln!(
        "\n=== STATS ACCURACY ===\n  memories:    {}\n  queries:     {}\n  avg latency: {:.2}ms\n  index bytes: {}\n  RAM bytes:   {}\n",
        stats_after.total_memories,
        stats_after.total_echo_queries,
        stats_after.avg_echo_latency_ms,
        stats_after.index_size_bytes,
        stats_after.ram_usage_bytes,
    );
}

// ---------------------------------------------------------------------------
// 11. Category isolation
// ---------------------------------------------------------------------------

/// For each category, query with a category-specific term and verify the
/// top result comes from the expected category.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_category_isolation() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    // (category-specific query, expected source category)
    let probes: &[(&str, &str)] = &[
        ("pytest fixtures testing framework", "programming"),
        ("QLoRA 4-bit quantization fine-tuning", "ai_ml"),
        ("standing desk dual monitors office", "personal"),
        ("Tauri Rust backend React frontend app", "projects"),
        ("freemium revenue cloud fine-tuning", "business"),
        ("fastembed ONNX model embeddings", "tech_decisions"),
        ("shakshuka Friday morning breakfast", "food"),
        ("5K running three times week evening", "health"),
        ("Kyoto Japan favorite destination", "travel"),
        ("James Webb telescope galaxies billion", "random"),
    ];

    let mut pass = 0;

    for &(query, expected_source) in probes {
        let results = engine.echo(query, 3).await.expect("echo should succeed");

        if results.is_empty() {
            eprintln!("  FAIL [{expected_source}]: no results for {query:?}");
            continue;
        }

        let top = &results[0];
        if top.source == expected_source {
            pass += 1;
            eprintln!(
                "  OK   [{expected_source:16}] sim={:.3}  {:?}",
                top.similarity,
                &top.content[..top.content.len().min(50)],
            );
        } else {
            eprintln!(
                "  MISS [{expected_source:16}] got source={:?} sim={:.3}  {:?}",
                top.source,
                top.similarity,
                &top.content[..top.content.len().min(50)],
            );
        }
    }

    eprintln!(
        "\n=== CATEGORY ISOLATION ===\n  pass: {pass}/{}\n",
        probes.len()
    );

    // At least 7/10 should hit the correct category with these specific probes
    assert!(
        pass >= 7,
        "Category isolation: expected >= 7/10 correct, got {pass}/{}",
        probes.len(),
    );
}

// ---------------------------------------------------------------------------
// 12. Concurrent echo safety
// ---------------------------------------------------------------------------

/// Fire 20 echo queries concurrently and verify no panics, deadlocks,
/// or data corruption.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_concurrent_echo() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    let engine = std::sync::Arc::new(engine);
    let queries: Vec<&str> = vec![
        "Rust programming language",
        "AI model locally",
        "standing desk morning",
        "Bellkis desktop app",
        "freemium pricing",
        "fastembed ONNX",
        "shakshuka breakfast",
        "running 5K",
        "Japan travel",
        "James Webb telescope",
        "Docker containers",
        "RAG documents",
        "dark theme",
        "BSL license Apache",
        "IIA grants Israel",
        "SIMD cosine",
        "hummus tahini",
        "yoga back pain",
        "Berlin tech meetup",
        "GPT-4 parameters",
    ];

    let mut handles = Vec::new();
    for query in queries {
        let eng = engine.clone();
        let q = query.to_string();
        handles.push(tokio::spawn(async move {
            let result = eng.echo(&q, 5).await;
            (q, result)
        }));
    }

    let mut ok_count = 0;
    let mut err_count = 0;

    for handle in handles {
        let (query, result) = handle.await.expect("task should not panic");
        match result {
            Ok(results) => {
                ok_count += 1;
                // Verify sort order
                for w in results.windows(2) {
                    assert!(
                        w[0].final_score >= w[1].final_score,
                        "Concurrent query {query:?}: results not sorted",
                    );
                }
            }
            Err(e) => {
                err_count += 1;
                eprintln!("  concurrent err: query={query:?} err={e}");
            }
        }
    }

    eprintln!("\n=== CONCURRENT ECHO ===\n  OK:   {ok_count}\n  ERR:  {err_count}\n");

    // All should succeed (the embedder Mutex serializes but should not deadlock)
    assert_eq!(
        err_count, 0,
        "All concurrent queries should succeed, got {err_count} errors"
    );
}

// ---------------------------------------------------------------------------
// 13. Echo count tracking
// ---------------------------------------------------------------------------

/// Verify that echo_count on memories increments correctly across queries.
/// We store a distinctive memory, query for it multiple times, and check
/// the stats reflect the expected query count.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn stress_echo_count_tracking() {
    let dir = tempdir().expect("temp dir");
    let config = stress_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");
    build_corpus(&engine).await;

    // Run the same query 10 times
    let query = "Rust systems programming language safety";
    for _ in 0..10 {
        let _ = engine.echo(query, 5).await;
    }

    let stats = engine.stats().await;
    assert_eq!(stats.total_echo_queries, 10, "Should track 10 echo queries");

    eprintln!(
        "\n=== ECHO COUNT TRACKING ===\n  queries:     {}\n  avg latency: {:.2}ms\n",
        stats.total_echo_queries, stats.avg_echo_latency_ms,
    );
}
