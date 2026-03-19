//! Precision tuning test suite for Echo Memory.
//!
//! Systematically tests different similarity thresholds, embedding strategies,
//! and query formulations to find the optimal configuration for the Echo engine.
//!
//! Context: at 110 memories, some direct recall queries scored below the 0.4
//! threshold. For example:
//!   - "What's my preferred Python web framework?" -> 0.371
//!   - "What am I currently building?" -> 0.322
//!
//! The gap is between HOW the user asks vs HOW the memory was stored.
//! These tests find the sweet spot.
//!
//! All tests are `#[ignore]` because they require the fastembed model
//! (all-MiniLM-L6-v2, ~23MB ONNX). Run with:
//!
//!     cargo test --test echo_precision_tuning -- --ignored --nocapture
//!
//! Or run a single test:
//!
//!     cargo test --test echo_precision_tuning threshold_range_sweep -- --ignored --nocapture

use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use shrimpk_memory::embedder::Embedder;
use shrimpk_memory::similarity::cosine_similarity;
use std::path::PathBuf;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an EchoConfig for precision tuning tests.
/// Uses a very low threshold so we capture all candidates for analysis.
fn tuning_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.05, // near-zero: capture everything for analysis
        max_echo_results: 50,       // wide window to see full ranking
        ram_budget_bytes: 200_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

/// Build a config with a specific threshold for sweep tests.
fn config_with_threshold(data_dir: PathBuf, threshold: f32) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: threshold,
        max_echo_results: 50,
        ram_budget_bytes: 200_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Test data: POSITIVE pairs (memory, query) that SHOULD match
// ---------------------------------------------------------------------------

/// Each entry: (memory_text, query_text, pair_label)
fn positive_pairs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // FastAPI cluster -- different query styles for the same memory
        (
            "I prefer FastAPI for building REST APIs",
            "What web framework for APIs?",
            "fastapi:direct_question",
        ),
        (
            "I prefer FastAPI for building REST APIs",
            "best Python API framework",
            "fastapi:keyword_phrase",
        ),
        (
            "I prefer FastAPI for building REST APIs",
            "What's my preferred Python web framework?",
            "fastapi:natural_indirect",
        ),
        (
            "I prefer FastAPI for building REST APIs",
            "FastAPI",
            "fastapi:exact_keyword",
        ),
        // Python cluster
        (
            "Python is my main programming language",
            "What language do I use?",
            "python:direct_question",
        ),
        (
            "Python is my main programming language",
            "my coding language",
            "python:keyword_phrase",
        ),
        (
            "Python is my main programming language",
            "programming language preference",
            "python:formal_phrase",
        ),
        // Ollama cluster
        (
            "I run Ollama locally for most AI tasks",
            "How do I run AI?",
            "ollama:direct_question",
        ),
        (
            "I run Ollama locally for most AI tasks",
            "local AI setup",
            "ollama:keyword_phrase",
        ),
        (
            "I run Ollama locally for most AI tasks",
            "What AI tool do I use?",
            "ollama:natural_question",
        ),
        // Bellkis cluster
        (
            "Currently building Bellkis, an AI hub desktop application",
            "What am I building?",
            "bellkis:direct_question",
        ),
        (
            "Currently building Bellkis, an AI hub desktop application",
            "What is my current project?",
            "bellkis:rephrased_question",
        ),
        (
            "Currently building Bellkis, an AI hub desktop application",
            "Tell me about Bellkis",
            "bellkis:topic_request",
        ),
        // Shakshuka cluster
        (
            "I make shakshuka every Friday morning",
            "What do I cook on Fridays?",
            "shakshuka:temporal_question",
        ),
        (
            "I make shakshuka every Friday morning",
            "breakfast recipe",
            "shakshuka:category_keyword",
        ),
        (
            "I make shakshuka every Friday morning",
            "Friday morning food",
            "shakshuka:time_context",
        ),
        // Running cluster
        (
            "I run 5K three times a week",
            "exercise routine",
            "running:category_keyword",
        ),
        (
            "I run 5K three times a week",
            "how often do I run?",
            "running:frequency_question",
        ),
        // Travel cluster
        (
            "My favorite travel destination is Japan",
            "where do I like to travel?",
            "travel:direct_question",
        ),
        (
            "My favorite travel destination is Japan",
            "Japan trip",
            "travel:keyword_phrase",
        ),
    ]
}

// ---------------------------------------------------------------------------
// Test data: NEGATIVE pairs (memory, query) that should NOT match
// ---------------------------------------------------------------------------

fn negative_pairs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "I prefer FastAPI for building REST APIs",
            "What's the weather like today?",
            "fastapi:weather",
        ),
        (
            "I prefer FastAPI for building REST APIs",
            "recipe for chocolate cake",
            "fastapi:cake",
        ),
        (
            "Python is my main programming language",
            "how to fix a car engine",
            "python:car",
        ),
        (
            "Python is my main programming language",
            "best hiking trails in Colorado",
            "python:hiking",
        ),
        (
            "I run Ollama locally for most AI tasks",
            "who won the 1998 World Cup",
            "ollama:worldcup",
        ),
        (
            "I run Ollama locally for most AI tasks",
            "how to knit a sweater",
            "ollama:knitting",
        ),
        (
            "Currently building Bellkis, an AI hub desktop application",
            "recipe for hummus",
            "bellkis:hummus",
        ),
        (
            "Currently building Bellkis, an AI hub desktop application",
            "what is quantum entanglement",
            "bellkis:quantum",
        ),
        (
            "I make shakshuka every Friday morning",
            "how do nuclear reactors work",
            "shakshuka:nuclear",
        ),
        (
            "I make shakshuka every Friday morning",
            "stock market predictions",
            "shakshuka:stocks",
        ),
        (
            "I run 5K three times a week",
            "how to write a compiler",
            "running:compiler",
        ),
        (
            "I run 5K three times a week",
            "the speed of light is 300000 km/s",
            "running:physics",
        ),
        (
            "My favorite travel destination is Japan",
            "how to tune a guitar",
            "travel:guitar",
        ),
        (
            "My favorite travel destination is Japan",
            "what is the periodic table",
            "travel:chemistry",
        ),
        // Cross-domain negatives: tech query against personal memory
        (
            "I make shakshuka every Friday morning",
            "best Python web framework",
            "shakshuka:python_framework",
        ),
        (
            "My favorite travel destination is Japan",
            "Kubernetes orchestration",
            "travel:kubernetes",
        ),
        (
            "I run 5K three times a week",
            "machine learning embeddings",
            "running:ml",
        ),
        (
            "Python is my main programming language",
            "yoga for back pain",
            "python:yoga",
        ),
        (
            "I prefer FastAPI for building REST APIs",
            "sourdough bread recipe",
            "fastapi:bread",
        ),
        (
            "Currently building Bellkis, an AI hub desktop application",
            "history of the Roman Empire",
            "bellkis:rome",
        ),
    ]
}

// ---------------------------------------------------------------------------
// Test data: Memory formulation variants for Test 3
// ---------------------------------------------------------------------------

struct FormulationVariant {
    label: &'static str,
    text: &'static str,
}

fn fastapi_formulations() -> Vec<FormulationVariant> {
    vec![
        FormulationVariant {
            label: "A:natural",
            text: "I prefer FastAPI for building REST APIs",
        },
        FormulationVariant {
            label: "B:rewritten",
            text: "FastAPI is my preferred web framework for REST API development",
        },
        FormulationVariant {
            label: "C:structured",
            text: "Web framework preference: FastAPI. Used for REST APIs.",
        },
        FormulationVariant {
            label: "D:third_person_detailed",
            text: "User prefers FastAPI for REST APIs because of async support and auto docs",
        },
    ]
}

fn bellkis_formulations() -> Vec<FormulationVariant> {
    vec![
        FormulationVariant {
            label: "A:natural",
            text: "Currently building Bellkis, an AI hub desktop application",
        },
        FormulationVariant {
            label: "B:rewritten",
            text: "Bellkis is an AI hub desktop app currently under development",
        },
        FormulationVariant {
            label: "C:structured",
            text: "Current project: Bellkis. Type: AI hub desktop application.",
        },
        FormulationVariant {
            label: "D:third_person_detailed",
            text: "User is building Bellkis, a desktop application that serves as an AI hub",
        },
    ]
}

fn cooking_formulations() -> Vec<FormulationVariant> {
    vec![
        FormulationVariant {
            label: "A:natural",
            text: "I make shakshuka every Friday morning",
        },
        FormulationVariant {
            label: "B:rewritten",
            text: "Shakshuka is my regular Friday morning breakfast dish",
        },
        FormulationVariant {
            label: "C:structured",
            text: "Friday morning routine: cook shakshuka for breakfast.",
        },
        FormulationVariant {
            label: "D:third_person_detailed",
            text: "User makes shakshuka every Friday morning as a breakfast tradition",
        },
    ]
}

// ---------------------------------------------------------------------------
// Full 110-memory stress corpus (from echo_stress_test.rs)
// ---------------------------------------------------------------------------

fn full_corpus() -> Vec<(&'static str, &'static str)> {
    let mut corpus: Vec<(&str, &str)> = Vec::with_capacity(120);

    for text in PROGRAMMING {
        corpus.push(("programming", text));
    }
    for text in AI_ML {
        corpus.push(("ai_ml", text));
    }
    for text in PERSONAL {
        corpus.push(("personal", text));
    }
    for text in PROJECTS {
        corpus.push(("projects", text));
    }
    for text in BUSINESS {
        corpus.push(("business", text));
    }
    for text in TECH_DECISIONS {
        corpus.push(("tech_decisions", text));
    }
    for text in FOOD {
        corpus.push(("food", text));
    }
    for text in HEALTH {
        corpus.push(("health", text));
    }
    for text in TRAVEL {
        corpus.push(("travel", text));
    }
    for text in RANDOM {
        corpus.push(("random", text));
    }

    corpus
}

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
// Test 1: Threshold Range Sweep
// ---------------------------------------------------------------------------
//
// Stores 20 known positive pairs and 20 known negative pairs, then sweeps
// thresholds from 0.10 to 0.90 in 0.05 increments. For each threshold,
// measures precision, recall, and F1 score.

#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn threshold_range_sweep() {
    let dir = tempdir().expect("temp dir");
    let config = tuning_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let positives = positive_pairs();
    let negatives = negative_pairs();

    // Collect unique memory texts
    let mut unique_memories: Vec<&str> = Vec::new();
    for &(mem, _, _) in &positives {
        if !unique_memories.contains(&mem) {
            unique_memories.push(mem);
        }
    }
    for &(mem, _, _) in &negatives {
        if !unique_memories.contains(&mem) {
            unique_memories.push(mem);
        }
    }

    // Store all unique memories
    let mut memory_map: Vec<(&str, String)> = Vec::new(); // (text, memory_id as string)
    for &text in &unique_memories {
        let id = engine
            .store(text, "tuning_test")
            .await
            .unwrap_or_else(|e| panic!("Failed to store: {text} -- {e}"));
        memory_map.push((text, id.to_string()));
    }

    // For each positive pair, query with threshold=0.05 and record the
    // similarity score between the query and its expected memory.
    println!();
    println!("====================================================================");
    println!("  TEST 1: THRESHOLD RANGE SWEEP");
    println!("====================================================================");
    println!();

    // Collect similarity scores for all pairs using raw embeddings
    let mut embedder = Embedder::new().expect("embedder init");

    // Build memory embeddings map
    let mut mem_embeddings: Vec<(&str, Vec<f32>)> = Vec::new();
    for &text in &unique_memories {
        let emb = embedder.embed(text).expect("embed memory");
        mem_embeddings.push((text, emb));
    }

    // Score all positive pairs
    let mut positive_scores: Vec<(&str, &str, &str, f32)> = Vec::new();
    for &(mem_text, query_text, label) in &positives {
        let query_emb = embedder.embed(query_text).expect("embed query");
        let mem_emb = mem_embeddings
            .iter()
            .find(|(t, _)| *t == mem_text)
            .map(|(_, e)| e)
            .expect("memory embedding should exist");
        let score = cosine_similarity(&query_emb, mem_emb);
        positive_scores.push((mem_text, query_text, label, score));
    }

    // Score all negative pairs
    let mut negative_scores: Vec<(&str, &str, &str, f32)> = Vec::new();
    for &(mem_text, query_text, label) in &negatives {
        let query_emb = embedder.embed(query_text).expect("embed query");
        let mem_emb = mem_embeddings
            .iter()
            .find(|(t, _)| *t == mem_text)
            .map(|(_, e)| e)
            .expect("memory embedding should exist");
        let score = cosine_similarity(&query_emb, mem_emb);
        negative_scores.push((mem_text, query_text, label, score));
    }

    // Sweep thresholds
    println!(
        "{:>9} | {:>9} | {:>6} | {:>6} | {:>5} | {:>6} | {:>6}",
        "Threshold", "Precision", "Recall", "  F1  ", "True+", "False+", "False-"
    );
    println!(
        "{:-<9}-+-{:-<9}-+-{:-<6}-+-{:-<6}-+-{:-<5}-+-{:-<6}-+-{:-<6}",
        "", "", "", "", "", "", ""
    );

    let mut best_f1: f32 = 0.0;
    let mut best_threshold: f32 = 0.0;

    let mut threshold = 0.10_f32;
    while threshold <= 0.90 {
        let true_positives = positive_scores
            .iter()
            .filter(|(_, _, _, s)| *s >= threshold)
            .count();
        let false_negatives = positive_scores
            .iter()
            .filter(|(_, _, _, s)| *s < threshold)
            .count();
        let false_positives = negative_scores
            .iter()
            .filter(|(_, _, _, s)| *s >= threshold)
            .count();

        let precision = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            1.0 // no predictions => vacuously precise
        };
        let recall = if true_positives + false_negatives > 0 {
            true_positives as f32 / (true_positives + false_negatives) as f32
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = threshold;
        }

        println!(
            "{:>9.2} | {:>9.4} | {:>6.4} | {:>6.4} | {:>5} | {:>6} | {:>6}",
            threshold, precision, recall, f1, true_positives, false_positives, false_negatives
        );

        threshold += 0.05;
    }

    println!();
    println!(
        ">>> Best F1: {:.4} at threshold {:.2}",
        best_f1, best_threshold
    );
    println!();

    // Also print the score distributions
    let mut pos_sorted: Vec<f32> = positive_scores.iter().map(|(_, _, _, s)| *s).collect();
    pos_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut neg_sorted: Vec<f32> = negative_scores.iter().map(|(_, _, _, s)| *s).collect();
    neg_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    println!("  Positive score distribution (ascending):");
    println!(
        "    min={:.4}  p25={:.4}  median={:.4}  p75={:.4}  max={:.4}",
        pos_sorted[0],
        pos_sorted[pos_sorted.len() / 4],
        pos_sorted[pos_sorted.len() / 2],
        pos_sorted[3 * pos_sorted.len() / 4],
        pos_sorted[pos_sorted.len() - 1],
    );
    println!("  Negative score distribution (descending):");
    println!(
        "    max={:.4}  p75={:.4}  median={:.4}  p25={:.4}  min={:.4}",
        neg_sorted[0],
        neg_sorted[neg_sorted.len() / 4],
        neg_sorted[neg_sorted.len() / 2],
        neg_sorted[3 * neg_sorted.len() / 4],
        neg_sorted[neg_sorted.len() - 1],
    );
    println!();

    // Sanity: best F1 should be meaningfully above random chance
    assert!(
        best_f1 > 0.50,
        "Best F1 ({best_f1:.4}) should be > 0.50; the model cannot distinguish positive from negative"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Query Formulation Analysis
// ---------------------------------------------------------------------------
//
// For each positive pair, shows the actual similarity score.
// Reveals which query styles work best: direct question, keyword, natural.

#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn query_formulation_analysis() {
    let mut embedder = Embedder::new().expect("embedder init");

    let positives = positive_pairs();

    println!();
    println!("====================================================================");
    println!("  TEST 2: QUERY FORMULATION ANALYSIS");
    println!("====================================================================");
    println!();

    // Group by memory text
    let mut groups: Vec<(&str, Vec<(&str, &str, f32)>)> = Vec::new();
    let mut current_memory: Option<&str> = None;

    for &(mem_text, query_text, label) in &positives {
        let query_emb = embedder.embed(query_text).expect("embed query");
        let mem_emb = embedder.embed(mem_text).expect("embed memory");
        let score = cosine_similarity(&query_emb, &mem_emb);

        if current_memory != Some(mem_text) {
            groups.push((mem_text, Vec::new()));
            current_memory = Some(mem_text);
        }
        groups
            .last_mut()
            .unwrap()
            .1
            .push((query_text, label, score));
    }

    // Statistics for query style analysis
    let mut style_scores: Vec<(&str, Vec<f32>)> = Vec::new();

    for (mem_text, queries) in &groups {
        println!("  Memory: {:?}", mem_text);
        let mut best_score: f32 = 0.0;
        let mut best_label = "";
        for &(query_text, label, score) in queries {
            let marker = if score < 0.30 {
                " ** BELOW 0.30 **"
            } else if score < 0.40 {
                " * below 0.40"
            } else {
                ""
            };
            println!("    {:>50} -> {:.4}{}", query_text, score, marker);

            // Track by query style (extract style from label suffix)
            let style = label.split(':').last().unwrap_or("unknown");
            if let Some(entry) = style_scores.iter_mut().find(|(s, _)| *s == style) {
                entry.1.push(score);
            } else {
                style_scores.push((style, vec![score]));
            }

            if score > best_score {
                best_score = score;
                best_label = label;
            }
        }
        let best_style = best_label.split(':').last().unwrap_or("unknown");
        println!(
            "    >>> Best: {} ({:.4}) -- style: {}",
            best_label, best_score, best_style
        );
        println!();
    }

    // Summary: average score per query style
    println!("  ---- Query Style Summary ----");
    println!(
        "  {:>25} | {:>6} | {:>6} | {:>6} | {:>5}",
        "Style", "Avg", "Min", "Max", "Count"
    );
    println!(
        "  {:-<25}-+-{:-<6}-+-{:-<6}-+-{:-<6}-+-{:-<5}",
        "", "", "", "", ""
    );

    let mut style_summary: Vec<(&str, f32, f32, f32, usize)> = style_scores
        .iter()
        .map(|(style, scores)| {
            let avg = scores.iter().sum::<f32>() / scores.len() as f32;
            let min = scores
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let max = scores
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            (*style, avg, min, max, scores.len())
        })
        .collect();

    style_summary.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (style, avg, min, max, count) in &style_summary {
        println!(
            "  {:>25} | {:>6.4} | {:>6.4} | {:>6.4} | {:>5}",
            style, avg, min, max, count
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Test 3: Memory Formulation Analysis
// ---------------------------------------------------------------------------
//
// Tests whether HOW we store memories affects recall. Same semantic content,
// different phrasings. Reveals if structured storage outperforms natural language.

#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn memory_formulation_analysis() {
    let mut embedder = Embedder::new().expect("embedder init");

    println!();
    println!("====================================================================");
    println!("  TEST 3: MEMORY FORMULATION ANALYSIS");
    println!("====================================================================");
    println!();

    // Test FastAPI formulations
    let fastapi_queries = vec![
        "What web framework for APIs?",
        "best Python API framework",
        "What's my preferred Python web framework?",
        "FastAPI",
        "REST API development tool",
    ];

    println!("  --- FastAPI memory formulations ---");
    run_formulation_test(&mut embedder, &fastapi_formulations(), &fastapi_queries);

    // Test Bellkis formulations
    let bellkis_queries = vec![
        "What am I building?",
        "What is my current project?",
        "Tell me about Bellkis",
        "AI desktop app project",
        "current work in progress",
    ];

    println!("  --- Bellkis memory formulations ---");
    run_formulation_test(&mut embedder, &bellkis_formulations(), &bellkis_queries);

    // Test Cooking formulations
    let cooking_queries = vec![
        "What do I cook on Fridays?",
        "breakfast recipe",
        "Friday morning food",
        "shakshuka",
        "weekly cooking routine",
    ];

    println!("  --- Cooking memory formulations ---");
    run_formulation_test(&mut embedder, &cooking_formulations(), &cooking_queries);
}

fn run_formulation_test(
    embedder: &mut Embedder,
    formulations: &[FormulationVariant],
    queries: &[&str],
) {
    // Build a matrix: rows = formulations, columns = queries
    let mut score_matrix: Vec<Vec<f32>> = Vec::new();

    for variant in formulations {
        let mem_emb = embedder.embed(variant.text).expect("embed memory variant");
        let mut row: Vec<f32> = Vec::new();
        for &query in queries {
            let query_emb = embedder.embed(query).expect("embed query");
            let score = cosine_similarity(&query_emb, &mem_emb);
            row.push(score);
        }
        score_matrix.push(row);
    }

    // Print header
    print!("  {:>25} |", "Formulation");
    for &q in queries {
        let truncated: String = q.chars().take(22).collect();
        print!(" {:>22} |", truncated);
    }
    println!(" {:>6}", "AVG");

    let separator_width = 25 + 3 + queries.len() * 25 + 8;
    println!("  {}", "-".repeat(separator_width));

    // Print each formulation's scores
    let mut best_avg: f32 = 0.0;
    let mut best_label = "";

    for (i, variant) in formulations.iter().enumerate() {
        let avg = score_matrix[i].iter().sum::<f32>() / score_matrix[i].len() as f32;
        print!("  {:>25} |", variant.label);
        for &score in &score_matrix[i] {
            let marker = if score < 0.30 { "*" } else { " " };
            print!(" {:>21.4}{} |", score, marker);
        }
        println!(" {:>6.4}", avg);

        if avg > best_avg {
            best_avg = avg;
            best_label = variant.label;
        }
    }

    println!();
    println!(
        "    >>> Best formulation: {} (avg {:.4})",
        best_label, best_avg
    );
    println!();

    // Show per-query which formulation wins
    println!("    Per-query winners:");
    for (j, &query) in queries.iter().enumerate() {
        let mut best_score: f32 = 0.0;
        let mut best_form = "";
        for (i, variant) in formulations.iter().enumerate() {
            if score_matrix[i][j] > best_score {
                best_score = score_matrix[i][j];
                best_form = variant.label;
            }
        }
        let truncated: String = query.chars().take(40).collect();
        println!(
            "      {:>40} -> {} ({:.4})",
            truncated, best_form, best_score
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Test 4: Context Window Simulation
// ---------------------------------------------------------------------------
//
// Loads all 110 memories from the stress corpus, then for each of the known
// positive pairs checks what rank the expected memory appears at, what
// irrelevant memories appear above it, and at what threshold it is included
// vs excluded.

#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn context_window_simulation() {
    let dir = tempdir().expect("temp dir");
    let config = tuning_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // Build the full 110-memory corpus
    let corpus = full_corpus();
    for &(category, text) in &corpus {
        engine
            .store(text, category)
            .await
            .unwrap_or_else(|e| panic!("Failed to store [{category}]: {text} -- {e}"));
    }

    println!();
    println!("====================================================================");
    println!("  TEST 4: CONTEXT WINDOW SIMULATION (110 memories)");
    println!("====================================================================");
    println!();

    // Representative queries -- one per memory cluster
    let test_queries: Vec<(&str, &str)> = vec![
        (
            "What web framework for APIs?",
            "I prefer FastAPI for building REST APIs because of async support and auto-generated docs",
        ),
        (
            "What's my preferred Python web framework?",
            "I prefer FastAPI for building REST APIs because of async support and auto-generated docs",
        ),
        (
            "What language do I use?",
            "Python is my main programming language",
        ), // Note: not in the corpus as-is but "Python" appears in PROGRAMMING[0]
        (
            "How do I run AI?",
            "I run Ollama locally for most AI tasks to avoid API costs",
        ),
        (
            "local AI setup",
            "I run Ollama locally for most AI tasks to avoid API costs",
        ),
        (
            "What am I building?",
            "Currently building Bellkis, an AI hub desktop application",
        ),
        (
            "What is my current project?",
            "Currently building Bellkis, an AI hub desktop application",
        ),
        (
            "What do I cook on Fridays?",
            "I make shakshuka every Friday morning for breakfast",
        ),
        (
            "exercise routine",
            "I run 5K three times a week, usually in the evening",
        ),
        (
            "how often do I run?",
            "I run 5K three times a week, usually in the evening",
        ),
        (
            "where do I like to travel?",
            "Favorite travel destination is Japan, especially Kyoto",
        ),
        (
            "Japan trip",
            "Favorite travel destination is Japan, especially Kyoto",
        ),
        (
            "What database do I use?",
            "For database work I always choose PostgreSQL over MySQL",
        ),
        (
            "What's my morning routine?",
            "I work best in the morning, usually starting at 7am",
        ),
        (
            "breakfast recipe",
            "I make shakshuka every Friday morning for breakfast",
        ),
    ];

    let mut rank_found: Vec<usize> = Vec::new();
    let mut not_found_count = 0;
    let mut safe_thresholds: Vec<f32> = Vec::new();

    for (query_text, expected_substring) in &test_queries {
        let results = engine
            .echo(query_text, 50)
            .await
            .expect("echo should succeed");

        println!("  Query: {:?}", query_text);

        // Find the rank of the expected memory
        let mut found_rank: Option<usize> = None;
        let mut found_score: f32 = 0.0;

        for (rank, result) in results.iter().enumerate() {
            let is_target = result.content.contains(expected_substring)
                || expected_substring.contains(&result.content);
            if is_target && found_rank.is_none() {
                found_rank = Some(rank + 1);
                found_score = result.similarity;
            }
        }

        // Print top 5 results
        let display_count = results.len().min(5);
        for (rank, result) in results.iter().take(display_count).enumerate() {
            let is_target = result.content.contains(expected_substring)
                || expected_substring.contains(&result.content);
            let marker = if is_target { " <-- TARGET" } else { "" };
            let truncated: String = result.content.chars().take(60).collect();
            println!(
                "    Rank {:>2}: [{:.4}] {:?}{}",
                rank + 1,
                result.similarity,
                truncated,
                marker
            );
        }

        if let Some(rank) = found_rank {
            rank_found.push(rank);

            // Calculate safe threshold: the score just above rank 2 (if rank 1 is target)
            // or just above the first irrelevant result
            let safe_threshold = if rank == 1 && results.len() > 1 {
                // Target is rank 1. Safe threshold is midpoint between target and next.
                (found_score + results[1].similarity) / 2.0
            } else {
                // Target is not rank 1. Need threshold low enough to include it.
                found_score - 0.01
            };
            safe_thresholds.push(safe_threshold);

            println!(
                "    >>> Found at rank {} (score {:.4}), safe threshold: {:.4}",
                rank, found_score, safe_threshold
            );
        } else {
            not_found_count += 1;
            println!(
                "    >>> NOT FOUND in top 50 results! Expected: {:?}",
                &expected_substring[..expected_substring.len().min(60)]
            );
        }
        println!();
    }

    // Summary
    println!("  ---- Context Window Summary ----");
    if !rank_found.is_empty() {
        let avg_rank = rank_found.iter().sum::<usize>() as f32 / rank_found.len() as f32;
        let rank_1_count = rank_found.iter().filter(|&&r| r == 1).count();
        let top_3_count = rank_found.iter().filter(|&&r| r <= 3).count();
        let top_5_count = rank_found.iter().filter(|&&r| r <= 5).count();

        println!("    Total queries: {}", test_queries.len());
        println!(
            "    Found: {} | Not found: {}",
            rank_found.len(),
            not_found_count
        );
        println!("    Average rank when found: {:.1}", avg_rank);
        println!(
            "    Rank 1: {} ({:.0}%)",
            rank_1_count,
            rank_1_count as f32 / rank_found.len() as f32 * 100.0
        );
        println!(
            "    Top 3:  {} ({:.0}%)",
            top_3_count,
            top_3_count as f32 / rank_found.len() as f32 * 100.0
        );
        println!(
            "    Top 5:  {} ({:.0}%)",
            top_5_count,
            top_5_count as f32 / rank_found.len() as f32 * 100.0
        );
    }

    if !safe_thresholds.is_empty() {
        let mut sorted_thresholds = safe_thresholds.clone();
        sorted_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!();
        println!(
            "    Safe threshold range: {:.4} - {:.4}",
            sorted_thresholds[0],
            sorted_thresholds[sorted_thresholds.len() - 1]
        );
        println!(
            "    Median safe threshold: {:.4}",
            sorted_thresholds[sorted_thresholds.len() / 2]
        );
        println!(
            "    Conservative (p25): {:.4}",
            sorted_thresholds[sorted_thresholds.len() / 4]
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Test 5: Recommended Configuration
// ---------------------------------------------------------------------------
//
// Runs all analysis from Tests 1-4 and synthesizes the final recommendation.

#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn recommended_configuration() {
    let mut embedder = Embedder::new().expect("embedder init");

    let positives = positive_pairs();
    let negatives = negative_pairs();

    println!();
    println!("====================================================================");
    println!("  TEST 5: RECOMMENDED CONFIGURATION");
    println!("====================================================================");
    println!();

    // --- Phase A: Compute all scores ---

    // Unique memories
    let mut unique_memories: Vec<&str> = Vec::new();
    for &(mem, _, _) in &positives {
        if !unique_memories.contains(&mem) {
            unique_memories.push(mem);
        }
    }

    // Embed all unique memories
    let mut mem_embeddings: Vec<(&str, Vec<f32>)> = Vec::new();
    for &text in &unique_memories {
        let emb = embedder.embed(text).expect("embed memory");
        mem_embeddings.push((text, emb));
    }

    // Score positive pairs
    let mut positive_scores: Vec<f32> = Vec::new();
    for &(mem_text, query_text, _) in &positives {
        let query_emb = embedder.embed(query_text).expect("embed query");
        let mem_emb = mem_embeddings
            .iter()
            .find(|(t, _)| *t == mem_text)
            .map(|(_, e)| e)
            .expect("memory embedding");
        let score = cosine_similarity(&query_emb, mem_emb);
        positive_scores.push(score);
    }

    // Score negative pairs
    let mut negative_scores: Vec<f32> = Vec::new();
    for &(mem_text, query_text, _) in &negatives {
        let query_emb = embedder.embed(query_text).expect("embed query");
        let mem_emb = mem_embeddings
            .iter()
            .find(|(t, _)| *t == mem_text)
            .map(|(_, e)| e)
            .expect("memory embedding");
        let score = cosine_similarity(&query_emb, mem_emb);
        negative_scores.push(score);
    }

    // --- Phase B: Find optimal threshold via F1 maximization ---

    let mut best_f1: f32 = 0.0;
    let mut best_threshold: f32 = 0.0;
    let mut best_precision: f32 = 0.0;
    let mut best_recall: f32 = 0.0;

    // Fine-grained sweep: 0.05 to 0.85 in 0.01 increments
    let mut threshold = 0.05_f32;
    while threshold <= 0.85 {
        let tp = positive_scores.iter().filter(|&&s| s >= threshold).count();
        let fn_ = positive_scores.iter().filter(|&&s| s < threshold).count();
        let fp = negative_scores.iter().filter(|&&s| s >= threshold).count();

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            1.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = threshold;
            best_precision = precision;
            best_recall = recall;
        }

        threshold += 0.01;
    }

    // --- Phase C: Analyze score gap ---

    positive_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    negative_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending

    let pos_min = positive_scores[0];
    let pos_p10 = positive_scores[positive_scores.len() / 10];
    let pos_median = positive_scores[positive_scores.len() / 2];
    let neg_max = negative_scores[0];
    let neg_p10 = negative_scores[negative_scores.len() / 10];
    let neg_median = negative_scores[negative_scores.len() / 2];

    let separation_gap = pos_min - neg_max;

    // --- Phase D: Determine max_echo_results recommendation ---
    // Use the context window simulation logic: in 110 memories, how many
    // results typically score above the threshold?
    let dir = tempdir().expect("temp dir");
    let config = config_with_threshold(dir.path().to_path_buf(), best_threshold);
    let engine = EchoEngine::new(config).expect("engine init");

    let corpus = full_corpus();
    for &(category, text) in &corpus {
        engine
            .store(text, category)
            .await
            .unwrap_or_else(|e| panic!("Failed to store [{category}]: {text} -- {e}"));
    }

    // Run a few representative queries and count results
    let sample_queries = vec![
        "What web framework for APIs?",
        "What am I building?",
        "exercise routine",
        "where do I like to travel?",
        "local AI setup",
    ];

    let mut result_counts: Vec<usize> = Vec::new();
    for query in &sample_queries {
        let results = engine.echo(query, 50).await.expect("echo");
        result_counts.push(results.len());
    }

    let avg_results = result_counts.iter().sum::<usize>() as f32 / result_counts.len() as f32;
    let max_results = *result_counts.iter().max().unwrap_or(&0);

    // Recommend max_echo_results that covers p95 of result counts + headroom
    let recommended_max_results = if max_results <= 5 {
        5
    } else if max_results <= 10 {
        10
    } else {
        (max_results + 5).min(20)
    };

    // --- Phase E: Score gap analysis ---
    let needs_gap_filter = separation_gap < 0.05;

    // --- Phase F: Keyword boost analysis ---
    // Test: do queries with exact keywords in the memory score higher?
    let keyword_hit_scores: Vec<f32> = positives
        .iter()
        .zip(positive_scores.iter())
        .filter(|((mem, query, _), _)| {
            // Check if any significant word from the query appears verbatim in the memory
            let query_words: Vec<&str> = query.split_whitespace().filter(|w| w.len() > 3).collect();
            query_words
                .iter()
                .any(|w| mem.to_lowercase().contains(&w.to_lowercase()))
        })
        .map(|(_, &score)| score)
        .collect();

    let keyword_miss_scores: Vec<f32> = positives
        .iter()
        .zip(positive_scores.iter())
        .filter(|((mem, query, _), _)| {
            let query_words: Vec<&str> = query.split_whitespace().filter(|w| w.len() > 3).collect();
            !query_words
                .iter()
                .any(|w| mem.to_lowercase().contains(&w.to_lowercase()))
        })
        .map(|(_, &score)| score)
        .collect();

    let avg_keyword_hit = if keyword_hit_scores.is_empty() {
        0.0
    } else {
        keyword_hit_scores.iter().sum::<f32>() / keyword_hit_scores.len() as f32
    };
    let avg_keyword_miss = if keyword_miss_scores.is_empty() {
        0.0
    } else {
        keyword_miss_scores.iter().sum::<f32>() / keyword_miss_scores.len() as f32
    };
    let keyword_boost_delta = avg_keyword_hit - avg_keyword_miss;
    let needs_keyword_boost = keyword_boost_delta > 0.05 && avg_keyword_miss < best_threshold;

    // --- Output the recommendation ---

    println!("  === SCORE DISTRIBUTIONS ===");
    println!();
    println!("  Positive pairs (should match):");
    println!(
        "    min={:.4}  p10={:.4}  median={:.4}  count={}",
        pos_min,
        pos_p10,
        pos_median,
        positive_scores.len()
    );
    println!("  Negative pairs (should NOT match):");
    println!(
        "    max={:.4}  p10={:.4}  median={:.4}  count={}",
        neg_max,
        neg_p10,
        neg_median,
        negative_scores.len()
    );
    println!();
    println!(
        "  Separation gap (pos_min - neg_max): {:.4}",
        separation_gap
    );
    if separation_gap > 0.0 {
        println!(
            "    Clean separation -- a threshold exists that perfectly separates the classes."
        );
    } else {
        println!(
            "    OVERLAP of {:.4} -- some positive/negative scores overlap.",
            -separation_gap
        );
        println!(
            "    Pairs that fall in the overlap zone will be misclassified at any single threshold."
        );
    }
    println!();

    println!("  === OPTIMAL THRESHOLD (F1 maximized) ===");
    println!();
    println!("  similarity_threshold: {:.2}", best_threshold);
    println!(
        "    Precision: {:.4} ({:.0}% of returned results are relevant)",
        best_precision,
        best_precision * 100.0
    );
    println!(
        "    Recall:    {:.4} ({:.0}% of relevant memories are returned)",
        best_recall,
        best_recall * 100.0
    );
    println!("    F1:        {:.4}", best_f1);
    println!();

    println!("  === MAX ECHO RESULTS ===");
    println!();
    println!(
        "  At threshold {:.2} with 110 memories, query result counts:",
        best_threshold
    );
    for (i, query) in sample_queries.iter().enumerate() {
        println!("    {:?} -> {} results", query, result_counts[i]);
    }
    println!("  Average: {:.1} | Max: {}", avg_results, max_results);
    println!("  max_echo_results: {}", recommended_max_results);
    println!();

    println!("  === SCORE GAP FILTER ===");
    println!();
    if needs_gap_filter {
        println!("  RECOMMENDED: Yes, enable minimum score gap filter.");
        println!("    The positive/negative distributions overlap, so a gap filter");
        println!("    (e.g., top result must be >0.10 above the second result)");
        println!("    will help discard ambiguous matches.");
    } else {
        println!("  RECOMMENDED: No, gap filter not needed.");
        println!("    The distributions are well-separated at the optimal threshold.");
    }
    println!();

    println!("  === KEYWORD BOOST ===");
    println!();
    println!(
        "  Avg score when query has keyword overlap with memory: {:.4} (n={})",
        avg_keyword_hit,
        keyword_hit_scores.len()
    );
    println!(
        "  Avg score when NO keyword overlap:                    {:.4} (n={})",
        avg_keyword_miss,
        keyword_miss_scores.len()
    );
    println!("  Delta: {:.4}", keyword_boost_delta);
    if needs_keyword_boost {
        println!("  RECOMMENDED: Yes, boost exact keyword matches.");
        println!("    Queries without keyword overlap score below threshold on average.");
        println!("    A +0.05 to +0.10 keyword boost would rescue these matches.");
    } else {
        println!("  RECOMMENDED: No keyword boost needed.");
        println!("    Semantic similarity alone is sufficient at the chosen threshold.");
    }
    println!();

    println!("  ============================================");
    println!("  FINAL RECOMMENDED CONFIGURATION:");
    println!("  ============================================");
    println!();
    println!("  EchoConfig {{");
    println!("      similarity_threshold: {:.2},", best_threshold);
    println!("      max_echo_results: {},", recommended_max_results);
    println!("      // gap_filter: {},", needs_gap_filter);
    println!("      // keyword_boost: {},", needs_keyword_boost);
    println!("  }}");
    println!();

    // Assertions: ensure the test produces meaningful results
    assert!(
        best_f1 > 0.50,
        "Optimal F1 ({best_f1:.4}) should be > 0.50 -- model is not discriminating"
    );
    assert!(
        best_threshold > 0.10 && best_threshold < 0.80,
        "Optimal threshold ({best_threshold:.2}) should be between 0.10 and 0.80"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Hardest pairs deep dive
// ---------------------------------------------------------------------------
//
// Zooms into the pairs that originally triggered this investigation:
//   - "What's my preferred Python web framework?" -> 0.371
//   - "What am I currently building?" -> 0.322
//
// Provides exhaustive diagnostics: token overlap, embedding distance,
// comparison against reformulated queries, and a concrete fix recommendation.

#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn hardest_pairs_deep_dive() {
    let mut embedder = Embedder::new().expect("embedder init");

    println!();
    println!("====================================================================");
    println!("  TEST 6: HARDEST PAIRS DEEP DIVE");
    println!("====================================================================");
    println!();

    let hard_pairs: Vec<(&str, &str, f32)> = vec![
        (
            "I prefer FastAPI for building REST APIs because of async support and auto-generated docs",
            "What's my preferred Python web framework?",
            0.371,
        ),
        (
            "Currently building Bellkis, an AI hub desktop application",
            "What am I currently building?",
            0.322,
        ),
    ];

    for (memory, query, historical_score) in &hard_pairs {
        let mem_emb = embedder.embed(memory).expect("embed memory");
        let query_emb = embedder.embed(query).expect("embed query");
        let actual_score = cosine_similarity(&query_emb, &mem_emb);

        println!("  Memory: {:?}", memory);
        println!("  Query:  {:?}", query);
        println!(
            "  Historical score: {:.3} | Current score: {:.4}",
            historical_score, actual_score
        );
        println!();

        // Generate alternative queries and score each
        let reformulations: Vec<(&str, &str)> = if memory.contains("FastAPI") {
            vec![
                ("exact_entity", "FastAPI"),
                ("entity_context", "FastAPI REST APIs"),
                ("direct_domain", "web framework for APIs"),
                (
                    "preference_question",
                    "what framework do I prefer for APIs?",
                ),
                ("python_specific", "Python API framework preference"),
                ("keyword_rich", "prefer FastAPI building REST APIs"),
                ("how_question", "how do I build REST APIs?"),
                ("why_question", "why do I use FastAPI?"),
                ("completion_style", "My preferred API framework is"),
                ("third_person", "user's Python web framework preference"),
            ]
        } else {
            vec![
                ("exact_entity", "Bellkis"),
                ("entity_context", "Bellkis AI hub desktop"),
                ("direct_question", "what project am I working on?"),
                ("building_question", "what am I developing right now?"),
                (
                    "keyword_rich",
                    "currently building AI hub desktop application",
                ),
                ("short_informal", "current project"),
                ("how_question", "what application am I building?"),
                ("status_question", "project status"),
                ("completion_style", "I am currently building"),
                ("third_person", "user's current development project"),
            ]
        };

        println!("  Reformulation analysis:");
        let mut scored_reformulations: Vec<(&str, &str, f32)> = Vec::new();
        for (style, alt_query) in &reformulations {
            let alt_emb = embedder.embed(alt_query).expect("embed reformulation");
            let score = cosine_similarity(&alt_emb, &mem_emb);
            scored_reformulations.push((style, alt_query, score));
        }

        scored_reformulations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        for (style, alt_query, score) in &scored_reformulations {
            let delta = score - actual_score;
            let marker = if *score > actual_score + 0.05 {
                " ++ BETTER"
            } else if *score < actual_score - 0.05 {
                " -- WORSE"
            } else {
                ""
            };
            println!(
                "    [{:>20}] {:>55} -> {:.4} (delta {:+.4}){}",
                style, alt_query, score, delta, marker
            );
        }

        // Identify what makes high-scoring reformulations succeed
        let top_3: Vec<&&str> = scored_reformulations
            .iter()
            .take(3)
            .map(|(_, q, _)| q)
            .collect();
        let bottom_3: Vec<&&str> = scored_reformulations
            .iter()
            .rev()
            .take(3)
            .map(|(_, q, _)| q)
            .collect();

        println!();
        println!("    Top 3 query styles:    {:?}", top_3);
        println!("    Bottom 3 query styles: {:?}", bottom_3);
        println!(
            "    Score spread: {:.4} (best) - {:.4} (worst) = {:.4} range",
            scored_reformulations[0].2,
            scored_reformulations.last().unwrap().2,
            scored_reformulations[0].2 - scored_reformulations.last().unwrap().2,
        );
        println!();
        println!("  ---");
        println!();
    }
}
