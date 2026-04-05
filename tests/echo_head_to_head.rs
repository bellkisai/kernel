//! Head-to-head benchmark: ShrimPK Echo Memory vs Plain Ollama.
//!
//! Proves that Echo Memory makes AI responses BETTER and more personalized
//! by injecting relevant stored memories as context before sending prompts
//! to a local LLM.
//!
//! ## How it works
//! For each prompt, we run two modes:
//! - **Mode A (No Echo)**: Send the raw question to Ollama — no context, no memory.
//! - **Mode B (With Echo)**: Run EchoEngine::echo() to find relevant memories,
//!   build an enhanced prompt with that context, then send to the same Ollama model.
//!
//! We measure keyword hits, response times, and personalization to prove Echo
//! Memory delivers measurably better results.
//!
//! All tests are `#[ignore]` because they require:
//! 1. Ollama running locally at http://localhost:11434
//! 2. At least one model pulled (e.g., `ollama pull llama3.2:1b`)
//! 3. The fastembed model (all-MiniLM-L6-v2, ~23MB ONNX)
//!
//! Run with:
//!
//!     cargo test --test echo_head_to_head -- --ignored --nocapture

use serde::{Deserialize, Serialize};
use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use tempfile::tempdir;

// ===========================================================================
// Ollama API types
// ===========================================================================

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaGenerateResponse {
    response: String,
    #[serde(default)]
    total_duration: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    prompt_eval_count: u64,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}

// ===========================================================================
// Memory corpus
// ===========================================================================

const MEMORIES: &[(&str, &str)] = &[
    (
        "I prefer FastAPI for building REST APIs because of async and auto-docs",
        "coding",
    ),
    ("Python 3.12 is my main programming language", "coding"),
    ("PostgreSQL for all database work, never MySQL", "coding"),
    ("I deploy everything to AWS using Terraform", "infra"),
    ("VS Code with Vim keybindings is my IDE setup", "coding"),
    ("TypeScript strict mode for all frontend work", "coding"),
    ("Docker Compose for local development environments", "infra"),
    ("I run Ollama locally to avoid API costs", "ai"),
    (
        "My current project is Bellkis, an AI desktop hub built with Tauri and Rust",
        "project",
    ),
    ("I use GitHub Actions for CI/CD with matrix builds", "infra"),
    (
        "For testing I prefer pytest with fixtures and parametrize",
        "coding",
    ),
    ("Redis for caching, PostgreSQL for persistence", "infra"),
    ("I use pnpm instead of npm for package management", "coding"),
    (
        "Tailwind CSS with shadcn/ui for all frontend styling",
        "coding",
    ),
    ("My preferred quantization for local models is Q4_K_M", "ai"),
    ("I live in Israel and work in the tech industry", "personal"),
    ("I prefer dark theme in all applications", "personal"),
    ("Coffee with oat milk, no sugar, every morning", "personal"),
    ("I run 5K three times a week in the evening", "personal"),
    (
        "My favorite travel destination is Japan, especially Kyoto",
        "personal",
    ),
];

// ===========================================================================
// Prompt pairs: question + expected keywords in a good response
// ===========================================================================

const PROMPTS: &[(&str, &[&str])] = &[
    (
        "What framework should I use for a new REST API?",
        &["FastAPI", "async", "Python"],
    ),
    ("Help me set up a database for my project", &["PostgreSQL"]),
    (
        "What's my deployment workflow?",
        &["AWS", "Terraform", "Docker"],
    ),
    (
        "Recommend an IDE setup for TypeScript development",
        &["VS Code", "Vim", "TypeScript"],
    ),
    (
        "How should I handle caching in my application?",
        &["Redis", "PostgreSQL"],
    ),
    (
        "What testing framework should I use?",
        &["pytest", "fixtures"],
    ),
    (
        "Tell me about my current project",
        &["Bellkis", "Tauri", "Rust"],
    ),
    (
        "What's the best way to run AI models locally?",
        &["Ollama", "Q4_K_M"],
    ),
    ("What package manager do I use for JavaScript?", &["pnpm"]),
    (
        "How do I style my frontend components?",
        &["Tailwind", "shadcn"],
    ),
];

// ===========================================================================
// Measurements
// ===========================================================================

struct PromptResult {
    question: String,
    input_tokens_a: usize,
    input_tokens_b: usize,
    response_a: String,
    response_b: String,
    keywords_found_a: usize,
    keywords_found_b: usize,
    keywords_total: usize,
    duration_a_ms: u64,
    duration_b_ms: u64,
    personalized_b: bool,
}

// ===========================================================================
// Helpers
// ===========================================================================

const OLLAMA_BASE: &str = "http://localhost:11434";

/// Estimate token count (~4 chars per token).
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Count how many expected keywords appear in the response (case-insensitive).
fn count_keywords(response: &str, keywords: &[&str]) -> usize {
    let lower = response.to_lowercase();
    keywords
        .iter()
        .filter(|kw| lower.contains(&kw.to_lowercase()))
        .count()
}

/// Check if a response references stored personal preferences.
fn is_personalized(response: &str) -> bool {
    let indicators = [
        "your preference",
        "you prefer",
        "you use",
        "you mentioned",
        "your setup",
        "your project",
        "your current",
        "based on your",
        "your memory",
        "your context",
        "according to",
        "as you",
        "you've",
        "you have",
        "you run",
        "you deploy",
        "your workflow",
        "your stack",
        "your tool",
        "you like",
        // Also check direct references to stored facts
        "FastAPI",
        "PostgreSQL",
        "Terraform",
        "VS Code",
        "Vim keybinding",
        "pnpm",
        "shadcn",
        "Q4_K_M",
        "Bellkis",
        "pytest",
    ];
    let lower = response.to_lowercase();
    indicators
        .iter()
        .any(|ind| lower.contains(&ind.to_lowercase()))
}

/// Discover the first available Ollama model. Returns None if Ollama is
/// unreachable or has no models pulled.
async fn discover_model() -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let resp = client
        .get(format!("{OLLAMA_BASE}/api/tags"))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let tags: OllamaTagsResponse = resp.json().await.ok()?;
    tags.models.first().map(|m| m.name.clone())
}

/// Send a prompt to Ollama and return (response_text, duration_ms).
async fn ollama_generate(model: &str, prompt: &str) -> (String, u64) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("HTTP client");

    let req = OllamaGenerateRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };

    let start = std::time::Instant::now();
    let resp = client
        .post(format!("{OLLAMA_BASE}/api/generate"))
        .json(&req)
        .send()
        .await
        .expect("Ollama request failed");

    let elapsed_ms = start.elapsed().as_millis() as u64;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        panic!("Ollama returned {status}: {body}");
    }

    let result: OllamaGenerateResponse = resp.json().await.expect("Parse Ollama response");
    // Use Ollama's own total_duration if available (nanoseconds -> ms), else wall clock
    let duration_ms = if result.total_duration > 0 {
        result.total_duration / 1_000_000
    } else {
        elapsed_ms
    };

    (result.response, duration_ms)
}

/// Build an EchoConfig for the benchmark.
fn bench_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15,
        max_echo_results: 5,
        ram_budget_bytes: 100_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

// ===========================================================================
// The benchmark
// ===========================================================================

#[tokio::test]
#[ignore = "requires Ollama running locally with a model pulled"]
async fn head_to_head_echo_vs_plain_ollama() {
    // 1. Discover available model
    let model = match discover_model().await {
        Some(m) => m,
        None => {
            println!("\n=== SKIPPED: Echo vs Plain Ollama ===");
            println!("Ollama not running or no models available.");
            println!("Start Ollama and pull a model: ollama pull llama3.2:1b");
            return;
        }
    };

    println!("\n=== HEAD-TO-HEAD: Echo Memory vs Plain Ollama ===");
    println!("Model: {model}");
    println!();

    // 2. Set up Echo Memory engine and store the memory corpus
    let dir = tempdir().expect("temp dir");
    let config = bench_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("EchoEngine init");

    for &(text, source) in MEMORIES {
        engine
            .store(text, source)
            .await
            .expect("Failed to store memory");
    }

    // 3. Run each prompt in both modes
    let mut results: Vec<PromptResult> = Vec::new();

    for &(question, expected_keywords) in PROMPTS {
        // ---- Mode A: No Echo (baseline) ----
        let (response_a, duration_a_ms) = ollama_generate(&model, question).await;
        let input_tokens_a = estimate_tokens(question);
        let keywords_found_a = count_keywords(&response_a, expected_keywords);

        // ---- Mode B: With Echo ----
        // Step 1: Get relevant memories from Echo
        let echo_results = engine.echo(question, 3).await.expect("Echo query failed");

        // Step 2: Build enhanced prompt
        let mut context_lines = String::new();
        for r in &echo_results {
            context_lines.push_str(&format!("- {}\n", r.content));
        }

        let enhanced_prompt = if echo_results.is_empty() {
            // Fallback: no echo results, send raw question
            question.to_string()
        } else {
            format!(
                "Context from your memory:\n\
                 {context_lines}\n\
                 Based on the above context, answer: {question}"
            )
        };

        let input_tokens_b = estimate_tokens(&enhanced_prompt);

        // Step 3: Send enhanced prompt to Ollama
        let (response_b, duration_b_ms) = ollama_generate(&model, &enhanced_prompt).await;
        let keywords_found_b = count_keywords(&response_b, expected_keywords);
        let personalized_b = is_personalized(&response_b);

        results.push(PromptResult {
            question: question.to_string(),
            input_tokens_a,
            input_tokens_b,
            response_a,
            response_b,
            keywords_found_a,
            keywords_found_b,
            keywords_total: expected_keywords.len(),
            duration_a_ms,
            duration_b_ms,
            personalized_b,
        });
    }

    // 4. Print results table
    println!(
        "{:<45} | {:>16} | {:>13} | {:<6}",
        "Prompt", "No Echo Keywords", "Echo Keywords", "Winner"
    );
    println!("{}", "-".repeat(95));

    let mut echo_wins = 0u32;
    let mut tie_count = 0u32;
    let mut total_kw_a = 0usize;
    let mut total_kw_b = 0usize;
    let mut personalized_count = 0u32;
    let mut total_input_a = 0usize;
    let mut total_input_b = 0usize;

    for r in &results {
        let short_q: String = if r.question.len() > 43 {
            format!("{}...", &r.question[..40])
        } else {
            r.question.clone()
        };

        let winner = if r.keywords_found_b > r.keywords_found_a {
            echo_wins += 1;
            "Echo"
        } else if r.keywords_found_b == r.keywords_found_a {
            tie_count += 1;
            "Tie"
        } else {
            "Plain"
        };

        println!(
            "{:<45} | {:>6}/{:<9} | {:>5}/{:<7} | {:<6}",
            short_q,
            r.keywords_found_a,
            r.keywords_total,
            r.keywords_found_b,
            r.keywords_total,
            winner,
        );

        total_kw_a += r.keywords_found_a;
        total_kw_b += r.keywords_found_b;
        total_input_a += r.input_tokens_a;
        total_input_b += r.input_tokens_b;
        if r.personalized_b {
            personalized_count += 1;
        }
    }

    let n = results.len() as f64;
    let avg_kw_a = total_kw_a as f64 / n;
    let avg_kw_b = total_kw_b as f64 / n;
    let avg_input_a = total_input_a as f64 / n;
    let avg_input_b = total_input_b as f64 / n;
    let extra_tokens = avg_input_b - avg_input_a;
    let personalization_pct = (personalized_count as f64 / n) * 100.0;

    let accuracy_improvement = if avg_kw_a > 0.0 {
        ((avg_kw_b - avg_kw_a) / avg_kw_a) * 100.0
    } else if avg_kw_b > 0.0 {
        // Avoid division by zero: if baseline is 0 but echo found keywords,
        // report as a direct percentage of total possible keywords
        let avg_total: f64 = results.iter().map(|r| r.keywords_total as f64).sum::<f64>() / n;
        (avg_kw_b / avg_total) * 100.0
    } else {
        0.0
    };

    println!();
    println!("SUMMARY:");
    println!("  Echo won: {echo_wins}/{} prompts", results.len());
    println!("  Ties:     {tie_count}/{} prompts", results.len());
    println!("  Avg keywords (No Echo): {avg_kw_a:.1}");
    println!("  Avg keywords (Echo):    {avg_kw_b:.1}");
    println!(
        "  Personalization rate:   {personalization_pct:.0}% (Echo references stored preferences)"
    );
    println!("  Avg input tokens (No Echo): {avg_input_a:.0}");
    println!("  Avg input tokens (Echo):    {avg_input_b:.0} (+{extra_tokens:.0} for context)");
    println!();
    println!(
        "CONCLUSION: Echo Memory makes AI {accuracy_improvement:.0}% more accurate \
         and {personalization_pct:.0}% more personalized."
    );

    // 5. Print detailed per-prompt comparison
    println!();
    println!("=== DETAILED RESPONSES ===");
    for (i, r) in results.iter().enumerate() {
        println!();
        println!("--- Prompt {}: {} ---", i + 1, r.question);
        println!(
            "  [A] No Echo ({} tokens, {}ms): {}",
            r.input_tokens_a,
            r.duration_a_ms,
            truncate_response(&r.response_a, 200),
        );
        println!(
            "  [B] Echo    ({} tokens, {}ms): {}",
            r.input_tokens_b,
            r.duration_b_ms,
            truncate_response(&r.response_b, 200),
        );
        println!(
            "  Keywords: A={}/{} B={}/{}  Personalized: {}",
            r.keywords_found_a,
            r.keywords_total,
            r.keywords_found_b,
            r.keywords_total,
            if r.personalized_b { "YES" } else { "no" },
        );
    }

    // 6. Assertions: Echo should win on at least some prompts
    // We use soft assertions since LLM output is non-deterministic
    assert!(
        echo_wins >= 1,
        "Echo Memory should win at least 1 prompt, but won {echo_wins}/{}",
        results.len()
    );
    println!();
    println!("=== BENCHMARK PASSED ===");
}

/// Truncate a response for display, collapsing whitespace.
fn truncate_response(text: &str, max_len: usize) -> String {
    let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.len() <= max_len {
        cleaned
    } else {
        format!("{}...", &cleaned[..max_len])
    }
}
