//! Micro-benchmark for fast KS67 dev testing (~2 min with Ollama consolidation).
//!
//! 20 curated memories covering ALL hard cases:
//! - Identity gravity well (name appears everywhere)
//! - Supersession (job change, city move, tool switch)
//! - Multi-entity facts (project names, tools, people)
//! - Dense multi-fact content (long memories with 5+ facts)
//! - Temporal reasoning (dates, timelines, "when did")
//! - Preference evolution (IDE, OS, framework changes)
//!
//! Run WITHOUT consolidation (embedding-only, instant):
//!     cargo test --test echo_micro_benchmark -- --ignored --nocapture
//!
//! Run WITH consolidation (requires Ollama, ~2 min):
//!     cargo test --test echo_micro_benchmark -- --ignored --nocapture consolidation
//!
//! Expects fastembed model (all-MiniLM-L6-v2, ~23MB ONNX).

use shrimpk_core::{EchoConfig, EchoResult};
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use tempfile::tempdir;

// ===========================================================================
// Config
// ===========================================================================

fn micro_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15,
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        supersedes_demotion: 0.15,
        child_rescue_only: false, // KS68.3: let children compete in Pipe A (dedup guard active)
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

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

fn print_results(label: &str, results: &[EchoResult]) {
    println!("\n{label}:");
    for (i, r) in results.iter().enumerate().take(5) {
        println!(
            "  #{}: sim={:.3} score={:.3} {}",
            i + 1,
            r.similarity,
            r.final_score,
            &r.content[..r.content.len().min(100)]
        );
    }
}

// ===========================================================================
// The 20 curated memories — designed for maximum coverage in minimum size
// ===========================================================================

/// Seed 20 memories that cover every hard case pattern.
/// Returns the engine for querying.
fn seed_micro_dataset(engine: &EchoEngine, rt: &tokio::runtime::Runtime) {
    rt.block_on(async {
        // --- Identity & basics (gravity well bait) ---
        // M1: Identity fact — the gravity well source
        engine.store(
            "My name is Sam Torres and I'm a backend engineer. I'm 29 years old.",
            "session1"
        ).await.unwrap();

        // M2: Relationship
        engine.store(
            "My partner Jordan is a UX designer at Figma. We met at a Rust meetup in 2022.",
            "session1"
        ).await.unwrap();

        // M3: Pet (should not be confused with identity)
        engine.store(
            "I have a golden retriever named Pixel and I'm allergic to cats.",
            "session1"
        ).await.unwrap();

        // --- Work: supersession chain ---
        // M4: OLD job (should be superseded by M5)
        engine.store(
            "I work at Shopify on the payments team. I joined in 2019 after graduating.",
            "session2"
        ).await.unwrap();

        // M5: NEW job (supersedes M4)
        engine.store(
            "I just started at Stripe as a senior engineer on the billing infrastructure team. I left Shopify last month.",
            "session3"
        ).await.unwrap();

        // --- Location: supersession chain ---
        // M6: OLD location (should be superseded by M7)
        engine.store(
            "I live in Oakland, California. I moved here from Vancouver after college.",
            "session1"
        ).await.unwrap();

        // M7: NEW location (supersedes M6)
        engine.store(
            "I moved from Oakland to San Francisco last month to be closer to the Stripe office.",
            "session4"
        ).await.unwrap();

        // --- Multi-entity project facts ---
        // M8: Project with specific tech stack
        engine.store(
            "ShrimPK is my open-source AI memory kernel written in Rust. It uses fastembed for embeddings and has 11 crates in the workspace.",
            "session2"
        ).await.unwrap();

        // M9: Second project with different stack
        engine.store(
            "MLTK is my VS Code extension for ML workflows. It's built with React and TypeScript and has 200 downloads on the marketplace.",
            "session2"
        ).await.unwrap();

        // --- Preference evolution ---
        // M10: OLD IDE preference
        engine.store(
            "I use VS Code with the GitHub Copilot extension for all my development work.",
            "session1"
        ).await.unwrap();

        // M11: NEW IDE preference (supersedes M10)
        engine.store(
            "I switched from VS Code to Neovim with LazyVim config. The modal editing is so much faster for Rust development.",
            "session4"
        ).await.unwrap();

        // M12: OLD OS preference
        engine.store(
            "I run Windows 11 on my main machine and macOS on my work laptop.",
            "session1"
        ).await.unwrap();

        // M13: NEW OS preference (supersedes M12)
        engine.store(
            "I switched to Arch Linux with Hyprland on my personal machine. Tiling window managers are a game changer.",
            "session5"
        ).await.unwrap();

        // --- Technical preferences (non-evolving) ---
        // M14: Language preferences
        engine.store(
            "I prefer Rust for systems programming and Go for microservices. Python is only for scripts.",
            "session2"
        ).await.unwrap();

        // M15: Database preferences
        engine.store(
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics. I tried MongoDB but went back to Postgres.",
            "session3"
        ).await.unwrap();

        // --- Dense multi-fact memories ---
        // M16: Dense lifestyle (should extract 5+ facts)
        engine.store(
            "My daily routine: I wake up at 6am, brew pour-over coffee with a Hario V60, meditate for 15 minutes, then do a 30-minute jiu-jitsu drill. I practice Brazilian jiu-jitsu three times a week at a Gracie gym. I'm vegetarian and meal prep on Sundays.",
            "session3"
        ).await.unwrap();

        // M17: Dense career/education (should extract 4+ facts)
        engine.store(
            "I graduated from the University of British Columbia with a CS degree in 2019. I interned at Google the summer before. My thesis was on distributed consensus algorithms. I'm now studying for the AWS Solutions Architect certification.",
            "session2"
        ).await.unwrap();

        // --- Temporal facts ---
        // M18: Travel with specific dates
        engine.store(
            "I visited Tokyo last November for two weeks and stayed in Shinjuku. I practiced my Japanese — I'm at JLPT N3 level. Planning to go to Barcelona in April 2027.",
            "session4"
        ).await.unwrap();

        // M19: Goal with deadline
        engine.store(
            "I'm working on a patent for ShrimPK's push-based memory architecture. The filing deadline is April 15, 2026. I also want to present at ROSCon which is on April 26.",
            "session5"
        ).await.unwrap();

        // M20: Recent status update
        engine.store(
            "Just finished KS67 sprint for ShrimPK — added schema-driven fact extraction with dynamic max_facts, embedding supersession, and subject diversity caps. Recall benchmark target is 75%.",
            "session5"
        ).await.unwrap();
    });
}

// ===========================================================================
// Benchmark runner
// ===========================================================================

/// Run all 20 questions and report scores.
fn run_benchmark(engine: &EchoEngine, rt: &tokio::runtime::Runtime) -> (usize, usize) {
    let questions: Vec<(&str, &[&str], &str)> = vec![
        // --- Information Extraction (5) ---
        (
            "What is Sam's job? Where does Sam work?",
            &["Stripe", "senior engineer", "billing"],
            "IE-1: Current job",
        ),
        (
            "Does Sam have any pets?",
            &["golden retriever", "Pixel"],
            "IE-2: Pet recall",
        ),
        (
            "Where did Sam go to university?",
            &["University of British Columbia", "UBC"],
            "IE-3: Education",
        ),
        (
            "What food allergies or dietary restrictions does Sam have?",
            &["vegetarian", "cats", "allergic"],
            "IE-4: Health/diet",
        ),
        (
            "What martial art does Sam practice?",
            &["jiu-jitsu", "Brazilian", "Gracie"],
            "IE-5: Hobby recall",
        ),
        // --- Knowledge Update / Supersession (5) ---
        (
            "Where does Sam work now?",
            &["Stripe"], // NOT Shopify
            "KU-1: Current job (supersession)",
        ),
        (
            "Where does Sam live currently?",
            &["San Francisco"], // NOT Oakland
            "KU-2: Current location (supersession)",
        ),
        (
            "What IDE does Sam use?",
            &["Neovim", "LazyVim"], // NOT VS Code
            "KU-3: Current IDE (preference evolution)",
        ),
        (
            "What OS does Sam run on personal machines?",
            &["Arch Linux", "Hyprland"], // NOT Windows
            "KU-4: Current OS (preference evolution)",
        ),
        (
            "What is Sam currently working on at ShrimPK?",
            &["KS67", "fact extraction", "schema"],
            "KU-5: Recent status",
        ),
        // --- Temporal Reasoning (3) ---
        (
            "When did Sam start at the current job?",
            &["Stripe", "just started", "last month"],
            "TR-1: Job start timing",
        ),
        (
            "Where has Sam traveled recently?",
            &["Tokyo", "November", "Shinjuku"],
            "TR-2: Recent travel",
        ),
        (
            "What upcoming deadlines does Sam have?",
            &["patent", "April 15", "ROSCon", "April 26"],
            "TR-3: Future deadlines",
        ),
        // --- Multi-Entity / Project Facts (4) ---
        (
            "What is ShrimPK? What tech stack does it use?",
            &["Rust", "memory kernel", "fastembed"],
            "ME-1: Project ShrimPK",
        ),
        (
            "What is MLTK? What is it built with?",
            &["VS Code", "React", "TypeScript", "extension"],
            "ME-2: Project MLTK",
        ),
        (
            "What programming languages does Sam prefer?",
            &["Rust", "Go"],
            "ME-3: Language preferences",
        ),
        (
            "What databases does Sam use?",
            &["PostgreSQL", "ClickHouse"],
            "ME-4: Database preferences",
        ),
        // --- Preference Tracking (3) ---
        (
            "How does Sam make coffee?",
            &["pour-over", "Hario", "V60"],
            "PT-1: Coffee method",
        ),
        (
            "Who is Sam's partner?",
            &["Jordan", "UX designer", "Figma"],
            "PT-2: Relationship",
        ),
        (
            "What language is Sam learning?",
            &["Japanese", "JLPT", "N3"],
            "PT-3: Language learning",
        ),
    ];

    let mut passed = 0;
    let total = questions.len();

    for (query, needles, label) in &questions {
        let results = rt.block_on(async { engine.echo(query, 5).await.expect("echo") });

        let hit = any_needle_in_top_n(&results, 3, needles);
        if hit {
            passed += 1;
        }

        let status = if hit { "PASS" } else { "FAIL" };
        print_results(&format!("[{status}] {label} — \"{query}\""), &results);
    }

    println!("\n============================================================");
    println!(
        "MICRO-BENCHMARK RESULT: {passed}/{total} ({:.0}%)",
        passed as f64 / total as f64 * 100.0
    );
    println!("============================================================");

    (passed, total)
}

// ===========================================================================
// Test: embedding-only (no consolidation, instant)
// ===========================================================================

#[test]
#[ignore = "requires fastembed model download"]
fn micro_benchmark_embedding_only() {
    let dir = tempdir().expect("temp dir");
    let config = micro_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_micro_dataset(&engine, &rt);

    println!("\n=== MICRO-BENCHMARK: Embedding-only (no enrichment) ===\n");
    let (passed, total) = run_benchmark(&engine, &rt);
    drop(rt);

    println!("\nBaseline (no consolidation): {passed}/{total}");
    // This measures raw embedding recall — no enriched children, no supersession
}

// ===========================================================================
// Test: with consolidation (requires Ollama, ~2 min)
// ===========================================================================

#[test]
#[ignore = "requires fastembed model + Ollama"]
fn micro_benchmark_with_consolidation() {
    let dir = tempdir().expect("temp dir");
    let config = micro_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_micro_dataset(&engine, &rt);

    // Run consolidation (enriches all 20 memories with the v2 extraction pipeline)
    println!("\n=== Running consolidation (this takes ~2 min with Ollama) ===");
    let start = std::time::Instant::now();
    let consol_result = rt.block_on(async { engine.consolidate_now().await });
    let elapsed = start.elapsed();
    println!("Consolidation completed in {:.1}s", elapsed.as_secs_f64());
    println!("Result: {:?}", consol_result);

    // Run a second consolidation pass for remaining un-enriched memories
    // (MAX_ENRICHMENTS_PER_CYCLE = 10, so 20 memories need 2 passes)
    println!("\n=== Running second consolidation pass ===");
    let start2 = std::time::Instant::now();
    let consol_result2 = rt.block_on(async { engine.consolidate_now().await });
    let elapsed2 = start2.elapsed();
    println!("Second pass completed in {:.1}s", elapsed2.as_secs_f64());
    println!("Result: {:?}", consol_result2);

    println!("\n=== MICRO-BENCHMARK: With KS67 enrichment ===\n");
    let (passed, total) = run_benchmark(&engine, &rt);
    drop(rt);

    println!("\nWith consolidation: {passed}/{total}");
    assert!(
        passed >= 15,
        "KS67 target: at least 75% (15/20). Got {passed}/20"
    );
}
