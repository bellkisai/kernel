//! LME (LongMemEval) micro-benchmarks — deterministic retrieval tests targeting
//! the four dominant failure modes found in the LME-S 500-question overnight run.
//!
//! Each benchmark uses `#[ignore]` because it loads the fastembed model (~23 MB ONNX)
//! which is too slow for the regular `cargo test` cycle.
//!
//! Run all:
//!     cargo test -p shrimpk-memory --features test-helpers --test lme_micro_benchmark -- --ignored --nocapture
//!
//! Run one:
//!     cargo test -p shrimpk-memory --features test-helpers --test lme_micro_benchmark benchmark_multi_session -- --ignored --nocapture
//!
//! ## Failure modes tested
//!
//! 1. **multi-session scatter** (25.9% of LME-S wrong) — facts spread across sessions,
//!    reader hallucinates aggregates when not all fragments are retrieved.
//! 2. **supersession / knowledge-update** (10.6%) — old value still returned after update.
//! 3. **preference recall** (5.8%) — user preferences not surfaced for related queries.
//! 4. **temporal reasoning** (22.0%) — date-stamped events not retrieved for temporal queries.

use chrono::{Duration, Utc};
use shrimpk_core::{EchoConfig, EchoResult, MemoryEntry, MemoryId};
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use tempfile::tempdir;

// ===========================================================================
// Config
// ===========================================================================

fn lme_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.10,
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        supersedes_demotion: 0.15,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Check if ANY of the top-N results contains the needle (case-insensitive).
fn top_n_contains(results: &[EchoResult], n: usize, needle: &str) -> bool {
    let lc = needle.to_lowercase();
    results
        .iter()
        .take(n)
        .any(|r| r.content.to_lowercase().contains(&lc))
}

/// Check if any needle in the list appears in top-N results.
fn any_needle_in_top_n(results: &[EchoResult], n: usize, needles: &[&str]) -> bool {
    needles
        .iter()
        .any(|needle| top_n_contains(results, n, needle))
}

/// Print top-5 results for diagnostic output.
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

/// Build a MemoryEntry with a real embedding from the engine's embedder,
/// backdated by `days_ago` days.
fn make_entry_aged(engine: &EchoEngine, content: &str, source: &str, days_ago: i64) -> MemoryEntry {
    let embedding = engine
        .embed_text_for_test(content)
        .expect("embedding should succeed");
    let mut entry = MemoryEntry::new(content.to_string(), embedding, source.to_string());
    entry.created_at = Utc::now() - Duration::days(days_ago);
    entry
}

// ===========================================================================
// Benchmark 1: multi-session scatter
// ===========================================================================
// Tracks multi-session scatter failure (25.9% of LME-S failures).
// Target: 1/5 pre-fix, 4/5 post cross-session consolidation.

#[test]
#[ignore = "requires fastembed model download"]
fn benchmark_multi_session() {
    let dir = tempdir().expect("temp dir");
    let config = lme_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Seed 4 camping trips across different sessions for user "Alex"
    rt.block_on(async {
        let e1 = make_entry_aged(
            &engine,
            "Alex went camping in Yosemite for 3 days in June",
            "session-1",
            90,
        );
        engine.inject_entry(e1).await;

        let e2 = make_entry_aged(
            &engine,
            "Alex went camping in Zion for 2 days in July",
            "session-2",
            75,
        );
        engine.inject_entry(e2).await;

        let e3 = make_entry_aged(
            &engine,
            "Alex went camping in Olympic for 3 days in August",
            "session-3",
            60,
        );
        engine.inject_entry(e3).await;

        // No session 4 — intentional gap

        let e4 = make_entry_aged(
            &engine,
            "Alex went on a camping trip to Acadia for 2 days in September",
            "session-5",
            30,
        );
        engine.inject_entry(e4).await;

        // Distractors — outdoor but not camping
        let d1 = make_entry_aged(
            &engine,
            "Alex went hiking at Mount Whitney last weekend",
            "session-1",
            85,
        );
        engine.inject_entry(d1).await;

        let d2 = make_entry_aged(
            &engine,
            "Alex visited the San Diego Zoo with friends on Saturday",
            "session-2",
            70,
        );
        engine.inject_entry(d2).await;

        let d3 = make_entry_aged(
            &engine,
            "Alex started a new rock climbing class at the gym",
            "session-3",
            50,
        );
        engine.inject_entry(d3).await;
    });

    // Questions — pass if expected answer appears in top-3 results
    let questions: Vec<(&str, &[&str], &str)> = vec![
        (
            "How many total days did Alex spend camping?",
            &["Yosemite", "Zion", "Olympic", "Acadia"],
            "MS-1: Total camping days (needs all 4 trips)",
        ),
        (
            "How many camping trips did Alex take?",
            &["Yosemite", "Zion", "Olympic", "Acadia"],
            "MS-2: Trip count (needs all 4 trips)",
        ),
        (
            "Which parks did Alex visit for camping?",
            &["Yosemite", "Zion", "Olympic", "Acadia"],
            "MS-3: Park enumeration",
        ),
        (
            "What outdoor activities did Alex do in the summer?",
            &["camping", "Yosemite", "Zion", "Olympic"],
            "MS-4: Summer activities",
        ),
        (
            "Where did Alex go camping most recently?",
            &["Acadia"],
            "MS-5: Most recent trip",
        ),
    ];

    // For multi-session, we count how many of the 4 park names appear
    // in the top-10 results for each question. A question passes if
    // ALL expected needles appear (for MS-1..MS-3) or any needle (for MS-4..MS-5).
    let mut passed = 0;
    let total = questions.len();

    for (query, needles, label) in &questions {
        let results = rt.block_on(async { engine.echo(query, 10).await.expect("echo") });

        // For multi-session: check if ALL needles are found in top-10
        let all_found = needles
            .iter()
            .all(|needle| top_n_contains(&results, 10, needle));
        let any_found = any_needle_in_top_n(&results, 3, needles);

        // MS-5 only needs one needle, MS-1..MS-4 need all
        let hit = if needles.len() == 1 {
            any_found
        } else {
            all_found
        };

        if hit {
            passed += 1;
        }
        let status = if hit { "PASS" } else { "FAIL" };
        let found_count = needles
            .iter()
            .filter(|n| top_n_contains(&results, 10, n))
            .count();
        print_results(
            &format!(
                "[{status}] {label} — \"{query}\" ({found_count}/{} markers)",
                needles.len()
            ),
            &results,
        );
    }

    println!("\nBENCHMARK RESULT: {passed}/{total}");

    rt.block_on(async { engine.shutdown().await });
}

// ===========================================================================
// Benchmark 2: knowledge-update / supersession (strict)
// ===========================================================================
// Tracks supersession failure (10.6% of LME-S failures). Current NLI: 0/5.
// Target: 5/5 after NLI-based contradiction detection (KS74 fix).

#[test]
#[ignore = "requires fastembed model download"]
fn benchmark_knowledge_update_strict() {
    let dir = tempdir().expect("temp dir");
    let config = lme_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Seed old-then-new fact pairs for user "Sam"
    // Each old fact is stored 60 days ago, new fact 5 days ago
    let pairs: Vec<(&str, &str, &str, &str, &str)> = vec![
        (
            "Sam works at Shopify as a senior engineer",
            "Sam just started a new job at Stripe as a staff engineer",
            "Where does Sam work?",
            "Stripe",
            "Shopify",
        ),
        (
            "Sam lives in Oakland, California",
            "Sam recently moved to San Francisco",
            "Where does Sam live?",
            "San Francisco",
            "Oakland",
        ),
        (
            "Sam drives a Honda Civic",
            "Sam sold the Honda and bought a Tesla Model 3",
            "What car does Sam drive?",
            "Tesla",
            "Honda Civic",
        ),
        (
            "Sam's favorite programming language is Ruby",
            "Sam has been using Rust exclusively for the past year",
            "What is Sam's favorite programming language?",
            "Rust",
            "Ruby",
        ),
        (
            "Sam's gym is Planet Fitness",
            "Sam switched to Equinox last month",
            "What gym does Sam go to?",
            "Equinox",
            "Planet Fitness",
        ),
    ];

    // Store all pairs and create supersession edges
    let mut old_ids: Vec<MemoryId> = Vec::new();
    let mut new_ids: Vec<MemoryId> = Vec::new();

    rt.block_on(async {
        for (old_content, new_content, _query, _new_marker, _old_marker) in &pairs {
            let old_entry = make_entry_aged(&engine, old_content, "conversation", 60);
            let old_id = old_entry.id.clone();
            engine.inject_entry(old_entry).await;
            old_ids.push(old_id);

            let new_entry = make_entry_aged(&engine, new_content, "conversation", 5);
            let new_id = new_entry.id.clone();
            engine.inject_entry(new_entry).await;
            new_ids.push(new_id);
        }

        // Create supersession edges: old → new
        for (old_id, new_id) in old_ids.iter().zip(new_ids.iter()) {
            engine.inject_supersedes_edge(old_id, new_id).await;
        }
    });

    let mut passed = 0;
    let total = pairs.len();

    for (i, (_old_content, _new_content, query, new_marker, old_marker)) in pairs.iter().enumerate()
    {
        let results = rt.block_on(async { engine.echo(query, 5).await.expect("echo") });

        // Pass: NEW value in top-1 AND OLD value NOT in top-1
        let top1_has_new = top_n_contains(&results, 1, new_marker);
        let top1_has_old = top_n_contains(&results, 1, old_marker);
        let hit = top1_has_new && !top1_has_old;

        if hit {
            passed += 1;
        }
        let status = if hit { "PASS" } else { "FAIL" };
        print_results(
            &format!(
                "[{status}] KU-{}: \"{query}\" (want={new_marker}, reject={old_marker})",
                i + 1
            ),
            &results,
        );
    }

    println!("\nBENCHMARK RESULT: {passed}/{total}");

    rt.block_on(async { engine.shutdown().await });
}

// ===========================================================================
// Benchmark 3: preference recall
// ===========================================================================
// Tracks preference recall via embedding-only (no reader LLM). Validates that
// ShrimPK stores and retrieves preference facts correctly.
// Target: 5/5 (these are simple single-session, failure here = embedding or storage bug).

#[test]
#[ignore = "requires fastembed model download"]
fn benchmark_preference_recall() {
    let dir = tempdir().expect("temp dir");
    let config = lme_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Seed 5 preferences for user "Jordan"
    rt.block_on(async {
        let prefs = [
            "Jordan's favorite coffee drink is a cortado with oat milk",
            "Jordan loves hiking and prefers trails with elevation gain over flat walks",
            "Jordan's favorite framework for web development is SvelteKit",
            "Jordan listens to jazz and especially likes Miles Davis",
            "Jordan's preferred text editor is Neovim with a custom Lua config",
        ];
        for pref in &prefs {
            let entry = make_entry_aged(&engine, pref, "conversation", 14);
            engine.inject_entry(entry).await;
        }

        // Distractors — related topics but not direct preferences
        let distractors = [
            "Jordan had a great latte at the new coffee shop downtown",
            "Jordan completed the Pacific Crest Trail section hike last month",
            "Jordan deployed a new microservice using Next.js at work",
            "Jordan went to a blues concert with friends last Friday",
            "Jordan set up a new CI/CD pipeline using GitHub Actions",
        ];
        for dist in &distractors {
            let entry = make_entry_aged(&engine, dist, "conversation", 7);
            engine.inject_entry(entry).await;
        }
    });

    let questions: Vec<(&str, &[&str], &str)> = vec![
        (
            "What coffee drink does Jordan prefer?",
            &["cortado", "oat milk"],
            "PR-1: Coffee preference",
        ),
        (
            "What kind of hiking trails does Jordan prefer?",
            &["elevation"],
            "PR-2: Hiking preference",
        ),
        (
            "What web framework does Jordan like?",
            &["SvelteKit"],
            "PR-3: Web framework",
        ),
        (
            "What music genre does Jordan enjoy?",
            &["jazz", "Miles Davis"],
            "PR-4: Music preference",
        ),
        (
            "What text editor does Jordan use?",
            &["Neovim"],
            "PR-5: Editor preference",
        ),
    ];

    let mut passed = 0;
    let total = questions.len();

    for (query, needles, label) in &questions {
        let results = rt.block_on(async { engine.echo(query, 5).await.expect("echo") });

        let hit = any_needle_in_top_n(&results, 1, needles);
        if hit {
            passed += 1;
        }
        let status = if hit { "PASS" } else { "FAIL" };
        print_results(&format!("[{status}] {label} — \"{query}\""), &results);
    }

    println!("\nBENCHMARK RESULT: {passed}/{total}");

    rt.block_on(async { engine.shutdown().await });
}

// ===========================================================================
// Benchmark 4: temporal reasoning
// ===========================================================================
// Tracks temporal reasoning gaps (temporal-reasoning is 59% wrong on LME-S,
// held back by date arithmetic). Target: 3/5 pre-reader-fix, 5/5 after
// qwen2.5:3b reader upgrade.

#[test]
#[ignore = "requires fastembed model download"]
fn benchmark_temporal_reasoning() {
    let dir = tempdir().expect("temp dir");
    let config = lme_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Seed time-stamped travel events for user "Casey"
    rt.block_on(async {
        let events = [
            (
                "Casey visited Tokyo in March 2024 for 10 days",
                "session-1",
                400,
            ),
            (
                "Casey visited Paris in August 2024 for 7 days",
                "session-2",
                250,
            ),
            (
                "Casey visited New York in November 2024 for 4 days",
                "session-3",
                160,
            ),
            (
                "Casey's most recent trip was to Buenos Aires in February 2025 for 5 days",
                "session-4",
                60,
            ),
            (
                "Casey is planning a trip to Cape Town in June 2025",
                "session-5",
                10,
            ),
        ];
        for (content, source, days_ago) in &events {
            let entry = make_entry_aged(&engine, content, source, *days_ago);
            engine.inject_entry(entry).await;
        }

        // Distractors — non-travel temporal content
        let distractors = [
            (
                "Casey started a new job at a tech company in January 2025",
                "session-3",
                90,
            ),
            (
                "Casey signed up for a cooking class in December 2024",
                "session-2",
                130,
            ),
            ("Casey's lease expires in September 2025", "session-5", 5),
        ];
        for (content, source, days_ago) in &distractors {
            let entry = make_entry_aged(&engine, content, source, *days_ago);
            engine.inject_entry(entry).await;
        }
    });

    let questions: Vec<(&str, &[&str], &str)> = vec![
        (
            "What was Casey's most recent completed trip?",
            &["Buenos Aires"],
            "TE-1: Most recent trip",
        ),
        (
            "How many days did Casey spend in Paris?",
            &["Paris", "7 days"],
            "TE-2: Paris duration",
        ),
        (
            "Which trip did Casey take before New York?",
            &["Paris"],
            "TE-3: Trip ordering (before NY)",
        ),
        (
            "How many international trips has Casey taken?",
            &["Tokyo", "Paris", "Buenos Aires"],
            "TE-4: Trip count (needs multiple)",
        ),
        (
            "Where is Casey planning to travel next?",
            &["Cape Town"],
            "TE-5: Future trip",
        ),
    ];

    let mut passed = 0;
    let total = questions.len();

    for (query, needles, label) in &questions {
        let results = rt.block_on(async { engine.echo(query, 10).await.expect("echo") });

        // TE-4 needs all markers; others need any marker in top-1
        let hit = if needles.len() > 2 {
            // Multi-marker: all needles must appear in top-10
            needles.iter().all(|n| top_n_contains(&results, 10, n))
        } else {
            any_needle_in_top_n(&results, 1, needles)
        };

        if hit {
            passed += 1;
        }
        let status = if hit { "PASS" } else { "FAIL" };
        print_results(&format!("[{status}] {label} — \"{query}\""), &results);
    }

    println!("\nBENCHMARK RESULT: {passed}/{total}");

    rt.block_on(async { engine.shutdown().await });
}
