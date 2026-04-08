//! LME (LongMemEval) micro-benchmarks — deterministic retrieval tests inspired
//! by the failure modes found in the LME-S 500-question overnight run.
//!
//! Each benchmark uses `#[ignore]` because it loads the fastembed model (~30 MB)
//! which is too slow for the regular `cargo test` cycle.
//! Run with: `cargo test -p shrimpk-memory -- --ignored`
//!
//! ## Failure modes tested
//!
//! 1. **multi-session-scatter** (LME 22% of total wrong) — facts about the same
//!    topic are spread across multiple conversation sessions. The engine must
//!    retrieve all relevant fragments, not just the single best match.
//!
//! 2. **knowledge-update / supersession** (LME 9% of total wrong) — a fact is
//!    updated (e.g., salary changes from $80k to $95k). The engine must return
//!    the newest value and demote or hide the old one.
//!
//! 3. **preference-recall** (LME 5.8% of total wrong) — user states a personal
//!    preference (e.g., "I only drink oat milk"). A later question about a
//!    related topic should surface that preference.
//!
//! 4. **temporal-reasoning** (LME 22% of total wrong) — events stored with
//!    explicit dates. Queries ask for ordering or duration. The engine must
//!    retrieve the right events so a reader LLM can compute the answer.

use chrono::{Duration, Utc};
use shrimpk_core::{EchoConfig, MemoryEntry};
use shrimpk_memory::EchoEngine;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal EchoConfig pointing at a temp directory.
/// Low threshold (0.10) to ensure we don't miss anything during benchmarks.
fn bench_config(dir: &std::path::Path) -> EchoConfig {
    EchoConfig {
        max_memories: 500,
        similarity_threshold: 0.10,
        max_echo_results: 20,
        data_dir: dir.to_path_buf(),
        use_lsh: false,    // brute-force for determinism
        use_bloom: false,  // skip Bloom for small fixture
        use_labels: false, // skip label classification overhead
        ..Default::default()
    }
}

/// Build a MemoryEntry with a real embedding from the engine's embedder.
fn make_entry(engine: &EchoEngine, content: &str, source: &str) -> MemoryEntry {
    let embedding = engine
        .embed_text_for_test(content)
        .expect("embedding should succeed");
    MemoryEntry::new(content.to_string(), embedding, source.to_string())
}

/// Build a MemoryEntry backdated by `days_ago` days.
fn make_entry_aged(engine: &EchoEngine, content: &str, source: &str, days_ago: i64) -> MemoryEntry {
    let mut entry = make_entry(engine, content, source);
    entry.created_at = Utc::now() - Duration::days(days_ago);
    entry
}

/// Check if any echo result content contains the expected substring (case-insensitive).
fn any_result_contains(results: &[shrimpk_core::EchoResult], needle: &str) -> bool {
    let needle_lower = needle.to_lowercase();
    results
        .iter()
        .any(|r| r.content.to_lowercase().contains(&needle_lower))
}

/// Count how many of the expected needles appear in at least one echo result.
fn count_retrieved(results: &[shrimpk_core::EchoResult], needles: &[&str]) -> usize {
    needles
        .iter()
        .filter(|needle| any_result_contains(results, needle))
        .count()
}

// ===========================================================================
// Benchmark 1: multi-session scatter
// ===========================================================================
//
// Failure mode: facts about the same topic are stored across separate sessions.
// The user asks an aggregation question ("How many X?") and the engine must
// retrieve ALL relevant fragments, not just the top-1.
//
// LME examples:
//   Q: "How many projects have I led?" — expected 2, got 5 (hallucinated)
//   Q: "How many model kits have I worked on?" — expected 5, got 30
//
// Target: retrieve >= 4/5 camping trips so a reader LLM can count correctly.
// Current expected baseline: TBD (first run establishes it).

#[tokio::test]
#[ignore]
async fn benchmark_multi_session() {
    let tmp = TempDir::new().expect("tempdir");
    let config = bench_config(tmp.path());
    let engine = EchoEngine::new(config).expect("engine init");

    // 5 camping trips stored across different "sessions" (sources)
    let trips = [
        (
            "I went camping at Yosemite National Park for 3 days last June.",
            "session-1",
            90,
        ),
        (
            "My camping trip to Joshua Tree was 2 days in August.",
            "session-2",
            60,
        ),
        (
            "Spent a weekend camping at Big Sur with friends in September.",
            "session-3",
            45,
        ),
        (
            "I did a solo camping trip at Death Valley for 1 day in October.",
            "session-4",
            30,
        ),
        (
            "Our family camping trip to Sequoia was 2 days in November.",
            "session-5",
            15,
        ),
    ];

    // Distractor memories (outdoor activities but not camping trips)
    let distractors = [
        (
            "I went hiking at Mount Whitney last weekend.",
            "session-1",
            85,
        ),
        ("We visited the San Diego Zoo on Saturday.", "session-2", 70),
        (
            "I started a new rock climbing class at the local gym.",
            "session-3",
            50,
        ),
        (
            "Bought new running shoes for the half marathon.",
            "session-4",
            35,
        ),
        (
            "Signed up for a kayaking tour on Lake Tahoe.",
            "session-5",
            20,
        ),
    ];

    for (content, source, days_ago) in &trips {
        let entry = make_entry_aged(&engine, content, source, *days_ago);
        engine.inject_entry(entry).await;
    }
    for (content, source, days_ago) in &distractors {
        let entry = make_entry_aged(&engine, content, source, *days_ago);
        engine.inject_entry(entry).await;
    }

    // The aggregation query — requires retrieving all 5 trip memories
    let results = engine
        .echo("How many camping trips did I go on?", 10)
        .await
        .expect("echo should succeed");

    // Check how many of the 5 trips appear in the results
    let trip_markers = [
        "Yosemite",
        "Joshua Tree",
        "Big Sur",
        "Death Valley",
        "Sequoia",
    ];
    let retrieved = count_retrieved(&results, &trip_markers);
    let total = trip_markers.len();

    println!("BENCHMARK RESULT: {retrieved}/{total}");
    println!("  Query: How many camping trips did I go on?");
    for r in &results {
        println!(
            "  [{:.3}] {}",
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // Soft assertion — this benchmark tracks progress, not gates CI.
    // Target: 5/5 (all trips retrieved). Acceptable: >= 4/5.
    assert!(
        retrieved >= 3,
        "multi-session: expected >= 3/{total} trips, got {retrieved}/{total}"
    );

    engine.shutdown().await;
}

// ===========================================================================
// Benchmark 2: knowledge-update / supersession (strict)
// ===========================================================================
//
// Failure mode: a fact is updated but the old version still surfaces in echo.
// The engine should demote or hide superseded memories.
//
// LME examples:
//   Q: "Mortgage pre-approval amount?" — expected $400k, got $350k (old)
//   Q: "Personal best 5K time?" — expected 25:50, got old time
//
// We store 3 entity updates (salary, city, car) with old then new values.
// Target: echo for each entity returns the NEW value ranked above the old.

#[tokio::test]
#[ignore]
async fn benchmark_knowledge_update_strict() {
    let tmp = TempDir::new().expect("tempdir");
    let config = bench_config(tmp.path());
    let engine = EchoEngine::new(config).expect("engine init");

    // 3 knowledge updates: (old_content, new_content, query, new_marker, old_marker)
    let updates: Vec<(&str, &str, &str, &str, &str)> = vec![
        (
            "My salary is $80,000 per year at the tech company.",
            "I got a raise, my salary is now $95,000 per year.",
            "What is my current salary?",
            "95,000",
            "80,000",
        ),
        (
            "I live in Portland, Oregon with my partner.",
            "We just moved to Denver, Colorado last month.",
            "Where do I live now?",
            "Denver",
            "Portland",
        ),
        (
            "I drive a 2018 Honda Civic for my daily commute.",
            "I traded in my car and now drive a 2024 Toyota RAV4.",
            "What car do I drive?",
            "RAV4",
            "Civic",
        ),
    ];

    let mut passed = 0;
    let total = updates.len();

    for (old_content, new_content, query, new_marker, old_marker) in &updates {
        // Store old value first (30 days ago), then new value (2 days ago)
        let old_entry = make_entry_aged(&engine, old_content, "conversation", 30);
        let old_id = old_entry.id.clone();
        engine.inject_entry(old_entry).await;

        let new_entry = make_entry_aged(&engine, new_content, "conversation", 2);
        let new_id = new_entry.id.clone();
        engine.inject_entry(new_entry).await;

        // Create supersession edge: old -> new
        engine.inject_supersedes_edge(&old_id, &new_id).await;

        let results = engine.echo(query, 5).await.expect("echo should succeed");

        // Check: new value should appear; ideally ranked above old value
        let has_new = any_result_contains(&results, new_marker);
        let new_rank = results.iter().position(|r| {
            r.content
                .to_lowercase()
                .contains(&new_marker.to_lowercase())
        });
        let old_rank = results.iter().position(|r| {
            r.content
                .to_lowercase()
                .contains(&old_marker.to_lowercase())
        });

        let new_above_old = match (new_rank, old_rank) {
            (Some(n), Some(o)) => n < o, // lower index = higher rank
            (Some(_), None) => true,     // new present, old absent = perfect
            _ => false,
        };

        let this_passed = has_new && new_above_old;
        if this_passed {
            passed += 1;
        }

        println!(
            "  [{}] Q: {} | new_rank={:?} old_rank={:?}",
            if this_passed { "PASS" } else { "FAIL" },
            query,
            new_rank,
            old_rank,
        );
    }

    println!("BENCHMARK RESULT: {passed}/{total}");

    // Target: 3/3. Supersession demotion should handle all cases.
    assert!(
        passed >= 2,
        "knowledge-update: expected >= 2/{total}, got {passed}/{total}"
    );

    engine.shutdown().await;
}

// ===========================================================================
// Benchmark 3: preference recall
// ===========================================================================
//
// Failure mode: user states preferences in conversation, but when asked a
// related question later the preference is not surfaced.
//
// LME examples:
//   Q: "Recommend video editing resources" — should know user uses Premiere Pro
//   Q: "Suggest photography accessories" — should know user has Sony gear
//
// We store 4 preferences + distractors, then query with related topics.
// Target: at least 3/4 preferences surfaced.

#[tokio::test]
#[ignore]
async fn benchmark_preference_recall() {
    let tmp = TempDir::new().expect("tempdir");
    let config = bench_config(tmp.path());
    let engine = EchoEngine::new(config).expect("engine init");

    // Preferences stated in conversation
    let preferences = [
        "I only drink oat milk because I'm lactose intolerant.",
        "I use Vim as my primary text editor and refuse to switch to VS Code.",
        "I always fly Delta airlines because of their frequent flyer program.",
        "I prefer reading physical books over e-books or audiobooks.",
    ];

    // Distractor memories
    let distractors = [
        "I had a great latte at the new coffee shop downtown.",
        "The software update broke my build pipeline yesterday.",
        "My flight to Chicago was delayed by 2 hours.",
        "I finished reading The Great Gatsby last week.",
    ];

    for pref in &preferences {
        let entry = make_entry_aged(&engine, pref, "conversation", 14);
        engine.inject_entry(entry).await;
    }
    for dist in &distractors {
        let entry = make_entry_aged(&engine, dist, "conversation", 7);
        engine.inject_entry(entry).await;
    }

    // Queries that should surface the preferences
    let queries: Vec<(&str, &str)> = vec![
        ("What kind of milk should I add to the recipe?", "oat milk"),
        ("What text editor plugins should I recommend?", "Vim"),
        ("Which airline should I book for the trip?", "Delta"),
        (
            "Should I get the Kindle version of this book?",
            "physical books",
        ),
    ];

    let mut passed = 0;
    let total = queries.len();

    for (query, expected_marker) in &queries {
        let results = engine.echo(query, 10).await.expect("echo should succeed");
        let found = any_result_contains(&results, expected_marker);
        if found {
            passed += 1;
        }
        println!(
            "  [{}] Q: {} | looking for '{}'",
            if found { "PASS" } else { "FAIL" },
            query,
            expected_marker,
        );
    }

    println!("BENCHMARK RESULT: {passed}/{total}");

    // Target: 4/4. Preference memories should be semantically close to related queries.
    assert!(
        passed >= 2,
        "preference-recall: expected >= 2/{total}, got {passed}/{total}"
    );

    engine.shutdown().await;
}

// ===========================================================================
// Benchmark 4: temporal reasoning (retrieval component)
// ===========================================================================
//
// Failure mode: events with explicit dates are stored. Queries ask about time
// between events, ordering, or "how many days ago". The engine must retrieve
// the correct events so a reader LLM can compute the answer.
//
// LME examples:
//   Q: "How many days between MoMA visit and Ancient Civilizations exhibit?"
//   Q: "Which three events happened in order?"
//   Q: "How many weeks since I received the crystal chandelier?"
//
// We store 5 dated events and query for specific pairs/ordering.
// Target: for each temporal query, both relevant events are in top-10 results.

#[tokio::test]
#[ignore]
async fn benchmark_temporal_reasoning() {
    let tmp = TempDir::new().expect("tempdir");
    let config = bench_config(tmp.path());
    let engine = EchoEngine::new(config).expect("engine init");

    // Events with explicit dates baked into the content
    let events = [
        (
            "On March 1st, I visited the Museum of Modern Art with Sarah.",
            "session-1",
            38,
        ),
        (
            "On March 8th, I went to the Ancient Civilizations exhibit at the Metropolitan Museum.",
            "session-2",
            31,
        ),
        (
            "On February 14th, I volunteered at the animal shelter fundraising dinner.",
            "session-1",
            53,
        ),
        (
            "On March 15th, I started learning piano at the community center.",
            "session-3",
            24,
        ),
        (
            "On February 28th, my aunt gave me a crystal chandelier when we met for lunch.",
            "session-2",
            39,
        ),
    ];

    // Distractor events (no specific dates, or unrelated)
    let distractors = [
        (
            "I've been going to the gym more regularly this year.",
            "session-1",
            20,
        ),
        (
            "The weather has been unusually warm for this time of year.",
            "session-3",
            15,
        ),
        (
            "I need to schedule a dentist appointment soon.",
            "session-2",
            10,
        ),
    ];

    for (content, source, days_ago) in &events {
        let entry = make_entry_aged(&engine, content, source, *days_ago);
        engine.inject_entry(entry).await;
    }
    for (content, source, days_ago) in &distractors {
        let entry = make_entry_aged(&engine, content, source, *days_ago);
        engine.inject_entry(entry).await;
    }

    // Temporal queries: (query, [markers that must both be retrieved])
    let temporal_queries: Vec<(&str, Vec<&str>)> = vec![
        (
            "How many days passed between my visit to the Museum of Modern Art and the Ancient Civilizations exhibit?",
            vec!["Museum of Modern Art", "Ancient Civilizations"],
        ),
        (
            "When did I volunteer at the animal shelter fundraising dinner?",
            vec!["animal shelter", "February 14th"],
        ),
        (
            "How many weeks ago did I receive the crystal chandelier from my aunt?",
            vec!["crystal chandelier", "aunt"],
        ),
        (
            "Did I start learning piano before or after visiting MoMA?",
            vec!["piano", "Museum of Modern Art"],
        ),
    ];

    let mut passed = 0;
    let total = temporal_queries.len();

    for (query, required_markers) in &temporal_queries {
        let results = engine.echo(query, 10).await.expect("echo should succeed");
        let all_found = required_markers
            .iter()
            .all(|marker| any_result_contains(&results, marker));
        if all_found {
            passed += 1;
        }
        let found_count = required_markers
            .iter()
            .filter(|marker| any_result_contains(&results, marker))
            .count();
        println!(
            "  [{}] Q: {}... | {}/{} markers found",
            if all_found { "PASS" } else { "FAIL" },
            &query[..query.len().min(70)],
            found_count,
            required_markers.len(),
        );
    }

    println!("BENCHMARK RESULT: {passed}/{total}");

    // Target: 4/4. All temporal events should be retrievable by their content.
    assert!(
        passed >= 2,
        "temporal-reasoning: expected >= 2/{total}, got {passed}/{total}"
    );

    engine.shutdown().await;
}
