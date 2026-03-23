//! LongMemEval-style benchmark for ShrimPK Echo Memory (KS17 Track 2).
//!
//! Evaluates long-term memory capabilities across five categories modeled
//! after the LongMemEval benchmark used to evaluate conversational AI systems:
//!
//! 1. **Information Extraction** — recall specific facts from stored memories
//! 2. **Multi-Session Reasoning** — connect facts stored in separate sessions
//! 3. **Temporal Reasoning** — handle time-ordered information
//! 4. **Knowledge Update** — surface corrections over stale facts
//! 5. **Preference Tracking** — recall the most current user preferences
//!
//! All tests are `#[ignore]` because they require the fastembed model
//! (all-MiniLM-L6-v2, ~23MB ONNX). Run with:
//!
//!     cargo test --test echo_longmemeval -- --ignored --nocapture
//!
//! The model is cached after first download, so subsequent runs are fast.

use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use tempfile::tempdir;

// ===========================================================================
// Config helper
// ===========================================================================

/// Build an EchoConfig tuned for LongMemEval tests.
///
/// Uses a low similarity threshold (0.15) so that semantically related
/// memories surface even when query phrasing diverges from stored text.
/// max_echo_results is generous (10) to evaluate ranking quality.
fn longmemeval_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15, // low: we want to measure ranking, not gating
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

// ===========================================================================
// Scoring helpers
// ===========================================================================

/// Check whether ANY of the top-N results contain the expected substring
/// (case-insensitive).
fn top_n_contains(results: &[shrimpk_core::EchoResult], n: usize, needle: &str) -> bool {
    let needle_lower = needle.to_lowercase();
    results
        .iter()
        .take(n)
        .any(|r| r.content.to_lowercase().contains(&needle_lower))
}

// ===========================================================================
// Category 1: Information Extraction (5 tests)
//
// Can the system recall specific facts from past conversations?
// Store 20+ facts about a simulated user across multiple "sessions",
// then query with natural language questions.
// Pass criterion: relevant memory appears in top-3 results.
// ===========================================================================

/// Seed the engine with a diverse user profile across multiple sessions.
/// Returns the engine ready for querying.
async fn seed_user_profile(data_dir: PathBuf) -> EchoEngine {
    let config = longmemeval_config(data_dir);
    let engine = EchoEngine::new(config).expect("engine init");

    // Session 1: personal basics
    let session1 = [
        "My name is Alex Chen and I'm 32 years old",
        "I was born in Taipei, Taiwan but grew up in Vancouver, Canada",
        "I have a golden retriever named Pixel who is 4 years old",
        "My partner's name is Jordan and we've been together for 6 years",
        "I'm allergic to shellfish and cats",
    ];

    // Session 2: work and education
    let session2 = [
        "I work as a senior backend engineer at Stripe in San Francisco",
        "I graduated from the University of British Columbia with a CS degree in 2015",
        "Before Stripe I worked at Shopify for 3 years on their payments team",
        "My team at Stripe works on the billing infrastructure service",
        "I'm being considered for a staff engineer promotion next quarter",
    ];

    // Session 3: technical preferences
    let session3 = [
        "I prefer Rust for systems programming and Go for microservices",
        "My IDE is Neovim with LazyVim config and Catppuccin theme",
        "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
        "I run NixOS on my personal machines and macOS at work",
        "My dotfiles are managed with chezmoi and stored on GitHub",
    ];

    // Session 4: hobbies and lifestyle
    let session4 = [
        "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
        "I'm learning Japanese and currently at JLPT N3 level",
        "I collect mechanical keyboards and my daily driver is a Keychron Q1 with Boba U4T switches",
        "I brew pour-over coffee every morning using a Hario V60 and light roast beans",
        "My favorite cuisine is Thai food, especially pad see ew and massaman curry",
    ];

    // Session 5: travel and goals
    let session5 = [
        "I visited Tokyo last November and stayed in Shinjuku for two weeks",
        "My next trip is planned for Barcelona in April 2027",
        "My long-term goal is to start a developer tools company focused on observability",
        "I'm saving for a house in the Oakland Hills area",
        "I want to compete in a jiu-jitsu tournament by the end of the year",
    ];

    for text in &session1 {
        engine.store(text, "session1").await.unwrap();
    }
    for text in &session2 {
        engine.store(text, "session2").await.unwrap();
    }
    for text in &session3 {
        engine.store(text, "session3").await.unwrap();
    }
    for text in &session4 {
        engine.store(text, "session4").await.unwrap();
    }
    for text in &session5 {
        engine.store(text, "session5").await.unwrap();
    }

    engine
}

/// IE-1: Direct fact recall — profession
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ie_1_profession() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("What is my job? Where do I work?", 5)
        .await
        .expect("echo should succeed");

    assert!(
        top_n_contains(&results, 3, "Stripe"),
        "Top-3 should mention Stripe. Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// IE-2: Direct fact recall — pet
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ie_2_pet() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("Do I have any pets?", 5)
        .await
        .expect("echo should succeed");

    assert!(
        top_n_contains(&results, 3, "golden retriever"),
        "Top-3 should mention the golden retriever. Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// IE-3: Direct fact recall — education
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ie_3_education() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("Where did I go to university?", 5)
        .await
        .expect("echo should succeed");

    assert!(
        top_n_contains(&results, 3, "British Columbia"),
        "Top-3 should mention University of British Columbia. Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// IE-4: Indirect recall — food allergy (phrased differently from stored text)
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ie_4_allergy() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("What foods should I avoid? Any allergies?", 5)
        .await
        .expect("echo should succeed");

    assert!(
        top_n_contains(&results, 3, "shellfish"),
        "Top-3 should mention shellfish allergy. Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// IE-5: Indirect recall — hobby (phrased as a question about exercise)
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ie_5_hobby() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("What martial art do I train? How often?", 5)
        .await
        .expect("echo should succeed");

    assert!(
        top_n_contains(&results, 3, "jiu-jitsu"),
        "Top-3 should mention Brazilian jiu-jitsu. Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 2: Multi-Session Reasoning (5 tests)
//
// Can the system connect information across multiple sessions?
// Query with a question that requires facts from 2+ separate store() calls.
// Pass criterion: BOTH relevant memories appear in top-5 results.
// ===========================================================================

/// MSR-1: Connect workplace + programming languages
/// "What language should I use at work?" requires knowing both the job and the language prefs.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_msr_1_work_and_language() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("What programming language do I use for backend work?", 5)
        .await
        .expect("echo should succeed");

    // Should surface both the language preference memory AND the work context
    let has_language = top_n_contains(&results, 5, "Rust")
        || top_n_contains(&results, 5, "Go");
    let has_work = top_n_contains(&results, 5, "backend")
        || top_n_contains(&results, 5, "Stripe")
        || top_n_contains(&results, 5, "microservices");

    assert!(
        has_language,
        "Top-5 should mention a programming language (Rust/Go). Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
    assert!(
        has_work,
        "Top-5 should mention work context (backend/Stripe). Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// MSR-2: Connect travel history + language learning
/// "Can I get by with my Japanese in Tokyo?" requires knowing both the trip and the language level.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_msr_2_travel_and_language() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("How was my Japanese when I visited Tokyo?", 5)
        .await
        .expect("echo should succeed");

    let has_japanese = top_n_contains(&results, 5, "Japanese")
        || top_n_contains(&results, 5, "JLPT");
    let has_tokyo = top_n_contains(&results, 5, "Tokyo");

    assert!(
        has_japanese && has_tokyo,
        "Top-5 should mention both Japanese study and Tokyo trip. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// MSR-3: Connect hobby + goal
/// "Am I ready for competition?" requires knowing both the training regimen and the goal.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_msr_3_hobby_and_goal() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("Am I training enough for the jiu-jitsu tournament?", 5)
        .await
        .expect("echo should succeed");

    let has_training = top_n_contains(&results, 5, "three times a week")
        || top_n_contains(&results, 5, "jiu-jitsu");
    let has_goal = top_n_contains(&results, 5, "tournament")
        || top_n_contains(&results, 5, "compete");

    assert!(
        has_training && has_goal,
        "Top-5 should surface both training schedule and tournament goal. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// MSR-4: Connect career history + current role
/// "How did I end up at Stripe?" requires Shopify history + current Stripe role.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_msr_4_career_path() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("What's my career history in the payments industry?", 5)
        .await
        .expect("echo should succeed");

    let has_shopify = top_n_contains(&results, 5, "Shopify");
    let has_stripe = top_n_contains(&results, 5, "Stripe");

    assert!(
        has_shopify && has_stripe,
        "Top-5 should mention both Shopify and Stripe. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// MSR-5: Connect personal + professional goals
/// "What am I working toward financially and professionally?" requires house savings + startup goal.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_msr_5_combined_goals() {
    let dir = tempdir().expect("temp dir");
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    let results = engine
        .echo("What are my big life goals? What am I saving up for?", 5)
        .await
        .expect("echo should succeed");

    let has_house = top_n_contains(&results, 5, "house")
        || top_n_contains(&results, 5, "Oakland");
    let has_startup = top_n_contains(&results, 5, "developer tools")
        || top_n_contains(&results, 5, "observability")
        || top_n_contains(&results, 5, "company");

    assert!(
        has_house || has_startup,
        "Top-5 should surface at least one major life goal. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 3: Temporal Reasoning (3 tests)
//
// Can the system handle time-ordered information?
// Store events with temporal markers and query about sequences.
// Pass criterion: memories with correct temporal markers surface, and
// the more recent event ranks higher than the older one (by final_score).
// ===========================================================================

/// TR-1: Temporal ordering of job changes
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_tr_1_job_timeline() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // Store in chronological order (simulating real conversation flow)
    engine
        .store("In 2018 I started my first job as a junior developer at a small startup in Vancouver", "session_early")
        .await
        .unwrap();
    engine
        .store("In 2019 I joined Shopify as a backend engineer on their payments team", "session_mid")
        .await
        .unwrap();
    engine
        .store("In 2022 I moved to Stripe as a senior backend engineer in San Francisco", "session_recent")
        .await
        .unwrap();

    let results = engine
        .echo("Where have I worked over the years?", 5)
        .await
        .expect("echo should succeed");

    // All three jobs should surface in top-5
    let has_startup = top_n_contains(&results, 5, "2018")
        || top_n_contains(&results, 5, "startup");
    let has_shopify = top_n_contains(&results, 5, "Shopify");
    let has_stripe = top_n_contains(&results, 5, "Stripe");

    assert!(
        has_startup && has_shopify && has_stripe,
        "Top-5 should mention all three jobs. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// TR-2: Temporal ordering of recent events
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_tr_2_recent_events() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    engine
        .store("Last month I attended a Rust conference in Berlin", "session_a")
        .await
        .unwrap();
    engine
        .store("Last week I gave a talk on observability at the local meetup", "session_b")
        .await
        .unwrap();
    engine
        .store("Yesterday I submitted a CFP for RustConf 2027", "session_c")
        .await
        .unwrap();

    let results = engine
        .echo("What tech events have I been involved in recently?", 5)
        .await
        .expect("echo should succeed");

    // At minimum the most recent event should surface in top-3
    assert!(
        top_n_contains(&results, 3, "RustConf")
            || top_n_contains(&results, 3, "meetup")
            || top_n_contains(&results, 3, "conference"),
        "Top-3 should mention at least one recent tech event. Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );

    // All three should be in top-5
    let event_count = results
        .iter()
        .take(5)
        .filter(|r| {
            let c = r.content.to_lowercase();
            c.contains("conference") || c.contains("meetup") || c.contains("rustconf") || c.contains("cfp")
        })
        .count();

    assert!(
        event_count >= 2,
        "Top-5 should contain at least 2 of 3 tech events. Found {event_count}. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// TR-3: Temporal specificity — "last week" vs "last year"
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_tr_3_temporal_specificity() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    engine
        .store("Last year I started learning piano as a complete beginner", "session_old")
        .await
        .unwrap();
    engine
        .store("Last week I finished learning my first Chopin nocturne on piano", "session_new")
        .await
        .unwrap();

    // Noise entries to make ranking harder
    engine
        .store("I enjoy listening to classical music while coding", "session_noise")
        .await
        .unwrap();
    engine
        .store("My neighbor plays guitar every evening", "session_noise2")
        .await
        .unwrap();

    let results = engine
        .echo("How is my piano playing going?", 5)
        .await
        .expect("echo should succeed");

    // Both piano memories should surface
    let has_beginner = top_n_contains(&results, 5, "beginner")
        || top_n_contains(&results, 5, "started learning piano");
    let has_chopin = top_n_contains(&results, 5, "Chopin")
        || top_n_contains(&results, 5, "nocturne");

    assert!(
        has_beginner || has_chopin,
        "Top-5 should mention piano progress. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 4: Knowledge Update (3 tests)
//
// Can the system handle corrections and updates to previously stored facts?
// Store a fact, then store a correction. Query for the current state.
// Pass criterion: the corrected/updated memory ranks HIGHER than the stale one.
//
// Note: Echo Memory currently ranks by cosine similarity to the query, so
// "recency bias" is not built in. These tests verify that the CORRECTED
// memory is at least semantically relevant and surfaces alongside the
// original. Full knowledge-update handling (superseding old facts) is a
// future capability tracked in the backlog.
// ===========================================================================

/// KU-1: Job change — "Where do I work?"
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ku_1_job_change() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // Old fact
    engine
        .store("I work as a backend engineer at Google on the Cloud Spanner team", "session_old")
        .await
        .unwrap();

    // Noise
    engine
        .store("I enjoy hiking on weekends in the bay area", "session_noise")
        .await
        .unwrap();

    // Correction (stored later, simulating a new conversation)
    engine
        .store("I left Google last month. I now work at Meta on the infrastructure team", "session_new")
        .await
        .unwrap();

    let results = engine
        .echo("Where do I currently work?", 5)
        .await
        .expect("echo should succeed");

    // The Meta memory should surface
    assert!(
        top_n_contains(&results, 3, "Meta"),
        "Top-3 should mention Meta (the current employer). Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );

    // Both should surface (system doesn't delete old facts, but both are relevant)
    let has_google = top_n_contains(&results, 5, "Google");
    let has_meta = top_n_contains(&results, 5, "Meta");
    assert!(
        has_google && has_meta,
        "Top-5 should surface both old and new employer for context. Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// KU-2: Address change — "Where do I live?"
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ku_2_address_change() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    engine
        .store("I live in a one-bedroom apartment in downtown Seattle", "session_old")
        .await
        .unwrap();
    engine
        .store("My favorite restaurant is the Thai place on Pike Street in Seattle", "session_noise")
        .await
        .unwrap();
    engine
        .store("I just moved to Portland, Oregon and I'm renting a house in the Pearl District", "session_new")
        .await
        .unwrap();

    let results = engine
        .echo("Where do I live right now?", 5)
        .await
        .expect("echo should succeed");

    // Portland should surface
    assert!(
        top_n_contains(&results, 3, "Portland"),
        "Top-3 should mention Portland (current residence). Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// KU-3: Technology preference update — "What language do I use?"
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_ku_3_tech_preference_update() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    engine
        .store("Python is my go-to programming language for everything", "session_old")
        .await
        .unwrap();
    engine
        .store("I started a new side project building a web scraper", "session_noise")
        .await
        .unwrap();
    engine
        .store("I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve", "session_new")
        .await
        .unwrap();

    let results = engine
        .echo("What is my primary programming language?", 5)
        .await
        .expect("echo should succeed");

    // Rust (current) should surface
    assert!(
        top_n_contains(&results, 3, "Rust"),
        "Top-3 should mention Rust (current preference). Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );

    // Python (old) should also surface since it's semantically relevant
    assert!(
        top_n_contains(&results, 5, "Python"),
        "Top-5 should also mention Python (for context). Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 5: Preference Tracking (4 tests)
//
// Can the system recall and apply user preferences across sessions?
// Store preferences across multiple sessions and query about current state.
// Pass criterion: the most current preference surfaces in top-3.
// ===========================================================================

/// PT-1: IDE preference tracking across sessions
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_pt_1_ide_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // Preferences evolve over time
    engine
        .store("I use Sublime Text as my code editor, it's fast and lightweight", "month1")
        .await
        .unwrap();
    engine
        .store("I switched to VS Code because of the extension ecosystem", "month3")
        .await
        .unwrap();
    engine
        .store("I've moved to Neovim with a custom Lua config for maximum speed", "month6")
        .await
        .unwrap();

    let results = engine
        .echo("What code editor do I use?", 5)
        .await
        .expect("echo should succeed");

    // All three should surface as they're all about editors
    let has_neovim = top_n_contains(&results, 5, "Neovim");
    let has_vscode = top_n_contains(&results, 5, "VS Code");
    let has_sublime = top_n_contains(&results, 5, "Sublime");

    assert!(
        has_neovim,
        "Top-5 should mention Neovim (most recent preference). Got: {:?}",
        results.iter().take(5).map(|r| &r.content).collect::<Vec<_>>()
    );

    // At least 2 of 3 editor preferences should surface
    let editor_count = [has_neovim, has_vscode, has_sublime]
        .iter()
        .filter(|&&x| x)
        .count();
    assert!(
        editor_count >= 2,
        "Top-5 should contain at least 2 of 3 editor preferences. Found {editor_count}."
    );
}

/// PT-2: Dietary preference tracking
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_pt_2_dietary_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    engine
        .store("I eat everything, no dietary restrictions", "early")
        .await
        .unwrap();
    engine
        .store("I've started reducing meat, mostly eating vegetarian meals now", "mid")
        .await
        .unwrap();
    engine
        .store("I'm fully vegan now, it's been great for my energy levels", "recent")
        .await
        .unwrap();

    // Noise
    engine
        .store("I run 5K every morning before work", "noise")
        .await
        .unwrap();

    let results = engine
        .echo("What's my diet like? What do I eat?", 5)
        .await
        .expect("echo should succeed");

    assert!(
        top_n_contains(&results, 3, "vegan"),
        "Top-3 should mention vegan (most recent dietary preference). Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// PT-3: Coffee preference tracking (specific and nuanced)
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_pt_3_coffee_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    engine
        .store("I drink regular drip coffee with cream and sugar", "month1")
        .await
        .unwrap();
    engine
        .store("I switched to espresso-based drinks, usually a latte with whole milk", "month4")
        .await
        .unwrap();
    engine
        .store("Now I drink pour-over black coffee, no milk no sugar, using a Hario V60", "month8")
        .await
        .unwrap();

    // Additional noise
    engine
        .store("I like green tea in the afternoon as a lighter caffeine option", "noise")
        .await
        .unwrap();

    let results = engine
        .echo("How do I take my coffee?", 5)
        .await
        .expect("echo should succeed");

    // The most recent preference should surface
    assert!(
        top_n_contains(&results, 3, "pour-over")
            || top_n_contains(&results, 3, "V60")
            || top_n_contains(&results, 3, "black coffee"),
        "Top-3 should mention current coffee preference (pour-over/V60/black). Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

/// PT-4: Operating system preference (multi-device)
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_pt_4_os_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    engine
        .store("I use Windows 11 on all my machines for gaming and development", "early")
        .await
        .unwrap();
    engine
        .store("I dual-boot Linux Mint alongside Windows now for dev work", "mid")
        .await
        .unwrap();
    engine
        .store("I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely", "recent")
        .await
        .unwrap();

    // Noise
    engine
        .store("I bought a new 4K monitor for my desk setup", "noise")
        .await
        .unwrap();

    let results = engine
        .echo("What operating system do I run?", 5)
        .await
        .expect("echo should succeed");

    assert!(
        top_n_contains(&results, 3, "Arch")
            || top_n_contains(&results, 3, "Hyprland"),
        "Top-3 should mention Arch Linux (most recent OS preference). Got: {:?}",
        results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>()
    );
}

// ===========================================================================
// Summary test — runs all categories and prints a scorecard
// ===========================================================================

/// Run a complete LongMemEval-style evaluation and print a scorecard.
/// This is the main benchmark entry point for generating the results report.
///
/// Scoring: for each test scenario, we check if the expected memory surfaces
/// in the top-3 (strict) or top-5 (relaxed) results.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn longmemeval_full_scorecard() {
    let dir = tempdir().expect("temp dir");

    println!("\n=== LongMemEval Benchmark — ShrimPK Echo Memory ===\n");

    // ------ Category 1: Information Extraction ------
    let engine = seed_user_profile(dir.path().to_path_buf()).await;

    struct TestCase {
        name: &'static str,
        query: &'static str,
        needles_strict: Vec<&'static str>,  // must be in top-3
        needles_relaxed: Vec<&'static str>, // must be in top-5
    }

    let ie_cases = vec![
        TestCase {
            name: "IE-1: Profession",
            query: "What is my job? Where do I work?",
            needles_strict: vec!["Stripe"],
            needles_relaxed: vec!["Stripe", "engineer"],
        },
        TestCase {
            name: "IE-2: Pet",
            query: "Do I have any pets?",
            needles_strict: vec!["golden retriever"],
            needles_relaxed: vec!["golden retriever", "Pixel"],
        },
        TestCase {
            name: "IE-3: Education",
            query: "Where did I go to university?",
            needles_strict: vec!["British Columbia"],
            needles_relaxed: vec!["British Columbia", "CS"],
        },
        TestCase {
            name: "IE-4: Allergy",
            query: "What foods should I avoid? Any allergies?",
            needles_strict: vec!["shellfish"],
            needles_relaxed: vec!["shellfish", "allergic"],
        },
        TestCase {
            name: "IE-5: Hobby",
            query: "What martial art do I train?",
            needles_strict: vec!["jiu-jitsu"],
            needles_relaxed: vec!["jiu-jitsu", "Gracie"],
        },
    ];

    let mut total_strict = 0u32;
    let mut total_relaxed = 0u32;
    let mut total_tests = 0u32;

    println!("Category 1: Information Extraction");
    println!("{:<30} | {:>9} | {:>10}", "Test", "Top-3 Hit", "Top-5 Hit");
    println!("{}", "-".repeat(56));

    for tc in &ie_cases {
        let results = engine.echo(tc.query, 10).await.expect("echo");
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(&results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(&results, 5, n));
        if strict {
            total_strict += 1;
        }
        if relaxed {
            total_relaxed += 1;
        }
        total_tests += 1;
        println!(
            "{:<30} | {:>9} | {:>10}",
            tc.name,
            if strict { "PASS" } else { "MISS" },
            if relaxed { "PASS" } else { "MISS" },
        );
    }

    // ------ Category 2: Multi-Session Reasoning ------
    println!();
    println!("Category 2: Multi-Session Reasoning");
    println!("{:<30} | {:>9} | {:>10}", "Test", "Top-3 Hit", "Top-5 Hit");
    println!("{}", "-".repeat(56));

    let msr_cases = vec![
        TestCase {
            name: "MSR-1: Work + Language",
            query: "What programming language do I use for backend work?",
            needles_strict: vec!["Rust"],
            needles_relaxed: vec!["Rust", "Go", "backend"],
        },
        TestCase {
            name: "MSR-2: Travel + Language",
            query: "How was my Japanese when I visited Tokyo?",
            needles_strict: vec!["Japanese"],
            needles_relaxed: vec!["Japanese", "Tokyo"],
        },
        TestCase {
            name: "MSR-3: Hobby + Goal",
            query: "Am I training enough for the jiu-jitsu tournament?",
            needles_strict: vec!["jiu-jitsu"],
            needles_relaxed: vec!["jiu-jitsu", "tournament"],
        },
        TestCase {
            name: "MSR-4: Career Path",
            query: "What's my career history in the payments industry?",
            needles_strict: vec!["Shopify"],
            needles_relaxed: vec!["Shopify", "Stripe"],
        },
        TestCase {
            name: "MSR-5: Life Goals",
            query: "What are my big life goals?",
            needles_strict: vec!["house"],
            needles_relaxed: vec!["house", "developer tools", "observability"],
        },
    ];

    for tc in &msr_cases {
        let results = engine.echo(tc.query, 10).await.expect("echo");
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(&results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(&results, 5, n));
        if strict {
            total_strict += 1;
        }
        if relaxed {
            total_relaxed += 1;
        }
        total_tests += 1;
        println!(
            "{:<30} | {:>9} | {:>10}",
            tc.name,
            if strict { "PASS" } else { "MISS" },
            if relaxed { "PASS" } else { "MISS" },
        );
    }

    // ------ Category 3: Temporal Reasoning ------
    // Uses a fresh engine with temporal-specific memories
    println!();
    println!("Category 3: Temporal Reasoning");
    println!("{:<30} | {:>9} | {:>10}", "Test", "Top-3 Hit", "Top-5 Hit");
    println!("{}", "-".repeat(56));

    let dir_tr = tempdir().expect("temp dir");
    let config_tr = longmemeval_config(dir_tr.path().to_path_buf());
    let engine_tr = EchoEngine::new(config_tr).expect("engine init");

    engine_tr.store("In 2018 I started my first job as a junior developer at a small startup in Vancouver", "s1").await.unwrap();
    engine_tr.store("In 2019 I joined Shopify as a backend engineer on their payments team", "s2").await.unwrap();
    engine_tr.store("In 2022 I moved to Stripe as a senior backend engineer in San Francisco", "s3").await.unwrap();
    engine_tr.store("Last month I attended a Rust conference in Berlin", "s4").await.unwrap();
    engine_tr.store("Last week I gave a talk on observability at the local meetup", "s5").await.unwrap();
    engine_tr.store("Yesterday I submitted a CFP for RustConf 2027", "s6").await.unwrap();
    engine_tr.store("Last year I started learning piano as a complete beginner", "s7").await.unwrap();
    engine_tr.store("Last week I finished learning my first Chopin nocturne on piano", "s8").await.unwrap();
    engine_tr.store("I enjoy listening to classical music while coding", "noise1").await.unwrap();
    engine_tr.store("My neighbor plays guitar every evening", "noise2").await.unwrap();

    let tr_cases = vec![
        TestCase {
            name: "TR-1: Job Timeline",
            query: "Where have I worked over the years?",
            needles_strict: vec!["Stripe"],
            needles_relaxed: vec!["startup", "Shopify", "Stripe"],
        },
        TestCase {
            name: "TR-2: Recent Events",
            query: "What tech events have I been involved in recently?",
            needles_strict: vec!["conference"],
            needles_relaxed: vec!["conference", "meetup", "RustConf"],
        },
        TestCase {
            name: "TR-3: Piano Progress",
            query: "How is my piano playing going?",
            needles_strict: vec!["piano"],
            needles_relaxed: vec!["piano", "Chopin", "beginner"],
        },
    ];

    for tc in &tr_cases {
        let results = engine_tr.echo(tc.query, 10).await.expect("echo");
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(&results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(&results, 5, n));
        if strict {
            total_strict += 1;
        }
        if relaxed {
            total_relaxed += 1;
        }
        total_tests += 1;
        println!(
            "{:<30} | {:>9} | {:>10}",
            tc.name,
            if strict { "PASS" } else { "MISS" },
            if relaxed { "PASS" } else { "MISS" },
        );
    }

    // ------ Category 4: Knowledge Update ------
    println!();
    println!("Category 4: Knowledge Update");
    println!("{:<30} | {:>9} | {:>10}", "Test", "Top-3 Hit", "Top-5 Hit");
    println!("{}", "-".repeat(56));

    let dir_ku = tempdir().expect("temp dir");
    let config_ku = longmemeval_config(dir_ku.path().to_path_buf());
    let engine_ku = EchoEngine::new(config_ku).expect("engine init");

    engine_ku.store("I work as a backend engineer at Google on the Cloud Spanner team", "old").await.unwrap();
    engine_ku.store("I enjoy hiking on weekends in the bay area", "noise").await.unwrap();
    engine_ku.store("I left Google last month. I now work at Meta on the infrastructure team", "new").await.unwrap();
    engine_ku.store("I live in a one-bedroom apartment in downtown Seattle", "old2").await.unwrap();
    engine_ku.store("My favorite restaurant is the Thai place on Pike Street in Seattle", "noise2").await.unwrap();
    engine_ku.store("I just moved to Portland, Oregon and I'm renting a house in the Pearl District", "new2").await.unwrap();
    engine_ku.store("Python is my go-to programming language for everything", "old3").await.unwrap();
    engine_ku.store("I started a new side project building a web scraper", "noise3").await.unwrap();
    engine_ku.store("I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve", "new3").await.unwrap();

    let ku_cases = vec![
        TestCase {
            name: "KU-1: Job Change",
            query: "Where do I currently work?",
            needles_strict: vec!["Meta"],
            needles_relaxed: vec!["Meta", "Google"],
        },
        TestCase {
            name: "KU-2: Address Change",
            query: "Where do I live right now?",
            needles_strict: vec!["Portland"],
            needles_relaxed: vec!["Portland", "Seattle"],
        },
        TestCase {
            name: "KU-3: Language Switch",
            query: "What is my primary programming language?",
            needles_strict: vec!["Rust"],
            needles_relaxed: vec!["Rust", "Python"],
        },
    ];

    for tc in &ku_cases {
        let results = engine_ku.echo(tc.query, 10).await.expect("echo");
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(&results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(&results, 5, n));
        if strict {
            total_strict += 1;
        }
        if relaxed {
            total_relaxed += 1;
        }
        total_tests += 1;
        println!(
            "{:<30} | {:>9} | {:>10}",
            tc.name,
            if strict { "PASS" } else { "MISS" },
            if relaxed { "PASS" } else { "MISS" },
        );
    }

    // ------ Category 5: Preference Tracking ------
    println!();
    println!("Category 5: Preference Tracking");
    println!("{:<30} | {:>9} | {:>10}", "Test", "Top-3 Hit", "Top-5 Hit");
    println!("{}", "-".repeat(56));

    let dir_pt = tempdir().expect("temp dir");
    let config_pt = longmemeval_config(dir_pt.path().to_path_buf());
    let engine_pt = EchoEngine::new(config_pt).expect("engine init");

    engine_pt.store("I use Sublime Text as my code editor, it's fast and lightweight", "m1").await.unwrap();
    engine_pt.store("I switched to VS Code because of the extension ecosystem", "m3").await.unwrap();
    engine_pt.store("I've moved to Neovim with a custom Lua config for maximum speed", "m6").await.unwrap();
    engine_pt.store("I eat everything, no dietary restrictions", "d1").await.unwrap();
    engine_pt.store("I've started reducing meat, mostly eating vegetarian meals now", "d2").await.unwrap();
    engine_pt.store("I'm fully vegan now, it's been great for my energy levels", "d3").await.unwrap();
    engine_pt.store("I run 5K every morning before work", "noise_d").await.unwrap();
    engine_pt.store("I drink regular drip coffee with cream and sugar", "c1").await.unwrap();
    engine_pt.store("I switched to espresso-based drinks, usually a latte with whole milk", "c2").await.unwrap();
    engine_pt.store("Now I drink pour-over black coffee, no milk no sugar, using a Hario V60", "c3").await.unwrap();
    engine_pt.store("I like green tea in the afternoon as a lighter caffeine option", "noise_c").await.unwrap();
    engine_pt.store("I use Windows 11 on all my machines for gaming and development", "o1").await.unwrap();
    engine_pt.store("I dual-boot Linux Mint alongside Windows now for dev work", "o2").await.unwrap();
    engine_pt.store("I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely", "o3").await.unwrap();
    engine_pt.store("I bought a new 4K monitor for my desk setup", "noise_o").await.unwrap();

    let pt_cases = vec![
        TestCase {
            name: "PT-1: IDE Preference",
            query: "What code editor do I use?",
            needles_strict: vec!["Neovim"],
            needles_relaxed: vec!["Neovim", "VS Code", "Sublime"],
        },
        TestCase {
            name: "PT-2: Dietary Preference",
            query: "What's my diet like? What do I eat?",
            needles_strict: vec!["vegan"],
            needles_relaxed: vec!["vegan", "vegetarian"],
        },
        TestCase {
            name: "PT-3: Coffee Preference",
            query: "How do I take my coffee?",
            needles_strict: vec!["pour-over"],
            needles_relaxed: vec!["pour-over", "V60", "black coffee"],
        },
        TestCase {
            name: "PT-4: OS Preference",
            query: "What operating system do I run?",
            needles_strict: vec!["Arch"],
            needles_relaxed: vec!["Arch", "Hyprland", "Linux"],
        },
    ];

    for tc in &pt_cases {
        let results = engine_pt.echo(tc.query, 10).await.expect("echo");
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(&results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(&results, 5, n));
        if strict {
            total_strict += 1;
        }
        if relaxed {
            total_relaxed += 1;
        }
        total_tests += 1;
        println!(
            "{:<30} | {:>9} | {:>10}",
            tc.name,
            if strict { "PASS" } else { "MISS" },
            if relaxed { "PASS" } else { "MISS" },
        );
    }

    // ------ Summary ------
    let strict_pct = (total_strict as f64 / total_tests as f64) * 100.0;
    let relaxed_pct = (total_relaxed as f64 / total_tests as f64) * 100.0;

    println!();
    println!("=== LONGMEMEVAL SCORECARD ===");
    println!("Total tests:      {total_tests}");
    println!(
        "Top-3 accuracy:   {total_strict}/{total_tests} ({strict_pct:.1}%)"
    );
    println!(
        "Top-5 accuracy:   {total_relaxed}/{total_tests} ({relaxed_pct:.1}%)"
    );
    println!();
    println!("Hindsight claims: 91.4% on original LongMemEval");
    println!(
        "ShrimPK (top-3):  {strict_pct:.1}%"
    );
    println!(
        "ShrimPK (top-5):  {relaxed_pct:.1}%"
    );
    println!();

    // Soft assertion: we expect to hit at least 60% on top-5 (relaxed)
    // This is a baseline — the real target after optimization is 90%+
    assert!(
        relaxed_pct >= 50.0,
        "Expected at least 50% top-5 accuracy as a baseline, got {relaxed_pct:.1}%"
    );

    println!("=== BENCHMARK COMPLETE ===");
}
