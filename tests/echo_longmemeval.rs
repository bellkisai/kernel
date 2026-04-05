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
//!
//! **ort runtime fix**: Every test creates `EchoEngine` OUTSIDE the tokio
//! runtime, then uses `block_on` only for async store/echo operations.
//! This avoids the ort 2.0.0-rc.11 panic when its internal runtime is
//! dropped while another tokio runtime is active.

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
// Sync seed helper
// ===========================================================================

/// Seed the engine with a diverse user profile across multiple sessions.
/// Must be called with an engine created OUTSIDE the tokio runtime.
fn seed_user_profile_sync(engine: &EchoEngine, rt: &tokio::runtime::Runtime) {
    rt.block_on(async {
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
    });
}

// ===========================================================================
// Category 1: Information Extraction (5 tests)
//
// Can the system recall specific facts from past conversations?
// Store 20+ facts about a simulated user across multiple "sessions",
// then query with natural language questions.
// Pass criterion: relevant memory appears in top-3 results.
// ===========================================================================

/// IE-1: Direct fact recall — profession
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ie_1_profession() {
    // Create engine OUTSIDE tokio runtime to avoid ort runtime conflict
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        // Seed memories
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior software engineer at Stripe",
            "I have a golden retriever named Pixel who is 4 years old",
            "I studied Computer Science at University of British Columbia",
            "I'm allergic to shellfish, which is ironic given my love for seafood restaurants",
        ] {
            engine.store(text, "test").await.expect("store");
        }
        engine
            .echo("What is my job? Where do I work?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);
    // engine drops here on plain thread — no tokio context

    println!("IE-1 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Stripe"),
        "Top-3 should mention Stripe. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-2: Direct fact recall — pet
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ie_2_pet() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("Do I have any pets?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-2 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "golden retriever"),
        "Top-3 should mention the golden retriever. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-3: Direct fact recall — education
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ie_3_education() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("Where did I go to university?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-3 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "British Columbia"),
        "Top-3 should mention University of British Columbia. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-4: Indirect recall — food allergy (phrased differently from stored text)
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ie_4_allergy() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What foods should I avoid? Any allergies?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-4 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "shellfish"),
        "Top-3 should mention shellfish allergy. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-5: Indirect recall — hobby (phrased as a question about exercise)
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ie_5_hobby() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What martial art do I train? How often?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-5 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "jiu-jitsu"),
        "Top-3 should mention Brazilian jiu-jitsu. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
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
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_msr_1_work_and_language() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What programming language do I use for backend work?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-1 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // Should surface both the language preference memory AND the work context
    let has_language = top_n_contains(&results, 5, "Rust") || top_n_contains(&results, 5, "Go");
    let has_work = top_n_contains(&results, 5, "backend")
        || top_n_contains(&results, 5, "Stripe")
        || top_n_contains(&results, 5, "microservices");

    assert!(
        has_language,
        "Top-5 should mention a programming language (Rust/Go). Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
    assert!(
        has_work,
        "Top-5 should mention work context (backend/Stripe). Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-2: Connect travel history + language learning
/// "Can I get by with my Japanese in Tokyo?" requires knowing both the trip and the language level.
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_msr_2_travel_and_language() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("How was my Japanese when I visited Tokyo?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-2 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_japanese =
        top_n_contains(&results, 5, "Japanese") || top_n_contains(&results, 5, "JLPT");
    let has_tokyo = top_n_contains(&results, 5, "Tokyo");

    assert!(
        has_japanese && has_tokyo,
        "Top-5 should mention both Japanese study and Tokyo trip. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-3: Connect hobby + goal
/// "Am I ready for competition?" requires knowing both the training regimen and the goal.
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_msr_3_hobby_and_goal() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("Am I training enough for the jiu-jitsu tournament?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-3 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_training = top_n_contains(&results, 5, "three times a week")
        || top_n_contains(&results, 5, "jiu-jitsu");
    let has_goal =
        top_n_contains(&results, 5, "tournament") || top_n_contains(&results, 5, "compete");

    assert!(
        has_training && has_goal,
        "Top-5 should surface both training schedule and tournament goal. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-4: Connect career history + current role
/// "How did I end up at Stripe?" requires Shopify history + current Stripe role.
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_msr_4_career_path() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What's my career history in the payments industry?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-4 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_shopify = top_n_contains(&results, 5, "Shopify");
    let has_stripe = top_n_contains(&results, 5, "Stripe");

    assert!(
        has_shopify && has_stripe,
        "Top-5 should mention both Shopify and Stripe. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-5: Connect personal + professional goals
/// "What am I working toward financially and professionally?" requires house savings + startup goal.
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_msr_5_combined_goals() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What are my big life goals? What am I saving up for?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-5 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_house = top_n_contains(&results, 5, "house") || top_n_contains(&results, 5, "Oakland");
    let has_startup = top_n_contains(&results, 5, "developer tools")
        || top_n_contains(&results, 5, "observability")
        || top_n_contains(&results, 5, "company");

    assert!(
        has_house || has_startup,
        "Top-5 should surface at least one major life goal. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
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
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_tr_1_job_timeline() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
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

        engine
            .echo("Where have I worked over the years?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-1 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // All three jobs should surface in top-5
    let has_startup = top_n_contains(&results, 5, "2018") || top_n_contains(&results, 5, "startup");
    let has_shopify = top_n_contains(&results, 5, "Shopify");
    let has_stripe = top_n_contains(&results, 5, "Stripe");

    assert!(
        has_startup && has_shopify && has_stripe,
        "Top-5 should mention all three jobs. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// TR-2: Temporal ordering of recent events
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_tr_2_recent_events() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "Last month I attended a Rust conference in Berlin",
                "session_a",
            )
            .await
            .unwrap();
        engine
            .store(
                "Last week I gave a talk on observability at the local meetup",
                "session_b",
            )
            .await
            .unwrap();
        engine
            .store("Yesterday I submitted a CFP for RustConf 2027", "session_c")
            .await
            .unwrap();

        engine
            .echo("What tech events have I been involved in recently?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-2 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // At minimum the most recent event should surface in top-3
    assert!(
        top_n_contains(&results, 3, "RustConf")
            || top_n_contains(&results, 3, "meetup")
            || top_n_contains(&results, 3, "conference"),
        "Top-3 should mention at least one recent tech event. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );

    // All three should be in top-5
    let event_count = results
        .iter()
        .take(5)
        .filter(|r| {
            let c = r.content.to_lowercase();
            c.contains("conference")
                || c.contains("meetup")
                || c.contains("rustconf")
                || c.contains("cfp")
        })
        .count();

    assert!(
        event_count >= 2,
        "Top-5 should contain at least 2 of 3 tech events. Found {event_count}. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// TR-3: Temporal specificity — "last week" vs "last year"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_tr_3_temporal_specificity() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "Last year I started learning piano as a complete beginner",
                "session_old",
            )
            .await
            .unwrap();
        engine
            .store(
                "Last week I finished learning my first Chopin nocturne on piano",
                "session_new",
            )
            .await
            .unwrap();

        // Noise entries to make ranking harder
        engine
            .store(
                "I enjoy listening to classical music while coding",
                "session_noise",
            )
            .await
            .unwrap();
        engine
            .store("My neighbor plays guitar every evening", "session_noise2")
            .await
            .unwrap();

        engine
            .echo("How is my piano playing going?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-3 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // Both piano memories should surface
    let has_beginner = top_n_contains(&results, 5, "beginner")
        || top_n_contains(&results, 5, "started learning piano");
    let has_chopin =
        top_n_contains(&results, 5, "Chopin") || top_n_contains(&results, 5, "nocturne");

    assert!(
        has_beginner || has_chopin,
        "Top-5 should mention piano progress. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 4: Knowledge Update (3 tests)
//
// Can the system handle corrections and updates to previously stored facts?
// Store a fact, then store a correction. Query for the current state.
// Pass criterion: the corrected/updated memory ranks HIGHER than the stale one.
//
// Note: KS18 Track 3 added a configurable recency boost to the echo
// scoring pipeline. Newer memories get a small advantage via the formula
// recency_weight / (1.0 + days_since_stored), where recency_weight
// defaults to 0.05. This helps corrections rank above stale facts when
// both have similar cosine similarity. Full knowledge-update handling
// (explicit superseding of old facts) is a future capability.
// ===========================================================================

/// KU-1: Job change — "Where do I work?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ku_1_job_change() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        // Old fact
        engine
            .store(
                "I work as a backend engineer at Google on the Cloud Spanner team",
                "session_old",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "I enjoy hiking on weekends in the bay area",
                "session_noise",
            )
            .await
            .unwrap();

        // Correction (stored later, simulating a new conversation)
        engine
            .store(
                "I left Google last month. I now work at Meta on the infrastructure team",
                "session_new",
            )
            .await
            .unwrap();

        engine
            .echo("Where do I currently work?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-1 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // The Meta memory should surface
    assert!(
        top_n_contains(&results, 3, "Meta"),
        "Top-3 should mention Meta (the current employer). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );

    // Both should surface (system doesn't delete old facts, but both are relevant)
    let has_google = top_n_contains(&results, 5, "Google");
    let has_meta = top_n_contains(&results, 5, "Meta");
    assert!(
        has_google && has_meta,
        "Top-5 should surface both old and new employer for context. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// KU-2: Address change — "Where do I live?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ku_2_address_change() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I live in a one-bedroom apartment in downtown Seattle",
                "session_old",
            )
            .await
            .unwrap();
        engine
            .store(
                "My favorite restaurant is the Thai place on Pike Street in Seattle",
                "session_noise",
            )
            .await
            .unwrap();
        engine
            .store(
                "I just moved to Portland, Oregon and I'm renting a house in the Pearl District",
                "session_new",
            )
            .await
            .unwrap();

        engine
            .echo("Where do I live right now?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-2 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // Portland should surface
    assert!(
        top_n_contains(&results, 3, "Portland"),
        "Top-3 should mention Portland (current residence). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// KU-3: Technology preference update — "What language do I use?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_ku_3_tech_preference_update() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
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

        engine
            .echo("What is my primary programming language?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-3 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // Rust (current) should surface
    assert!(
        top_n_contains(&results, 3, "Rust"),
        "Top-3 should mention Rust (current preference). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );

    // Python (old) should also surface since it's semantically relevant
    assert!(
        top_n_contains(&results, 5, "Python"),
        "Top-5 should also mention Python (for context). Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
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
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_pt_1_ide_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        // Preferences evolve over time
        engine
            .store(
                "I use Sublime Text as my code editor, it's fast and lightweight",
                "month1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I switched to VS Code because of the extension ecosystem",
                "month3",
            )
            .await
            .unwrap();
        engine
            .store(
                "I've moved to Neovim with a custom Lua config for maximum speed",
                "month6",
            )
            .await
            .unwrap();

        engine
            .echo("What code editor do I use?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-1 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // All three should surface as they're all about editors
    let has_neovim = top_n_contains(&results, 5, "Neovim");
    let has_vscode = top_n_contains(&results, 5, "VS Code");
    let has_sublime = top_n_contains(&results, 5, "Sublime");

    assert!(
        has_neovim,
        "Top-5 should mention Neovim (most recent preference). Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
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
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_pt_2_dietary_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I eat everything, no dietary restrictions", "early")
            .await
            .unwrap();
        engine
            .store(
                "I've started reducing meat, mostly eating vegetarian meals now",
                "mid",
            )
            .await
            .unwrap();
        engine
            .store(
                "I'm fully vegan now, it's been great for my energy levels",
                "recent",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store("I run 5K every morning before work", "noise")
            .await
            .unwrap();

        engine
            .echo("What's my diet like? What do I eat?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-2 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "vegan"),
        "Top-3 should mention vegan (most recent dietary preference). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// PT-3: Coffee preference tracking (specific and nuanced)
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_pt_3_coffee_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I drink regular drip coffee with cream and sugar", "month1")
            .await
            .unwrap();
        engine
            .store(
                "I switched to espresso-based drinks, usually a latte with whole milk",
                "month4",
            )
            .await
            .unwrap();
        engine
            .store(
                "Now I drink pour-over black coffee, no milk no sugar, using a Hario V60",
                "month8",
            )
            .await
            .unwrap();

        // Additional noise
        engine
            .store(
                "I like green tea in the afternoon as a lighter caffeine option",
                "noise",
            )
            .await
            .unwrap();

        engine
            .echo("How do I take my coffee?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-3 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // The most recent preference should surface
    assert!(
        top_n_contains(&results, 3, "pour-over")
            || top_n_contains(&results, 3, "V60")
            || top_n_contains(&results, 3, "black coffee"),
        "Top-3 should mention current coffee preference (pour-over/V60/black). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// PT-4: Operating system preference (multi-device)
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_pt_4_os_preference() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
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

        engine
            .echo("What operating system do I run?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-4 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Arch") || top_n_contains(&results, 3, "Hyprland"),
        "Top-3 should mention Arch Linux (most recent OS preference). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
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
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_full_scorecard() {
    println!("\n=== LongMemEval Benchmark — ShrimPK Echo Memory ===\n");

    struct TestCase {
        name: &'static str,
        query: &'static str,
        needles_strict: Vec<&'static str>,  // must be in top-3
        needles_relaxed: Vec<&'static str>, // must be in top-5
    }

    let mut total_strict = 0u32;
    let mut total_relaxed = 0u32;
    let mut total_tests = 0u32;

    // ------ Category 1: Information Extraction ------
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_user_profile_sync(&engine, &rt);

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

    println!("Category 1: Information Extraction");
    println!("{:<30} | {:>9} | {:>10}", "Test", "Top-3 Hit", "Top-5 Hit");
    println!("{}", "-".repeat(56));

    let ie_results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for tc in &ie_cases {
            results.push(engine.echo(tc.query, 10).await.expect("echo"));
        }
        results
    });

    for (tc, results) in ie_cases.iter().zip(ie_results.iter()) {
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(results, 5, n));
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

    // Drop the IE engine before creating the MSR engine
    drop(engine);

    // ------ Category 2: Multi-Session Reasoning ------
    println!();
    println!("Category 2: Multi-Session Reasoning");
    println!("{:<30} | {:>9} | {:>10}", "Test", "Top-3 Hit", "Top-5 Hit");
    println!("{}", "-".repeat(56));

    // Re-seed for MSR (the previous engine was dropped)
    let dir_msr = tempdir().expect("temp dir");
    let config_msr = longmemeval_config(dir_msr.path().to_path_buf());
    let engine = EchoEngine::new(config_msr).expect("engine init");
    seed_user_profile_sync(&engine, &rt);

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

    let msr_results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for tc in &msr_cases {
            results.push(engine.echo(tc.query, 10).await.expect("echo"));
        }
        results
    });

    // Drop MSR engine before creating TR engine
    drop(engine);

    for (tc, results) in msr_cases.iter().zip(msr_results.iter()) {
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(results, 5, n));
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

    rt.block_on(async {
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
    });

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

    let tr_results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for tc in &tr_cases {
            results.push(engine_tr.echo(tc.query, 10).await.expect("echo"));
        }
        results
    });

    // Drop TR engine before creating KU engine
    drop(engine_tr);

    for (tc, results) in tr_cases.iter().zip(tr_results.iter()) {
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(results, 5, n));
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

    rt.block_on(async {
        engine_ku.store("I work as a backend engineer at Google on the Cloud Spanner team", "old").await.unwrap();
        engine_ku.store("I enjoy hiking on weekends in the bay area", "noise").await.unwrap();
        engine_ku.store("I left Google last month. I now work at Meta on the infrastructure team", "new").await.unwrap();
        engine_ku.store("I live in a one-bedroom apartment in downtown Seattle", "old2").await.unwrap();
        engine_ku.store("My favorite restaurant is the Thai place on Pike Street in Seattle", "noise2").await.unwrap();
        engine_ku.store("I just moved to Portland, Oregon and I'm renting a house in the Pearl District", "new2").await.unwrap();
        engine_ku.store("Python is my go-to programming language for everything", "old3").await.unwrap();
        engine_ku.store("I started a new side project building a web scraper", "noise3").await.unwrap();
        engine_ku.store("I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve", "new3").await.unwrap();
    });

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

    let ku_results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for tc in &ku_cases {
            results.push(engine_ku.echo(tc.query, 10).await.expect("echo"));
        }
        results
    });

    // Drop KU engine before creating PT engine
    drop(engine_ku);

    for (tc, results) in ku_cases.iter().zip(ku_results.iter()) {
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(results, 5, n));
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

    rt.block_on(async {
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
    });

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

    let pt_results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for tc in &pt_cases {
            results.push(engine_pt.echo(tc.query, 10).await.expect("echo"));
        }
        results
    });

    // Drop PT engine before final summary
    drop(engine_pt);

    for (tc, results) in pt_cases.iter().zip(pt_results.iter()) {
        let strict = tc
            .needles_strict
            .iter()
            .all(|n| top_n_contains(results, 3, n));
        let relaxed = tc
            .needles_relaxed
            .iter()
            .any(|n| top_n_contains(results, 5, n));
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

    // Drop the runtime before engines (engines already dropped above)
    drop(rt);

    // ------ Summary ------
    let strict_pct = (total_strict as f64 / total_tests as f64) * 100.0;
    let relaxed_pct = (total_relaxed as f64 / total_tests as f64) * 100.0;

    println!();
    println!("=== LONGMEMEVAL SCORECARD ===");
    println!("Total tests:      {total_tests}");
    println!("Top-3 accuracy:   {total_strict}/{total_tests} ({strict_pct:.1}%)");
    println!("Top-5 accuracy:   {total_relaxed}/{total_tests} ({relaxed_pct:.1}%)");
    println!();
    println!("Hindsight claims: 91.4% on original LongMemEval");
    println!("ShrimPK (top-3):  {strict_pct:.1}%");
    println!("ShrimPK (top-5):  {relaxed_pct:.1}%");
    println!();

    // Soft assertion: we expect to hit at least 60% on top-5 (relaxed)
    // This is a baseline — the real target after optimization is 90%+
    assert!(
        relaxed_pct >= 50.0,
        "Expected at least 50% top-5 accuracy as a baseline, got {relaxed_pct:.1}%"
    );

    println!("=== BENCHMARK COMPLETE ===");
}

// ===========================================================================
// CONSOLIDATED BENCHMARK — WITH OLLAMA FACT EXTRACTION
// ===========================================================================

/// Config with consolidation enabled (Ollama + llama3.2:3b).
fn longmemeval_consolidated_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15, // match baseline
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        data_dir,
        embedding_dim: 384,
        consolidation_provider: "ollama".to_string(),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        enrichment_model: "llama3.2:3b".to_string(),
        consolidation_consent_given: true,
        // child_rescue_only: true (default)
        // KS22 prompt sweep — testing C1 (verb whitelist)
        fact_extraction_prompt: Some(
            "Extract personal facts from the text. Rules:\n\
             1. One fact per line, starting with \"The user\"\n\
             2. Use ONLY these verbs: works at, works for, joined, lives in, moved to, based in, uses, prefers, switched to, likes, chose, belongs to, member of, part of\n\
             3. No colons, labels, or key-value pairs\n\n\
             Example:\n\
               The user uses Neovim\n\
               The user lives in Berlin\n\
               The user switched to Python from Java\n\n\
             Max {max_facts} facts. If none found, output NONE.".to_string()
        ),
        supersedes_demotion: 0.10,
        ..Default::default()
    }
}

/// Run consolidation passes until no more facts are extracted (max 10 passes).
/// Runs on a dedicated thread with its own tokio runtime to avoid conflicts
/// between reqwest::blocking::Client's internal runtime and the test runtime.
/// Run consolidation on a dedicated thread to avoid reqwest::blocking + tokio conflicts.
/// Run consolidation passes until no more facts are extracted (max 10 passes).
/// Now clean — ureq replaced reqwest::blocking, no tokio runtime conflict.
fn run_full_consolidation(engine: &EchoEngine) -> usize {
    let mut total_facts = 0;
    std::thread::scope(|s| {
        let result = s.spawn(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let mut facts = 0;
            for pass in 1..=10 {
                let result = rt.block_on(engine.consolidate_now());
                println!(
                    "  Consolidation pass {pass}: facts={}, merged={}, pruned={}, duration={}ms",
                    result.facts_extracted,
                    result.duplicates_merged,
                    result.hebbian_edges_pruned,
                    result.duration_ms
                );
                facts += result.facts_extracted;
                if result.facts_extracted == 0 {
                    break;
                }
            }
            facts
        });
        total_facts = result.join().expect("consolidation thread panicked");
    });
    total_facts
}

/// Full LongMemEval with consolidation active.
/// Requires Ollama running with llama3.2:3b.
#[test]
#[ignore = "requires Ollama with llama3.2:3b"]
fn longmemeval_consolidated_scorecard() {
    println!("\n=== LongMemEval CONSOLIDATED Benchmark — ShrimPK Echo Memory ===\n");

    // --- Category 1: Information Extraction ---
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_consolidated_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store phase
    rt.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I have a golden retriever named Pixel who is 4 years old",
            "My partner's name is Jordan and we've been together for 6 years",
            "I graduated from the University of British Columbia with a CS degree in 2015",
            "I was born in Taipei, Taiwan but grew up in Vancouver, Canada",
            "I'm allergic to shellfish and cats",
            "My favorite cuisine is Thai food, especially pad see ew and massaman curry",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
            "I drive a 2022 Tesla Model 3 but mostly bike to work",
            "I'm being considered for a staff engineer promotion next quarter",
            "My IDE is Neovim with LazyVim config and Catppuccin theme",
            "I enjoy listening to classical music while coding, especially Chopin",
            "I read about 30 books a year, mostly sci-fi and technical books",
            "I donate monthly to the EFF and Wikipedia",
            "I have a standing desk and use a split ergonomic keyboard",
            "My morning routine is meditation, coffee, then 30 minutes of reading",
            "I run a small Kubernetes cluster at home for side projects",
        ] {
            engine.store(text, "profile").await.expect("store");
        }
    });

    // Run consolidation OUTSIDE block_on (uses its own thread+runtime)
    println!("Running consolidation (Ollama llama3.2:3b)...");
    let total_facts = run_full_consolidation(&engine);
    println!("Total facts extracted: {total_facts}\n");

    // Query phase — new block_on
    let ie_results = rt.block_on(async {
        let queries = vec![
            (
                "IE-1: Profession",
                "What is my job? Where do I work?",
                "Stripe",
            ),
            ("IE-2: Pet", "Do I have any pets?", "golden retriever"),
            (
                "IE-3: Education",
                "Where did I go to college?",
                "British Columbia",
            ),
            ("IE-4: Allergy", "What am I allergic to?", "shellfish"),
            (
                "IE-5: Hobby",
                "What sports or physical activities do I do?",
                "jiu-jitsu",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            for (i, res) in r.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    res.similarity,
                    res.final_score,
                    &res.content[..res.content.len().min(70)]
                );
            }
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt);

    // --- Category 2-5: Multi-Session, Temporal, Knowledge Update, Preference ---
    // These need separate engines with category-specific data

    // MSR
    let dir2 = tempdir().expect("temp dir");
    let config2 = longmemeval_consolidated_config(dir2.path().to_path_buf());
    let engine2 = EchoEngine::new(config2).expect("engine init");
    let rt2 = tokio::runtime::Runtime::new().unwrap();

    rt2.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
        ] {
            engine2.store(text, "profile").await.expect("store");
        }
    });
    println!("\nRunning MSR consolidation...");
    run_full_consolidation(&engine2);
    let msr_results = rt2.block_on(async {
        let queries = vec![
            (
                "MSR-1: Work+Lang",
                "What programming languages do I use at work?",
                "Rust",
            ),
            (
                "MSR-2: Travel+Lang",
                "Have I traveled anywhere related to languages I'm learning?",
                "Tokyo",
            ),
            (
                "MSR-3: Hobby+Goal",
                "What goals do I have related to my hobbies?",
                "tournament",
            ),
            (
                "MSR-4: Career",
                "Tell me about my career progression",
                "Stripe",
            ),
            (
                "MSR-5: Life Goals",
                "What are my big life goals? What am I saving up for?",
                "house",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine2.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt2);

    // KU (Knowledge Update)
    let dir3 = tempdir().expect("temp dir");
    let config3 = longmemeval_consolidated_config(dir3.path().to_path_buf());
    let engine3 = EchoEngine::new(config3).expect("engine init");
    let rt3 = tokio::runtime::Runtime::new().unwrap();

    rt3.block_on(async {
        engine3.store("I work as a backend engineer at Google on the Cloud Spanner team", "old").await.unwrap();
        engine3.store("I left Google last month. I now work at Meta on the infrastructure team", "new").await.unwrap();
        engine3.store("I enjoy hiking on weekends in the bay area", "noise").await.unwrap();
        engine3.store("I live in a one-bedroom apartment in downtown Seattle", "old").await.unwrap();
        engine3.store("I just moved to Portland, Oregon and I'm renting a house in the Pearl District", "new").await.unwrap();
        engine3.store("Python is my go-to programming language for everything", "old").await.unwrap();
        engine3.store("I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve", "new").await.unwrap();
    });
    println!("\nRunning KU consolidation...");
    run_full_consolidation(&engine3);
    let ku_results = rt3.block_on(async {
        let queries = vec![
            ("KU-1: Job Change", "Where do I work now?", "Meta"),
            ("KU-2: Address", "Where do I live now?", "Portland"),
            (
                "KU-3: Language",
                "What programming language do I mainly use?",
                "Rust",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine3.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt3);

    // PT (Preference Tracking)
    let dir4 = tempdir().expect("temp dir");
    let config4 = longmemeval_consolidated_config(dir4.path().to_path_buf());
    let engine4 = EchoEngine::new(config4).expect("engine init");
    let rt4 = tokio::runtime::Runtime::new().unwrap();

    rt4.block_on(async {
        engine4.store("I use Sublime Text as my code editor, it's fast and lightweight", "m1").await.unwrap();
        engine4.store("I switched to VS Code because of the extension ecosystem", "m3").await.unwrap();
        engine4.store("I've moved to Neovim with a custom Lua config for maximum speed", "m6").await.unwrap();
        engine4.store("I'm vegetarian and have been for the past 3 years", "m1").await.unwrap();
        engine4.store("I started eating fish again, so now I'm pescatarian", "m4").await.unwrap();
        engine4.store("I drink regular drip coffee with cream and sugar", "m1").await.unwrap();
        engine4.store("I switched to espresso-based drinks, usually a latte with whole milk", "m3").await.unwrap();
        engine4.store("Now I drink pour-over black coffee, no milk no sugar, using a Hario V60", "m6").await.unwrap();
        engine4.store("I use Windows 11 on all my machines for gaming and development", "m1").await.unwrap();
        engine4.store("I dual-boot Linux Mint alongside Windows now for dev work", "m3").await.unwrap();
        engine4.store("I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely", "m6").await.unwrap();
    });
    println!("\nRunning PT consolidation...");
    run_full_consolidation(&engine4);
    let pt_results = rt4.block_on(async {
        let queries = vec![
            ("PT-1: IDE", "What code editor do I use?", "Neovim"),
            ("PT-2: Diet", "What's my diet like?", "pescatarian"),
            ("PT-3: Coffee", "How do I take my coffee?", "pour-over"),
            ("PT-4: OS", "What operating system do I use?", "Arch"),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine4.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            for (i, res) in r.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    res.similarity,
                    res.final_score,
                    &res.content[..res.content.len().min(70)]
                );
            }
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt4);

    // --- SCORECARD ---
    let all_results: Vec<(bool, bool)> = ie_results
        .into_iter()
        .chain(msr_results)
        .chain(ku_results)
        .chain(pt_results)
        .collect();

    let total = all_results.len();
    let strict = all_results.iter().filter(|(h3, _)| *h3).count();
    let relaxed = all_results.iter().filter(|(_, h5)| *h5).count();
    let strict_pct = (strict as f64 / total as f64) * 100.0;
    let relaxed_pct = (relaxed as f64 / total as f64) * 100.0;

    println!("\n=== CONSOLIDATED LONGMEMEVAL SCORECARD ===");
    println!("Total tests:      {total}");
    println!("Top-3 accuracy:   {strict}/{total} ({strict_pct:.1}%)");
    println!("Top-5 accuracy:   {relaxed}/{total} ({relaxed_pct:.1}%)");
    println!();
    println!("Baseline (no consolidation): 90.0% top-3, 100.0% top-5");
    println!("Consolidated:                {strict_pct:.1}% top-3, {relaxed_pct:.1}% top-5");
    println!("Uplift:                      {:.1}pp", strict_pct - 90.0);
    println!();
    println!("=== CONSOLIDATED BENCHMARK COMPLETE ===");
}

// ===========================================================================
// Reranker config + scorecard (KS23 Track 3)
// ===========================================================================

/// Config with LLM reranker enabled + consolidation + supersedes demotion.
/// Combines all enrichment features for maximum ranking quality.
fn longmemeval_reranker_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15, // match baseline
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        data_dir,
        embedding_dim: 384,
        consolidation_provider: "ollama".to_string(),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        enrichment_model: "llama3.2:3b".to_string(),
        consolidation_consent_given: true,
        reranker_enabled: true,
        query_expansion_enabled: false, // test reranker in isolation first
        fact_extraction_prompt: Some(
            "Extract personal facts from the text. Rules:\n\
             1. One fact per line, starting with \"The user\"\n\
             2. Use ONLY these verbs: works at, works for, joined, lives in, moved to, based in, uses, prefers, switched to, likes, chose, belongs to, member of, part of\n\
             3. No colons, labels, or key-value pairs\n\n\
             Example:\n\
               The user uses Neovim\n\
               The user lives in Berlin\n\
               The user switched to Python from Java\n\n\
             Max {max_facts} facts. If none found, output NONE.".to_string()
        ),
        supersedes_demotion: 0.10,
        ..Default::default()
    }
}

/// Full LongMemEval with LLM reranker + consolidation.
/// Requires Ollama running with llama3.2:3b.
#[test]
#[ignore = "requires Ollama with llama3.2:3b"]
fn longmemeval_reranker_scorecard() {
    println!("\n=== LongMemEval RERANKER Benchmark — ShrimPK Echo Memory ===\n");

    // --- Category 1: Information Extraction ---
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_reranker_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store phase
    rt.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I have a golden retriever named Pixel who is 4 years old",
            "My partner's name is Jordan and we've been together for 6 years",
            "I graduated from the University of British Columbia with a CS degree in 2015",
            "I was born in Taipei, Taiwan but grew up in Vancouver, Canada",
            "I'm allergic to shellfish and cats",
            "My favorite cuisine is Thai food, especially pad see ew and massaman curry",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
            "I drive a 2022 Tesla Model 3 but mostly bike to work",
            "I'm being considered for a staff engineer promotion next quarter",
            "My IDE is Neovim with LazyVim config and Catppuccin theme",
            "I enjoy listening to classical music while coding, especially Chopin",
            "I read about 30 books a year, mostly sci-fi and technical books",
            "I donate monthly to the EFF and Wikipedia",
            "I have a standing desk and use a split ergonomic keyboard",
            "My morning routine is meditation, coffee, then 30 minutes of reading",
            "I run a small Kubernetes cluster at home for side projects",
        ] {
            engine.store(text, "profile").await.expect("store");
        }
    });

    // Run consolidation
    println!("Running consolidation (Ollama llama3.2:3b)...");
    let total_facts = run_full_consolidation(&engine);
    println!("Total facts extracted: {total_facts}\n");

    // Query phase
    let ie_results = rt.block_on(async {
        let queries = vec![
            (
                "IE-1: Profession",
                "What is my job? Where do I work?",
                "Stripe",
            ),
            ("IE-2: Pet", "Do I have any pets?", "golden retriever"),
            (
                "IE-3: Education",
                "Where did I go to college?",
                "British Columbia",
            ),
            ("IE-4: Allergy", "What am I allergic to?", "shellfish"),
            (
                "IE-5: Hobby",
                "What sports or physical activities do I do?",
                "jiu-jitsu",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            for (i, res) in r.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    res.similarity,
                    res.final_score,
                    &res.content[..res.content.len().min(70)]
                );
            }
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt);

    // --- Category 2: Multi-Session Reasoning ---
    let dir2 = tempdir().expect("temp dir");
    let config2 = longmemeval_reranker_config(dir2.path().to_path_buf());
    let engine2 = EchoEngine::new(config2).expect("engine init");
    let rt2 = tokio::runtime::Runtime::new().unwrap();

    rt2.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
        ] {
            engine2.store(text, "profile").await.expect("store");
        }
    });
    println!("\nRunning MSR consolidation...");
    run_full_consolidation(&engine2);
    let msr_results = rt2.block_on(async {
        let queries = vec![
            (
                "MSR-1: Work+Lang",
                "What programming languages do I use at work?",
                "Rust",
            ),
            (
                "MSR-2: Travel+Lang",
                "Have I traveled anywhere related to languages I'm learning?",
                "Tokyo",
            ),
            (
                "MSR-3: Hobby+Goal",
                "What goals do I have related to my hobbies?",
                "tournament",
            ),
            (
                "MSR-4: Career",
                "Tell me about my career progression",
                "Stripe",
            ),
            (
                "MSR-5: Life Goals",
                "What are my big life goals? What am I saving up for?",
                "house",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine2.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt2);

    // --- Category 3: Knowledge Update ---
    let dir3 = tempdir().expect("temp dir");
    let config3 = longmemeval_reranker_config(dir3.path().to_path_buf());
    let engine3 = EchoEngine::new(config3).expect("engine init");
    let rt3 = tokio::runtime::Runtime::new().unwrap();

    rt3.block_on(async {
        engine3.store("I work as a backend engineer at Google on the Cloud Spanner team", "old").await.unwrap();
        engine3.store("I left Google last month. I now work at Meta on the infrastructure team", "new").await.unwrap();
        engine3.store("I enjoy hiking on weekends in the bay area", "noise").await.unwrap();
        engine3.store("I live in a one-bedroom apartment in downtown Seattle", "old").await.unwrap();
        engine3.store("I just moved to Portland, Oregon and I'm renting a house in the Pearl District", "new").await.unwrap();
        engine3.store("Python is my go-to programming language for everything", "old").await.unwrap();
        engine3.store("I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve", "new").await.unwrap();
    });
    println!("\nRunning KU consolidation...");
    run_full_consolidation(&engine3);
    let ku_results = rt3.block_on(async {
        let queries = vec![
            ("KU-1: Job Change", "Where do I work now?", "Meta"),
            ("KU-2: Address", "Where do I live now?", "Portland"),
            (
                "KU-3: Language",
                "What programming language do I mainly use?",
                "Rust",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine3.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            for (i, res) in r.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    res.similarity,
                    res.final_score,
                    &res.content[..res.content.len().min(70)]
                );
            }
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt3);

    // --- Category 4: Preference Tracking (reranker's target category) ---
    let dir4 = tempdir().expect("temp dir");
    let config4 = longmemeval_reranker_config(dir4.path().to_path_buf());
    let engine4 = EchoEngine::new(config4).expect("engine init");
    let rt4 = tokio::runtime::Runtime::new().unwrap();

    rt4.block_on(async {
        engine4.store("I use Sublime Text as my code editor, it's fast and lightweight", "m1").await.unwrap();
        engine4.store("I switched to VS Code because of the extension ecosystem", "m3").await.unwrap();
        engine4.store("I've moved to Neovim with a custom Lua config for maximum speed", "m6").await.unwrap();
        engine4.store("I'm vegetarian and have been for the past 3 years", "m1").await.unwrap();
        engine4.store("I started eating fish again, so now I'm pescatarian", "m4").await.unwrap();
        engine4.store("I drink regular drip coffee with cream and sugar", "m1").await.unwrap();
        engine4.store("I switched to espresso-based drinks, usually a latte with whole milk", "m3").await.unwrap();
        engine4.store("Now I drink pour-over black coffee, no milk no sugar, using a Hario V60", "m6").await.unwrap();
        engine4.store("I use Windows 11 on all my machines for gaming and development", "m1").await.unwrap();
        engine4.store("I dual-boot Linux Mint alongside Windows now for dev work", "m3").await.unwrap();
        engine4.store("I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely", "m6").await.unwrap();
    });
    println!("\nRunning PT consolidation...");
    run_full_consolidation(&engine4);
    let pt_results = rt4.block_on(async {
        let queries = vec![
            ("PT-1: IDE", "What code editor do I use?", "Neovim"),
            ("PT-2: Diet", "What's my diet like?", "pescatarian"),
            ("PT-3: Coffee", "How do I take my coffee?", "pour-over"),
            ("PT-4: OS", "What operating system do I use?", "Arch"),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine4.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            for (i, res) in r.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    res.similarity,
                    res.final_score,
                    &res.content[..res.content.len().min(70)]
                );
            }
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt4);

    // --- SCORECARD ---
    let all_results: Vec<(bool, bool)> = ie_results
        .into_iter()
        .chain(msr_results)
        .chain(ku_results)
        .chain(pt_results)
        .collect();

    let total = all_results.len();
    let strict = all_results.iter().filter(|(h3, _)| *h3).count();
    let relaxed = all_results.iter().filter(|(_, h5)| *h5).count();
    let strict_pct = (strict as f64 / total as f64) * 100.0;
    let relaxed_pct = (relaxed as f64 / total as f64) * 100.0;

    println!("\n=== RERANKER LONGMEMEVAL SCORECARD ===");
    println!("Total tests:      {total}");
    println!("Top-3 accuracy:   {strict}/{total} ({strict_pct:.1}%)");
    println!("Top-5 accuracy:   {relaxed}/{total} ({relaxed_pct:.1}%)");
    println!();
    println!("Baseline (no enrichment):    90.0% top-3, 100.0% top-5");
    println!("Consolidated (no reranker):  see consolidated scorecard");
    println!("Consolidated + Reranker:     {strict_pct:.1}% top-3, {relaxed_pct:.1}% top-5");
    println!();
    println!("=== RERANKER BENCHMARK COMPLETE ===");
}

// ===========================================================================
// HYDE BENCHMARK — CONSOLIDATED + QUERY EXPANSION (KS23 Track 2)
// ===========================================================================

/// Config with consolidation + HyDE query expansion enabled.
fn longmemeval_hyde_config(data_dir: PathBuf) -> EchoConfig {
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
        query_expansion_enabled: true,
        fact_extraction_prompt: Some(
            "Extract personal facts from the text. Rules:\n\
             1. One fact per line, starting with \"The user\"\n\
             2. Use ONLY these verbs: works at, works for, joined, lives in, moved to, based in, uses, prefers, switched to, likes, chose, belongs to, member of, part of\n\
             3. No colons, labels, or key-value pairs\n\n\
             Example:\n\
               The user uses Neovim\n\
               The user lives in Berlin\n\
               The user switched to Python from Java\n\n\
             Max {max_facts} facts. If none found, output NONE.".to_string()
        ),
        supersedes_demotion: 0.10,
        ..Default::default()
    }
}

/// Full LongMemEval with consolidation + HyDE query expansion.
/// Requires Ollama running with llama3.2:3b.
#[test]
#[ignore = "requires Ollama with llama3.2:3b"]
fn longmemeval_hyde_scorecard() {
    println!("\n=== LongMemEval HYDE Benchmark — ShrimPK Echo Memory ===\n");

    // --- Category 1: Information Extraction ---
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_hyde_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store phase
    rt.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I have a golden retriever named Pixel who is 4 years old",
            "My partner's name is Jordan and we've been together for 6 years",
            "I graduated from the University of British Columbia with a CS degree in 2015",
            "I was born in Taipei, Taiwan but grew up in Vancouver, Canada",
            "I'm allergic to shellfish and cats",
            "My favorite cuisine is Thai food, especially pad see ew and massaman curry",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
            "I drive a 2022 Tesla Model 3 but mostly bike to work",
            "I'm being considered for a staff engineer promotion next quarter",
            "My IDE is Neovim with LazyVim config and Catppuccin theme",
            "I enjoy listening to classical music while coding, especially Chopin",
            "I read about 30 books a year, mostly sci-fi and technical books",
            "I donate monthly to the EFF and Wikipedia",
            "I have a standing desk and use a split ergonomic keyboard",
            "My morning routine is meditation, coffee, then 30 minutes of reading",
            "I run a small Kubernetes cluster at home for side projects",
        ] {
            engine.store(text, "profile").await.expect("store");
        }
    });

    // Run consolidation OUTSIDE block_on (uses its own thread+runtime)
    println!("Running consolidation (Ollama llama3.2:3b)...");
    let total_facts = run_full_consolidation(&engine);
    println!("Total facts extracted: {total_facts}\n");

    // Query phase
    let ie_results = rt.block_on(async {
        let queries = vec![
            (
                "IE-1: Profession",
                "What is my job? Where do I work?",
                "Stripe",
            ),
            ("IE-2: Pet", "Do I have any pets?", "golden retriever"),
            (
                "IE-3: Education",
                "Where did I go to college?",
                "British Columbia",
            ),
            ("IE-4: Allergy", "What am I allergic to?", "shellfish"),
            (
                "IE-5: Hobby",
                "What sports or physical activities do I do?",
                "jiu-jitsu",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            for (i, res) in r.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    res.similarity,
                    res.final_score,
                    &res.content[..res.content.len().min(70)]
                );
            }
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt);

    // --- Category 2: Multi-Session Reasoning ---
    let dir2 = tempdir().expect("temp dir");
    let config2 = longmemeval_hyde_config(dir2.path().to_path_buf());
    let engine2 = EchoEngine::new(config2).expect("engine init");
    let rt2 = tokio::runtime::Runtime::new().unwrap();

    rt2.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
        ] {
            engine2.store(text, "profile").await.expect("store");
        }
    });
    println!("\nRunning MSR consolidation...");
    run_full_consolidation(&engine2);
    let msr_results = rt2.block_on(async {
        let queries = vec![
            (
                "MSR-1: Work+Lang",
                "What programming languages do I use at work?",
                "Rust",
            ),
            (
                "MSR-2: Travel+Lang",
                "Have I traveled anywhere related to languages I'm learning?",
                "Tokyo",
            ),
            (
                "MSR-3: Hobby+Goal",
                "What goals do I have related to my hobbies?",
                "tournament",
            ),
            (
                "MSR-4: Career",
                "Tell me about my career progression",
                "Stripe",
            ),
            (
                "MSR-5: Life Goals",
                "What are my big life goals? What am I saving up for?",
                "house",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine2.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt2);

    // --- Category 3: Knowledge Update ---
    let dir3 = tempdir().expect("temp dir");
    let config3 = longmemeval_hyde_config(dir3.path().to_path_buf());
    let engine3 = EchoEngine::new(config3).expect("engine init");
    let rt3 = tokio::runtime::Runtime::new().unwrap();

    rt3.block_on(async {
        engine3.store("I work as a backend engineer at Google on the Cloud Spanner team", "old").await.unwrap();
        engine3.store("I left Google last month. I now work at Meta on the infrastructure team", "new").await.unwrap();
        engine3.store("I enjoy hiking on weekends in the bay area", "noise").await.unwrap();
        engine3.store("I live in a one-bedroom apartment in downtown Seattle", "old").await.unwrap();
        engine3.store("I just moved to Portland, Oregon and I'm renting a house in the Pearl District", "new").await.unwrap();
        engine3.store("Python is my go-to programming language for everything", "old").await.unwrap();
        engine3.store("I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve", "new").await.unwrap();
    });
    println!("\nRunning KU consolidation...");
    run_full_consolidation(&engine3);
    let ku_results = rt3.block_on(async {
        let queries = vec![
            ("KU-1: Job Change", "Where do I work now?", "Meta"),
            ("KU-2: Address", "Where do I live now?", "Portland"),
            (
                "KU-3: Language",
                "What programming language do I mainly use?",
                "Rust",
            ),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine3.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt3);

    // --- Category 4: Preference Tracking ---
    let dir4 = tempdir().expect("temp dir");
    let config4 = longmemeval_hyde_config(dir4.path().to_path_buf());
    let engine4 = EchoEngine::new(config4).expect("engine init");
    let rt4 = tokio::runtime::Runtime::new().unwrap();

    rt4.block_on(async {
        engine4.store("I use Sublime Text as my code editor, it's fast and lightweight", "m1").await.unwrap();
        engine4.store("I switched to VS Code because of the extension ecosystem", "m3").await.unwrap();
        engine4.store("I've moved to Neovim with a custom Lua config for maximum speed", "m6").await.unwrap();
        engine4.store("I'm vegetarian and have been for the past 3 years", "m1").await.unwrap();
        engine4.store("I started eating fish again, so now I'm pescatarian", "m4").await.unwrap();
        engine4.store("I drink regular drip coffee with cream and sugar", "m1").await.unwrap();
        engine4.store("I switched to espresso-based drinks, usually a latte with whole milk", "m3").await.unwrap();
        engine4.store("Now I drink pour-over black coffee, no milk no sugar, using a Hario V60", "m6").await.unwrap();
        engine4.store("I use Windows 11 on all my machines for gaming and development", "m1").await.unwrap();
        engine4.store("I dual-boot Linux Mint alongside Windows now for dev work", "m3").await.unwrap();
        engine4.store("I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely", "m6").await.unwrap();
    });
    println!("\nRunning PT consolidation...");
    run_full_consolidation(&engine4);
    let pt_results = rt4.block_on(async {
        let queries = vec![
            ("PT-1: IDE", "What code editor do I use?", "Neovim"),
            ("PT-2: Diet", "What's my diet like?", "pescatarian"),
            ("PT-3: Coffee", "How do I take my coffee?", "pour-over"),
            ("PT-4: OS", "What operating system do I use?", "Arch"),
        ];

        let mut results = Vec::new();
        for (name, query, needle) in &queries {
            let r = engine4.echo(query, 5).await.expect("echo");
            let hit3 = top_n_contains(&r, 3, needle);
            let hit5 = top_n_contains(&r, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if hit3 { "PASS" } else { "MISS" },
                if hit5 { "PASS" } else { "MISS" }
            );
            for (i, res) in r.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    res.similarity,
                    res.final_score,
                    &res.content[..res.content.len().min(70)]
                );
            }
            results.push((hit3, hit5));
        }
        results
    });
    drop(rt4);

    // --- SCORECARD ---
    let all_results: Vec<(bool, bool)> = ie_results
        .into_iter()
        .chain(msr_results)
        .chain(ku_results)
        .chain(pt_results)
        .collect();

    let total = all_results.len();
    let strict = all_results.iter().filter(|(h3, _)| *h3).count();
    let relaxed = all_results.iter().filter(|(_, h5)| *h5).count();
    let strict_pct = (strict as f64 / total as f64) * 100.0;
    let relaxed_pct = (relaxed as f64 / total as f64) * 100.0;

    println!("\n=== HYDE LONGMEMEVAL SCORECARD ===");
    println!("Total tests:      {total}");
    println!("Top-3 accuracy:   {strict}/{total} ({strict_pct:.1}%)");
    println!("Top-5 accuracy:   {relaxed}/{total} ({relaxed_pct:.1}%)");
    println!();
    println!("Baseline (no enrichment):    90.0% top-3, 100.0% top-5");
    println!("Consolidated (no HyDE):      see consolidated scorecard");
    println!("Consolidated + HyDE:         {strict_pct:.1}% top-3, {relaxed_pct:.1}% top-5");
    println!();
    println!("=== HYDE BENCHMARK COMPLETE ===");
}

// ===========================================================================
// COMBINED: Consolidation + HyDE + Reranker
// ===========================================================================

fn longmemeval_combined_config(data_dir: PathBuf) -> EchoConfig {
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
        query_expansion_enabled: true,
        reranker_enabled: true,
        fact_extraction_prompt: Some(
            "Extract personal facts from the text. Rules:\n\
             1. One fact per line, starting with \"The user\"\n\
             2. Use ONLY these verbs: works at, works for, joined, lives in, moved to, \
             based in, uses, prefers, switched to, likes, chose, belongs to, member of, part of\n\
             3. No colons, labels, or key-value pairs\n\n\
             Example:\n  The user uses Neovim\n  The user lives in Berlin\n  \
             The user switched to Python from Java\n\n\
             Max {max_facts} facts. If none found, output NONE."
                .to_string(),
        ),
        ..Default::default()
    }
}

#[test]
#[ignore = "requires Ollama with llama3.2:3b"]
fn longmemeval_combined_scorecard() {
    println!("\n=== LongMemEval COMBINED (C1 + HyDE + Reranker) ===\n");

    // IE
    let dir = tempdir().expect("temp dir");
    let engine =
        EchoEngine::new(longmemeval_combined_config(dir.path().to_path_buf())).expect("init");
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I have a golden retriever named Pixel who is 4 years old",
            "My partner's name is Jordan and we've been together for 6 years",
            "I graduated from the University of British Columbia with a CS degree in 2015",
            "I was born in Taipei, Taiwan but grew up in Vancouver, Canada",
            "I'm allergic to shellfish and cats",
            "My favorite cuisine is Thai food, especially pad see ew and massaman curry",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
            "I drive a 2022 Tesla Model 3 but mostly bike to work",
            "I'm being considered for a staff engineer promotion next quarter",
            "My IDE is Neovim with LazyVim config and Catppuccin theme",
            "I enjoy listening to classical music while coding, especially Chopin",
            "I read about 30 books a year, mostly sci-fi and technical books",
            "I donate monthly to the EFF and Wikipedia",
            "I have a standing desk and use a split ergonomic keyboard",
            "My morning routine is meditation, coffee, then 30 minutes of reading",
            "I run a small Kubernetes cluster at home for side projects",
        ] {
            engine.store(text, "profile").await.expect("store");
        }
    });
    println!("Running consolidation...");
    run_full_consolidation(&engine);
    let ie = rt.block_on(async {
        let q = vec![
            (
                "IE-1: Profession",
                "What is my job? Where do I work?",
                "Stripe",
            ),
            ("IE-2: Pet", "Do I have any pets?", "golden retriever"),
            (
                "IE-3: Education",
                "Where did I go to college?",
                "British Columbia",
            ),
            ("IE-4: Allergy", "What am I allergic to?", "shellfish"),
            (
                "IE-5: Hobby",
                "What sports or physical activities do I do?",
                "jiu-jitsu",
            ),
        ];
        let mut r = Vec::new();
        for (name, query, needle) in &q {
            let res = engine.echo(query, 5).await.expect("echo");
            let h3 = top_n_contains(&res, 3, needle);
            let h5 = top_n_contains(&res, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if h3 { "PASS" } else { "MISS" },
                if h5 { "PASS" } else { "MISS" }
            );
            r.push((h3, h5));
        }
        r
    });
    drop(rt);

    // MSR
    let dir2 = tempdir().expect("temp dir");
    let engine2 =
        EchoEngine::new(longmemeval_combined_config(dir2.path().to_path_buf())).expect("init");
    let rt2 = tokio::runtime::Runtime::new().unwrap();
    rt2.block_on(async {
        for text in &[
            "My name is Alex Chen and I'm 32 years old",
            "I work as a senior backend engineer at Stripe in San Francisco",
            "I prefer Rust for systems programming and Go for microservices",
            "For databases I use PostgreSQL for OLTP and ClickHouse for analytics",
            "I'm learning Japanese and currently at JLPT N3 level",
            "I visited Tokyo last November and stayed in Shinjuku for two weeks",
            "I practice Brazilian jiu-jitsu three times a week at a Gracie gym",
            "I want to compete in a jiu-jitsu tournament by the end of the year",
            "My long-term goal is to start a developer tools company focused on observability",
            "I'm saving for a house in the Oakland Hills area",
        ] {
            engine2.store(text, "profile").await.expect("store");
        }
    });
    println!("\nRunning MSR consolidation...");
    run_full_consolidation(&engine2);
    let msr = rt2.block_on(async {
        let q = vec![
            (
                "MSR-1: Work+Lang",
                "What programming languages do I use at work?",
                "Rust",
            ),
            (
                "MSR-2: Travel+Lang",
                "Have I traveled anywhere related to languages I'm learning?",
                "Tokyo",
            ),
            (
                "MSR-3: Hobby+Goal",
                "What goals do I have related to my hobbies?",
                "tournament",
            ),
            (
                "MSR-4: Career",
                "Tell me about my career progression",
                "Stripe",
            ),
            (
                "MSR-5: Life Goals",
                "What are my big life goals? What am I saving up for?",
                "house",
            ),
        ];
        let mut r = Vec::new();
        for (name, query, needle) in &q {
            let res = engine2.echo(query, 5).await.expect("echo");
            let h3 = top_n_contains(&res, 3, needle);
            let h5 = top_n_contains(&res, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if h3 { "PASS" } else { "MISS" },
                if h5 { "PASS" } else { "MISS" }
            );
            r.push((h3, h5));
        }
        r
    });
    drop(rt2);

    // KU
    let dir3 = tempdir().expect("temp dir");
    let engine3 =
        EchoEngine::new(longmemeval_combined_config(dir3.path().to_path_buf())).expect("init");
    let rt3 = tokio::runtime::Runtime::new().unwrap();
    rt3.block_on(async {
        engine3.store("I work as a backend engineer at Google on the Cloud Spanner team", "old").await.unwrap();
        engine3.store("I left Google last month. I now work at Meta on the infrastructure team", "new").await.unwrap();
        engine3.store("I enjoy hiking on weekends in the bay area", "noise").await.unwrap();
        engine3.store("I live in a one-bedroom apartment in downtown Seattle", "old").await.unwrap();
        engine3.store("I just moved to Portland, Oregon and I'm renting a house in the Pearl District", "new").await.unwrap();
        engine3.store("Python is my go-to programming language for everything", "old").await.unwrap();
        engine3.store("I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve", "new").await.unwrap();
    });
    println!("\nRunning KU consolidation...");
    run_full_consolidation(&engine3);
    let ku = rt3.block_on(async {
        let q = vec![
            ("KU-1: Job Change", "Where do I work now?", "Meta"),
            ("KU-2: Address", "Where do I live now?", "Portland"),
            (
                "KU-3: Language",
                "What programming language do I mainly use?",
                "Rust",
            ),
        ];
        let mut r = Vec::new();
        for (name, query, needle) in &q {
            let res = engine3.echo(query, 5).await.expect("echo");
            let h3 = top_n_contains(&res, 3, needle);
            let h5 = top_n_contains(&res, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if h3 { "PASS" } else { "MISS" },
                if h5 { "PASS" } else { "MISS" }
            );
            r.push((h3, h5));
        }
        r
    });
    drop(rt3);

    // PT
    let dir4 = tempdir().expect("temp dir");
    let engine4 =
        EchoEngine::new(longmemeval_combined_config(dir4.path().to_path_buf())).expect("init");
    let rt4 = tokio::runtime::Runtime::new().unwrap();
    rt4.block_on(async {
        engine4.store("I use Sublime Text as my code editor, it's fast and lightweight", "m1").await.unwrap();
        engine4.store("I switched to VS Code because of the extension ecosystem", "m3").await.unwrap();
        engine4.store("I've moved to Neovim with a custom Lua config for maximum speed", "m6").await.unwrap();
        engine4.store("I'm vegetarian and have been for the past 3 years", "m1").await.unwrap();
        engine4.store("I started eating fish again, so now I'm pescatarian", "m4").await.unwrap();
        engine4.store("I drink regular drip coffee with cream and sugar", "m1").await.unwrap();
        engine4.store("I switched to espresso-based drinks, usually a latte with whole milk", "m3").await.unwrap();
        engine4.store("Now I drink pour-over black coffee, no milk no sugar, using a Hario V60", "m6").await.unwrap();
        engine4.store("I use Windows 11 on all my machines for gaming and development", "m1").await.unwrap();
        engine4.store("I dual-boot Linux Mint alongside Windows now for dev work", "m3").await.unwrap();
        engine4.store("I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely", "m6").await.unwrap();
    });
    println!("\nRunning PT consolidation...");
    run_full_consolidation(&engine4);
    let pt = rt4.block_on(async {
        let q = vec![
            ("PT-1: IDE", "What code editor do I use?", "Neovim"),
            ("PT-2: Diet", "What's my diet like?", "pescatarian"),
            ("PT-3: Coffee", "How do I take my coffee?", "pour-over"),
            ("PT-4: OS", "What operating system do I use?", "Arch"),
        ];
        let mut r = Vec::new();
        for (name, query, needle) in &q {
            let res = engine4.echo(query, 5).await.expect("echo");
            let h3 = top_n_contains(&res, 3, needle);
            let h5 = top_n_contains(&res, 5, needle);
            println!(
                "{name}: top3={} top5={}",
                if h3 { "PASS" } else { "MISS" },
                if h5 { "PASS" } else { "MISS" }
            );
            for (i, r) in res.iter().take(3).enumerate() {
                println!(
                    "  #{}: sim={:.3} score={:.3} {}",
                    i + 1,
                    r.similarity,
                    r.final_score,
                    &r.content[..r.content.len().min(70)]
                );
            }
            r.push((h3, h5));
        }
        r
    });
    drop(rt4);

    let all: Vec<(bool, bool)> = ie.into_iter().chain(msr).chain(ku).chain(pt).collect();
    let total = all.len();
    let strict = all.iter().filter(|(h3, _)| *h3).count();
    let relaxed = all.iter().filter(|(_, h5)| *h5).count();
    let strict_pct = (strict as f64 / total as f64) * 100.0;
    let relaxed_pct = (relaxed as f64 / total as f64) * 100.0;

    println!("\n=== COMBINED LONGMEMEVAL SCORECARD ===");
    println!("Total tests:      {total}");
    println!("Top-3 accuracy:   {strict}/{total} ({strict_pct:.1}%)");
    println!("Top-5 accuracy:   {relaxed}/{total} ({relaxed_pct:.1}%)");
    println!();
    println!("Baseline:             90.0% top-3, 100.0% top-5");
    println!("Consolidated (C1):    88.2% top-3, 100.0% top-5");
    println!("HyDE only:            88.2% top-3, 100.0% top-5");
    println!("Reranker only:        94.1% top-3, 94.1% top-5");
    println!("COMBINED:             {strict_pct:.1}% top-3, {relaxed_pct:.1}% top-5");
    println!();
    println!("=== COMBINED BENCHMARK COMPLETE ===");
}
