//! Expanded LongMemEval benchmark for ShrimPK Echo Memory (KS24 Track 2).
//!
//! Adds 30 new tests (6 per category) to the original 20, bringing the total
//! to 50 tests across all 5 LongMemEval categories:
//!
//! 1. **Information Extraction** (IE-6 through IE-11)
//! 2. **Multi-Session Reasoning** (MSR-6 through MSR-11)
//! 3. **Temporal Reasoning** (TR-4 through TR-9)
//! 4. **Knowledge Update** (KU-4 through KU-9)
//! 5. **Preference Tracking** (PT-5 through PT-10)
//!
//! Run with:
//!
//!     cargo test --test echo_longmemeval_expanded -- --ignored --nocapture
//!
//! **ort runtime fix**: Every test creates `EchoEngine` OUTSIDE the tokio
//! runtime, then uses `block_on` only for async store/echo operations.

use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use tempfile::tempdir;

// ===========================================================================
// Config helper (identical to original)
// ===========================================================================

fn longmemeval_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.15,
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

// ===========================================================================
// Scoring helper (identical to original)
// ===========================================================================

fn top_n_contains(results: &[shrimpk_core::EchoResult], n: usize, needle: &str) -> bool {
    let needle_lower = needle.to_lowercase();
    results
        .iter()
        .take(n)
        .any(|r| r.content.to_lowercase().contains(&needle_lower))
}

// ===========================================================================
// Extended seed helper — 25 original + 10 new memories = 35 total
//
// The 10 new memories cover facts needed by IE-6..IE-11 and MSR-6..MSR-11
// that the original 25-memory profile did not include.
// ===========================================================================

fn seed_extended_profile_sync(engine: &EchoEngine, rt: &tokio::runtime::Runtime) {
    rt.block_on(async {
        // --- Original 25 memories (sessions 1-5) ---

        // Session 1: personal basics
        engine.store("My name is Alex Chen and I'm 32 years old", "session1").await.unwrap();
        engine.store("I was born in Taipei, Taiwan but grew up in Vancouver, Canada", "session1").await.unwrap();
        engine.store("I have a golden retriever named Pixel who is 4 years old", "session1").await.unwrap();
        engine.store("My partner's name is Jordan and we've been together for 6 years", "session1").await.unwrap();
        engine.store("I'm allergic to shellfish and cats", "session1").await.unwrap();

        // Session 2: work and education
        engine.store("I work as a senior backend engineer at Stripe in San Francisco", "session2").await.unwrap();
        engine.store("I graduated from the University of British Columbia with a CS degree in 2015", "session2").await.unwrap();
        engine.store("Before Stripe I worked at Shopify for 3 years on their payments team", "session2").await.unwrap();
        engine.store("My team at Stripe works on the billing infrastructure service", "session2").await.unwrap();
        engine.store("I'm being considered for a staff engineer promotion next quarter", "session2").await.unwrap();

        // Session 3: technical preferences
        engine.store("I prefer Rust for systems programming and Go for microservices", "session3").await.unwrap();
        engine.store("My IDE is Neovim with LazyVim config and Catppuccin theme", "session3").await.unwrap();
        engine.store("For databases I use PostgreSQL for OLTP and ClickHouse for analytics", "session3").await.unwrap();
        engine.store("I run NixOS on my personal machines and macOS at work", "session3").await.unwrap();
        engine.store("My dotfiles are managed with chezmoi and stored on GitHub", "session3").await.unwrap();

        // Session 4: hobbies and lifestyle
        engine.store("I practice Brazilian jiu-jitsu three times a week at a Gracie gym", "session4").await.unwrap();
        engine.store("I'm learning Japanese and currently at JLPT N3 level", "session4").await.unwrap();
        engine.store("I collect mechanical keyboards and my daily driver is a Keychron Q1 with Boba U4T switches", "session4").await.unwrap();
        engine.store("I brew pour-over coffee every morning using a Hario V60 and light roast beans", "session4").await.unwrap();
        engine.store("My favorite cuisine is Thai food, especially pad see ew and massaman curry", "session4").await.unwrap();

        // Session 5: travel and goals
        engine.store("I visited Tokyo last November and stayed in Shinjuku for two weeks", "session5").await.unwrap();
        engine.store("My next trip is planned for Barcelona in April 2027", "session5").await.unwrap();
        engine.store("My long-term goal is to start a developer tools company focused on observability", "session5").await.unwrap();
        engine.store("I'm saving for a house in the Oakland Hills area", "session5").await.unwrap();
        engine.store("I want to compete in a jiu-jitsu tournament by the end of the year", "session5").await.unwrap();

        // --- 10 NEW memories for expanded tests (sessions 6-7) ---

        // Session 6: family, health, vehicle, reading, donations
        engine.store("I have a younger sister named Maya who lives in Toronto and works as a graphic designer", "session6").await.unwrap();
        engine.store("My morning routine is meditation for 10 minutes, then pour-over coffee, then 30 minutes of reading before work", "session6").await.unwrap();
        engine.store("I drive a 2022 Tesla Model 3 but mostly bike to work on my Canyon Grail gravel bike", "session6").await.unwrap();
        engine.store("I read about 30 books a year, mostly science fiction novels and technical books about distributed systems", "session6").await.unwrap();
        engine.store("I donate monthly to the EFF and Wikipedia, and I sponsor two open-source maintainers on GitHub", "session6").await.unwrap();

        // Session 7: music, commute, community, desk setup, study
        engine.store("I listen to lo-fi hip hop and Chopin while coding, and jazz when I'm relaxing", "session7").await.unwrap();
        engine.store("I usually bike to the Stripe office, it's a 25-minute ride from my apartment in the Mission District", "session7").await.unwrap();
        engine.store("I'm active on the Rust subreddit and a member of the San Francisco Rust meetup group", "session7").await.unwrap();
        engine.store("I have a standing desk from Uplift and use a split ergonomic Kinesis Advantage 360 keyboard", "session7").await.unwrap();
        engine.store("I'm currently studying for the AWS Solutions Architect certification in my spare time", "session7").await.unwrap();
    });
}

// ===========================================================================
// Category 1: Information Extraction — 6 NEW tests (IE-6 through IE-11)
// ===========================================================================

/// IE-6: Workplace details — "What team am I on?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ie_6_team() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What team am I on at work?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-6 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "billing infrastructure"),
        "Top-3 should mention the billing infrastructure team. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-7: Family — "Do I have siblings?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ie_7_siblings() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("Do I have any brothers or sisters?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-7 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Maya") || top_n_contains(&results, 3, "sister"),
        "Top-3 should mention sister Maya. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-8: Health routine — "What's my morning routine?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ie_8_morning_routine() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What do I do every morning? What's my morning routine?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-8 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "meditation") || top_n_contains(&results, 3, "morning routine"),
        "Top-3 should mention morning routine (meditation/coffee/reading). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-9: Vehicle — "What car do I drive?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ie_9_vehicle() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What car do I drive? How do I get around?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-9 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Tesla") || top_n_contains(&results, 3, "Model 3"),
        "Top-3 should mention Tesla Model 3. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-10: Reading habits — "What kind of books do I read?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ie_10_reading() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What kind of books do I like to read?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-10 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "sci-fi")
            || top_n_contains(&results, 3, "science fiction")
            || top_n_contains(&results, 3, "technical books"),
        "Top-3 should mention reading habits (sci-fi / technical). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// IE-11: Financial — "What do I donate to?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ie_11_donations() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What charities or causes do I support financially?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("IE-11 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "EFF")
            || top_n_contains(&results, 3, "Wikipedia")
            || top_n_contains(&results, 3, "donate"),
        "Top-3 should mention donations (EFF/Wikipedia). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 2: Multi-Session Reasoning — 6 NEW tests (MSR-6 through MSR-11)
// ===========================================================================

/// MSR-6: Tech stack connection — "What database goes with my language?"
/// Requires connecting Rust preference (session3) with database choice (session3).
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_msr_6_tech_stack() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What database do I use for my backend services?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-6 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_db =
        top_n_contains(&results, 5, "PostgreSQL") || top_n_contains(&results, 5, "ClickHouse");
    let has_backend = top_n_contains(&results, 5, "backend")
        || top_n_contains(&results, 5, "Stripe")
        || top_n_contains(&results, 5, "Rust");

    assert!(
        has_db,
        "Top-5 should mention a database (PostgreSQL/ClickHouse). Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
    assert!(
        has_backend,
        "Top-5 should also surface backend context. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-7: Location + work — "Where is my office?"
/// Requires connecting Stripe (session2) with San Francisco / Mission District (session7).
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_msr_7_office_location() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("Where is my office? How far is my commute?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-7 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_stripe =
        top_n_contains(&results, 5, "Stripe") || top_n_contains(&results, 5, "San Francisco");
    let has_commute = top_n_contains(&results, 5, "bike")
        || top_n_contains(&results, 5, "Mission District")
        || top_n_contains(&results, 5, "25-minute");

    assert!(
        has_stripe && has_commute,
        "Top-5 should mention both workplace and commute details. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-8: Music + coding — "What do I listen to while working?"
/// Requires connecting music preference (session7) with coding context.
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_msr_8_music_while_coding() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("What music do I listen to while programming?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-8 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_music = top_n_contains(&results, 5, "lo-fi")
        || top_n_contains(&results, 5, "Chopin")
        || top_n_contains(&results, 5, "hip hop");

    assert!(
        has_music,
        "Top-5 should mention music listening habits (lo-fi/Chopin). Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-9: Pet + allergy connection — "What animals do I have despite allergies?"
/// Requires connecting golden retriever (session1) with cat allergy (session1).
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_msr_9_pet_allergy() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("I want to get a new pet. What should I know about my animal allergies and current pets?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-9 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_pet =
        top_n_contains(&results, 5, "golden retriever") || top_n_contains(&results, 5, "Pixel");
    let has_allergy =
        top_n_contains(&results, 5, "allergic") || top_n_contains(&results, 5, "cats");

    assert!(
        has_pet && has_allergy,
        "Top-5 should surface both pet info and allergy info. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-10: Skills + career — "What skills relate to my career goal?"
/// Requires connecting observability startup goal (session5) with Rust/Go skills (session3).
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_msr_10_skills_career() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo(
                "What technical skills do I have that would help me start my own company?",
                5,
            )
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-10 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_goal = top_n_contains(&results, 5, "developer tools")
        || top_n_contains(&results, 5, "observability")
        || top_n_contains(&results, 5, "company");
    let has_skill = top_n_contains(&results, 5, "Rust")
        || top_n_contains(&results, 5, "Go")
        || top_n_contains(&results, 5, "backend")
        || top_n_contains(&results, 5, "engineer");

    assert!(
        has_goal || has_skill,
        "Top-5 should surface startup goal or relevant technical skills. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// MSR-11: Hobby + location — "Where do I practice my sport?"
/// Requires connecting jiu-jitsu (session4) with Gracie gym location.
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_msr_11_hobby_location() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    seed_extended_profile_sync(&engine, &rt);
    let results = rt.block_on(async {
        engine
            .echo("Where do I train jiu-jitsu? What gym do I go to?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("MSR-11 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_bjj = top_n_contains(&results, 3, "jiu-jitsu");
    let has_gym = top_n_contains(&results, 5, "Gracie") || top_n_contains(&results, 5, "gym");

    assert!(
        has_bjj && has_gym,
        "Top-5 should mention both jiu-jitsu and the Gracie gym. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 3: Temporal Reasoning — 6 NEW tests (TR-4 through TR-9)
// ===========================================================================

/// TR-4: Education timeline — "When did I graduate?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_tr_4_graduation() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I started university at UBC in September 2011",
                "session_edu1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I did a summer internship at Amazon in 2013",
                "session_edu2",
            )
            .await
            .unwrap();
        engine
            .store(
                "I graduated from UBC with a CS degree in June 2015",
                "session_edu3",
            )
            .await
            .unwrap();
        engine
            .store(
                "After graduating I spent 3 months backpacking in Southeast Asia",
                "session_edu4",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "My sister graduated from art school in 2018",
                "session_noise",
            )
            .await
            .unwrap();

        engine
            .echo("When did I finish university? What year did I graduate?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-4 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "2015") || top_n_contains(&results, 3, "graduated"),
        "Top-3 should mention graduation in 2015. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// TR-5: Relationship duration — "How long have I been with my partner?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_tr_5_relationship() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I met Jordan at a coffee shop in Vancouver in 2018",
                "session_rel1",
            )
            .await
            .unwrap();
        engine
            .store("Jordan and I started dating in March 2019", "session_rel2")
            .await
            .unwrap();
        engine
            .store(
                "We moved in together in San Francisco in 2021",
                "session_rel3",
            )
            .await
            .unwrap();
        engine
            .store(
                "Jordan and I celebrated our 5th anniversary last month",
                "session_rel4",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "My coworker just got engaged to their partner",
                "session_noise",
            )
            .await
            .unwrap();

        engine
            .echo(
                "How long have I been with my partner? When did we start dating?",
                5,
            )
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-5 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_dating =
        top_n_contains(&results, 5, "2019") || top_n_contains(&results, 5, "started dating");
    let has_anniversary =
        top_n_contains(&results, 5, "anniversary") || top_n_contains(&results, 5, "5th");

    assert!(
        has_dating || has_anniversary,
        "Top-5 should mention relationship timeline. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// TR-6: Pet age tracking — "How old is my dog?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_tr_6_pet_age() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I adopted Pixel as a puppy from a rescue shelter in 2021",
                "session_pet1",
            )
            .await
            .unwrap();
        engine
            .store(
                "Pixel turned 2 years old in March 2023, we had a little birthday party",
                "session_pet2",
            )
            .await
            .unwrap();
        engine
            .store(
                "Pixel is now 4 years old and has calmed down a lot from his puppy energy",
                "session_pet3",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "My neighbor's cat is 12 years old and still very active",
                "session_noise",
            )
            .await
            .unwrap();

        engine
            .echo("How old is my dog now?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-6 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "4 years old") || top_n_contains(&results, 3, "Pixel"),
        "Top-3 should mention Pixel's current age. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// TR-7: Career timing — "When did I start at my current job?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_tr_7_career_timing() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "In 2016 I got my first programming job at a small Vancouver startup",
                "session_c1",
            )
            .await
            .unwrap();
        engine
            .store(
                "In 2019 I joined Shopify as a backend engineer on their payments team",
                "session_c2",
            )
            .await
            .unwrap();
        engine
            .store(
                "In January 2022 I started at Stripe as a senior backend engineer",
                "session_c3",
            )
            .await
            .unwrap();
        engine
            .store(
                "I've been at Stripe for over 2 years now and I'm up for a promotion",
                "session_c4",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "My friend just started a new job at Apple last week",
                "session_noise",
            )
            .await
            .unwrap();

        engine
            .echo("When did I start working at my current company?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-7 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    let has_stripe_start =
        top_n_contains(&results, 3, "2022") || top_n_contains(&results, 3, "Stripe");

    assert!(
        has_stripe_start,
        "Top-3 should mention starting at Stripe in 2022. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// TR-8: Recent activity — "What did I do last November?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_tr_8_recent_activity() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("Last March I ran a half marathon in San Francisco", "session_a1")
            .await
            .unwrap();
        engine
            .store("In July I attended a music festival in Portland", "session_a2")
            .await
            .unwrap();
        engine
            .store("Last November I traveled to Tokyo and explored Shinjuku and Akihabara for two weeks", "session_a3")
            .await
            .unwrap();
        engine
            .store("In December I started a new side project building a CLI tool in Rust", "session_a4")
            .await
            .unwrap();

        // Noise
        engine
            .store("I prefer warm weather over cold weather", "session_noise")
            .await
            .unwrap();

        engine
            .echo("What did I do last November? Any trips?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-8 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Tokyo") || top_n_contains(&results, 3, "November"),
        "Top-3 should mention the Tokyo trip in November. Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// TR-9: Future plans — "When is my next competition?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_tr_9_future_plans() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I competed in my first jiu-jitsu tournament last September",
                "session_f1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I've registered for the IBJJF San Francisco Open in March 2027",
                "session_f2",
            )
            .await
            .unwrap();
        engine
            .store(
                "After that I want to try the Pan American championships in April 2027",
                "session_f3",
            )
            .await
            .unwrap();
        engine
            .store(
                "My next trip is planned for Barcelona in April 2027",
                "session_f4",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store("I should buy new running shoes soon", "session_noise")
            .await
            .unwrap();

        engine
            .echo("When is my next jiu-jitsu competition?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("TR-9 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "IBJJF")
            || top_n_contains(&results, 3, "San Francisco Open")
            || top_n_contains(&results, 3, "March 2027"),
        "Top-3 should mention the next competition (IBJJF SF Open). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 4: Knowledge Update — 6 NEW tests (KU-4 through KU-9)
// ===========================================================================

/// KU-4: Hobby change — "I stopped running, now I swim"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ku_4_hobby_change() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I run 5 kilometers every morning before work, it's my main cardio exercise", "session_old")
            .await
            .unwrap();
        engine
            .store("I enjoy listening to podcasts during my morning run", "session_noise")
            .await
            .unwrap();
        engine
            .store("I had to stop running due to a knee injury. Now I swim laps at the YMCA pool three times a week instead", "session_new")
            .await
            .unwrap();

        engine
            .echo("What cardio exercise do I do?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-4 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "swim") || top_n_contains(&results, 3, "YMCA"),
        "Top-3 should mention swimming (current exercise). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );

    // Old hobby should also surface for context
    assert!(
        top_n_contains(&results, 5, "run") || top_n_contains(&results, 5, "running"),
        "Top-5 should also mention running (old hobby) for context. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// KU-5: Relationship status update
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ku_5_relationship_update() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I've been dating Jordan for about 3 years, we live together in an apartment", "session_old")
            .await
            .unwrap();
        engine
            .store("I love cooking dinner with Jordan on weekends", "session_noise")
            .await
            .unwrap();
        engine
            .store("Jordan and I got engaged last weekend! We're planning a small wedding for next spring", "session_new")
            .await
            .unwrap();

        engine
            .echo("What's my relationship status? Am I married or single?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-5 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "engaged") || top_n_contains(&results, 3, "wedding"),
        "Top-3 should mention engagement (most recent status). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// KU-6: Multiple updates to same fact — 3 address changes
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ku_6_triple_address_change() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I live in a studio apartment in downtown Vancouver near Gastown",
                "session_addr1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I moved to Seattle for a new job, renting a place in Capitol Hill",
                "session_addr2",
            )
            .await
            .unwrap();
        engine
            .store(
                "I relocated to San Francisco last month, now living in the Mission District",
                "session_addr3",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "I love exploring new neighborhoods and finding good coffee shops",
                "session_noise",
            )
            .await
            .unwrap();

        engine
            .echo("Where do I currently live?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-6 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // Most recent address should surface
    assert!(
        top_n_contains(&results, 3, "San Francisco")
            || top_n_contains(&results, 3, "Mission District"),
        "Top-3 should mention San Francisco / Mission District (most recent address). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );

    // All addresses should appear in top-5 for full history
    let addr_count = [
        top_n_contains(&results, 5, "Vancouver"),
        top_n_contains(&results, 5, "Seattle"),
        top_n_contains(&results, 5, "San Francisco"),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    assert!(
        addr_count >= 2,
        "Top-5 should contain at least 2 of 3 addresses. Found {addr_count}. Got: {:?}",
        results
            .iter()
            .take(5)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// KU-7: Partial update — salary changed but same company
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ku_7_partial_update() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I work at Stripe as a senior engineer with a salary of $185k base plus equity", "session_old")
            .await
            .unwrap();
        engine
            .store("I got promoted to staff engineer at Stripe with a salary increase to $230k base plus larger equity grant", "session_new")
            .await
            .unwrap();

        // Noise
        engine
            .store("Tech salaries in the Bay Area have been going up this year", "session_noise")
            .await
            .unwrap();

        engine
            .echo("What's my current role and compensation?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-7 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    // Updated info should surface
    assert!(
        top_n_contains(&results, 3, "staff engineer") || top_n_contains(&results, 3, "230k"),
        "Top-3 should mention staff engineer / $230k (updated compensation). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// KU-8: Skill level update — "JLPT N3 -> N2"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ku_8_skill_level_update() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I'm learning Japanese and just passed the JLPT N4 test",
                "session_skill1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I passed the JLPT N3 exam last December after months of studying",
                "session_skill2",
            )
            .await
            .unwrap();
        engine
            .store(
                "I just got my JLPT N2 certification! Next goal is N1",
                "session_skill3",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "I want to visit Kyoto to practice my Japanese in a traditional setting",
                "session_noise",
            )
            .await
            .unwrap();

        engine
            .echo(
                "What level is my Japanese at? What JLPT level did I reach?",
                5,
            )
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-8 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "N2"),
        "Top-3 should mention JLPT N2 (most recent level). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// KU-9: Diet evolution — 3 stages
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_ku_9_diet_evolution() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I eat everything, no restrictions. I love burgers and steaks",
                "session_diet1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I cut out red meat for health reasons, now I only eat chicken and fish",
                "session_diet2",
            )
            .await
            .unwrap();
        engine
            .store(
                "I've gone fully plant-based vegan. No animal products at all, and I feel amazing",
                "session_diet3",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "I started a new workout routine focusing on strength training",
                "session_noise",
            )
            .await
            .unwrap();

        engine
            .echo("What's my current diet? Do I eat meat?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("KU-9 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "vegan") || top_n_contains(&results, 3, "plant-based"),
        "Top-3 should mention vegan/plant-based (current diet). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Category 5: Preference Tracking — 6 NEW tests (PT-5 through PT-10)
// ===========================================================================

/// PT-5: Music preference — "What genre do I listen to?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_pt_5_music() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I mostly listen to pop music and top 40 hits", "month1")
            .await
            .unwrap();
        engine
            .store(
                "I've been getting into indie rock and alternative music lately",
                "month3",
            )
            .await
            .unwrap();
        engine
            .store(
                "Now I mainly listen to lo-fi hip hop and ambient electronic music for focus",
                "month6",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store("I bought new headphones, the Sony WH-1000XM5", "noise")
            .await
            .unwrap();

        engine
            .echo(
                "What kind of music do I listen to? What's my music taste?",
                5,
            )
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-5 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "lo-fi")
            || top_n_contains(&results, 3, "ambient")
            || top_n_contains(&results, 3, "electronic"),
        "Top-3 should mention lo-fi/ambient (most recent preference). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// PT-6: Transportation preference — "How do I commute?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_pt_6_commute() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I drive my car to work every day, about a 40-minute commute on the highway",
                "month1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I started taking the BART train to reduce my carbon footprint",
                "month4",
            )
            .await
            .unwrap();
        engine
            .store(
                "I now bike to work every day, it's a 25-minute ride and I love the fresh air",
                "month8",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store("Gas prices have been going up a lot this year", "noise")
            .await
            .unwrap();

        engine
            .echo("How do I get to work? What's my commute like?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-6 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "bike"),
        "Top-3 should mention biking (most recent commute). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// PT-7: Work setup — "What kind of desk/keyboard do I use?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_pt_7_desk_setup() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I work at a regular sitting desk with a standard Dell keyboard and mouse", "month1")
            .await
            .unwrap();
        engine
            .store("I got a sit-stand desk converter for my existing desk, and a mechanical keyboard", "month4")
            .await
            .unwrap();
        engine
            .store("I upgraded to a full Uplift standing desk and a split ergonomic Kinesis Advantage 360", "month8")
            .await
            .unwrap();

        // Noise
        engine
            .store("I need to clean my office this weekend", "noise")
            .await
            .unwrap();

        engine
            .echo("What's my desk and keyboard setup at work?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-7 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Uplift")
            || top_n_contains(&results, 3, "Kinesis")
            || top_n_contains(&results, 3, "standing desk"),
        "Top-3 should mention Uplift/Kinesis (most recent setup). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// PT-8: Learning preference — "What am I currently studying?"
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_pt_8_learning() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store(
                "I'm taking an online course on machine learning fundamentals",
                "month1",
            )
            .await
            .unwrap();
        engine
            .store(
                "I finished the ML course and started studying distributed systems design",
                "month3",
            )
            .await
            .unwrap();
        engine
            .store(
                "I'm currently studying for the AWS Solutions Architect certification exam",
                "month6",
            )
            .await
            .unwrap();

        // Noise
        engine
            .store(
                "I find that studying in the morning works best for me",
                "noise",
            )
            .await
            .unwrap();

        engine
            .echo("What am I currently learning or studying?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-8 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "AWS") || top_n_contains(&results, 3, "Solutions Architect"),
        "Top-3 should mention AWS certification (most recent study). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// PT-9: Social media / communities
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_pt_9_communities() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I'm mostly on Twitter for tech discussions and memes", "month1")
            .await
            .unwrap();
        engine
            .store("I left Twitter and moved to Mastodon for the tech community", "month4")
            .await
            .unwrap();
        engine
            .store("I'm now most active on the Rust subreddit and Hacker News, and I'm a member of the San Francisco Rust meetup", "month8")
            .await
            .unwrap();

        // Noise
        engine
            .store("Social media can be really distracting during work hours", "noise")
            .await
            .unwrap();

        engine
            .echo("What online communities or social media am I active on?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-9 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Rust subreddit")
            || top_n_contains(&results, 3, "Hacker News")
            || top_n_contains(&results, 3, "San Francisco Rust"),
        "Top-3 should mention current communities (Reddit/HN/meetup). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

/// PT-10: Cooking preference
#[test]
#[ignore = "requires fastembed model download"]
fn longmemeval_expanded_pt_10_cooking() {
    let dir = tempdir().expect("temp dir");
    let config = longmemeval_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        engine
            .store("I mostly order takeout and eat at restaurants, I don't cook much", "month1")
            .await
            .unwrap();
        engine
            .store("I started meal prepping on Sundays, mostly simple pasta and salad recipes", "month4")
            .await
            .unwrap();
        engine
            .store("I've gotten into Thai and Japanese home cooking, I make pad see ew and ramen from scratch now", "month8")
            .await
            .unwrap();

        // Noise
        engine
            .store("I need to buy a new set of kitchen knives", "noise")
            .await
            .unwrap();

        engine
            .echo("What do I like to cook? What are my cooking habits?", 5)
            .await
            .expect("echo should succeed")
    });
    drop(rt);

    println!("PT-10 results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: sim={:.3} content={}",
            i + 1,
            r.similarity,
            &r.content[..r.content.len().min(80)]
        );
    }

    assert!(
        top_n_contains(&results, 3, "Thai")
            || top_n_contains(&results, 3, "Japanese")
            || top_n_contains(&results, 3, "pad see ew")
            || top_n_contains(&results, 3, "ramen"),
        "Top-3 should mention Thai/Japanese cooking (most recent). Got: {:?}",
        results
            .iter()
            .take(3)
            .map(|r| &r.content)
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Expanded Combined Scorecard — all 50 tests
//
// Runs the combined pipeline (C1 + HyDE + reranker + consolidation) across
// all 50 tests (20 original + 30 new) and prints a full scorecard.
//
// Requires Ollama running with llama3.2:3b.
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

/// Run consolidation passes until no more facts are extracted (max 10 passes).
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

/// Full 50-test LongMemEval scorecard with combined pipeline.
#[test]
#[ignore = "requires Ollama with llama3.2:3b"]
fn longmemeval_expanded_combined_scorecard() {
    println!("\n=== LongMemEval EXPANDED COMBINED Scorecard (50 tests) ===\n");

    #[allow(dead_code)]
    struct TestResult {
        name: &'static str,
        top3: bool,
        top5: bool,
    }

    let mut all_results: Vec<TestResult> = Vec::new();

    // =====================================================================
    // IE: Information Extraction (11 tests)
    // =====================================================================
    println!("--- Category 1: Information Extraction (11 tests) ---");

    let dir_ie = tempdir().expect("temp dir");
    let engine_ie =
        EchoEngine::new(longmemeval_combined_config(dir_ie.path().to_path_buf())).expect("init");
    let rt_ie = tokio::runtime::Runtime::new().unwrap();

    // Store extended profile
    rt_ie.block_on(async {
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
            "My IDE is Neovim with LazyVim config and Catppuccin theme",
            "My team at Stripe works on the billing infrastructure service",
            "Before Stripe I worked at Shopify for 3 years on their payments team",
            "I'm being considered for a staff engineer promotion next quarter",
            "I run NixOS on my personal machines and macOS at work",
            "My dotfiles are managed with chezmoi and stored on GitHub",
            "I collect mechanical keyboards and my daily driver is a Keychron Q1 with Boba U4T switches",
            "I brew pour-over coffee every morning using a Hario V60 and light roast beans",
            "My next trip is planned for Barcelona in April 2027",
            // Extended profile memories
            "I have a younger sister named Maya who lives in Toronto and works as a graphic designer",
            "My morning routine is meditation for 10 minutes, then pour-over coffee, then 30 minutes of reading before work",
            "I drive a 2022 Tesla Model 3 but mostly bike to work on my Canyon Grail gravel bike",
            "I read about 30 books a year, mostly science fiction novels and technical books about distributed systems",
            "I donate monthly to the EFF and Wikipedia, and I sponsor two open-source maintainers on GitHub",
            "I listen to lo-fi hip hop and Chopin while coding, and jazz when I'm relaxing",
            "I usually bike to the Stripe office, it's a 25-minute ride from my apartment in the Mission District",
            "I'm active on the Rust subreddit and a member of the San Francisco Rust meetup group",
            "I have a standing desk from Uplift and use a split ergonomic Kinesis Advantage 360 keyboard",
            "I'm currently studying for the AWS Solutions Architect certification in my spare time",
        ] {
            engine_ie.store(text, "profile").await.expect("store");
        }
    });

    println!("Running IE consolidation...");
    run_full_consolidation(&engine_ie);

    let ie_queries: Vec<(&str, &str, &[&str])> = vec![
        (
            "IE-1: Profession",
            "What is my job? Where do I work?",
            &["Stripe"],
        ),
        ("IE-2: Pet", "Do I have any pets?", &["golden retriever"]),
        (
            "IE-3: Education",
            "Where did I go to college?",
            &["British Columbia"],
        ),
        ("IE-4: Allergy", "What am I allergic to?", &["shellfish"]),
        (
            "IE-5: Hobby",
            "What sports or physical activities do I do?",
            &["jiu-jitsu"],
        ),
        (
            "IE-6: Team",
            "What team am I on at work?",
            &["billing infrastructure"],
        ),
        (
            "IE-7: Siblings",
            "Do I have any brothers or sisters?",
            &["sister", "Maya"],
        ),
        (
            "IE-8: Morning routine",
            "What do I do every morning?",
            &["meditation", "morning routine"],
        ),
        (
            "IE-9: Vehicle",
            "What car do I drive?",
            &["Tesla", "Model 3"],
        ),
        (
            "IE-10: Reading",
            "What kind of books do I read?",
            &["sci-fi", "science fiction", "technical"],
        ),
        (
            "IE-11: Donations",
            "What charities or causes do I support financially?",
            &["EFF", "Wikipedia", "donate"],
        ),
    ];

    rt_ie.block_on(async {
        for (name, query, needles) in &ie_queries {
            let res = engine_ie.echo(query, 5).await.expect("echo");
            let h3 = needles.iter().any(|n| top_n_contains(&res, 3, n));
            let h5 = needles.iter().any(|n| top_n_contains(&res, 5, n));
            println!(
                "  {name}: top3={} top5={}",
                if h3 { "PASS" } else { "MISS" },
                if h5 { "PASS" } else { "MISS" }
            );
            all_results.push(TestResult {
                name,
                top3: h3,
                top5: h5,
            });
        }
    });
    drop(engine_ie);

    // =====================================================================
    // MSR: Multi-Session Reasoning (11 tests)
    // =====================================================================
    println!("\n--- Category 2: Multi-Session Reasoning (11 tests) ---");

    let dir_msr = tempdir().expect("temp dir");
    let engine_msr =
        EchoEngine::new(longmemeval_combined_config(dir_msr.path().to_path_buf())).expect("init");

    // Reuse same extended profile
    rt_ie.block_on(async {
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
            "My IDE is Neovim with LazyVim config and Catppuccin theme",
            "My team at Stripe works on the billing infrastructure service",
            "Before Stripe I worked at Shopify for 3 years on their payments team",
            "I'm being considered for a staff engineer promotion next quarter",
            "I collect mechanical keyboards and my daily driver is a Keychron Q1 with Boba U4T switches",
            "I brew pour-over coffee every morning using a Hario V60 and light roast beans",
            "My next trip is planned for Barcelona in April 2027",
            "I have a younger sister named Maya who lives in Toronto and works as a graphic designer",
            "My morning routine is meditation for 10 minutes, then pour-over coffee, then 30 minutes of reading before work",
            "I drive a 2022 Tesla Model 3 but mostly bike to work on my Canyon Grail gravel bike",
            "I read about 30 books a year, mostly science fiction novels and technical books about distributed systems",
            "I donate monthly to the EFF and Wikipedia, and I sponsor two open-source maintainers on GitHub",
            "I listen to lo-fi hip hop and Chopin while coding, and jazz when I'm relaxing",
            "I usually bike to the Stripe office, it's a 25-minute ride from my apartment in the Mission District",
            "I'm active on the Rust subreddit and a member of the San Francisco Rust meetup group",
            "I have a standing desk from Uplift and use a split ergonomic Kinesis Advantage 360 keyboard",
            "I'm currently studying for the AWS Solutions Architect certification in my spare time",
        ] {
            engine_msr.store(text, "profile").await.expect("store");
        }
    });

    println!("Running MSR consolidation...");
    run_full_consolidation(&engine_msr);

    let msr_queries: Vec<(&str, &str, &[&str])> = vec![
        (
            "MSR-1: Work+Lang",
            "What programming languages do I use at work?",
            &["Rust"],
        ),
        (
            "MSR-2: Travel+Lang",
            "Have I traveled anywhere related to languages I'm learning?",
            &["Tokyo"],
        ),
        (
            "MSR-3: Hobby+Goal",
            "What goals do I have related to my hobbies?",
            &["tournament"],
        ),
        (
            "MSR-4: Career",
            "Tell me about my career progression",
            &["Stripe", "Shopify"],
        ),
        (
            "MSR-5: Life Goals",
            "What are my big life goals? What am I saving up for?",
            &["house", "developer tools", "observability"],
        ),
        (
            "MSR-6: Tech Stack",
            "What database do I use for my backend services?",
            &["PostgreSQL", "ClickHouse"],
        ),
        (
            "MSR-7: Office Location",
            "Where is my office? How far is my commute?",
            &["Stripe", "Mission District", "bike"],
        ),
        (
            "MSR-8: Music+Coding",
            "What music do I listen to while programming?",
            &["lo-fi", "Chopin"],
        ),
        (
            "MSR-9: Pet+Allergy",
            "What should I know about my animal allergies and current pets?",
            &["golden retriever", "Pixel", "allergic", "cats"],
        ),
        (
            "MSR-10: Skills+Career",
            "What technical skills would help me start my own company?",
            &["Rust", "Go", "developer tools", "observability"],
        ),
        (
            "MSR-11: Hobby+Location",
            "Where do I train jiu-jitsu? What gym?",
            &["Gracie", "jiu-jitsu"],
        ),
    ];

    rt_ie.block_on(async {
        for (name, query, needles) in &msr_queries {
            let res = engine_msr.echo(query, 5).await.expect("echo");
            let h3 = needles.iter().any(|n| top_n_contains(&res, 3, n));
            let h5 = needles.iter().any(|n| top_n_contains(&res, 5, n));
            println!(
                "  {name}: top3={} top5={}",
                if h3 { "PASS" } else { "MISS" },
                if h5 { "PASS" } else { "MISS" }
            );
            all_results.push(TestResult {
                name,
                top3: h3,
                top5: h5,
            });
        }
    });
    drop(engine_msr);

    // =====================================================================
    // TR: Temporal Reasoning (9 tests)
    // =====================================================================
    println!("\n--- Category 3: Temporal Reasoning (9 tests) ---");

    let dir_tr = tempdir().expect("temp dir");
    let engine_tr =
        EchoEngine::new(longmemeval_combined_config(dir_tr.path().to_path_buf())).expect("init");

    rt_ie.block_on(async {
        // Original TR memories
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
        // New TR memories
        engine_tr.store("I started university at UBC in September 2011", "edu1").await.unwrap();
        engine_tr.store("I did a summer internship at Amazon in 2013", "edu2").await.unwrap();
        engine_tr.store("I graduated from UBC with a CS degree in June 2015", "edu3").await.unwrap();
        engine_tr.store("After graduating I spent 3 months backpacking in Southeast Asia", "edu4").await.unwrap();
        engine_tr.store("My sister graduated from art school in 2018", "noise3").await.unwrap();
        engine_tr.store("I met Jordan at a coffee shop in Vancouver in 2018", "rel1").await.unwrap();
        engine_tr.store("Jordan and I started dating in March 2019", "rel2").await.unwrap();
        engine_tr.store("We moved in together in San Francisco in 2021", "rel3").await.unwrap();
        engine_tr.store("Jordan and I celebrated our 5th anniversary last month", "rel4").await.unwrap();
        engine_tr.store("My coworker just got engaged to their partner", "noise4").await.unwrap();
        engine_tr.store("I adopted Pixel as a puppy from a rescue shelter in 2021", "pet1").await.unwrap();
        engine_tr.store("Pixel turned 2 years old in March 2023, we had a little birthday party", "pet2").await.unwrap();
        engine_tr.store("Pixel is now 4 years old and has calmed down a lot from his puppy energy", "pet3").await.unwrap();
        engine_tr.store("My neighbor's cat is 12 years old and still very active", "noise5").await.unwrap();
        engine_tr.store("I've been at Stripe for over 2 years now and I'm up for a promotion", "c4").await.unwrap();
        engine_tr.store("My friend just started a new job at Apple last week", "noise6").await.unwrap();
        engine_tr.store("Last March I ran a half marathon in San Francisco", "act1").await.unwrap();
        engine_tr.store("In July I attended a music festival in Portland", "act2").await.unwrap();
        engine_tr.store("Last November I traveled to Tokyo and explored Shinjuku and Akihabara for two weeks", "act3").await.unwrap();
        engine_tr.store("In December I started a new side project building a CLI tool in Rust", "act4").await.unwrap();
        engine_tr.store("I prefer warm weather over cold weather", "noise7").await.unwrap();
        engine_tr.store("I competed in my first jiu-jitsu tournament last September", "comp1").await.unwrap();
        engine_tr.store("I've registered for the IBJJF San Francisco Open in March 2027", "comp2").await.unwrap();
        engine_tr.store("After that I want to try the Pan American championships in April 2027", "comp3").await.unwrap();
        engine_tr.store("My next trip is planned for Barcelona in April 2027", "trip1").await.unwrap();
        engine_tr.store("I should buy new running shoes soon", "noise8").await.unwrap();
    });

    println!("Running TR consolidation...");
    run_full_consolidation(&engine_tr);

    let tr_queries: Vec<(&str, &str, &[&str])> = vec![
        (
            "TR-1: Job Timeline",
            "Where have I worked over the years?",
            &["startup", "Shopify", "Stripe"],
        ),
        (
            "TR-2: Recent Events",
            "What tech events have I been involved in recently?",
            &["conference", "meetup", "RustConf"],
        ),
        (
            "TR-3: Piano Progress",
            "How is my piano playing going?",
            &["Chopin", "nocturne", "beginner", "piano"],
        ),
        (
            "TR-4: Graduation",
            "When did I finish university? What year?",
            &["2015", "graduated"],
        ),
        (
            "TR-5: Relationship",
            "How long have I been with my partner?",
            &["2019", "started dating", "anniversary"],
        ),
        (
            "TR-6: Pet Age",
            "How old is my dog now?",
            &["4 years old", "Pixel"],
        ),
        (
            "TR-7: Career Timing",
            "When did I start working at my current company?",
            &["2022", "Stripe"],
        ),
        (
            "TR-8: November Trip",
            "What did I do last November?",
            &["Tokyo", "November"],
        ),
        (
            "TR-9: Next Competition",
            "When is my next jiu-jitsu competition?",
            &["IBJJF", "San Francisco Open", "March 2027"],
        ),
    ];

    rt_ie.block_on(async {
        for (name, query, needles) in &tr_queries {
            let res = engine_tr.echo(query, 5).await.expect("echo");
            let h3 = needles.iter().any(|n| top_n_contains(&res, 3, n));
            let h5 = needles.iter().any(|n| top_n_contains(&res, 5, n));
            println!(
                "  {name}: top3={} top5={}",
                if h3 { "PASS" } else { "MISS" },
                if h5 { "PASS" } else { "MISS" }
            );
            all_results.push(TestResult {
                name,
                top3: h3,
                top5: h5,
            });
        }
    });
    drop(engine_tr);

    // =====================================================================
    // KU: Knowledge Update (9 tests) — ISOLATED engine per test
    // Each test gets ONLY its own old/new/noise memories to eliminate
    // cross-update contamination (KS28 Principle 1).
    // =====================================================================
    println!("\n--- Category 4: Knowledge Update (9 tests, isolated engines) ---");

    #[allow(clippy::type_complexity)]
    let ku_tests: Vec<(&str, &str, &[&str], Vec<(&str, &str)>)> = vec![
        (
            "KU-1: Job Change",
            "Where do I work now?",
            &["Meta"],
            vec![
                (
                    "I work as a backend engineer at Google on the Cloud Spanner team",
                    "old",
                ),
                (
                    "I left Google last month. I now work at Meta on the infrastructure team",
                    "new",
                ),
                ("I enjoy hiking on weekends in the bay area", "noise"),
            ],
        ),
        (
            "KU-2: Address",
            "Where do I live now?",
            &["Portland"],
            vec![
                (
                    "I live in a one-bedroom apartment in downtown Seattle",
                    "old",
                ),
                (
                    "I just moved to Portland, Oregon and I'm renting a house in the Pearl District",
                    "new",
                ),
            ],
        ),
        (
            "KU-3: Language",
            "What programming language do I mainly use?",
            &["Rust"],
            vec![
                (
                    "Python is my go-to programming language for everything",
                    "old",
                ),
                (
                    "I've switched from Python to Rust as my main language. The type system and performance are worth the learning curve",
                    "new",
                ),
            ],
        ),
        (
            "KU-4: Hobby Change",
            "What cardio exercise do I do?",
            &["swim", "YMCA"],
            vec![
                (
                    "I run 5 kilometers every morning before work, it's my main cardio exercise",
                    "old",
                ),
                (
                    "I enjoy listening to podcasts during my morning run",
                    "noise",
                ),
                (
                    "I had to stop running due to a knee injury. Now I swim laps at the YMCA pool three times a week instead",
                    "new",
                ),
            ],
        ),
        (
            "KU-5: Relationship",
            "What's my relationship status?",
            &["engaged", "wedding"],
            vec![
                (
                    "I've been dating Jordan for about 3 years, we live together in an apartment",
                    "old",
                ),
                ("I love cooking dinner with Jordan on weekends", "noise"),
                (
                    "Jordan and I got engaged last weekend! We're planning a small wedding for next spring",
                    "new",
                ),
            ],
        ),
        (
            "KU-6: Triple Address",
            "Where do I currently live?",
            &["San Francisco", "Mission District"],
            vec![
                (
                    "I live in a studio apartment in downtown Vancouver near Gastown",
                    "addr1",
                ),
                (
                    "I moved to Seattle for a new job, renting a place in Capitol Hill",
                    "addr2",
                ),
                (
                    "I relocated to San Francisco last month, now living in the Mission District",
                    "addr3",
                ),
                (
                    "I love exploring new neighborhoods and finding good coffee shops",
                    "noise",
                ),
            ],
        ),
        (
            "KU-7: Partial Update",
            "What's my current role and compensation?",
            &["staff engineer", "230k"],
            vec![
                (
                    "I work at Stripe as a senior engineer with a salary of $185k base plus equity",
                    "old",
                ),
                (
                    "I got promoted to staff engineer at Stripe with a salary increase to $230k base plus larger equity grant",
                    "new",
                ),
                (
                    "Tech salaries in the Bay Area have been going up this year",
                    "noise",
                ),
            ],
        ),
        (
            "KU-8: Skill Level",
            "What JLPT level did I reach?",
            &["N2"],
            vec![
                (
                    "I'm learning Japanese and just passed the JLPT N4 test",
                    "skill1",
                ),
                (
                    "I passed the JLPT N3 exam last December after months of studying",
                    "skill2",
                ),
                (
                    "I just got my JLPT N2 certification! Next goal is N1",
                    "skill3",
                ),
                (
                    "I want to visit Kyoto to practice my Japanese in a traditional setting",
                    "noise",
                ),
            ],
        ),
        (
            "KU-9: Diet Evolution",
            "What's my current diet? Do I eat meat?",
            &["vegan", "plant-based"],
            vec![
                (
                    "I eat everything, no restrictions. I love burgers and steaks",
                    "diet1",
                ),
                (
                    "I cut out red meat for health reasons, now I only eat chicken and fish",
                    "diet2",
                ),
                (
                    "I've gone fully plant-based vegan. No animal products at all, and I feel amazing",
                    "diet3",
                ),
                (
                    "I started a new workout routine focusing on strength training",
                    "noise",
                ),
            ],
        ),
    ];

    for (name, query, needles, memories) in &ku_tests {
        let dir_ku = tempdir().expect("temp dir");
        let engine_ku = EchoEngine::new(longmemeval_combined_config(dir_ku.path().to_path_buf()))
            .expect("init");
        rt_ie.block_on(async {
            for (text, src) in memories {
                engine_ku.store(text, src).await.unwrap();
            }
        });
        println!("  {name}: consolidating ({} memories)...", memories.len());
        run_full_consolidation(&engine_ku);
        let res = rt_ie.block_on(async { engine_ku.echo(query, 5).await.expect("echo") });
        let h3 = needles.iter().any(|n| top_n_contains(&res, 3, n));
        let h5 = needles.iter().any(|n| top_n_contains(&res, 5, n));
        println!(
            "  {name}: top3={} top5={}",
            if h3 { "PASS" } else { "MISS" },
            if h5 { "PASS" } else { "MISS" }
        );
        all_results.push(TestResult {
            name,
            top3: h3,
            top5: h5,
        });
        drop(engine_ku);
        drop(dir_ku);
    }

    // =====================================================================
    // PT: Preference Tracking (10 tests) — ISOLATED engine per test
    // Each test gets ONLY its own evolution memories + noise to eliminate
    // cross-preference contamination (KS28 Principle 1).
    // =====================================================================
    println!("\n--- Category 5: Preference Tracking (10 tests, isolated engines) ---");

    #[allow(clippy::type_complexity)]
    let pt_tests: Vec<(&str, &str, &[&str], Vec<(&str, &str)>)> = vec![
        (
            "PT-1: IDE",
            "What code editor do I use?",
            &["Neovim"],
            vec![
                (
                    "I use Sublime Text as my code editor, it's fast and lightweight",
                    "m1",
                ),
                (
                    "I switched to VS Code because of the extension ecosystem",
                    "m3",
                ),
                (
                    "I've moved to Neovim with a custom Lua config for maximum speed",
                    "m6",
                ),
            ],
        ),
        (
            "PT-2: Diet",
            "What's my diet like?",
            &["pescatarian"],
            vec![
                ("I'm vegetarian and have been for the past 3 years", "m1"),
                ("I started eating fish again, so now I'm pescatarian", "m4"),
            ],
        ),
        (
            "PT-3: Coffee",
            "How do I take my coffee?",
            &["pour-over", "V60", "black coffee"],
            vec![
                ("I drink regular drip coffee with cream and sugar", "m1"),
                (
                    "I switched to espresso-based drinks, usually a latte with whole milk",
                    "m3",
                ),
                (
                    "Now I drink pour-over black coffee, no milk no sugar, using a Hario V60",
                    "m6",
                ),
            ],
        ),
        (
            "PT-4: OS",
            "What operating system do I use?",
            &["Arch", "Hyprland"],
            vec![
                (
                    "I use Windows 11 on all my machines for gaming and development",
                    "m1",
                ),
                (
                    "I dual-boot Linux Mint alongside Windows now for dev work",
                    "m3",
                ),
                (
                    "I've gone all-in on Arch Linux with Hyprland compositor, retired Windows completely",
                    "m6",
                ),
            ],
        ),
        (
            "PT-5: Music",
            "What kind of music do I listen to?",
            &["lo-fi", "ambient", "electronic"],
            vec![
                ("I mostly listen to pop music and top 40 hits", "m1"),
                (
                    "I've been getting into indie rock and alternative music lately",
                    "m3",
                ),
                (
                    "Now I mainly listen to lo-fi hip hop and ambient electronic music for focus",
                    "m6",
                ),
                ("I bought new headphones, the Sony WH-1000XM5", "noise"),
            ],
        ),
        (
            "PT-6: Commute",
            "How do I get to work?",
            &["bike"],
            vec![
                (
                    "I drive my car to work every day, about a 40-minute commute on the highway",
                    "m1",
                ),
                (
                    "I started taking the BART train to reduce my carbon footprint",
                    "m4",
                ),
                (
                    "I now bike to work every day, it's a 25-minute ride and I love the fresh air",
                    "m8",
                ),
                ("Gas prices have been going up a lot this year", "noise"),
            ],
        ),
        (
            "PT-7: Desk Setup",
            "What's my desk and keyboard setup?",
            &["Uplift", "Kinesis", "standing desk"],
            vec![
                (
                    "I work at a regular sitting desk with a standard Dell keyboard and mouse",
                    "m1",
                ),
                (
                    "I got a sit-stand desk converter for my existing desk, and a mechanical keyboard",
                    "m4",
                ),
                (
                    "I upgraded to a full Uplift standing desk and a split ergonomic Kinesis Advantage 360",
                    "m8",
                ),
                ("I need to clean my office this weekend", "noise"),
            ],
        ),
        (
            "PT-8: Learning",
            "What am I currently learning or studying?",
            &["AWS", "Solutions Architect"],
            vec![
                (
                    "I'm taking an online course on machine learning fundamentals",
                    "m1",
                ),
                (
                    "I finished the ML course and started studying distributed systems design",
                    "m3",
                ),
                (
                    "I'm currently studying for the AWS Solutions Architect certification exam",
                    "m6",
                ),
                (
                    "I find that studying in the morning works best for me",
                    "noise",
                ),
            ],
        ),
        (
            "PT-9: Communities",
            "What online communities am I active on?",
            &["Rust subreddit", "Hacker News"],
            vec![
                ("I'm mostly on Twitter for tech discussions and memes", "m1"),
                (
                    "I left Twitter and moved to Mastodon for the tech community",
                    "m4",
                ),
                (
                    "I'm now most active on the Rust subreddit and Hacker News, and I'm a member of the San Francisco Rust meetup",
                    "m8",
                ),
                (
                    "Social media can be really distracting during work hours",
                    "noise",
                ),
            ],
        ),
        (
            "PT-10: Cooking",
            "What do I like to cook?",
            &["Thai", "Japanese", "pad see ew", "ramen"],
            vec![
                (
                    "I mostly order takeout and eat at restaurants, I don't cook much",
                    "m1",
                ),
                (
                    "I started meal prepping on Sundays, mostly simple pasta and salad recipes",
                    "m4",
                ),
                (
                    "I've gotten into Thai and Japanese home cooking, I make pad see ew and ramen from scratch now",
                    "m8",
                ),
                ("I need to buy a new set of kitchen knives", "noise"),
            ],
        ),
    ];

    for (name, query, needles, memories) in &pt_tests {
        let dir_pt = tempdir().expect("temp dir");
        let engine_pt = EchoEngine::new(longmemeval_combined_config(dir_pt.path().to_path_buf()))
            .expect("init");
        rt_ie.block_on(async {
            for (text, src) in memories {
                engine_pt.store(text, src).await.unwrap();
            }
        });
        println!("  {name}: consolidating ({} memories)...", memories.len());
        run_full_consolidation(&engine_pt);
        let res = rt_ie.block_on(async { engine_pt.echo(query, 5).await.expect("echo") });
        let h3 = needles.iter().any(|n| top_n_contains(&res, 3, n));
        let h5 = needles.iter().any(|n| top_n_contains(&res, 5, n));
        println!(
            "  {name}: top3={} top5={}",
            if h3 { "PASS" } else { "MISS" },
            if h5 { "PASS" } else { "MISS" }
        );
        all_results.push(TestResult {
            name,
            top3: h3,
            top5: h5,
        });
        drop(engine_pt);
        drop(dir_pt);
    }

    drop(rt_ie);

    // =====================================================================
    // Final Scorecard
    // =====================================================================
    let total = all_results.len();
    let strict = all_results.iter().filter(|r| r.top3).count();
    let relaxed = all_results.iter().filter(|r| r.top5).count();
    let strict_pct = (strict as f64 / total as f64) * 100.0;
    let relaxed_pct = (relaxed as f64 / total as f64) * 100.0;

    println!("\n============================================================");
    println!("=== EXPANDED COMBINED LONGMEMEVAL SCORECARD (50 tests) ===");
    println!("============================================================");
    println!();
    println!("{:<30} | {:>9} | {:>9}", "Category", "Top-3", "Top-5");
    println!("{}", "-".repeat(55));

    // Per-category breakdown
    let categories = [
        ("IE (Info Extraction)", 0..11),
        ("MSR (Multi-Session)", 11..22),
        ("TR (Temporal Reasoning)", 22..31),
        ("KU (Knowledge Update)", 31..40),
        ("PT (Preference Tracking)", 40..50),
    ];

    for (cat_name, range) in &categories {
        let cat_results = &all_results[range.clone()];
        let cat_total = cat_results.len();
        let cat_s = cat_results.iter().filter(|r| r.top3).count();
        let cat_r = cat_results.iter().filter(|r| r.top5).count();
        println!(
            "{:<30} | {:>4}/{:<4} | {:>4}/{:<4}",
            cat_name, cat_s, cat_total, cat_r, cat_total
        );
    }

    println!("{}", "-".repeat(55));
    println!(
        "{:<30} | {:>4}/{:<4} | {:>4}/{:<4}",
        "TOTAL", strict, total, relaxed, total
    );
    println!();
    println!("Top-3 accuracy:   {strict}/{total} ({strict_pct:.1}%)");
    println!("Top-5 accuracy:   {relaxed}/{total} ({relaxed_pct:.1}%)");
    println!();
    println!("=== EXPANDED BENCHMARK COMPLETE ===");
}
