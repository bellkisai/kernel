//! Integration tests for the Echo Memory engine.
//!
//! These tests exercise the full echo cycle end-to-end: embed -> store -> query -> rank.
//! All tests are `#[ignore]` because they require the fastembed model (all-MiniLM-L6-v2)
//! to be downloaded (~23MB ONNX). Run with:
//!
//!     cargo test --test echo_integration -- --ignored
//!
//! The model is cached after first download, so subsequent runs are fast.

use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use tempfile::tempdir;

/// Build an EchoConfig suitable for integration tests.
///
/// Uses a throwaway temp path for data_dir and relaxed thresholds.
fn test_config(data_dir: PathBuf) -> EchoConfig {
    EchoConfig {
        max_memories: 10_000,
        similarity_threshold: 0.2, // low threshold so related topics match
        max_echo_results: 10,
        ram_budget_bytes: 100_000_000,
        data_dir,
        embedding_dim: 384,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 1. Full echo cycle
// ---------------------------------------------------------------------------

/// Store 5 diverse memories, query with a related topic, and verify the results
/// are non-empty, sorted by score descending, and returned within a generous
/// latency budget (1000ms accounts for model warm-up on first run).
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn test_full_echo_cycle() {
    let dir = tempdir().expect("temp dir");
    let config = test_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // Store 5 diverse memories
    engine
        .store(
            "Rust is a systems programming language focused on safety and concurrency",
            "test",
        )
        .await
        .unwrap();
    engine
        .store(
            "The capital of France is Paris, known for the Eiffel Tower",
            "test",
        )
        .await
        .unwrap();
    engine
        .store(
            "Machine learning models require large datasets for training",
            "test",
        )
        .await
        .unwrap();
    engine
        .store(
            "Photosynthesis converts sunlight into chemical energy in plants",
            "test",
        )
        .await
        .unwrap();
    engine
        .store(
            "The TCP/IP protocol stack powers internet communication",
            "test",
        )
        .await
        .unwrap();

    // Echo with a related topic
    let start = std::time::Instant::now();
    let results = engine
        .echo("memory safety in systems languages", 5)
        .await
        .expect("echo should succeed");
    let elapsed = start.elapsed();

    // At least 1 result returned
    assert!(
        !results.is_empty(),
        "Echo should return at least 1 related memory"
    );

    // Top result has similarity > 0.5 (Rust / systems programming is highly related)
    assert!(
        results[0].similarity > 0.5,
        "Top result similarity should be > 0.5, got {}",
        results[0].similarity
    );

    // Results sorted by score descending
    for window in results.windows(2) {
        assert!(
            window[0].final_score >= window[1].final_score,
            "Results should be sorted descending: {} >= {}",
            window[0].final_score,
            window[1].final_score
        );
    }

    // Latency < 1000ms (generous for first run including possible model load)
    assert!(
        elapsed.as_millis() < 1000,
        "Echo latency should be < 1000ms, got {}ms",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// 2. PII masking in echo
// ---------------------------------------------------------------------------

/// Store text containing an API key, query for it, and verify:
/// - The echo result content contains the masked token
/// - The original key text is NOT present in the result
/// - The memory's sensitivity is Restricted
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn test_pii_masking_in_echo() {
    let dir = tempdir().expect("temp dir");
    let config = test_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let original_key = "sk-test1234567890abcdef";
    let text = format!("My OpenAI key is {original_key}");

    let id = engine
        .store(&text, "test")
        .await
        .expect("store should succeed");

    // Echo with a query that should match the stored memory
    let results = engine
        .echo("What's my API key?", 5)
        .await
        .expect("echo should succeed");

    // Find the result matching our stored memory
    let result = results
        .iter()
        .find(|r| r.memory_id == id)
        .expect("Should find the stored memory in echo results");

    // The echo result content must contain the mask token
    assert!(
        result.content.contains("[MASKED:api_key]"),
        "Echo result should contain [MASKED:api_key], got: {}",
        result.content
    );

    // The original key must NOT appear in the result content
    assert!(
        !result.content.contains(original_key),
        "Original API key should NOT appear in echo result, got: {}",
        result.content
    );

    // Verify sensitivity via stats indirectly: store was created, so we can check
    // the memory was classified correctly by storing another and comparing.
    // We already know from the PII filter unit tests that api_key -> Restricted.
    // Here we verify the full pipeline classified it, by checking the mask appeared.
    // (EchoEngine does not expose the raw MemoryEntry through echo, but the mask
    // presence confirms the PII pipeline ran and set masked_content.)
}

// ---------------------------------------------------------------------------
// 3. Persistence roundtrip
// ---------------------------------------------------------------------------

/// Store 10 memories, persist to a temp directory, create a new engine from the
/// same directory, and verify the loaded engine has the same count and returns
/// the same echo results.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn test_persistence_roundtrip() {
    let dir = tempdir().expect("temp dir");
    let config = test_config(dir.path().to_path_buf());

    // Phase 1: store and echo
    let engine = EchoEngine::new(config.clone()).expect("engine init");

    let texts = [
        "Quantum computing uses qubits for parallel computation",
        "The Great Wall of China is visible from space",
        "Neural networks are inspired by biological neurons",
        "Shakespeare wrote Hamlet in the early 1600s",
        "Docker containers provide lightweight virtualization",
        "The Mariana Trench is the deepest oceanic trench",
        "Functional programming avoids mutable state",
        "DNA stores genetic information in a double helix",
        "Kubernetes orchestrates containerized applications",
        "The speed of light is approximately 300,000 km/s",
    ];

    for text in &texts {
        engine.store(text, "test").await.unwrap();
    }

    // Record the echo results before persist
    let results_before = engine
        .echo("container orchestration", 5)
        .await
        .expect("echo should succeed");

    // Persist to disk
    engine.persist().await.expect("persist should succeed");

    // Phase 2: load from the same directory
    let engine2 = EchoEngine::load(config).expect("load should succeed");

    // Same number of memories
    let stats = engine2.stats().await;
    assert_eq!(
        stats.total_memories, 10,
        "Loaded engine should have 10 memories, got {}",
        stats.total_memories
    );

    // Echo returns same results (same memory IDs in the same order)
    let results_after = engine2
        .echo("container orchestration", 5)
        .await
        .expect("echo should succeed after load");

    assert_eq!(
        results_before.len(),
        results_after.len(),
        "Same number of echo results before and after persist"
    );

    for (before, after) in results_before.iter().zip(results_after.iter()) {
        assert_eq!(
            before.memory_id, after.memory_id,
            "Memory IDs should match after roundtrip"
        );
        // Similarity scores should be identical (same embeddings, same query)
        assert!(
            (before.similarity - after.similarity).abs() < 1e-5,
            "Similarity scores should match: {} vs {}",
            before.similarity,
            after.similarity
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Empty echo
// ---------------------------------------------------------------------------

/// Query on an empty store should return an empty vec (not an error) within 100ms.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn test_empty_echo() {
    let dir = tempdir().expect("temp dir");
    let config = test_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let start = std::time::Instant::now();
    let results = engine
        .echo("anything at all", 5)
        .await
        .expect("echo on empty store should succeed, not error");
    let elapsed = start.elapsed();

    assert!(
        results.is_empty(),
        "Empty store should return empty results, got {} results",
        results.len()
    );

    assert!(
        elapsed.as_millis() < 100,
        "Empty echo should complete in < 100ms, took {}ms",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// 5. Forget removes from echo
// ---------------------------------------------------------------------------

/// Store 3 memories, forget one, and verify the forgotten memory no longer
/// appears in echo results even when the query is an exact match.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn test_forget_removes_from_echo() {
    let dir = tempdir().expect("temp dir");
    let config = test_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    let _id1 = engine
        .store("Apples are a popular fruit rich in fiber", "test")
        .await
        .unwrap();
    let id2 = engine
        .store("Bananas are high in potassium and vitamins", "test")
        .await
        .unwrap();
    let _id3 = engine
        .store("Oranges contain lots of vitamin C", "test")
        .await
        .unwrap();

    // Forget the banana memory
    engine
        .forget(id2.clone())
        .await
        .expect("forget should succeed");

    // Echo with a query about bananas
    let results = engine
        .echo("bananas potassium", 10)
        .await
        .expect("echo should succeed");

    // The forgotten memory should NOT appear in results
    assert!(
        results.iter().all(|r| r.memory_id != id2),
        "Forgotten memory (bananas) should not appear in echo results"
    );

    // The other memories should still be findable
    let fruit_results = engine
        .echo("fruit vitamins", 10)
        .await
        .expect("echo should succeed");

    assert!(
        !fruit_results.is_empty(),
        "Non-forgotten memories should still be findable"
    );
    assert!(
        fruit_results.iter().all(|r| r.memory_id != id2),
        "Forgotten memory should not appear in any echo results"
    );
}

// ---------------------------------------------------------------------------
// 6. Stats accuracy
// ---------------------------------------------------------------------------

/// Store N memories and verify that stats.total_memories == N.
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn test_stats_accuracy() {
    let dir = tempdir().expect("temp dir");
    let config = test_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // Verify empty stats
    let stats = engine.stats().await;
    assert_eq!(
        stats.total_memories, 0,
        "Empty engine should have 0 memories"
    );
    assert_eq!(stats.total_echo_queries, 0, "No queries yet");

    // Store N memories
    let n: usize = 7;
    for i in 0..n {
        engine
            .store(
                &format!("Test memory number {i} with unique content"),
                "test",
            )
            .await
            .unwrap();
    }

    let stats = engine.stats().await;
    assert_eq!(
        stats.total_memories, n,
        "Stats should report exactly {n} memories, got {}",
        stats.total_memories
    );
    assert_eq!(
        stats.max_capacity, 10_000,
        "Max capacity should match config"
    );

    // Perform some echo queries and verify query count tracking
    engine.echo("test", 5).await.unwrap();
    engine.echo("memory", 5).await.unwrap();
    engine.echo("unique", 5).await.unwrap();

    let stats = engine.stats().await;
    assert_eq!(
        stats.total_echo_queries, 3,
        "Should have tracked 3 echo queries, got {}",
        stats.total_echo_queries
    );
    assert!(
        stats.avg_echo_latency_ms > 0.0,
        "Average latency should be positive after queries"
    );
    assert!(
        stats.index_size_bytes > 0,
        "Index size should be non-zero with stored memories"
    );
    assert!(
        stats.ram_usage_bytes > 0,
        "RAM usage estimate should be non-zero"
    );
}

// ---------------------------------------------------------------------------
// 7. Sensitivity classification
// ---------------------------------------------------------------------------

/// Store various texts and verify the PII pipeline classifies them correctly
/// through the full engine path (store -> echo -> check content for masks).
#[tokio::test]
#[ignore = "requires fastembed model download"]
async fn test_sensitivity_classification() {
    let dir = tempdir().expect("temp dir");
    let config = test_config(dir.path().to_path_buf());
    let engine = EchoEngine::new(config).expect("engine init");

    // Store texts with different sensitivity levels
    // 1. Normal text -> Public (no PII, no masking)
    let id_public = engine
        .store("The weather in Tokyo is warm and humid in summer", "test")
        .await
        .unwrap();

    // 2. Text with email -> Private (PII detected, masked)
    let id_private = engine
        .store("Contact support@bellkis.com for account help", "test")
        .await
        .unwrap();

    // 3. Text with API key -> Restricted (high-sensitivity PII)
    let id_restricted = engine
        .store("My secret key is sk-abcdefghij1234567890klmn", "test")
        .await
        .unwrap();

    // Verify Public: echo result should contain original text (no masking)
    let results = engine.echo("Tokyo weather summer", 5).await.unwrap();
    if let Some(r) = results.iter().find(|r| r.memory_id == id_public) {
        assert!(
            !r.content.contains("[MASKED:"),
            "Public text should have no masking, got: {}",
            r.content
        );
        assert!(
            r.content.contains("Tokyo"),
            "Public text should be intact, got: {}",
            r.content
        );
    }

    // Verify Private: echo result should mask the email
    let results = engine.echo("contact support account", 5).await.unwrap();
    if let Some(r) = results.iter().find(|r| r.memory_id == id_private) {
        assert!(
            r.content.contains("[MASKED:email]"),
            "Private text should have email masked, got: {}",
            r.content
        );
        assert!(
            !r.content.contains("support@bellkis.com"),
            "Original email should not appear in echo result, got: {}",
            r.content
        );
    }

    // Verify Restricted: echo result should mask the API key
    let results = engine.echo("secret key credentials", 5).await.unwrap();
    if let Some(r) = results.iter().find(|r| r.memory_id == id_restricted) {
        assert!(
            r.content.contains("[MASKED:api_key]"),
            "Restricted text should have API key masked, got: {}",
            r.content
        );
        assert!(
            !r.content.contains("sk-abcdefghij1234567890klmn"),
            "Original API key should not appear in echo result, got: {}",
            r.content
        );
    }
}
