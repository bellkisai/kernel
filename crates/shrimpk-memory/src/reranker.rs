//! Reranker backends for Echo results.
//!
//! After cosine similarity + Hebbian boost produces the top-N results,
//! the reranker reorders them by true semantic relevance. Two backends:
//!
//! - **LLM** (KS23): Sends results to a local LLM via Ollama (~2s latency).
//! - **Cross-encoder** (KS24 Track 3): Uses fastembed's `TextRerank` ONNX model
//!   (~5-15ms latency). Same local inference infrastructure as the embedder.
//!
//! The reranker is opt-in (`reranker_backend: "cross_encoder"` or `"llm"` in config)
//! and gracefully falls back to original ordering if the backend call fails.

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use shrimpk_core::{EchoConfig, EchoResult, RerankerBackend};
use std::collections::HashSet;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Lazy-initialized cross-encoder singleton
// ---------------------------------------------------------------------------

/// The cross-encoder model we use. JINARerankerV1TurboEn is the fastest option
/// in the fastembed model zoo (~33MB ONNX, English-optimized, low latency).
const RERANKER_MODEL: RerankerModel = RerankerModel::JINARerankerV1TurboEn;

/// Thread-safe lazy singleton for the fastembed cross-encoder model.
/// The model (~33MB jina-reranker-v1-turbo-en) is downloaded on first use
/// and cached in the fastembed model cache directory.
/// We keep it alive for the process lifetime to avoid repeated init costs.
static CROSS_ENCODER: std::sync::LazyLock<Mutex<Option<TextRerank>>> =
    std::sync::LazyLock::new(|| {
        let start = std::time::Instant::now();
        let opts = RerankInitOptions::new(RERANKER_MODEL);
        match TextRerank::try_new(opts) {
            Ok(model) => {
                let elapsed = start.elapsed();
                tracing::info!(
                    elapsed_ms = elapsed.as_millis(),
                    model = "jina-reranker-v1-turbo-en",
                    "Cross-encoder reranker initialized (fastembed)"
                );
                Mutex::new(Some(model))
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Cross-encoder reranker failed to initialize, will fall back to original order"
                );
                Mutex::new(None)
            }
        }
    });

// ---------------------------------------------------------------------------
// Public dispatch function
// ---------------------------------------------------------------------------

/// Rerank echo results using the configured backend.
///
/// Uses `EchoConfig::effective_reranker_backend()` to resolve backward-compatible
/// backend selection (respects both `reranker_backend` and legacy `reranker_enabled`).
///
/// Returns `Some(reranked_results)` on success, `None` on failure or if disabled.
/// The caller should keep the original ordering when `None` is returned.
pub fn rerank(
    config: &EchoConfig,
    query: &str,
    results: &[EchoResult],
) -> Option<Vec<EchoResult>> {
    match config.effective_reranker_backend() {
        RerankerBackend::None => None,
        RerankerBackend::Llm => rerank_with_llm(config, query, results),
        RerankerBackend::CrossEncoder => rerank_with_cross_encoder(query, results),
    }
}

// ---------------------------------------------------------------------------
// Cross-encoder backend (KS24 Track 3)
// ---------------------------------------------------------------------------

/// Rerank echo results using a fastembed cross-encoder model.
///
/// Uses `TextRerank` with the default model (jina-reranker-v1-turbo-en, ~33MB ONNX).
/// The cross-encoder scores each (query, document) pair directly, producing much
/// more accurate relevance scores than cosine similarity of independent embeddings.
///
/// Typical latency: 5-15ms for 10 documents (vs ~2s for LLM reranker).
///
/// Returns `Some(reranked_results)` on success, `None` on failure.
pub fn rerank_with_cross_encoder(
    query: &str,
    results: &[EchoResult],
) -> Option<Vec<EchoResult>> {
    if results.is_empty() {
        return None;
    }

    let rerank_count = results.len().min(10);

    // Extract document strings for the cross-encoder.
    // We use &str slices to match the query type for the generic `rerank<S>` method.
    let documents: Vec<&str> = results
        .iter()
        .take(rerank_count)
        .map(|r| r.content.as_str())
        .collect();

    let start = std::time::Instant::now();

    // Acquire the cross-encoder model
    let mut guard = match CROSS_ENCODER.lock() {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                target: "shrimpk::audit",
                error = %e,
                "Cross-encoder: mutex poisoned, keeping original order"
            );
            return None;
        }
    };

    let model = match guard.as_mut() {
        Some(m) => m,
        None => {
            tracing::warn!(
                target: "shrimpk::audit",
                "Cross-encoder: model not available, keeping original order"
            );
            return None;
        }
    };

    tracing::debug!(
        target: "shrimpk::audit",
        query = query,
        rerank_count = rerank_count,
        "Cross-encoder: reranking top-{} results",
        rerank_count,
    );

    // Call the cross-encoder: returns Vec<RerankResult> sorted by score descending
    let rerank_results = match model.rerank(query, documents.as_slice(), false, None) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                target: "shrimpk::audit",
                error = %e,
                "Cross-encoder: rerank failed, keeping original order"
            );
            return None;
        }
    };

    let elapsed_ms = start.elapsed().as_millis();

    // Map cross-encoder output back to EchoResults.
    // RerankResult.index is the original position in `documents`, and results
    // come pre-sorted by score descending.
    let mut reranked: Vec<EchoResult> = Vec::with_capacity(results.len());
    let mut seen = HashSet::new();
    for rr in &rerank_results {
        let idx = rr.index;
        if seen.insert(idx) && idx < results.len() {
            reranked.push(results[idx].clone());
        }
    }
    // Append any results not covered by reranking (preserving original order)
    for (i, r) in results.iter().enumerate() {
        if !seen.contains(&i) {
            reranked.push(r.clone());
        }
    }

    tracing::info!(
        target: "shrimpk::audit",
        reranked_count = reranked.len(),
        elapsed_ms = elapsed_ms,
        top_score = rerank_results.first().map(|r| r.score).unwrap_or(0.0),
        "Cross-encoder: reordered {} results in {}ms",
        reranked.len(),
        elapsed_ms,
    );

    Some(reranked)
}

// ---------------------------------------------------------------------------
// LLM backend (KS23 Track 3) — original Ollama-based reranker
// ---------------------------------------------------------------------------

/// Ask the LLM to reorder the top-N echo results by true relevance.
///
/// Returns `Some(reranked_results)` on success, `None` on failure
/// (timeout, parse error, Ollama unreachable). The caller should
/// keep the original ordering when `None` is returned.
pub fn rerank_with_llm(
    config: &EchoConfig,
    query: &str,
    results: &[EchoResult],
) -> Option<Vec<EchoResult>> {
    if results.is_empty() {
        return None;
    }

    let rerank_count = results.len().min(10);

    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(std::time::Duration::from_secs(15)))
            .build(),
    );

    // Format memories for the LLM -- truncate each to 100 chars for token efficiency
    let memories: String = results
        .iter()
        .enumerate()
        .take(rerank_count)
        .map(|(i, r)| format!("{}. {}", i + 1, &r.content[..r.content.len().min(100)]))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Given this query and numbered memories, output ONLY the numbers in order of relevance \
         (most relevant first). Output numbers separated by commas, nothing else.\n\n\
         Query: {query}\n\nMemories:\n{memories}"
    );

    let body = serde_json::json!({
        "model": config.enrichment_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a relevance ranker. Output ONLY comma-separated numbers, \
                    most relevant first. Consider recency — if multiple memories describe the \
                    same topic, the MOST RECENT one should rank highest."
            },
            {"role": "user", "content": prompt}
        ],
        "stream": false,
        "options": {"temperature": 0.0, "num_predict": 64}
    });

    let endpoint = format!("{}/api/chat", config.ollama_url.trim_end_matches('/'));

    tracing::debug!(
        target: "shrimpk::audit",
        query = query,
        rerank_count = rerank_count,
        model = %config.enrichment_model,
        "Reranker: sending top-{} results to LLM",
        rerank_count,
    );

    let start = std::time::Instant::now();
    let mut resp = match agent.post(&endpoint).send_json(&body) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                target: "shrimpk::audit",
                error = %e,
                "Reranker: Ollama unreachable, keeping original order"
            );
            return None;
        }
    };

    let json: serde_json::Value = match resp.body_mut().read_json() {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!(
                target: "shrimpk::audit",
                error = %e,
                "Reranker: failed to parse LLM response, keeping original order"
            );
            return None;
        }
    };

    let content = json["message"]["content"].as_str()?;
    let elapsed_ms = start.elapsed().as_millis();

    tracing::debug!(
        target: "shrimpk::audit",
        raw_response = content,
        elapsed_ms = elapsed_ms,
        "Reranker: LLM response received"
    );

    // Parse the ranking: "3, 1, 5, 2, 4" -> [3, 1, 5, 2, 4]
    let indices: Vec<usize> = content
        .split(|c: char| c == ',' || c == ' ' || c == '\n')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|&i| i >= 1 && i <= results.len())
        .map(|i| i - 1) // 1-indexed to 0-indexed
        .collect();

    if indices.is_empty() {
        tracing::warn!(
            target: "shrimpk::audit",
            raw = content,
            "Reranker: failed to parse indices from LLM response, keeping original order"
        );
        return None;
    }

    // Reorder results based on LLM ranking
    let mut reranked: Vec<EchoResult> = Vec::with_capacity(results.len());
    let mut seen = HashSet::new();
    for idx in indices {
        if seen.insert(idx) && idx < results.len() {
            reranked.push(results[idx].clone());
        }
    }
    // Append any results the LLM didn't rank (preserving original order)
    for (i, r) in results.iter().enumerate() {
        if !seen.contains(&i) {
            reranked.push(r.clone());
        }
    }

    tracing::info!(
        target: "shrimpk::audit",
        reranked_count = reranked.len(),
        elapsed_ms = elapsed_ms,
        "Reranker: reordered {} results in {}ms",
        reranked.len(),
        elapsed_ms,
    );

    Some(reranked)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use shrimpk_core::EchoConfig;

    fn make_result(_id: &str, content: &str, score: f64) -> EchoResult {
        EchoResult {
            memory_id: shrimpk_core::MemoryId::new(),
            content: content.to_string(),
            similarity: score as f32,
            final_score: score,
            source: "test".to_string(),
            echoed_at: Utc::now(),
        }
    }

    #[test]
    fn rerank_with_cross_encoder_empty_returns_none() {
        let results: Vec<EchoResult> = vec![];
        assert!(rerank_with_cross_encoder("test query", &results).is_none());
    }

    #[test]
    fn rerank_dispatch_none_backend_returns_none() {
        let config = EchoConfig {
            reranker_backend: RerankerBackend::None,
            ..Default::default()
        };
        let results = vec![make_result("1", "hello world", 0.9)];
        assert!(rerank(&config, "hello", &results).is_none());
    }

    #[test]
    fn reranker_backend_roundtrip() {
        for backend in [
            RerankerBackend::None,
            RerankerBackend::Llm,
            RerankerBackend::CrossEncoder,
        ] {
            let s = backend.to_string();
            let parsed: RerankerBackend = s.parse().unwrap();
            assert_eq!(backend, parsed);
        }
    }

    #[test]
    fn reranker_backend_parse_aliases() {
        assert_eq!(
            "ollama".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::Llm
        );
        assert_eq!(
            "ce".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::CrossEncoder
        );
        assert_eq!(
            "fastembed".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::CrossEncoder
        );
        assert_eq!(
            "disabled".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::None
        );
        assert_eq!(
            "off".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::None
        );
    }

    #[test]
    fn reranker_backend_parse_invalid() {
        assert!("unknown".parse::<RerankerBackend>().is_err());
    }

    #[test]
    #[ignore = "requires fastembed reranker model download (~33MB)"]
    fn cross_encoder_reranks_results() {
        let results = vec![
            make_result("1", "I use Sublime Text for editing code", 0.8),
            make_result("2", "My preferred editor is Neovim with LSP", 0.75),
            make_result("3", "The weather today is sunny and warm", 0.7),
        ];

        let reranked = rerank_with_cross_encoder("What text editor do you prefer?", &results);
        assert!(reranked.is_some());
        let reranked = reranked.unwrap();

        // All results should be present
        assert_eq!(reranked.len(), 3);

        // The weather result should be ranked last since it's irrelevant
        let weather_pos = reranked
            .iter()
            .position(|r| r.content.contains("weather"))
            .unwrap();
        assert_eq!(
            weather_pos, 2,
            "Weather result should be ranked last, was at position {}",
            weather_pos
        );
    }

    #[test]
    #[ignore = "requires fastembed reranker model download (~33MB)"]
    fn cross_encoder_preserves_all_results() {
        let results: Vec<EchoResult> = (0..5)
            .map(|i| make_result(&i.to_string(), &format!("Memory content number {i}"), 0.5))
            .collect();

        let reranked = rerank_with_cross_encoder("memory content", &results);
        assert!(reranked.is_some());
        let reranked = reranked.unwrap();
        assert_eq!(reranked.len(), 5, "All results should be preserved");

        // Check no duplicates
        let ids: HashSet<_> = reranked.iter().map(|r| &r.memory_id).collect();
        assert_eq!(ids.len(), 5, "No duplicate results");
    }

    #[test]
    fn rerank_llm_empty_returns_none() {
        let config = EchoConfig::default();
        let results: Vec<EchoResult> = vec![];
        assert!(rerank_with_llm(&config, "test query", &results).is_none());
    }
}
