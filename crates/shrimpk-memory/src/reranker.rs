//! LLM-based reranker for Echo results (KS23 Track 3).
//!
//! After cosine similarity + Hebbian boost produces the top-N results,
//! the reranker sends them to a local LLM (via Ollama) to reorder by
//! true semantic relevance. This helps cases where keyword overlap
//! misleads cosine ranking (e.g., PT-1: Sublime outranks Neovim).
//!
//! The reranker is opt-in (`reranker_enabled: true` in config) and
//! gracefully falls back to original ordering if the LLM call fails.

use shrimpk_core::{EchoConfig, EchoResult};
use std::collections::HashSet;

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

    // Format memories for the LLM — truncate each to 100 chars for token efficiency
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
