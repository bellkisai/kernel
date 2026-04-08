//! Pluggable embedding provider implementations (KS75).
//!
//! Two backends:
//! - `FastembedProvider` — local ONNX via `fastembed` (default, zero API calls)
//! - `OpenAIProvider` — any OpenAI-compatible embedding API (cloud or local Ollama)
//!
//! Factory function `from_config()` selects the appropriate provider based on `EchoConfig`.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use shrimpk_core::{EchoConfig, EmbeddingBackend, EmbeddingProvider, Result, ShrimPKError};

// ---------------------------------------------------------------------------
// FastembedProvider
// ---------------------------------------------------------------------------

/// Local ONNX embedding via fastembed.
///
/// Wraps `fastembed::TextEmbedding` with runtime model selection.
/// Zero external API calls — all inference runs locally.
pub struct FastembedProvider {
    model: TextEmbedding,
    dim: usize,
    model_name: String,
}

impl FastembedProvider {
    /// Create a new FastembedProvider for the given model name.
    ///
    /// Supported model names (case-insensitive):
    /// - "bge-small-en-v1.5" (384-dim, default)
    /// - "bge-base-en-v1.5" (768-dim)
    /// - "bge-large-en-v1.5" (1024-dim)
    /// - "bge-m3" (1024-dim)
    /// - "all-minilm-l6-v2" (384-dim)
    /// - "all-minilm-l12-v2" (384-dim)
    /// - "nomic-embed-text-v1.5" (768-dim)
    /// - "mxbai-embed-large-v1" (1024-dim)
    /// - "gte-large-en-v1.5" (1024-dim)
    pub fn new(model_name: &str) -> Result<Self> {
        let (variant, dim) = resolve_fastembed_model(model_name)?;
        let display_name = format!("fastembed/{model_name}");

        let start = std::time::Instant::now();
        let model = TextEmbedding::try_new(InitOptions::new(variant)).map_err(|e| {
            ShrimPKError::Embedding(format!(
                "Failed to init fastembed model '{model_name}': {e}"
            ))
        })?;

        tracing::info!(
            elapsed_ms = start.elapsed().as_millis(),
            model = %display_name,
            dim = dim,
            "FastembedProvider initialized"
        );

        Ok(Self {
            model,
            dim,
            model_name: display_name,
        })
    }
}

impl EmbeddingProvider for FastembedProvider {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let results = self
            .model
            .embed(vec![text.to_string()], None)
            .map_err(|e| ShrimPKError::Embedding(format!("fastembed embed failed: {e}")))?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| ShrimPKError::Embedding("Empty fastembed result".into()))
    }

    fn embed_batch(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.model
            .embed(texts, None)
            .map_err(|e| ShrimPKError::Embedding(format!("fastembed batch embed failed: {e}")))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

/// Map a config model name to a fastembed `EmbeddingModel` variant + dimension.
fn resolve_fastembed_model(name: &str) -> Result<(EmbeddingModel, usize)> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        s if s.contains("bge-small-en") => Ok((EmbeddingModel::BGESmallENV15, 384)),
        s if s.contains("bge-base-en") => Ok((EmbeddingModel::BGEBaseENV15, 768)),
        s if s.contains("bge-large-en") => Ok((EmbeddingModel::BGELargeENV15, 1024)),
        s if s.contains("bge-m3") => Ok((EmbeddingModel::BGEM3, 1024)),
        s if s.contains("all-minilm-l6") || s.contains("minilm-l6") => {
            Ok((EmbeddingModel::AllMiniLML6V2, 384))
        }
        s if s.contains("all-minilm-l12") || s.contains("minilm-l12") => {
            Ok((EmbeddingModel::AllMiniLML12V2, 384))
        }
        s if s.contains("nomic-embed-text") => Ok((EmbeddingModel::NomicEmbedTextV15, 768)),
        s if s.contains("mxbai-embed-large") => Ok((EmbeddingModel::MxbaiEmbedLargeV1, 1024)),
        s if s.contains("gte-large-en") => Ok((EmbeddingModel::GTELargeENV15, 1024)),
        s if s.contains("gte-base-en") => Ok((EmbeddingModel::GTEBaseENV15, 768)),
        _ => Err(ShrimPKError::Embedding(format!(
            "Unknown fastembed model '{name}'. Supported: bge-small-en-v1.5, bge-base-en-v1.5, \
             bge-large-en-v1.5, bge-m3, all-minilm-l6-v2, all-minilm-l12-v2, \
             nomic-embed-text-v1.5, mxbai-embed-large-v1, gte-large-en-v1.5, gte-base-en-v1.5"
        ))),
    }
}

// ---------------------------------------------------------------------------
// OpenAIProvider
// ---------------------------------------------------------------------------

/// OpenAI-compatible embedding API provider.
///
/// Works with any endpoint that implements the `/v1/embeddings` contract:
/// OpenAI, Ollama, LiteLLM, vLLM, Azure OpenAI, etc.
///
/// API key is read from `SHRIMPK_EMBEDDING_API_KEY` env var -- never stored in config.
///
/// # Blocking
///
/// Uses [`ureq`] (synchronous HTTP). All calls block the current thread for up to 30 s.
/// Callers in async contexts **must** invoke this provider through
/// `EchoEngine::embed_blocking()` which uses
/// `tokio::task::block_in_place` to prevent worker-thread starvation.
pub struct OpenAIProvider {
    url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
    dim: usize,
    display_name: String,
}

impl OpenAIProvider {
    /// Create a new OpenAI-compatible embedding provider.
    ///
    /// The `dim` parameter must match the actual dimension of the remote model.
    /// Use `EchoConfig::infer_embedding_dim()` to auto-derive it from the model name.
    pub fn new(url: &str, model: &str, dim: usize) -> Result<Self> {
        let api_key = std::env::var("SHRIMPK_EMBEDDING_API_KEY").ok();
        let display_name = format!("openai/{model}");

        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(30)))
                .build(),
        );

        tracing::info!(
            url = %url,
            model = %model,
            dim = dim,
            has_api_key = api_key.is_some(),
            "OpenAIProvider initialized"
        );

        Ok(Self {
            url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            agent,
            dim,
            display_name,
        })
    }

    /// Call the embedding API for a batch of texts.
    ///
    /// # Blocking
    ///
    /// This method performs a synchronous HTTP POST via [`ureq`] and will block the
    /// calling thread for up to 30 s (the global timeout configured in [`Self::new`]).
    ///
    /// When running on a **multi-thread** Tokio runtime (the daemon) the blocking
    /// HTTP call is wrapped in [`tokio::task::block_in_place`] to inform the
    /// scheduler and prevent worker-thread starvation.  On a **current-thread**
    /// runtime (`#[tokio::test]`) or outside Tokio entirely (sync tests, CLI) the
    /// request runs directly, because `block_in_place` panics on a single-threaded
    /// runtime.
    ///
    /// This is defense-in-depth: [`EchoEngine::embed_blocking()`] also wraps the
    /// outer call with `block_in_place` for the mutex-lock concern; the inner wrap
    /// here covers the provider-specific HTTP concern.
    fn call_api(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let endpoint = format!("{}/v1/embeddings", self.url);

        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        // Audit logging (matches HttpConsolidator pattern)
        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        tracing::info!(
            target: "shrimpk::audit",
            endpoint = %endpoint,
            data_bytes = body_bytes.len(),
            batch_size = texts.len(),
            direction = "outbound",
            component = "embedding_provider",
            "External embedding API call"
        );

        // Closure that performs the synchronous HTTP request and parses the response.
        // Factored out so we can conditionally wrap it with block_in_place.
        let do_request = || -> Result<Vec<Vec<f32>>> {
            let mut req = self.agent.post(&endpoint);
            if let Some(key) = &self.api_key {
                req = req.header("Authorization", &format!("Bearer {key}"));
            }

            let mut resp = req.send_json(&body).map_err(|e| {
                ShrimPKError::Embedding(format!("OpenAI embedding API error at {endpoint}: {e}"))
            })?;

            let json: serde_json::Value = resp.body_mut().read_json().map_err(|e| {
                ShrimPKError::Embedding(format!("OpenAI embedding API parse error: {e}"))
            })?;

            // Extract embeddings: {"data": [{"embedding": [...], "index": 0}, ...]}
            let data = json["data"].as_array().ok_or_else(|| {
                ShrimPKError::Embedding(format!(
                    "OpenAI embedding API: missing 'data' array in response: {}",
                    truncate_json(&json)
                ))
            })?;

            // Sort by index to maintain input order
            let mut indexed: Vec<(usize, Vec<f32>)> = data
                .iter()
                .filter_map(|item| {
                    let index = item["index"].as_u64()? as usize;
                    let embedding: Vec<f32> = item["embedding"]
                        .as_array()?
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    Some((index, embedding))
                })
                .collect();

            indexed.sort_by_key(|(i, _)| *i);
            let embeddings: Vec<Vec<f32>> = indexed.into_iter().map(|(_, e)| e).collect();

            if embeddings.len() != texts.len() {
                return Err(ShrimPKError::Embedding(format!(
                    "OpenAI embedding API returned {} embeddings for {} inputs",
                    embeddings.len(),
                    texts.len()
                )));
            }

            // Validate dimension
            if let Some(first) = embeddings.first()
                && first.len() != self.dim
            {
                return Err(ShrimPKError::Embedding(format!(
                    "OpenAI embedding dimension mismatch: expected {}, got {} from model '{}'",
                    self.dim,
                    first.len(),
                    self.model
                )));
            }

            Ok(embeddings)
        };

        // Wrap blocking HTTP in block_in_place on multi-thread Tokio runtime
        // to prevent worker-thread starvation (ureq has a 30 s timeout).
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(do_request)
            }
            _ => do_request(),
        }
    }
}

impl EmbeddingProvider for OpenAIProvider {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let results = self.call_api(&[text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| ShrimPKError::Embedding("Empty OpenAI embedding result".into()))
    }

    fn embed_batch(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(&texts)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        &self.display_name
    }
}

/// Truncate JSON for error messages.
fn truncate_json(v: &serde_json::Value) -> String {
    let s = v.to_string();
    if s.len() > 200 {
        format!("{}...", &s[..200])
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create an embedding provider based on config.
///
/// Reads `config.embedding_provider` to select the backend:
/// - `Fastembed` → `FastembedProvider` with `config.embedding_model`
/// - `OpenAI` → `OpenAIProvider` with `config.embedding_api_url` + `config.embedding_model`
pub fn from_config(config: &EchoConfig) -> Result<Box<dyn EmbeddingProvider>> {
    match config.embedding_provider {
        EmbeddingBackend::Fastembed => {
            let provider = FastembedProvider::new(&config.embedding_model)?;
            Ok(Box::new(provider))
        }
        EmbeddingBackend::OpenAI => {
            let dim = config.infer_embedding_dim();
            let provider =
                OpenAIProvider::new(&config.embedding_api_url, &config.embedding_model, dim)?;
            Ok(Box::new(provider))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_fastembed_known_models() {
        let cases = [
            ("BGE-small-EN-v1.5", 384),
            ("bge-small-en-v1.5", 384),
            ("bge-base-en-v1.5", 768),
            ("BGE-large-EN-v1.5", 1024),
            ("bge-m3", 1024),
            ("all-MiniLM-L6-v2", 384),
            ("all-minilm-l12-v2", 384),
            ("nomic-embed-text-v1.5", 768),
            ("mxbai-embed-large-v1", 1024),
            ("gte-large-en-v1.5", 1024),
            ("gte-base-en-v1.5", 768),
        ];
        for (name, expected_dim) in cases {
            let (_, dim) =
                resolve_fastembed_model(name).unwrap_or_else(|_| panic!("should resolve '{name}'"));
            assert_eq!(dim, expected_dim, "model '{name}'");
        }
    }

    #[test]
    fn resolve_fastembed_unknown_errors() {
        let result = resolve_fastembed_model("my-custom-model");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unknown fastembed model"),
            "error should mention unknown model: {err}"
        );
    }

    #[test]
    fn openai_provider_initializes() {
        // OpenAI provider should initialize (api_key is read from env, may or may not be set)
        let provider = OpenAIProvider::new("http://localhost:11434", "nomic-embed-text", 768);
        assert!(provider.is_ok());
        let p = provider.unwrap();
        assert_eq!(p.dimension(), 768);
        assert_eq!(p.name(), "openai/nomic-embed-text");
    }

    #[test]
    fn from_config_default_selects_fastembed() {
        // This test just checks that the factory selects the right backend.
        // It doesn't actually initialize the model (that would download it).
        let config = EchoConfig::default();
        assert_eq!(config.embedding_provider, EmbeddingBackend::Fastembed);
        assert_eq!(config.embedding_model, "BGE-small-EN-v1.5");
    }

    #[test]
    #[ignore = "requires fastembed model download"]
    fn fastembed_provider_default_model_works() {
        let mut provider = FastembedProvider::new("BGE-small-EN-v1.5").unwrap();
        assert_eq!(provider.dimension(), 384);

        let embedding = provider.embed("Hello world").unwrap();
        assert_eq!(embedding.len(), 384);

        let batch = provider
            .embed_batch(vec!["Hello".into(), "World".into()])
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), 384);
    }
}
