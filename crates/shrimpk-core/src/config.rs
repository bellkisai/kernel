//! Configuration types for the Echo Memory engine.
//!
//! Config resolution follows a priority chain:
//! **env vars > config file (TOML) > auto-detect defaults**

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Default maximum disk usage: 2 GB.
const DEFAULT_MAX_DISK_BYTES: u64 = 2_147_483_648;

/// Reranker backend for echo result reranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RerankerBackend {
    /// No reranking (disabled).
    #[default]
    None,
    /// LLM-based reranking via Ollama (~2s latency). Original KS23 implementation.
    Llm,
    /// Cross-encoder reranking via fastembed ONNX (~5-15ms latency). KS24 Track 3.
    CrossEncoder,
}

impl std::fmt::Display for RerankerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Llm => write!(f, "llm"),
            Self::CrossEncoder => write!(f, "cross_encoder"),
        }
    }
}

impl std::str::FromStr for RerankerBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "disabled" | "off" => Ok(Self::None),
            "llm" | "ollama" => Ok(Self::Llm),
            "cross_encoder" | "crossencoder" | "ce" | "fastembed" => Ok(Self::CrossEncoder),
            _ => Err(format!(
                "invalid reranker backend '{s}': expected none, llm, or cross_encoder"
            )),
        }
    }
}

/// Quantization mode for embedding vectors in the echo index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantizationMode {
    /// Full 32-bit float (1,536 bytes per 384-dim vector). Best quality.
    #[default]
    F32,
    /// Half precision (768 bytes per vector). ~0.1% quality loss.
    F16,
    /// 8-bit integer (384 bytes per vector). ~1% quality loss.
    Int8,
    /// Binary 1-bit (48 bytes per vector). ~5% quality loss, needs re-ranking.
    Binary,
}

impl QuantizationMode {
    /// Bytes per embedding vector at this quantization level.
    pub fn bytes_per_vector(&self, dim: usize) -> usize {
        match self {
            Self::F32 => dim * 4,
            Self::F16 => dim * 2,
            Self::Int8 => dim,
            Self::Binary => dim.div_ceil(8),
        }
    }
}

impl std::fmt::Display for QuantizationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Int8 => write!(f, "int8"),
            Self::Binary => write!(f, "binary"),
        }
    }
}

impl std::str::FromStr for QuantizationMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "int8" => Ok(Self::Int8),
            "binary" => Ok(Self::Binary),
            _ => Err(format!(
                "invalid quantization mode '{s}': expected f32, f16, int8, or binary"
            )),
        }
    }
}

/// Configuration for the Echo Memory engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoConfig {
    /// Maximum number of memories to store.
    pub max_memories: usize,
    /// Cosine similarity threshold for echo activation (0.0 to 1.0).
    /// Lower = more results, higher = more precise.
    pub similarity_threshold: f32,
    /// Maximum number of echo results per query.
    pub max_echo_results: usize,
    /// RAM budget for the echo index in bytes.
    pub ram_budget_bytes: u64,
    /// Directory for persistent storage.
    pub data_dir: PathBuf,
    /// Embedding vector quantization mode.
    pub quantization: QuantizationMode,
    /// Embedding dimension (384 for all-MiniLM-L6-v2).
    pub embedding_dim: usize,
    /// Whether to use LSH for sub-linear candidate retrieval.
    pub use_lsh: bool,
    /// Whether to use Bloom filter pre-screening before LSH.
    pub use_bloom: bool,
    /// Whether to use label-based pre-filtering for candidate retrieval (ADR-015).
    /// When true, semantic labels on memories provide an inverted index that narrows
    /// candidates before cosine scoring. When false, falls back to pure LSH.
    pub use_labels: bool,
    /// Maximum disk usage in bytes for the data directory.
    #[serde(default = "default_max_disk_bytes")]
    pub max_disk_bytes: u64,
    /// Ollama API URL for LLM-powered memory enrichment during consolidation.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    /// Ollama model for fact extraction during consolidation.
    #[serde(default = "default_enrichment_model")]
    pub enrichment_model: String,
    /// Maximum number of extracted facts per parent memory.
    #[serde(default = "default_max_facts_per_memory")]
    pub max_facts_per_memory: usize,
    /// Consolidation provider: "ollama" (default), "http", or "none".
    #[serde(default = "default_consolidation_provider")]
    pub consolidation_provider: String,
    /// URL of the backend LLM provider for the proxy (default: Ollama).
    #[serde(default = "default_proxy_target")]
    pub proxy_target: String,
    /// Whether the OpenAI-compatible proxy is enabled.
    #[serde(default = "default_proxy_enabled")]
    pub proxy_enabled: bool,
    /// Maximum echo results to inject into proxy requests.
    #[serde(default = "default_proxy_max_echo_results")]
    pub proxy_max_echo_results: usize,
    /// Context window size (in tokens) for proxy token budgeting.
    #[serde(default = "default_proxy_context_window")]
    pub proxy_context_window: usize,
    /// Rate limit for the daemon API in requests per second.
    #[serde(default = "default_daemon_rate_limit")]
    pub daemon_rate_limit: u64,
    /// Whether the user has given explicit consent for external consolidation.
    /// Required when `consolidation_provider = "http"`. Default: false.
    #[serde(default)]
    pub consolidation_consent_given: bool,
    /// Recency weight for echo scoring (KS18 Track 3).
    /// Newer memories get a small boost: `recency_weight / (1.0 + days_since_stored)`.
    /// This helps Knowledge Update queries surface corrections over stale facts.
    /// Default: 0.05 (small — cosine similarity should still dominate).
    #[serde(default = "default_recency_weight")]
    pub recency_weight: f32,
    /// When true, child memories (from consolidation) only participate via Pipe B rescue,
    /// never in direct Pipe A ranking. Prevents child fragments from displacing parents.
    #[serde(default = "default_child_rescue_only")]
    pub child_rescue_only: bool,
    /// Score penalty applied to child memories when promoted via Pipe B rescue.
    /// Default: 0.0 (no penalty). Negative values demote children relative to parents.
    #[serde(default)]
    pub child_memory_penalty: f32,
    /// Demotion applied to older (superseded) memories when a Supersedes edge exists.
    /// Default: 0.0 (disabled). Positive values penalize stale facts.
    #[serde(default)]
    pub supersedes_demotion: f32,
    /// Custom system prompt for the consolidator LLM fact extraction.
    /// Use `{max_facts}` placeholder for the max facts count.
    /// Default: built-in prompt. Set via config.toml or SHRIMPK_FACT_PROMPT env var.
    #[serde(default)]
    pub fact_extraction_prompt: Option<String>,
    /// HyDE query expansion: ask an LLM for a hypothetical answer before embedding.
    /// Improves recall for preference-tracking and indirect queries at the cost of
    /// ~100-500ms latency per echo call (Ollama round-trip).
    /// Default: false (raw query embedding only).
    #[serde(default)]
    pub query_expansion_enabled: bool,
    /// LLM-based reranking of top-N echo results.
    /// When enabled, the top-10 results from cosine+Hebbian scoring are sent to
    /// a local LLM (via Ollama) which reorders them by true semantic relevance.
    /// This helps cases like PT-1 where keyword overlap misleads cosine ranking.
    /// Default: false (reranker is opt-in).
    /// DEPRECATED: Use `reranker_backend` instead. Kept for backward compatibility.
    #[serde(default)]
    pub reranker_enabled: bool,
    /// Reranker backend selection (KS24 Track 3).
    /// - `None`: disabled (default)
    /// - `Llm`: Ollama-based LLM reranking (~2s latency, original KS23)
    /// - `CrossEncoder`: fastembed ONNX cross-encoder (~5-15ms latency)
    ///
    /// When `reranker_enabled = true` and `reranker_backend = None`, falls back
    /// to `Llm` for backward compatibility.
    #[serde(default)]
    pub reranker_backend: RerankerBackend,

    // --- Intelligence scoring (KS50) ---
    /// Enable ACT-R Optimized Learning activation (replaces recency_boost).
    #[serde(default)]
    pub use_actr_activation: bool,
    /// Enable FSRS power-law decay (replaces exponential).
    #[serde(default = "default_true")]
    pub use_power_law_decay: bool,
    /// Enable importance scoring for consolidation priority.
    #[serde(default = "default_true")]
    pub use_importance: bool,
    /// Weight of ACT-R activation term in scoring formula.
    #[serde(default = "default_activation_weight")]
    pub activation_weight: f32,
    /// Weight of importance boost in scoring formula (0.0 = consolidation only).
    #[serde(default)]
    pub importance_weight: f32,
    /// Enable full ACT-R retrieval history (`Vec<u32>` ring buffer). Future.
    #[serde(default)]
    pub use_full_actr_history: bool,

    // --- Hebbian (KS60) ---
    /// Hebbian co-activation half-life in seconds. Default: 604800 (7 days).
    /// Lower values = faster decay = only recent co-activations matter.
    /// Higher values = longer memory of co-activation patterns.
    #[serde(default = "default_hebbian_half_life")]
    pub hebbian_half_life_secs: f64,
    /// Hebbian edge prune threshold. Edges with decayed weight below this
    /// are removed during consolidation. Default: 0.01.
    #[serde(default = "default_hebbian_prune_threshold")]
    pub hebbian_prune_threshold: f64,

    // --- GraphRAG (KS62) ---
    /// Enable entity-anchored graph traversal during echo queries.
    /// When true and query mentions known entities, BFS from entity-anchored
    /// memories via Hebbian edges, merged with vector results via RRF.
    #[serde(default = "default_true")]
    pub graph_traversal_enabled: bool,
    /// Maximum BFS hops for graph traversal. Default: 2.
    #[serde(default = "default_graph_max_hops")]
    pub graph_max_hops: usize,
    /// RRF k parameter. Higher values reduce the impact of rank position.
    /// Default: 60 (standard value from literature).
    #[serde(default = "default_graph_rrf_k")]
    pub graph_rrf_k: usize,

    // --- Community Summaries (KS64) ---
    /// Enable per-label community summary generation during consolidation.
    #[serde(default = "default_true")]
    pub community_summaries_enabled: bool,
    /// Echo final_score threshold below which community summaries are injected.
    /// Default: 0.25 — only triggers when results are weak.
    #[serde(default = "default_community_summary_threshold")]
    pub community_summary_threshold: f32,
    /// Minimum label cluster size to generate a summary. Default: 5.
    #[serde(default = "default_community_min_members")]
    pub community_min_members: usize,

    // --- Context Assembly (KS60) ---
    /// Maximum conversation turns to include in proxy context. Default: 20.
    #[serde(default = "default_proxy_max_conversation_turns")]
    pub proxy_max_conversation_turns: usize,

    // --- Multimodal (KS31) ---
    /// Enabled modalities. Default: `[Text]`.
    /// Add `Vision` to enable CLIP image embedding, `Speech` for audio embedding.
    #[serde(default = "default_modalities")]
    pub enabled_modalities: Vec<crate::Modality>,
    /// Embedding dimension for vision channel (CLIP). Default: 512.
    #[serde(default = "default_vision_dim")]
    pub vision_embedding_dim: usize,
    /// Embedding dimension for speech channel. Default: 640 (ECAPA-TDNN 256 + Whisper-tiny 384).
    #[serde(default = "default_speech_dim")]
    pub speech_embedding_dim: usize,
}

fn default_true() -> bool {
    true
}
fn default_activation_weight() -> f32 {
    0.1
}
fn default_hebbian_half_life() -> f64 {
    604_800.0
}
fn default_hebbian_prune_threshold() -> f64 {
    0.01
}
fn default_community_summary_threshold() -> f32 {
    0.25
}
fn default_community_min_members() -> usize {
    5
}
fn default_graph_max_hops() -> usize {
    2
}
fn default_graph_rrf_k() -> usize {
    60
}
fn default_proxy_max_conversation_turns() -> usize {
    20
}

fn default_modalities() -> Vec<crate::Modality> {
    vec![crate::Modality::Text]
}
fn default_vision_dim() -> usize {
    512
}
fn default_speech_dim() -> usize {
    640
}

fn default_proxy_target() -> String {
    "http://127.0.0.1:11434".to_string()
}
fn default_proxy_enabled() -> bool {
    true
}
fn default_proxy_max_echo_results() -> usize {
    5
}
fn default_proxy_context_window() -> usize {
    8000
}

fn default_daemon_rate_limit() -> u64 {
    100
}
fn default_child_rescue_only() -> bool {
    true
}

fn default_recency_weight() -> f32 {
    0.05
}

fn default_max_disk_bytes() -> u64 {
    DEFAULT_MAX_DISK_BYTES
}

fn default_ollama_url() -> String {
    "http://127.0.0.1:11434".to_string()
}

fn default_enrichment_model() -> String {
    "qwen2.5:1.5b".to_string()
}

fn default_max_facts_per_memory() -> usize {
    5
}

fn default_consolidation_provider() -> String {
    "ollama".to_string()
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            max_memories: 1_000_000,
            similarity_threshold: 0.14,
            max_echo_results: 20,
            ram_budget_bytes: 1_800_000_000,
            data_dir: config_dir(),
            quantization: QuantizationMode::F32,
            embedding_dim: 384,
            use_lsh: true,
            use_bloom: true,
            use_labels: true,
            max_disk_bytes: DEFAULT_MAX_DISK_BYTES,
            ollama_url: default_ollama_url(),
            enrichment_model: default_enrichment_model(),
            max_facts_per_memory: default_max_facts_per_memory(),
            consolidation_provider: default_consolidation_provider(),
            proxy_target: default_proxy_target(),
            proxy_enabled: default_proxy_enabled(),
            proxy_max_echo_results: default_proxy_max_echo_results(),
            proxy_context_window: default_proxy_context_window(),
            daemon_rate_limit: default_daemon_rate_limit(),
            consolidation_consent_given: false,
            recency_weight: default_recency_weight(),
            child_rescue_only: default_child_rescue_only(),
            child_memory_penalty: 0.0,
            supersedes_demotion: 0.15,
            fact_extraction_prompt: None,
            query_expansion_enabled: false,
            reranker_enabled: false,
            reranker_backend: RerankerBackend::None,
            use_actr_activation: false,
            use_power_law_decay: default_true(),
            use_importance: default_true(),
            activation_weight: default_activation_weight(),
            importance_weight: 0.0,
            use_full_actr_history: false,
            community_summaries_enabled: default_true(),
            community_summary_threshold: default_community_summary_threshold(),
            community_min_members: default_community_min_members(),
            graph_traversal_enabled: default_true(),
            graph_max_hops: default_graph_max_hops(),
            graph_rrf_k: default_graph_rrf_k(),
            hebbian_half_life_secs: default_hebbian_half_life(),
            hebbian_prune_threshold: default_hebbian_prune_threshold(),
            proxy_max_conversation_turns: default_proxy_max_conversation_turns(),
            enabled_modalities: default_modalities(),
            vision_embedding_dim: default_vision_dim(),
            speech_embedding_dim: default_speech_dim(),
        }
    }
}

impl EchoConfig {
    /// Resolve the effective reranker backend, handling backward compatibility.
    ///
    /// If `reranker_backend` is explicitly set to a non-None value, use it.
    /// If `reranker_enabled = true` but `reranker_backend = None`, fall back to `Llm`
    /// for backward compatibility with KS23 configs.
    pub fn effective_reranker_backend(&self) -> RerankerBackend {
        match self.reranker_backend {
            RerankerBackend::None if self.reranker_enabled => RerankerBackend::Llm,
            other => other,
        }
    }

    /// Auto-detect optimal configuration based on available system RAM.
    pub fn auto_detect() -> Self {
        let total_ram_bytes = get_total_ram();
        let ram_gb = total_ram_bytes / 1_073_741_824;

        match ram_gb {
            0..=7 => Self::minimal(),
            8..=15 => Self::standard(),
            16..=31 => Self::full(),
            _ => Self::maximum(),
        }
    }

    /// Minimal config for 8GB RAM machines.
    pub fn minimal() -> Self {
        Self {
            max_memories: 100_000,
            similarity_threshold: 0.16,
            max_echo_results: 10,
            ram_budget_bytes: 50_000_000,
            quantization: QuantizationMode::Binary,
            ..Default::default()
        }
    }

    /// Standard config for 16GB RAM machines.
    pub fn standard() -> Self {
        Self {
            max_memories: 500_000,
            ram_budget_bytes: 900_000_000,
            ..Default::default()
        }
    }

    /// Full config for 32GB RAM machines (default).
    pub fn full() -> Self {
        Self::default()
    }

    /// Maximum config for 64GB+ RAM machines.
    pub fn maximum() -> Self {
        Self {
            max_memories: 5_000_000,
            ram_budget_bytes: 9_000_000_000,
            ..Default::default()
        }
    }

    /// Estimated index size in bytes for the current config.
    pub fn estimated_index_bytes(&self) -> u64 {
        let bytes_per_entry = self.quantization.bytes_per_vector(self.embedding_dim) + 100;
        (self.max_memories as u64) * (bytes_per_entry as u64)
    }
}

// ---------------------------------------------------------------------------
// File-based config (TOML)
// ---------------------------------------------------------------------------

/// User-editable config file (`~/.shrimpk-kernel/config.toml`).
/// All fields are optional — missing fields fall back to auto-detect defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileConfig {
    pub max_memories: Option<usize>,
    pub similarity_threshold: Option<f32>,
    pub max_echo_results: Option<usize>,
    pub ram_budget_bytes: Option<u64>,
    pub data_dir: Option<PathBuf>,
    pub quantization: Option<QuantizationMode>,
    pub embedding_dim: Option<usize>,
    pub use_lsh: Option<bool>,
    pub use_bloom: Option<bool>,
    pub use_labels: Option<bool>,
    pub max_disk_bytes: Option<u64>,
    pub ollama_url: Option<String>,
    pub enrichment_model: Option<String>,
    pub max_facts_per_memory: Option<usize>,
    pub consolidation_provider: Option<String>,
    pub proxy_target: Option<String>,
    pub proxy_enabled: Option<bool>,
    pub proxy_max_echo_results: Option<usize>,
    pub proxy_context_window: Option<usize>,
    pub daemon_rate_limit: Option<u64>,
    pub consolidation_consent_given: Option<bool>,
    pub recency_weight: Option<f32>,
    pub child_rescue_only: Option<bool>,
    pub child_memory_penalty: Option<f32>,
    pub supersedes_demotion: Option<f32>,
    pub query_expansion_enabled: Option<bool>,
    pub reranker_enabled: Option<bool>,
    pub reranker_backend: Option<RerankerBackend>,
    pub use_actr_activation: Option<bool>,
    pub use_power_law_decay: Option<bool>,
    pub use_importance: Option<bool>,
    pub activation_weight: Option<f32>,
    pub importance_weight: Option<f32>,
    pub use_full_actr_history: Option<bool>,
    pub community_summaries_enabled: Option<bool>,
    pub community_summary_threshold: Option<f32>,
    pub community_min_members: Option<usize>,
    pub graph_traversal_enabled: Option<bool>,
    pub graph_max_hops: Option<usize>,
    pub graph_rrf_k: Option<usize>,
    pub hebbian_half_life_secs: Option<f64>,
    pub hebbian_prune_threshold: Option<f64>,
    pub proxy_max_conversation_turns: Option<usize>,
    pub enabled_modalities: Option<Vec<crate::Modality>>,
    pub vision_embedding_dim: Option<usize>,
    pub speech_embedding_dim: Option<usize>,
}

/// Default data directory: `~/.shrimpk-kernel/`
pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".shrimpk-kernel")
}

/// Path to the config file: `~/.shrimpk-kernel/config.toml`
pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

/// Load the TOML config file. Returns `Ok(None)` if the file does not exist.
pub fn load_config_file() -> crate::Result<Option<FileConfig>> {
    let path = config_path();
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| crate::ShrimPKError::Config(format!("reading {}: {e}", path.display())))?;
    let fc: FileConfig = toml::from_str(&content)
        .map_err(|e| crate::ShrimPKError::Config(format!("parsing {}: {e}", path.display())))?;
    Ok(Some(fc))
}

/// Save a `FileConfig` to the TOML config file, creating parent dirs if needed.
pub fn save_config_file(config: &FileConfig) -> crate::Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            crate::ShrimPKError::Config(format!("creating {}: {e}", parent.display()))
        })?;
    }
    let content = toml::to_string_pretty(config)
        .map_err(|e| crate::ShrimPKError::Config(format!("serializing config: {e}")))?;
    std::fs::write(&path, content)
        .map_err(|e| crate::ShrimPKError::Config(format!("writing {}: {e}", path.display())))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Environment variable overrides
// ---------------------------------------------------------------------------

fn env_usize(name: &str) -> crate::Result<Option<usize>> {
    match std::env::var(name) {
        Ok(val) => val
            .parse()
            .map(Some)
            .map_err(|_| crate::ShrimPKError::Config(format!("{name}={val}: invalid integer"))),
        Err(_) => Ok(None),
    }
}

fn env_u64(name: &str) -> crate::Result<Option<u64>> {
    match std::env::var(name) {
        Ok(val) => val
            .parse()
            .map(Some)
            .map_err(|_| crate::ShrimPKError::Config(format!("{name}={val}: invalid integer"))),
        Err(_) => Ok(None),
    }
}

fn env_f32(name: &str) -> crate::Result<Option<f32>> {
    match std::env::var(name) {
        Ok(val) => val
            .parse()
            .map(Some)
            .map_err(|_| crate::ShrimPKError::Config(format!("{name}={val}: invalid float"))),
        Err(_) => Ok(None),
    }
}

fn env_f64(name: &str) -> crate::Result<Option<f64>> {
    match std::env::var(name) {
        Ok(val) => val
            .parse()
            .map(Some)
            .map_err(|_| crate::ShrimPKError::Config(format!("{name}={val}: invalid float"))),
        Err(_) => Ok(None),
    }
}

fn env_quantization() -> crate::Result<Option<QuantizationMode>> {
    match std::env::var("SHRIMPK_QUANTIZATION") {
        Ok(val) => val
            .parse()
            .map(Some)
            .map_err(|e| crate::ShrimPKError::Config(format!("SHRIMPK_QUANTIZATION: {e}"))),
        Err(_) => Ok(None),
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

/// Resolve the final config: auto-detect → overlay file → overlay env vars.
pub fn resolve_config() -> crate::Result<EchoConfig> {
    let mut config = EchoConfig::auto_detect();

    // Layer 2: file overrides
    if let Some(fc) = load_config_file()? {
        if let Some(v) = fc.max_memories {
            config.max_memories = v;
        }
        if let Some(v) = fc.similarity_threshold {
            config.similarity_threshold = v;
        }
        if let Some(v) = fc.max_echo_results {
            config.max_echo_results = v;
        }
        if let Some(v) = fc.ram_budget_bytes {
            config.ram_budget_bytes = v;
        }
        if let Some(v) = fc.data_dir {
            config.data_dir = v;
        }
        if let Some(v) = fc.quantization {
            config.quantization = v;
        }
        if let Some(v) = fc.embedding_dim {
            config.embedding_dim = v;
        }
        if let Some(v) = fc.use_lsh {
            config.use_lsh = v;
        }
        if let Some(v) = fc.use_bloom {
            config.use_bloom = v;
        }
        if let Some(v) = fc.use_labels {
            config.use_labels = v;
        }
        if let Some(v) = fc.max_disk_bytes {
            config.max_disk_bytes = v;
        }
        if let Some(v) = fc.ollama_url {
            config.ollama_url = v;
        }
        if let Some(v) = fc.enrichment_model {
            config.enrichment_model = v;
        }
        if let Some(v) = fc.max_facts_per_memory {
            config.max_facts_per_memory = v;
        }
        if let Some(v) = fc.consolidation_provider {
            config.consolidation_provider = v;
        }
        if let Some(v) = fc.proxy_target {
            config.proxy_target = v;
        }
        if let Some(v) = fc.proxy_enabled {
            config.proxy_enabled = v;
        }
        if let Some(v) = fc.proxy_max_echo_results {
            config.proxy_max_echo_results = v;
        }
        if let Some(v) = fc.proxy_context_window {
            config.proxy_context_window = v;
        }
        if let Some(v) = fc.daemon_rate_limit {
            config.daemon_rate_limit = v;
        }
        if let Some(v) = fc.consolidation_consent_given {
            config.consolidation_consent_given = v;
        }
        if let Some(v) = fc.recency_weight {
            config.recency_weight = v;
        }
        if let Some(v) = fc.child_rescue_only {
            config.child_rescue_only = v;
        }
        if let Some(v) = fc.child_memory_penalty {
            config.child_memory_penalty = v;
        }
        if let Some(v) = fc.supersedes_demotion {
            config.supersedes_demotion = v;
        }
        if let Some(v) = fc.query_expansion_enabled {
            config.query_expansion_enabled = v;
        }
        if let Some(v) = fc.reranker_enabled {
            config.reranker_enabled = v;
        }
        if let Some(v) = fc.reranker_backend {
            config.reranker_backend = v;
        }
        if let Some(v) = fc.use_actr_activation {
            config.use_actr_activation = v;
        }
        if let Some(v) = fc.use_power_law_decay {
            config.use_power_law_decay = v;
        }
        if let Some(v) = fc.use_importance {
            config.use_importance = v;
        }
        if let Some(v) = fc.activation_weight {
            config.activation_weight = v;
        }
        if let Some(v) = fc.importance_weight {
            config.importance_weight = v;
        }
        if let Some(v) = fc.use_full_actr_history {
            config.use_full_actr_history = v;
        }
        if let Some(v) = fc.community_summaries_enabled {
            config.community_summaries_enabled = v;
        }
        if let Some(v) = fc.community_summary_threshold {
            config.community_summary_threshold = v;
        }
        if let Some(v) = fc.community_min_members {
            config.community_min_members = v;
        }
        if let Some(v) = fc.graph_traversal_enabled {
            config.graph_traversal_enabled = v;
        }
        if let Some(v) = fc.graph_max_hops {
            config.graph_max_hops = v;
        }
        if let Some(v) = fc.graph_rrf_k {
            config.graph_rrf_k = v;
        }
        if let Some(v) = fc.hebbian_half_life_secs {
            config.hebbian_half_life_secs = v;
        }
        if let Some(v) = fc.hebbian_prune_threshold {
            config.hebbian_prune_threshold = v;
        }
        if let Some(v) = fc.proxy_max_conversation_turns {
            config.proxy_max_conversation_turns = v;
        }
        if let Some(v) = fc.enabled_modalities {
            config.enabled_modalities = v;
        }
        if let Some(v) = fc.vision_embedding_dim {
            config.vision_embedding_dim = v;
        }
        if let Some(v) = fc.speech_embedding_dim {
            config.speech_embedding_dim = v;
        }
    }

    // Layer 3: env var overrides (highest priority)
    if let Some(v) = env_usize("SHRIMPK_MAX_MEMORIES")? {
        config.max_memories = v;
    }
    if let Some(v) = env_f32("SHRIMPK_SIMILARITY_THRESHOLD")? {
        config.similarity_threshold = v;
    }
    if let Some(v) = env_u64("SHRIMPK_MAX_DISK_BYTES")? {
        config.max_disk_bytes = v;
    }
    if let Some(v) = env_quantization()? {
        config.quantization = v;
    }
    if let Some(v) = env_path("SHRIMPK_DATA_DIR") {
        config.data_dir = v;
    }
    if let Ok(v) = std::env::var("SHRIMPK_OLLAMA_URL") {
        config.ollama_url = v;
    }
    if let Ok(v) = std::env::var("SHRIMPK_ENRICHMENT_MODEL") {
        config.enrichment_model = v;
    }
    if let Ok(v) = std::env::var("SHRIMPK_CONSOLIDATION_PROVIDER") {
        config.consolidation_provider = v;
    }
    if let Ok(v) = std::env::var("SHRIMPK_PROXY_TARGET") {
        config.proxy_target = v;
    }
    if let Ok(v) = std::env::var("SHRIMPK_PROXY_ENABLED") {
        config.proxy_enabled = v.parse().unwrap_or(true);
    }
    if let Some(v) = env_usize("SHRIMPK_PROXY_CONTEXT_WINDOW")? {
        config.proxy_context_window = v;
    }
    if let Some(v) = env_u64("SHRIMPK_DAEMON_RATE_LIMIT")? {
        config.daemon_rate_limit = v;
    }
    if let Ok(v) = std::env::var("SHRIMPK_CONSOLIDATION_CONSENT") {
        config.consolidation_consent_given = v.parse().unwrap_or(false);
    }
    if let Some(v) = env_f32("SHRIMPK_RECENCY_WEIGHT")? {
        config.recency_weight = v;
    }
    if let Ok(v) = std::env::var("SHRIMPK_QUERY_EXPANSION") {
        config.query_expansion_enabled = v.parse().unwrap_or(false);
    }
    if let Ok(v) = std::env::var("SHRIMPK_RERANKER") {
        config.reranker_enabled = v.parse().unwrap_or(false);
    }
    if let Ok(v) = std::env::var("SHRIMPK_RERANKER_BACKEND")
        && let Ok(backend) = v.parse::<RerankerBackend>()
    {
        config.reranker_backend = backend;
    }
    if let Some(v) = env_f64("SHRIMPK_HEBBIAN_HALF_LIFE")? {
        config.hebbian_half_life_secs = v;
    }
    if let Some(v) = env_f64("SHRIMPK_HEBBIAN_PRUNE_THRESHOLD")? {
        config.hebbian_prune_threshold = v;
    }

    // Backward compatibility: if reranker_enabled=true but backend=None, default to Llm
    if config.reranker_enabled && config.reranker_backend == RerankerBackend::None {
        config.reranker_backend = RerankerBackend::Llm;
    }

    Ok(config)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get total system RAM in bytes.
fn get_total_ram() -> u64 {
    #[cfg(feature = "sysinfo")]
    {
        use sysinfo::System;
        let sys = System::new_all();
        sys.total_memory()
    }
    #[cfg(not(feature = "sysinfo"))]
    {
        16 * 1_073_741_824 // assume 16GB if sysinfo unavailable
    }
}

/// Calculate total disk usage of a directory (non-recursive for the data dir).
pub fn disk_usage(dir: &std::path::Path) -> std::io::Result<u64> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut total: u64 = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total += meta.len();
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detect_returns_valid_config() {
        let config = EchoConfig::auto_detect();
        assert!(config.max_memories > 0);
        assert!(config.similarity_threshold > 0.0);
        assert!(config.similarity_threshold < 1.0);
        assert!(config.max_echo_results > 0);
        assert_eq!(config.max_disk_bytes, DEFAULT_MAX_DISK_BYTES);
    }

    #[test]
    fn quantization_bytes_per_vector() {
        assert_eq!(QuantizationMode::F32.bytes_per_vector(384), 1536);
        assert_eq!(QuantizationMode::F16.bytes_per_vector(384), 768);
        assert_eq!(QuantizationMode::Int8.bytes_per_vector(384), 384);
        assert_eq!(QuantizationMode::Binary.bytes_per_vector(384), 48);
    }

    #[test]
    fn minimal_config_fits_8gb() {
        let config = EchoConfig::minimal();
        let estimated = config.estimated_index_bytes();
        assert!(
            estimated < 100_000_000,
            "Minimal index should be under 100MB, got {}",
            estimated
        );
    }

    #[test]
    fn full_config_fits_16gb() {
        let config = EchoConfig::full();
        let estimated = config.estimated_index_bytes();
        assert!(
            estimated < 2_000_000_000,
            "Full index should be under 2GB, got {}",
            estimated
        );
    }

    #[test]
    fn tier_progression() {
        let minimal = EchoConfig::minimal();
        let standard = EchoConfig::standard();
        let full = EchoConfig::full();
        let maximum = EchoConfig::maximum();
        assert!(minimal.max_memories < standard.max_memories);
        assert!(standard.max_memories < full.max_memories);
        assert!(full.max_memories < maximum.max_memories);
    }

    #[test]
    fn quantization_mode_roundtrip() {
        for mode in [
            QuantizationMode::F32,
            QuantizationMode::F16,
            QuantizationMode::Int8,
            QuantizationMode::Binary,
        ] {
            let s = mode.to_string();
            let parsed: QuantizationMode = s.parse().unwrap();
            assert_eq!(mode, parsed);
        }
    }

    #[test]
    fn quantization_mode_parse_case_insensitive() {
        assert_eq!(
            "F32".parse::<QuantizationMode>().unwrap(),
            QuantizationMode::F32
        );
        assert_eq!(
            "BINARY".parse::<QuantizationMode>().unwrap(),
            QuantizationMode::Binary
        );
    }

    #[test]
    fn quantization_mode_parse_invalid() {
        assert!("unknown".parse::<QuantizationMode>().is_err());
    }

    #[test]
    fn file_config_toml_roundtrip() {
        let fc = FileConfig {
            max_memories: Some(50_000),
            similarity_threshold: Some(0.2),
            max_disk_bytes: Some(1_000_000_000),
            quantization: Some(QuantizationMode::Binary),
            ..Default::default()
        };
        let toml_str = toml::to_string_pretty(&fc).unwrap();
        let parsed: FileConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.max_memories, Some(50_000));
        assert_eq!(parsed.similarity_threshold, Some(0.2));
        assert_eq!(parsed.max_disk_bytes, Some(1_000_000_000));
        assert_eq!(parsed.quantization, Some(QuantizationMode::Binary));
        assert!(parsed.data_dir.is_none());
    }

    #[test]
    fn file_config_empty_toml_produces_all_none() {
        let parsed: FileConfig = toml::from_str("").unwrap();
        assert!(parsed.max_memories.is_none());
        assert!(parsed.similarity_threshold.is_none());
        assert!(parsed.max_disk_bytes.is_none());
    }

    #[test]
    fn resolve_config_returns_auto_detect_when_no_file_or_env() {
        // This test works because no config file exists in CI/test environments
        // and we don't set SHRIMPK_ env vars by default.
        let config = resolve_config().unwrap();
        assert!(config.max_memories > 0);
        assert_eq!(config.max_disk_bytes, DEFAULT_MAX_DISK_BYTES);
    }

    #[test]
    fn config_dir_is_under_home() {
        let dir = config_dir();
        assert!(
            dir.to_string_lossy().contains(".shrimpk-kernel"),
            "config dir should contain .shrimpk-kernel, got {}",
            dir.display()
        );
    }

    #[test]
    fn config_path_is_toml() {
        let path = config_path();
        assert_eq!(path.extension().unwrap(), "toml");
    }

    #[test]
    fn disk_usage_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let usage = disk_usage(tmp.path()).unwrap();
        assert_eq!(usage, 0);
    }

    #[test]
    fn disk_usage_with_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.bin"), vec![0u8; 1000]).unwrap();
        std::fs::write(tmp.path().join("b.bin"), vec![0u8; 2000]).unwrap();
        let usage = disk_usage(tmp.path()).unwrap();
        assert_eq!(usage, 3000);
    }

    #[test]
    fn disk_usage_nonexistent_dir() {
        let usage = disk_usage(std::path::Path::new("/nonexistent/path/shrimpk")).unwrap();
        assert_eq!(usage, 0);
    }

    // --- Env var parsing unit tests (test helpers directly, no global state) ---

    #[test]
    fn env_usize_parses_valid() {
        // Use a unique env var name to avoid races
        unsafe { std::env::set_var("_SHRIMPK_TEST_USIZE", "42") };
        let val = env_usize("_SHRIMPK_TEST_USIZE").unwrap();
        unsafe { std::env::remove_var("_SHRIMPK_TEST_USIZE") };
        assert_eq!(val, Some(42));
    }

    #[test]
    fn env_usize_returns_none_when_unset() {
        let val = env_usize("_SHRIMPK_NONEXISTENT_VAR_12345").unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn env_usize_rejects_invalid() {
        unsafe { std::env::set_var("_SHRIMPK_TEST_BAD_USIZE", "xyz") };
        let result = env_usize("_SHRIMPK_TEST_BAD_USIZE");
        unsafe { std::env::remove_var("_SHRIMPK_TEST_BAD_USIZE") };
        assert!(result.is_err());
    }

    #[test]
    fn env_f32_parses_valid() {
        unsafe { std::env::set_var("_SHRIMPK_TEST_F32", "0.25") };
        let val = env_f32("_SHRIMPK_TEST_F32").unwrap();
        unsafe { std::env::remove_var("_SHRIMPK_TEST_F32") };
        assert_eq!(val, Some(0.25));
    }

    #[test]
    fn env_quantization_parses_valid() {
        unsafe { std::env::set_var("SHRIMPK_QUANTIZATION", "binary") };
        let val = env_quantization().unwrap();
        unsafe { std::env::remove_var("SHRIMPK_QUANTIZATION") };
        assert_eq!(val, Some(QuantizationMode::Binary));
    }

    #[test]
    fn env_path_returns_some() {
        unsafe { std::env::set_var("_SHRIMPK_TEST_PATH", "/tmp/test-dir") };
        let val = env_path("_SHRIMPK_TEST_PATH");
        unsafe { std::env::remove_var("_SHRIMPK_TEST_PATH") };
        assert_eq!(val, Some(PathBuf::from("/tmp/test-dir")));
    }

    #[test]
    fn env_path_returns_none_when_unset() {
        let val = env_path("_SHRIMPK_NONEXISTENT_PATH_12345");
        assert_eq!(val, None);
    }

    #[test]
    fn save_and_load_config_file_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");

        let fc = FileConfig {
            max_memories: Some(99_999),
            similarity_threshold: Some(0.3),
            ..Default::default()
        };

        // Write to custom path
        let content = toml::to_string_pretty(&fc).unwrap();
        std::fs::write(&path, &content).unwrap();

        // Read back
        let read_content = std::fs::read_to_string(&path).unwrap();
        let parsed: FileConfig = toml::from_str(&read_content).unwrap();
        assert_eq!(parsed.max_memories, Some(99_999));
        assert_eq!(parsed.similarity_threshold, Some(0.3));
    }

    #[test]
    fn default_config_has_disk_limit() {
        let config = EchoConfig::default();
        assert_eq!(config.max_disk_bytes, DEFAULT_MAX_DISK_BYTES);
    }

    #[test]
    fn default_config_has_recency_weight() {
        let config = EchoConfig::default();
        assert!((config.recency_weight - 0.05).abs() < f32::EPSILON);
    }

    // --- RerankerBackend tests ---

    #[test]
    fn reranker_backend_default_is_none() {
        let config = EchoConfig::default();
        assert_eq!(config.reranker_backend, RerankerBackend::None);
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
        // Llm aliases
        assert_eq!(
            "ollama".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::Llm
        );
        assert_eq!(
            "LLM".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::Llm
        );

        // CrossEncoder aliases
        assert_eq!(
            "ce".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::CrossEncoder
        );
        assert_eq!(
            "fastembed".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::CrossEncoder
        );
        assert_eq!(
            "crossencoder".parse::<RerankerBackend>().unwrap(),
            RerankerBackend::CrossEncoder
        );

        // None aliases
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
        assert!("".parse::<RerankerBackend>().is_err());
    }

    #[test]
    fn reranker_backend_serde_roundtrip() {
        let fc = FileConfig {
            reranker_backend: Some(RerankerBackend::CrossEncoder),
            ..Default::default()
        };
        let toml_str = toml::to_string_pretty(&fc).unwrap();
        let parsed: FileConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.reranker_backend, Some(RerankerBackend::CrossEncoder));
    }

    #[test]
    fn effective_backend_respects_explicit_setting() {
        let config = EchoConfig {
            reranker_enabled: false,
            reranker_backend: RerankerBackend::CrossEncoder,
            ..Default::default()
        };
        assert_eq!(
            config.effective_reranker_backend(),
            RerankerBackend::CrossEncoder
        );
    }

    #[test]
    fn effective_backend_backward_compat_reranker_enabled() {
        // Legacy: reranker_enabled=true, no explicit backend
        let config = EchoConfig {
            reranker_enabled: true,
            reranker_backend: RerankerBackend::None,
            ..Default::default()
        };
        assert_eq!(
            config.effective_reranker_backend(),
            RerankerBackend::Llm,
            "reranker_enabled=true should fall back to Llm"
        );
    }

    #[test]
    fn effective_backend_none_when_both_disabled() {
        let config = EchoConfig {
            reranker_enabled: false,
            reranker_backend: RerankerBackend::None,
            ..Default::default()
        };
        assert_eq!(config.effective_reranker_backend(), RerankerBackend::None);
    }

    #[test]
    fn effective_backend_explicit_overrides_legacy() {
        // Explicit cross_encoder set, even with reranker_enabled=false
        let config = EchoConfig {
            reranker_enabled: false,
            reranker_backend: RerankerBackend::CrossEncoder,
            ..Default::default()
        };
        assert_eq!(
            config.effective_reranker_backend(),
            RerankerBackend::CrossEncoder,
            "Explicit backend should override legacy reranker_enabled"
        );
    }
}
