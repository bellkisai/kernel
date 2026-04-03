//! # shrimpk-memory
//!
//! Echo Memory engine — push-based AI memory where memories self-activate
//! based on contextual relevance, like human associative recall.
//!
//! ## Architecture
//! - **Store**: In-memory vector store with disk persistence
//! - **MultiEmbedder**: Multi-channel embedding via fastembed (text: all-MiniLM-L6-v2)
//! - **Similarity**: SIMD-accelerated cosine similarity via simsimd
//! - **Echo**: The activation cycle — embed query -> search -> rank -> return
//! - **PII Filter**: Regex-based secret/PII detection and masking
//!
//! ## Phase 1 (KS1)
//! Flat brute-force similarity. Phase 2 adds LSH, Bloom filters, Hebbian learning.

pub mod activation;
pub mod bloom;
pub mod consolidation;
pub mod consolidator;
pub mod echo;
pub mod embedder;
pub mod hebbian;
pub mod importance;
pub mod labels;
pub mod lsh;
pub mod persistence;
pub mod pii;
pub mod reformulator;
pub mod reranker;
pub mod similarity;
// Speech module is always compiled — constants, l2_normalize, resample_linear are always available.
// The `speech` feature gates the real ONNX inference implementation.
pub mod speech;
pub mod store;

// Re-export the main types
pub use activation::{actr_ol_activation, power_law_decay};
pub use consolidation::{ConsolidationResult, detect_relationship};
pub use echo::EchoEngine;
pub use hebbian::{HebbianGraph, RelationshipType};
pub use importance::{compute_embedding_mean, compute_importance};
pub use pii::PiiFilter;
pub use reformulator::MemoryReformulator;
pub use shrimpk_core::{
    EchoConfig, EchoResult, MemoryCategory, MemoryEntry, MemoryId, MemoryStats, QueryMode,
    SensitivityLevel,
};
pub use store::EchoStore;
