//! # shrimpk-memory
//!
//! Echo Memory engine — push-based AI memory where memories self-activate
//! based on contextual relevance, like human associative recall.
//!
//! ## Architecture
//! - **Store**: In-memory vector store with disk persistence
//! - **Embedder**: Sentence embedding via fastembed (all-MiniLM-L6-v2)
//! - **Similarity**: SIMD-accelerated cosine similarity via simsimd
//! - **Echo**: The activation cycle — embed query -> search -> rank -> return
//! - **PII Filter**: Regex-based secret/PII detection and masking
//!
//! ## Phase 1 (KS1)
//! Flat brute-force similarity. Phase 2 adds LSH, Bloom filters, Hebbian learning.

pub mod bloom;
pub mod consolidation;
pub mod embedder;
pub mod hebbian;
pub mod lsh;
pub mod persistence;
pub mod similarity;
pub mod pii;
pub mod reformulator;
pub mod store;
pub mod echo;

// Re-export the main types
pub use consolidation::ConsolidationResult;
pub use echo::EchoEngine;
pub use hebbian::HebbianGraph;
pub use pii::PiiFilter;
pub use reformulator::MemoryReformulator;
pub use store::EchoStore;
pub use shrimpk_core::{EchoConfig, EchoResult, MemoryCategory, MemoryEntry, MemoryId, MemoryStats, SensitivityLevel};
