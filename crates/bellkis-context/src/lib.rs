//! # bellkis-context
//!
//! Smart prompt compilation — the "compiler" that builds optimal context windows.
//! Combines system prompt + echo memories + RAG + conversation history within token budget.
//!
//! ## Planned Features (KS2+)
//! - Token budget allocation per context source
//! - Priority-ordered context assembly
//! - Adaptive compression based on model context window
//! - Integration with Echo Memory for automatic context enrichment

// TODO: Implement in KS2
pub use bellkis_core::EchoConfig;
