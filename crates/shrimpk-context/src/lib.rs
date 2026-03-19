//! # shrimpk-context
//!
//! Smart prompt compilation — the "compiler" that builds optimal context windows.
//! Combines system prompt + echo memories + RAG + conversation history within token budget.
//!
//! ## Architecture
//!
//! The context assembler works in a priority pipeline:
//! 1. **Budget** — allocates token slots proportionally across sources
//! 2. **Segments** — typed context blocks (system, echo, RAG, conversation, query)
//! 3. **Assembly** — fills slots by priority, truncating lowest-priority first
//! 4. **Sensitivity** — filters echo results based on provider locality
//!
//! ## Usage
//!
//! ```rust
//! use shrimpk_context::{ContextAssembler, ContextConfig};
//!
//! let assembler = ContextAssembler::new(ContextConfig::default());
//! let result = assembler.assemble(
//!     8_000,                    // context window
//!     None,                     // use default system prompt
//!     &[],                      // no echo results
//!     &[],                      // no RAG chunks
//!     &[],                      // no conversation
//!     "What is the meaning of life?",
//! );
//! assert!(result.total_tokens < 8_000);
//! ```

pub mod budget;
pub mod segment;
pub mod assembler;
pub mod sensitivity;

pub use assembler::{ContextAssembler, ContextConfig, AssembledContext};
pub use budget::TokenBudget;
pub use segment::ContextSegment;
