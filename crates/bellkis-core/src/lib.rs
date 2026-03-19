//! # bellkis-core
//!
//! Core types, traits, and error framework for the Bellkis AI-OS kernel.
//! This crate defines the contract layer — shared types used across all kernel crates.
//! It has zero heavy dependencies (no ML, no async runtime, no I/O).

pub mod error;
pub mod memory;
pub mod config;
pub mod pii;
pub mod traits;

// Re-export commonly used types at crate root
pub use error::{BellkisError, Result};
pub use memory::{MemoryId, MemoryEntry, EchoResult, MemoryStats, SensitivityLevel};
pub use config::{EchoConfig, QuantizationMode};
pub use pii::{PiiMatch, PiiType};
pub use traits::{Provider, ModelBackend};
