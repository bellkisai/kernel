//! # shrimpk-core
//!
//! Core types, traits, and error framework for the ShrimPK AI-OS kernel.
//! This crate defines the contract layer — shared types used across all kernel crates.
//! It has zero heavy dependencies (no ML, no async runtime, no I/O).

pub mod config;
pub mod error;
pub mod memory;
pub mod pii;
pub mod traits;

// Re-export commonly used types at crate root
pub use config::{
    EchoConfig, FileConfig, QuantizationMode, RerankerBackend, config_dir, config_path,
    disk_usage, load_config_file, resolve_config, save_config_file,
};
pub use error::{Result, ShrimPKError};
pub use memory::{
    EchoResult, MemoryCategory, MemoryEntry, MemoryEntrySummary, MemoryId, MemoryStats,
    SensitivityLevel,
};
pub use pii::{PiiMatch, PiiType};
pub use traits::{Consolidator, ModelBackend, Provider};
