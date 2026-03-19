//! Error types for the ShrimPK kernel.
//!
//! Uses a unified `ShrimPKError` enum with per-domain variants.
//! Each kernel crate can define its own error type and convert into `ShrimPKError`.

use serde::Serialize;

/// Unified error type for the ShrimPK kernel.
#[derive(Debug, thiserror::Error, Serialize)]
pub enum ShrimPKError {
    /// Echo Memory errors (store, retrieve, index)
    #[error("Memory error: {0}")]
    Memory(String),

    /// Embedding generation errors
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Persistence errors (save, load, corruption)
    #[error("Persistence error: {0}")]
    Persistence(String),

    /// Configuration errors (invalid params, auto-detect failure)
    #[error("Config error: {0}")]
    Config(String),

    /// PII filter errors (regex compilation, scan failure)
    #[error("PII filter error: {0}")]
    PiiFilter(String),

    /// Provider routing errors
    #[error("Router error: {0}")]
    Router(String),

    /// Context assembly errors
    #[error("Context error: {0}")]
    Context(String),

    /// Security/sandbox errors
    #[error("Security error: {0}")]
    Security(String),

    /// Disk limit exceeded errors
    #[error("Disk limit: {0}")]
    DiskLimit(String),

    /// Generic I/O errors
    #[error("IO error: {0}")]
    Io(String),
}

/// Convenience Result type using ShrimPKError.
pub type Result<T> = std::result::Result<T, ShrimPKError>;

// Convert std::io::Error into ShrimPKError
impl From<std::io::Error> for ShrimPKError {
    fn from(err: std::io::Error) -> Self {
        ShrimPKError::Io(err.to_string())
    }
}

// Convert serde_json::Error into ShrimPKError
impl From<serde_json::Error> for ShrimPKError {
    fn from(err: serde_json::Error) -> Self {
        ShrimPKError::Persistence(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = ShrimPKError::Memory("index corrupted".into());
        assert_eq!(err.to_string(), "Memory error: index corrupted");
    }

    #[test]
    fn error_serializes() {
        let err = ShrimPKError::Embedding("model not found".into());
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("model not found"));
    }
}
