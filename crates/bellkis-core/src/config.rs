//! Configuration types for the Echo Memory engine.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Quantization mode for embedding vectors in the echo index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Full 32-bit float (1,536 bytes per 384-dim vector). Best quality.
    F32,
    /// Half precision (768 bytes per vector). ~0.1% quality loss.
    F16,
    /// 8-bit integer (384 bytes per vector). ~1% quality loss.
    Int8,
    /// Binary 1-bit (48 bytes per vector). ~5% quality loss, needs re-ranking.
    Binary,
}

impl Default for QuantizationMode {
    fn default() -> Self {
        Self::F32
    }
}

impl QuantizationMode {
    /// Bytes per embedding vector at this quantization level.
    pub fn bytes_per_vector(&self, dim: usize) -> usize {
        match self {
            Self::F32 => dim * 4,
            Self::F16 => dim * 2,
            Self::Int8 => dim,
            Self::Binary => (dim + 7) / 8, // ceil(dim/8) bytes
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
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            max_memories: 1_000_000,
            similarity_threshold: 0.14, // KS1 precision tuning: perfect F1 at 0.14
            max_echo_results: 20,     // KS1 tuning: avg 20.6 results at optimal threshold
            ram_budget_bytes: 1_800_000_000, // ~1.8 GB
            data_dir: dirs_default(),
            quantization: QuantizationMode::F32,
            embedding_dim: 384,
        }
    }
}

impl EchoConfig {
    /// Auto-detect optimal configuration based on available system RAM.
    ///
    /// Uses binary quantization on low-RAM machines (8GB) for 150MB index,
    /// full f32 on high-RAM machines (32GB+) for maximum quality.
    /// User can override any setting.
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
    /// Binary quantized: 100K memories in ~5MB index.
    pub fn minimal() -> Self {
        Self {
            max_memories: 100_000,
            similarity_threshold: 0.16, // slightly higher than default to compensate for binary quant
            max_echo_results: 10,
            ram_budget_bytes: 50_000_000, // ~50 MB
            quantization: QuantizationMode::Binary,
            ..Default::default()
        }
    }

    /// Standard config for 16GB RAM machines.
    /// f32: 500K memories in ~900MB index.
    pub fn standard() -> Self {
        Self {
            max_memories: 500_000,
            ram_budget_bytes: 900_000_000, // ~900 MB
            ..Default::default()
        }
    }

    /// Full config for 32GB RAM machines.
    /// f32: 1M memories in ~1.8GB index.
    pub fn full() -> Self {
        Self::default()
    }

    /// Maximum config for 64GB+ RAM machines.
    /// f32: 5M memories in ~9GB index.
    pub fn maximum() -> Self {
        Self {
            max_memories: 5_000_000,
            ram_budget_bytes: 9_000_000_000, // ~9 GB
            ..Default::default()
        }
    }

    /// Estimated index size in bytes for the current config.
    pub fn estimated_index_bytes(&self) -> u64 {
        let bytes_per_entry = self.quantization.bytes_per_vector(self.embedding_dim)
            + 100; // metadata overhead per entry
        (self.max_memories as u64) * (bytes_per_entry as u64)
    }
}

/// Get total system RAM in bytes.
fn get_total_ram() -> u64 {
    // sysinfo is an optional dependency — fallback to 16GB assumption
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

/// Default data directory: ~/.bellkis-kernel/
fn dirs_default() -> PathBuf {
    dirs_home().join(".bellkis-kernel")
}

/// Home directory (cross-platform).
fn dirs_home() -> PathBuf {
    // Simple cross-platform home directory detection
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
    } else if let Ok(profile) = std::env::var("USERPROFILE") {
        PathBuf::from(profile)
    } else {
        PathBuf::from(".")
    }
}

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
        assert!(estimated < 100_000_000, "Minimal index should be under 100MB, got {}", estimated);
    }

    #[test]
    fn full_config_fits_16gb() {
        let config = EchoConfig::full();
        let estimated = config.estimated_index_bytes();
        assert!(estimated < 2_000_000_000, "Full index should be under 2GB, got {}", estimated);
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
}
