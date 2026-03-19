//! Shared formatting helpers for MCP tool responses.

use shrimpk_core::{EchoConfig, QuantizationMode};

pub fn tier_name(config: &EchoConfig) -> &'static str {
    match (config.quantization, config.max_memories) {
        (QuantizationMode::Binary, _) => "minimal",
        (QuantizationMode::F32, m) if m >= 5_000_000 => "maximum",
        (QuantizationMode::F32, m) if m >= 1_000_000 => "full",
        (QuantizationMode::F32, m) if m >= 500_000 => "standard",
        _ => "custom",
    }
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(c);
    }
    result
}

pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = s
            .char_indices()
            .take_while(|&(i, _)| i < max_len.saturating_sub(3))
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        format!("{}...", &s[..end])
    }
}

pub fn detect_ram_gb() -> u64 {
    use sysinfo::System;
    let sys = System::new_all();
    sys.total_memory() / 1_073_741_824
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_boundaries() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn format_number_with_commas() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1_000), "1,000");
        assert_eq!(format_number(1_000_000), "1,000,000");
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let result = truncate("hello world this is long", 10);
        assert!(result.len() <= 10);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn tier_name_detects_correctly() {
        let mut config = EchoConfig::default();
        assert_eq!(tier_name(&config), "full");

        config.max_memories = 500_000;
        assert_eq!(tier_name(&config), "standard");

        config.quantization = QuantizationMode::Binary;
        assert_eq!(tier_name(&config), "minimal");
    }

    #[test]
    fn detect_ram_returns_nonzero() {
        assert!(detect_ram_gb() > 0);
    }
}
