//! PII (Personally Identifiable Information) detection types.
//!
//! Types for identifying and masking sensitive data in memory content.
//! The actual regex patterns and scanning logic live in `shrimpk-memory`.
//! This module defines the shared types.

use serde::{Deserialize, Serialize};

/// Type of PII/secret detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiiType {
    /// API key patterns (sk-, pk_, AKIA, etc.)
    ApiKey,
    /// Credit card numbers (16 digits with optional separators)
    CreditCard,
    /// Social Security Numbers (XXX-XX-XXXX)
    Ssn,
    /// Email addresses
    Email,
    /// Phone numbers (various formats)
    PhoneNumber,
    /// Passwords following "password:", "pwd:", etc.
    Password,
    /// Custom user-defined pattern
    Custom(String),
}

impl std::fmt::Display for PiiType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey => write!(f, "api_key"),
            Self::CreditCard => write!(f, "credit_card"),
            Self::Ssn => write!(f, "ssn"),
            Self::Email => write!(f, "email"),
            Self::PhoneNumber => write!(f, "phone"),
            Self::Password => write!(f, "password"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// A match found by the PII scanner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiMatch {
    /// What type of PII was found.
    pub pattern_type: PiiType,
    /// Byte offset of the start of the match in the original text.
    pub start: usize,
    /// Byte offset of the end of the match in the original text.
    pub end: usize,
    /// The masked replacement string (e.g., "[MASKED:api_key]").
    pub masked_value: String,
}

impl PiiMatch {
    /// Create a new PII match.
    pub fn new(pattern_type: PiiType, start: usize, end: usize) -> Self {
        let masked_value = format!("[MASKED:{}]", pattern_type);
        Self {
            pattern_type,
            start,
            end,
            masked_value,
        }
    }

    /// Length of the matched text in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the match is empty (shouldn't happen, but for safety).
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pii_type_display() {
        assert_eq!(PiiType::ApiKey.to_string(), "api_key");
        assert_eq!(PiiType::CreditCard.to_string(), "credit_card");
        assert_eq!(PiiType::Custom("token".into()).to_string(), "token");
    }

    #[test]
    fn pii_match_creates_mask() {
        let m = PiiMatch::new(PiiType::ApiKey, 10, 30);
        assert_eq!(m.masked_value, "[MASKED:api_key]");
        assert_eq!(m.len(), 20);
    }
}
