//! PII/secret detection and masking.
//!
//! Scans text for sensitive patterns (API keys, credit cards, SSNs, emails,
//! phone numbers, passwords) and replaces them with masked tokens.
//! Masking happens at **store** time, not at echo time.

use regex::Regex;
use shrimpk_core::{PiiMatch, PiiType, SensitivityLevel};
use tracing::instrument;

/// A compiled pattern for PII detection.
struct CompiledPattern {
    regex: Regex,
    pii_type: PiiType,
}

/// PII/secret filter with pre-compiled regex patterns.
///
/// Thread-safe (Regex is Send+Sync). Create once, reuse across the engine lifetime.
pub struct PiiFilter {
    patterns: Vec<CompiledPattern>,
}

impl PiiFilter {
    /// Create a new PII filter with all default patterns compiled.
    ///
    /// # Panics
    /// Panics if any built-in regex pattern fails to compile (indicates a bug).
    pub fn new() -> Self {
        let pattern_defs: Vec<(&str, PiiType)> = vec![
            // API keys — longer/more-specific prefixes first to handle overlaps
            // OpenAI project keys: sk-proj-...
            (r"sk-proj-[a-zA-Z0-9_-]{20,}", PiiType::ApiKey),
            // Anthropic keys: sk-ant-...
            (r"sk-ant-[a-zA-Z0-9_-]{20,}", PiiType::ApiKey),
            // Generic sk- keys (OpenAI legacy, Stripe, etc.) — allow dashes/underscores
            (r"sk-[a-zA-Z0-9_-]{20,}", PiiType::ApiKey),
            // Publishable keys: pk_...
            (r"pk_[a-zA-Z0-9_-]{20,}", PiiType::ApiKey),
            // AWS access keys: AKIA...
            (r"AKIA[A-Z0-9]{16}", PiiType::ApiKey),
            // Groq keys: gsk_...
            (r"gsk_[a-zA-Z0-9]{20,}", PiiType::ApiKey),
            // xAI/Grok keys: xai-...
            (r"xai-[a-zA-Z0-9]{20,}", PiiType::ApiKey),
            // GitHub personal access tokens: ghp_...
            (r"ghp_[a-zA-Z0-9]{36}", PiiType::ApiKey),
            // GitLab personal access tokens: glpat-...
            (r"glpat-[a-zA-Z0-9_-]{20,}", PiiType::ApiKey),
            // Credit card numbers (16 digits with optional separators)
            (
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                PiiType::CreditCard,
            ),
            // SSN (US format)
            (r"\b\d{3}-\d{2}-\d{4}\b", PiiType::Ssn),
            // Email addresses
            (
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                PiiType::Email,
            ),
            // Phone numbers (US-style)
            (r"\b\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b", PiiType::PhoneNumber),
            // Passwords following "password:", "pwd:", etc.
            (
                r"(?i)(?:password|pwd|passwd|pass)\s*[:=]\s*\S+",
                PiiType::Password,
            ),
        ];

        let patterns = pattern_defs
            .into_iter()
            .map(|(pat, pii_type)| {
                let regex = Regex::new(pat).unwrap_or_else(|e| {
                    panic!("Bug: built-in PII pattern '{pat}' failed to compile: {e}")
                });
                CompiledPattern { regex, pii_type }
            })
            .collect();

        Self { patterns }
    }

    /// Scan text and return all PII matches found.
    ///
    /// Matches may overlap. They are returned in order of appearance.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn scan(&self, text: &str) -> Vec<PiiMatch> {
        let mut matches = Vec::new();

        for pattern in &self.patterns {
            for m in pattern.regex.find_iter(text) {
                matches.push(PiiMatch::new(pattern.pii_type.clone(), m.start(), m.end()));
            }
        }

        // Sort by start position for predictable ordering
        matches.sort_by_key(|m| m.start);
        matches
    }

    /// Mask PII in text, replacing only matched patterns with `[MASKED:type]`.
    ///
    /// Returns the masked text and all matches found.
    /// Surrounding text is preserved exactly.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn mask(&self, text: &str) -> (String, Vec<PiiMatch>) {
        let matches = self.scan(text);

        if matches.is_empty() {
            return (text.to_string(), matches);
        }

        // Build the masked string by replacing matches back-to-front
        // to preserve byte offsets. We process non-overlapping matches.
        let non_overlapping = deduplicate_overlapping(&matches);

        let mut result = String::with_capacity(text.len());
        let mut last_end = 0;

        for m in &non_overlapping {
            // Append text before this match
            result.push_str(&text[last_end..m.start]);
            // Append the mask token
            result.push_str(&m.masked_value);
            last_end = m.end;
        }
        // Append remaining text after last match
        result.push_str(&text[last_end..]);

        tracing::debug!(matches = non_overlapping.len(), "PII masking complete");

        (result, matches)
    }

    /// Classify the sensitivity level of text based on detected PII.
    ///
    /// - No matches: `Public`
    /// - Email/Phone/SSN/CreditCard: `Private`
    /// - API key/Password: `Restricted`
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn classify(&self, text: &str) -> SensitivityLevel {
        let matches = self.scan(text);

        if matches.is_empty() {
            return SensitivityLevel::Public;
        }

        // Check for high-sensitivity types first
        for m in &matches {
            match m.pattern_type {
                PiiType::ApiKey | PiiType::Password => {
                    return SensitivityLevel::Restricted;
                }
                _ => {}
            }
        }

        // Any other PII detected
        SensitivityLevel::Private
    }
}

impl Default for PiiFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Remove overlapping matches, preferring longer matches and earlier position.
///
/// Input must be sorted by start position.
fn deduplicate_overlapping(matches: &[PiiMatch]) -> Vec<&PiiMatch> {
    let mut result: Vec<&PiiMatch> = Vec::new();

    for m in matches {
        if let Some(last) = result.last() {
            // Skip if this match overlaps with the previous one
            if m.start < last.end {
                continue;
            }
        }
        result.push(m);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_api_keys() {
        let filter = PiiFilter::new();
        let text = "My key is sk-abc123def456ghi789jkl012mno";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect API key");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_openai_project_keys() {
        let filter = PiiFilter::new();
        let text = "Key: sk-proj-abc123def456ghi789jkl";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect OpenAI project key");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_anthropic_keys() {
        let filter = PiiFilter::new();
        let text = "Key: sk-ant-abc123def456ghi789";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect Anthropic key");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_groq_keys() {
        let filter = PiiFilter::new();
        let text = "Key: gsk_abc123def456ghi789jklmno";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect Groq key");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_xai_keys() {
        let filter = PiiFilter::new();
        let text = "Key: xai-abc123def456ghi789jklmno";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect xAI key");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_github_tokens() {
        let filter = PiiFilter::new();
        let text = "Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect GitHub PAT");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_gitlab_tokens() {
        let filter = PiiFilter::new();
        let text = "Token: glpat-abc123def456ghi789jkl";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect GitLab token");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_aws_keys() {
        let filter = PiiFilter::new();
        let text = "AWS key: AKIAIOSFODNN7EXAMPLE";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect AWS access key");
        assert_eq!(matches[0].pattern_type, PiiType::ApiKey);
    }

    #[test]
    fn detects_credit_cards() {
        let filter = PiiFilter::new();
        let text = "Card: 4111 1111 1111 1111";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect credit card");
        assert_eq!(matches[0].pattern_type, PiiType::CreditCard);
    }

    #[test]
    fn detects_ssn() {
        let filter = PiiFilter::new();
        let text = "SSN is 123-45-6789";
        let matches = filter.scan(text);
        assert!(
            matches.iter().any(|m| m.pattern_type == PiiType::Ssn),
            "should detect SSN"
        );
    }

    #[test]
    fn detects_emails() {
        let filter = PiiFilter::new();
        let text = "Contact me at user@example.com for details";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect email");
        assert_eq!(matches[0].pattern_type, PiiType::Email);
    }

    #[test]
    fn detects_passwords() {
        let filter = PiiFilter::new();
        let text = "password: mysecretP@ss123";
        let matches = filter.scan(text);
        assert!(!matches.is_empty(), "should detect password");
        assert_eq!(matches[0].pattern_type, PiiType::Password);
    }

    #[test]
    fn mask_replaces_only_pii() {
        let filter = PiiFilter::new();
        let text = "Send to user@example.com and call 555.123.4567 please";
        let (masked, matches) = filter.mask(text);

        assert!(!matches.is_empty());
        assert!(
            masked.contains("[MASKED:email]"),
            "should mask email, got: {masked}"
        );
        assert!(
            masked.contains("Send to"),
            "should preserve surrounding text"
        );
        assert!(masked.contains("please"), "should preserve trailing text");
        assert!(
            !masked.contains("user@example.com"),
            "should not contain original email"
        );
    }

    #[test]
    fn mask_preserves_clean_text() {
        let filter = PiiFilter::new();
        let text = "This is a clean sentence with no PII.";
        let (masked, matches) = filter.mask(text);
        assert!(matches.is_empty());
        assert_eq!(masked, text);
    }

    #[test]
    fn mask_sk_proj_key_preserves_surrounding_text() {
        let filter = PiiFilter::new();
        let text = "The project sk-proj-test123456789abcdef was created";
        let (masked, _) = filter.mask(text);

        assert!(
            masked.contains("The project "),
            "should preserve text before key, got: {masked}"
        );
        assert!(
            masked.contains(" was created"),
            "should preserve text after key, got: {masked}"
        );
        assert!(
            masked.contains("[MASKED:api_key]"),
            "should mask the key, got: {masked}"
        );
        assert!(
            !masked.contains("sk-proj-test123456789abcdef"),
            "should not contain original key"
        );
    }

    #[test]
    fn classify_public_for_clean_text() {
        let filter = PiiFilter::new();
        assert_eq!(filter.classify("Hello world"), SensitivityLevel::Public);
    }

    #[test]
    fn classify_private_for_email() {
        let filter = PiiFilter::new();
        assert_eq!(
            filter.classify("Contact admin@company.com"),
            SensitivityLevel::Private
        );
    }

    #[test]
    fn classify_restricted_for_api_key() {
        let filter = PiiFilter::new();
        assert_eq!(
            filter.classify("Use sk-abc123def456ghi789jkl012mno"),
            SensitivityLevel::Restricted
        );
    }

    #[test]
    fn classify_restricted_for_password() {
        let filter = PiiFilter::new();
        assert_eq!(
            filter.classify("password=hunter2"),
            SensitivityLevel::Restricted
        );
    }

    #[test]
    fn classify_restricted_for_groq_key() {
        let filter = PiiFilter::new();
        assert_eq!(
            filter.classify("Use gsk_abc123def456ghi789jklmno"),
            SensitivityLevel::Restricted
        );
    }

    #[test]
    fn classify_restricted_for_xai_key() {
        let filter = PiiFilter::new();
        assert_eq!(
            filter.classify("Use xai-abc123def456ghi789jklmno"),
            SensitivityLevel::Restricted
        );
    }
}
