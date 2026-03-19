//! Memory reformulation for improved echo recall.
//!
//! KS1 precision tuning showed rewritten memories score ~9% higher on
//! cosine similarity than natural text. This module rewrites common
//! preference/fact patterns into structured forms that embed better.
//!
//! Phase 1: regex-based pattern matching (no LLM dependency).
//! Phase 2: optional LLM-based reformulation for complex sentences.

use regex::Regex;

/// A compiled reformulation rule: regex pattern + replacement template.
struct ReformulationRule {
    regex: Regex,
    /// Template string with capture group references ($1, $2, etc.).
    template: &'static str,
}

/// Reformulates natural-language memory text into structured forms
/// that produce higher cosine similarity scores during echo recall.
///
/// Example:
/// - "I prefer FastAPI for REST APIs" -> "Preference: FastAPI for REST APIs"
/// - "I use Neovim" -> "Tool/technology: Neovim"
///
/// If no pattern matches, returns `None` (the original text is kept as-is).
pub struct MemoryReformulator {
    rules: Vec<ReformulationRule>,
}

impl MemoryReformulator {
    /// Create a new reformulator with all built-in rules compiled.
    ///
    /// # Panics
    /// Panics if any built-in regex pattern fails to compile (indicates a bug).
    pub fn new() -> Self {
        // Rules are ordered from most specific to least specific.
        // Case-insensitive matching on the leading verb/phrase.
        let rule_defs: Vec<(&str, &'static str)> = vec![
            // "I always choose X over Y" -> "Preference: X over Y"
            (r"(?i)^I\s+always\s+choose\s+(.+?)\s+over\s+(.+)$", "Preference: $1 over $2"),
            // "My favorite X is Y" -> "Favorite X: Y"
            (r"(?i)^My\s+fav(?:orite|ourite)\s+(.+?)\s+is\s+(.+)$", "Favorite $1: $2"),
            // "I prefer X for Y" -> "Preference: X for Y"
            (r"(?i)^I\s+prefer\s+(.+?)\s+for\s+(.+)$", "Preference: $1 for $2"),
            // "I prefer X" (without "for") -> "Preference: X"
            (r"(?i)^I\s+prefer\s+(.+)$", "Preference: $1"),
            // "I'm building X" / "I am building X" -> "Active project: X"
            (r"(?i)^I'?m\s+building\s+(.+)$", "Active project: $1"),
            (r"(?i)^I\s+am\s+building\s+(.+)$", "Active project: $1"),
            // "Currently building X" -> "Active project: X"
            (r"(?i)^Currently\s+building\s+(.+)$", "Active project: $1"),
            // "I live in X" -> "Location: X"
            (r"(?i)^I\s+live\s+in\s+(.+)$", "Location: $1"),
            // "I speak X" -> "Language: X"
            (r"(?i)^I\s+speak\s+(.+)$", "Language: $1"),
            // "I work with X" -> "Technology: X"
            (r"(?i)^I\s+work\s+with\s+(.+)$", "Technology: $1"),
            // "I use X" -> "Tool/technology: X"
            (r"(?i)^I\s+use\s+(.+)$", "Tool/technology: $1"),
        ];

        let rules = rule_defs
            .into_iter()
            .map(|(pat, template)| {
                let regex = Regex::new(pat).unwrap_or_else(|e| {
                    panic!("Bug: reformulation pattern '{pat}' failed to compile: {e}")
                });
                ReformulationRule { regex, template }
            })
            .collect();

        Self { rules }
    }

    /// Try to reformulate text into a structured form for better embedding.
    ///
    /// Returns `Some(reformulated)` if a pattern matched, `None` otherwise.
    /// The caller should embed the reformulated text but store the original
    /// for display to the user.
    pub fn reformulate(&self, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        for rule in &self.rules {
            if rule.regex.is_match(trimmed) {
                let result = rule.regex.replace(trimmed, rule.template).to_string();
                // Only return if the reformulation actually changed the text
                if result != trimmed {
                    return Some(result);
                }
            }
        }

        None
    }
}

impl Default for MemoryReformulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reformulate_prefer_for() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I prefer FastAPI for REST APIs"),
            Some("Preference: FastAPI for REST APIs".to_string())
        );
    }

    #[test]
    fn reformulate_prefer_simple() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I prefer dark mode"),
            Some("Preference: dark mode".to_string())
        );
    }

    #[test]
    fn reformulate_use() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I use Neovim"),
            Some("Tool/technology: Neovim".to_string())
        );
    }

    #[test]
    fn reformulate_favorite() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("My favorite language is Rust"),
            Some("Favorite language: Rust".to_string())
        );
    }

    #[test]
    fn reformulate_choose_over() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I always choose Rust over Go"),
            Some("Preference: Rust over Go".to_string())
        );
    }

    #[test]
    fn reformulate_work_with() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I work with PostgreSQL"),
            Some("Technology: PostgreSQL".to_string())
        );
    }

    #[test]
    fn reformulate_building_contraction() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I'm building Bellkis"),
            Some("Active project: Bellkis".to_string())
        );
    }

    #[test]
    fn reformulate_currently_building() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("Currently building Bellkis"),
            Some("Active project: Bellkis".to_string())
        );
    }

    #[test]
    fn reformulate_live_in() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I live in Tel Aviv"),
            Some("Location: Tel Aviv".to_string())
        );
    }

    #[test]
    fn reformulate_speak() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I speak Hebrew"),
            Some("Language: Hebrew".to_string())
        );
    }

    #[test]
    fn reformulate_no_match_returns_none() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("Random text with no pattern"),
            None
        );
    }

    #[test]
    fn reformulate_empty_returns_none() {
        let r = MemoryReformulator::new();
        assert_eq!(r.reformulate(""), None);
        assert_eq!(r.reformulate("  "), None);
    }

    #[test]
    fn reformulate_case_insensitive() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("i prefer Python for scripting"),
            Some("Preference: Python for scripting".to_string())
        );
    }

    #[test]
    fn reformulate_i_am_building() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("I am building a startup"),
            Some("Active project: a startup".to_string())
        );
    }

    #[test]
    fn reformulate_favourite_british_spelling() {
        let r = MemoryReformulator::new();
        assert_eq!(
            r.reformulate("My favourite editor is VSCode"),
            Some("Favorite editor: VSCode".to_string())
        );
    }
}
