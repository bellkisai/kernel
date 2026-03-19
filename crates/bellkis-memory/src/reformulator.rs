//! Memory reformulation for improved echo recall.
//!
//! KS1 precision tuning showed rewritten memories score ~9% higher on
//! cosine similarity than natural text. This module rewrites common
//! preference/fact patterns into structured forms that embed better.
//!
//! Phase 1: regex-based pattern matching (no LLM dependency).
//! Phase 2: optional LLM-based reformulation for complex sentences.

use bellkis_core::MemoryCategory;
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

    /// Classify text into a [`MemoryCategory`] for adaptive decay.
    ///
    /// Uses keyword heuristics to assign the most appropriate category.
    /// Falls back to `Conversation` for unmatched text (short-lived by default).
    pub fn categorize(&self, text: &str) -> MemoryCategory {
        let lower = text.to_lowercase();

        // Identity patterns — personal facts that rarely change
        if lower.contains("my name is")
            || lower.contains("i live in")
            || lower.contains("i am from")
            || lower.contains("i speak")
            || lower.contains("my email")
            || lower.contains("i was born")
        {
            return MemoryCategory::Identity;
        }

        // Active project patterns — current work context
        if lower.contains("currently building")
            || lower.contains("working on")
            || lower.contains("my project")
            || lower.contains("i'm building")
            || lower.contains("the project uses")
            || lower.contains("our app")
        {
            return MemoryCategory::ActiveProject;
        }

        // Preference patterns — tool/workflow choices
        if lower.contains("i prefer")
            || lower.contains("i always use")
            || lower.contains("my favorite")
            || lower.contains("i always choose")
            || lower.contains("my go-to")
            || lower.contains("i recommend")
            || lower.contains("my preferred")
            || lower.contains("i like to use")
        {
            return MemoryCategory::Preference;
        }

        // Fact patterns — learned information
        if lower.contains("the speed of")
            || lower.contains("the population")
            || lower.contains("was invented")
            || lower.contains("is known for")
            || lower.contains("was founded")
            || lower.contains("according to")
        {
            return MemoryCategory::Fact;
        }

        // Default: conversation — one-off discussions, fades fastest
        MemoryCategory::Conversation
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

    // --- Auto-categorization tests ---

    #[test]
    fn categorize_preference() {
        let r = MemoryReformulator::new();
        assert_eq!(r.categorize("I prefer Python for backend work"), MemoryCategory::Preference);
        assert_eq!(r.categorize("My favorite tool is Neovim"), MemoryCategory::Preference);
        assert_eq!(r.categorize("I always use dark mode"), MemoryCategory::Preference);
        assert_eq!(r.categorize("I always choose Rust over Go"), MemoryCategory::Preference);
        assert_eq!(r.categorize("My go-to framework is React"), MemoryCategory::Preference);
        assert_eq!(r.categorize("I recommend FastAPI"), MemoryCategory::Preference);
        assert_eq!(r.categorize("My preferred editor is VSCode"), MemoryCategory::Preference);
        assert_eq!(r.categorize("I like to use TypeScript"), MemoryCategory::Preference);
    }

    #[test]
    fn categorize_active_project() {
        let r = MemoryReformulator::new();
        assert_eq!(r.categorize("Currently building Bellkis"), MemoryCategory::ActiveProject);
        assert_eq!(r.categorize("I'm working on a new feature"), MemoryCategory::ActiveProject);
        assert_eq!(r.categorize("My project is a Tauri app"), MemoryCategory::ActiveProject);
        assert_eq!(r.categorize("I'm building an AI assistant"), MemoryCategory::ActiveProject);
        assert_eq!(r.categorize("The project uses Rust and React"), MemoryCategory::ActiveProject);
        assert_eq!(r.categorize("Our app supports multiple languages"), MemoryCategory::ActiveProject);
    }

    #[test]
    fn categorize_identity() {
        let r = MemoryReformulator::new();
        assert_eq!(r.categorize("My name is Lior"), MemoryCategory::Identity);
        assert_eq!(r.categorize("I live in Tel Aviv"), MemoryCategory::Identity);
        assert_eq!(r.categorize("I am from Israel"), MemoryCategory::Identity);
        assert_eq!(r.categorize("I speak Hebrew and English"), MemoryCategory::Identity);
        assert_eq!(r.categorize("My email is user@example.com"), MemoryCategory::Identity);
        assert_eq!(r.categorize("I was born in 1990"), MemoryCategory::Identity);
    }

    #[test]
    fn categorize_fact() {
        let r = MemoryReformulator::new();
        assert_eq!(r.categorize("The speed of light is 299,792,458 m/s"), MemoryCategory::Fact);
        assert_eq!(r.categorize("The population of Tokyo is about 14 million"), MemoryCategory::Fact);
        assert_eq!(r.categorize("The telephone was invented by Alexander Graham Bell"), MemoryCategory::Fact);
        assert_eq!(r.categorize("Rust is known for memory safety"), MemoryCategory::Fact);
        assert_eq!(r.categorize("Google was founded in 1998"), MemoryCategory::Fact);
        assert_eq!(r.categorize("According to the docs, this API requires auth"), MemoryCategory::Fact);
    }

    #[test]
    fn categorize_conversation_default() {
        let r = MemoryReformulator::new();
        assert_eq!(r.categorize("We discussed the bug yesterday"), MemoryCategory::Conversation);
        assert_eq!(r.categorize("Can you help me fix this error?"), MemoryCategory::Conversation);
        assert_eq!(r.categorize("Thanks, that worked!"), MemoryCategory::Conversation);
        assert_eq!(r.categorize("Random unmatched text"), MemoryCategory::Conversation);
    }

    #[test]
    fn categorize_case_insensitive() {
        let r = MemoryReformulator::new();
        assert_eq!(r.categorize("MY NAME IS LIOR"), MemoryCategory::Identity);
        assert_eq!(r.categorize("i prefer python"), MemoryCategory::Preference);
        assert_eq!(r.categorize("CURRENTLY BUILDING something"), MemoryCategory::ActiveProject);
    }
}
