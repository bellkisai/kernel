//! Context segments — typed building blocks for assembled prompts.
//!
//! Each segment represents a distinct source of context (system prompt,
//! echo memory, RAG document, conversation message, or user query) and
//! carries metadata for priority-based assembly and token estimation.

use serde::{Deserialize, Serialize};

/// A typed segment of context to be assembled into a prompt.
///
/// Segments are ordered by priority during assembly: lower priority
/// values are placed first and protected from truncation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextSegment {
    /// The system prompt — always included, never truncated.
    SystemPrompt(String),
    /// An echo memory result — high-value recalled context.
    EchoMemory {
        content: String,
        similarity: f32,
        source: String,
    },
    /// A RAG document chunk — retrieved from document search.
    RagDocument { content: String, source: String },
    /// A conversation message — recent chat history.
    ConversationMessage { role: String, content: String },
    /// The current user query — always included, never truncated.
    UserQuery(String),
}

impl ContextSegment {
    /// Estimate the token count for this segment.
    ///
    /// Uses the ~4 characters per token heuristic, which is a reasonable
    /// approximation for English text with most tokenizers.
    pub fn token_estimate(&self) -> usize {
        let chars = match self {
            Self::SystemPrompt(s) => s.len(),
            Self::EchoMemory {
                content, source, ..
            } => content.len() + source.len() + 20,
            Self::RagDocument { content, source } => content.len() + source.len() + 10,
            Self::ConversationMessage { role, content } => role.len() + content.len() + 5,
            Self::UserQuery(s) => s.len(),
        };
        // Ceiling division: round up to avoid underestimation
        chars.div_ceil(4)
    }

    /// Priority value for assembly ordering.
    ///
    /// Lower values = higher priority = placed first and last to be truncated.
    /// - 0: System prompt (never truncated)
    /// - 1: Echo memories (protected)
    /// - 2: RAG documents
    /// - 3: Conversation history (first to be truncated)
    /// - 0: User query (never truncated, same as system)
    pub fn priority(&self) -> u8 {
        match self {
            Self::SystemPrompt(_) => 0,
            Self::EchoMemory { .. } => 1,
            Self::RagDocument { .. } => 2,
            Self::ConversationMessage { .. } => 3,
            Self::UserQuery(_) => 0,
        }
    }

    /// Extract the text content of this segment.
    pub fn content(&self) -> &str {
        match self {
            Self::SystemPrompt(s) | Self::UserQuery(s) => s,
            Self::EchoMemory { content, .. }
            | Self::RagDocument { content, .. }
            | Self::ConversationMessage { content, .. } => content,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_approximately_4_chars_per_token() {
        let segment = ContextSegment::SystemPrompt("a".repeat(400));
        // 400 chars → ~100 tokens (with ceiling: (400+3)/4 = 100)
        assert_eq!(segment.token_estimate(), 100);
    }

    #[test]
    fn token_estimate_rounds_up() {
        let segment = ContextSegment::UserQuery("hello".into()); // 5 chars
        // (5+3)/4 = 2
        assert_eq!(segment.token_estimate(), 2);
    }

    #[test]
    fn token_estimate_empty_string() {
        let segment = ContextSegment::SystemPrompt(String::new());
        // (0+3)/4 = 0 (integer division)
        assert_eq!(segment.token_estimate(), 0);
    }

    #[test]
    fn token_estimate_echo_includes_metadata() {
        let segment = ContextSegment::EchoMemory {
            content: "memory text".into(),
            similarity: 0.95,
            source: "conversation".into(),
        };
        // content(11) + source(12) + overhead(20) = 43 chars → (43+3)/4 = 11
        assert_eq!(segment.token_estimate(), 11);
    }

    #[test]
    fn priority_system_prompt_is_highest() {
        let sys = ContextSegment::SystemPrompt("test".into());
        assert_eq!(sys.priority(), 0);
    }

    #[test]
    fn priority_echo_before_rag() {
        let echo = ContextSegment::EchoMemory {
            content: "x".into(),
            similarity: 0.9,
            source: "s".into(),
        };
        let rag = ContextSegment::RagDocument {
            content: "x".into(),
            source: "s".into(),
        };
        assert!(echo.priority() < rag.priority());
    }

    #[test]
    fn priority_rag_before_conversation() {
        let rag = ContextSegment::RagDocument {
            content: "x".into(),
            source: "s".into(),
        };
        let conv = ContextSegment::ConversationMessage {
            role: "user".into(),
            content: "x".into(),
        };
        assert!(rag.priority() < conv.priority());
    }

    #[test]
    fn priority_user_query_same_as_system() {
        let query = ContextSegment::UserQuery("test".into());
        assert_eq!(query.priority(), 0);
    }

    #[test]
    fn content_extraction() {
        let seg = ContextSegment::ConversationMessage {
            role: "assistant".into(),
            content: "hello world".into(),
        };
        assert_eq!(seg.content(), "hello world");
    }
}
