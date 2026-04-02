//! Context assembler — the main compilation logic.
//!
//! Combines system prompt, echo memories, RAG documents, and conversation
//! history into a single assembled context that fits within the model's
//! token budget. Priority-based truncation ensures the most valuable
//! context is preserved when space is tight.

use shrimpk_core::EchoResult;
use tracing::debug;

use crate::budget::TokenBudget;
use crate::segment::ContextSegment;

/// Configuration for the context assembler.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Default system prompt used when no override is provided.
    pub default_system_prompt: String,
    /// Maximum number of echo results to include (default: 10).
    pub max_echo_results: usize,
    /// Maximum number of RAG document chunks to include (default: 5).
    pub max_rag_chunks: usize,
    /// Maximum number of conversation turns to include (default: 20).
    pub max_conversation_turns: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            default_system_prompt: String::new(),
            max_echo_results: 10,
            max_rag_chunks: 5,
            max_conversation_turns: 20,
        }
    }
}

/// The result of context assembly — a compiled prompt ready for the model.
#[derive(Debug, Clone)]
pub struct AssembledContext {
    /// Ordered segments composing the final prompt.
    pub segments: Vec<ContextSegment>,
    /// Total estimated token count of all segments.
    pub total_tokens: usize,
    /// The token budget that was used for assembly.
    pub budget: TokenBudget,
    /// Descriptions of what was truncated to fit within budget.
    pub truncated: Vec<String>,
}

/// Assembles context from multiple sources into a token-budgeted prompt.
///
/// The assembler respects a strict priority order:
/// 1. System prompt (always included, never truncated)
/// 2. Echo memories (protected from truncation)
/// 3. RAG documents (truncated before echo)
/// 4. Conversation history (first to be truncated, most recent kept)
/// 5. User query (always included, never truncated)
#[derive(Debug, Clone)]
pub struct ContextAssembler {
    config: ContextConfig,
}

impl ContextAssembler {
    /// Create a new assembler with the given configuration.
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    /// Assemble context from multiple sources, fitting within the token budget.
    ///
    /// # Arguments
    /// - `context_window`: the model's maximum context window in tokens
    /// - `system_prompt`: optional override for the default system prompt
    /// - `echo_results`: echo memory results (from Echo Memory subsystem)
    /// - `rag_chunks`: RAG document chunks (from document retrieval)
    /// - `conversation`: conversation history as `(role, content)` pairs
    /// - `user_query`: the current user query
    ///
    /// # Returns
    /// An `AssembledContext` containing the ordered segments, token count,
    /// budget, and a log of what was truncated.
    pub fn assemble(
        &self,
        context_window: usize,
        system_prompt: Option<&str>,
        echo_results: &[EchoResult],
        rag_chunks: &[String],
        conversation: &[(String, String)],
        user_query: &str,
    ) -> AssembledContext {
        let echo_count = echo_results.len().min(self.config.max_echo_results);
        let budget = TokenBudget::for_context_window(context_window).with_echo_count(echo_count);

        let mut segments = Vec::new();
        let mut truncated = Vec::new();
        let mut used_tokens: usize = 0;

        // The usable budget is total minus response reserve.
        let usable = budget.total.saturating_sub(budget.response_reserve);

        // --- Step 1: System prompt (always included, priority 0) ---
        let sys_text = system_prompt.unwrap_or(&self.config.default_system_prompt);
        let sys_segment = ContextSegment::SystemPrompt(sys_text.to_string());
        let sys_tokens = sys_segment.token_estimate();
        used_tokens += sys_tokens;
        segments.push(sys_segment);

        // --- Step 2: Echo memories up to echo budget (priority 1) ---
        let echo_limit = echo_count;
        let mut echo_tokens_used: usize = 0;
        let mut echo_added = 0;
        for result in echo_results.iter().take(echo_limit) {
            let seg = ContextSegment::EchoMemory {
                content: result.content.clone(),
                similarity: result.similarity,
                source: result.source.clone(),
            };
            let tok = seg.token_estimate();
            if echo_tokens_used + tok <= budget.echo_memories {
                echo_tokens_used += tok;
                used_tokens += tok;
                segments.push(seg);
                echo_added += 1;
            } else {
                truncated.push(format!(
                    "Echo memory from '{}' (similarity {:.2}) — exceeded echo budget",
                    result.source, result.similarity,
                ));
            }
        }
        if echo_added < echo_results.len() && echo_results.len() > echo_limit {
            truncated.push(format!(
                "{} echo results exceeded max_echo_results limit ({})",
                echo_results.len() - echo_limit,
                self.config.max_echo_results,
            ));
        }

        // --- Step 3: RAG chunks up to RAG budget (priority 2) ---
        let rag_limit = rag_chunks.len().min(self.config.max_rag_chunks);
        let mut rag_tokens_used: usize = 0;
        let mut rag_added = 0;
        for (i, chunk) in rag_chunks.iter().take(rag_limit).enumerate() {
            let seg = ContextSegment::RagDocument {
                content: chunk.clone(),
                source: format!("chunk_{}", i),
            };
            let tok = seg.token_estimate();
            if rag_tokens_used + tok <= budget.rag_context {
                rag_tokens_used += tok;
                used_tokens += tok;
                segments.push(seg);
                rag_added += 1;
            } else {
                truncated.push(format!(
                    "RAG chunk {} ({} tokens) — exceeded RAG budget",
                    i, tok,
                ));
            }
        }
        if rag_added < rag_chunks.len() && rag_chunks.len() > rag_limit {
            truncated.push(format!(
                "{} RAG chunks exceeded max_rag_chunks limit ({})",
                rag_chunks.len() - rag_limit,
                self.config.max_rag_chunks,
            ));
        }

        // --- Step 4: User query (always included, priority 0) ---
        let query_segment = ContextSegment::UserQuery(user_query.to_string());
        let query_tokens = query_segment.token_estimate();
        used_tokens += query_tokens;

        // --- Step 5: Conversation history, most recent first, up to budget ---
        let conv_limit = conversation.len().min(self.config.max_conversation_turns);
        let mut conv_segments: Vec<ContextSegment> = Vec::new();
        let mut conv_tokens_used: usize = 0;
        // Iterate from most recent backwards, keeping the most recent messages.
        for (i, (role, content)) in conversation.iter().rev().take(conv_limit).enumerate() {
            let seg = ContextSegment::ConversationMessage {
                role: role.clone(),
                content: content.clone(),
            };
            let tok = seg.token_estimate();
            if conv_tokens_used + tok <= budget.conversation
                && used_tokens + conv_tokens_used + tok <= usable
            {
                conv_tokens_used += tok;
                conv_segments.push(seg);
            } else {
                let remaining = conversation.len().saturating_sub(i);
                truncated.push(format!(
                    "{} older conversation turn(s) truncated to fit budget",
                    remaining,
                ));
                break;
            }
        }
        // Reverse to restore chronological order.
        conv_segments.reverse();
        used_tokens += conv_tokens_used;
        segments.extend(conv_segments);

        // Add user query at the end.
        segments.push(query_segment);

        // --- Step 6: Over-budget truncation ---
        // If we're still over budget, truncate conversation first, then RAG.
        // Never truncate echo or system.
        while used_tokens > usable {
            // Try removing conversation messages (from the oldest).
            if let Some(pos) = segments
                .iter()
                .position(|s| matches!(s, ContextSegment::ConversationMessage { .. }))
            {
                let removed = segments.remove(pos);
                let tok = removed.token_estimate();
                used_tokens = used_tokens.saturating_sub(tok);
                truncated.push(format!(
                    "Conversation message truncated during over-budget pass ({} tokens freed)",
                    tok,
                ));
                continue;
            }
            // Try removing RAG chunks (from the last).
            if let Some(pos) = segments
                .iter()
                .rposition(|s| matches!(s, ContextSegment::RagDocument { .. }))
            {
                let removed = segments.remove(pos);
                let tok = removed.token_estimate();
                used_tokens = used_tokens.saturating_sub(tok);
                truncated.push(format!(
                    "RAG chunk truncated during over-budget pass ({} tokens freed)",
                    tok,
                ));
                continue;
            }
            // Nothing left to truncate — break to avoid infinite loop.
            break;
        }

        debug!(
            total_tokens = used_tokens,
            segments = segments.len(),
            truncated = truncated.len(),
            "Context assembled"
        );

        AssembledContext {
            segments,
            total_tokens: used_tokens,
            budget,
            truncated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use shrimpk_core::{EchoResult, MemoryId};

    fn make_echo(content: &str, similarity: f32) -> EchoResult {
        EchoResult {
            memory_id: MemoryId::new(),
            content: content.into(),
            similarity,
            final_score: similarity as f64,
            source: "test".into(),
            echoed_at: Utc::now(),
            modality: Default::default(),
            labels: Vec::new(),
        }
    }

    fn make_config() -> ContextConfig {
        ContextConfig {
            default_system_prompt: "You are a helpful assistant.".into(),
            max_echo_results: 10,
            max_rag_chunks: 5,
            max_conversation_turns: 20,
        }
    }

    #[test]
    fn assemble_fits_within_budget() {
        let assembler = ContextAssembler::new(make_config());
        let echoes = vec![make_echo("memory one", 0.9)];
        let rag = vec!["document chunk".to_string()];
        let conv = vec![
            ("user".to_string(), "hello".to_string()),
            ("assistant".to_string(), "hi there".to_string()),
        ];

        let result = assembler.assemble(8_000, None, &echoes, &rag, &conv, "what is X?");

        // Total tokens must not exceed usable budget (total - response_reserve)
        let usable = result.budget.total - result.budget.response_reserve;
        assert!(
            result.total_tokens <= usable,
            "total {} exceeds usable {}",
            result.total_tokens,
            usable,
        );
    }

    #[test]
    fn echo_memories_appear_before_conversation() {
        let assembler = ContextAssembler::new(make_config());
        let echoes = vec![make_echo("recalled fact", 0.85)];
        let conv = vec![("user".to_string(), "earlier message".to_string())];

        let result = assembler.assemble(8_000, None, &echoes, &[], &conv, "query");

        let echo_pos = result
            .segments
            .iter()
            .position(|s| matches!(s, ContextSegment::EchoMemory { .. }));
        let conv_pos = result
            .segments
            .iter()
            .position(|s| matches!(s, ContextSegment::ConversationMessage { .. }));

        assert!(echo_pos.is_some(), "echo segment missing");
        assert!(conv_pos.is_some(), "conversation segment missing");
        assert!(
            echo_pos.unwrap() < conv_pos.unwrap(),
            "echo (pos {}) should come before conversation (pos {})",
            echo_pos.unwrap(),
            conv_pos.unwrap(),
        );
    }

    #[test]
    fn truncation_removes_conversation_first_never_echo() {
        let config = ContextConfig {
            default_system_prompt: "sys".into(),
            max_echo_results: 10,
            max_rag_chunks: 5,
            max_conversation_turns: 100,
        };
        let assembler = ContextAssembler::new(config);

        let echoes = vec![make_echo("important memory", 0.95)];
        // Create large conversation that will exceed budget.
        let conv: Vec<(String, String)> = (0..100)
            .map(|i| {
                (
                    "user".to_string(),
                    format!("message {} {}", i, "x".repeat(200)),
                )
            })
            .collect();

        // Tiny context window to force truncation.
        let result = assembler.assemble(500, None, &echoes, &[], &conv, "q");

        // Echo must still be present.
        let has_echo = result
            .segments
            .iter()
            .any(|s| matches!(s, ContextSegment::EchoMemory { .. }));
        assert!(has_echo, "echo memories must never be truncated");

        // System prompt must still be present.
        let has_system = result
            .segments
            .iter()
            .any(|s| matches!(s, ContextSegment::SystemPrompt(_)));
        assert!(has_system, "system prompt must never be truncated");

        // Something should have been truncated.
        assert!(
            !result.truncated.is_empty(),
            "truncation log should not be empty"
        );
    }

    #[test]
    fn empty_echo_produces_valid_context() {
        let assembler = ContextAssembler::new(make_config());

        let result = assembler.assemble(
            8_000,
            Some("custom system prompt"),
            &[], // no echoes
            &[], // no RAG
            &[("user".to_string(), "hi".to_string())],
            "what's up?",
        );

        assert!(
            result.segments.len() >= 3,
            "should have system + conv + query"
        );

        let has_system = result
            .segments
            .iter()
            .any(|s| matches!(s, ContextSegment::SystemPrompt(_)));
        let has_query = result
            .segments
            .iter()
            .any(|s| matches!(s, ContextSegment::UserQuery(_)));
        let has_conv = result
            .segments
            .iter()
            .any(|s| matches!(s, ContextSegment::ConversationMessage { .. }));

        assert!(has_system);
        assert!(has_query);
        assert!(has_conv);
    }

    #[test]
    fn over_budget_truncation_log_records_cuts() {
        let config = ContextConfig {
            default_system_prompt: "system".into(),
            max_echo_results: 10,
            max_rag_chunks: 50,
            max_conversation_turns: 100,
        };
        let assembler = ContextAssembler::new(config);

        // Large RAG chunks + conversation to overflow a small window.
        let rag: Vec<String> = (0..10)
            .map(|i| format!("chunk {} {}", i, "y".repeat(500)))
            .collect();
        let conv: Vec<(String, String)> = (0..50)
            .map(|i| ("user".to_string(), format!("msg {} {}", i, "z".repeat(300))))
            .collect();

        let result = assembler.assemble(1_000, None, &[], &rag, &conv, "query");

        assert!(
            !result.truncated.is_empty(),
            "should have truncation entries"
        );

        let usable = result.budget.total - result.budget.response_reserve;
        assert!(
            result.total_tokens <= usable,
            "final tokens {} must fit usable {}",
            result.total_tokens,
            usable,
        );
    }

    #[test]
    fn system_prompt_override_works() {
        let assembler = ContextAssembler::new(make_config());

        let result = assembler.assemble(8_000, Some("custom override"), &[], &[], &[], "q");

        if let ContextSegment::SystemPrompt(ref text) = result.segments[0] {
            assert_eq!(text, "custom override");
        } else {
            panic!("first segment should be SystemPrompt");
        }
    }

    #[test]
    fn user_query_is_last_segment() {
        let assembler = ContextAssembler::new(make_config());
        let echoes = vec![make_echo("mem", 0.8)];
        let conv = vec![("user".to_string(), "hi".to_string())];

        let result = assembler.assemble(8_000, None, &echoes, &[], &conv, "my question");

        let last = result.segments.last().expect("should have segments");
        match last {
            ContextSegment::UserQuery(q) => assert_eq!(q, "my question"),
            other => panic!("last segment should be UserQuery, got {:?}", other),
        }
    }

    #[test]
    fn adaptive_budget_more_echo_less_conversation() {
        let assembler = ContextAssembler::new(make_config());
        let echoes: Vec<EchoResult> = (0..10)
            .map(|i| make_echo(&format!("echo {}", i), 0.9))
            .collect();

        let result = assembler.assemble(100_000, None, &echoes, &[], &[], "q");

        // With 10 echo results, budget shifts 5% from conversation to echo.
        assert_eq!(result.budget.echo_memories, 20_000);
        assert_eq!(result.budget.conversation, 30_000);
    }
}
