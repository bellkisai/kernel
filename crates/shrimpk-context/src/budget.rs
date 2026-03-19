//! Token budget allocation for context assembly.
//!
//! Splits a model's context window into proportional budgets for each
//! context source: system prompt, echo memories, RAG documents,
//! conversation history, and response reserve.

use serde::{Deserialize, Serialize};

/// Token budget allocation across context sources.
///
/// Given a model's total context window, the budget allocates space
/// proportionally so that each source gets fair representation without
/// exceeding the model's limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Model's total context window in tokens.
    pub total: usize,
    /// Fixed allocation for the system prompt.
    pub system_prompt: usize,
    /// Allocation for Echo Memory results.
    pub echo_memories: usize,
    /// Allocation for RAG/document chunks.
    pub rag_context: usize,
    /// Allocation for recent conversation history.
    pub conversation: usize,
    /// Reserved for the model's output generation.
    pub response_reserve: usize,
}

impl TokenBudget {
    /// Create a budget for a given context window size.
    ///
    /// Allocates proportionally:
    /// - System prompt: 5%
    /// - Echo memories: 15%
    /// - RAG context: 20%
    /// - Conversation: 35%
    /// - Response reserve: 25%
    pub fn for_context_window(total_tokens: usize) -> Self {
        Self {
            total: total_tokens,
            system_prompt: total_tokens * 5 / 100,
            echo_memories: total_tokens * 15 / 100,
            rag_context: total_tokens * 20 / 100,
            conversation: total_tokens * 35 / 100,
            response_reserve: total_tokens * 25 / 100,
        }
    }

    /// Adaptive rebalancing: if Echo provides rich context, shift some
    /// conversation budget to echo to accommodate more memories.
    ///
    /// For each echo result beyond 5, transfer 1% of total from
    /// conversation to echo (up to 10% shift).
    pub fn with_echo_count(mut self, echo_count: usize) -> Self {
        if echo_count > 5 {
            let extra = (echo_count - 5).min(10);
            let shift = self.total * extra / 100;
            // Only shift if conversation has enough headroom
            if self.conversation > shift {
                self.conversation -= shift;
                self.echo_memories += shift;
            }
        }
        self
    }

    /// How many tokens remain for conversation after all fixed allocations.
    ///
    /// This is the conversation budget minus any overhead. If the sum of
    /// fixed allocations exceeds total, returns 0.
    pub fn conversation_budget(&self) -> usize {
        let fixed =
            self.system_prompt + self.echo_memories + self.rag_context + self.response_reserve;
        self.total.saturating_sub(fixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_allocation_proportions() {
        let budget = TokenBudget::for_context_window(100_000);
        assert_eq!(budget.system_prompt, 5_000);
        assert_eq!(budget.echo_memories, 15_000);
        assert_eq!(budget.rag_context, 20_000);
        assert_eq!(budget.conversation, 35_000);
        assert_eq!(budget.response_reserve, 25_000);
    }

    #[test]
    fn budget_proportions_small_window() {
        let budget = TokenBudget::for_context_window(4_096);
        assert_eq!(budget.system_prompt, 204); // 5% of 4096
        assert_eq!(budget.echo_memories, 614); // 15% of 4096
        assert_eq!(budget.rag_context, 819); // 20% of 4096
        assert_eq!(budget.conversation, 1_433); // 35% of 4096
        assert_eq!(budget.response_reserve, 1_024); // 25% of 4096
    }

    #[test]
    fn conversation_budget_returns_remaining() {
        let budget = TokenBudget::for_context_window(100_000);
        // fixed = 5000 + 15000 + 20000 + 25000 = 65000
        // remaining = 100000 - 65000 = 35000
        assert_eq!(budget.conversation_budget(), 35_000);
    }

    #[test]
    fn adaptive_budget_shifts_for_rich_echo() {
        let budget = TokenBudget::for_context_window(100_000).with_echo_count(10);
        // 10 results: 5 extra beyond threshold, shift 5% = 5000 tokens
        assert_eq!(budget.echo_memories, 20_000); // 15000 + 5000
        assert_eq!(budget.conversation, 30_000); // 35000 - 5000
    }

    #[test]
    fn adaptive_budget_no_shift_for_few_echoes() {
        let budget = TokenBudget::for_context_window(100_000).with_echo_count(3);
        assert_eq!(budget.echo_memories, 15_000);
        assert_eq!(budget.conversation, 35_000);
    }

    #[test]
    fn adaptive_budget_caps_shift_at_10_percent() {
        let budget = TokenBudget::for_context_window(100_000).with_echo_count(50);
        // 50 results: 45 extra, capped at 10 → shift 10% = 10000
        assert_eq!(budget.echo_memories, 25_000); // 15000 + 10000
        assert_eq!(budget.conversation, 25_000); // 35000 - 10000
    }

    #[test]
    fn budget_zero_window() {
        let budget = TokenBudget::for_context_window(0);
        assert_eq!(budget.total, 0);
        assert_eq!(budget.conversation_budget(), 0);
    }
}
