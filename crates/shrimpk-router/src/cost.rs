//! Cost tracking and budget enforcement.
//!
//! Records per-provider token usage, computes running totals,
//! and enforces daily/monthly spending budgets.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use shrimpk_core::traits::ProviderId;

/// Tracks costs across all providers with optional budget limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTracker {
    /// Per-provider usage records.
    usage: HashMap<ProviderId, ProviderUsage>,
    /// Optional daily budget limit in USD.
    daily_budget: Option<f64>,
    /// Optional monthly budget limit in USD.
    monthly_budget: Option<f64>,
    /// Running total for the current day.
    daily_total: f64,
    /// Running total for the current month.
    monthly_total: f64,
}

/// Usage statistics for a single provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderUsage {
    /// Total input tokens consumed.
    pub total_input_tokens: u64,
    /// Total output tokens consumed.
    pub total_output_tokens: u64,
    /// Total cost in USD.
    pub total_cost_usd: f64,
    /// Number of requests made.
    pub request_count: u64,
    /// Timestamp of the most recent request.
    pub last_used: DateTime<Utc>,
}

impl CostTracker {
    /// Create a new cost tracker with no budget limits.
    pub fn new() -> Self {
        Self {
            usage: HashMap::new(),
            daily_budget: None,
            monthly_budget: None,
            daily_total: 0.0,
            monthly_total: 0.0,
        }
    }

    /// Create a cost tracker with budget limits.
    pub fn with_budgets(daily: Option<f64>, monthly: Option<f64>) -> Self {
        Self {
            usage: HashMap::new(),
            daily_budget: daily,
            monthly_budget: monthly,
            daily_total: 0.0,
            monthly_total: 0.0,
        }
    }

    /// Record a completed request's usage.
    pub fn record(
        &mut self,
        provider: &ProviderId,
        input_tokens: u64,
        output_tokens: u64,
        cost: f64,
    ) {
        let entry = self
            .usage
            .entry(provider.clone())
            .or_insert_with(|| ProviderUsage {
                total_input_tokens: 0,
                total_output_tokens: 0,
                total_cost_usd: 0.0,
                request_count: 0,
                last_used: Utc::now(),
            });

        entry.total_input_tokens += input_tokens;
        entry.total_output_tokens += output_tokens;
        entry.total_cost_usd += cost;
        entry.request_count += 1;
        entry.last_used = Utc::now();

        self.daily_total += cost;
        self.monthly_total += cost;
    }

    /// Get usage for a specific provider.
    pub fn get_usage(&self, provider: &ProviderId) -> Option<&ProviderUsage> {
        self.usage.get(provider)
    }

    /// Total cost across all providers.
    pub fn total_cost(&self) -> f64 {
        self.usage.values().map(|u| u.total_cost_usd).sum()
    }

    /// Check if spending is within both daily and monthly budgets.
    pub fn is_within_budget(&self) -> bool {
        if let Some(daily) = self.daily_budget
            && self.daily_total >= daily
        {
            return false;
        }
        if let Some(monthly) = self.monthly_budget
            && self.monthly_total >= monthly
        {
            return false;
        }
        true
    }

    /// Check if a proposed cost would stay within budget.
    pub fn would_be_within_budget(&self, additional_cost: f64) -> bool {
        if let Some(daily) = self.daily_budget
            && self.daily_total + additional_cost > daily
        {
            return false;
        }
        if let Some(monthly) = self.monthly_budget
            && self.monthly_total + additional_cost > monthly
        {
            return false;
        }
        true
    }

    /// Reset daily counters (called at the start of a new day).
    pub fn reset_daily(&mut self) {
        self.daily_total = 0.0;
    }

    /// Reset monthly counters (called at the start of a new month).
    pub fn reset_monthly(&mut self) {
        self.monthly_total = 0.0;
    }

    /// Get the current daily spend.
    pub fn daily_spend(&self) -> f64 {
        self.daily_total
    }

    /// Get the current monthly spend.
    pub fn monthly_spend(&self) -> f64 {
        self.monthly_total
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_usage_correctly() {
        let mut tracker = CostTracker::new();
        let provider = ProviderId::new("openai");

        tracker.record(&provider, 100, 50, 0.05);
        tracker.record(&provider, 200, 100, 0.10);

        let usage = tracker.get_usage(&provider).unwrap();
        assert_eq!(usage.total_input_tokens, 300);
        assert_eq!(usage.total_output_tokens, 150);
        assert!((usage.total_cost_usd - 0.15).abs() < 1e-10);
        assert_eq!(usage.request_count, 2);
    }

    #[test]
    fn total_cost_across_providers() {
        let mut tracker = CostTracker::new();
        let p1 = ProviderId::new("openai");
        let p2 = ProviderId::new("anthropic");

        tracker.record(&p1, 100, 50, 0.05);
        tracker.record(&p2, 200, 100, 0.10);

        assert!((tracker.total_cost() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn daily_budget_enforcement() {
        let mut tracker = CostTracker::with_budgets(Some(1.0), None);
        let provider = ProviderId::new("openai");

        tracker.record(&provider, 100, 50, 0.80);
        assert!(tracker.is_within_budget());

        tracker.record(&provider, 100, 50, 0.30);
        assert!(!tracker.is_within_budget());
    }

    #[test]
    fn monthly_budget_enforcement() {
        let mut tracker = CostTracker::with_budgets(None, Some(10.0));
        let provider = ProviderId::new("openai");

        tracker.record(&provider, 1000, 500, 9.50);
        assert!(tracker.is_within_budget());

        tracker.record(&provider, 100, 50, 0.60);
        assert!(!tracker.is_within_budget());
    }

    #[test]
    fn would_be_within_budget_check() {
        let mut tracker = CostTracker::with_budgets(Some(1.0), None);
        let provider = ProviderId::new("openai");

        tracker.record(&provider, 100, 50, 0.80);
        assert!(tracker.would_be_within_budget(0.10));
        assert!(!tracker.would_be_within_budget(0.30));
    }

    #[test]
    fn reset_daily_clears_daily_total() {
        let mut tracker = CostTracker::with_budgets(Some(1.0), Some(10.0));
        let provider = ProviderId::new("openai");

        tracker.record(&provider, 100, 50, 0.90);
        assert!(tracker.is_within_budget());

        tracker.record(&provider, 100, 50, 0.20);
        assert!(!tracker.is_within_budget()); // daily exceeded

        tracker.reset_daily();
        assert!(tracker.is_within_budget()); // daily reset, monthly still OK
    }

    #[test]
    fn no_usage_returns_none() {
        let tracker = CostTracker::new();
        assert!(tracker.get_usage(&ProviderId::new("nonexistent")).is_none());
    }

    #[test]
    fn no_budget_always_within() {
        let mut tracker = CostTracker::new();
        let provider = ProviderId::new("openai");
        tracker.record(&provider, 1_000_000, 500_000, 999.99);
        assert!(tracker.is_within_budget());
    }
}
