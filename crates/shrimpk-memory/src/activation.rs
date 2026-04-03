//! ACT-R activation model + FSRS power-law decay.
//!
//! Replaces the exponential decay `exp(-age * ln2 / half_life)` with
//! power-law `(1 + FSRS_FACTOR * age / stability)^(FSRS_DECAY)`.
//!
//! ACT-R Optimized Learning approximation uses existing echo_count + created_at
//! instead of a full retrieval timestamp history (~85% accuracy, zero schema change).
//!
//! References:
//! - Anderson & Lebiere (1998), The Atomic Components of Thought
//! - FSRS-6 (2024), open-source spaced repetition

use chrono::{DateTime, Utc};

/// FSRS constants (from FSRS-6).
const FSRS_FACTOR: f64 = 19.0 / 81.0; // 0.2346...
const FSRS_DECAY: f64 = -0.5;

/// Power-law decay (FSRS forgetting curve).
///
/// `R(t, S) = (1 + FACTOR * t / S) ^ DECAY`
///
/// Returns retention probability in [0.0, 1.0].
/// `age_secs`: seconds since memory creation.
/// `stability_days`: days at which retention = 0.9 (category-dependent).
pub fn power_law_decay(age_secs: f64, stability_days: f64) -> f64 {
    if stability_days <= 0.0 {
        return 0.0;
    }
    let age_days = age_secs / 86400.0;
    (1.0 + FSRS_FACTOR * age_days / stability_days)
        .powf(FSRS_DECAY)
        .clamp(0.0, 1.0)
}

/// ACT-R Optimized Learning approximation for base-level activation.
///
/// `B ≈ ln((n + 1) / ((1 - d) * L^(1-d)))`
///
/// Uses existing echo_count (n) and lifetime (L = age in seconds).
/// No timestamp history needed. ~85% accuracy vs full BLA.
///
/// Returns activation value (typically in range [-6, 4]).
/// Clamped to [-6, 4] to prevent extreme values from dominating scoring.
///
/// `echo_count`: number of times retrieved.
/// `created_at`: when the memory was stored.
/// `last_echoed`: last retrieval time (for recency component).
/// `d`: decay parameter (category-dependent, 0.3-0.7).
pub fn actr_ol_activation(
    echo_count: u32,
    created_at: DateTime<Utc>,
    last_echoed: Option<DateTime<Utc>>,
    d: f64,
) -> f64 {
    let now = Utc::now();
    let lifetime_secs = (now - created_at).num_seconds().max(1) as f64;

    // Base-level activation (Optimized Learning approximation)
    let n = echo_count as f64;
    let one_minus_d = 1.0 - d;
    let bla = if one_minus_d > 0.001 {
        ((n + 1.0) / (one_minus_d * lifetime_secs.powf(one_minus_d))).ln()
    } else {
        // Edge case: d ≈ 1.0 — just use log of count
        (n + 1.0).ln() - lifetime_secs.ln()
    };

    // Recency component: boost for recently-echoed memories
    let recency = if let Some(last) = last_echoed {
        let since_echo = (now - last).num_seconds().max(1) as f64;
        -(d * since_echo.ln())
    } else {
        0.0
    };

    (bla + recency * 0.3).clamp(-6.0, 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn power_law_decay_fresh_memory() {
        let decay = power_law_decay(0.0, 7.0);
        assert!((decay - 1.0).abs() < 0.001, "fresh memory: decay={decay}");
    }

    #[test]
    fn power_law_decay_at_half_life() {
        // FSRS power-law decays slower than exponential at half-life.
        // At t=7 days, S=7/1.73≈4.05: R = (1 + 0.2346 * 7/4.05)^(-0.5) ≈ 0.84
        // This is above 0.5 — the whole point of power-law is gentler long-tail.
        let half_life_secs = 7.0 * 86400.0;
        let stability = 7.0 / 1.73; // ~4.05 days
        let decay = power_law_decay(half_life_secs, stability);
        assert!(
            decay < 0.95 && decay > 0.7,
            "at half-life: decay={decay}, expected power-law retention ~0.84"
        );
    }

    #[test]
    fn power_law_decay_old_memory_retains() {
        // Identity memory at 1 year should still retain something
        let age = 365.0 * 86400.0;
        let stability = 365.0 / 1.73; // ~211 days
        let decay = power_law_decay(age, stability);
        assert!(
            decay > 0.1,
            "1-year Identity should retain >10%, got {decay}"
        );
    }

    #[test]
    fn power_law_beats_exponential_long_term() {
        let age = 60.0 * 86400.0; // 60 days
        let stability = 14.0 / 1.73; // ActiveProject
        let pl = power_law_decay(age, stability);
        let exp = (-age * 0.693 / (14.0 * 86400.0)).exp();
        assert!(
            pl > exp,
            "power-law {pl} should beat exponential {exp} at 60 days"
        );
    }

    #[test]
    fn power_law_zero_stability() {
        assert_eq!(power_law_decay(100.0, 0.0), 0.0);
    }

    #[test]
    fn power_law_decay_monotonically_decreasing() {
        let stability = 10.0;
        let d1 = power_law_decay(1.0 * 86400.0, stability);
        let d2 = power_law_decay(10.0 * 86400.0, stability);
        let d3 = power_law_decay(100.0 * 86400.0, stability);
        assert!(d1 > d2, "1 day ({d1}) > 10 days ({d2})");
        assert!(d2 > d3, "10 days ({d2}) > 100 days ({d3})");
    }

    #[test]
    fn actr_fresh_memory_low_activation() {
        let created = Utc::now() - Duration::seconds(60);
        let act = actr_ol_activation(0, created, None, 0.5);
        // Fresh, never retrieved — should be low
        assert!(act < 1.0, "fresh memory activation={act}");
    }

    #[test]
    fn actr_frequently_accessed_higher() {
        let created = Utc::now() - Duration::days(7);
        let act0 = actr_ol_activation(0, created, None, 0.5);
        let act10 = actr_ol_activation(10, created, None, 0.5);
        assert!(act10 > act0, "10 echoes ({act10}) > 0 echoes ({act0})");
    }

    #[test]
    fn actr_recently_echoed_higher() {
        // Use a shorter lifetime (1 day) with enough echoes to stay within clamp range.
        let created = Utc::now() - Duration::days(1);
        let recent = Some(Utc::now() - Duration::minutes(5));
        let old = Some(Utc::now() - Duration::hours(20));
        let act_recent = actr_ol_activation(20, created, recent, 0.5);
        let act_old = actr_ol_activation(20, created, old, 0.5);
        assert!(
            act_recent > act_old,
            "recently echoed ({act_recent}) > old echo ({act_old})"
        );
    }

    #[test]
    fn actr_clamped() {
        let created = Utc::now() - Duration::seconds(1);
        let act = actr_ol_activation(10000, created, Some(Utc::now()), 0.5);
        assert!(act <= 4.0, "activation must be clamped to 4.0, got {act}");
        assert!(act >= -6.0);
    }

    #[test]
    fn actr_category_decay_affects_result() {
        // In the OL approximation `B = ln((n+1) / ((1-d) * L^(1-d)))`, the d parameter
        // controls how the lifetime denominator grows. Different d values produce
        // measurably different activation levels, which is the key property we need.
        let created = Utc::now() - Duration::hours(1);
        let act_low_d = actr_ol_activation(10, created, None, 0.3);
        let act_high_d = actr_ol_activation(10, created, None, 0.7);
        assert!(
            (act_low_d - act_high_d).abs() > 0.5,
            "Different decay rates should produce meaningfully different activations: \
             d=0.3 ({act_low_d}) vs d=0.7 ({act_high_d})"
        );
    }

    #[test]
    fn actr_edge_case_d_near_one() {
        // d ≈ 1.0 triggers the alternate formula path
        let created = Utc::now() - Duration::days(7);
        let act = actr_ol_activation(5, created, None, 0.999);
        assert!(
            (-6.0..=4.0).contains(&act),
            "d~1.0 should still be clamped, got {act}"
        );
    }
}
