//! Echo sensitivity filtering for provider locality.
//!
//! Private memories must not leak to cloud providers. This module
//! filters echo results based on their sensitivity level and
//! whether the target provider is local or cloud-based.

use shrimpk_core::{EchoResult, MemoryEntry, SensitivityLevel};

/// Filter echo results based on provider locality.
///
/// Private memories should NOT be sent to cloud providers. Restricted
/// and Blocked memories are never sent to any provider.
///
/// # Arguments
/// - `results`: echo results to filter
/// - `entries`: the full memory entries (to check sensitivity levels)
/// - `is_local_provider`: whether the target model runs locally
///
/// # Returns
/// References to echo results that are safe to send to the given provider.
pub fn filter_for_provider<'a>(
    results: &'a [EchoResult],
    entries: &[MemoryEntry],
    is_local_provider: bool,
) -> Vec<&'a EchoResult> {
    results
        .iter()
        .filter(|result| {
            // Look up the memory entry to check its sensitivity.
            let sensitivity = entries
                .iter()
                .find(|e| e.id == result.memory_id)
                .map(|e| e.sensitivity)
                .unwrap_or(SensitivityLevel::Public); // default to Public if not found

            match sensitivity {
                SensitivityLevel::Public => true,
                SensitivityLevel::Private => is_local_provider,
                SensitivityLevel::Restricted | SensitivityLevel::Blocked => false,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use shrimpk_core::MemoryId;

    fn make_entry(id: MemoryId, sensitivity: SensitivityLevel) -> MemoryEntry {
        let mut entry = MemoryEntry::new("test content".into(), vec![], "test".into());
        entry.id = id;
        entry.sensitivity = sensitivity;
        entry
    }

    fn make_echo_result(id: MemoryId) -> EchoResult {
        EchoResult {
            memory_id: id,
            content: "test".into(),
            similarity: 0.9,
            final_score: 0.9,
            source: "test".into(),
            echoed_at: Utc::now(),
            modality: Default::default(),
            labels: Vec::new(),
            matched_child_content: None,
        }
    }

    #[test]
    fn public_memories_pass_for_cloud() {
        let id = MemoryId::new();
        let entries = vec![make_entry(id.clone(), SensitivityLevel::Public)];
        let results = vec![make_echo_result(id)];

        let filtered = filter_for_provider(&results, &entries, false);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn public_memories_pass_for_local() {
        let id = MemoryId::new();
        let entries = vec![make_entry(id.clone(), SensitivityLevel::Public)];
        let results = vec![make_echo_result(id)];

        let filtered = filter_for_provider(&results, &entries, true);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn private_memories_excluded_for_cloud() {
        let id = MemoryId::new();
        let entries = vec![make_entry(id.clone(), SensitivityLevel::Private)];
        let results = vec![make_echo_result(id)];

        let filtered = filter_for_provider(&results, &entries, false);
        assert!(filtered.is_empty(), "private memories must not go to cloud");
    }

    #[test]
    fn private_memories_included_for_local() {
        let id = MemoryId::new();
        let entries = vec![make_entry(id.clone(), SensitivityLevel::Private)];
        let results = vec![make_echo_result(id)];

        let filtered = filter_for_provider(&results, &entries, true);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn restricted_memories_excluded_everywhere() {
        let id = MemoryId::new();
        let entries = vec![make_entry(id.clone(), SensitivityLevel::Restricted)];
        let results = vec![make_echo_result(id)];

        assert!(filter_for_provider(&results, &entries, true).is_empty());
        assert!(filter_for_provider(&results, &entries, false).is_empty());
    }

    #[test]
    fn blocked_memories_excluded_everywhere() {
        let id = MemoryId::new();
        let entries = vec![make_entry(id.clone(), SensitivityLevel::Blocked)];
        let results = vec![make_echo_result(id)];

        assert!(filter_for_provider(&results, &entries, true).is_empty());
        assert!(filter_for_provider(&results, &entries, false).is_empty());
    }

    #[test]
    fn mixed_sensitivities_cloud_filter() {
        let pub_id = MemoryId::new();
        let priv_id = MemoryId::new();
        let restr_id = MemoryId::new();

        let entries = vec![
            make_entry(pub_id.clone(), SensitivityLevel::Public),
            make_entry(priv_id.clone(), SensitivityLevel::Private),
            make_entry(restr_id.clone(), SensitivityLevel::Restricted),
        ];
        let results = vec![
            make_echo_result(pub_id),
            make_echo_result(priv_id),
            make_echo_result(restr_id),
        ];

        // Cloud: only public passes.
        let filtered = filter_for_provider(&results, &entries, false);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn mixed_sensitivities_local_filter() {
        let pub_id = MemoryId::new();
        let priv_id = MemoryId::new();
        let restr_id = MemoryId::new();

        let entries = vec![
            make_entry(pub_id.clone(), SensitivityLevel::Public),
            make_entry(priv_id.clone(), SensitivityLevel::Private),
            make_entry(restr_id.clone(), SensitivityLevel::Restricted),
        ];
        let results = vec![
            make_echo_result(pub_id),
            make_echo_result(priv_id),
            make_echo_result(restr_id),
        ];

        // Local: public + private pass, restricted does not.
        let filtered = filter_for_provider(&results, &entries, true);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn unknown_memory_id_defaults_to_public() {
        let unknown_id = MemoryId::new();
        let entries: Vec<MemoryEntry> = vec![]; // no matching entry
        let results = vec![make_echo_result(unknown_id)];

        // Should pass for both local and cloud (defaults to Public).
        assert_eq!(filter_for_provider(&results, &entries, false).len(), 1);
        assert_eq!(filter_for_provider(&results, &entries, true).len(), 1);
    }

    #[test]
    fn empty_results_returns_empty() {
        let entries = vec![make_entry(MemoryId::new(), SensitivityLevel::Public)];
        let results: Vec<EchoResult> = vec![];

        assert!(filter_for_provider(&results, &entries, true).is_empty());
    }
}
