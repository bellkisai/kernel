//! Hebbian learning for Echo Memory — "neurons that fire together wire together."
//!
//! Tracks co-activation of memories: when two memories are returned together in
//! an echo query, their Hebbian edge is strengthened. Edges decay exponentially
//! over time (default half-life: 7 days) so stale associations fade naturally.
//!
//! The graph is sparse (HashMap-backed) with an adjacency index for O(degree)
//! neighbor lookups. During echo queries, Hebbian associations boost the final
//! score of results that have been co-activated in the past.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single co-activation edge between two memories.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoActivation {
    /// Current (pre-decay) weight.
    pub weight: f64,
    /// Timestamp of last co-activation in seconds since epoch.
    pub last_activated: f64,
    /// How many times this pair has co-activated.
    pub activation_count: u32,
}

/// Sparse co-activation graph implementing Hebbian learning.
///
/// Edges are stored with the invariant `a < b` to avoid duplicates.
/// An adjacency index provides O(degree) neighbor lookups for the
/// echo boost pass.
///
/// **Design note (KS4):** The Hebbian graph uses a single `half_life` for ALL
/// edges (default: 7 days).  Category-aware adaptive decay (see
/// [`MemoryCategory`](bellkis_core::MemoryCategory)) affects *memory*
/// persistence, not *association* persistence.  Memories decay at
/// per-category rates; the links between them decay uniformly.  This
/// keeps the association graph simple and avoids the complexity of
/// per-edge decay curves.
pub struct HebbianGraph {
    /// (memory_index_a, memory_index_b) -> co-activation record.
    /// Invariant: a < b for every key.
    edges: HashMap<(u32, u32), CoActivation>,
    /// Per-node neighbor list for O(degree) association lookups.
    adjacency: HashMap<u32, Vec<u32>>,
    /// Decay half-life in seconds (default: 604,800 = 7 days).
    half_life: f64,
    /// Decay constant: ln(2) / half_life.
    lambda: f64,
    /// Minimum decayed weight before an edge is pruned.
    prune_threshold: f64,
    /// Total number of co-activation events recorded (for stats).
    activation_count: u64,
}

/// Serializable representation for persistence.
#[derive(Serialize, Deserialize)]
struct HebbianSnapshot {
    edges: Vec<((u32, u32), CoActivation)>,
    activation_count: u64,
}

impl HebbianGraph {
    /// Create a new empty Hebbian graph.
    ///
    /// # Arguments
    /// * `half_life_seconds` - Exponential decay half-life (e.g., 604,800 for 7 days)
    /// * `prune_threshold` - Edges with decayed weight below this are pruned on consolidation
    pub fn new(half_life_seconds: f64, prune_threshold: f64) -> Self {
        let lambda = (2.0_f64).ln() / half_life_seconds;
        Self {
            edges: HashMap::new(),
            adjacency: HashMap::new(),
            half_life: half_life_seconds,
            lambda,
            prune_threshold,
            activation_count: 0,
        }
    }

    /// Current time as seconds since UNIX epoch.
    fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Normalize a pair so that a < b (avoids duplicate edges).
    fn key(a: u32, b: u32) -> (u32, u32) {
        if a < b { (a, b) } else { (b, a) }
    }

    /// Ensure `other` appears in the adjacency list of `node`.
    fn ensure_adjacency(&mut self, node: u32, other: u32) {
        let neighbors = self.adjacency.entry(node).or_default();
        if !neighbors.contains(&other) {
            neighbors.push(other);
        }
    }

    /// Record co-activation of two memories.
    ///
    /// Hebbian rule: "neurons that fire together wire together."
    /// Applies exponential decay since last activation, then adds `strength`.
    pub fn co_activate(&mut self, id_a: u32, id_b: u32, strength: f64) {
        if id_a == id_b {
            return;
        }
        let key = Self::key(id_a, id_b);
        let now = Self::now();

        let entry = self.edges.entry(key).or_insert(CoActivation {
            weight: 0.0,
            last_activated: now,
            activation_count: 0,
        });

        // Apply exponential decay since last activation
        let elapsed = now - entry.last_activated;
        if elapsed > 0.0 {
            entry.weight *= (-self.lambda * elapsed).exp();
        }

        // Hebbian strengthening
        entry.weight += strength;
        entry.last_activated = now;
        entry.activation_count += 1;
        self.activation_count += 1;

        // Update adjacency index for both nodes
        self.ensure_adjacency(id_a, id_b);
        self.ensure_adjacency(id_b, id_a);
    }

    /// Co-activate using a manually-specified timestamp (for testing).
    #[cfg(test)]
    fn co_activate_at(&mut self, id_a: u32, id_b: u32, strength: f64, timestamp: f64) {
        if id_a == id_b {
            return;
        }
        let key = Self::key(id_a, id_b);

        let entry = self.edges.entry(key).or_insert(CoActivation {
            weight: 0.0,
            last_activated: timestamp,
            activation_count: 0,
        });

        let elapsed = timestamp - entry.last_activated;
        if elapsed > 0.0 {
            entry.weight *= (-self.lambda * elapsed).exp();
        }

        entry.weight += strength;
        entry.last_activated = timestamp;
        entry.activation_count += 1;
        self.activation_count += 1;

        self.ensure_adjacency(id_a, id_b);
        self.ensure_adjacency(id_b, id_a);
    }

    /// Get the current (decayed) weight between two memories.
    ///
    /// Returns 0.0 if no edge exists.
    pub fn get_weight(&self, id_a: u32, id_b: u32) -> f64 {
        if id_a == id_b {
            return 0.0;
        }
        let key = Self::key(id_a, id_b);
        match self.edges.get(&key) {
            Some(co) => {
                let elapsed = Self::now() - co.last_activated;
                co.weight * (-self.lambda * elapsed).exp()
            }
            None => 0.0,
        }
    }

    /// Get the weight using a specific reference timestamp (for testing).
    #[cfg(test)]
    fn get_weight_at(&self, id_a: u32, id_b: u32, at: f64) -> f64 {
        if id_a == id_b {
            return 0.0;
        }
        let key = Self::key(id_a, id_b);
        match self.edges.get(&key) {
            Some(co) => {
                let elapsed = at - co.last_activated;
                co.weight * (-self.lambda * elapsed).exp()
            }
            None => 0.0,
        }
    }

    /// Get all strong associations for a memory, using the adjacency index
    /// for O(degree) lookup rather than scanning the full edge map.
    ///
    /// Returns `(neighbor_id, decayed_weight)` pairs where weight >= `min_weight`.
    pub fn get_associations(&self, id: u32, min_weight: f64) -> Vec<(u32, f64)> {
        let now = Self::now();
        self.get_associations_at(id, min_weight, now)
    }

    /// Get associations using a specific reference timestamp.
    fn get_associations_at(&self, id: u32, min_weight: f64, at: f64) -> Vec<(u32, f64)> {
        let neighbors = match self.adjacency.get(&id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        neighbors
            .iter()
            .filter_map(|&neighbor| {
                let key = Self::key(id, neighbor);
                self.edges.get(&key).and_then(|co| {
                    let elapsed = at - co.last_activated;
                    let decayed = co.weight * (-self.lambda * elapsed).exp();
                    if decayed >= min_weight {
                        Some((neighbor, decayed))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Consolidation: prune edges whose decayed weight has fallen below the threshold.
    ///
    /// Also cleans up the adjacency index for pruned edges.
    /// Returns the number of pruned edges.
    pub fn consolidate(&mut self) -> usize {
        self.consolidate_at(Self::now())
    }

    /// Consolidation at a specific reference timestamp.
    fn consolidate_at(&mut self, at: f64) -> usize {
        let lambda = self.lambda;
        let threshold = self.prune_threshold;

        // Collect keys to remove
        let to_remove: Vec<(u32, u32)> = self
            .edges
            .iter()
            .filter(|(_, co)| {
                let elapsed = at - co.last_activated;
                let decayed = co.weight * (-lambda * elapsed).exp();
                decayed < threshold
            })
            .map(|(&key, _)| key)
            .collect();

        let pruned = to_remove.len();

        for key in &to_remove {
            self.edges.remove(key);
        }

        // Rebuild adjacency index from remaining edges
        self.adjacency.clear();
        for &(a, b) in self.edges.keys() {
            self.adjacency.entry(a).or_default().push(b);
            self.adjacency.entry(b).or_default().push(a);
        }

        pruned
    }

    /// Remove all edges involving a given node.
    ///
    /// Called when a memory is forgotten/deleted.
    pub fn remove_node(&mut self, id: u32) {
        // Remove all edges where this node is a participant
        self.edges.retain(|&(a, b), _| a != id && b != id);

        // Remove from adjacency
        if let Some(neighbors) = self.adjacency.remove(&id) {
            for neighbor in neighbors {
                if let Some(n_list) = self.adjacency.get_mut(&neighbor) {
                    n_list.retain(|&n| n != id);
                    if n_list.is_empty() {
                        self.adjacency.remove(&neighbor);
                    }
                }
            }
        }
    }

    /// Number of edges in the graph.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Total co-activation events recorded (for stats).
    pub fn total_activations(&self) -> u64 {
        self.activation_count
    }

    /// The configured decay half-life in seconds.
    pub fn half_life(&self) -> f64 {
        self.half_life
    }

    /// Save the graph to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let snapshot = HebbianSnapshot {
            edges: self.edges.iter().map(|(&k, v)| (k, v.clone())).collect(),
            activation_count: self.activation_count,
        };

        let json = serde_json::to_string(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load a graph from a JSON file.
    ///
    /// The `half_life` and `prune_threshold` are supplied from config, not stored
    /// in the file, so they can be tuned between sessions.
    pub fn load(path: &Path, half_life: f64, prune_threshold: f64) -> std::io::Result<Self> {
        if !path.exists() {
            return Ok(Self::new(half_life, prune_threshold));
        }

        let json = std::fs::read_to_string(path)?;
        let snapshot: HebbianSnapshot = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let lambda = (2.0_f64).ln() / half_life;
        let mut adjacency: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut edges = HashMap::new();

        for ((a, b), co) in snapshot.edges {
            edges.insert((a, b), co);
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }

        Ok(Self {
            edges,
            adjacency,
            half_life,
            lambda,
            prune_threshold,
            activation_count: snapshot.activation_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_LIFE: f64 = 604_800.0; // 7 days
    const PRUNE_THRESHOLD: f64 = 0.01;

    #[test]
    fn co_activate_increases_weight() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        assert_eq!(graph.len(), 0);

        graph.co_activate(1, 2, 0.5);
        let w = graph.get_weight(1, 2);
        assert!(w > 0.4, "Weight should be close to 0.5, got {w}");

        // Second co-activation should increase weight
        graph.co_activate(1, 2, 0.3);
        let w2 = graph.get_weight(1, 2);
        assert!(w2 > w, "Weight should increase after second co-activation: {w2} > {w}");
        assert_eq!(graph.len(), 1, "Should still be one edge");
    }

    #[test]
    fn weight_decays_over_time() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let t0 = 1_000_000.0;

        // Activate at t0
        graph.co_activate_at(1, 2, 1.0, t0);

        // Immediately after: weight ~1.0
        let w_immediate = graph.get_weight_at(1, 2, t0);
        assert!(
            (w_immediate - 1.0).abs() < 0.001,
            "Weight at t0 should be ~1.0, got {w_immediate}"
        );

        // After one half-life: weight ~0.5
        let w_half = graph.get_weight_at(1, 2, t0 + HALF_LIFE);
        assert!(
            (w_half - 0.5).abs() < 0.01,
            "Weight after one half-life should be ~0.5, got {w_half}"
        );

        // After two half-lives: weight ~0.25
        let w_quarter = graph.get_weight_at(1, 2, t0 + 2.0 * HALF_LIFE);
        assert!(
            (w_quarter - 0.25).abs() < 0.01,
            "Weight after two half-lives should be ~0.25, got {w_quarter}"
        );
    }

    #[test]
    fn get_associations_returns_only_strong_connections() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let t0 = 1_000_000.0;

        // Node 1 strongly associated with node 2
        graph.co_activate_at(1, 2, 1.0, t0);
        // Node 1 weakly associated with node 3
        graph.co_activate_at(1, 3, 0.005, t0);
        // Node 1 associated with node 4
        graph.co_activate_at(1, 4, 0.5, t0);

        let associations = graph.get_associations_at(1, 0.1, t0);
        let ids: Vec<u32> = associations.iter().map(|(id, _)| *id).collect();

        assert!(ids.contains(&2), "Should include strongly connected node 2");
        assert!(ids.contains(&4), "Should include connected node 4");
        assert!(!ids.contains(&3), "Should NOT include weakly connected node 3 (weight 0.005 < 0.1)");
    }

    #[test]
    fn consolidate_prunes_weak_edges() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let t0 = 1_000_000.0;
        let t_consolidate = t0 + 0.1; // consolidate just after activation

        // Strong edge
        graph.co_activate_at(1, 2, 1.0, t0);
        // Weak edge: set its last_activated to 2 half-lives ago
        // so decayed weight ~ 0.02 * 0.25 = 0.005 < 0.01
        graph.co_activate_at(3, 4, 0.02, t0 - 2.0 * HALF_LIFE);

        assert_eq!(graph.len(), 2);

        let pruned = graph.consolidate_at(t_consolidate);
        assert_eq!(pruned, 1, "Should prune exactly the weak edge");
        assert_eq!(graph.len(), 1, "One edge should remain");
        assert!(graph.get_weight_at(1, 2, t_consolidate) > 0.0, "Strong edge should survive");
        assert_eq!(graph.get_weight_at(3, 4, t_consolidate), 0.0, "Pruned edge should be gone");
    }

    #[test]
    fn remove_node_clears_all_edges() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);

        graph.co_activate(1, 2, 1.0);
        graph.co_activate(1, 3, 1.0);
        graph.co_activate(2, 3, 1.0);
        assert_eq!(graph.len(), 3);

        graph.remove_node(1);
        assert_eq!(graph.len(), 1, "Only the (2,3) edge should remain");
        assert_eq!(graph.get_weight(1, 2), 0.0, "Edge (1,2) should be gone");
        assert_eq!(graph.get_weight(1, 3), 0.0, "Edge (1,3) should be gone");
        assert!(graph.get_weight(2, 3) > 0.0, "Edge (2,3) should remain");
    }

    #[test]
    fn adjacency_stays_consistent() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);

        graph.co_activate(10, 20, 1.0);
        graph.co_activate(10, 30, 1.0);
        graph.co_activate(20, 30, 1.0);

        // Node 10 should have neighbors 20 and 30
        let adj_10 = graph.adjacency.get(&10).expect("Node 10 should have adjacency");
        assert!(adj_10.contains(&20));
        assert!(adj_10.contains(&30));

        // Remove node 20
        graph.remove_node(20);

        // Node 10 should only have neighbor 30 now
        let adj_10 = graph.adjacency.get(&10).expect("Node 10 should still have adjacency");
        assert!(!adj_10.contains(&20), "20 should be removed from 10's neighbors");
        assert!(adj_10.contains(&30), "30 should remain in 10's neighbors");

        // Node 20 should be completely gone from adjacency
        assert!(graph.adjacency.get(&20).is_none(), "Node 20 should have no adjacency entry");
    }

    #[test]
    fn adjacency_consistent_after_consolidate() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let t0 = 1_000_000.0;
        let t_consolidate = t0 + 0.1;

        // Strong edge at t0
        graph.co_activate_at(1, 2, 1.0, t0);
        // Weak edge set far in the past so it decays below threshold
        graph.co_activate_at(1, 3, 0.02, t0 - 3.0 * HALF_LIFE);

        graph.consolidate_at(t_consolidate);

        // After consolidation, node 1 should only have neighbor 2
        let adj_1 = graph.adjacency.get(&1).expect("Node 1 should have adjacency");
        assert!(adj_1.contains(&2), "Node 2 should be neighbor of 1");
        assert!(!adj_1.contains(&3), "Node 3 should be pruned from adjacency");

        // Node 3 should have no adjacency entry (its only edge was pruned)
        assert!(
            graph.adjacency.get(&3).is_none() || graph.adjacency.get(&3).unwrap().is_empty(),
            "Node 3 should have no neighbors after its edge was pruned"
        );
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("hebbian.json");

        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(1, 2, 0.8);
        graph.co_activate(3, 4, 0.6);
        let original_len = graph.len();
        let original_activations = graph.total_activations();

        graph.save(&path).expect("Should save");

        let loaded = HebbianGraph::load(&path, HALF_LIFE, PRUNE_THRESHOLD)
            .expect("Should load");

        assert_eq!(loaded.len(), original_len, "Edge count should match");
        assert_eq!(
            loaded.total_activations(),
            original_activations,
            "Activation count should match"
        );

        // Verify weights are preserved (within decay tolerance since time passes)
        let w12 = loaded.get_weight(1, 2);
        assert!(w12 > 0.5, "Weight (1,2) should be close to 0.8, got {w12}");
        let w34 = loaded.get_weight(3, 4);
        assert!(w34 > 0.3, "Weight (3,4) should be close to 0.6, got {w34}");

        // Verify adjacency was rebuilt
        let adj_1 = loaded.adjacency.get(&1).expect("Node 1 should have adjacency");
        assert!(adj_1.contains(&2));
        let adj_3 = loaded.adjacency.get(&3).expect("Node 3 should have adjacency");
        assert!(adj_3.contains(&4));
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let path = std::path::Path::new("/tmp/nonexistent_hebbian_99999.json");
        let graph = HebbianGraph::load(path, HALF_LIFE, PRUNE_THRESHOLD)
            .expect("Should return empty graph");
        assert!(graph.is_empty());
        assert_eq!(graph.total_activations(), 0);
    }

    #[test]
    fn self_activation_is_ignored() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(5, 5, 1.0);
        assert_eq!(graph.len(), 0, "Self-edges should be ignored");
        assert_eq!(graph.get_weight(5, 5), 0.0);
    }

    #[test]
    fn key_normalization() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(5, 3, 1.0);
        // Should be the same edge regardless of order
        let w_53 = graph.get_weight(5, 3);
        let w_35 = graph.get_weight(3, 5);
        assert!(
            (w_53 - w_35).abs() < 0.001,
            "get_weight should be symmetric: {w_53} vs {w_35}"
        );
        assert_eq!(graph.len(), 1, "Should be exactly one edge");
    }

    #[test]
    fn reinforcement_accumulates() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let t0 = 1_000_000.0;

        graph.co_activate_at(1, 2, 0.5, t0);
        let w1 = graph.get_weight_at(1, 2, t0);

        // Activate again 1 second later (negligible decay)
        graph.co_activate_at(1, 2, 0.3, t0 + 1.0);
        let w2 = graph.get_weight_at(1, 2, t0 + 1.0);

        assert!(
            w2 > w1,
            "Weight should increase with reinforcement: {w2} > {w1}"
        );
        // Should be close to 0.5 + 0.3 = 0.8 (negligible decay in 1 second)
        assert!(
            (w2 - 0.8).abs() < 0.01,
            "Weight should be ~0.8 after two activations, got {w2}"
        );
    }
}
