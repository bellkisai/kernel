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

/// Typed relationship between two memories in the Hebbian graph.
///
/// Enhances co-activation edges with semantic meaning. The default
/// `CoActivation` variant preserves existing behavior. Typed variants
/// are detected during sleep consolidation via regex-based entity
/// extraction from fact text.
///
/// The `Supersedes` variant is especially important for knowledge
/// updates: when a newer memory contradicts an older one, the newer
/// memory gets a stronger echo boost (freshness signal).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    /// Default — memories co-occurred in an echo result set.
    CoActivation,
    /// "X works at Y" / "X employed at Y" / "X joined Y".
    WorksAt(String),
    /// "X lives in Y" / "X moved to Y" / "X based in Y".
    LivesIn(String),
    /// "X prefers Y" / "X uses Y" / "X likes Y" / "X switched to Y".
    PrefersTool(String),
    /// "X part of Y" / "X belongs to Y" / "X member of Y".
    PartOf(String),
    /// Event A happened before Event B (temporal ordering).
    TemporalSequence,
    /// New information replaces old information (knowledge update).
    /// The edge points from the OLD memory to the NEW memory.
    /// During echo ranking, the newer memory gets an extra boost.
    Supersedes,
    /// Extensible catch-all for domain-specific relationships.
    Custom(String),
}

/// A single co-activation edge between two memories.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoActivation {
    /// Current (pre-decay) weight.
    pub weight: f64,
    /// Timestamp of last co-activation in seconds since epoch.
    pub last_activated: f64,
    /// How many times this pair has co-activated.
    pub activation_count: u32,
    /// Optional typed relationship between the two memories.
    /// `None` for legacy edges (treated as `CoActivation` during ranking).
    /// New edges created by co_activate() default to `None` (pure co-activation).
    /// Typed relationships are set by the consolidator during sleep consolidation.
    #[serde(default)]
    pub relationship: Option<RelationshipType>,
    /// When this edge became valid (epoch seconds). `None` = valid since creation.
    /// Set during consolidation when a new typed relationship is established.
    #[serde(default)]
    pub valid_from: Option<f64>,
    /// When this edge expired (epoch seconds). `None` = still valid.
    /// Set via `set_valid_until()` when a Supersedes edge retires an old relationship.
    #[serde(default)]
    pub valid_until: Option<f64>,
}

/// Sparse co-activation graph implementing Hebbian learning.
///
/// Edges are stored with the invariant `a < b` to avoid duplicates.
/// An adjacency index provides O(degree) neighbor lookups for the
/// echo boost pass.
///
/// **Design note (KS4):** The Hebbian graph uses a single `half_life` for ALL
/// edges (default: 7 days).  Category-aware adaptive decay (see
/// [`MemoryCategory`](shrimpk_core::MemoryCategory)) affects *memory*
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
            relationship: None,
            valid_from: None,
            valid_until: None,
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
            relationship: None,
            valid_from: None,
            valid_until: None,
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

    /// Record co-activation of two memories with a typed relationship.
    ///
    /// Same as [`co_activate`] but also sets the relationship type on the edge.
    /// If the edge already exists, the relationship is updated (overwritten).
    pub fn co_activate_with_relationship(
        &mut self,
        id_a: u32,
        id_b: u32,
        strength: f64,
        relationship: RelationshipType,
    ) {
        self.co_activate(id_a, id_b, strength);
        let key = Self::key(id_a, id_b);
        if let Some(entry) = self.edges.get_mut(&key) {
            entry.relationship = Some(relationship);
        }
    }

    /// Set or update the relationship type on an existing edge.
    ///
    /// Returns `true` if the edge exists and was updated, `false` if no edge
    /// exists between the two nodes.
    pub fn set_relationship(
        &mut self,
        id_a: u32,
        id_b: u32,
        relationship: RelationshipType,
    ) -> bool {
        let key = Self::key(id_a, id_b);
        if let Some(entry) = self.edges.get_mut(&key) {
            entry.relationship = Some(relationship);
            true
        } else {
            false
        }
    }

    /// Get the relationship type for an edge, if one exists.
    ///
    /// Returns `None` if the edge doesn't exist or has no typed relationship.
    pub fn get_relationship(&self, id_a: u32, id_b: u32) -> Option<&RelationshipType> {
        let key = Self::key(id_a, id_b);
        self.edges.get(&key).and_then(|co| co.relationship.as_ref())
    }

    /// Get the raw co-activation edge between two memories, if it exists.
    pub fn get_edge(&self, id_a: u32, id_b: u32) -> Option<&CoActivation> {
        let key = Self::key(id_a, id_b);
        self.edges.get(&key)
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

    /// Get associations with their relationship types, for use in typed echo ranking.
    ///
    /// Returns `(neighbor_id, decayed_weight, Option<&RelationshipType>)` triples
    /// where weight >= `min_weight`.
    pub fn get_associations_typed(
        &self,
        id: u32,
        min_weight: f64,
    ) -> Vec<(u32, f64, Option<&RelationshipType>)> {
        let now = Self::now();
        let neighbors = match self.adjacency.get(&id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        neighbors
            .iter()
            .filter_map(|&neighbor| {
                let key = Self::key(id, neighbor);
                self.edges.get(&key).and_then(|co| {
                    let elapsed = now - co.last_activated;
                    let decayed = co.weight * (-self.lambda * elapsed).exp();
                    if decayed >= min_weight {
                        Some((neighbor, decayed, co.relationship.as_ref()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Get associations that are temporally valid at a given point in time (KS63).
    ///
    /// Filters out edges where `valid_until < at_time` (expired) or
    /// `valid_from > at_time` (not yet valid). Edges with `None` validity
    /// are treated as always-valid (backward compat).
    pub fn get_valid_associations(
        &self,
        id: u32,
        at_time: f64,
        min_weight: f64,
    ) -> Vec<(u32, f64, Option<&RelationshipType>)> {
        let neighbors = match self.adjacency.get(&id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        neighbors
            .iter()
            .filter_map(|&neighbor| {
                let key = Self::key(id, neighbor);
                self.edges.get(&key).and_then(|co| {
                    // Temporal validity check
                    if let Some(until) = co.valid_until
                        && at_time > until
                    {
                        return None; // expired
                    }
                    if let Some(from) = co.valid_from
                        && at_time < from
                    {
                        return None; // not yet valid
                    }
                    // Decay weight
                    let elapsed = at_time - co.last_activated;
                    let decayed = co.weight * (-self.lambda * elapsed).exp();
                    if decayed >= min_weight {
                        Some((neighbor, decayed, co.relationship.as_ref()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Set the expiry timestamp on an existing edge (KS63).
    ///
    /// Called when a Supersedes edge retires an old relationship.
    /// Returns `true` if the edge existed and was updated.
    pub fn set_valid_until(&mut self, id_a: u32, id_b: u32, until: f64) -> bool {
        let key = Self::key(id_a, id_b);
        if let Some(entry) = self.edges.get_mut(&key) {
            entry.valid_until = Some(until);
            true
        } else {
            false
        }
    }

    /// Set the start-of-validity timestamp on an existing edge (KS63).
    ///
    /// Called when a new typed relationship is established during consolidation.
    /// Returns `true` if the edge existed and was updated.
    pub fn set_valid_from(&mut self, id_a: u32, id_b: u32, from: f64) -> bool {
        let key = Self::key(id_a, id_b);
        if let Some(entry) = self.edges.get_mut(&key) {
            entry.valid_from = Some(from);
            true
        } else {
            false
        }
    }

    /// BFS graph traversal from anchor nodes via Hebbian edges (KS62).
    ///
    /// Walks the co-activation graph outward from `anchors` up to `max_hops` hops.
    /// Path weight is multiplicative: if A→B has weight 0.8 and B→C has weight 0.5,
    /// the path A→B→C has weight 0.4. A visited set prevents cycles.
    ///
    /// Returns `(node_id, accumulated_weight)` sorted by weight descending,
    /// truncated to `max_results`.
    pub fn graph_traverse(
        &self,
        anchors: &[u32],
        max_hops: usize,
        max_results: usize,
    ) -> Vec<(u32, f64)> {
        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut results: Vec<(u32, f64)> = Vec::new();
        let mut frontier: Vec<(u32, f64)> = Vec::new();

        // Seed with anchors (weight 1.0 = direct entity match)
        for &anchor in anchors {
            if visited.insert(anchor) {
                frontier.push((anchor, 1.0));
                results.push((anchor, 1.0));
            }
        }

        for _hop in 0..max_hops {
            let mut next_frontier: Vec<(u32, f64)> = Vec::new();
            for &(node, path_weight) in &frontier {
                for (neighbor, edge_weight) in self.get_associations(node, self.prune_threshold) {
                    if visited.insert(neighbor) {
                        let new_weight = path_weight * edge_weight;
                        next_frontier.push((neighbor, new_weight));
                        results.push((neighbor, new_weight));
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);
        results
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

        let json = serde_json::to_string(&snapshot).map_err(std::io::Error::other)?;
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
        let snapshot: HebbianSnapshot =
            serde_json::from_str(&json).map_err(std::io::Error::other)?;

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
        assert!(
            w2 > w,
            "Weight should increase after second co-activation: {w2} > {w}"
        );
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
        assert!(
            !ids.contains(&3),
            "Should NOT include weakly connected node 3 (weight 0.005 < 0.1)"
        );
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
        assert!(
            graph.get_weight_at(1, 2, t_consolidate) > 0.0,
            "Strong edge should survive"
        );
        assert_eq!(
            graph.get_weight_at(3, 4, t_consolidate),
            0.0,
            "Pruned edge should be gone"
        );
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
        let adj_10 = graph
            .adjacency
            .get(&10)
            .expect("Node 10 should have adjacency");
        assert!(adj_10.contains(&20));
        assert!(adj_10.contains(&30));

        // Remove node 20
        graph.remove_node(20);

        // Node 10 should only have neighbor 30 now
        let adj_10 = graph
            .adjacency
            .get(&10)
            .expect("Node 10 should still have adjacency");
        assert!(
            !adj_10.contains(&20),
            "20 should be removed from 10's neighbors"
        );
        assert!(adj_10.contains(&30), "30 should remain in 10's neighbors");

        // Node 20 should be completely gone from adjacency
        assert!(
            !graph.adjacency.contains_key(&20),
            "Node 20 should have no adjacency entry"
        );
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
        let adj_1 = graph
            .adjacency
            .get(&1)
            .expect("Node 1 should have adjacency");
        assert!(adj_1.contains(&2), "Node 2 should be neighbor of 1");
        assert!(
            !adj_1.contains(&3),
            "Node 3 should be pruned from adjacency"
        );

        // Node 3 should have no adjacency entry (its only edge was pruned)
        assert!(
            graph.adjacency.get(&3).is_none_or(|v| v.is_empty()),
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

        let loaded = HebbianGraph::load(&path, HALF_LIFE, PRUNE_THRESHOLD).expect("Should load");

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
        let adj_1 = loaded
            .adjacency
            .get(&1)
            .expect("Node 1 should have adjacency");
        assert!(adj_1.contains(&2));
        let adj_3 = loaded
            .adjacency
            .get(&3)
            .expect("Node 3 should have adjacency");
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

    // ---- Typed relationship edge tests (KS18 Track 4) ----

    #[test]
    fn co_activate_default_has_no_relationship() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(1, 2, 0.5);

        let edge = graph.get_edge(1, 2).expect("Edge should exist");
        assert!(
            edge.relationship.is_none(),
            "Default co_activate should have no relationship"
        );
        assert!(
            graph.get_relationship(1, 2).is_none(),
            "get_relationship should return None for default edges"
        );
    }

    #[test]
    fn co_activate_with_relationship_sets_type() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate_with_relationship(
            1,
            2,
            0.5,
            RelationshipType::WorksAt("Google".to_string()),
        );

        let rel = graph
            .get_relationship(1, 2)
            .expect("Relationship should be set");
        assert_eq!(*rel, RelationshipType::WorksAt("Google".to_string()));

        // Weight should still be there
        let w = graph.get_weight(1, 2);
        assert!(w > 0.4, "Weight should be close to 0.5, got {w}");
    }

    #[test]
    fn set_relationship_updates_existing_edge() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(1, 2, 0.5);

        // Initially no relationship
        assert!(graph.get_relationship(1, 2).is_none());

        // Set relationship
        let updated = graph.set_relationship(1, 2, RelationshipType::Supersedes);
        assert!(updated, "Should return true for existing edge");

        let rel = graph
            .get_relationship(1, 2)
            .expect("Should have relationship");
        assert_eq!(*rel, RelationshipType::Supersedes);
    }

    #[test]
    fn set_relationship_returns_false_for_missing_edge() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let updated = graph.set_relationship(1, 2, RelationshipType::Supersedes);
        assert!(!updated, "Should return false when no edge exists");
    }

    #[test]
    fn relationship_survives_reinforcement() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);

        // Create edge with relationship
        graph.co_activate_with_relationship(
            1,
            2,
            0.5,
            RelationshipType::LivesIn("Tel Aviv".to_string()),
        );

        // Reinforce with plain co_activate (should NOT clear relationship)
        graph.co_activate(1, 2, 0.3);

        let rel = graph
            .get_relationship(1, 2)
            .expect("Relationship should survive reinforcement");
        assert_eq!(*rel, RelationshipType::LivesIn("Tel Aviv".to_string()));
    }

    #[test]
    fn relationship_serialization_roundtrip() {
        let types = vec![
            RelationshipType::CoActivation,
            RelationshipType::WorksAt("Anthropic".to_string()),
            RelationshipType::LivesIn("San Francisco".to_string()),
            RelationshipType::PrefersTool("Rust".to_string()),
            RelationshipType::PartOf("Bellkis".to_string()),
            RelationshipType::TemporalSequence,
            RelationshipType::Supersedes,
            RelationshipType::Custom("mentor-of".to_string()),
        ];

        for rel_type in &types {
            let json = serde_json::to_string(rel_type)
                .unwrap_or_else(|e| panic!("Failed to serialize {rel_type:?}: {e}"));
            let deserialized: RelationshipType = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("Failed to deserialize {rel_type:?}: {e}"));
            assert_eq!(*rel_type, deserialized, "Roundtrip failed for {rel_type:?}");
        }
    }

    #[test]
    fn edge_with_relationship_serialization_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("hebbian_rel.json");

        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate_with_relationship(
            1,
            2,
            0.8,
            RelationshipType::WorksAt("Google".to_string()),
        );
        graph.co_activate_with_relationship(3, 4, 0.6, RelationshipType::Supersedes);
        graph.co_activate(5, 6, 0.4); // plain edge — no relationship

        graph.save(&path).expect("Should save");

        let loaded = HebbianGraph::load(&path, HALF_LIFE, PRUNE_THRESHOLD).expect("Should load");

        // Check typed edges survived
        let rel_12 = loaded
            .get_relationship(1, 2)
            .expect("Relationship (1,2) should persist");
        assert_eq!(*rel_12, RelationshipType::WorksAt("Google".to_string()));

        let rel_34 = loaded
            .get_relationship(3, 4)
            .expect("Relationship (3,4) should persist");
        assert_eq!(*rel_34, RelationshipType::Supersedes);

        // Plain edge should have no relationship
        assert!(
            loaded.get_relationship(5, 6).is_none(),
            "Plain edge should have no relationship after load"
        );
    }

    #[test]
    fn legacy_edges_deserialize_without_relationship() {
        // Simulate loading a legacy JSON that has no `relationship` field.
        // The #[serde(default)] attribute should make it deserialize as None.
        let legacy_json = r#"{
            "edges":[[[1,2],{"weight":0.5,"last_activated":1000000.0,"activation_count":3}]],
            "activation_count":3
        }"#;
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("legacy_hebbian.json");
        std::fs::write(&path, legacy_json).expect("Write legacy JSON");

        let loaded =
            HebbianGraph::load(&path, HALF_LIFE, PRUNE_THRESHOLD).expect("Should load legacy");
        assert_eq!(loaded.len(), 1);

        // The edge should exist but have no relationship (legacy compat)
        let edge = loaded.get_edge(1, 2).expect("Edge should exist");
        assert!(
            edge.relationship.is_none(),
            "Legacy edges should have relationship = None"
        );
    }

    #[test]
    fn get_associations_typed_returns_relationships() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate_with_relationship(
            1,
            2,
            1.0,
            RelationshipType::WorksAt("ACME".to_string()),
        );
        graph.co_activate_with_relationship(1, 3, 0.8, RelationshipType::Supersedes);
        graph.co_activate(1, 4, 0.5); // no typed relationship

        let assocs = graph.get_associations_typed(1, 0.1);
        assert_eq!(assocs.len(), 3, "Should return all three associations");

        // Find the typed associations
        let works_at = assocs.iter().find(|(id, _, _)| *id == 2);
        let supersedes = assocs.iter().find(|(id, _, _)| *id == 3);
        let plain = assocs.iter().find(|(id, _, _)| *id == 4);

        assert!(works_at.is_some(), "Should find WorksAt association");
        assert_eq!(
            *works_at.unwrap().2.unwrap(),
            RelationshipType::WorksAt("ACME".to_string())
        );

        assert!(supersedes.is_some(), "Should find Supersedes association");
        assert_eq!(
            *supersedes.unwrap().2.unwrap(),
            RelationshipType::Supersedes
        );

        assert!(plain.is_some(), "Should find plain association");
        assert!(
            plain.unwrap().2.is_none(),
            "Plain association should have no relationship"
        );
    }

    // --- KS62: graph_traverse tests ---

    #[test]
    fn graph_traverse_single_hop() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        // A --0.5-- B --0.3-- C
        graph.co_activate(1, 2, 0.5);
        graph.co_activate(2, 3, 0.3);

        let results = graph.graph_traverse(&[1], 1, 10);
        let ids: Vec<u32> = results.iter().map(|&(id, _)| id).collect();
        assert!(ids.contains(&1), "Anchor should be in results");
        assert!(ids.contains(&2), "1-hop neighbor should be in results");
        assert!(
            !ids.contains(&3),
            "2-hop neighbor should NOT be in 1-hop results"
        );
    }

    #[test]
    fn graph_traverse_multi_hop() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(1, 2, 0.8);
        graph.co_activate(2, 3, 0.6);
        graph.co_activate(3, 4, 0.4);

        let results = graph.graph_traverse(&[1], 2, 10);
        let ids: Vec<u32> = results.iter().map(|&(id, _)| id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3), "2-hop neighbor should be reached");
        assert!(
            !ids.contains(&4),
            "3-hop should NOT be reached with max_hops=2"
        );

        // Verify multiplicative weights
        let w3 = results.iter().find(|&&(id, _)| id == 3).unwrap().1;
        assert!(
            w3 < 0.6,
            "2-hop weight should be < direct edge weight, got {w3}"
        );
    }

    #[test]
    fn graph_traverse_handles_cycles() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        // Triangle: 1--2--3--1
        graph.co_activate(1, 2, 0.5);
        graph.co_activate(2, 3, 0.5);
        graph.co_activate(3, 1, 0.5);

        let results = graph.graph_traverse(&[1], 3, 10);
        // Should visit each node exactly once despite cycle
        assert_eq!(results.len(), 3, "Should visit 3 unique nodes");
        let ids: Vec<u32> = results.iter().map(|&(id, _)| id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn graph_traverse_empty_graph() {
        let graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let results = graph.graph_traverse(&[1, 2], 2, 10);
        // Anchors still appear even with no edges
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn graph_traverse_respects_max_results() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        // Star graph: center 0, spokes 1..10
        for i in 1..=10 {
            graph.co_activate(0, i, 0.5);
        }
        let results = graph.graph_traverse(&[0], 1, 5);
        assert_eq!(results.len(), 5, "Should truncate to max_results");
    }

    // --- KS63: temporal validity tests ---

    #[test]
    fn temporal_fields_serde_backward_compat() {
        // Legacy edge without valid_from/valid_until should deserialize fine
        let json = r#"{"weight":0.5,"last_activated":1000.0,"activation_count":1}"#;
        let co: CoActivation = serde_json::from_str(json).unwrap();
        assert!(co.valid_from.is_none());
        assert!(co.valid_until.is_none());
        assert_eq!(co.weight, 0.5);
    }

    #[test]
    fn temporal_fields_serde_roundtrip() {
        let co = CoActivation {
            weight: 0.8,
            last_activated: 2000.0,
            activation_count: 3,
            relationship: Some(RelationshipType::WorksAt("Acme".into())),
            valid_from: Some(1000.0),
            valid_until: Some(3000.0),
        };
        let json = serde_json::to_string(&co).unwrap();
        let co2: CoActivation = serde_json::from_str(&json).unwrap();
        assert_eq!(co2.valid_from, Some(1000.0));
        assert_eq!(co2.valid_until, Some(3000.0));
    }

    #[test]
    fn set_valid_until_expires_edge() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(1, 2, 0.5);
        assert!(graph.set_valid_until(1, 2, 5000.0));
        let edge = graph.get_edge(1, 2).unwrap();
        assert_eq!(edge.valid_until, Some(5000.0));
    }

    #[test]
    fn set_valid_from_on_edge() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        graph.co_activate(3, 4, 0.3);
        assert!(graph.set_valid_from(3, 4, 1000.0));
        let edge = graph.get_edge(3, 4).unwrap();
        assert_eq!(edge.valid_from, Some(1000.0));
    }

    #[test]
    fn set_valid_returns_false_for_missing_edge() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        assert!(!graph.set_valid_until(99, 100, 5000.0));
        assert!(!graph.set_valid_from(99, 100, 1000.0));
    }

    #[test]
    fn get_valid_associations_filters_expired() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let t0 = 1_000_000.0;

        // Create two edges at t0
        graph.co_activate_at(1, 2, 0.5, t0);
        graph.co_activate_at(1, 3, 0.5, t0);

        // Expire edge 1-2 at t0 + 100
        graph.set_valid_until(1, 2, t0 + 100.0);

        // At t0 + 50: both edges valid
        let at_50 = graph.get_valid_associations(1, t0 + 50.0, 0.0);
        assert_eq!(at_50.len(), 2, "Both edges should be valid at t0+50");

        // At t0 + 200: edge 1-2 expired
        let at_200 = graph.get_valid_associations(1, t0 + 200.0, 0.0);
        assert_eq!(at_200.len(), 1, "Only edge 1-3 should be valid at t0+200");
        assert_eq!(at_200[0].0, 3);
    }

    #[test]
    fn get_valid_associations_filters_not_yet_valid() {
        let mut graph = HebbianGraph::new(HALF_LIFE, PRUNE_THRESHOLD);
        let t0 = 1_000_000.0;

        graph.co_activate_at(1, 2, 0.5, t0);
        graph.set_valid_from(1, 2, t0 + 500.0);

        // Before valid_from: should NOT appear
        let before = graph.get_valid_associations(1, t0 + 100.0, 0.0);
        assert!(
            before.is_empty(),
            "Edge should not be valid before valid_from"
        );

        // After valid_from: should appear
        let after = graph.get_valid_associations(1, t0 + 600.0, 0.0);
        assert_eq!(after.len(), 1, "Edge should be valid after valid_from");
    }
}
