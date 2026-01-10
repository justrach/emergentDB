//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! Based on the paper: "Efficient and robust approximate nearest neighbor search using
//! Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2016).
//!
//! Provides O(log N) search complexity with high recall.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use parking_lot::RwLock;

use crate::distance::{cosine_distance_simd, euclidean_distance_simd, negative_dot_product_simd};
use crate::{DistanceMetric, Embedding, NodeId, Result, SearchResult, VectorError};

use super::{IndexConfig, VectorIndex};

/// A node in the HNSW graph.
#[derive(Clone)]
struct HnswNode {
    id: NodeId,
    embedding: Embedding,
    /// Neighbors at each level (level 0 = bottom, most neighbors).
    neighbors: Vec<Vec<NodeId>>,
}

/// Distance-ID pair for priority queue.
#[derive(Clone, Copy)]
struct DistanceId {
    distance: f32,
    id: NodeId,
}

impl PartialEq for DistanceId {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for DistanceId {}

impl PartialOrd for DistanceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceId {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap variant for furthest neighbor tracking.
#[derive(Clone, Copy)]
struct MaxDistanceId {
    distance: f32,
    id: NodeId,
}

impl PartialEq for MaxDistanceId {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for MaxDistanceId {}

impl PartialOrd for MaxDistanceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxDistanceId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// HNSW index for approximate nearest neighbor search.
pub struct HnswIndex {
    config: IndexConfig,
    /// All nodes in the graph.
    nodes: RwLock<HashMap<NodeId, HnswNode>>,
    /// Entry point node ID (top level).
    entry_point: RwLock<Option<NodeId>>,
    /// Maximum level in the graph.
    max_level: RwLock<usize>,
    /// Multiplier for level generation.
    level_multiplier: f64,
}

impl HnswIndex {
    /// Create a new HNSW index with the given configuration.
    pub fn new(config: IndexConfig) -> Self {
        // ml = 1 / ln(M) is the standard choice
        let level_multiplier = 1.0 / (config.m as f64).ln();

        Self {
            config,
            nodes: RwLock::new(HashMap::new()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            level_multiplier,
        }
    }

    /// Create with default HNSW config for given dimension.
    pub fn with_dim(dim: usize) -> Self {
        Self::new(IndexConfig::hnsw(dim, DistanceMetric::Cosine))
    }

    /// Get distance function based on metric.
    fn distance_fn(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => cosine_distance_simd,
            DistanceMetric::Euclidean => euclidean_distance_simd,
            DistanceMetric::DotProduct => negative_dot_product_simd,
        }
    }

    /// Generate random level for a new node.
    fn random_level(&self) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let uniform: f64 = rng.r#gen();
        (-uniform.ln() * self.level_multiplier).floor() as usize
    }

    /// Get distance between node and query.
    fn distance_to_query(&self, node_id: NodeId, query: &[f32], nodes: &HashMap<NodeId, HnswNode>) -> f32 {
        let distance_fn = self.distance_fn();
        if let Some(node) = nodes.get(&node_id) {
            distance_fn(query, node.embedding.as_slice())
        } else {
            f32::MAX
        }
    }

    /// Search layer for nearest neighbors.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<NodeId>,
        ef: usize,
        level: usize,
        nodes: &HashMap<NodeId, HnswNode>,
    ) -> Vec<DistanceId> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut candidates: BinaryHeap<DistanceId> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxDistanceId> = BinaryHeap::new();

        // Initialize with entry points
        for ep in entry_points {
            let dist = self.distance_to_query(ep, query, nodes);
            candidates.push(DistanceId { distance: dist, id: ep });
            results.push(MaxDistanceId { distance: dist, id: ep });
            visited.insert(ep);
        }

        while let Some(current) = candidates.pop() {
            // Check if we can stop
            if let Some(furthest) = results.peek() {
                if current.distance > furthest.distance {
                    break;
                }
            }

            // Explore neighbors
            if let Some(node) = nodes.get(&current.id) {
                if level < node.neighbors.len() {
                    for &neighbor_id in &node.neighbors[level] {
                        if visited.contains(&neighbor_id) {
                            continue;
                        }
                        visited.insert(neighbor_id);

                        let dist = self.distance_to_query(neighbor_id, query, nodes);

                        // Check if this neighbor is worth exploring
                        let dominated = results.len() >= ef
                            && results
                                .peek()
                                .map(|f| dist >= f.distance)
                                .unwrap_or(false);

                        if !dominated {
                            candidates.push(DistanceId {
                                distance: dist,
                                id: neighbor_id,
                            });
                            results.push(MaxDistanceId {
                                distance: dist,
                                id: neighbor_id,
                            });

                            // Keep only ef best results
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert results to sorted vec
        let mut result_vec: Vec<DistanceId> = results
            .into_iter()
            .map(|m| DistanceId {
                distance: m.distance,
                id: m.id,
            })
            .collect();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result_vec
    }

    /// Select M best neighbors using simple heuristic.
    fn select_neighbors(&self, candidates: &[DistanceId], m: usize) -> Vec<NodeId> {
        candidates.iter().take(m).map(|d| d.id).collect()
    }

    /// Search with custom ef parameter.
    pub fn search_with_ef(&self, query: &Embedding, k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if query.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: query.dim(),
            });
        }

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = *self.max_level.read();

        let ep = match *entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let query_slice = query.as_slice();

        // Start from top level, descend to level 0
        let mut current_nearest = vec![ep];

        for level in (1..=max_level).rev() {
            let results = self.search_layer(query_slice, current_nearest, 1, level, &nodes);
            current_nearest = results.into_iter().map(|d| d.id).collect();
        }

        // Search layer 0 with full ef
        let results = self.search_layer(query_slice, current_nearest, ef.max(k), 0, &nodes);

        // Convert to SearchResult, take k best
        let search_results: Vec<SearchResult> = results
            .into_iter()
            .take(k)
            .map(|d| {
                let score = if self.config.metric == DistanceMetric::Cosine {
                    1.0 - d.distance // Convert distance to similarity
                } else {
                    d.distance
                };
                SearchResult::new(d.id, score)
            })
            .collect();

        Ok(search_results)
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()> {
        if embedding.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: embedding.dim(),
            });
        }

        let level = self.random_level();
        let m = self.config.m;
        let m_max = m * 2; // Max neighbors at level 0
        let ef_construction = self.config.ef_construction;

        // Clone embedding data for search before moving into node
        let query_vec: Vec<f32> = embedding.as_slice().to_vec();

        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();

        // Create new node with empty neighbor lists
        let new_node = HnswNode {
            id,
            embedding,
            neighbors: vec![Vec::new(); level + 1],
        };

        // First node insertion
        if entry_point.is_none() {
            nodes.insert(id, new_node);
            *entry_point = Some(id);
            *max_level = level;
            return Ok(());
        }

        let ep = entry_point.unwrap();

        // Find entry point at the target level
        let mut current_nearest = vec![ep];

        // Descend from top to level + 1
        for lv in (level + 1..=*max_level).rev() {
            let results = self.search_layer(&query_vec, current_nearest, 1, lv, &nodes);
            current_nearest = results.into_iter().map(|d| d.id).collect();
        }

        // Insert node first so we can update neighbors
        nodes.insert(id, new_node);

        // Connect at each level from level down to 0
        for lv in (0..=level.min(*max_level)).rev() {
            let candidates =
                self.search_layer(&query_vec, current_nearest.clone(), ef_construction, lv, &nodes);

            // Select neighbors for this level
            let max_neighbors = if lv == 0 { m_max } else { m };
            let neighbors = self.select_neighbors(&candidates, max_neighbors);

            // Update new node's neighbors
            if let Some(node) = nodes.get_mut(&id) {
                if lv < node.neighbors.len() {
                    node.neighbors[lv] = neighbors.clone();
                }
            }

            // Add bidirectional connections and collect nodes that need pruning
            let mut to_prune: Vec<(NodeId, Vec<f32>, Vec<NodeId>)> = Vec::new();

            for &neighbor_id in &neighbors {
                if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                    if lv < neighbor.neighbors.len() {
                        neighbor.neighbors[lv].push(id);

                        // Check if pruning is needed
                        if neighbor.neighbors[lv].len() > max_neighbors {
                            // Collect info for later pruning
                            let emb_copy = neighbor.embedding.as_slice().to_vec();
                            let neighbor_ids = neighbor.neighbors[lv].clone();
                            to_prune.push((neighbor_id, emb_copy, neighbor_ids));
                        }
                    }
                }
            }

            // Perform pruning after releasing mutable borrows
            let distance_fn = self.distance_fn();
            for (neighbor_id, neighbor_emb, neighbor_ids) in to_prune {
                let mut neighbor_dists: Vec<DistanceId> = neighbor_ids
                    .iter()
                    .filter_map(|&nid| {
                        nodes.get(&nid).map(|n| DistanceId {
                            distance: distance_fn(&neighbor_emb, n.embedding.as_slice()),
                            id: nid,
                        })
                    })
                    .collect();

                neighbor_dists.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                let pruned: Vec<NodeId> = neighbor_dists
                    .into_iter()
                    .take(max_neighbors)
                    .map(|d| d.id)
                    .collect();

                if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                    if lv < neighbor.neighbors.len() {
                        neighbor.neighbors[lv] = pruned;
                    }
                }
            }

            current_nearest = candidates.into_iter().map(|d| d.id).collect();
        }

        // Update entry point if new node has higher level
        if level > *max_level {
            *max_level = level;
            *entry_point = Some(id);
        }

        Ok(())
    }

    fn remove(&mut self, id: NodeId) -> Result<bool> {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();

        // Remove the node
        let removed = nodes.remove(&id).is_some();

        if removed {
            // Remove references from all neighbors
            for node in nodes.values_mut() {
                for level_neighbors in &mut node.neighbors {
                    level_neighbors.retain(|&nid| nid != id);
                }
            }

            // Update entry point if needed
            if *entry_point == Some(id) {
                *entry_point = nodes.keys().next().copied();
            }
        }

        Ok(removed)
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    fn get(&self, id: NodeId) -> Option<&Embedding> {
        // Similar limitation as flat index
        None
    }

    fn len(&self) -> usize {
        self.nodes.read().len()
    }

    fn metric(&self) -> DistanceMetric {
        self.config.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors() -> Vec<(NodeId, Embedding)> {
        vec![
            (NodeId::new(1), Embedding::new(vec![1.0, 0.0, 0.0])),
            (NodeId::new(2), Embedding::new(vec![0.0, 1.0, 0.0])),
            (NodeId::new(3), Embedding::new(vec![0.0, 0.0, 1.0])),
            (NodeId::new(4), Embedding::new(vec![0.707, 0.707, 0.0])),
            (NodeId::new(5), Embedding::new(vec![0.577, 0.577, 0.577])),
        ]
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = HnswIndex::new(IndexConfig::hnsw(3, DistanceMetric::Cosine));

        for (id, emb) in create_test_vectors() {
            index.insert(id, emb).unwrap();
        }

        assert_eq!(index.len(), 5);

        // Search for vector similar to [1, 0, 0]
        let query = Embedding::new(vec![0.9, 0.1, 0.0]);
        let results = index.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be ID 1 (most similar)
        assert_eq!(results[0].id, NodeId::new(1));
    }

    #[test]
    fn test_empty_search() {
        let index = HnswIndex::new(IndexConfig::hnsw(3, DistanceMetric::Cosine));
        let query = Embedding::new(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut index = HnswIndex::new(IndexConfig::hnsw(3, DistanceMetric::Cosine));

        index
            .insert(NodeId::new(1), Embedding::new(vec![1.0, 0.0, 0.0]))
            .unwrap();

        let query = Embedding::new(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, NodeId::new(1));
        assert!((results[0].score - 1.0).abs() < 1e-6); // Identical vectors
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = HnswIndex::new(IndexConfig::hnsw(3, DistanceMetric::Cosine));

        let result = index.insert(NodeId::new(1), Embedding::new(vec![1.0, 0.0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_remove() {
        let mut index = HnswIndex::new(IndexConfig::hnsw(3, DistanceMetric::Cosine));

        for (id, emb) in create_test_vectors() {
            index.insert(id, emb).unwrap();
        }

        assert_eq!(index.len(), 5);

        let removed = index.remove(NodeId::new(1)).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 4);

        // Search should not return removed node
        let query = Embedding::new(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 5).unwrap();
        assert!(!results.iter().any(|r| r.id == NodeId::new(1)));
    }

    #[test]
    fn test_larger_dataset() {
        use rand::Rng;
        let mut index = HnswIndex::new(IndexConfig::hnsw(128, DistanceMetric::Cosine));

        // Insert 100 random vectors
        let mut rng = rand::thread_rng();
        for i in 0..100 {
            let data: Vec<f32> = (0..128).map(|_| rng.r#gen::<f32>()).collect();
            let mut emb = Embedding::new(data);
            emb.normalize();
            index.insert(NodeId::new(i), emb).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search should return results
        let query_data: Vec<f32> = (0..128).map(|_| rng.r#gen::<f32>()).collect();
        let mut query = Embedding::new(query_data);
        query.normalize();

        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);

        // Results should be sorted by score (descending for cosine similarity)
        for window in results.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }
}
