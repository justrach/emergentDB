//! Graph traversal algorithms.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering;

use uuid::Uuid;

use crate::graph::ContextGraph;
use crate::ontology::RelationType;

/// A node in the traversal path with cost.
#[derive(Clone)]
struct PathNode {
    id: Uuid,
    cost: f32,
    path: Vec<Uuid>,
}

impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for PathNode {}

impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

/// Traversal configuration.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum depth to traverse.
    pub max_depth: usize,
    /// Relationship types to follow.
    pub relation_types: Option<Vec<RelationType>>,
    /// Minimum relationship weight to follow.
    pub min_weight: f32,
    /// Whether to follow reverse (incoming) edges.
    pub follow_incoming: bool,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            relation_types: None,
            min_weight: 0.0,
            follow_incoming: false,
        }
    }
}

/// Breadth-first traversal from a starting concept.
pub fn bfs_traverse(
    graph: &ContextGraph,
    start: Uuid,
    config: &TraversalConfig,
) -> Vec<Uuid> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    queue.push_back((start, 0));
    visited.insert(start);

    while let Some((current, depth)) = queue.pop_front() {
        if depth > config.max_depth {
            continue;
        }

        result.push(current);

        // Get outgoing relationships
        let mut rels = graph.get_outgoing(current);

        // Optionally include incoming
        if config.follow_incoming {
            rels.extend(graph.get_incoming(current));
        }

        for rel in rels {
            // Filter by relationship type
            if let Some(ref types) = config.relation_types {
                if !types.contains(&rel.relation_type) {
                    continue;
                }
            }

            // Filter by weight
            if rel.weight < config.min_weight {
                continue;
            }

            let next = if rel.source == current {
                rel.target
            } else {
                rel.source
            };

            if !visited.contains(&next) {
                visited.insert(next);
                queue.push_back((next, depth + 1));
            }
        }
    }

    result
}

/// Find shortest path between two concepts using Dijkstra's algorithm.
pub fn shortest_path(
    graph: &ContextGraph,
    start: Uuid,
    end: Uuid,
    config: &TraversalConfig,
) -> Option<Vec<Uuid>> {
    let mut distances: HashMap<Uuid, f32> = HashMap::new();
    let mut heap = BinaryHeap::new();

    distances.insert(start, 0.0);
    heap.push(PathNode {
        id: start,
        cost: 0.0,
        path: vec![start],
    });

    while let Some(PathNode { id, cost, path }) = heap.pop() {
        if id == end {
            return Some(path);
        }

        if path.len() > config.max_depth {
            continue;
        }

        if let Some(&best) = distances.get(&id) {
            if cost > best {
                continue;
            }
        }

        // Get neighbors
        let mut rels = graph.get_outgoing(id);
        if config.follow_incoming {
            rels.extend(graph.get_incoming(id));
        }

        for rel in rels {
            // Filter by relationship type
            if let Some(ref types) = config.relation_types {
                if !types.contains(&rel.relation_type) {
                    continue;
                }
            }

            // Filter by weight
            if rel.weight < config.min_weight {
                continue;
            }

            let next = if rel.source == id { rel.target } else { rel.source };
            let edge_cost = 1.0 / rel.weight.max(0.01); // Higher weight = lower cost
            let next_cost = cost + edge_cost;

            let better = distances.get(&next).map(|&d| next_cost < d).unwrap_or(true);
            if better {
                distances.insert(next, next_cost);
                let mut next_path = path.clone();
                next_path.push(next);
                heap.push(PathNode {
                    id: next,
                    cost: next_cost,
                    path: next_path,
                });
            }
        }
    }

    None
}

/// Find all paths between two concepts up to max length.
pub fn all_paths(
    graph: &ContextGraph,
    start: Uuid,
    end: Uuid,
    max_length: usize,
) -> Vec<Vec<Uuid>> {
    let mut result = Vec::new();
    let mut stack = vec![(start, vec![start])];

    while let Some((current, path)) = stack.pop() {
        if current == end {
            result.push(path);
            continue;
        }

        if path.len() >= max_length {
            continue;
        }

        for rel in graph.get_outgoing(current) {
            if !path.contains(&rel.target) {
                let mut new_path = path.clone();
                new_path.push(rel.target);
                stack.push((rel.target, new_path));
            }
        }
    }

    result
}

/// Get the semantic neighborhood of a concept.
/// Returns concepts reachable within n hops, weighted by path length.
pub fn semantic_neighborhood(
    graph: &ContextGraph,
    center: Uuid,
    max_hops: usize,
) -> Vec<(Uuid, f32)> {
    let mut scores: HashMap<Uuid, f32> = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back((center, 0));
    visited.insert(center);

    while let Some((current, depth)) = queue.pop_front() {
        if depth > max_hops {
            continue;
        }

        // Score decreases with distance
        let score = 1.0 / (depth as f32 + 1.0);
        scores.insert(current, score);

        for rel in graph.get_outgoing(current) {
            if !visited.contains(&rel.target) {
                visited.insert(rel.target);
                queue.push_back((rel.target, depth + 1));
            }
        }
    }

    let mut result: Vec<_> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{Concept, Relationship};

    fn create_test_graph() -> (ContextGraph, Uuid, Uuid, Uuid, Uuid) {
        let graph = ContextGraph::new();
        let a = graph.add_concept(Concept::new("A"));
        let b = graph.add_concept(Concept::new("B"));
        let c = graph.add_concept(Concept::new("C"));
        let d = graph.add_concept(Concept::new("D"));

        // A -> B -> C -> D
        graph.add_relationship(Relationship::new(a, b, RelationType::RelatedTo)).unwrap();
        graph.add_relationship(Relationship::new(b, c, RelationType::RelatedTo)).unwrap();
        graph.add_relationship(Relationship::new(c, d, RelationType::RelatedTo)).unwrap();

        (graph, a, b, c, d)
    }

    #[test]
    fn test_bfs_traverse() {
        let (graph, a, b, c, d) = create_test_graph();

        let config = TraversalConfig::default();
        let result = bfs_traverse(&graph, a, &config);

        assert!(result.contains(&a));
        assert!(result.contains(&b));
        assert!(result.contains(&c));
        assert!(result.contains(&d));
    }

    #[test]
    fn test_shortest_path() {
        let (graph, a, _, _, d) = create_test_graph();

        let config = TraversalConfig::default();
        let path = shortest_path(&graph, a, d, &config).unwrap();

        assert_eq!(path.len(), 4);
        assert_eq!(path[0], a);
        assert_eq!(path[3], d);
    }

    #[test]
    fn test_all_paths() {
        let graph = ContextGraph::new();
        let a = graph.add_concept(Concept::new("A"));
        let b = graph.add_concept(Concept::new("B"));
        let c = graph.add_concept(Concept::new("C"));

        // A -> B -> C and A -> C
        graph.add_relationship(Relationship::new(a, b, RelationType::RelatedTo)).unwrap();
        graph.add_relationship(Relationship::new(b, c, RelationType::RelatedTo)).unwrap();
        graph.add_relationship(Relationship::new(a, c, RelationType::RelatedTo)).unwrap();

        let paths = all_paths(&graph, a, c, 5);
        assert_eq!(paths.len(), 2); // Direct and through B
    }
}
