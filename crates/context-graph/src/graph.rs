//! Context graph storage and operations.

use std::collections::{HashMap, HashSet};

use dashmap::DashMap;
use parking_lot::RwLock;
use uuid::Uuid;

use crate::ontology::{Concept, Relationship, RelationType};

/// Error types for context graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Concept not found: {0}")]
    ConceptNotFound(Uuid),

    #[error("Relationship already exists")]
    RelationshipExists,

    #[error("Cycle detected in IS_A hierarchy")]
    CycleDetected,
}

pub type Result<T> = std::result::Result<T, GraphError>;

/// Thread-safe context graph with ontological relationships.
pub struct ContextGraph {
    /// Concepts indexed by ID.
    concepts: DashMap<Uuid, Concept>,
    /// Outgoing relationships: source -> [(target, relationship)].
    outgoing: DashMap<Uuid, Vec<Relationship>>,
    /// Incoming relationships: target -> [(source, relationship)].
    incoming: DashMap<Uuid, Vec<Relationship>>,
    /// Name to ID index for fast lookup.
    name_index: DashMap<String, Uuid>,
}

impl ContextGraph {
    /// Create a new empty context graph.
    pub fn new() -> Self {
        Self {
            concepts: DashMap::new(),
            outgoing: DashMap::new(),
            incoming: DashMap::new(),
            name_index: DashMap::new(),
        }
    }

    /// Add a concept to the graph.
    pub fn add_concept(&self, concept: Concept) -> Uuid {
        let id = concept.id;
        self.name_index.insert(concept.name.clone(), id);
        self.concepts.insert(id, concept);
        id
    }

    /// Get concept by ID.
    pub fn get_concept(&self, id: Uuid) -> Option<Concept> {
        self.concepts.get(&id).map(|c| c.clone())
    }

    /// Get concept by name.
    pub fn get_concept_by_name(&self, name: &str) -> Option<Concept> {
        self.name_index
            .get(name)
            .and_then(|id| self.concepts.get(&*id).map(|c| c.clone()))
    }

    /// Add a relationship between concepts.
    pub fn add_relationship(&self, rel: Relationship) -> Result<()> {
        // Verify both concepts exist
        if !self.concepts.contains_key(&rel.source) {
            return Err(GraphError::ConceptNotFound(rel.source));
        }
        if !self.concepts.contains_key(&rel.target) {
            return Err(GraphError::ConceptNotFound(rel.target));
        }

        // Check for IS_A cycles
        if rel.relation_type == RelationType::IsA {
            if self.would_create_cycle(rel.source, rel.target) {
                return Err(GraphError::CycleDetected);
            }
        }

        // Add to outgoing
        self.outgoing
            .entry(rel.source)
            .or_insert_with(Vec::new)
            .push(rel.clone());

        // Add to incoming
        self.incoming
            .entry(rel.target)
            .or_insert_with(Vec::new)
            .push(rel);

        Ok(())
    }

    /// Check if adding an IS_A relationship would create a cycle.
    fn would_create_cycle(&self, source: Uuid, target: Uuid) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![target];

        while let Some(current) = stack.pop() {
            if current == source {
                return true;
            }
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            // Follow IS_A relationships
            if let Some(rels) = self.outgoing.get(&current) {
                for rel in rels.iter() {
                    if rel.relation_type == RelationType::IsA {
                        stack.push(rel.target);
                    }
                }
            }
        }

        false
    }

    /// Get all relationships from a concept.
    pub fn get_outgoing(&self, id: Uuid) -> Vec<Relationship> {
        self.outgoing
            .get(&id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get all relationships to a concept.
    pub fn get_incoming(&self, id: Uuid) -> Vec<Relationship> {
        self.incoming
            .get(&id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get relationships of a specific type from a concept.
    pub fn get_outgoing_by_type(&self, id: Uuid, rel_type: RelationType) -> Vec<Relationship> {
        self.outgoing
            .get(&id)
            .map(|v| v.iter().filter(|r| r.relation_type == rel_type).cloned().collect())
            .unwrap_or_default()
    }

    /// Get all ancestors in IS_A hierarchy.
    pub fn get_ancestors(&self, id: Uuid) -> Vec<Uuid> {
        let mut ancestors = Vec::new();
        let mut stack = vec![id];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for rel in self.get_outgoing_by_type(current, RelationType::IsA) {
                ancestors.push(rel.target);
                stack.push(rel.target);
            }
        }

        ancestors
    }

    /// Get all descendants in IS_A hierarchy.
    pub fn get_descendants(&self, id: Uuid) -> Vec<Uuid> {
        let mut descendants = Vec::new();
        let mut stack = vec![id];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if let Some(rels) = self.incoming.get(&current) {
                for rel in rels.iter() {
                    if rel.relation_type == RelationType::IsA {
                        descendants.push(rel.source);
                        stack.push(rel.source);
                    }
                }
            }
        }

        descendants
    }

    /// Number of concepts in the graph.
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Number of relationships in the graph.
    pub fn relationship_count(&self) -> usize {
        self.outgoing.iter().map(|e| e.len()).sum()
    }
}

impl Default for ContextGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_concept() {
        let graph = ContextGraph::new();
        let concept = Concept::new("Animal");
        let id = graph.add_concept(concept);

        assert!(graph.get_concept(id).is_some());
        assert!(graph.get_concept_by_name("Animal").is_some());
    }

    #[test]
    fn test_add_relationship() {
        let graph = ContextGraph::new();
        let animal = graph.add_concept(Concept::new("Animal"));
        let dog = graph.add_concept(Concept::new("Dog"));

        let rel = Relationship::new(dog, animal, RelationType::IsA);
        graph.add_relationship(rel).unwrap();

        let outgoing = graph.get_outgoing(dog);
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].target, animal);
    }

    #[test]
    fn test_cycle_detection() {
        let graph = ContextGraph::new();
        let a = graph.add_concept(Concept::new("A"));
        let b = graph.add_concept(Concept::new("B"));
        let c = graph.add_concept(Concept::new("C"));

        // A -> B -> C
        graph.add_relationship(Relationship::new(a, b, RelationType::IsA)).unwrap();
        graph.add_relationship(Relationship::new(b, c, RelationType::IsA)).unwrap();

        // C -> A would create cycle
        let result = graph.add_relationship(Relationship::new(c, a, RelationType::IsA));
        assert!(matches!(result, Err(GraphError::CycleDetected)));
    }

    #[test]
    fn test_get_ancestors() {
        let graph = ContextGraph::new();
        let living = graph.add_concept(Concept::new("Living"));
        let animal = graph.add_concept(Concept::new("Animal"));
        let mammal = graph.add_concept(Concept::new("Mammal"));
        let dog = graph.add_concept(Concept::new("Dog"));

        graph.add_relationship(Relationship::new(animal, living, RelationType::IsA)).unwrap();
        graph.add_relationship(Relationship::new(mammal, animal, RelationType::IsA)).unwrap();
        graph.add_relationship(Relationship::new(dog, mammal, RelationType::IsA)).unwrap();

        let ancestors = graph.get_ancestors(dog);
        assert!(ancestors.contains(&mammal));
        assert!(ancestors.contains(&animal));
        assert!(ancestors.contains(&living));
    }
}
