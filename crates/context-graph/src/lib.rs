//! Context Graph - Ontology-based semantic relationships.
//!
//! This crate provides:
//! - Typed relationships (IS_A, PART_OF, RELATED_TO, etc.)
//! - Hierarchical concept organization
//! - Graph traversal with relationship filtering
//! - Integration with vector similarity search

pub mod ontology;
pub mod graph;
pub mod traversal;

pub use ontology::{Concept, Relationship, RelationType};
pub use graph::ContextGraph;
