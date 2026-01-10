//! Ontology types for the context graph.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A concept in the ontology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Unique identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Parent concept (for IS_A hierarchy).
    pub parent: Option<Uuid>,
    /// Embedding vector ID (links to vector store).
    pub embedding_id: Option<u64>,
}

impl Concept {
    /// Create a new concept.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            parent: None,
            embedding_id: None,
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set parent concept.
    pub fn with_parent(mut self, parent: Uuid) -> Self {
        self.parent = Some(parent);
        self
    }
}

/// Types of relationships between concepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// Hierarchical: "X is a Y" (e.g., Dog IS_A Animal)
    IsA,
    /// Compositional: "X is part of Y" (e.g., Wheel PART_OF Car)
    PartOf,
    /// General semantic relation (e.g., Coffee RELATED_TO Energy)
    RelatedTo,
    /// Causal: "X causes Y" (e.g., Rain CAUSES Flood)
    Causes,
    /// Temporal: "X precedes Y" (e.g., Monday PRECEDES Tuesday)
    Precedes,
    /// Antonymy: "X is opposite of Y" (e.g., Hot OPPOSITE_OF Cold)
    OppositeOf,
    /// Synonymy: "X is same as Y" (e.g., Big SAME_AS Large)
    SameAs,
    /// Domain membership: "X belongs to domain Y"
    InDomain,
    /// Custom relationship type.
    Custom(u32),
}

/// A relationship between two concepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source concept.
    pub source: Uuid,
    /// Target concept.
    pub target: Uuid,
    /// Type of relationship.
    pub relation_type: RelationType,
    /// Strength/weight of relationship [0, 1].
    pub weight: f32,
    /// Optional metadata.
    pub metadata: Option<serde_json::Value>,
}

impl Relationship {
    /// Create a new relationship.
    pub fn new(source: Uuid, target: Uuid, relation_type: RelationType) -> Self {
        Self {
            source,
            target,
            relation_type,
            weight: 1.0,
            metadata: None,
        }
    }

    /// Set weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }
}
