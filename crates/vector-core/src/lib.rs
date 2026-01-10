//! # EmergentDB Vector Core
//!
//! Self-optimizing vector database using MAP-Elites quality-diversity.
//!
//! EmergentDB automatically evolves the optimal index configuration for your data
//! by exploring different strategies (HNSW, IVF, PQ) and selecting the best one
//! based on your optimization priorities (recall, speed, memory).
//!
//! ## Index Types
//!
//! - **Flat**: Exact brute-force search (baseline)
//! - **HNSW**: Hierarchical Navigable Small World graphs (high recall)
//! - **IVF**: Inverted File Index with k-means clustering (scalable)
//! - **PQ**: Product Quantization (memory efficient)
//! - **Emergent**: MAP-Elites adaptive index that evolves optimal configuration
//!
//! ## Example: Self-Optimizing Index
//!
//! ```rust,ignore
//! use vector_core::{EmergentIndex, EmergentConfig, Embedding, NodeId, OptimizationPriority};
//!
//! // Create emergent index
//! let config = EmergentConfig {
//!     dim: 384,
//!     priority: OptimizationPriority::recall_first(),
//!     ..Default::default()
//! };
//! let mut index = EmergentIndex::new(config);
//!
//! // Insert vectors
//! for i in 0..10000 {
//!     index.insert(NodeId::new(i), embedding).unwrap();
//! }
//!
//! // Evolve to find optimal configuration
//! let best = index.evolve().unwrap();
//! println!("Best config: {:?} with recall {:.2}%", best.genome.index_type, best.metrics.recall_at_10 * 100.0);
//!
//! // Search uses the evolved optimal index
//! let results = index.search(&query, 10).unwrap();
//! ```

pub mod simd;
pub mod distance;
pub mod index;

// Re-export commonly used items
pub use distance::{cosine_similarity_simd, euclidean_distance_simd, l2_norm_simd};
pub use index::{IndexConfig, VectorIndex};
pub use index::flat::FlatIndex;
pub use index::hnsw::HnswIndex;
pub use index::ivf::IvfIndex;
pub use index::pq::PqIndex;
pub use index::emergent::{EmergentIndex, EmergentConfig, IndexGenome, IndexType, OptimizationPriority, Elite};

use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// A dense vector embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    data: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding from raw data
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Create a zero embedding of given dimension
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
        }
    }

    /// Get the dimension of this embedding
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Normalize the embedding in-place (L2 norm)
    pub fn normalize(&mut self) {
        let norm = distance::l2_norm_simd(&self.data);
        if norm > 1e-10 {
            for x in &mut self.data {
                *x /= norm;
            }
        }
    }

    /// Return a normalized copy
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Get the raw slice
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable raw slice
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl Deref for Embedding {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Embedding {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(data: Vec<f32>) -> Self {
        Self::new(data)
    }
}

/// Node ID in the vector index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// A search result with distance/similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: NodeId,
    pub score: f32,
}

impl SearchResult {
    pub fn new(id: NodeId, score: f32) -> Self {
        Self { id, score }
    }
}

/// Distance metric for vector comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine distance)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Dot product (for normalized vectors, same as cosine)
    DotProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Error types for vector operations
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Empty input")]
    EmptyInput,
}

pub type Result<T> = std::result::Result<T, VectorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_normalize() {
        let mut emb = Embedding::new(vec![3.0, 4.0]);
        emb.normalize();

        let norm = distance::l2_norm_simd(emb.as_slice());
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_zeros() {
        let emb = Embedding::zeros(384);
        assert_eq!(emb.dim(), 384);
        assert!(emb.iter().all(|&x| x == 0.0));
    }
}
