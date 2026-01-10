//! Vector index implementations for EmergentDB.
//!
//! Provides multiple indexing strategies:
//! - Flat: Exact brute-force search
//! - HNSW: Hierarchical Navigable Small World graphs
//! - IVF: Inverted File Index with k-means clustering
//! - PQ: Product Quantization for memory-efficient search
//! - Emergent: MAP-Elites adaptive index that evolves optimal configuration

pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod pq;
pub mod emergent;

use crate::{DistanceMetric, Embedding, NodeId, Result, SearchResult};

/// Trait for vector indices.
pub trait VectorIndex: Send + Sync {
    /// Insert a vector with the given ID.
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()>;

    /// Remove a vector by ID.
    fn remove(&mut self, id: NodeId) -> Result<bool>;

    /// Search for k nearest neighbors.
    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>>;

    /// Get vector by ID.
    fn get(&self, id: NodeId) -> Option<&Embedding>;

    /// Number of vectors in the index.
    fn len(&self) -> usize;

    /// Check if index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the distance metric used by this index.
    fn metric(&self) -> DistanceMetric;
}

/// Index configuration.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Dimensionality of vectors.
    pub dim: usize,
    /// Distance metric to use.
    pub metric: DistanceMetric,
    /// HNSW-specific: number of neighbors per node.
    pub m: usize,
    /// HNSW-specific: size of dynamic candidate list during construction.
    pub ef_construction: usize,
    /// HNSW-specific: size of dynamic candidate list during search.
    pub ef_search: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dim: 1536, // Default to OpenAI embedding dimension
            metric: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

impl IndexConfig {
    /// Create config for flat (exact) index.
    pub fn flat(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            dim,
            metric,
            ..Default::default()
        }
    }

    /// Create config for HNSW index with default parameters.
    pub fn hnsw(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            dim,
            metric,
            ..Default::default()
        }
    }

    /// Create high-accuracy HNSW config (slower construction, faster search).
    pub fn hnsw_high_accuracy(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            dim,
            metric,
            m: 32,
            ef_construction: 400,
            ef_search: 100,
        }
    }

    /// Create fast HNSW config (faster construction, lower accuracy).
    pub fn hnsw_fast(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            dim,
            metric,
            m: 8,
            ef_construction: 100,
            ef_search: 20,
        }
    }
}
