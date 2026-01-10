//! Flat (brute-force) vector index for exact nearest neighbor search.
//!
//! Simple but exact - suitable for small to medium datasets (<100k vectors).

use std::collections::HashMap;

use parking_lot::RwLock;

use crate::distance::{cosine_distance_simd, euclidean_distance_simd, negative_dot_product_simd};
use crate::{DistanceMetric, Embedding, NodeId, Result, SearchResult, VectorError};

use super::{IndexConfig, VectorIndex};

/// Flat index for exact nearest neighbor search.
///
/// Uses brute-force comparison with SIMD acceleration.
/// Best for small datasets or when exact results are required.
pub struct FlatIndex {
    config: IndexConfig,
    /// Vectors stored by ID.
    vectors: RwLock<HashMap<NodeId, Embedding>>,
}

impl FlatIndex {
    /// Create a new flat index with the given configuration.
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            vectors: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default config for given dimension.
    pub fn with_dim(dim: usize) -> Self {
        Self::new(IndexConfig::flat(dim, DistanceMetric::Cosine))
    }

    /// Get distance function based on metric.
    fn distance_fn(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => cosine_distance_simd,
            DistanceMetric::Euclidean => euclidean_distance_simd,
            DistanceMetric::DotProduct => negative_dot_product_simd,
        }
    }

    /// Batch insert multiple vectors.
    pub fn insert_batch(&mut self, items: Vec<(NodeId, Embedding)>) -> Result<()> {
        let mut vectors = self.vectors.write();
        for (id, emb) in items {
            if emb.dim() != self.config.dim {
                return Err(VectorError::DimensionMismatch {
                    expected: self.config.dim,
                    got: emb.dim(),
                });
            }
            vectors.insert(id, emb);
        }
        Ok(())
    }

    /// Search with custom ef (number of candidates to consider).
    /// For flat index, this is ignored since we always consider all vectors.
    pub fn search_with_ef(&self, query: &Embedding, k: usize, _ef: usize) -> Result<Vec<SearchResult>> {
        self.search(query, k)
    }
}

impl VectorIndex for FlatIndex {
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()> {
        if embedding.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: embedding.dim(),
            });
        }

        let mut vectors = self.vectors.write();
        vectors.insert(id, embedding);
        Ok(())
    }

    fn remove(&mut self, id: NodeId) -> Result<bool> {
        let mut vectors = self.vectors.write();
        Ok(vectors.remove(&id).is_some())
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        if query.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: query.dim(),
            });
        }

        let vectors = self.vectors.read();
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let distance_fn = self.distance_fn();

        // Compute all distances
        let mut results: Vec<SearchResult> = vectors
            .iter()
            .map(|(id, emb)| {
                let dist = distance_fn(query.as_slice(), emb.as_slice());
                SearchResult::new(*id, dist)
            })
            .collect();

        // Sort by distance (ascending)
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

        // Take top k
        results.truncate(k);

        // Convert distances to similarities for cosine metric
        if self.config.metric == DistanceMetric::Cosine {
            for r in &mut results {
                r.score = 1.0 - r.score; // Convert distance to similarity
            }
        }

        Ok(results)
    }

    fn get(&self, id: NodeId) -> Option<&Embedding> {
        // Note: This is a simplified implementation. In production,
        // you'd want to return a guard or use a different approach
        // to avoid lifetime issues.
        None // Placeholder - would need interior mutability pattern
    }

    fn len(&self) -> usize {
        self.vectors.read().len()
    }

    fn metric(&self) -> DistanceMetric {
        self.config.metric
    }
}

/// Parallel flat index using rayon for search.
pub struct ParallelFlatIndex {
    inner: FlatIndex,
}

impl ParallelFlatIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            inner: FlatIndex::new(config),
        }
    }

    /// Parallel search using rayon.
    pub fn search_parallel(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        use rayon::prelude::*;

        if query.dim() != self.inner.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.inner.config.dim,
                got: query.dim(),
            });
        }

        let vectors = self.inner.vectors.read();
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let distance_fn = self.inner.distance_fn();
        let query_slice = query.as_slice();

        // Parallel distance computation
        let mut results: Vec<SearchResult> = vectors
            .par_iter()
            .map(|(id, emb)| {
                let dist = distance_fn(query_slice, emb.as_slice());
                SearchResult::new(*id, dist)
            })
            .collect();

        // Sort (single-threaded is fine for final sort)
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        results.truncate(k);

        // Convert distances to similarities for cosine metric
        if self.inner.config.metric == DistanceMetric::Cosine {
            for r in &mut results {
                r.score = 1.0 - r.score;
            }
        }

        Ok(results)
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
        let mut index = FlatIndex::new(IndexConfig::flat(3, DistanceMetric::Cosine));

        for (id, emb) in create_test_vectors() {
            index.insert(id, emb).unwrap();
        }

        assert_eq!(index.len(), 5);

        // Search for vector similar to [1, 0, 0]
        let query = Embedding::new(vec![0.9, 0.1, 0.0]);
        let results = index.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be ID 1 (most similar to query)
        assert_eq!(results[0].id, NodeId::new(1));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = FlatIndex::new(IndexConfig::flat(3, DistanceMetric::Cosine));

        // Try to insert wrong dimension
        let result = index.insert(NodeId::new(1), Embedding::new(vec![1.0, 0.0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_remove() {
        let mut index = FlatIndex::new(IndexConfig::flat(3, DistanceMetric::Cosine));

        index
            .insert(NodeId::new(1), Embedding::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(index.len(), 1);

        let removed = index.remove(NodeId::new(1)).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 0);

        // Remove non-existent
        let removed = index.remove(NodeId::new(1)).unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_batch_insert() {
        let mut index = FlatIndex::new(IndexConfig::flat(3, DistanceMetric::Cosine));

        index.insert_batch(create_test_vectors()).unwrap();
        assert_eq!(index.len(), 5);
    }

    #[test]
    fn test_euclidean_metric() {
        let mut index = FlatIndex::new(IndexConfig::flat(3, DistanceMetric::Euclidean));

        index
            .insert(NodeId::new(1), Embedding::new(vec![0.0, 0.0, 0.0]))
            .unwrap();
        index
            .insert(NodeId::new(2), Embedding::new(vec![3.0, 4.0, 0.0]))
            .unwrap();

        let query = Embedding::new(vec![0.0, 0.0, 0.0]);
        let results = index.search(&query, 2).unwrap();

        // Closer vector (ID 1) should be first
        assert_eq!(results[0].id, NodeId::new(1));
        assert!((results[0].score - 0.0).abs() < 1e-6);
        assert!((results[1].score - 5.0).abs() < 1e-6); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_empty_search() {
        let index = FlatIndex::new(IndexConfig::flat(3, DistanceMetric::Cosine));
        let query = Embedding::new(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 10).unwrap();
        assert!(results.is_empty());
    }
}
