//! IVF (Inverted File Index) for scalable approximate nearest neighbor search.
//!
//! Based on techniques from FAISS and LanceDB:
//! - Clusters vectors using k-means
//! - Searches only the most relevant clusters (nprobe)
//! - O(N/num_partitions) search complexity
//!
//! References:
//! - https://ann-benchmarks.com/
//! - LanceDB IVF implementation

use std::collections::HashMap;

use parking_lot::RwLock;
use rayon::prelude::*;

use crate::distance::{cosine_distance_simd, euclidean_distance_simd, negative_dot_product_simd};
use crate::{DistanceMetric, Embedding, NodeId, Result, SearchResult, VectorError};

use super::{IndexConfig, VectorIndex};

/// IVF index configuration.
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Number of partitions (clusters).
    pub num_partitions: usize,
    /// Number of partitions to search (nprobe).
    pub nprobe: usize,
    /// K-means iterations for training.
    pub kmeans_iterations: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            dim: 1536,
            metric: DistanceMetric::Cosine,
            num_partitions: 256,
            nprobe: 16,
            kmeans_iterations: 20,
        }
    }
}

impl IvfConfig {
    pub fn new(dim: usize, num_partitions: usize) -> Self {
        Self {
            dim,
            num_partitions,
            nprobe: (num_partitions / 16).max(1),
            ..Default::default()
        }
    }
}

/// A partition (cluster) in the IVF index.
struct IvfPartition {
    /// Centroid vector.
    centroid: Vec<f32>,
    /// Vectors in this partition.
    vectors: Vec<(NodeId, Vec<f32>)>,
}

/// IVF (Inverted File Index) for approximate nearest neighbor search.
pub struct IvfIndex {
    config: IvfConfig,
    /// Partitions (clusters).
    partitions: RwLock<Vec<IvfPartition>>,
    /// Whether the index has been trained.
    trained: RwLock<bool>,
    /// Training vectors (used before training).
    training_vectors: RwLock<Vec<(NodeId, Vec<f32>)>>,
}

impl IvfIndex {
    /// Create a new IVF index.
    pub fn new(config: IvfConfig) -> Self {
        Self {
            config,
            partitions: RwLock::new(Vec::new()),
            trained: RwLock::new(false),
            training_vectors: RwLock::new(Vec::new()),
        }
    }

    /// Get distance function based on metric.
    fn distance_fn(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => cosine_distance_simd,
            DistanceMetric::Euclidean => euclidean_distance_simd,
            DistanceMetric::DotProduct => negative_dot_product_simd,
        }
    }

    /// Train the index using k-means clustering.
    pub fn train(&self) -> Result<()> {
        let training_vectors = self.training_vectors.read();
        if training_vectors.len() < self.config.num_partitions {
            return Err(VectorError::IndexError(format!(
                "Need at least {} vectors to train, got {}",
                self.config.num_partitions,
                training_vectors.len()
            )));
        }

        // Initialize centroids randomly (reservoir sampling)
        let mut centroids: Vec<Vec<f32>> = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let mut selected: Vec<usize> = (0..training_vectors.len()).collect();

            // Fisher-Yates shuffle for first num_partitions
            for i in 0..self.config.num_partitions {
                let j = i + rng.r#gen::<usize>() % (training_vectors.len() - i);
                selected.swap(i, j);
            }

            selected[..self.config.num_partitions]
                .iter()
                .map(|&i| training_vectors[i].1.clone())
                .collect()
        };

        let distance_fn = self.distance_fn();

        // K-means iterations
        for _ in 0..self.config.kmeans_iterations {
            // Assign vectors to nearest centroid
            let assignments: Vec<usize> = training_vectors
                .par_iter()
                .map(|(_, vec)| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, distance_fn(vec, c)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();

            // Update centroids
            let dim = self.config.dim;
            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dim]; self.config.num_partitions];
            let mut counts: Vec<usize> = vec![0; self.config.num_partitions];

            for (i, (_, vec)) in training_vectors.iter().enumerate() {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for (j, &v) in vec.iter().enumerate() {
                    new_centroids[cluster][j] += v;
                }
            }

            // Average
            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for v in centroid.iter_mut() {
                        *v /= counts[i] as f32;
                    }
                } else {
                    // Keep old centroid for empty clusters
                    *centroid = centroids[i].clone();
                }
            }

            centroids = new_centroids;
        }

        // Create partitions with trained centroids
        let mut partitions: Vec<IvfPartition> = centroids
            .into_iter()
            .map(|centroid| IvfPartition {
                centroid,
                vectors: Vec::new(),
            })
            .collect();

        // Assign all training vectors to partitions
        for (id, vec) in training_vectors.iter() {
            let nearest = partitions
                .iter()
                .enumerate()
                .map(|(i, p)| (i, distance_fn(vec, &p.centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            partitions[nearest].vectors.push((*id, vec.clone()));
        }

        *self.partitions.write() = partitions;
        *self.trained.write() = true;

        Ok(())
    }

    /// Check if the index is trained.
    pub fn is_trained(&self) -> bool {
        *self.trained.read()
    }

    /// Get number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.config.num_partitions
    }

    /// Set nprobe (number of partitions to search).
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.config.nprobe = nprobe.min(self.config.num_partitions);
    }
}

impl VectorIndex for IvfIndex {
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()> {
        if embedding.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: embedding.dim(),
            });
        }

        let vec = embedding.as_slice().to_vec();

        if !*self.trained.read() {
            // Add to training vectors
            self.training_vectors.write().push((id, vec));
            return Ok(());
        }

        // Find nearest partition and add
        let distance_fn = self.distance_fn();
        let mut partitions = self.partitions.write();

        let nearest = partitions
            .iter()
            .enumerate()
            .map(|(i, p)| (i, distance_fn(&vec, &p.centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        partitions[nearest].vectors.push((id, vec));
        Ok(())
    }

    fn remove(&mut self, id: NodeId) -> Result<bool> {
        if !*self.trained.read() {
            let mut training = self.training_vectors.write();
            let len_before = training.len();
            training.retain(|(vid, _)| *vid != id);
            return Ok(training.len() < len_before);
        }

        let mut partitions = self.partitions.write();
        for partition in partitions.iter_mut() {
            let len_before = partition.vectors.len();
            partition.vectors.retain(|(vid, _)| *vid != id);
            if partition.vectors.len() < len_before {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        if query.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: query.dim(),
            });
        }

        if !*self.trained.read() {
            // Fallback to brute force on training vectors
            let training = self.training_vectors.read();
            let distance_fn = self.distance_fn();
            let query_slice = query.as_slice();

            let mut results: Vec<SearchResult> = training
                .iter()
                .map(|(id, vec)| SearchResult::new(*id, distance_fn(query_slice, vec)))
                .collect();

            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            results.truncate(k);

            if self.config.metric == DistanceMetric::Cosine {
                for r in &mut results {
                    r.score = 1.0 - r.score;
                }
            }
            return Ok(results);
        }

        let partitions = self.partitions.read();
        let distance_fn = self.distance_fn();
        let query_slice = query.as_slice();

        // Find nprobe nearest centroids
        let mut centroid_distances: Vec<(usize, f32)> = partitions
            .iter()
            .enumerate()
            .map(|(i, p)| (i, distance_fn(query_slice, &p.centroid)))
            .collect();

        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let probe_partitions: Vec<usize> = centroid_distances
            .iter()
            .take(self.config.nprobe)
            .map(|(i, _)| *i)
            .collect();

        // Search in selected partitions
        let mut results: Vec<SearchResult> = probe_partitions
            .par_iter()
            .flat_map(|&pi| {
                partitions[pi]
                    .vectors
                    .iter()
                    .map(|(id, vec)| SearchResult::new(*id, distance_fn(query_slice, vec)))
                    .collect::<Vec<_>>()
            })
            .collect();

        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        results.truncate(k);

        if self.config.metric == DistanceMetric::Cosine {
            for r in &mut results {
                r.score = 1.0 - r.score;
            }
        }

        Ok(results)
    }

    fn get(&self, _id: NodeId) -> Option<&Embedding> {
        None // Not implemented for IVF
    }

    fn len(&self) -> usize {
        if !*self.trained.read() {
            return self.training_vectors.read().len();
        }
        self.partitions.read().iter().map(|p| p.vectors.len()).sum()
    }

    fn metric(&self) -> DistanceMetric {
        self.config.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_training() {
        let config = IvfConfig::new(3, 4);
        let mut index = IvfIndex::new(config);

        // Insert training vectors
        for i in 0..100 {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / 100.0;
            let vec = vec![angle.cos(), angle.sin(), 0.0];
            index.insert(NodeId::new(i), Embedding::new(vec)).unwrap();
        }

        assert!(!index.is_trained());

        // Train
        index.train().unwrap();
        assert!(index.is_trained());
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_ivf_search() {
        let config = IvfConfig {
            dim: 3,
            num_partitions: 4,
            nprobe: 2,
            ..Default::default()
        };
        let mut index = IvfIndex::new(config);

        // Create clustered data
        for i in 0..100 {
            let cluster = i / 25;
            let offset = (i % 25) as f32 * 0.01;
            let vec = match cluster {
                0 => vec![1.0 + offset, 0.0, 0.0],
                1 => vec![0.0, 1.0 + offset, 0.0],
                2 => vec![-1.0 - offset, 0.0, 0.0],
                _ => vec![0.0, -1.0 - offset, 0.0],
            };
            index.insert(NodeId::new(i), Embedding::new(vec)).unwrap();
        }

        index.train().unwrap();

        // Search for vector in first cluster
        let query = Embedding::new(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // All results should be from cluster 0 (IDs 0-24)
        for r in &results {
            assert!(r.id.0 < 25);
        }
    }
}
