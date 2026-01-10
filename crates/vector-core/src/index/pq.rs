//! Product Quantization (PQ) for memory-efficient vector search.
//!
//! Based on techniques from FAISS:
//! - Divides vectors into M subvectors
//! - Learns K centroids per subspace (codebook)
//! - Encodes each vector as M bytes (assuming K=256)
//! - Uses asymmetric distance computation (ADC)
//!
//! References:
//! - "Product Quantization for Nearest Neighbor Search" (Jégou et al.)
//! - LanceDB IVF-PQ implementation

use std::collections::HashMap;

use parking_lot::RwLock;
use rayon::prelude::*;

use crate::distance::{euclidean_distance_simd};
use crate::{DistanceMetric, Embedding, NodeId, Result, SearchResult, VectorError};

use super::VectorIndex;

/// PQ index configuration.
#[derive(Debug, Clone)]
pub struct PqConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Number of subvectors (M). dim must be divisible by M.
    pub num_subvectors: usize,
    /// Number of centroids per subspace (K). Usually 256 for byte encoding.
    pub num_centroids: usize,
    /// K-means iterations for training.
    pub kmeans_iterations: usize,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            dim: 1536,
            num_subvectors: 96, // 1536 / 96 = 16 dimensions per subvector
            num_centroids: 256,
            kmeans_iterations: 25,
        }
    }
}

impl PqConfig {
    pub fn new(dim: usize, num_subvectors: usize) -> Result<Self> {
        if dim % num_subvectors != 0 {
            return Err(VectorError::IndexError(format!(
                "dim ({}) must be divisible by num_subvectors ({})",
                dim, num_subvectors
            )));
        }
        Ok(Self {
            dim,
            num_subvectors,
            ..Default::default()
        })
    }

    /// Dimension of each subvector.
    pub fn subvector_dim(&self) -> usize {
        self.dim / self.num_subvectors
    }
}

/// Codebook for one subspace.
#[derive(Debug, Clone)]
struct Codebook {
    /// Centroids for this subspace. Shape: [num_centroids, subvector_dim]
    centroids: Vec<Vec<f32>>,
}

impl Codebook {
    fn new(num_centroids: usize, subvector_dim: usize) -> Self {
        Self {
            centroids: vec![vec![0.0; subvector_dim]; num_centroids],
        }
    }

    /// Find nearest centroid index for a subvector.
    fn encode(&self, subvector: &[f32]) -> u8 {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, euclidean_distance_simd(subvector, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Get centroid by index.
    fn get_centroid(&self, idx: u8) -> &[f32] {
        &self.centroids[idx as usize]
    }
}

/// Encoded vector (compressed representation).
#[derive(Debug, Clone)]
struct EncodedVector {
    id: NodeId,
    /// One code per subspace.
    codes: Vec<u8>,
}

/// Product Quantization index.
pub struct PqIndex {
    config: PqConfig,
    /// Codebooks for each subspace.
    codebooks: RwLock<Vec<Codebook>>,
    /// Encoded vectors.
    encoded_vectors: RwLock<Vec<EncodedVector>>,
    /// Original vectors (kept for retrieval, optional).
    original_vectors: RwLock<HashMap<NodeId, Vec<f32>>>,
    /// Whether the index has been trained.
    trained: RwLock<bool>,
    /// Training vectors.
    training_vectors: RwLock<Vec<(NodeId, Vec<f32>)>>,
}

impl PqIndex {
    /// Create a new PQ index.
    pub fn new(config: PqConfig) -> Self {
        Self {
            config,
            codebooks: RwLock::new(Vec::new()),
            encoded_vectors: RwLock::new(Vec::new()),
            original_vectors: RwLock::new(HashMap::new()),
            trained: RwLock::new(false),
            training_vectors: RwLock::new(Vec::new()),
        }
    }

    /// Train the PQ codebooks using k-means on each subspace.
    pub fn train(&self) -> Result<()> {
        let training_vectors = self.training_vectors.read();
        let min_training = self.config.num_centroids;

        if training_vectors.len() < min_training {
            return Err(VectorError::IndexError(format!(
                "Need at least {} vectors to train, got {}",
                min_training,
                training_vectors.len()
            )));
        }

        let subvector_dim = self.config.subvector_dim();
        let num_subvectors = self.config.num_subvectors;
        let num_centroids = self.config.num_centroids;

        // Train codebook for each subspace in parallel
        let codebooks: Vec<Codebook> = (0..num_subvectors)
            .into_par_iter()
            .map(|m| {
                // Extract subvectors for this subspace
                let start = m * subvector_dim;
                let end = start + subvector_dim;

                let subvectors: Vec<&[f32]> = training_vectors
                    .iter()
                    .map(|(_, v)| &v[start..end])
                    .collect();

                // K-means for this subspace
                self.train_codebook(&subvectors, num_centroids, subvector_dim)
            })
            .collect();

        *self.codebooks.write() = codebooks;

        // Encode all training vectors
        let codebooks = self.codebooks.read();
        let encoded: Vec<EncodedVector> = training_vectors
            .par_iter()
            .map(|(id, vec)| {
                let codes = self.encode_vector(vec, &codebooks);
                EncodedVector { id: *id, codes }
            })
            .collect();

        *self.encoded_vectors.write() = encoded;

        // Store originals for retrieval
        let mut originals = self.original_vectors.write();
        for (id, vec) in training_vectors.iter() {
            originals.insert(*id, vec.clone());
        }

        *self.trained.write() = true;
        Ok(())
    }

    fn train_codebook(&self, subvectors: &[&[f32]], num_centroids: usize, dim: usize) -> Codebook {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize centroids with random samples
        let mut centroids: Vec<Vec<f32>> = {
            let mut indices: Vec<usize> = (0..subvectors.len()).collect();
            for i in 0..num_centroids.min(indices.len()) {
                let j = i + rng.r#gen::<usize>() % (indices.len() - i);
                indices.swap(i, j);
            }
            indices[..num_centroids.min(indices.len())]
                .iter()
                .map(|&i| subvectors[i].to_vec())
                .collect()
        };

        // Pad with zeros if not enough samples
        while centroids.len() < num_centroids {
            centroids.push(vec![0.0; dim]);
        }

        // K-means iterations
        for _ in 0..self.config.kmeans_iterations {
            // Assign to nearest centroid
            let assignments: Vec<usize> = subvectors
                .par_iter()
                .map(|sv| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, euclidean_distance_simd(sv, c)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();

            // Update centroids
            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dim]; num_centroids];
            let mut counts: Vec<usize> = vec![0; num_centroids];

            for (i, sv) in subvectors.iter().enumerate() {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for (j, &v) in sv.iter().enumerate() {
                    new_centroids[cluster][j] += v;
                }
            }

            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for v in centroid.iter_mut() {
                        *v /= counts[i] as f32;
                    }
                } else {
                    *centroid = centroids[i].clone();
                }
            }

            centroids = new_centroids;
        }

        let mut codebook = Codebook::new(num_centroids, dim);
        codebook.centroids = centroids;
        codebook
    }

    fn encode_vector(&self, vec: &[f32], codebooks: &[Codebook]) -> Vec<u8> {
        let subvector_dim = self.config.subvector_dim();

        (0..self.config.num_subvectors)
            .map(|m| {
                let start = m * subvector_dim;
                let end = start + subvector_dim;
                codebooks[m].encode(&vec[start..end])
            })
            .collect()
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        *self.trained.read()
    }

    /// Compute asymmetric distance (ADC) between query and encoded vector.
    fn asymmetric_distance(&self, distance_tables: &[Vec<f32>], codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(m, &code)| distance_tables[m][code as usize])
            .sum()
    }

    /// Precompute distance tables for a query.
    fn compute_distance_tables(&self, query: &[f32], codebooks: &[Codebook]) -> Vec<Vec<f32>> {
        let subvector_dim = self.config.subvector_dim();

        (0..self.config.num_subvectors)
            .map(|m| {
                let start = m * subvector_dim;
                let end = start + subvector_dim;
                let query_sub = &query[start..end];

                codebooks[m]
                    .centroids
                    .iter()
                    .map(|centroid| {
                        let diff: f32 = query_sub
                            .iter()
                            .zip(centroid.iter())
                            .map(|(q, c)| (q - c).powi(2))
                            .sum();
                        diff
                    })
                    .collect()
            })
            .collect()
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.config.dim * 4; // f32
        let compressed_bytes = self.config.num_subvectors; // u8 per subvector
        original_bytes as f32 / compressed_bytes as f32
    }
}

impl VectorIndex for PqIndex {
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()> {
        if embedding.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: embedding.dim(),
            });
        }

        let vec = embedding.as_slice().to_vec();

        if !*self.trained.read() {
            self.training_vectors.write().push((id, vec));
            return Ok(());
        }

        // Encode and add
        let codebooks = self.codebooks.read();
        let codes = self.encode_vector(&vec, &codebooks);
        drop(codebooks);

        self.encoded_vectors.write().push(EncodedVector { id, codes });
        self.original_vectors.write().insert(id, vec);

        Ok(())
    }

    fn remove(&mut self, id: NodeId) -> Result<bool> {
        if !*self.trained.read() {
            let mut training = self.training_vectors.write();
            let len_before = training.len();
            training.retain(|(vid, _)| *vid != id);
            return Ok(training.len() < len_before);
        }

        let mut encoded = self.encoded_vectors.write();
        let len_before = encoded.len();
        encoded.retain(|ev| ev.id != id);

        self.original_vectors.write().remove(&id);

        Ok(encoded.len() < len_before)
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        if query.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: query.dim(),
            });
        }

        if !*self.trained.read() {
            // Fallback to brute force
            let training = self.training_vectors.read();
            let query_slice = query.as_slice();

            let mut results: Vec<SearchResult> = training
                .iter()
                .map(|(id, vec)| SearchResult::new(*id, euclidean_distance_simd(query_slice, vec)))
                .collect();

            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            results.truncate(k);
            return Ok(results);
        }

        let codebooks = self.codebooks.read();
        let encoded = self.encoded_vectors.read();

        // Precompute distance tables
        let distance_tables = self.compute_distance_tables(query.as_slice(), &codebooks);

        // Compute ADC distances
        let mut results: Vec<SearchResult> = encoded
            .par_iter()
            .map(|ev| {
                let dist = self.asymmetric_distance(&distance_tables, &ev.codes);
                SearchResult::new(ev.id, dist)
            })
            .collect();

        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        results.truncate(k);

        Ok(results)
    }

    fn get(&self, _id: NodeId) -> Option<&Embedding> {
        None // Would require lifetime gymnastics; use get_original instead
    }

    fn len(&self) -> usize {
        if !*self.trained.read() {
            return self.training_vectors.read().len();
        }
        self.encoded_vectors.read().len()
    }

    fn metric(&self) -> DistanceMetric {
        DistanceMetric::Euclidean // PQ uses L2 for quantization
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_config() {
        let config = PqConfig::new(128, 8).unwrap();
        assert_eq!(config.subvector_dim(), 16);

        // Should fail if not divisible
        assert!(PqConfig::new(100, 8).is_err());
    }

    #[test]
    fn test_pq_training() {
        let config = PqConfig {
            dim: 16,
            num_subvectors: 4,
            num_centroids: 8,
            kmeans_iterations: 10,
        };
        let mut index = PqIndex::new(config);

        // Insert training vectors
        for i in 0..100 {
            let vec: Vec<f32> = (0..16).map(|j| (i * j) as f32 % 10.0).collect();
            index.insert(NodeId::new(i), Embedding::new(vec)).unwrap();
        }

        assert!(!index.is_trained());
        index.train().unwrap();
        assert!(index.is_trained());

        // Check compression ratio
        let ratio = index.compression_ratio();
        assert!(ratio > 10.0); // 64 bytes -> 4 bytes = 16x
    }

    #[test]
    fn test_pq_search() {
        let config = PqConfig {
            dim: 16,
            num_subvectors: 4,
            num_centroids: 16,
            kmeans_iterations: 15,
        };
        let mut index = PqIndex::new(config);

        // Insert clustered data
        for i in 0..100 {
            let cluster = i / 25;
            let offset = (i % 25) as f32 * 0.01;
            let vec = match cluster {
                0 => vec![1.0 + offset; 16],
                1 => vec![2.0 + offset; 16],
                2 => vec![3.0 + offset; 16],
                _ => vec![4.0 + offset; 16],
            };
            index.insert(NodeId::new(i), Embedding::new(vec)).unwrap();
        }

        index.train().unwrap();

        // Query for first cluster
        let query = Embedding::new(vec![1.0; 16]);
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be from cluster 0
        for r in &results {
            assert!(r.id.0 < 25, "Expected cluster 0 results, got id {}", r.id.0);
        }
    }
}
