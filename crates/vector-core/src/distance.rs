//! Distance metrics for vector similarity using SIMD.
//!
//! Provides optimized implementations of:
//! - Cosine similarity
//! - Euclidean (L2) distance
//! - Dot product distance

use crate::simd::{dot_product_simd, squared_norm_simd};

/// Compute cosine similarity between two vectors.
///
/// Returns value in range [-1, 1] where:
/// - 1 = identical direction
/// - 0 = orthogonal
/// - -1 = opposite direction
///
/// For normalized vectors, this equals dot product.
#[inline]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = squared_norm_simd(a).sqrt();
    let norm_b = squared_norm_simd(b).sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute cosine distance (1 - cosine similarity).
///
/// Returns value in range [0, 2] where:
/// - 0 = identical direction
/// - 1 = orthogonal
/// - 2 = opposite direction
#[inline]
pub fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_simd(a, b)
}

/// Compute L2 (Euclidean) norm of a vector.
#[inline]
pub fn l2_norm_simd(a: &[f32]) -> f32 {
    squared_norm_simd(a).sqrt()
}

/// Compute squared Euclidean distance between two vectors.
///
/// More efficient than euclidean_distance when you only need relative ordering.
/// Uses ARM NEON on Apple M-series, AVX2/SSE on x86_64.
#[inline]
pub fn squared_euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::squared_euclidean_simd(a, b)
}

/// Compute Euclidean (L2) distance between two vectors.
#[inline]
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean_distance_simd(a, b).sqrt()
}

/// Compute negative dot product (for use as a distance metric).
///
/// For normalized vectors, minimizing this is equivalent to maximizing cosine similarity.
#[inline]
pub fn negative_dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    -dot_product_simd(a, b)
}

/// Batch compute distances from query to multiple vectors.
///
/// Returns distances in same order as input vectors.
pub fn batch_distances<F>(query: &[f32], vectors: &[&[f32]], distance_fn: F) -> Vec<f32>
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    vectors.iter().map(|v| distance_fn(query, v)).collect()
}

/// Parallel batch distance computation using rayon.
#[cfg(feature = "parallel")]
pub fn batch_distances_parallel<F>(query: &[f32], vectors: &[&[f32]], distance_fn: F) -> Vec<f32>
where
    F: Fn(&[f32], &[f32]) -> f32 + Sync,
{
    use rayon::prelude::*;

    vectors.par_iter().map(|v| distance_fn(query, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity_simd(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity_simd(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity_simd(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_45_degrees() {
        let a = vec![1.0, 0.0];
        let b = vec![0.707107, 0.707107]; // Normalized 45-degree vector
        let sim = cosine_similarity_simd(&a, &b);
        // cos(45°) ≈ 0.707
        assert!((sim - 0.707107).abs() < 1e-4);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = euclidean_distance_simd(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_squared_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        let dist = squared_euclidean_distance_simd(&a, &b);
        assert!((dist - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let a = vec![3.0, 4.0];
        let norm = l2_norm_simd(&a);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector_cosine() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity_simd(&a, &b);
        assert_eq!(sim, 0.0); // Zero vector should return 0
    }

    #[test]
    fn test_large_vector_distance() {
        let dim = 1536; // OpenAI embedding dimension
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((dim - i) as f32) * 0.001).collect();

        let sim = cosine_similarity_simd(&a, &b);
        assert!(sim > 0.0 && sim < 1.0); // Should be positive but not identical

        let dist = euclidean_distance_simd(&a, &b);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_batch_distances() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let distances = batch_distances(&query, &refs, cosine_distance_simd);

        assert!((distances[0] - 0.0).abs() < 1e-6); // Same direction
        assert!((distances[1] - 1.0).abs() < 1e-6); // Orthogonal
        assert!((distances[2] - 2.0).abs() < 1e-6); // Opposite
    }
}
