//! Behavior descriptors for MAP-Elites characterization.

use serde::{Deserialize, Serialize};

/// A behavior descriptor representing a point in behavior space.
///
/// Each dimension should be in [0, 1] range for proper grid mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorDescriptor {
    /// Feature values (each in [0, 1] range).
    features: Vec<f32>,
}

impl BehaviorDescriptor {
    /// Create a new behavior descriptor.
    pub fn new(features: Vec<f32>) -> Self {
        // Clamp features to [0, 1] range
        let features = features.into_iter().map(|f| f.clamp(0.0, 1.0)).collect();
        Self { features }
    }

    /// Create from raw values with automatic normalization.
    pub fn from_raw(values: Vec<f32>, mins: &[f32], maxs: &[f32]) -> Self {
        let features = values
            .iter()
            .zip(mins.iter().zip(maxs.iter()))
            .map(|(&v, (&min, &max))| {
                if max > min {
                    (v - min) / (max - min)
                } else {
                    0.5
                }
            })
            .collect();

        Self::new(features)
    }

    /// Get the feature values.
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// Number of dimensions.
    pub fn dims(&self) -> usize {
        self.features.len()
    }

    /// Euclidean distance to another behavior descriptor.
    pub fn distance(&self, other: &Self) -> f32 {
        self.features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Trait for computing behavior descriptors from solutions.
pub trait BehaviorCharacterizer<T> {
    /// Compute the behavior descriptor for a solution.
    fn characterize(&self, solution: &T) -> BehaviorDescriptor;
}

/// Query behavior characterizer based on embedding similarity.
pub struct QueryBehaviorCharacterizer {
    /// Reference embeddings for each behavior dimension.
    reference_embeddings: Vec<Vec<f32>>,
}

impl QueryBehaviorCharacterizer {
    /// Create a new characterizer with reference embeddings.
    ///
    /// Each reference embedding defines a behavior dimension.
    /// The behavior value is the cosine similarity to that reference.
    pub fn new(reference_embeddings: Vec<Vec<f32>>) -> Self {
        Self { reference_embeddings }
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 1e-10 && norm_b > 1e-10 {
            // Map [-1, 1] to [0, 1]
            (dot / (norm_a * norm_b) + 1.0) / 2.0
        } else {
            0.5
        }
    }
}

impl BehaviorCharacterizer<Vec<f32>> for QueryBehaviorCharacterizer {
    fn characterize(&self, embedding: &Vec<f32>) -> BehaviorDescriptor {
        let features: Vec<f32> = self
            .reference_embeddings
            .iter()
            .map(|ref_emb| Self::cosine_similarity(embedding, ref_emb))
            .collect();

        BehaviorDescriptor::new(features)
    }
}

/// Simple text-based behavior characterizer.
pub struct TextBehaviorCharacterizer {
    /// Words that indicate dimension values.
    dimension_keywords: Vec<Vec<String>>,
}

impl TextBehaviorCharacterizer {
    pub fn new(dimension_keywords: Vec<Vec<String>>) -> Self {
        Self { dimension_keywords }
    }
}

impl BehaviorCharacterizer<String> for TextBehaviorCharacterizer {
    fn characterize(&self, text: &String) -> BehaviorDescriptor {
        let text_lower = text.to_lowercase();

        let features: Vec<f32> = self
            .dimension_keywords
            .iter()
            .map(|keywords| {
                let count = keywords
                    .iter()
                    .filter(|kw| text_lower.contains(&kw.to_lowercase()))
                    .count();
                (count as f32 / keywords.len().max(1) as f32).min(1.0)
            })
            .collect();

        BehaviorDescriptor::new(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_descriptor_clamping() {
        let bd = BehaviorDescriptor::new(vec![-0.5, 1.5, 0.5]);
        assert_eq!(bd.features(), &[0.0, 1.0, 0.5]);
    }

    #[test]
    fn test_behavior_distance() {
        let bd1 = BehaviorDescriptor::new(vec![0.0, 0.0]);
        let bd2 = BehaviorDescriptor::new(vec![1.0, 0.0]);
        let bd3 = BehaviorDescriptor::new(vec![1.0, 1.0]);

        assert!((bd1.distance(&bd2) - 1.0).abs() < 1e-6);
        assert!((bd1.distance(&bd3) - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_from_raw_normalization() {
        let bd = BehaviorDescriptor::from_raw(
            vec![50.0, 0.0],
            &[0.0, -100.0],
            &[100.0, 100.0],
        );

        assert!((bd.features()[0] - 0.5).abs() < 1e-6);
        assert!((bd.features()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_text_characterizer() {
        let char = TextBehaviorCharacterizer::new(vec![
            vec!["fast".to_string(), "quick".to_string()],
            vec!["accurate".to_string(), "precise".to_string()],
        ]);

        let text = "This is a fast and accurate system".to_string();
        let bd = char.characterize(&text);

        assert!(bd.features()[0] > 0.0); // Contains "fast"
        assert!(bd.features()[1] > 0.0); // Contains "accurate"
    }
}
