//! EmergentIndex - Self-optimizing index using MAP-Elites quality-diversity.
//!
//! This is the core of EmergentDB: an index that evolves its own configuration
//! based on actual performance characteristics of the data.
//!
//! Key improvements:
//! - 3D behavior grid: recall × latency × memory (more balanced coverage)
//! - Fast ingestion with early termination and heuristic seeding
//! - Balanced fitness using geometric mean of normalized scores
//! - Throughput and build time in quality metrics
//!
//! Behavior dimensions:
//! - Recall@10 (accuracy)
//! - Query latency (microseconds per query)
//! - Memory efficiency (bytes per vector)

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rayon::prelude::*;

use crate::simd::{InsertStrategy, benchmark_insert_strategy};
use crate::{DistanceMetric, Embedding, NodeId, Result, SearchResult, VectorError};

use super::flat::FlatIndex;
use super::hnsw::HnswIndex;
use super::ivf::{IvfConfig, IvfIndex};
use super::{IndexConfig, VectorIndex};

/// Index type variants for the emergent system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexType {
    Flat,
    Hnsw,
    Ivf,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexType::Flat => write!(f, "Flat"),
            IndexType::Hnsw => write!(f, "HNSW"),
            IndexType::Ivf => write!(f, "IVF"),
        }
    }
}

/// Pre-evolved preset configurations for common use cases.
/// These are battle-tested configurations that skip the evolution process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexPreset {
    /// Brute-force flat index. 100% recall, O(n) search.
    /// Best for: Small datasets (<10K vectors), perfect accuracy requirements.
    Flat,

    /// Fast HNSW with low memory footprint.
    /// Best for: Speed-critical applications, moderate recall acceptable.
    /// Config: m=8, ef_construction=100, ef_search=20
    HnswFast,

    /// Balanced HNSW with good recall/speed tradeoff.
    /// Best for: General purpose, most use cases.
    /// Config: m=16, ef_construction=200, ef_search=50
    HnswBalanced,

    /// High-recall HNSW for accuracy-critical applications.
    /// Best for: When 99%+ recall is required.
    /// Config: m=32, ef_construction=300, ef_search=100
    HnswAccurate,

    /// Maximum quality HNSW with high memory usage.
    /// Best for: When recall must be as close to 100% as possible.
    /// Config: m=48, ef_construction=400, ef_search=200
    HnswMaxQuality,

    /// Fast IVF with low nprobe.
    /// Best for: Very large datasets (>1M), speed over accuracy.
    /// Config: 256 partitions, nprobe=8
    IvfFast,

    /// Balanced IVF with moderate nprobe.
    /// Best for: Large datasets with moderate recall requirements.
    /// Config: 256 partitions, nprobe=32
    IvfBalanced,

    /// High-recall IVF with high nprobe.
    /// Best for: Large datasets needing good recall.
    /// Config: 512 partitions, nprobe=64
    IvfAccurate,
}

impl IndexPreset {
    /// Convert preset to a concrete IndexGenome configuration.
    pub fn to_genome(&self) -> IndexGenome {
        match self {
            IndexPreset::Flat => IndexGenome {
                index_type: IndexType::Flat,
                ..Default::default()
            },
            IndexPreset::HnswFast => IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 8,
                hnsw_ef_construction: 100,
                hnsw_ef_search: 20,
                ..Default::default()
            },
            IndexPreset::HnswBalanced => IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 16,
                hnsw_ef_construction: 200,
                hnsw_ef_search: 50,
                ..Default::default()
            },
            IndexPreset::HnswAccurate => IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 32,
                hnsw_ef_construction: 300,
                hnsw_ef_search: 100,
                ..Default::default()
            },
            IndexPreset::HnswMaxQuality => IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 48,
                hnsw_ef_construction: 400,
                hnsw_ef_search: 200,
                ..Default::default()
            },
            IndexPreset::IvfFast => IndexGenome {
                index_type: IndexType::Ivf,
                ivf_partitions: 256,
                ivf_nprobe: 8,
                ..Default::default()
            },
            IndexPreset::IvfBalanced => IndexGenome {
                index_type: IndexType::Ivf,
                ivf_partitions: 256,
                ivf_nprobe: 32,
                ..Default::default()
            },
            IndexPreset::IvfAccurate => IndexGenome {
                index_type: IndexType::Ivf,
                ivf_partitions: 512,
                ivf_nprobe: 64,
                ..Default::default()
            },
        }
    }

    /// Get all available presets.
    pub fn all() -> Vec<Self> {
        vec![
            IndexPreset::Flat,
            IndexPreset::HnswFast,
            IndexPreset::HnswBalanced,
            IndexPreset::HnswAccurate,
            IndexPreset::HnswMaxQuality,
            IndexPreset::IvfFast,
            IndexPreset::IvfBalanced,
            IndexPreset::IvfAccurate,
        ]
    }

    /// Get recommended preset based on dataset size and priority.
    pub fn recommend(num_vectors: usize, priority: &str) -> Self {
        match (num_vectors, priority) {
            // Small datasets: use Flat for perfect recall
            (0..=5_000, _) => IndexPreset::Flat,

            // Medium datasets: HNSW variants
            (5_001..=100_000, "speed") => IndexPreset::HnswFast,
            (5_001..=100_000, "accuracy") => IndexPreset::HnswAccurate,
            (5_001..=100_000, _) => IndexPreset::HnswBalanced,

            // Large datasets: IVF or HNSW depending on priority
            (100_001..=1_000_000, "speed") => IndexPreset::IvfFast,
            (100_001..=1_000_000, "accuracy") => IndexPreset::HnswMaxQuality,
            (100_001..=1_000_000, _) => IndexPreset::IvfBalanced,

            // Very large: IVF is the only practical option
            (_, "accuracy") => IndexPreset::IvfAccurate,
            (_, "speed") => IndexPreset::IvfFast,
            (_, _) => IndexPreset::IvfBalanced,
        }
    }
}

impl std::fmt::Display for IndexPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexPreset::Flat => write!(f, "Flat (Brute Force)"),
            IndexPreset::HnswFast => write!(f, "HNSW Fast (m=8, ef=20)"),
            IndexPreset::HnswBalanced => write!(f, "HNSW Balanced (m=16, ef=50)"),
            IndexPreset::HnswAccurate => write!(f, "HNSW Accurate (m=32, ef=100)"),
            IndexPreset::HnswMaxQuality => write!(f, "HNSW Max Quality (m=48, ef=200)"),
            IndexPreset::IvfFast => write!(f, "IVF Fast (nprobe=8)"),
            IndexPreset::IvfBalanced => write!(f, "IVF Balanced (nprobe=32)"),
            IndexPreset::IvfAccurate => write!(f, "IVF Accurate (nprobe=64)"),
        }
    }
}

// ============================================================================
// PRE-COMPUTED ELITES GRID
// Industry-standard configurations based on OpenSearch, Milvus, and ANN-benchmarks
// Reference: https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/
// ============================================================================

/// A pre-computed elite configuration with expected performance metrics.
/// These are derived from industry benchmarks and research papers.
#[derive(Debug, Clone)]
pub struct PrecomputedElite {
    /// The index configuration
    pub genome: IndexGenome,
    /// Expected recall@10 (0.0 - 1.0)
    pub expected_recall: f32,
    /// Expected query latency factor (1.0 = baseline flat search)
    pub latency_factor: f32,
    /// Memory overhead factor (1.0 = vector data only)
    pub memory_factor: f32,
    /// Recommended minimum dataset size
    pub min_vectors: usize,
    /// Recommended maximum dataset size
    pub max_vectors: usize,
    /// Human-readable description
    pub description: &'static str,
}

/// Pre-computed elites grid - a 3D archive indexed by [recall_tier][speed_tier][scale_tier].
/// This replaces runtime evolution with battle-tested configurations.
pub struct PrecomputedElitesGrid {
    /// Grid dimensions: [recall: 0=low, 1=medium, 2=high][speed: 0=fast, 1=balanced, 2=accurate][scale: 0=small, 1=medium, 2=large]
    grid: Vec<Vec<Vec<PrecomputedElite>>>,
}

impl PrecomputedElitesGrid {
    /// Create the pre-computed elites grid with industry-standard configurations.
    /// Based on benchmarks from OpenSearch, Milvus, Pinecone, and ANN-benchmarks.
    pub fn new() -> Self {
        let mut grid = vec![vec![vec![]; 3]; 3];

        // =====================================================================
        // RECALL TIER 0: Speed Priority (70-85% recall acceptable)
        // =====================================================================

        // Speed + Fast + Small (< 10K)
        grid[0][0].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Flat,
                ..Default::default()
            },
            expected_recall: 1.0,
            latency_factor: 1.0,
            memory_factor: 1.0,
            min_vectors: 0,
            max_vectors: 10_000,
            description: "Flat index - perfect for small datasets",
        });

        // Speed + Fast + Medium (10K - 100K)
        grid[0][1].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 8,
                hnsw_ef_construction: 64,
                hnsw_ef_search: 16,
                ..Default::default()
            },
            expected_recall: 0.75,
            latency_factor: 0.05,
            memory_factor: 1.3,
            min_vectors: 10_000,
            max_vectors: 100_000,
            description: "Ultra-fast HNSW (OpenSearch low config)",
        });

        // Speed + Fast + Large (100K - 1M)
        grid[0][2].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Ivf,
                ivf_partitions: 256,
                ivf_nprobe: 4,
                ..Default::default()
            },
            expected_recall: 0.70,
            latency_factor: 0.02,
            memory_factor: 1.1,
            min_vectors: 100_000,
            max_vectors: 10_000_000,
            description: "Fast IVF for large-scale speed priority",
        });

        // =====================================================================
        // RECALL TIER 1: Balanced (85-95% recall)
        // =====================================================================

        // Balanced + Fast + Small
        grid[1][0].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 12,
                hnsw_ef_construction: 100,
                hnsw_ef_search: 32,
                ..Default::default()
            },
            expected_recall: 0.90,
            latency_factor: 0.08,
            memory_factor: 1.4,
            min_vectors: 1_000,
            max_vectors: 50_000,
            description: "Balanced HNSW for small-medium datasets",
        });

        // Balanced + Balanced + Medium (OpenSearch recommended)
        grid[1][1].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 16,
                hnsw_ef_construction: 128,
                hnsw_ef_search: 64,
                ..Default::default()
            },
            expected_recall: 0.92,
            latency_factor: 0.10,
            memory_factor: 1.5,
            min_vectors: 10_000,
            max_vectors: 500_000,
            description: "OpenSearch default - good all-rounder",
        });

        // Balanced + Accurate + Large
        grid[1][2].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Ivf,
                ivf_partitions: 512,
                ivf_nprobe: 32,
                ..Default::default()
            },
            expected_recall: 0.88,
            latency_factor: 0.05,
            memory_factor: 1.2,
            min_vectors: 100_000,
            max_vectors: 10_000_000,
            description: "Balanced IVF for million-scale",
        });

        // =====================================================================
        // RECALL TIER 2: High Recall (95-99%+ recall)
        // =====================================================================

        // High + Fast + Small
        grid[2][0].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 24,
                hnsw_ef_construction: 200,
                hnsw_ef_search: 100,
                ..Default::default()
            },
            expected_recall: 0.97,
            latency_factor: 0.15,
            memory_factor: 1.8,
            min_vectors: 1_000,
            max_vectors: 100_000,
            description: "High-recall HNSW (Milvus recommended)",
        });

        // High + Balanced + Medium (OpenSearch high config)
        grid[2][1].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 32,
                hnsw_ef_construction: 256,
                hnsw_ef_search: 128,
                ..Default::default()
            },
            expected_recall: 0.98,
            latency_factor: 0.20,
            memory_factor: 2.0,
            min_vectors: 10_000,
            max_vectors: 1_000_000,
            description: "High-recall HNSW (OpenSearch high config)",
        });

        // High + Accurate + Large (Maximum quality)
        grid[2][2].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 48,
                hnsw_ef_construction: 400,
                hnsw_ef_search: 200,
                ..Default::default()
            },
            expected_recall: 0.99,
            latency_factor: 0.30,
            memory_factor: 2.5,
            min_vectors: 10_000,
            max_vectors: 5_000_000,
            description: "Maximum quality HNSW (research-grade)",
        });

        // Add ultra-high quality variant
        grid[2][2].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 64,
                hnsw_ef_construction: 512,
                hnsw_ef_search: 300,
                ..Default::default()
            },
            expected_recall: 0.995,
            latency_factor: 0.40,
            memory_factor: 3.0,
            min_vectors: 50_000,
            max_vectors: 2_000_000,
            description: "Ultra-high quality (99.5% recall target)",
        });

        // Add OpenSearch portfolio configurations
        grid[1][1].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 32,
                hnsw_ef_construction: 128,
                hnsw_ef_search: 32,
                ..Default::default()
            },
            expected_recall: 0.90,
            latency_factor: 0.08,
            memory_factor: 1.7,
            min_vectors: 10_000,
            max_vectors: 500_000,
            description: "OpenSearch portfolio config #2",
        });

        grid[2][1].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 64,
                hnsw_ef_construction: 128,
                hnsw_ef_search: 128,
                ..Default::default()
            },
            expected_recall: 0.96,
            latency_factor: 0.18,
            memory_factor: 2.2,
            min_vectors: 10_000,
            max_vectors: 1_000_000,
            description: "OpenSearch portfolio config #4",
        });

        grid[2][2].push(PrecomputedElite {
            genome: IndexGenome {
                index_type: IndexType::Hnsw,
                hnsw_m: 128,
                hnsw_ef_construction: 256,
                hnsw_ef_search: 256,
                ..Default::default()
            },
            expected_recall: 0.998,
            latency_factor: 0.50,
            memory_factor: 4.0,
            min_vectors: 100_000,
            max_vectors: 1_000_000,
            description: "OpenSearch portfolio config #5 (max quality)",
        });

        Self { grid }
    }

    /// Get all elites in a specific cell of the grid.
    pub fn get_cell(&self, recall_tier: usize, speed_tier: usize) -> &[PrecomputedElite] {
        &self.grid[recall_tier.min(2)][speed_tier.min(2)]
    }

    /// Select the best elite for given requirements.
    pub fn select_best(
        &self,
        num_vectors: usize,
        min_recall: f32,
        max_latency_factor: f32,
    ) -> Option<&PrecomputedElite> {
        // Determine tiers based on requirements
        let recall_tier = if min_recall >= 0.95 { 2 } else if min_recall >= 0.85 { 1 } else { 0 };
        let speed_tier = if max_latency_factor <= 0.1 { 0 } else if max_latency_factor <= 0.25 { 1 } else { 2 };

        // Search in the target cell first
        let candidates = self.get_cell(recall_tier, speed_tier);

        candidates
            .iter()
            .filter(|e| {
                e.expected_recall >= min_recall
                    && e.latency_factor <= max_latency_factor
                    && num_vectors >= e.min_vectors
                    && num_vectors <= e.max_vectors
            })
            .max_by(|a, b| a.expected_recall.partial_cmp(&b.expected_recall).unwrap())
    }

    /// Get all elites that match the given dataset size.
    pub fn get_for_scale(&self, num_vectors: usize) -> Vec<&PrecomputedElite> {
        self.grid
            .iter()
            .flatten()
            .flatten()
            .filter(|e| num_vectors >= e.min_vectors && num_vectors <= e.max_vectors)
            .collect()
    }

    /// Get the recommended configuration based on dataset size and priority.
    pub fn recommend(&self, num_vectors: usize, priority: &str) -> &PrecomputedElite {
        let (recall_tier, speed_tier) = match priority {
            "speed" | "fast" => (0, 0),
            "balanced" | "default" => (1, 1),
            "accuracy" | "recall" | "high" => (2, 1),
            "max" | "maximum" | "ultra" => (2, 2),
            _ => (1, 1), // default to balanced
        };

        // Get candidates and find best match for scale
        let candidates = self.get_cell(recall_tier, speed_tier);
        candidates
            .iter()
            .filter(|e| num_vectors >= e.min_vectors && num_vectors <= e.max_vectors)
            .next()
            .unwrap_or_else(|| {
                // Fallback to any matching scale
                self.get_for_scale(num_vectors)
                    .into_iter()
                    .next()
                    .unwrap_or(&self.grid[1][1][0]) // Ultimate fallback
            })
    }

    /// Get total number of elite configurations.
    pub fn len(&self) -> usize {
        self.grid.iter().flatten().map(|v| v.len()).sum()
    }

    /// Check if grid is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all unique configurations for benchmarking.
    pub fn all_elites(&self) -> Vec<&PrecomputedElite> {
        self.grid.iter().flatten().flatten().collect()
    }

    /// Convert to serializable format for saving.
    pub fn to_serializable(&self) -> Vec<SerializableElite> {
        self.all_elites()
            .iter()
            .map(|e| SerializableElite {
                index_type: match e.genome.index_type {
                    IndexType::Flat => "flat".to_string(),
                    IndexType::Hnsw => "hnsw".to_string(),
                    IndexType::Ivf => "ivf".to_string(),
                },
                hnsw_m: e.genome.hnsw_m,
                hnsw_ef_construction: e.genome.hnsw_ef_construction,
                hnsw_ef_search: e.genome.hnsw_ef_search,
                ivf_partitions: e.genome.ivf_partitions,
                ivf_nprobe: e.genome.ivf_nprobe,
                expected_recall: e.expected_recall,
                latency_factor: e.latency_factor,
                memory_factor: e.memory_factor,
                min_vectors: e.min_vectors,
                max_vectors: e.max_vectors,
                description: e.description.to_string(),
            })
            .collect()
    }
}

impl Default for PrecomputedElitesGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable elite configuration for JSON export.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializableElite {
    pub index_type: String,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub ivf_partitions: usize,
    pub ivf_nprobe: usize,
    pub expected_recall: f32,
    pub latency_factor: f32,
    pub memory_factor: f32,
    pub min_vectors: usize,
    pub max_vectors: usize,
    pub description: String,
}

/// A genome representing an index configuration.
#[derive(Debug, Clone)]
pub struct IndexGenome {
    pub index_type: IndexType,
    /// HNSW: m parameter (neighbors per node)
    pub hnsw_m: usize,
    /// HNSW: ef_construction
    pub hnsw_ef_construction: usize,
    /// HNSW: ef_search
    pub hnsw_ef_search: usize,
    /// IVF: number of partitions
    pub ivf_partitions: usize,
    /// IVF: nprobe (partitions to search)
    pub ivf_nprobe: usize,
}

impl Default for IndexGenome {
    fn default() -> Self {
        Self {
            index_type: IndexType::Hnsw,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            ivf_partitions: 256,
            ivf_nprobe: 16,
        }
    }
}

impl IndexGenome {
    /// Create a random genome.
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let index_type = match rng.r#gen::<u8>() % 3 {
            0 => IndexType::Flat,
            1 => IndexType::Hnsw,
            _ => IndexType::Ivf,
        };

        Self {
            index_type,
            hnsw_m: [8, 12, 16, 24, 32, 48][rng.r#gen::<usize>() % 6],
            hnsw_ef_construction: [50, 100, 150, 200, 300][rng.r#gen::<usize>() % 5],
            hnsw_ef_search: [10, 20, 40, 50, 80, 100, 150][rng.r#gen::<usize>() % 7],
            ivf_partitions: [32, 64, 128, 256, 512][rng.r#gen::<usize>() % 5],
            ivf_nprobe: [2, 4, 8, 16, 32, 64][rng.r#gen::<usize>() % 6],
        }
    }

    /// Create heuristic seed genomes (known good configurations).
    /// These cover diverse regions of the behavior space for better archive coverage.
    pub fn heuristic_seeds() -> Vec<Self> {
        vec![
            // ===== HNSW variants (high recall, varying speed/memory) =====
            // Balanced HNSW
            Self {
                index_type: IndexType::Hnsw,
                hnsw_m: 16,
                hnsw_ef_construction: 200,
                hnsw_ef_search: 50,
                ..Default::default()
            },
            // Fast HNSW (lower recall, faster)
            Self {
                index_type: IndexType::Hnsw,
                hnsw_m: 8,
                hnsw_ef_construction: 50,
                hnsw_ef_search: 10,
                ..Default::default()
            },
            // High-recall HNSW
            Self {
                index_type: IndexType::Hnsw,
                hnsw_m: 32,
                hnsw_ef_construction: 300,
                hnsw_ef_search: 100,
                ..Default::default()
            },
            // Ultra-fast HNSW (minimal memory)
            Self {
                index_type: IndexType::Hnsw,
                hnsw_m: 8,
                hnsw_ef_construction: 100,
                hnsw_ef_search: 20,
                ..Default::default()
            },
            // Memory-heavy HNSW (max neighbors)
            Self {
                index_type: IndexType::Hnsw,
                hnsw_m: 48,
                hnsw_ef_construction: 300,
                hnsw_ef_search: 150,
                ..Default::default()
            },
            // ===== Flat (baseline, perfect recall) =====
            Self {
                index_type: IndexType::Flat,
                ..Default::default()
            },
            // ===== IVF variants (trade recall for speed) =====
            // Fast IVF (few partitions, low nprobe)
            Self {
                index_type: IndexType::Ivf,
                ivf_partitions: 32,
                ivf_nprobe: 4,
                ..Default::default()
            },
            // Balanced IVF
            Self {
                index_type: IndexType::Ivf,
                ivf_partitions: 128,
                ivf_nprobe: 16,
                ..Default::default()
            },
            // High-recall IVF (many partitions, high nprobe)
            Self {
                index_type: IndexType::Ivf,
                ivf_partitions: 256,
                ivf_nprobe: 32,
                ..Default::default()
            },
            // Extreme IVF (max partitions)
            Self {
                index_type: IndexType::Ivf,
                ivf_partitions: 512,
                ivf_nprobe: 64,
                ..Default::default()
            },
        ]
    }

    /// Mutate the genome with adaptive step sizes.
    pub fn mutate(&self) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut new = self.clone();

        // Multi-mutation: mutate 1-2 parameters
        let num_mutations = if rng.r#gen::<bool>() { 1 } else { 2 };

        for _ in 0..num_mutations {
            match rng.r#gen::<u8>() % 6 {
                0 => {
                    new.index_type = match rng.r#gen::<u8>() % 3 {
                        0 => IndexType::Flat,
                        1 => IndexType::Hnsw,
                        _ => IndexType::Ivf,
                    }
                }
                1 => new.hnsw_m = [8, 12, 16, 24, 32, 48][rng.r#gen::<usize>() % 6],
                2 => new.hnsw_ef_construction = [50, 100, 150, 200, 300][rng.r#gen::<usize>() % 5],
                3 => new.hnsw_ef_search = [10, 20, 40, 50, 80, 100, 150][rng.r#gen::<usize>() % 7],
                4 => new.ivf_partitions = [32, 64, 128, 256, 512][rng.r#gen::<usize>() % 5],
                _ => new.ivf_nprobe = [2, 4, 8, 16, 32, 64][rng.r#gen::<usize>() % 6],
            }
        }

        new
    }

    /// Crossover two genomes.
    pub fn crossover(&self, other: &Self) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        Self {
            index_type: if rng.r#gen::<bool>() { self.index_type } else { other.index_type },
            hnsw_m: if rng.r#gen::<bool>() { self.hnsw_m } else { other.hnsw_m },
            hnsw_ef_construction: if rng.r#gen::<bool>() {
                self.hnsw_ef_construction
            } else {
                other.hnsw_ef_construction
            },
            hnsw_ef_search: if rng.r#gen::<bool>() {
                self.hnsw_ef_search
            } else {
                other.hnsw_ef_search
            },
            ivf_partitions: if rng.r#gen::<bool>() {
                self.ivf_partitions
            } else {
                other.ivf_partitions
            },
            ivf_nprobe: if rng.r#gen::<bool>() { self.ivf_nprobe } else { other.ivf_nprobe },
        }
    }
}

// ============================================================================
// InsertQD: Quality-Diversity for Insert Strategies (Meta-Framework)
// ============================================================================

/// A genome representing an insert strategy configuration.
/// This is evolved by the InsertQD system to find optimal insert strategies.
#[derive(Debug, Clone)]
pub struct InsertGenome {
    /// The SIMD strategy to use for normalization
    pub strategy: InsertStrategy,
    /// Batch size for insert operations (1 = sequential)
    pub batch_size: usize,
    /// Whether to use parallel processing (rayon)
    pub use_parallelism: bool,
    /// Prefetch distance for memory optimization
    pub prefetch_distance: usize,
}

impl Default for InsertGenome {
    fn default() -> Self {
        Self {
            strategy: InsertStrategy::SimdSequential,
            batch_size: 1,
            use_parallelism: false,
            prefetch_distance: 0,
        }
    }
}

impl InsertGenome {
    /// Create a random insert genome.
    /// Note: PreNormalized is excluded as it's for user-provided normalized data.
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Use all 6 SIMD strategies (not PreNormalized)
        let strategy = match rng.r#gen::<u8>() % 6 {
            0 => InsertStrategy::SimdSequential,
            1 => InsertStrategy::SimdBatch,
            2 => InsertStrategy::SimdParallel,
            3 => InsertStrategy::SimdChunked,
            4 => InsertStrategy::SimdUnrolled,
            _ => InsertStrategy::SimdInterleaved,
        };

        Self {
            strategy,
            batch_size: [1, 10, 50, 100, 500, 1000][rng.r#gen::<usize>() % 6],
            use_parallelism: rng.r#gen::<bool>(),
            prefetch_distance: [0, 2, 4, 8][rng.r#gen::<usize>() % 4],
        }
    }

    /// Heuristic seed configurations for insert strategies.
    /// These cover diverse regions of the insert behavior space.
    pub fn heuristic_seeds() -> Vec<Self> {
        vec![
            // ===== Sequential strategies (low overhead, single-threaded) =====
            Self {
                strategy: InsertStrategy::SimdSequential,
                batch_size: 1,
                use_parallelism: false,
                prefetch_distance: 0,
            },
            // ===== Batch strategies (balanced throughput) =====
            Self {
                strategy: InsertStrategy::SimdBatch,
                batch_size: 100,
                use_parallelism: false,
                prefetch_distance: 4,
            },
            // ===== Parallel strategies (max throughput) =====
            Self {
                strategy: InsertStrategy::SimdParallel,
                batch_size: 500,
                use_parallelism: true,
                prefetch_distance: 8,
            },
            // ===== Chunked - cache-friendly =====
            Self {
                strategy: InsertStrategy::SimdChunked,
                batch_size: 64,
                use_parallelism: false,
                prefetch_distance: 4,
            },
            // ===== Unrolled - CPU pipelining =====
            Self {
                strategy: InsertStrategy::SimdUnrolled,
                batch_size: 100,
                use_parallelism: false,
                prefetch_distance: 0,
            },
            // ===== Interleaved - memory bandwidth =====
            Self {
                strategy: InsertStrategy::SimdInterleaved,
                batch_size: 200,
                use_parallelism: false,
                prefetch_distance: 2,
            },
            // ===== Parallel + Unrolled combo =====
            Self {
                strategy: InsertStrategy::SimdUnrolled,
                batch_size: 400,
                use_parallelism: true,
                prefetch_distance: 4,
            },
            // ===== Parallel + Interleaved combo =====
            Self {
                strategy: InsertStrategy::SimdInterleaved,
                batch_size: 500,
                use_parallelism: true,
                prefetch_distance: 8,
            },
        ]
    }

    /// Mutate the insert genome.
    pub fn mutate(&self) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut new = self.clone();

        // Multi-mutation for better exploration
        let num_mutations = if rng.r#gen::<bool>() { 1 } else { 2 };

        for _ in 0..num_mutations {
            match rng.r#gen::<u8>() % 4 {
                0 => {
                    // Use all 6 SIMD strategies (not PreNormalized)
                    new.strategy = match rng.r#gen::<u8>() % 6 {
                        0 => InsertStrategy::SimdSequential,
                        1 => InsertStrategy::SimdBatch,
                        2 => InsertStrategy::SimdParallel,
                        3 => InsertStrategy::SimdChunked,
                        4 => InsertStrategy::SimdUnrolled,
                        _ => InsertStrategy::SimdInterleaved,
                    }
                }
                1 => new.batch_size = [1, 10, 50, 100, 500, 1000][rng.r#gen::<usize>() % 6],
                2 => new.use_parallelism = !new.use_parallelism,
                _ => new.prefetch_distance = [0, 2, 4, 8][rng.r#gen::<usize>() % 4],
            }
        }

        new
    }

    /// Crossover two insert genomes.
    pub fn crossover(&self, other: &Self) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        Self {
            strategy: if rng.r#gen::<bool>() { self.strategy } else { other.strategy },
            batch_size: if rng.r#gen::<bool>() { self.batch_size } else { other.batch_size },
            use_parallelism: if rng.r#gen::<bool>() {
                self.use_parallelism
            } else {
                other.use_parallelism
            },
            prefetch_distance: if rng.r#gen::<bool>() {
                self.prefetch_distance
            } else {
                other.prefetch_distance
            },
        }
    }
}

/// Performance metrics for an insert strategy.
#[derive(Debug, Clone)]
pub struct InsertMetrics {
    /// Throughput: vectors inserted per second
    pub throughput_vps: f32,
    /// Memory overhead per vector (beyond raw data)
    pub memory_overhead_bytes: f32,
    /// CPU efficiency (0-1): how well CPU is utilized
    pub cpu_efficiency: f32,
    /// Batch efficiency: throughput scaling with batch size
    pub batch_scaling_factor: f32,
}

impl InsertMetrics {
    /// Compute fitness for insert strategy.
    pub fn fitness(&self, priority: &InsertPriority) -> f32 {
        // Normalize throughput: 100k = 0.0, 10M = 1.0 (log scale)
        let throughput_score = ((self.throughput_vps.ln() - 100_000_f32.ln())
            / (10_000_000_f32.ln() - 100_000_f32.ln()))
        .clamp(0.0, 1.0);

        // Memory: 0 = 1.0, 100 bytes = 0.0
        let memory_score = 1.0 - (self.memory_overhead_bytes / 100.0).clamp(0.0, 1.0);

        // CPU efficiency is already 0-1
        let cpu_score = self.cpu_efficiency;

        // Batch scaling
        let batch_score = self.batch_scaling_factor.clamp(0.0, 1.0);

        // Weighted geometric mean
        let weights = [
            priority.throughput_weight,
            priority.memory_weight,
            priority.cpu_weight,
            priority.batch_weight,
        ];
        let scores = [throughput_score, memory_score, cpu_score, batch_score];

        let total_weight: f32 = weights.iter().sum();
        let mut product = 1.0_f32;
        for (score, weight) in scores.iter().zip(weights.iter()) {
            if *weight > 0.0 {
                product *= score.max(0.001).powf(*weight);
            }
        }
        product.powf(1.0 / total_weight)
    }

    /// Compute 2D behavior coordinates for InsertQD grid.
    /// Returns (throughput_bin, efficiency_bin) in [0, grid_size).
    ///
    /// The grid maps:
    /// - X-axis (throughput_bin): Log-scale throughput from 100K to 10M vectors/sec
    /// - Y-axis (efficiency_bin): Combined efficiency score (CPU + batch scaling)
    pub fn behavior_2d(&self, grid_size: usize) -> (usize, usize) {
        let gs = grid_size as f32;

        // Throughput dimension: 100K - 10M vectors/sec (log scale)
        // Using more realistic bounds for SIMD operations
        let tp_log = self.throughput_vps.max(100_000.0).ln();
        let tp_min = 100_000_f32.ln();
        let tp_max = 10_000_000_f32.ln();
        let throughput_norm = ((tp_log - tp_min) / (tp_max - tp_min)).clamp(0.0, 1.0);
        let throughput_bin = ((throughput_norm * (gs - 1.0)).round() as usize).min(grid_size - 1);

        // Efficiency dimension: weighted combination of CPU and batch efficiency
        let efficiency_score = (self.cpu_efficiency * 0.6 + self.batch_scaling_factor * 0.4).clamp(0.0, 1.0);
        let efficiency_bin = ((efficiency_score * (gs - 1.0)).round() as usize).min(grid_size - 1);

        (throughput_bin, efficiency_bin)
    }
}

/// Priority weights for insert optimization.
#[derive(Debug, Clone)]
pub struct InsertPriority {
    /// Weight for throughput (0-1)
    pub throughput_weight: f32,
    /// Weight for memory efficiency (0-1)
    pub memory_weight: f32,
    /// Weight for CPU efficiency (0-1)
    pub cpu_weight: f32,
    /// Weight for batch scaling (0-1)
    pub batch_weight: f32,
}

impl Default for InsertPriority {
    fn default() -> Self {
        Self {
            throughput_weight: 0.50,
            memory_weight: 0.15,
            cpu_weight: 0.20,
            batch_weight: 0.15,
        }
    }
}

impl InsertPriority {
    pub fn throughput_first() -> Self {
        Self {
            throughput_weight: 0.70,
            memory_weight: 0.10,
            cpu_weight: 0.10,
            batch_weight: 0.10,
        }
    }

    pub fn balanced() -> Self {
        Self {
            throughput_weight: 0.40,
            memory_weight: 0.20,
            cpu_weight: 0.20,
            batch_weight: 0.20,
        }
    }
}

/// Elite entry for InsertQD archive.
#[derive(Debug, Clone)]
pub struct InsertElite {
    pub genome: InsertGenome,
    pub metrics: InsertMetrics,
    pub fitness: f32,
}

/// 2D MAP-Elites archive for insert strategies.
type InsertArchive2D = Vec<Vec<Option<InsertElite>>>;

// ============================================================================
// End InsertQD Types
// ============================================================================

/// Performance metrics for an index configuration.
#[derive(Debug, Clone)]
pub struct IndexMetrics {
    /// Recall@10 (0.0 - 1.0)
    pub recall_at_10: f32,
    /// Average query latency in microseconds
    pub query_latency_us: f32,
    /// Memory usage in bytes per vector
    pub bytes_per_vector: f32,
    /// Build time in milliseconds
    pub build_time_ms: f32,
    /// Throughput: queries per second
    pub throughput_qps: f32,
}

impl IndexMetrics {
    /// Compute a balanced fitness score using geometric mean.
    /// This ensures all metrics must be good (no single metric dominance).
    /// Applies a recall floor: configurations with recall < 99% are heavily penalized.
    pub fn fitness(&self, priority: &OptimizationPriority) -> f32 {
        // RECALL FLOOR: Penalize configurations with recall < 99%
        // This prevents evolution from sacrificing recall for speed
        let recall_penalty = if self.recall_at_10 < 0.99 {
            // Cubic penalty below 99% recall - harsh penalty
            (self.recall_at_10 / 0.99).powi(3)
        } else {
            1.0
        };

        // Normalize each metric to 0-1 range
        let recall_score = self.recall_at_10;

        // Speed: 1us = 1.0, 10000us = 0.0 (log scale for better distribution)
        let speed_score = 1.0 - (self.query_latency_us.ln() / 10000_f32.ln()).clamp(0.0, 1.0);

        // Memory: 1000 bytes = 1.0, 10000 bytes = 0.0
        let memory_score = 1.0 - ((self.bytes_per_vector - 1000.0) / 9000.0).clamp(0.0, 1.0);

        // Build time: 1ms = 1.0, 10000ms = 0.0 (log scale)
        let build_score = 1.0 - (self.build_time_ms.ln() / 10000_f32.ln()).clamp(0.0, 1.0);

        // Weighted geometric mean for balanced optimization
        let weights = [
            priority.recall_weight,
            priority.speed_weight,
            priority.memory_weight,
            priority.build_weight,
        ];
        let scores = [recall_score, speed_score, memory_score, build_score];

        // Geometric mean: (s1^w1 * s2^w2 * ...)^(1/sum(w))
        let total_weight: f32 = weights.iter().sum();
        let mut product = 1.0_f32;
        for (score, weight) in scores.iter().zip(weights.iter()) {
            if *weight > 0.0 {
                product *= score.max(0.001).powf(*weight);
            }
        }

        // Apply recall floor penalty
        product.powf(1.0 / total_weight) * recall_penalty
    }

    /// Compute 3D behavior coordinates for MAP-Elites grid.
    /// Returns (recall_bin, latency_bin, memory_bin) in [0, grid_size).
    ///
    /// The grid maps:
    /// - X-axis (recall_bin): Recall from 0.0 to 1.0 (linear)
    /// - Y-axis (latency_bin): Query latency from 1us to 1000us (log scale)
    /// - Z-axis (memory_bin): Memory per vector from 3000 to 6000 bytes
    pub fn behavior_3d(&self, grid_size: usize) -> (usize, usize, usize) {
        let gs = grid_size as f32;

        // Recall: 0.0 - 1.0 mapped linearly
        let recall_bin = ((self.recall_at_10 * (gs - 1.0)).round() as usize).min(grid_size - 1);

        // Latency: log scale 1us - 1000us (tighter range for small datasets)
        let lat_log = self.query_latency_us.max(1.0).ln();
        let lat_min = 1_f32.ln();  // 1us minimum
        let lat_max = 1000_f32.ln(); // 1000us maximum
        let lat_norm = ((lat_log - lat_min) / (lat_max - lat_min)).clamp(0.0, 1.0);
        let latency_bin = ((lat_norm * (gs - 1.0)).round() as usize).min(grid_size - 1);

        // Memory: 3000 - 6000 bytes (tighter range for 768-dim embeddings)
        // 768 dims * 4 bytes = 3072 bytes minimum
        let mem_min = 3000.0;
        let mem_max = 6000.0;
        let mem_norm = ((self.bytes_per_vector - mem_min) / (mem_max - mem_min)).clamp(0.0, 1.0);
        let memory_bin = ((mem_norm * (gs - 1.0)).round() as usize).min(grid_size - 1);

        (recall_bin, latency_bin, memory_bin)
    }

    /// Legacy 2D behavior (for compatibility).
    pub fn behavior(&self, grid_size: usize) -> (usize, usize) {
        let (_, lat_bin, mem_bin) = self.behavior_3d(grid_size);
        (lat_bin, mem_bin)
    }
}

/// Elite entry in the MAP-Elites archive.
#[derive(Debug, Clone)]
pub struct Elite {
    pub genome: IndexGenome,
    pub metrics: IndexMetrics,
    pub fitness: f32,
}

/// Optimization priority for index selection.
#[derive(Debug, Clone)]
pub struct OptimizationPriority {
    /// Weight for recall (0-1)
    pub recall_weight: f32,
    /// Weight for speed (0-1)
    pub speed_weight: f32,
    /// Weight for memory efficiency (0-1)
    pub memory_weight: f32,
    /// Weight for build/ingestion speed (0-1)
    pub build_weight: f32,
}

impl Default for OptimizationPriority {
    fn default() -> Self {
        // Balanced across all 4 dimensions
        Self {
            recall_weight: 0.30,
            speed_weight: 0.30,
            memory_weight: 0.20,
            build_weight: 0.20,
        }
    }
}

impl OptimizationPriority {
    pub fn recall_first() -> Self {
        Self {
            recall_weight: 0.50,
            speed_weight: 0.25,
            memory_weight: 0.15,
            build_weight: 0.10,
        }
    }

    pub fn speed_first() -> Self {
        Self {
            recall_weight: 0.25,
            speed_weight: 0.50,
            memory_weight: 0.10,
            build_weight: 0.15,
        }
    }

    /// Search-optimized priority: maximize query speed, ignore build time.
    /// High recall (50%) with speed focus (40%), minimal build weight (5%).
    pub fn search_optimized() -> Self {
        Self {
            recall_weight: 0.50,
            speed_weight: 0.40,
            memory_weight: 0.05,
            build_weight: 0.05,  // Almost ignore build time
        }
    }

    pub fn memory_first() -> Self {
        Self {
            recall_weight: 0.20,
            speed_weight: 0.20,
            memory_weight: 0.45,
            build_weight: 0.15,
        }
    }

    pub fn fast_ingest() -> Self {
        Self {
            recall_weight: 0.25,
            speed_weight: 0.20,
            memory_weight: 0.10,
            build_weight: 0.45,
        }
    }

    /// Create a custom balanced priority.
    pub fn balanced(recall: f32, speed: f32, memory: f32, build: f32) -> Self {
        let sum = recall + speed + memory + build;
        Self {
            recall_weight: recall / sum,
            speed_weight: speed / sum,
            memory_weight: memory / sum,
            build_weight: build / sum,
        }
    }
}

/// EmergentDB configuration.
#[derive(Debug, Clone)]
pub struct EmergentConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// MAP-Elites grid size per dimension (3D grid = grid_size^3 cells).
    pub grid_size: usize,
    /// Number of evolution generations.
    pub generations: usize,
    /// Population size per generation.
    pub population_size: usize,
    /// Sample size for evaluation.
    pub eval_sample_size: usize,
    /// Number of queries for benchmark.
    pub benchmark_queries: usize,
    /// Optimization priority for index selection.
    pub priority: OptimizationPriority,
    /// Early termination: stop if fitness exceeds this threshold.
    pub early_stop_fitness: f32,
    /// Early termination: minimum generations before stopping.
    pub min_generations: usize,
    /// Use 3D behavior grid (recall × latency × memory).
    pub use_3d_grid: bool,
    /// InsertQD: Priority for insert strategy optimization.
    pub insert_priority: InsertPriority,
    /// InsertQD: Grid size for insert behavior grid.
    pub insert_grid_size: usize,
    /// InsertQD: Whether to run insert strategy evolution.
    pub evolve_inserts: bool,
}

impl Default for EmergentConfig {
    fn default() -> Self {
        Self {
            dim: 1536,
            metric: DistanceMetric::Cosine,
            grid_size: 6,  // 6^3 = 216 cells for 3D, or 6^2 = 36 for 2D
            generations: 12,
            population_size: 8,
            eval_sample_size: 300,  // Reduced for faster ingestion
            benchmark_queries: 30,
            priority: OptimizationPriority::default(),
            early_stop_fitness: 0.85,
            min_generations: 4,
            use_3d_grid: true,
            // InsertQD settings
            insert_priority: InsertPriority::default(),
            insert_grid_size: 4,  // 4x4 = 16 cells for insert strategies
            evolve_inserts: true,
        }
    }
}

impl EmergentConfig {
    /// Fast configuration for quick evolution.
    pub fn fast() -> Self {
        Self {
            grid_size: 4,
            generations: 6,
            population_size: 6,
            eval_sample_size: 200,
            benchmark_queries: 20,
            early_stop_fitness: 0.80,
            min_generations: 2,
            insert_grid_size: 3,  // Smaller grid for faster insert evolution
            evolve_inserts: true,
            ..Default::default()
        }
    }

    /// Thorough configuration for best results.
    pub fn thorough() -> Self {
        Self {
            grid_size: 8,
            generations: 20,
            population_size: 12,
            eval_sample_size: 500,
            benchmark_queries: 50,
            early_stop_fitness: 0.95,
            min_generations: 8,
            insert_grid_size: 5,  // Larger grid for better insert coverage
            evolve_inserts: true,
            ..Default::default()
        }
    }

    /// Speed-focused configuration (minimal insert evolution).
    pub fn speed_only() -> Self {
        Self {
            grid_size: 4,
            generations: 4,
            population_size: 4,
            eval_sample_size: 150,
            benchmark_queries: 15,
            early_stop_fitness: 0.75,
            min_generations: 2,
            evolve_inserts: false,  // Skip insert evolution for speed
            ..Default::default()
        }
    }

    /// Balanced configuration for good coverage with reasonable speed.
    /// Prioritizes archive diversity over pure fitness.
    pub fn balanced() -> Self {
        Self {
            grid_size: 5,           // 5^3 = 125 cells for good coverage
            generations: 10,        // More generations for exploration
            population_size: 10,    // Larger population for diversity
            eval_sample_size: 300,
            benchmark_queries: 30,
            early_stop_fitness: 0.90,  // Higher bar before early stop
            min_generations: 5,        // Run at least 5 generations
            insert_grid_size: 4,       // 4x4 = 16 cells for inserts
            evolve_inserts: true,
            ..Default::default()
        }
    }

    /// Search-first configuration: optimize for query latency.
    /// Ignores build time - use when you want fastest searches.
    pub fn search_first() -> Self {
        Self {
            grid_size: 6,              // 6^3 = 216 cells for finer resolution
            generations: 12,           // More exploration
            population_size: 15,       // Larger population
            eval_sample_size: 400,     // More samples for accuracy
            benchmark_queries: 50,     // More queries for stable measurements
            priority: OptimizationPriority::search_optimized(),
            early_stop_fitness: 0.92,
            min_generations: 6,
            insert_grid_size: 4,
            evolve_inserts: true,
            ..Default::default()
        }
    }
}

/// Dynamic index wrapper.
enum DynamicIndex {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
}

impl DynamicIndex {
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()> {
        match self {
            DynamicIndex::Flat(idx) => idx.insert(id, embedding),
            DynamicIndex::Hnsw(idx) => idx.insert(id, embedding),
            DynamicIndex::Ivf(idx) => idx.insert(id, embedding),
        }
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        match self {
            DynamicIndex::Flat(idx) => idx.search(query, k),
            DynamicIndex::Hnsw(idx) => idx.search(query, k),
            DynamicIndex::Ivf(idx) => idx.search(query, k),
        }
    }

    fn remove(&mut self, id: NodeId) -> Result<bool> {
        match self {
            DynamicIndex::Flat(idx) => idx.remove(id),
            DynamicIndex::Hnsw(idx) => idx.remove(id),
            DynamicIndex::Ivf(idx) => idx.remove(id),
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self {
            DynamicIndex::Flat(idx) => idx.len(),
            DynamicIndex::Hnsw(idx) => idx.len(),
            DynamicIndex::Ivf(idx) => idx.len(),
        }
    }
}

/// 3D MAP-Elites archive.
type Archive3D = Vec<Vec<Vec<Option<Elite>>>>;

/// EmergentIndex - The self-optimizing vector index.
///
/// This implements a dual-QD (Quality-Diversity) system:
/// 1. IndexQD: Evolves index type (HNSW, Flat, IVF) with 3D behavior grid
/// 2. InsertQD: Evolves insert strategy (SIMD variants) with 2D behavior grid
pub struct EmergentIndex {
    config: EmergentConfig,
    /// MAP-Elites archive: 3D grid[recall][latency][memory] = Elite
    archive_3d: RwLock<Archive3D>,
    /// Currently active index.
    active_index: RwLock<Option<DynamicIndex>>,
    /// Active genome.
    active_genome: RwLock<Option<IndexGenome>>,
    /// All vectors (for re-indexing during evolution).
    vectors: RwLock<Vec<(NodeId, Vec<f32>)>>,
    /// Whether evolution has been run.
    evolved: RwLock<bool>,
    /// Best elite found (IndexQD).
    best_elite: RwLock<Option<Elite>>,
    // ========================================================================
    // InsertQD: Quality-Diversity for Insert Strategies
    // ========================================================================
    /// InsertQD archive: 2D grid[batch][parallelism] = InsertElite
    insert_archive_2d: RwLock<InsertArchive2D>,
    /// Best insert elite found.
    best_insert_elite: RwLock<Option<InsertElite>>,
    /// Active insert strategy.
    active_insert_genome: RwLock<Option<InsertGenome>>,
}

impl EmergentIndex {
    /// Create a new EmergentIndex.
    pub fn new(config: EmergentConfig) -> Self {
        let gs = config.grid_size;
        let archive_3d = vec![vec![vec![None; gs]; gs]; gs];

        // Initialize InsertQD archive
        let igs = config.insert_grid_size;
        let insert_archive_2d = vec![vec![None; igs]; igs];

        Self {
            config,
            archive_3d: RwLock::new(archive_3d),
            active_index: RwLock::new(None),
            active_genome: RwLock::new(None),
            vectors: RwLock::new(Vec::new()),
            evolved: RwLock::new(false),
            best_elite: RwLock::new(None),
            // InsertQD
            insert_archive_2d: RwLock::new(insert_archive_2d),
            best_insert_elite: RwLock::new(None),
            active_insert_genome: RwLock::new(None),
        }
    }

    /// Run MAP-Elites evolution to find optimal configuration.
    pub fn evolve(&self) -> Result<Elite> {
        let vectors = self.vectors.read();
        let min_samples = self.config.eval_sample_size.min(vectors.len());
        if min_samples < 50 {
            return Err(VectorError::IndexError(format!(
                "Need at least 50 vectors to evolve, got {}",
                vectors.len()
            )));
        }

        // Sample vectors for evaluation
        let sample: Vec<(NodeId, Vec<f32>)> = {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            vectors
                .choose_multiple(&mut rng, min_samples)
                .cloned()
                .collect()
        };
        drop(vectors);

        // Ground truth from flat index
        let ground_truth = self.compute_ground_truth(&sample)?;

        // Initialize population with heuristic seeds + random
        let mut population: Vec<IndexGenome> = IndexGenome::heuristic_seeds();
        while population.len() < self.config.population_size {
            population.push(IndexGenome::random());
        }

        let mut best_fitness = 0.0_f32;
        let mut stagnant_generations = 0;

        // Evolution loop with early termination
        for generation in 0..self.config.generations {
            // Evaluate population in parallel
            let evaluations: Vec<(IndexGenome, Option<IndexMetrics>)> = population
                .par_iter()
                .map(|genome| {
                    let metrics = self.evaluate_genome(genome, &sample, &ground_truth);
                    (genome.clone(), metrics)
                })
                .collect();

            // Update 3D archive
            let mut archive = self.archive_3d.write();
            let mut generation_best = 0.0_f32;

            for (genome, metrics_opt) in evaluations {
                if let Some(metrics) = metrics_opt {
                    let (r_bin, l_bin, m_bin) = metrics.behavior_3d(self.config.grid_size);
                    let fitness = metrics.fitness(&self.config.priority);
                    generation_best = generation_best.max(fitness);

                    let should_update = match &archive[r_bin][l_bin][m_bin] {
                        None => true,
                        Some(existing) => fitness > existing.fitness,
                    };

                    if should_update {
                        archive[r_bin][l_bin][m_bin] = Some(Elite {
                            genome,
                            metrics,
                            fitness,
                        });
                    }
                }
            }
            drop(archive);

            // Track improvement
            if generation_best > best_fitness + 0.01 {
                best_fitness = generation_best;
                stagnant_generations = 0;
            } else {
                stagnant_generations += 1;
            }

            // Early termination conditions
            if generation >= self.config.min_generations {
                // Stop if we found a good enough solution
                if best_fitness >= self.config.early_stop_fitness {
                    tracing::info!(
                        generation = generation,
                        fitness = %best_fitness,
                        "Early stop: fitness threshold reached"
                    );
                    break;
                }

                // Stop if no improvement for 3 generations
                if stagnant_generations >= 3 {
                    tracing::info!(
                        generation = generation,
                        fitness = %best_fitness,
                        "Early stop: stagnant evolution"
                    );
                    break;
                }
            }

            // Generate new population from archive
            population = self.generate_offspring();
        }

        // Find best elite based on priority
        let best = self.select_best_elite()?;

        *self.best_elite.write() = Some(best.clone());
        *self.evolved.write() = true;

        // Build the winning index with all vectors
        let all_vectors = self.vectors.read();
        self.build_active_index_with_vectors(&best.genome, &all_vectors)?;
        drop(all_vectors);

        // Run InsertQD evolution if enabled (dual-QD system)
        if self.config.evolve_inserts {
            if let Ok(insert_elite) = self.evolve_inserts() {
                tracing::info!(
                    strategy = %insert_elite.genome.strategy,
                    throughput = %insert_elite.metrics.throughput_vps,
                    fitness = %insert_elite.fitness,
                    "InsertQD selected optimal strategy"
                );
            }
        }

        Ok(best)
    }

    fn compute_ground_truth(
        &self,
        sample: &[(NodeId, Vec<f32>)],
    ) -> Result<HashMap<u64, Vec<NodeId>>> {
        let config = super::IndexConfig::flat(self.config.dim, self.config.metric);
        let mut flat = FlatIndex::new(config);

        for (id, vec) in sample {
            flat.insert(*id, Embedding::new(vec.clone()))?;
        }

        let mut ground_truth = HashMap::new();
        let k = 10;

        // Use first N vectors as queries
        let num_queries = self.config.benchmark_queries.min(sample.len());
        for (id, vec) in sample.iter().take(num_queries) {
            let results = flat.search(&Embedding::new(vec.clone()), k)?;
            let ids: Vec<NodeId> = results.into_iter().map(|r| r.id).collect();
            ground_truth.insert(id.0, ids);
        }

        Ok(ground_truth)
    }

    fn evaluate_genome(
        &self,
        genome: &IndexGenome,
        sample: &[(NodeId, Vec<f32>)],
        ground_truth: &HashMap<u64, Vec<NodeId>>,
    ) -> Option<IndexMetrics> {
        // Build index
        let build_start = Instant::now();
        let mut index = self.build_index_from_genome(genome)?;

        for (id, vec) in sample {
            if index.insert(*id, Embedding::new(vec.clone())).is_err() {
                return None;
            }
        }

        // Train if needed (IVF)
        if let DynamicIndex::Ivf(ref ivf) = index {
            if ivf.train().is_err() {
                return None;
            }
        }

        let build_time_ms = build_start.elapsed().as_secs_f32() * 1000.0;

        // Benchmark queries
        let num_queries = ground_truth.len();
        let k = 10;

        let mut total_latency = Duration::ZERO;
        let mut total_recall = 0.0;

        for (query_id, expected) in ground_truth {
            let query_vec: Vec<f32> = sample
                .iter()
                .find(|(id, _)| id.0 == *query_id)
                .map(|(_, v)| v.clone())?;

            let start = Instant::now();
            let results = index.search(&Embedding::new(query_vec), k).ok()?;
            total_latency += start.elapsed();

            // Compute recall
            let result_ids: std::collections::HashSet<_> =
                results.iter().map(|r| r.id).collect();
            let expected_set: std::collections::HashSet<_> = expected.iter().cloned().collect();
            let hits = result_ids.intersection(&expected_set).count();
            total_recall += hits as f32 / k as f32;
        }

        let query_latency_us = total_latency.as_micros() as f32 / num_queries as f32;
        let throughput_qps = 1_000_000.0 / query_latency_us;

        // Estimate memory usage
        let bytes_per_vector = match genome.index_type {
            IndexType::Flat => (self.config.dim * 4) as f32, // f32 per dim
            IndexType::Hnsw => {
                // Vector + neighbors (rough estimate)
                (self.config.dim * 4 + genome.hnsw_m * 8 * 2) as f32
            }
            IndexType::Ivf => {
                // Vector + cluster assignment
                (self.config.dim * 4 + 8) as f32
            }
        };

        Some(IndexMetrics {
            recall_at_10: total_recall / num_queries as f32,
            query_latency_us,
            bytes_per_vector,
            build_time_ms,
            throughput_qps,
        })
    }

    fn build_index_from_genome(&self, genome: &IndexGenome) -> Option<DynamicIndex> {
        match genome.index_type {
            IndexType::Flat => {
                let config = super::IndexConfig::flat(self.config.dim, self.config.metric);
                Some(DynamicIndex::Flat(FlatIndex::new(config)))
            }
            IndexType::Hnsw => {
                let config = IndexConfig {
                    dim: self.config.dim,
                    metric: self.config.metric,
                    m: genome.hnsw_m,
                    ef_construction: genome.hnsw_ef_construction,
                    ef_search: genome.hnsw_ef_search,
                };
                Some(DynamicIndex::Hnsw(HnswIndex::new(config)))
            }
            IndexType::Ivf => {
                let config = IvfConfig {
                    dim: self.config.dim,
                    metric: self.config.metric,
                    num_partitions: genome.ivf_partitions,
                    nprobe: genome.ivf_nprobe,
                    kmeans_iterations: 15,  // Reduced from 20 for faster training
                };
                Some(DynamicIndex::Ivf(IvfIndex::new(config)))
            }
        }
    }

    fn generate_offspring(&self) -> Vec<IndexGenome> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let archive = self.archive_3d.read();
        let elites: Vec<&Elite> = archive
            .iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.as_ref())
            .collect();

        if elites.is_empty() {
            // Fall back to heuristic seeds
            let mut seeds = IndexGenome::heuristic_seeds();
            while seeds.len() < self.config.population_size {
                seeds.push(IndexGenome::random());
            }
            return seeds;
        }

        (0..self.config.population_size)
            .map(|_| {
                let op = rng.r#gen::<u8>() % 10;
                match op {
                    0..=3 => {
                        // Mutation (40%)
                        let parent = elites[rng.r#gen::<usize>() % elites.len()];
                        parent.genome.mutate()
                    }
                    4..=6 => {
                        // Crossover (30%)
                        let p1 = elites[rng.r#gen::<usize>() % elites.len()];
                        let p2 = elites[rng.r#gen::<usize>() % elites.len()];
                        p1.genome.crossover(&p2.genome)
                    }
                    7..=8 => {
                        // Elite reproduction (20%)
                        elites[rng.r#gen::<usize>() % elites.len()].genome.clone()
                    }
                    _ => {
                        // Random exploration (10%)
                        IndexGenome::random()
                    }
                }
            })
            .collect()
    }

    fn select_best_elite(&self) -> Result<Elite> {
        let archive = self.archive_3d.read();
        archive
            .iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.as_ref())
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .cloned()
            .ok_or_else(|| VectorError::IndexError("No elites found".to_string()))
    }

    fn build_active_index_with_vectors(&self, genome: &IndexGenome, vectors: &[(NodeId, Vec<f32>)]) -> Result<()> {
        let mut index = self
            .build_index_from_genome(genome)
            .ok_or_else(|| VectorError::IndexError("Failed to build index".to_string()))?;

        for (id, vec) in vectors.iter() {
            index.insert(*id, Embedding::new(vec.clone()))?;
        }

        // Train IVF if needed
        if let DynamicIndex::Ivf(ref ivf) = index {
            ivf.train()?;
        }

        *self.active_index.write() = Some(index);
        *self.active_genome.write() = Some(genome.clone());

        Ok(())
    }

    #[allow(dead_code)]
    fn build_active_index(&self, genome: &IndexGenome) -> Result<()> {
        let vectors = self.vectors.read();
        self.build_active_index_with_vectors(genome, &vectors)
    }

    /// Get archive coverage (percentage of cells filled).
    pub fn archive_coverage(&self) -> f32 {
        let archive = self.archive_3d.read();
        let filled = archive
            .iter()
            .flatten()
            .flatten()
            .filter(|e| e.is_some())
            .count();
        let total = self.config.grid_size.pow(3);
        filled as f32 / total as f32 * 100.0
    }

    /// Get the best elite found.
    pub fn get_best_elite(&self) -> Option<Elite> {
        self.best_elite.read().clone()
    }

    /// Get all elites from the archive.
    pub fn get_all_elites(&self) -> Vec<Elite> {
        self.archive_3d
            .read()
            .iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.clone())
            .collect()
    }

    /// Check if evolution has been run.
    pub fn is_evolved(&self) -> bool {
        *self.evolved.read()
    }

    /// Get the active index type.
    pub fn active_index_type(&self) -> Option<IndexType> {
        self.active_genome.read().as_ref().map(|g| g.index_type)
    }

    /// Re-evolve with new priority.
    pub fn set_priority(&mut self, priority: OptimizationPriority) -> Result<()> {
        self.config.priority = priority;
        if *self.evolved.read() {
            // Re-select best based on new priority
            let best = self.select_best_elite()?;
            let vectors = self.vectors.read();
            self.build_active_index_with_vectors(&best.genome, &vectors)?;
            *self.best_elite.write() = Some(best);
        }
        Ok(())
    }

    /// Manually set the index type and rebuild with specified configuration.
    /// This allows users to bypass evolution and use a specific index type.
    ///
    /// # Example
    /// ```ignore
    /// let genome = IndexGenome {
    ///     index_type: IndexType::Hnsw,
    ///     hnsw_m: 16,
    ///     hnsw_ef_construction: 200,
    ///     hnsw_ef_search: 50,
    ///     ..Default::default()
    /// };
    /// index.set_index_type(&genome)?;
    /// ```
    pub fn set_index_type(&self, genome: &IndexGenome) -> Result<()> {
        let vectors = self.vectors.read();
        if vectors.is_empty() {
            return Err(VectorError::IndexError("No vectors to index".to_string()));
        }
        self.build_active_index_with_vectors(genome, &vectors)?;
        *self.active_genome.write() = Some(genome.clone());
        Ok(())
    }

    /// Get the current active genome (index configuration).
    pub fn get_active_genome(&self) -> Option<IndexGenome> {
        self.active_genome.read().clone()
    }

    /// Clear the active index and reset to brute-force mode.
    pub fn reset_index(&self) {
        *self.active_index.write() = None;
        *self.active_genome.write() = None;
        *self.evolved.write() = false;
        *self.best_elite.write() = None;
    }

    /// Apply a pre-evolved preset configuration.
    /// This skips evolution and directly applies a battle-tested configuration.
    ///
    /// # Example
    /// ```ignore
    /// use vector_core::{EmergentIndex, EmergentConfig, IndexPreset};
    ///
    /// let mut index = EmergentIndex::new(EmergentConfig::default());
    /// // ... insert vectors ...
    /// index.apply_preset(IndexPreset::HnswBalanced)?;
    /// ```
    pub fn apply_preset(&self, preset: IndexPreset) -> Result<()> {
        let genome = preset.to_genome();
        self.set_index_type(&genome)?;
        tracing::info!(preset = %preset, "Applied preset configuration");
        Ok(())
    }

    /// Get the recommended preset based on current data size.
    /// Priority can be "speed", "accuracy", or "balanced" (default).
    pub fn recommend_preset(&self, priority: &str) -> IndexPreset {
        let num_vectors = self.vectors.read().len();
        IndexPreset::recommend(num_vectors, priority)
    }

    /// Apply the recommended preset for current data.
    /// This automatically selects the best preset based on dataset size and priority.
    pub fn apply_recommended_preset(&self, priority: &str) -> Result<IndexPreset> {
        let preset = self.recommend_preset(priority);
        self.apply_preset(preset)?;
        Ok(preset)
    }

    // ========================================================================
    // PRE-COMPUTED ELITES GRID (Industry-Standard Configurations)
    // ========================================================================

    /// Apply a configuration from the pre-computed elites grid.
    /// This uses industry-standard configurations from OpenSearch, Milvus, etc.
    ///
    /// # Arguments
    /// * `elite` - The pre-computed elite configuration to apply
    ///
    /// # Example
    /// ```ignore
    /// let grid = PrecomputedElitesGrid::new();
    /// let elite = grid.recommend(100_000, "balanced");
    /// index.apply_precomputed_elite(elite)?;
    /// ```
    pub fn apply_precomputed_elite(&self, elite: &PrecomputedElite) -> Result<()> {
        self.set_index_type(&elite.genome)?;
        tracing::info!(
            description = elite.description,
            expected_recall = %elite.expected_recall,
            "Applied pre-computed elite configuration"
        );
        Ok(())
    }

    /// Apply the best configuration from the pre-computed elites grid based on requirements.
    ///
    /// # Arguments
    /// * `min_recall` - Minimum acceptable recall (0.0 - 1.0)
    /// * `max_latency_factor` - Maximum acceptable latency factor (1.0 = flat baseline)
    ///
    /// # Example
    /// ```ignore
    /// // Get 95%+ recall with at most 20% of flat search latency
    /// index.apply_best_precomputed(0.95, 0.2)?;
    /// ```
    pub fn apply_best_precomputed(&self, min_recall: f32, max_latency_factor: f32) -> Result<PrecomputedElite> {
        let num_vectors = self.vectors.read().len();
        let grid = PrecomputedElitesGrid::new();

        let elite = grid
            .select_best(num_vectors, min_recall, max_latency_factor)
            .ok_or_else(|| VectorError::IndexError(
                format!("No configuration found for {} vectors with recall>={} and latency<={}",
                    num_vectors, min_recall, max_latency_factor)
            ))?
            .clone();

        self.apply_precomputed_elite(&elite)?;
        Ok(elite)
    }

    /// Apply the recommended pre-computed configuration based on dataset size and priority.
    /// This is the easiest way to get a good configuration without tuning.
    ///
    /// # Arguments
    /// * `priority` - One of: "speed", "balanced", "accuracy", "max"
    ///
    /// # Example
    /// ```ignore
    /// // Insert vectors first...
    /// index.apply_recommended_from_grid("balanced")?;
    /// ```
    pub fn apply_recommended_from_grid(&self, priority: &str) -> Result<PrecomputedElite> {
        let num_vectors = self.vectors.read().len();
        let grid = PrecomputedElitesGrid::new();

        let elite = grid.recommend(num_vectors, priority).clone();
        self.apply_precomputed_elite(&elite)?;
        Ok(elite)
    }

    /// Get all pre-computed configurations that match the current dataset size.
    /// Useful for benchmarking or letting users choose from available options.
    pub fn get_matching_precomputed(&self) -> Vec<PrecomputedElite> {
        let num_vectors = self.vectors.read().len();
        let grid = PrecomputedElitesGrid::new();

        grid.get_for_scale(num_vectors)
            .into_iter()
            .cloned()
            .collect()
    }

    // ========================================================================
    // InsertQD: Quality-Diversity Evolution for Insert Strategies
    // ========================================================================

    /// Run InsertQD evolution to find optimal insert strategy.
    ///
    /// This is a separate QD system that evolves insert strategies (SIMD variants)
    /// in parallel with the main index evolution. The two systems find their
    /// optimal configurations independently, then combine for the best overall.
    pub fn evolve_inserts(&self) -> Result<InsertElite> {
        let vectors = self.vectors.read();
        if vectors.len() < 100 {
            return Err(VectorError::IndexError(format!(
                "Need at least 100 vectors to evolve inserts, got {}",
                vectors.len()
            )));
        }

        // Sample vectors for benchmarking
        let sample_size = self.config.eval_sample_size.min(vectors.len());
        let sample: Vec<Vec<f32>> = {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            vectors
                .choose_multiple(&mut rng, sample_size)
                .map(|(_, v)| v.clone())
                .collect()
        };
        drop(vectors);

        // Initialize population with heuristic seeds (8 seeds now)
        let mut population: Vec<InsertGenome> = InsertGenome::heuristic_seeds();
        // Add random genomes for diversity
        while population.len() < 12 {
            population.push(InsertGenome::random());
        }

        let mut best_fitness = 0.0_f32;
        let insert_generations = 6;  // More generations for better coverage

        // Evolution loop
        for generation in 0..insert_generations {
            // Evaluate population
            let evaluations: Vec<(InsertGenome, Option<InsertMetrics>)> = population
                .iter()
                .map(|genome| {
                    let metrics = self.evaluate_insert_genome(genome, &sample);
                    (genome.clone(), metrics)
                })
                .collect();

            // Update 2D archive
            let mut archive = self.insert_archive_2d.write();
            let gs = self.config.insert_grid_size;

            for (genome, metrics_opt) in evaluations {
                if let Some(metrics) = metrics_opt {
                    let (batch_bin, para_bin) = metrics.behavior_2d(gs);
                    let fitness = metrics.fitness(&self.config.insert_priority);
                    best_fitness = best_fitness.max(fitness);

                    let should_update = match &archive[batch_bin][para_bin] {
                        None => true,
                        Some(existing) => fitness > existing.fitness,
                    };

                    if should_update {
                        archive[batch_bin][para_bin] = Some(InsertElite {
                            genome,
                            metrics,
                            fitness,
                        });
                    }
                }
            }
            drop(archive);

            // Early termination (only after min generations)
            if best_fitness >= 0.90 && generation >= 3 {
                break;
            }

            // Generate new population
            population = self.generate_insert_offspring();
        }

        // Select best insert elite
        let best = self.select_best_insert_elite()?;
        *self.best_insert_elite.write() = Some(best.clone());
        *self.active_insert_genome.write() = Some(best.genome.clone());

        Ok(best)
    }

    fn evaluate_insert_genome(
        &self,
        genome: &InsertGenome,
        sample: &[Vec<f32>],
    ) -> Option<InsertMetrics> {
        // Benchmark the insert strategy
        let throughput = benchmark_insert_strategy(genome.strategy, sample);

        // Measure memory overhead (approximate)
        let memory_overhead = match genome.strategy {
            InsertStrategy::SimdSequential => 0.0,
            InsertStrategy::SimdBatch => genome.batch_size as f32 * 4.0, // Batch buffer
            InsertStrategy::SimdParallel => genome.batch_size as f32 * 4.0 * 2.0, // Thread overhead
            InsertStrategy::PreNormalized => 0.0,
            InsertStrategy::SimdChunked => 64.0 * 4.0, // Fixed chunk buffer
            InsertStrategy::SimdUnrolled => 4.0 * 4.0, // 4 vectors in flight
            InsertStrategy::SimdInterleaved => genome.batch_size as f32 * 4.0, // Norms buffer
        };

        // CPU efficiency based on parallelism
        let cpu_efficiency = if genome.use_parallelism {
            // Higher efficiency for parallel strategies
            (throughput / 1_000_000.0).min(1.0) * 0.9 + 0.1
        } else {
            // Sequential has lower peak but consistent efficiency
            (throughput / 500_000.0).min(1.0) * 0.7 + 0.2
        };

        // Batch scaling factor
        let base_throughput = benchmark_insert_strategy(InsertStrategy::SimdSequential, sample);
        let batch_scaling = (throughput / base_throughput).clamp(0.5, 2.0) / 2.0;

        Some(InsertMetrics {
            throughput_vps: throughput,
            memory_overhead_bytes: memory_overhead,
            cpu_efficiency,
            batch_scaling_factor: batch_scaling,
        })
    }

    fn generate_insert_offspring(&self) -> Vec<InsertGenome> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let archive = self.insert_archive_2d.read();
        let elites: Vec<&InsertElite> = archive
            .iter()
            .flatten()
            .filter_map(|e| e.as_ref())
            .collect();

        if elites.is_empty() {
            let mut seeds = InsertGenome::heuristic_seeds();
            while seeds.len() < 12 {
                seeds.push(InsertGenome::random());
            }
            return seeds;
        }

        (0..12)  // Larger population for better diversity
            .map(|_| {
                let op = rng.r#gen::<u8>() % 10;
                match op {
                    0..=4 => {
                        // Mutation (50%)
                        let parent = elites[rng.r#gen::<usize>() % elites.len()];
                        parent.genome.mutate()
                    }
                    5..=7 => {
                        // Crossover (30%)
                        let p1 = elites[rng.r#gen::<usize>() % elites.len()];
                        let p2 = elites[rng.r#gen::<usize>() % elites.len()];
                        p1.genome.crossover(&p2.genome)
                    }
                    _ => {
                        // Random (20%)
                        InsertGenome::random()
                    }
                }
            })
            .collect()
    }

    fn select_best_insert_elite(&self) -> Result<InsertElite> {
        let archive = self.insert_archive_2d.read();
        archive
            .iter()
            .flatten()
            .filter_map(|e| e.as_ref())
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .cloned()
            .ok_or_else(|| VectorError::IndexError("No insert elites found".to_string()))
    }

    /// Get InsertQD archive coverage (percentage of cells filled).
    pub fn insert_archive_coverage(&self) -> f32 {
        let archive = self.insert_archive_2d.read();
        let filled = archive
            .iter()
            .flatten()
            .filter(|e| e.is_some())
            .count();
        let total = self.config.insert_grid_size.pow(2);
        filled as f32 / total as f32 * 100.0
    }

    /// Get the best insert elite found.
    pub fn get_best_insert_elite(&self) -> Option<InsertElite> {
        self.best_insert_elite.read().clone()
    }

    /// Get all insert elites from the archive.
    pub fn get_all_insert_elites(&self) -> Vec<InsertElite> {
        self.insert_archive_2d
            .read()
            .iter()
            .flatten()
            .filter_map(|e| e.clone())
            .collect()
    }

    /// Get the active insert strategy.
    pub fn active_insert_strategy(&self) -> Option<InsertStrategy> {
        self.active_insert_genome.read().as_ref().map(|g| g.strategy)
    }
}

impl VectorIndex for EmergentIndex {
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()> {
        if embedding.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: embedding.dim(),
            });
        }

        let vec = embedding.as_slice().to_vec();
        self.vectors.write().push((id, vec.clone()));

        // If we have an active index, insert there too
        if let Some(ref mut index) = *self.active_index.write() {
            index.insert(id, embedding)?;
        }

        Ok(())
    }

    fn remove(&mut self, id: NodeId) -> Result<bool> {
        let mut vectors = self.vectors.write();
        let len_before = vectors.len();
        vectors.retain(|(vid, _)| *vid != id);
        let removed_from_vectors = vectors.len() < len_before;
        drop(vectors);

        // Remove from active index if exists
        if let Some(ref mut index) = *self.active_index.write() {
            index.remove(id)?;
        }

        Ok(removed_from_vectors)
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        if query.dim() != self.config.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dim,
                got: query.dim(),
            });
        }

        // Use active index if available
        if let Some(ref index) = *self.active_index.read() {
            return index.search(query, k);
        }

        // Fallback to brute force
        let vectors = self.vectors.read();
        let query_slice = query.as_slice();

        use crate::distance::{cosine_distance_simd, euclidean_distance_simd, negative_dot_product_simd};

        let distance_fn: fn(&[f32], &[f32]) -> f32 = match self.config.metric {
            DistanceMetric::Cosine => cosine_distance_simd,
            DistanceMetric::Euclidean => euclidean_distance_simd,
            DistanceMetric::DotProduct => negative_dot_product_simd,
        };

        let mut results: Vec<SearchResult> = vectors
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

        Ok(results)
    }

    fn get(&self, _id: NodeId) -> Option<&Embedding> {
        None // Can't return reference to owned data
    }

    fn len(&self) -> usize {
        self.vectors.read().len()
    }

    fn metric(&self) -> DistanceMetric {
        self.config.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_mutation() {
        let genome = IndexGenome::default();
        let mutated = genome.mutate();
        assert!(matches!(
            mutated.index_type,
            IndexType::Flat | IndexType::Hnsw | IndexType::Ivf
        ));
    }

    #[test]
    fn test_genome_crossover() {
        let g1 = IndexGenome {
            index_type: IndexType::Hnsw,
            hnsw_m: 16,
            ..Default::default()
        };
        let g2 = IndexGenome {
            index_type: IndexType::Ivf,
            hnsw_m: 32,
            ..Default::default()
        };

        let child = g1.crossover(&g2);
        assert!(child.hnsw_m == 16 || child.hnsw_m == 32);
    }

    #[test]
    fn test_heuristic_seeds() {
        let seeds = IndexGenome::heuristic_seeds();
        assert!(seeds.len() >= 4);

        // Should have at least one of each type
        assert!(seeds.iter().any(|g| g.index_type == IndexType::Flat));
        assert!(seeds.iter().any(|g| g.index_type == IndexType::Hnsw));
        assert!(seeds.iter().any(|g| g.index_type == IndexType::Ivf));
    }

    #[test]
    fn test_metrics_behavior_3d() {
        let metrics = IndexMetrics {
            recall_at_10: 0.95,
            query_latency_us: 500.0,
            bytes_per_vector: 3500.0,
            build_time_ms: 100.0,
            throughput_qps: 2000.0,
        };

        let (r_bin, l_bin, m_bin) = metrics.behavior_3d(6);
        assert!(r_bin < 6);
        assert!(l_bin < 6);
        assert!(m_bin < 6);
    }

    #[test]
    fn test_balanced_fitness() {
        let metrics = IndexMetrics {
            recall_at_10: 0.95,
            query_latency_us: 100.0,
            bytes_per_vector: 3200.0,
            build_time_ms: 50.0,
            throughput_qps: 10000.0,
        };

        let priority = OptimizationPriority::default();
        let fitness = metrics.fitness(&priority);

        // Should be reasonably high given good metrics
        assert!(fitness > 0.5);
        assert!(fitness <= 1.0);
    }

    #[test]
    fn test_optimization_priorities() {
        let default = OptimizationPriority::default();
        let recall = OptimizationPriority::recall_first();
        let speed = OptimizationPriority::speed_first();
        let memory = OptimizationPriority::memory_first();
        let fast_ingest = OptimizationPriority::fast_ingest();

        assert!(recall.recall_weight > default.recall_weight);
        assert!(speed.speed_weight > default.speed_weight);
        assert!(memory.memory_weight > default.memory_weight);
        assert!(fast_ingest.build_weight > default.build_weight);
    }

    #[test]
    fn test_emergent_insert_search_before_evolve() {
        let config = EmergentConfig {
            dim: 8,
            ..Default::default()
        };
        let mut index = EmergentIndex::new(config);

        // Insert vectors
        for i in 0..100 {
            let vec: Vec<f32> = (0..8).map(|j| (i * j) as f32 % 10.0).collect();
            index.insert(NodeId::new(i), Embedding::new(vec)).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search should work (brute force fallback)
        let query = Embedding::new(vec![1.0; 8]);
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_config_presets() {
        let fast = EmergentConfig::fast();
        let thorough = EmergentConfig::thorough();
        let default = EmergentConfig::default();

        assert!(fast.generations < default.generations);
        assert!(thorough.generations > default.generations);
        assert!(fast.eval_sample_size < thorough.eval_sample_size);
    }
}
