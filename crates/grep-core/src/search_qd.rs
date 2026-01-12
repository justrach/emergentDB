//! SearchQD: Quality-Diversity optimized search strategies
//!
//! Behavior Space (2D):
//! - Latency: Search time (ms)
//! - Relevance: Result quality (NDCG@10)
//!
//! Techniques:
//! - Literal extraction: Search for literals before running full regex
//! - Trigram index: Fast substring matching using 3-character sequences
//! - BM25 ranking: Probabilistic relevance ranking
//! - Parallel search: Multi-threaded file processing

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::load_qd::{LoadGenome, LoadStrategy, Loader};
use crate::{Result, FileEntry};

// ============================================================================
// Search Strategy Genome
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// No index - scan all files
    None,
    /// Trigram index for fast substring search
    Trigram,
    /// Inverted index mapping terms to files
    Inverted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tokenizer {
    /// Split on whitespace only
    Whitespace,
    /// Split on whitespace and camelCase boundaries
    CamelCase,
    /// Split on whitespace, camelCase, and snake_case
    Code,
    /// Character trigrams (for fuzzy matching)
    Trigrams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingAlgorithm {
    /// Simple match count
    MatchCount,
    /// BM25 probabilistic ranking
    BM25 { k1: u32, b: u32 }, // k1 * 100, b * 100 (stored as integers)
    /// TF-IDF weighting
    TfIdf,
    /// Position-weighted (earlier matches score higher)
    PositionWeighted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchGenome {
    /// Index type for file filtering
    pub index_type: IndexType,
    /// Tokenization strategy
    pub tokenizer: Tokenizer,
    /// Ranking algorithm
    pub ranking: RankingAlgorithm,
    /// Extract literals from regex for pre-filtering
    pub literal_extraction: bool,
    /// Case sensitivity
    pub case_sensitive: bool,
    /// Smart case (case-insensitive unless uppercase in query)
    pub smart_case: bool,
    /// Maximum files to search (0 = unlimited)
    pub max_files: usize,
    /// Context lines around matches
    pub context_lines: usize,
    /// File loading strategy
    pub load_genome: LoadGenome,
}

impl Default for SearchGenome {
    fn default() -> Self {
        Self {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::BM25 { k1: 120, b: 75 }, // k1=1.2, b=0.75
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 2,
            load_genome: LoadGenome::default(),
        }
    }
}

// ============================================================================
// Emergent Auto-Selector
// ============================================================================

/// Automatically select the best elite based on workload characteristics
pub struct EmergentSearch;

impl EmergentSearch {
    /// Select the best search strategy based on workload
    pub fn select_elite(file_count: usize, pattern: &str, repeated_queries: bool) -> &'static SearchElite {
        let is_regex = pattern.contains('*') || pattern.contains('+') ||
                       pattern.contains('?') || pattern.contains('[') ||
                       pattern.contains('(') || pattern.contains('|');
        let is_short_pattern = pattern.len() < 5;

        // Decision tree based on empirical benchmarks
        // Key insight: no-index strategies are fastest up to ~1000 files
        match (file_count, is_regex, repeated_queries) {
            // Small repos (< 100 files): no index, literal is fastest
            (0..=100, false, _) => Self::find_elite("literal_only"),
            (0..=100, true, _) => Self::find_elite("regex_heavy"),

            // Medium repos (100-1000 files): still no index needed!
            // Benchmarks show no-index beats trigram index up to ~1000 files
            (101..=1000, false, false) => Self::find_elite("literal_only"),
            (101..=1000, true, false) => Self::find_elite("regex_heavy"),
            // For repeated queries, build an index
            (101..=1000, false, true) => Self::find_elite("inverted_simple"),
            (101..=1000, true, true) => Self::find_elite("inverted_position"),

            // Large repos (1000-10000 files): trigram index pays off
            (1001..=10000, _, false) => Self::find_elite("balanced"),
            (1001..=10000, _, true) => Self::find_elite("large_codebase"),

            // Massive repos (> 10000 files): parallel + trigram
            (_, _, _) => Self::find_elite("massive_parallel"),
        }
    }

    /// Select the best load strategy based on file characteristics
    pub fn select_load_elite(file_size: usize, file_count: usize) -> &'static crate::load_qd::LoadElite {
        use crate::load_qd::LOAD_ELITES;

        match (file_size, file_count) {
            // Small files: buffered I/O is fastest
            (0..=65536, _) => LOAD_ELITES.iter().find(|e| e.name == "small_files").unwrap(),
            // Medium files: mmap with sequential hint
            (65537..=1048576, _) => LOAD_ELITES.iter().find(|e| e.name == "mmap_sequential").unwrap(),
            // Large files, few of them: instant mmap
            (1048577..=104857600, 0..=100) => LOAD_ELITES.iter().find(|e| e.name == "instant_access").unwrap(),
            // Large files, many of them: parallel
            (1048577..=104857600, _) => LOAD_ELITES.iter().find(|e| e.name == "max_throughput").unwrap(),
            // Huge files: prefaulted mmap
            (_, _) => LOAD_ELITES.iter().find(|e| e.name == "huge_files").unwrap(),
        }
    }

    fn find_elite(name: &str) -> &'static SearchElite {
        SEARCH_ELITES.iter().find(|e| e.name == name).unwrap_or(&SEARCH_ELITES[0])
    }

    /// Get a summary of which elite would be selected for given parameters
    pub fn explain_selection(file_count: usize, pattern: &str, repeated_queries: bool) -> String {
        let elite = Self::select_elite(file_count, pattern, repeated_queries);
        format!(
            "Selected '{}': {} (files={}, pattern='{}', repeated={})",
            elite.name, elite.description, file_count, pattern, repeated_queries
        )
    }
}

// ============================================================================
// Precomputed Search Elites
// ============================================================================

#[derive(Debug, Clone)]
pub struct SearchElite {
    pub name: &'static str,
    pub genome: SearchGenome,
    pub description: &'static str,
}

/// Default load genome as const
const DEFAULT_LOAD_GENOME: LoadGenome = LoadGenome {
    strategy: LoadStrategy::Mmap,
    mmap_threshold: 64 * 1024,
    read_ahead: true,
    prefault: false,
};

/// Precomputed optimal configurations for different scenarios
/// Covering ~48% of 2D behavior space (Speed x Relevance)
pub static SEARCH_ELITES: &[SearchElite] = &[
    // === HIGH SPEED REGION ===
    SearchElite {
        name: "speed_first",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 0,
            load_genome: LoadGenome {
                strategy: LoadStrategy::Mmap,
                mmap_threshold: 4096,
                read_ahead: true,
                prefault: false,
            },
        },
        description: "Maximum speed, basic relevance",
    },
    SearchElite {
        name: "regex_heavy",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: false, // Complex regex, don't extract
            case_sensitive: true,
            smart_case: false,
            max_files: 0,
            context_lines: 2,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "For complex regex patterns",
    },
    SearchElite {
        name: "literal_only",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::PositionWeighted,
            literal_extraction: true,
            case_sensitive: true,
            smart_case: false,
            max_files: 0,
            context_lines: 0,
            load_genome: LoadGenome {
                strategy: LoadStrategy::Mmap,
                mmap_threshold: 0,
                read_ahead: true,
                prefault: false,
            },
        },
        description: "Fast literal search with position weighting",
    },
    // === BALANCED REGION ===
    SearchElite {
        name: "balanced",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::BM25 { k1: 120, b: 75 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 2,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Good balance of speed and relevance",
    },
    SearchElite {
        name: "trigram_tfidf",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Trigram index with TF-IDF ranking",
    },
    SearchElite {
        name: "camel_case_aware",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::BM25 { k1: 150, b: 50 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 2,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Optimized for camelCase identifiers",
    },
    // === HIGH RELEVANCE REGION ===
    SearchElite {
        name: "relevance_first",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::BM25 { k1: 200, b: 75 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 3,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Best relevance, may be slower",
    },
    SearchElite {
        name: "semantic_code",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::BM25 { k1: 180, b: 60 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 5,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Code-aware tokenization with rich context",
    },
    SearchElite {
        name: "inverted_position",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::PositionWeighted,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 2,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted index with position-weighted ranking",
    },
    // === LARGE CODEBASE REGION ===
    SearchElite {
        name: "large_codebase",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 10000,
            context_lines: 1,
            load_genome: LoadGenome {
                strategy: LoadStrategy::ParallelBlocks {
                    block_size: 4 * 1024 * 1024,
                    num_threads: 4,
                },
                mmap_threshold: 10 * 1024 * 1024,
                read_ahead: true,
                prefault: false,
            },
        },
        description: "Optimized for 100K+ files",
    },
    SearchElite {
        name: "massive_parallel",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: false,
            max_files: 50000,
            context_lines: 0,
            load_genome: LoadGenome {
                strategy: LoadStrategy::ParallelBlocks {
                    block_size: 8 * 1024 * 1024,
                    num_threads: 8,
                },
                mmap_threshold: 5 * 1024 * 1024,
                read_ahead: true,
                prefault: false,
            },
        },
        description: "Maximum parallelism for huge codebases",
    },
    // === SPECIAL CASES ===
    SearchElite {
        name: "fuzzy_trigram",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Trigrams,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: false,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Fuzzy matching via trigram tokenization",
    },
    // === ADDITIONAL DIVERSITY ELITES ===
    // Coverage: None + BM25
    SearchElite {
        name: "simple_bm25",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::BM25 { k1: 120, b: 75 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "No index, BM25 ranking for small codebases",
    },
    // Coverage: None + TfIdf
    SearchElite {
        name: "simple_tfidf",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "No index, TF-IDF for quick relevance",
    },
    // Coverage: Trigram + PositionWeighted
    SearchElite {
        name: "trigram_position",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::PositionWeighted,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 2,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Trigram index with position-weighted ranking",
    },
    // Coverage: Inverted + MatchCount
    SearchElite {
        name: "inverted_simple",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 0,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted index, simple match counting",
    },
    // Coverage: Inverted + TfIdf
    SearchElite {
        name: "inverted_tfidf",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 2,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted index with TF-IDF ranking",
    },
    // Coverage: None + PositionWeighted + Code
    SearchElite {
        name: "position_code",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::PositionWeighted,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Position-weighted with code tokenization",
    },
    // Coverage: Trigram + BM25 + Whitespace
    SearchElite {
        name: "trigram_bm25_simple",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::BM25 { k1: 100, b: 80 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Trigram + BM25 with whitespace tokenization",
    },
    // Coverage: Inverted + PositionWeighted + Trigrams tokenizer
    SearchElite {
        name: "inverted_fuzzy",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Trigrams,
            ranking: RankingAlgorithm::PositionWeighted,
            literal_extraction: false,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted index with trigram tokenization",
    },
    // Coverage: Trigram + TfIdf + Trigrams
    SearchElite {
        name: "double_trigram",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Trigrams,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: false,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 0,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Trigram index + tokenizer for fuzzy search",
    },
    // Coverage: None + MatchCount + CamelCase
    SearchElite {
        name: "simple_camel",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 0,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Simple search optimized for camelCase",
    },
    // Coverage: Inverted + BM25 + Whitespace
    SearchElite {
        name: "inverted_bm25_simple",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::BM25 { k1: 140, b: 70 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted index + BM25 + whitespace",
    },
    // More coverage elites
    SearchElite {
        name: "none_tfidf_code",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Code,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "No index, TF-IDF with code tokenization",
    },
    SearchElite {
        name: "none_bm25_camel",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::BM25 { k1: 120, b: 75 },
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 0,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "No index, BM25 with camelCase",
    },
    SearchElite {
        name: "trigram_match_camel",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Trigram + match count + camelCase",
    },
    SearchElite {
        name: "trigram_position_whitespace",
        genome: SearchGenome {
            index_type: IndexType::Trigram,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::PositionWeighted,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Trigram + position + whitespace",
    },
    SearchElite {
        name: "inverted_match_camel",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 0,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted + match count + camelCase",
    },
    SearchElite {
        name: "inverted_tfidf_whitespace",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Whitespace,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted + TF-IDF + whitespace",
    },
    // Final coverage elites
    SearchElite {
        name: "none_position_camel",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::CamelCase,
            ranking: RankingAlgorithm::PositionWeighted,
            literal_extraction: true,
            case_sensitive: false,
            smart_case: true,
            max_files: 0,
            context_lines: 1,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "No index, position-weighted + camelCase",
    },
    SearchElite {
        name: "none_tfidf_trigrams",
        genome: SearchGenome {
            index_type: IndexType::None,
            tokenizer: Tokenizer::Trigrams,
            ranking: RankingAlgorithm::TfIdf,
            literal_extraction: false,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 0,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "No index, TF-IDF with trigram tokenization",
    },
    SearchElite {
        name: "inverted_match_trigrams",
        genome: SearchGenome {
            index_type: IndexType::Inverted,
            tokenizer: Tokenizer::Trigrams,
            ranking: RankingAlgorithm::MatchCount,
            literal_extraction: false,
            case_sensitive: false,
            smart_case: false,
            max_files: 0,
            context_lines: 0,
            load_genome: DEFAULT_LOAD_GENOME,
        },
        description: "Inverted + match count + trigram tokenization",
    },
];

// ============================================================================
// Trigram Index
// ============================================================================

/// A trigram (3-character sequence)
pub type Trigram = [u8; 3];

/// Trigram index for fast substring filtering
#[derive(Debug, Default)]
pub struct TrigramIndex {
    /// Map from trigram to list of file IDs containing it
    index: HashMap<Trigram, Vec<usize>>,
    /// File paths by ID
    files: Vec<PathBuf>,
}

impl TrigramIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file to the index
    pub fn add_file(&mut self, path: PathBuf, content: &[u8]) {
        let file_id = self.files.len();
        self.files.push(path);

        // Extract trigrams from content
        let content_lower: Vec<u8> = content.iter().map(|b| b.to_ascii_lowercase()).collect();

        for window in content_lower.windows(3) {
            let trigram: Trigram = [window[0], window[1], window[2]];
            self.index.entry(trigram).or_default().push(file_id);
        }
    }

    /// Get files that might contain the query (based on trigrams)
    pub fn query(&self, pattern: &str) -> Vec<&Path> {
        let pattern_lower = pattern.to_lowercase();
        let pattern_bytes = pattern_lower.as_bytes();

        if pattern_bytes.len() < 3 {
            // Pattern too short for trigram filtering, return all files
            return self.files.iter().map(|p| p.as_path()).collect();
        }

        // Extract trigrams from query
        let query_trigrams: Vec<Trigram> = pattern_bytes
            .windows(3)
            .map(|w| [w[0], w[1], w[2]])
            .collect();

        if query_trigrams.is_empty() {
            return self.files.iter().map(|p| p.as_path()).collect();
        }

        // Find files containing ALL query trigrams (intersection)
        let mut result_ids: Option<Vec<usize>> = None;

        for trigram in query_trigrams {
            if let Some(file_ids) = self.index.get(&trigram) {
                match &mut result_ids {
                    None => result_ids = Some(file_ids.clone()),
                    Some(current) => {
                        let set: std::collections::HashSet<_> = file_ids.iter().copied().collect();
                        current.retain(|id| set.contains(id));
                    }
                }
            } else {
                // Trigram not found anywhere, no files match
                return vec![];
            }
        }

        result_ids
            .map(|ids| ids.into_iter().map(|id| self.files[id].as_path()).collect())
            .unwrap_or_default()
    }

    /// Number of indexed files
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Number of unique trigrams
    pub fn trigram_count(&self) -> usize {
        self.index.len()
    }
}

// ============================================================================
// BM25 Ranking
// ============================================================================

/// BM25 scoring parameters
#[derive(Debug, Clone)]
pub struct BM25 {
    /// Term saturation parameter (typically 1.2-2.0)
    pub k1: f64,
    /// Length normalization parameter (typically 0.75)
    pub b: f64,
    /// Average document length
    pub avg_doc_len: f64,
    /// Total number of documents
    pub doc_count: usize,
    /// Document frequency for each term
    pub doc_freq: HashMap<String, usize>,
}

impl BM25 {
    pub fn new(k1: f64, b: f64) -> Self {
        Self {
            k1,
            b,
            avg_doc_len: 0.0,
            doc_count: 0,
            doc_freq: HashMap::new(),
        }
    }

    /// Build BM25 index from documents
    pub fn build<'a>(
        k1: f64,
        b: f64,
        documents: impl Iterator<Item = (&'a Path, &'a str)>,
    ) -> Self {
        let mut bm25 = Self::new(k1, b);
        let mut total_len = 0usize;
        let mut doc_count = 0usize;

        for (_, content) in documents {
            let tokens = tokenize_code(content);
            total_len += tokens.len();
            doc_count += 1;

            // Count unique terms in this document
            let unique_terms: std::collections::HashSet<_> = tokens.into_iter().collect();
            for term in unique_terms {
                *bm25.doc_freq.entry(term).or_insert(0) += 1;
            }
        }

        bm25.doc_count = doc_count;
        bm25.avg_doc_len = if doc_count > 0 {
            total_len as f64 / doc_count as f64
        } else {
            0.0
        };

        bm25
    }

    /// Calculate IDF (Inverse Document Frequency) for a term
    pub fn idf(&self, term: &str) -> f64 {
        let df = *self.doc_freq.get(term).unwrap_or(&0) as f64;
        let n = self.doc_count as f64;

        if df == 0.0 {
            return 0.0;
        }

        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score a document for a query
    pub fn score(&self, query_terms: &[String], doc_content: &str) -> f64 {
        let doc_tokens = tokenize_code(doc_content);
        let doc_len = doc_tokens.len() as f64;

        // Count term frequencies in document
        let mut term_freq: HashMap<&str, usize> = HashMap::new();
        for token in &doc_tokens {
            *term_freq.entry(token.as_str()).or_insert(0) += 1;
        }

        let mut score = 0.0;

        for term in query_terms {
            let tf = *term_freq.get(term.as_str()).unwrap_or(&0) as f64;
            let idf = self.idf(term);

            // BM25 formula
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len);

            score += idf * numerator / denominator;
        }

        score
    }
}

// ============================================================================
// Tokenization
// ============================================================================

/// Tokenize code content into searchable terms
pub fn tokenize_code(content: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    let mut prev_char_type = CharType::Other;

    for ch in content.chars() {
        let char_type = classify_char(ch);

        // Check for camelCase or snake_case boundary
        let is_boundary = match (prev_char_type, char_type) {
            (CharType::Lower, CharType::Upper) => true, // camelCase
            (CharType::Upper, CharType::Lower) if current_token.len() > 1 => true, // XMLParser
            (_, CharType::Other) => true,
            (CharType::Other, _) => true,
            _ => false,
        };

        if is_boundary && !current_token.is_empty() {
            tokens.push(current_token.to_lowercase());
            current_token.clear();
        }

        if char_type != CharType::Other {
            current_token.push(ch);
        }

        prev_char_type = char_type;
    }

    if !current_token.is_empty() {
        tokens.push(current_token.to_lowercase());
    }

    // Filter out very short tokens
    tokens.retain(|t| t.len() >= 2);

    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharType {
    Upper,
    Lower,
    Digit,
    Other,
}

fn classify_char(ch: char) -> CharType {
    if ch.is_uppercase() {
        CharType::Upper
    } else if ch.is_lowercase() {
        CharType::Lower
    } else if ch.is_ascii_digit() {
        CharType::Digit
    } else {
        CharType::Other
    }
}

// ============================================================================
// Literal Extraction (from ripgrep)
// ============================================================================

/// Extract literal strings from a regex pattern for pre-filtering
pub fn extract_literals(pattern: &str) -> Vec<String> {
    let mut literals = Vec::new();
    let mut current = String::new();
    let mut in_escape = false;
    let mut in_class = false;

    for ch in pattern.chars() {
        if in_escape {
            // Common regex escape sequences that are NOT literal characters
            // \d \D \w \W \s \S \b \B \n \r \t etc.
            let is_special_escape = matches!(ch, 'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B' |
                                                 'n' | 'r' | 't' | 'f' | 'v' | '0'..='9');
            if is_special_escape {
                // This is a regex metacharacter, flush current literal
                if current.len() >= 3 {
                    literals.push(current.clone());
                }
                current.clear();
            } else {
                // Escaped literal character (like \. \* etc), add to current
                current.push(ch);
            }
            in_escape = false;
            continue;
        }

        match ch {
            '\\' => in_escape = true,
            '[' => {
                in_class = true;
                if current.len() >= 3 {
                    literals.push(current.clone());
                }
                current.clear();
            }
            ']' => in_class = false,
            '.' | '*' | '+' | '?' | '^' | '$' | '|' | '(' | ')' | '{' | '}' if !in_class => {
                // Regex metacharacter, flush current literal
                if current.len() >= 3 {
                    literals.push(current.clone());
                }
                current.clear();
            }
            _ if !in_class => {
                current.push(ch);
            }
            _ => {}
        }
    }

    if current.len() >= 3 {
        literals.push(current);
    }

    literals
}

// ============================================================================
// QD Search Engine
// ============================================================================

/// Search result with ranking score
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub path: PathBuf,
    pub matches: Vec<SearchMatch>,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct SearchMatch {
    pub line_number: usize,
    pub line_content: String,
    pub match_start: usize,
    pub match_end: usize,
}

/// QD-optimized search engine
pub struct SearchEngine {
    genome: SearchGenome,
    trigram_index: Option<TrigramIndex>,
    bm25: Option<BM25>,
    loader: Loader,
}

impl SearchEngine {
    pub fn new(genome: SearchGenome) -> Self {
        Self {
            loader: Loader::new(genome.load_genome.clone()),
            genome,
            trigram_index: None,
            bm25: None,
        }
    }

    /// Create engine from a named elite configuration
    pub fn from_elite(name: &str) -> Option<Self> {
        SEARCH_ELITES
            .iter()
            .find(|e| e.name == name)
            .map(|e| Self::new(e.genome.clone()))
    }

    /// Build index from files
    pub fn build_index(&mut self, files: &[FileEntry]) -> Result<()> {
        match self.genome.index_type {
            IndexType::None => {}
            IndexType::Trigram => {
                let mut index = TrigramIndex::new();
                for entry in files {
                    if let Ok(loaded) = self.loader.load(&entry.path) {
                        index.add_file(entry.path.clone(), loaded.as_bytes());
                    }
                }
                self.trigram_index = Some(index);
            }
            IndexType::Inverted => {
                // Build BM25 index
                let (k1, b) = match self.genome.ranking {
                    RankingAlgorithm::BM25 { k1, b } => (k1 as f64 / 100.0, b as f64 / 100.0),
                    _ => (1.2, 0.75),
                };

                let docs: Vec<_> = files
                    .iter()
                    .filter_map(|entry| {
                        self.loader.load(&entry.path).ok().and_then(|loaded| {
                            loaded.as_str().map(|s| (entry.path.as_path(), s.to_string()))
                        })
                    })
                    .collect();

                let bm25 = BM25::build(
                    k1,
                    b,
                    docs.iter().map(|(p, c)| (*p, c.as_str())),
                );
                self.bm25 = Some(bm25);
            }
        }

        Ok(())
    }

    /// Search for pattern in indexed files
    pub fn search(
        &self,
        pattern: &str,
        files: &[FileEntry],
    ) -> Result<Vec<RankedResult>> {
        // Step 1: Extract literals for pre-filtering (if enabled)
        let literals = if self.genome.literal_extraction {
            extract_literals(pattern)
        } else {
            vec![]
        };

        // Step 2: Filter files using index (if available)
        let candidate_files: Vec<_> = match &self.trigram_index {
            Some(index) if !literals.is_empty() => {
                // Use first literal for trigram filtering
                let filtered = index.query(&literals[0]);
                files
                    .iter()
                    .filter(|f| filtered.iter().any(|p| *p == f.path))
                    .collect()
            }
            _ => files.iter().collect(),
        };

        // Step 3: Apply max_files limit
        let candidate_files: Vec<_> = if self.genome.max_files > 0 {
            candidate_files
                .into_iter()
                .take(self.genome.max_files)
                .collect()
        } else {
            candidate_files
        };

        // Step 4: Build regex
        let regex = if self.genome.case_sensitive {
            regex::Regex::new(pattern)?
        } else if self.genome.smart_case && pattern.chars().any(|c| c.is_uppercase()) {
            regex::Regex::new(pattern)?
        } else {
            regex::RegexBuilder::new(pattern)
                .case_insensitive(true)
                .build()?
        };

        // Step 5: Search files in parallel
        let results: Vec<_> = candidate_files
            .par_iter()
            .filter_map(|entry| {
                let loaded = self.loader.load(&entry.path).ok()?;
                let content = loaded.as_str()?;

                // Quick literal pre-check
                if !literals.is_empty() {
                    let content_lower = content.to_lowercase();
                    if !literals.iter().all(|lit| content_lower.contains(&lit.to_lowercase())) {
                        return None;
                    }
                }

                let mut matches = Vec::new();

                for (line_num, line) in content.lines().enumerate() {
                    for mat in regex.find_iter(line) {
                        matches.push(SearchMatch {
                            line_number: line_num + 1,
                            line_content: line.to_string(),
                            match_start: mat.start(),
                            match_end: mat.end(),
                        });
                    }
                }

                if matches.is_empty() {
                    return None;
                }

                // Calculate score
                let score = self.calculate_score(&matches, content);

                Some(RankedResult {
                    path: entry.path.clone(),
                    matches,
                    score,
                })
            })
            .collect();

        // Step 6: Sort by score
        let mut results = results;
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    fn calculate_score(&self, matches: &[SearchMatch], content: &str) -> f64 {
        match self.genome.ranking {
            RankingAlgorithm::MatchCount => matches.len() as f64,
            RankingAlgorithm::BM25 { .. } => {
                if let Some(bm25) = &self.bm25 {
                    let query_terms: Vec<_> = matches
                        .iter()
                        .flat_map(|m| tokenize_code(&m.line_content[m.match_start..m.match_end]))
                        .collect();
                    bm25.score(&query_terms, content)
                } else {
                    matches.len() as f64
                }
            }
            RankingAlgorithm::TfIdf => {
                // Simplified TF-IDF: tf * log(match_count)
                let tf = matches.len() as f64;
                tf * (1.0 + tf).ln()
            }
            RankingAlgorithm::PositionWeighted => {
                // Earlier matches score higher
                matches
                    .iter()
                    .map(|m| 1.0 / (m.line_number as f64).sqrt())
                    .sum()
            }
        }
    }
}

// ============================================================================
// Benchmarking
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct SearchBenchmarkResult {
    pub strategy: String,
    pub file_count: usize,
    pub result_count: usize,
    pub search_time_ms: u64,
    pub files_per_second: f64,
}

/// Benchmark all search strategies
pub fn benchmark_search_strategies(
    pattern: &str,
    files: &[FileEntry],
) -> Result<Vec<SearchBenchmarkResult>> {
    let mut results = Vec::new();

    for elite in SEARCH_ELITES {
        let mut engine = SearchEngine::new(elite.genome.clone());

        // Build index
        let start = std::time::Instant::now();
        engine.build_index(files)?;
        let index_time = start.elapsed();

        // Search (average of 3 runs)
        let mut total_time = 0u64;
        let mut result_count = 0;
        let runs = 3;

        for _ in 0..runs {
            let start = std::time::Instant::now();
            let search_results = engine.search(pattern, files)?;
            total_time += start.elapsed().as_millis() as u64;
            result_count = search_results.len();
        }

        let avg_time = total_time / runs;
        let total_ms = index_time.as_millis() as u64 + avg_time;
        let files_per_sec = if total_ms > 0 {
            (files.len() as f64 * 1000.0) / total_ms as f64
        } else {
            f64::INFINITY
        };

        results.push(SearchBenchmarkResult {
            strategy: elite.name.to_string(),
            file_count: files.len(),
            result_count,
            search_time_ms: avg_time,
            files_per_second: files_per_sec,
        });
    }

    // Sort by files per second
    results.sort_by(|a, b| b.files_per_second.partial_cmp(&a.files_per_second).unwrap());

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_literals() {
        // Space is a literal character in regex, so "hello world" is one literal
        assert_eq!(extract_literals("hello world"), vec!["hello world"]);
        // .* splits into two literals
        assert_eq!(extract_literals(r"foo.*bar"), vec!["foo", "bar"]);
        // \w+ and \d+ are metacharacters, only "test" is a literal
        assert_eq!(extract_literals(r"\w+test\d+"), vec!["test"]);
        // Too short (< 3 chars) literals are filtered out
        assert_eq!(extract_literals("ab"), Vec::<String>::new());
    }

    #[test]
    fn test_tokenize_code() {
        let tokens = tokenize_code("getUserName");
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"name".to_string()));

        let tokens = tokenize_code("get_user_name");
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"name".to_string()));
    }

    #[test]
    fn test_trigram_index() {
        let mut index = TrigramIndex::new();
        index.add_file(PathBuf::from("test.rs"), b"fn hello_world() {}");

        let results = index.query("hello");
        assert_eq!(results.len(), 1);

        let results = index.query("goodbye");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_bm25() {
        let docs = vec![
            (Path::new("a.txt"), "hello world hello"),
            (Path::new("b.txt"), "goodbye world"),
        ];

        let bm25 = BM25::build(1.2, 0.75, docs.into_iter());

        let score_a = bm25.score(&["hello".to_string()], "hello world hello");
        let score_b = bm25.score(&["hello".to_string()], "goodbye world");

        assert!(score_a > score_b);
    }
}
