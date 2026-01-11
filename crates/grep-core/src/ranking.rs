//! Ranking strategies for search results

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingStrategy {
    /// Rank by number of matches
    MatchCount,
    /// Rank by match density (matches per line)
    Density,
    /// Rank by file recency
    Recency,
    /// Balanced ranking (default)
    Balanced,
    /// Prioritize specific file types
    FileTypePriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    pub strategy: RankingStrategy,
    pub match_count_weight: f32,
    pub density_weight: f32,
    pub recency_weight: f32,
    pub extension_boost: f32,
    pub priority_extensions: Vec<String>,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            strategy: RankingStrategy::Balanced,
            match_count_weight: 1.0,
            density_weight: 2.0,
            recency_weight: 0.5,
            extension_boost: 1.5,
            priority_extensions: vec![
                "rs".into(), "py".into(), "ts".into(), "js".into(),
                "go".into(), "java".into(), "cpp".into(), "c".into(),
            ],
        }
    }
}

impl RankingConfig {
    pub fn match_count_only() -> Self {
        Self {
            strategy: RankingStrategy::MatchCount,
            match_count_weight: 1.0,
            density_weight: 0.0,
            recency_weight: 0.0,
            extension_boost: 0.0,
            priority_extensions: vec![],
        }
    }

    pub fn density_focused() -> Self {
        Self {
            strategy: RankingStrategy::Density,
            match_count_weight: 0.5,
            density_weight: 3.0,
            recency_weight: 0.0,
            extension_boost: 0.0,
            priority_extensions: vec![],
        }
    }

    pub fn recency_focused() -> Self {
        Self {
            strategy: RankingStrategy::Recency,
            match_count_weight: 0.5,
            density_weight: 0.5,
            recency_weight: 3.0,
            extension_boost: 0.0,
            priority_extensions: vec![],
        }
    }

    pub fn for_code() -> Self {
        Self {
            strategy: RankingStrategy::FileTypePriority,
            match_count_weight: 1.0,
            density_weight: 2.0,
            recency_weight: 0.5,
            extension_boost: 2.0,
            priority_extensions: vec![
                "rs".into(), "py".into(), "ts".into(), "tsx".into(),
                "js".into(), "jsx".into(), "go".into(), "java".into(),
            ],
        }
    }
}
