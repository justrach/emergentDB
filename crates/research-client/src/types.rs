//! Types for the Deep Research API.

use serde::{Deserialize, Serialize};

use crate::tier::ResearchTier;

/// A research request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchRequest {
    /// The research query.
    pub query: String,
    /// Research tier (affects depth and cost).
    pub tier: Option<ResearchTier>,
    /// Maximum number of sources to return.
    pub max_sources: Option<usize>,
    /// Whether to include inline citations.
    pub include_citations: Option<bool>,
}

impl ResearchRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            tier: None,
            max_sources: None,
            include_citations: None,
        }
    }

    pub fn with_tier(mut self, tier: ResearchTier) -> Self {
        self.tier = Some(tier);
        self
    }

    pub fn with_max_sources(mut self, max: usize) -> Self {
        self.max_sources = Some(max);
        self
    }
}

/// A research response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResponse {
    /// The synthesized answer.
    pub answer: String,
    /// Sources used in the research.
    pub sources: Vec<Source>,
    /// Research metadata.
    pub metadata: ResearchMetadata,
}

/// A source used in research.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Source URL.
    pub url: String,
    /// Source title.
    pub title: String,
    /// Relevant snippet from the source.
    pub snippet: Option<String>,
    /// Publication date if known.
    pub date: Option<String>,
    /// Relevance score [0, 1].
    pub relevance: Option<f32>,
}

impl Source {
    pub fn new(url: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            title: title.into(),
            snippet: None,
            date: None,
            relevance: None,
        }
    }
}

/// Metadata about a research response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchMetadata {
    /// Time taken in milliseconds.
    pub latency_ms: u64,
    /// Number of sources consulted.
    pub sources_consulted: usize,
    /// Tier used for this request.
    pub tier: String,
    /// Estimated cost in dollars.
    pub estimated_cost: f64,
    /// Request ID for tracking.
    pub request_id: Option<String>,
}

/// Citation in the response text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Index into sources array.
    pub source_index: usize,
    /// Start character position in answer.
    pub start: usize,
    /// End character position in answer.
    pub end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder() {
        let req = ResearchRequest::new("test query")
            .with_tier(ResearchTier::Medium)
            .with_max_sources(5);

        assert_eq!(req.query, "test query");
        assert_eq!(req.tier, Some(ResearchTier::Medium));
        assert_eq!(req.max_sources, Some(5));
    }
}
