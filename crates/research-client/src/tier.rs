//! Research tiers with pricing.

use serde::{Deserialize, Serialize};

/// Research tier affecting depth and cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResearchTier {
    /// Basic tier - fast, minimal depth ($0.04)
    Basic,
    /// Medium tier - balanced ($0.07)
    Medium,
    /// Default tier - standard depth ($0.50)
    Default,
    /// Deeper tier - comprehensive ($1.30)
    Deeper,
}

impl ResearchTier {
    /// Cost per query in dollars.
    pub fn cost_per_query(&self) -> f64 {
        match self {
            Self::Basic => 0.04,
            Self::Medium => 0.07,
            Self::Default => 0.50,
            Self::Deeper => 1.30,
        }
    }

    /// API string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Basic => "basic",
            Self::Medium => "medium",
            Self::Default => "default",
            Self::Deeper => "deeper",
        }
    }

    /// Expected latency range in seconds.
    pub fn expected_latency(&self) -> (f64, f64) {
        match self {
            Self::Basic => (1.0, 5.0),
            Self::Medium => (5.0, 15.0),
            Self::Default => (15.0, 45.0),
            Self::Deeper => (45.0, 120.0),
        }
    }

    /// Expected number of sources.
    pub fn expected_sources(&self) -> (usize, usize) {
        match self {
            Self::Basic => (3, 5),
            Self::Medium => (5, 10),
            Self::Default => (10, 15),
            Self::Deeper => (15, 25),
        }
    }

    /// Choose tier based on budget.
    pub fn from_budget(budget: f64) -> Self {
        if budget >= 1.30 {
            Self::Deeper
        } else if budget >= 0.50 {
            Self::Default
        } else if budget >= 0.07 {
            Self::Medium
        } else {
            Self::Basic
        }
    }

    /// Choose tier based on urgency (seconds available).
    pub fn from_time_budget(seconds: f64) -> Self {
        if seconds >= 120.0 {
            Self::Deeper
        } else if seconds >= 45.0 {
            Self::Default
        } else if seconds >= 15.0 {
            Self::Medium
        } else {
            Self::Basic
        }
    }
}

impl Default for ResearchTier {
    fn default() -> Self {
        Self::Default
    }
}

impl std::fmt::Display for ResearchTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_costs() {
        assert!((ResearchTier::Basic.cost_per_query() - 0.04).abs() < 1e-6);
        assert!((ResearchTier::Deeper.cost_per_query() - 1.30).abs() < 1e-6);
    }

    #[test]
    fn test_from_budget() {
        assert_eq!(ResearchTier::from_budget(0.02), ResearchTier::Basic);
        assert_eq!(ResearchTier::from_budget(0.10), ResearchTier::Medium);
        assert_eq!(ResearchTier::from_budget(1.00), ResearchTier::Default);
        assert_eq!(ResearchTier::from_budget(2.00), ResearchTier::Deeper);
    }
}
