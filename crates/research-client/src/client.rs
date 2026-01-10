//! Deep Research API client.

use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::tier::ResearchTier;
use crate::types::{ResearchRequest, ResearchResponse, Source};

/// Error types for the research client.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },

    #[error("Rate limited, retry after {retry_after}s")]
    RateLimited { retry_after: u64 },

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Timeout")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, ClientError>;

/// Configuration for the research client.
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    /// API base URL.
    pub base_url: String,
    /// API key.
    pub api_key: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retries.
    pub max_retries: usize,
    /// Default research tier.
    pub default_tier: ResearchTier,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.example.com/v1".to_string(),
            api_key: String::new(),
            timeout: Duration::from_secs(120),
            max_retries: 3,
            default_tier: ResearchTier::Default,
        }
    }
}

impl ResearchConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        Self {
            api_key: std::env::var("RESEARCH_API_KEY").unwrap_or_default(),
            base_url: std::env::var("RESEARCH_BASE_URL")
                .unwrap_or_else(|_| "https://api.example.com/v1".to_string()),
            ..Default::default()
        }
    }
}

/// Deep Research API client.
pub struct ResearchClient {
    config: ResearchConfig,
    http: Client,
    /// Total cost tracked for this client instance.
    total_cost: std::sync::atomic::AtomicU64,
}

impl ResearchClient {
    /// Create a new client with the given configuration.
    pub fn new(config: ResearchConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(config.timeout)
            .build()?;

        Ok(Self {
            config,
            http,
            total_cost: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self> {
        Self::new(ResearchConfig::from_env())
    }

    /// Perform a research query.
    pub async fn research(&self, request: ResearchRequest) -> Result<ResearchResponse> {
        let tier = request.tier.unwrap_or(self.config.default_tier);
        let url = format!("{}/research", self.config.base_url);

        let body = serde_json::json!({
            "query": request.query,
            "tier": tier.as_str(),
            "max_sources": request.max_sources.unwrap_or(10),
            "include_citations": request.include_citations.unwrap_or(true),
        });

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2u64.pow(attempt as u32));
                tokio::time::sleep(delay).await;
            }

            match self.send_request(&url, &body).await {
                Ok(response) => {
                    // Track cost
                    let cost_cents = (tier.cost_per_query() * 100.0) as u64;
                    self.total_cost.fetch_add(cost_cents, std::sync::atomic::Ordering::Relaxed);

                    return Ok(response);
                }
                Err(ClientError::RateLimited { retry_after }) => {
                    warn!("Rate limited, waiting {}s", retry_after);
                    tokio::time::sleep(Duration::from_secs(retry_after)).await;
                    last_error = Some(ClientError::RateLimited { retry_after });
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or(ClientError::Timeout))
    }

    async fn send_request(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<ResearchResponse> {
        let response = self
            .http
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await?;

        let status = response.status();

        match status {
            StatusCode::OK => {
                let data: ResearchResponse = response.json().await?;
                Ok(data)
            }
            StatusCode::UNAUTHORIZED => Err(ClientError::InvalidApiKey),
            StatusCode::TOO_MANY_REQUESTS => {
                let retry_after = response
                    .headers()
                    .get("Retry-After")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60);
                Err(ClientError::RateLimited { retry_after })
            }
            _ => {
                let message = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                Err(ClientError::Api {
                    status: status.as_u16(),
                    message,
                })
            }
        }
    }

    /// Get total cost incurred (in dollars).
    pub fn total_cost(&self) -> f64 {
        self.total_cost.load(std::sync::atomic::Ordering::Relaxed) as f64 / 100.0
    }

    /// Perform a batch of research queries in parallel.
    pub async fn research_batch(&self, requests: Vec<ResearchRequest>) -> Vec<Result<ResearchResponse>> {
        let futures: Vec<_> = requests.into_iter().map(|r| self.research(r)).collect();
        futures::future::join_all(futures).await
    }

    /// Simple query with default tier.
    pub async fn query(&self, query: &str) -> Result<ResearchResponse> {
        self.research(ResearchRequest {
            query: query.to_string(),
            tier: Some(self.config.default_tier),
            max_sources: None,
            include_citations: None,
        })
        .await
    }

    /// Quick basic-tier query ($0.04).
    pub async fn query_basic(&self, query: &str) -> Result<ResearchResponse> {
        self.research(ResearchRequest {
            query: query.to_string(),
            tier: Some(ResearchTier::Basic),
            max_sources: Some(5),
            include_citations: Some(false),
        })
        .await
    }

    /// Deep research query ($1.30).
    pub async fn query_deep(&self, query: &str) -> Result<ResearchResponse> {
        self.research(ResearchRequest {
            query: query.to_string(),
            tier: Some(ResearchTier::Deeper),
            max_sources: Some(20),
            include_citations: Some(true),
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ResearchConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.default_tier, ResearchTier::Default);
    }
}
