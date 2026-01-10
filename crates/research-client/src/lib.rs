//! Research Client - Deep Research API integration.
//!
//! This crate provides:
//! - Research API client with tiered research modes
//! - Rate limiting and retry logic
//! - Response parsing and caching
//! - Cost tracking per research tier

pub mod client;
pub mod types;
pub mod tier;

pub use client::ResearchClient;
pub use types::{ResearchRequest, ResearchResponse, Source};
pub use tier::ResearchTier;
