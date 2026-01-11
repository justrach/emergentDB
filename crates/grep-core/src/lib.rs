//! grep-core: Fast text/code search for LLM agents
//!
//! A lightweight, embedding-free search engine for code and text.
//! Designed for AI agents that need to search through codebases quickly.
//!
//! # Features
//! - Regex and literal pattern matching
//! - File type filtering (by extension or gitignore-style globs)
//! - Parallel search across files
//! - Memory-mapped file reading for large codebases
//! - Ranked results by relevance (match count, recency, file importance)

mod index;
mod search;
mod ranking;
mod repo;

pub use index::{CodeIndex, IndexConfig, FileEntry};
pub use search::{SearchQuery, SearchResult, Match, Searcher, search, search_regex};
pub use ranking::{RankingStrategy, RankingConfig};
pub use repo::{RepoExplorer, RepoInfo, FileTree, explore};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GrepError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
    #[error("Path not found: {0}")]
    PathNotFound(String),
    #[error("Index error: {0}")]
    IndexError(String),
}

pub type Result<T> = std::result::Result<T, GrepError>;
