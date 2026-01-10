//! API Server - Tool call interface for LLM agents.
//!
//! This crate provides:
//! - REST API for vector operations
//! - Tool call schema definitions
//! - Rate limiting and authentication
//! - Streaming responses for long operations

pub mod routes;
pub mod tools;
pub mod handlers;

pub use handlers::{AppState, SharedState};
pub use routes::{create_router, create_router_with_middleware};
