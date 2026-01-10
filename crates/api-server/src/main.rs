//! EmergentDB API Server
//!
//! A self-optimizing vector database server with LLM tool integration.

use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use api_server::{AppState, create_router_with_middleware};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "api_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Get configuration from environment
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    let dim: usize = std::env::var("VECTOR_DIM")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(1536); // Default to OpenAI embedding dimension

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!("Starting EmergentDB API server on {}", addr);
    tracing::info!("Vector dimension: {}", dim);

    // Create shared state with vector index
    let state = Arc::new(AppState::new(dim));

    // Create router with middleware
    let app = create_router_with_middleware(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    tracing::info!("Server listening on http://{}", addr);

    axum::serve(listener, app).await.unwrap();
}
