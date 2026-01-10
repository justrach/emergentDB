//! EmergentDB API Server
//!
//! A self-optimizing vector database server with LLM tool integration.
//!
//! ## Environment Variables
//!
//! - `PORT`: Server port (default: 3000)
//! - `VECTOR_DIM`: Vector dimension (default: 768 for Gemini embeddings)
//! - `DATA_DIR`: Path to data directory for persistence (optional, in-memory if not set)

use std::net::SocketAddr;
use std::path::PathBuf;
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
        .unwrap_or(768); // Default to Gemini embedding dimension

    let data_dir: Option<PathBuf> = std::env::var("DATA_DIR")
        .ok()
        .map(PathBuf::from);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!("Starting EmergentDB API server on {}", addr);
    tracing::info!("Vector dimension: {}", dim);

    // Create shared state - with or without persistence
    let state = if let Some(ref dir) = data_dir {
        tracing::info!("Persistence enabled: {:?}", dir);
        match AppState::with_persistence(dim, dir.clone()) {
            Ok(state) => Arc::new(state),
            Err(e) => {
                tracing::error!("Failed to initialize persistent storage: {}", e);
                tracing::warn!("Falling back to in-memory mode");
                Arc::new(AppState::new(dim))
            }
        }
    } else {
        tracing::info!("Running in-memory mode (no persistence)");
        tracing::info!("Set DATA_DIR environment variable to enable persistence");
        Arc::new(AppState::new(dim))
    };

    // Create router with middleware
    let app = create_router_with_middleware(state);

    // Start server with graceful shutdown
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    tracing::info!("Server listening on http://{}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, shutting down...");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM, shutting down...");
        }
    }
}
