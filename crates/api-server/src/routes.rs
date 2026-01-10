//! API routes definition.

use axum::{
    routing::{get, post},
    Router,
};

use crate::handlers::{self, SharedState};

/// Create the API router with shared state.
pub fn create_router(state: SharedState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(handlers::health_check))
        // Vector operations
        .route("/vectors/search", post(handlers::vector_search))
        .route("/vectors/insert", post(handlers::vector_insert))
        .route("/vectors/batch_insert", post(handlers::vector_batch_insert))
        // Graph operations
        .route("/graph/concepts", post(handlers::add_concept))
        .route("/graph/concepts/{id}", get(handlers::get_concept))
        .route("/graph/relationships", post(handlers::add_relationship))
        .route("/graph/traverse", post(handlers::traverse_graph))
        // Research operations
        .route("/research/query", post(handlers::research_query))
        .route("/research/batch", post(handlers::research_batch))
        // QD operations
        .route("/qd/evolve", post(handlers::qd_evolve))
        .route("/qd/diverse", post(handlers::qd_diverse_solutions))
        // Tool call interface
        .route("/tools/call", post(handlers::tool_call))
        .route("/tools/list", get(handlers::list_tools))
        .with_state(state)
}

/// Create router with CORS and logging middleware.
pub fn create_router_with_middleware(state: SharedState) -> Router {
    use tower_http::cors::{Any, CorsLayer};
    use tower_http::trace::TraceLayer;

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(cors)
}
