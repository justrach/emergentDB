//! Request handlers for API endpoints.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use vector_core::{
    DistanceMetric, Embedding, NodeId, VectorIndex,
    RocksDbStorage, Storage, StorageConfig,
    EmergentIndex, EmergentConfig, OptimizationPriority,
};

// ============================================================================
// Application State
// ============================================================================

/// Shared application state containing the vector index and optional persistent storage.
pub struct AppState {
    pub index: RwLock<EmergentIndex>,
    pub storage: Option<Arc<RocksDbStorage>>,
    pub dim: usize,
}

impl AppState {
    /// Create a new in-memory only state (no persistence).
    pub fn new(dim: usize) -> Self {
        let config = EmergentConfig {
            dim,
            metric: DistanceMetric::Cosine,
            ..EmergentConfig::fast() // Use fast defaults for interactive use
        };
        Self {
            index: RwLock::new(EmergentIndex::new(config)),
            storage: None,
            dim,
        }
    }

    /// Create a new state with RocksDB persistence.
    /// On startup, loads all vectors from disk into the in-memory index.
    pub fn with_persistence(dim: usize, data_dir: PathBuf) -> Result<Self, String> {
        let config = EmergentConfig {
            dim,
            metric: DistanceMetric::Cosine,
            ..EmergentConfig::fast()
        };
        let mut index = EmergentIndex::new(config);

        // Configure and open RocksDB storage
        let storage_config = StorageConfig {
            data_dir: data_dir.clone(),
            enable_wal: true,
            sync_writes: false, // Async for performance
            ..Default::default()
        };

        let storage = RocksDbStorage::open_with_config(&data_dir, storage_config)
            .map_err(|e| format!("Failed to open storage at {:?}: {}", data_dir, e))?;

        // Recovery: Load all vectors from disk into memory
        let mut loaded = 0;
        for result in storage.iter().map_err(|e| format!("Failed to iterate storage: {}", e))? {
            match result {
                Ok((id, embedding)) => {
                    if let Err(e) = index.insert(id, embedding) {
                        tracing::warn!("Failed to load vector {}: {}", id.0, e);
                    } else {
                        loaded += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("Error reading from storage: {}", e);
                }
            }
        }

        if loaded > 0 {
            tracing::info!("Recovered {} vectors from disk", loaded);
        }

        Ok(Self {
            index: RwLock::new(index),
            storage: Some(Arc::new(storage)),
            dim,
        })
    }

    /// Check if persistence is enabled.
    pub fn has_persistence(&self) -> bool {
        self.storage.is_some()
    }
}

pub type SharedState = Arc<AppState>;

// ============================================================================
// Health Check
// ============================================================================

pub async fn health_check(State(state): State<SharedState>) -> impl IntoResponse {
    let index = state.index.read();
    let mut response = serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "vectors_count": index.len(),
        "dimension": state.dim,
        "persistence_enabled": state.storage.is_some(),
        "evolved": index.is_evolved(),
    });

    // Add evolution info if evolved
    if index.is_evolved() {
        if let Some(elite) = index.get_best_elite() {
            response["active_index"] = serde_json::json!({
                "type": format!("{}", elite.genome.index_type),
                "recall": elite.metrics.recall_at_10,
                "latency_us": elite.metrics.query_latency_us,
                "fitness": elite.fitness,
            });
        }
        response["archive_coverage"] = serde_json::json!(index.archive_coverage());
    }

    // Add storage stats if persistence is enabled
    if let Some(ref storage) = state.storage {
        let stats = storage.stats();
        response["storage"] = serde_json::json!({
            "disk_size_bytes": stats.disk_size_bytes,
            "reads": stats.reads,
            "writes": stats.writes,
        });
    }

    Json(response)
}

// ============================================================================
// Vector Operations
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct VectorSearchRequest {
    pub query: Vec<f32>,
    pub k: usize,
    #[serde(default)]
    pub metric: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VectorSearchResponse {
    pub results: Vec<SearchResult>,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

pub async fn vector_search(
    State(state): State<SharedState>,
    Json(req): Json<VectorSearchRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let index = state.index.read();

    let query = Embedding::new(req.query);
    let results: Vec<SearchResult> = match index.search(&query, req.k) {
        Ok(results) => results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id.0,
                score: r.score,
            })
            .collect(),
        Err(_) => vec![],
    };

    let latency_ms = start.elapsed().as_millis() as u64;

    Json(VectorSearchResponse { results, latency_ms })
}

#[derive(Debug, Deserialize)]
pub struct VectorInsertRequest {
    pub id: u64,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

pub async fn vector_insert(
    State(state): State<SharedState>,
    Json(req): Json<VectorInsertRequest>,
) -> impl IntoResponse {
    let embedding = Embedding::new(req.vector);
    let node_id = NodeId::new(req.id);

    // Insert into in-memory index
    {
        let mut index = state.index.write();
        if let Err(e) = index.insert(node_id, embedding.clone()) {
            return Json(serde_json::json!({
                "success": false,
                "error": e.to_string(),
            }));
        }
    }

    // Persist to disk if storage is enabled
    if let Some(ref storage) = state.storage {
        if let Err(e) = storage.put(node_id, &embedding) {
            tracing::error!("Failed to persist vector {}: {}", req.id, e);
            // Note: We don't fail the request since it's in memory
            // The data will be lost on restart but the current operation succeeds
        }
    }

    Json(serde_json::json!({
        "success": true,
        "id": req.id,
        "persisted": state.storage.is_some(),
    }))
}

#[derive(Debug, Deserialize)]
pub struct VectorBatchInsertRequest {
    pub vectors: Vec<VectorInsertRequest>,
}

pub async fn vector_batch_insert(
    State(state): State<SharedState>,
    Json(req): Json<VectorBatchInsertRequest>,
) -> impl IntoResponse {
    let mut inserted = 0;
    let mut to_persist: Vec<(NodeId, Embedding)> = Vec::with_capacity(req.vectors.len());

    // Insert into in-memory index
    {
        let mut index = state.index.write();
        for v in &req.vectors {
            let embedding = Embedding::new(v.vector.clone());
            let node_id = NodeId::new(v.id);
            if index.insert(node_id, embedding.clone()).is_ok() {
                inserted += 1;
                to_persist.push((node_id, embedding));
            }
        }
    }

    // Batch persist to disk if storage is enabled
    if let Some(ref storage) = state.storage {
        if !to_persist.is_empty() {
            if let Err(e) = storage.put_batch(&to_persist) {
                tracing::error!("Failed to persist batch: {}", e);
            }
        }
    }

    Json(serde_json::json!({
        "success": true,
        "count": inserted,
        "persisted": state.storage.is_some(),
    }))
}

// ============================================================================
// Graph Operations (placeholders for now)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AddConceptRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,
}

pub async fn add_concept(Json(req): Json<AddConceptRequest>) -> impl IntoResponse {
    Json(serde_json::json!({
        "success": true,
        "id": uuid::Uuid::new_v4().to_string(),
        "name": req.name,
    }))
}

pub async fn get_concept(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({
        "id": id,
        "name": "placeholder",
        "found": false,
    }))
}

#[derive(Debug, Deserialize)]
pub struct AddRelationshipRequest {
    pub source_id: String,
    pub target_id: String,
    pub relation_type: String,
    #[serde(default)]
    pub weight: Option<f32>,
}

pub async fn add_relationship(Json(req): Json<AddRelationshipRequest>) -> impl IntoResponse {
    Json(serde_json::json!({
        "success": true,
        "source": req.source_id,
        "target": req.target_id,
        "type": req.relation_type,
    }))
}

#[derive(Debug, Deserialize)]
pub struct TraverseRequest {
    pub start_id: String,
    #[serde(default)]
    pub max_depth: Option<usize>,
    #[serde(default)]
    pub relation_types: Option<Vec<String>>,
}

pub async fn traverse_graph(Json(_req): Json<TraverseRequest>) -> impl IntoResponse {
    Json(serde_json::json!({
        "nodes": [],
        "edges": [],
    }))
}

// ============================================================================
// Research Operations (placeholders)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ResearchQueryRequest {
    pub query: String,
    #[serde(default)]
    pub tier: Option<String>,
    #[serde(default)]
    pub max_sources: Option<usize>,
}

pub async fn research_query(Json(req): Json<ResearchQueryRequest>) -> impl IntoResponse {
    Json(serde_json::json!({
        "answer": "Research functionality requires RESEARCH_API_KEY to be configured",
        "sources": [],
        "tier": req.tier.unwrap_or_else(|| "default".to_string()),
    }))
}

#[derive(Debug, Deserialize)]
pub struct ResearchBatchRequest {
    pub queries: Vec<String>,
    #[serde(default)]
    pub tier: Option<String>,
}

pub async fn research_batch(Json(req): Json<ResearchBatchRequest>) -> impl IntoResponse {
    Json(serde_json::json!({
        "results": req.queries.iter().map(|q| {
            serde_json::json!({
                "query": q,
                "answer": "Research functionality requires RESEARCH_API_KEY",
                "sources": [],
            })
        }).collect::<Vec<_>>(),
    }))
}

// ============================================================================
// QD Operations - MAP-Elites Evolution
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct QdEvolveRequest {
    /// Optimization priority: "recall", "speed", "memory", "balanced"
    #[serde(default)]
    pub priority: Option<String>,
    /// Whether to use thorough evolution (slower but better)
    #[serde(default)]
    pub thorough: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct QdEvolveResponse {
    pub success: bool,
    pub best_index_type: String,
    pub recall: f32,
    pub latency_us: f32,
    pub fitness: f32,
    pub archive_coverage: f32,
    pub evolution_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub async fn qd_evolve(
    State(state): State<SharedState>,
    Json(req): Json<QdEvolveRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    // Check if we have enough vectors
    {
        let index = state.index.read();
        if index.len() < 50 {
            return Json(QdEvolveResponse {
                success: false,
                best_index_type: "none".to_string(),
                recall: 0.0,
                latency_us: 0.0,
                fitness: 0.0,
                archive_coverage: 0.0,
                evolution_time_ms: 0,
                error: Some(format!("Need at least 50 vectors to evolve, got {}", index.len())),
            });
        }
    }

    // Run evolution
    let result = {
        let index = state.index.read();
        index.evolve()
    };

    match result {
        Ok(elite) => {
            let index = state.index.read();
            Json(QdEvolveResponse {
                success: true,
                best_index_type: format!("{}", elite.genome.index_type),
                recall: elite.metrics.recall_at_10,
                latency_us: elite.metrics.query_latency_us,
                fitness: elite.fitness,
                archive_coverage: index.archive_coverage(),
                evolution_time_ms: start.elapsed().as_millis() as u64,
                error: None,
            })
        }
        Err(e) => {
            Json(QdEvolveResponse {
                success: false,
                best_index_type: "none".to_string(),
                recall: 0.0,
                latency_us: 0.0,
                fitness: 0.0,
                archive_coverage: 0.0,
                evolution_time_ms: start.elapsed().as_millis() as u64,
                error: Some(e.to_string()),
            })
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct QdDiverseRequest {
    pub n: usize,
}

pub async fn qd_diverse_solutions(Json(req): Json<QdDiverseRequest>) -> impl IntoResponse {
    Json(serde_json::json!({
        "solutions": [],
        "n": req.n,
    }))
}

// ============================================================================
// Index Configuration - Manual index type selection
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct IndexConfigureRequest {
    /// Index type: "flat", "hnsw", "ivf"
    pub index_type: String,
    /// HNSW: neighbors per node (default: 16)
    #[serde(default)]
    pub hnsw_m: Option<usize>,
    /// HNSW: ef_construction (default: 200)
    #[serde(default)]
    pub hnsw_ef_construction: Option<usize>,
    /// HNSW: ef_search (default: 50)
    #[serde(default)]
    pub hnsw_ef_search: Option<usize>,
    /// IVF: number of partitions (default: 256)
    #[serde(default)]
    pub ivf_partitions: Option<usize>,
    /// IVF: partitions to search (default: 16)
    #[serde(default)]
    pub ivf_nprobe: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct IndexConfigureResponse {
    pub success: bool,
    pub index_type: String,
    pub build_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub async fn index_configure(
    State(state): State<SharedState>,
    Json(req): Json<IndexConfigureRequest>,
) -> impl IntoResponse {
    use vector_core::{IndexGenome, IndexType};

    let start = Instant::now();

    // Parse index type
    let index_type = match req.index_type.to_lowercase().as_str() {
        "flat" => IndexType::Flat,
        "hnsw" => IndexType::Hnsw,
        "ivf" => IndexType::Ivf,
        other => {
            return Json(IndexConfigureResponse {
                success: false,
                index_type: other.to_string(),
                build_time_ms: 0,
                error: Some(format!("Unknown index type: {}. Use 'flat', 'hnsw', or 'ivf'", other)),
            });
        }
    };

    // Build genome with provided or default parameters
    let genome = IndexGenome {
        index_type,
        hnsw_m: req.hnsw_m.unwrap_or(16),
        hnsw_ef_construction: req.hnsw_ef_construction.unwrap_or(200),
        hnsw_ef_search: req.hnsw_ef_search.unwrap_or(50),
        ivf_partitions: req.ivf_partitions.unwrap_or(256),
        ivf_nprobe: req.ivf_nprobe.unwrap_or(16),
    };

    // Set the index type
    let result = {
        let index = state.index.read();
        index.set_index_type(&genome)
    };

    match result {
        Ok(()) => {
            Json(IndexConfigureResponse {
                success: true,
                index_type: format!("{}", index_type),
                build_time_ms: start.elapsed().as_millis() as u64,
                error: None,
            })
        }
        Err(e) => {
            Json(IndexConfigureResponse {
                success: false,
                index_type: format!("{}", index_type),
                build_time_ms: start.elapsed().as_millis() as u64,
                error: Some(e.to_string()),
            })
        }
    }
}

#[derive(Debug, Serialize)]
pub struct IndexStatusResponse {
    pub active_type: Option<String>,
    pub vector_count: usize,
    pub evolved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

pub async fn index_status(State(state): State<SharedState>) -> impl IntoResponse {
    let index = state.index.read();

    let (active_type, params) = if let Some(genome) = index.get_active_genome() {
        let params = match genome.index_type {
            vector_core::IndexType::Flat => None,
            vector_core::IndexType::Hnsw => Some(serde_json::json!({
                "m": genome.hnsw_m,
                "ef_construction": genome.hnsw_ef_construction,
                "ef_search": genome.hnsw_ef_search,
            })),
            vector_core::IndexType::Ivf => Some(serde_json::json!({
                "partitions": genome.ivf_partitions,
                "nprobe": genome.ivf_nprobe,
            })),
        };
        (Some(format!("{}", genome.index_type)), params)
    } else {
        (None, None)
    };

    Json(IndexStatusResponse {
        active_type,
        vector_count: index.len(),
        evolved: index.is_evolved(),
        params,
    })
}

pub async fn index_reset(State(state): State<SharedState>) -> impl IntoResponse {
    let index = state.index.read();
    index.reset_index();

    Json(serde_json::json!({
        "success": true,
        "message": "Index reset to brute-force mode"
    }))
}

// ============================================================================
// Tool Call Interface
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ToolCallRequest {
    pub tool: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ToolCallResponse {
    pub success: bool,
    pub result: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub async fn tool_call(
    State(state): State<SharedState>,
    Json(req): Json<ToolCallRequest>,
) -> impl IntoResponse {
    let result = match req.tool.as_str() {
        "vector_search" => {
            // Parse parameters and perform actual search
            if let (Some(query), Some(k)) = (
                req.parameters.get("query").and_then(|v| v.as_array()),
                req.parameters.get("k").and_then(|v| v.as_u64()),
            ) {
                let query_vec: Vec<f32> = query
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();

                let index = state.index.read();
                let embedding = Embedding::new(query_vec);

                match index.search(&embedding, k as usize) {
                    Ok(results) => {
                        let formatted: Vec<serde_json::Value> = results
                            .iter()
                            .map(|r| serde_json::json!({"id": r.id.0, "score": r.score}))
                            .collect();
                        serde_json::json!({"results": formatted})
                    }
                    Err(e) => {
                        let err_msg = e.to_string();
                        serde_json::json!({"error": err_msg})
                    }
                }
            } else {
                serde_json::json!({"error": "Missing query or k parameter"})
            }
        }
        "graph_query" => {
            serde_json::json!({
                "message": "Graph query tool called",
                "parameters": req.parameters,
            })
        }
        "research" => {
            serde_json::json!({
                "message": "Research tool called",
                "parameters": req.parameters,
            })
        }
        "qd_diversify" => {
            serde_json::json!({
                "message": "QD diversify tool called",
                "parameters": req.parameters,
            })
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ToolCallResponse {
                    success: false,
                    result: serde_json::Value::Null,
                    error: Some(format!("Unknown tool: {}", req.tool)),
                }),
            )
                .into_response();
        }
    };

    Json(ToolCallResponse {
        success: true,
        result,
        error: None,
    })
    .into_response()
}

pub async fn list_tools() -> impl IntoResponse {
    Json(serde_json::json!({
        "tools": [
            {
                "name": "vector_search",
                "description": "Search for similar vectors in the index",
                "parameters": {
                    "query": "vector to search for",
                    "k": "number of results",
                    "metric": "distance metric (cosine, euclidean, dot_product)"
                }
            },
            {
                "name": "graph_query",
                "description": "Query the context graph",
                "parameters": {
                    "start_id": "starting concept ID",
                    "relation_types": "types of relationships to follow",
                    "max_depth": "maximum traversal depth"
                }
            },
            {
                "name": "research",
                "description": "Perform deep research on a topic",
                "parameters": {
                    "query": "research query",
                    "tier": "research depth (basic, medium, default, deeper)",
                    "max_sources": "maximum sources to return"
                }
            },
            {
                "name": "qd_diversify",
                "description": "Generate diverse query variations using MAP-Elites",
                "parameters": {
                    "base_queries": "seed queries",
                    "n": "number of diverse variations"
                }
            }
        ]
    }))
}
