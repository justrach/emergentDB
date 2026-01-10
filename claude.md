# EmergentDB - Claude Code Context

This file provides context for Claude Code (or any AI assistant) to understand and work with EmergentDB effectively.

## Project Overview

EmergentDB is a self-optimizing vector database written in Rust that uses the MAP-Elites evolutionary algorithm to automatically discover optimal index configurations. It achieves 44-193x faster search than LanceDB and 10-25x faster than ChromaDB while maintaining 99%+ recall.

## Key Innovation

**Dual Quality-Diversity System:**
- **IndexQD**: Evolves index type and hyperparameters across a 3D behavior space (Recall x Latency x Memory)
- **InsertQD**: Evolves insertion strategies across a 2D behavior space (Throughput x Efficiency)

## Workspace Structure

```
emergentDB/
├── Cargo.toml              # Workspace definition
├── crates/
│   ├── vector-core/        # Core vector operations, indices, SIMD
│   ├── qd-engine/          # MAP-Elites evolution algorithm
│   ├── api-server/         # REST API (Axum)
│   ├── context-graph/      # Semantic relationship graph
│   └── research-client/    # External API integrations
├── examples/
│   ├── scale_benchmark.rs  # Performance benchmarks
│   ├── pdf_benchmark.rs    # Document embedding simulation
│   └── ingestion/          # Python ingestion CLI
├── frontend/               # Next.js 16.1 visualization
└── tests/                  # Python comparison benchmarks
```

## Crate Responsibilities

### vector-core (Core Library)
**Location:** `crates/vector-core/src/`

| File | Purpose |
|------|---------|
| `lib.rs` | Core types: `Embedding`, `NodeId`, `SearchResult`, `DistanceMetric` |
| `simd.rs` | Platform-specific SIMD operations (ARM NEON / x86 AVX2) |
| `distance.rs` | Distance metrics with SIMD acceleration |
| `index/mod.rs` | `VectorIndex` trait defining common interface |
| `index/flat.rs` | Brute-force O(N) exact search |
| `index/hnsw.rs` | Hierarchical Navigable Small World graphs |
| `index/ivf.rs` | Inverted File Index with k-means clustering |
| `index/pq.rs` | Product Quantization for compression |
| `index/emergent.rs` | Adaptive index with MAP-Elites evolution |

### qd-engine (Evolution)
**Location:** `crates/qd-engine/src/`

| File | Purpose |
|------|---------|
| `archive.rs` | Grid-based elite storage with niching |
| `behavior.rs` | Behavior characterization and descriptors |
| `evolution.rs` | MAP-Elites algorithm implementation |

### api-server (HTTP API)
**Location:** `crates/api-server/src/`

| File | Purpose |
|------|---------|
| `main.rs` | Server entry point |
| `routes.rs` | Endpoint definitions |
| `handlers.rs` | Request handlers |
| `tools.rs` | LLM tool schemas (OpenAI/Anthropic format) |

## Index Types and When to Use

| Index | Complexity | Use Case |
|-------|-----------|----------|
| `Flat` | O(N) | < 10K vectors, exact search baseline |
| `HNSW` | O(log N) | Medium-large datasets, high recall needs |
| `IVF` | O(N/partitions) | Large datasets, memory constraints |
| `PQ` | O(N) compressed | Extreme memory constraints |
| `Emergent` | Adaptive | Automatic optimization (recommended) |

## Key Traits and Types

### VectorIndex Trait
```rust
pub trait VectorIndex: Send + Sync {
    fn insert(&mut self, id: NodeId, embedding: Embedding) -> Result<()>;
    fn remove(&mut self, id: NodeId) -> Result<bool>;
    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>>;
    fn len(&self) -> usize;
    fn get(&self, id: NodeId) -> Option<&Embedding>;
}
```

### Core Types
```rust
pub struct Embedding { data: Vec<f32> }
pub struct NodeId(pub u64);
pub struct SearchResult { pub id: NodeId, pub score: f32 }
pub enum DistanceMetric { Cosine, Euclidean, DotProduct }
```

### Evolution Genomes
```rust
pub struct IndexGenome {
    pub index_type: IndexType,  // Flat, Hnsw, Ivf
    pub hnsw_m: usize,          // 8-48
    pub hnsw_ef_construction: usize,  // 50-400
    pub hnsw_ef_search: usize,  // 10-150
    pub ivf_partitions: usize,  // 32-512
    pub ivf_nprobe: usize,      // 2-64
}
```

## Common Development Tasks

### Building
```bash
# Standard release build
cargo build --release

# Apple Silicon optimized
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release

# x86 optimized
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Running Tests
```bash
# All workspace tests
cargo test --workspace

# Specific crate
cargo test -p vector-core
cargo test -p qd-engine

# With output
cargo test --workspace -- --nocapture
```

### Running Benchmarks
```bash
# Criterion benchmarks
cargo bench -p vector-core

# Scale benchmark example
cargo run --release --example scale_benchmark
```

### Starting the Server
```bash
# Default (port 3000, 768 dimensions)
cargo run --release -p api-server

# Custom settings
PORT=8080 VECTOR_DIM=1536 cargo run --release -p api-server
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/vectors/search` | k-NN search |
| POST | `/vectors/insert` | Insert single vector |
| POST | `/vectors/batch_insert` | Bulk insert |
| POST | `/ingest` | Ingest document with embedding |
| POST | `/ingest/batch` | Batch ingest documents |
| GET | `/ingest/stats` | Get ingestion statistics |
| POST | `/qd/evolve` | Run evolution |
| POST | `/tools/call` | LLM tool interface |

## API Usage Examples

### Insert a Vector
```bash
curl -X POST http://localhost:3000/vectors/insert \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "vector": [0.1, 0.2, ...],  # 768-dim for Gemini embeddings
    "metadata": {"source": "document.pdf"}
  }'
```

### Search for Similar Vectors
```bash
curl -X POST http://localhost:3000/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, ...],  # 768-dim query embedding
    "k": 5
  }'
```

**Response:**
```json
{
  "results": [
    {"id": 42, "score": 0.8521},
    {"id": 17, "score": 0.7834}
  ],
  "latency_ms": 1
}
```

### Batch Insert
```bash
curl -X POST http://localhost:3000/vectors/batch_insert \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": 1, "vector": [...], "metadata": {}},
      {"id": 2, "vector": [...], "metadata": {}}
    ]
  }'
```

### Check Ingestion Stats
```bash
curl http://localhost:3000/ingest/stats
# {"total_documents": 11, "vector_dimension": 768, "index_type": "flat"}
```

## Python Ingestion CLI

Located at `examples/ingestion/`, this CLI handles document processing with Gemini OCR and embeddings.

### Setup
```bash
cd examples/ingestion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Commands
```bash
# Ingest a document (PDF, image, text)
python ingest.py ingest document.pdf

# Ingest directory recursively
python ingest.py ingest-dir ./documents -r

# Search the database
python ingest.py query "What is machine learning?" --k 5

# List ingested documents
python ingest.py list

# Show statistics
python ingest.py stats

# Interactive mode
python ingest.py interactive
```

### Architecture
```
Document → Python CLI → Gemini API → EmergentDB Server
              ↓              ↓              ↓
         (chunking)    (OCR + embed)   (vector store)
              ↓                              ↓
         metadata.db                   Rust SIMD search
         (text only)                   (rankings)
```

## Configuration Presets

```rust
// Maximum search speed
EmergentConfig::search_first()  // 50% recall, 40% speed, 5% memory, 5% build

// Equal weight to all objectives
EmergentConfig::balanced()

// Memory-constrained environments
EmergentConfig::memory_efficient()

// Fast ingestion priority
EmergentConfig::fast_ingest()
```

## SIMD Details

The codebase uses platform-specific SIMD for performance:

- **ARM NEON** (Apple Silicon): 4-way unrolled FMA operations
- **x86 AVX2**: Via the `wide` crate (8xf32 vectors)
- **Fallback**: Scalar operations

Key SIMD functions in `simd.rs`:
- `dot_product_neon()` / `dot_product_simd()`
- `squared_euclidean_neon()` / `squared_euclidean_simd()`
- `normalize_simd()`
- `batch_normalize_simd()`

## Evolution Algorithm (MAP-Elites)

1. **Initialize**: Seed archive with 10 heuristic configurations
2. **For each generation**:
   - Select random elite from archive
   - Mutate to create offspring (1-2 parameter changes)
   - Evaluate fitness on sample vectors
   - Map to behavior cell (Recall x Latency x Memory)
   - If better than current cell elite, update
3. **Early stopping**: If fitness > threshold or no improvement for 3 generations
4. **Return**: Best elite across all cells

**Fitness Function** (Geometric Mean):
```
fitness = (recall^w1 x speed^w2 x memory^w3 x build^w4)^(1/sum(w))
```

**99% Recall Floor**: Configurations with recall < 99% get cubic penalty to prevent speed-first degradation.

## Testing Notes

- Tests are in each crate's `src/` directory
- Use `#[cfg(test)]` modules
- Integration tests for the API server test endpoint functionality
- Benchmarks use Criterion in `benches/` directories

## Common Issues

**Compilation warnings to expect:**
- Unused imports in some modules (can be ignored)
- Unused variables in SIMD code (chunking/remainder handling)
- Dead code warnings for helper methods

**Build requirements:**
- Rust 1.75+ required
- `clang` for some dependencies on Linux
- `xcode-select --install` on macOS

## Contributing Guidelines

1. Run `cargo test --workspace` before commits
2. Use `cargo fmt` for formatting
3. Use `cargo clippy` for linting
4. Add tests for new functionality
5. Update documentation for API changes

---

## TODO / Roadmap

### High Priority

- [ ] **On-Device Persistence**: Add RocksDB or SQLite backend for vector storage
  - Create database schema that initializes on first run
  - Support recovery from disk on server restart
  - Add `--data-dir` flag to specify storage location

- [ ] **On-Device Embeddings**: Support local embedding models
  - Integrate `candle` or `ort` for on-device inference
  - Support ONNX models (e.g., all-MiniLM-L6-v2)
  - Fallback to Gemini API when local model unavailable

- [ ] **Database Recovery**: Implement WAL (Write-Ahead Logging)
  - Persist index state to disk periodically
  - Recovery from crash with minimal data loss
  - Configurable sync intervals

### Medium Priority

- [ ] **Index Persistence**: Save/load evolved index configurations
  - Serialize best elite genome to disk
  - Skip evolution on restart if good config exists
  - Invalidate cache when data distribution changes significantly

- [ ] **Streaming Ingestion**: Support large file streaming
  - Process PDFs page-by-page without loading entire file
  - Chunked upload API for large documents
  - Progress callbacks during ingestion

- [ ] **Multi-Index Support**: Named collections/namespaces
  - Create multiple indexes per server
  - API routes: `/collections/{name}/vectors/...`
  - Per-collection evolution and configuration

### Lower Priority

- [ ] **Distributed Mode**: Multi-node vector search
  - Shard vectors across nodes
  - Merge results from multiple shards
  - Consistent hashing for shard assignment

- [ ] **GPU Acceleration**: CUDA/Metal support for distance calculations
  - Batch distance computation on GPU
  - Hybrid CPU/GPU for different workload sizes

- [ ] **Quantization**: Reduce memory footprint
  - int8 quantization for vectors
  - Binary quantization option
  - Automatic quantization based on memory pressure

### Completed

- [x] REST API with Axum
- [x] SIMD-optimized distance calculations
- [x] MAP-Elites evolution engine
- [x] Python ingestion CLI with Gemini OCR
- [x] Graceful server shutdown
- [x] Multiple index types (Flat, HNSW, IVF, PQ)
