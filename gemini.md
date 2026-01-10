# EmergentDB - Gemini Context

This file provides context for Google Gemini (or any AI assistant) to understand and work with EmergentDB effectively.

## What is EmergentDB?

EmergentDB is a **self-optimizing vector database** built in Rust that uses evolutionary algorithms (specifically MAP-Elites) to automatically discover the best index configuration for your specific workload. Instead of manually tuning hyperparameters, the system evolves optimal configurations through a quality-diversity approach.

**Performance claims:**
- 44-193x faster than LanceDB
- 10-25x faster than ChromaDB
- 99%+ recall guaranteed

## How It Works

### The Core Concept

Traditional vector databases require manual tuning of parameters like:
- HNSW: `M` (neighbors), `ef_construction`, `ef_search`
- IVF: `nlist` (partitions), `nprobe` (search breadth)

EmergentDB solves this by:
1. Defining a **behavior space** (Recall x Latency x Memory)
2. Evolving diverse configurations using **MAP-Elites**
3. Selecting the configuration with the best fitness for your priority

### Dual Quality-Diversity System

| System | Behavior Space | What It Evolves |
|--------|---------------|-----------------|
| IndexQD | 3D: Recall x Latency x Memory | Index type + hyperparameters |
| InsertQD | 2D: Throughput x Efficiency | SIMD insertion strategies |

## Project Layout

```
emergentDB/
├── Cargo.toml              # Rust workspace config
├── crates/
│   ├── vector-core/        # Vector ops, indices, SIMD
│   ├── qd-engine/          # MAP-Elites evolution
│   ├── api-server/         # REST API (Axum framework)
│   ├── context-graph/      # Semantic graph storage
│   └── research-client/    # External API client (Gemini, etc.)
├── examples/
│   ├── scale_benchmark.rs  # Performance testing
│   └── ingestion/          # Python document ingestion
├── frontend/               # Next.js dashboard
└── tests/                  # Python comparison tests
```

## Key Components

### 1. Vector Core (`crates/vector-core/`)

The performance-critical foundation with SIMD optimization.

**Main types:**
```rust
struct Embedding { data: Vec<f32> }    // Vector data
struct NodeId(pub u64);                 // Unique ID
struct SearchResult { id: NodeId, score: f32 }
enum DistanceMetric { Cosine, Euclidean, DotProduct }
```

**Index implementations:**
| File | Index Type | Complexity | Best For |
|------|-----------|-----------|----------|
| `flat.rs` | Flat (brute-force) | O(N) | Small datasets, exact search |
| `hnsw.rs` | HNSW | O(log N) | High recall needs |
| `ivf.rs` | IVF | O(N/partitions) | Large datasets |
| `pq.rs` | Product Quantization | O(N) compressed | Memory constraints |
| `emergent.rs` | Adaptive | Varies | Automatic optimization |

### 2. QD Engine (`crates/qd-engine/`)

Implements the MAP-Elites quality-diversity algorithm.

**Key files:**
- `archive.rs`: Grid-based storage of elite solutions
- `behavior.rs`: Maps configurations to behavior descriptors
- `evolution.rs`: The main evolution loop

**Algorithm overview:**
```
1. Seed archive with 10 heuristic configurations
2. Loop for N generations:
   a. Pick random elite from archive
   b. Mutate 1-2 parameters
   c. Evaluate fitness (recall, speed, memory, build time)
   d. Compute behavior descriptor (position in 3D grid)
   e. If better than current cell occupant, replace
3. Return best overall elite
```

### 3. API Server (`crates/api-server/`)

REST API built with Axum web framework.

**Endpoints:**
```
GET  /health              - Health check
POST /vectors/search      - k-NN search
POST /vectors/insert      - Add vector
POST /vectors/batch_insert - Bulk add
POST /ingest              - Ingest with embedding
POST /ingest/batch        - Batch ingest
GET  /ingest/stats        - Ingestion statistics
POST /qd/evolve           - Run evolution
POST /tools/call          - LLM tool interface
```

## API Usage with Gemini Embeddings

### Complete Flow: Document → EmergentDB

```python
from google import genai
from google.genai import types
import requests

# 1. Initialize Gemini client
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# 2. Get embedding from Gemini
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Your document text here",
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768
    )
)
embedding = list(result.embeddings[0].values)

# 3. Store in EmergentDB
response = requests.post(
    "http://localhost:3000/vectors/insert",
    json={
        "id": 1,
        "vector": embedding,
        "metadata": {"source": "document.pdf"}
    }
)
```

### Search with Query Embedding

```python
# 1. Get query embedding (use RETRIEVAL_QUERY task type)
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is machine learning?",
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768
    )
)
query_embedding = list(result.embeddings[0].values)

# 2. Search EmergentDB
response = requests.post(
    "http://localhost:3000/vectors/search",
    json={"query": query_embedding, "k": 5}
)
results = response.json()["results"]
# [{"id": 42, "score": 0.8521}, {"id": 17, "score": 0.7834}, ...]
```

### Batch Insert

```bash
curl -X POST http://localhost:3000/vectors/batch_insert \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": 1, "vector": [...768 floats...], "metadata": {"chunk": 0}},
      {"id": 2, "vector": [...768 floats...], "metadata": {"chunk": 1}}
    ]
  }'
```

## Python Ingestion CLI

A ready-to-use CLI that handles the complete pipeline:

### Setup
```bash
cd examples/ingestion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your Gemini API key in .env
echo "GEMINI_API_KEY=your_key_here" >> ../../.env
```

### Usage
```bash
# Ingest a PDF (auto OCR if scanned)
python ingest.py ingest document.pdf

# Ingest images with Gemini Vision OCR
python ingest.py ingest screenshot.png

# Ingest all files in directory
python ingest.py ingest-dir ./documents -r

# Semantic search
python ingest.py query "explain the algorithm" --k 5

# Check status
python ingest.py stats
```

### How It Works
```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Document   │────▶│  Gemini API │────▶│  EmergentDB     │
│  (PDF/IMG)  │     │  (OCR+Embed)│     │  (Rust Server)  │
└─────────────┘     └─────────────┘     └─────────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
   Chunking           768-dim             SIMD Search
   (1000 chars)       embeddings          (cosine sim)
       │                                       │
       ▼                                       ▼
   metadata.db                            Rankings
   (text content)                         (by score)
```

### 4. Research Client (`crates/research-client/`)

HTTP client for external APIs (used for Gemini OCR/embeddings).

## SIMD Optimization

Platform-specific vector operations for maximum performance:

**Apple Silicon (ARM NEON):**
- 4-way unrolled FMA (fused multiply-add)
- Functions: `dot_product_neon()`, `squared_euclidean_neon()`

**x86_64 (AVX2):**
- Uses `wide` crate for portable SIMD
- 8xf32 vector operations

**Six insert strategies compete in InsertQD:**
1. SimdSequential - One vector at a time
2. SimdBatch - Batch normalization
3. SimdParallel - Multi-threaded + SIMD
4. SimdChunked - L2 cache-friendly chunks
5. SimdUnrolled - 4-way loop unrolling
6. SimdInterleaved - Two-pass for memory bandwidth

**Result:** ~5.6M vectors/second on modern CPUs.

## Configuration Options

### EmergentConfig
```rust
EmergentConfig {
    dim: 1536,                    // Vector dimension
    metric: DistanceMetric::Cosine,
    grid_size: 6,                 // MAP-Elites grid (6^3 = 216 cells)
    generations: 10,
    population_size: 10,
    eval_sample_size: 1000,
    benchmark_queries: 100,
    // ... priority weights
}
```

### Presets
```rust
EmergentConfig::search_first()    // Fast queries (default)
EmergentConfig::balanced()        // Equal weight to all metrics
EmergentConfig::memory_efficient() // Low memory usage
EmergentConfig::fast_ingest()     // Fast insertions
```

## Fitness Calculation

**Geometric mean with weights:**
```
fitness = (recall^w1 * speed^w2 * memory^w3 * build^w4)^(1/sum_weights)
```

**99% Recall Floor:**
Configurations with recall < 99% receive a cubic penalty:
```
penalty = (recall / 0.99)^3  if recall < 0.99
```

This ensures accuracy is never sacrificed for speed.

## Common Commands

### Build
```bash
# Standard release
cargo build --release

# Apple Silicon optimized
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release
```

### Test
```bash
cargo test --workspace        # All tests
cargo test -p vector-core     # Specific crate
```

### Run Server
```bash
# Default settings
cargo run --release -p api-server

# Custom port/dimension
PORT=8080 VECTOR_DIM=768 cargo run --release -p api-server
```

### Benchmarks
```bash
cargo bench -p vector-core                    # Criterion benchmarks
cargo run --release --example scale_benchmark # Scale tests
```

## Usage Example

```rust
use vector_core::index::emergent::{EmergentConfig, EmergentIndex};

// Create index with search-optimized preset
let config = EmergentConfig::search_first();
let mut index = EmergentIndex::new(config);

// Insert vectors
for (id, embedding) in vectors {
    index.insert(id, embedding)?;
}

// Evolve to find optimal configuration
let elite = index.evolve()?;
println!("Selected: {} (fitness: {:.3})", 
    elite.genome.index_type, elite.fitness);

// Search (now 44-193x faster)
let results = index.search(&query, 10)?;
```

## Test Summary

| Crate | Tests | Focus |
|-------|-------|-------|
| vector-core | 51 | Distance metrics, SIMD ops, all index types |
| qd-engine | 8 | Archive, behavior, evolution |
| context-graph | 7 | Graph operations, traversal |
| api-server | 3 | Tool schemas, formatting |
| research-client | 4 | Config, tiers, types |

## Dependencies

**Runtime:**
- tokio 1.43+ (async runtime)
- axum 0.8+ (web framework)

**Performance:**
- wide 0.7 (portable SIMD)
- rayon 1.10 (parallelism)
- parking_lot 0.12 (fast locks)

**Serialization:**
- serde 1.0 / serde_json 1.0

## Extending EmergentDB

### Adding a New Index Type

1. Create `crates/vector-core/src/index/your_index.rs`
2. Implement the `VectorIndex` trait
3. Add to `IndexType` enum in `emergent.rs`
4. Add genome parameters if tunable
5. Include in evolution candidate generation

### Adding a New Distance Metric

1. Add variant to `DistanceMetric` enum
2. Implement in `distance.rs`
3. Add SIMD version in `simd.rs` if beneficial
4. Update tests

## Known Limitations

- Single-node only (no distributed mode yet)
- In-memory storage (no persistence built-in)
- Evolution requires sufficient sample data
- SIMD code has platform-specific paths

## Troubleshooting

**"linker 'cc' not found"**
```bash
# macOS
xcode-select --install
# Linux
sudo apt install build-essential
```

**"Address already in use"**
```bash
lsof -i :3000
kill -9 <PID>
```

**Slow performance**
- Ensure using `--release` flag
- Use CPU-specific flags: `RUSTFLAGS="-C target-cpu=native"`

---

## TODO / Roadmap

### High Priority

- [x] **On-Device Persistence**: RocksDB backend for vector storage ✅
  - Set `DATA_DIR` environment variable to enable persistence
  - Vectors automatically recovered from disk on server restart
  - Background persistence for inserts (async, non-blocking)

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
