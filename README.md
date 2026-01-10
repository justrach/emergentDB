# EmergentDB

A self-optimizing vector database that uses MAP-Elites evolutionary algorithm to automatically discover the optimal index configuration for your workload.

**44-193x faster than LanceDB. 10-25x faster than ChromaDB. 99%+ recall guaranteed.**

## The Problem

As embedding dimensions grow (768-3072), traditional vector databases struggle:

- **Manual Tuning Hell**: HNSW M=16? M=32? ef_construction=100? Most teams guess and hope.
- **Workload Mismatch**: Optimal config for 1K vectors ≠ optimal for 100K. Databases don't adapt.
- **Recall vs Speed**: Fast search often means lower recall. You shouldn't have to choose.

## The Solution

EmergentDB uses a **Dual Quality-Diversity System**:

1. **IndexQD** - 3D behavior space (Recall × Latency × Memory) evolves optimal index type and hyperparameters
2. **InsertQD** - 2D behavior space (Throughput × Efficiency) discovers fastest SIMD insertion strategy

The system automatically selects between HNSW, Flat, and IVF indices with evolved hyperparameters, achieving maximum search speed while enforcing a **99% recall floor**.

## Benchmark Results

768-dimensional vectors (OpenAI text-embedding-3-small size):

### 1,000 Vectors
| Database | Search Latency | Recall@10 |
|----------|---------------|-----------|
| **EmergentDB (Auto)** | **42.9μs** | 100% |
| ChromaDB | 1,078μs | 99% |
| LanceDB | 8,271μs | 100% |

**EmergentDB: 193x faster than LanceDB, 25x faster than ChromaDB**

### 10,000 Vectors
| Database | Search Latency | Recall@10 |
|----------|---------------|-----------|
| **EmergentDB (Auto)** | **122μs** | 100% |
| ChromaDB | 1,933μs | 76.5% |
| LanceDB | 7,442μs | 100% |

**EmergentDB: 61x faster than LanceDB, 16x faster than ChromaDB**

### 50,000 Vectors
| Database | Search Latency | Recall@10 |
|----------|---------------|-----------|
| **EmergentDB (Auto)** | **280μs** | 99.5% |
| ChromaDB | 2,698μs | 60% |
| LanceDB | 12,348μs | 100% |

**EmergentDB: 44x faster than LanceDB, 10x faster than ChromaDB**

## Quick Start

### Running the Server

```bash
# In-memory mode (default)
cargo run --release -p api-server

# With persistence (vectors survive restart)
DATA_DIR=./data cargo run --release -p api-server

# Custom settings
PORT=8080 VECTOR_DIM=768 DATA_DIR=./mydata cargo run --release -p api-server
```

### Using the Rust Library

```rust
use vector_core::index::emergent::{EmergentConfig, EmergentIndex};

// Create with search-optimized preset
let config = EmergentConfig::search_first();
let mut index = EmergentIndex::new(config);

// Insert your vectors
for (id, embedding) in vectors {
    index.insert(id, embedding)?;
}

// Evolve to find optimal configuration
let elite = index.evolve()?;
println!("Selected: {} (fitness: {:.3})",
    elite.genome.index_type, elite.fitness);

// Search - now 44-193x faster
let results = index.search(&query, 10)?;
```

## Configuration Presets

```rust
// Maximum search speed (default)
EmergentConfig::search_first()  // 50% recall, 40% speed, 5% memory, 5% build

// Balanced (equal weight to all objectives)
EmergentConfig::balanced()

// Memory-constrained environments
EmergentConfig::memory_efficient()
```

## Index Types

| Index | Complexity | Best For |
|-------|-----------|----------|
| **Flat** | O(N) | Small datasets, exact search baseline |
| **HNSW** | O(log N) | High recall requirements |
| **IVF** | O(N/partitions) | Large datasets with clustering |
| **PQ** | O(N) compressed | Memory-constrained environments |
| **Emergent** | Adaptive | Automatic optimization |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        EmergentDB                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   IndexQD    │    │   InsertQD   │    │   Archive    │   │
│  │  (3D Grid)   │    │  (2D Grid)   │    │  (Elites)    │   │
│  │              │    │              │    │              │   │
│  │ Recall       │    │ Throughput   │    │ Best configs │   │
│  │ Latency      │    │ Efficiency   │    │ per cell     │   │
│  │ Memory       │    │              │    │              │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│           │                  │                   │           │
│           └──────────────────┴───────────────────┘           │
│                              │                               │
│                    ┌─────────────────┐                       │
│                    │  MAP-Elites     │                       │
│                    │  Evolution      │                       │
│                    │  (6³ = 216      │                       │
│                    │   cells)        │                       │
│                    └─────────────────┘                       │
│                              │                               │
│                    ┌─────────────────┐                       │
│                    │  Optimal Index  │                       │
│                    │  Selection      │                       │
│                    └─────────────────┘                       │
│                              │                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │     HNSW     │  │     Flat     │  │     IVF      │       │
│  │  M, ef_c,    │  │   Brute      │  │  nlist,      │       │
│  │  ef_search   │  │   Force      │  │  nprobe      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Persistence

EmergentDB supports durable storage via RocksDB. When `DATA_DIR` is set:

```
┌─────────────────────────────────────────────────────────────┐
│                    PERSISTENCE MODE                         │
│                                                             │
│   Search: RAM ──SIMD──→ Results   (42μs, unchanged!)       │
│           ↑                                                 │
│           └── Loaded from disk on startup                   │
│                                                             │
│   Insert: RAM + async write → Disk  (non-blocking)         │
│                                                             │
│   Restart: Automatic recovery from RocksDB                  │
└─────────────────────────────────────────────────────────────┘
```

**Key benefits:**
- Vectors survive server restarts
- No impact on search performance (still in-memory SIMD)
- Automatic recovery on startup
- LZ4 compression for efficient storage

## Key Features

### 99% Recall Floor
Configurations with <99% recall get cubic fitness penalty, ensuring accuracy is never sacrificed for speed.

### SIMD-Optimized Insertions
Six insertion strategies compete in InsertQD:
- SimdSequential
- SimdBatch
- SimdParallel
- SimdChunked (L2 cache-friendly)
- SimdUnrolled (4-way loop unrolling)
- SimdInterleaved (two-pass for memory bandwidth)

Best strategy automatically selected: **5.6M vectors/second** on modern CPUs.

### Adaptive Index Selection
EmergentDB automatically selects:
- **HNSW**: For larger datasets (>5K vectors) - evolved M, ef_construction, ef_search
- **Flat**: For small datasets or when recall is paramount
- **IVF**: For very large datasets with memory constraints

## Running the Frontend

The visualization dashboard shows real-time benchmark comparisons with Airfoil design:

```bash
cd frontend
bun install
bun run dev
```

Open http://localhost:3000 to see the interactive benchmark visualization.

## Running Benchmarks

### Scale Benchmark (Rust)
```bash
cargo run --release --example scale_benchmark
```

### Comparison Benchmark (Python)
```bash
cd tests
source ../.venv312/bin/activate
python3 scale_comparison.py
```

## Project Structure

```
backend-new/
├── crates/
│   └── vector-core/
│       └── src/
│           ├── index/
│           │   ├── emergent.rs   # MAP-Elites evolution
│           │   ├── hnsw.rs       # HNSW index
│           │   ├── flat.rs       # Flat index
│           │   └── ivf.rs        # IVF index
│           └── simd.rs           # SIMD insert strategies
├── frontend/                     # Next.js 16.1 visualization
│   └── src/
│       ├── app/                  # App Router
│       └── components/           # React components
├── examples/
│   └── scale_benchmark.rs
└── tests/
    ├── scale_comparison.py
    └── benchmark_results/
        └── scale_comparison.json
```

## SIMD Optimization

EmergentDB uses platform-specific SIMD for maximum performance:

- **Apple Silicon (M1-M4)**: ARM NEON with fused multiply-accumulate
- **x86_64**: Wide SIMD via the `wide` crate
- **Fallback**: Scalar operations

Build for your platform:

```bash
# Apple M-series
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release

# Generic release
cargo build --release
```

## Why "Emergent"?

The optimal configuration *emerges* from evolutionary pressure. Instead of hand-tuning hyperparameters, EmergentDB:

1. Generates diverse configurations
2. Evaluates fitness on your actual data
3. Keeps the best per behavior cell (MAP-Elites)
4. Crosses over successful genomes
5. Mutates to explore new configurations
6. Repeats until convergence

The result: a configuration perfectly adapted to your specific workload, data distribution, and hardware.

## Configuration Options

### EmergentConfig

| Field | Default | Description |
|-------|---------|-------------|
| `dim` | 1536 | Vector dimensionality |
| `metric` | Cosine | Distance metric |
| `grid_size` | 6 | MAP-Elites grid resolution (6³ = 216 cells) |
| `generations` | 10 | Evolution iterations |
| `population_size` | 10 | Candidates per generation |
| `eval_sample_size` | 1000 | Vectors for benchmarking |
| `benchmark_queries` | 100 | Queries for recall measurement |

### Index-Specific Configs

**HNSW:**
- `m`: Neighbors per node (8-48)
- `ef_construction`: Build-time candidates (100-400)
- `ef_search`: Search-time candidates (20-200)

**IVF:**
- `num_partitions`: Cluster count (64-1024)
- `nprobe`: Partitions to search (4-64)
- `kmeans_iterations`: Training iterations

## Testing

```bash
# Run all tests
cargo test --workspace

# Run benchmarks
cargo bench

# Check compilation
cargo check --workspace
```

## License

MIT

## References

- HNSW: ["Efficient and robust approximate nearest neighbor search"](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2016)
- MAP-Elites: ["Illuminating search spaces"](https://arxiv.org/abs/1504.04909) (Mouret & Clune, 2015)
- Product Quantization: ["Product Quantization for Nearest Neighbor Search"](https://hal.inria.fr/inria-00514462v2) (Jégou et al., 2011)
- IVF: [FAISS](https://github.com/facebookresearch/faiss)
- LanceDB: [lancedb.com](https://lancedb.com)
- ChromaDB: [trychroma.com](https://trychroma.com)
