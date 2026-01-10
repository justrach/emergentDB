# EmergentDB

**A self-optimizing vector database that evolves its own configuration using quality-diversity algorithms.**

EmergentDB automatically discovers optimal index types and insert strategies for your specific data and workload - no manual tuning required.

## Benchmark Results

### vs Competition (303 vectors)

| Database | Search Latency | Recall@5 | Speedup |
|----------|---------------|----------|---------|
| **EmergentDB (Auto)** | **68 μs** | **100%** | **1x** |
| ChromaDB (HNSW) | 886 μs | 100% | 13x slower |
| LanceDB | 8,434 μs | 100% | **124x slower** |

### Scale Performance

| Scale | Emergent | HNSW Manual | Flat | Speedup |
|-------|----------|-------------|------|---------|
| **1K vectors** | 46 μs | 137 μs | 302 μs | **3x faster** |
| **10K vectors** | 67 μs | 457 μs | 4,000 μs | **7x faster** |
| **50K vectors** | 144 μs | 684 μs | 23,759 μs | **165x vs Flat** |

*768 dimensions, 100 document chunks + random vectors*

## How It Works

EmergentDB uses **MAP-Elites**, a quality-diversity algorithm, to evolve optimal configurations. Instead of searching for a single "best" solution, it maintains an archive of diverse high-performing solutions across multiple behavior dimensions.

### Dual-QD Architecture

EmergentDB runs **two independent evolution processes**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        EmergentDB                                │
├─────────────────────────────┬───────────────────────────────────┤
│         IndexQD             │           InsertQD                │
│   (Index Type Selection)    │    (Insert Strategy Selection)    │
├─────────────────────────────┼───────────────────────────────────┤
│ 3D Behavior Grid:           │ 2D Behavior Grid:                 │
│ • Recall (accuracy)         │ • Throughput (vectors/sec)        │
│ • Latency (query speed)     │ • CPU Efficiency                  │
│ • Memory (bytes/vector)     │                                   │
├─────────────────────────────┼───────────────────────────────────┤
│ Evolves:                    │ Evolves:                          │
│ • Index type (Flat/HNSW/IVF)│ • SIMD strategy                   │
│ • HNSW parameters (m, ef)   │ • Batch size                      │
│ • IVF partitions/nprobe     │ • Parallelism settings            │
└─────────────────────────────┴───────────────────────────────────┘
```

### Index Types

| Index | Best For | Trade-off |
|-------|----------|-----------|
| **Flat** | Small datasets (<10K) | Perfect recall, O(n) search |
| **HNSW** | Medium datasets | ~100% recall, O(log n) search |
| **IVF** | Large datasets | Tunable recall/speed |

### Insert Strategies (SIMD-Optimized)

| Strategy | Description | Best For |
|----------|-------------|----------|
| **SimdSequential** | One vector at a time | Low latency inserts |
| **SimdBatch** | Batch normalization | Bulk imports |
| **SimdParallel** | Multi-threaded + SIMD | Maximum throughput |
| **SimdChunked** | L2 cache-friendly chunks | Large batches |
| **SimdUnrolled** | 4-way loop unrolling | CPU pipelining |
| **SimdInterleaved** | Two-pass (norms then scale) | Memory bandwidth |

## Quick Start

```rust
use vector_core::index::emergent::{EmergentConfig, EmergentIndex};
use vector_core::{DistanceMetric, Embedding, NodeId};

// Create index with search-first priority
let mut config = EmergentConfig::search_first();
config.dim = 768;
config.metric = DistanceMetric::Cosine;

let mut index = EmergentIndex::new(config);

// Insert vectors
for (i, vector) in vectors.iter().enumerate() {
    index.insert(NodeId::new(i as u64), Embedding::new(vector.clone()))?;
}

// Run evolution - automatically finds best index type + insert strategy
let elite = index.evolve()?;

println!("Selected: {} with {:.1}μs latency",
    elite.genome.index_type,
    elite.metrics.query_latency_us
);

// Search with evolved configuration
let results = index.search(&query, 10)?;
```

## Configuration Presets

```rust
// For fastest search (ignores build time)
let config = EmergentConfig::search_first();

// Balanced across all metrics
let config = EmergentConfig::balanced();

// Prioritize recall over speed
let config = EmergentConfig::recall_first();

// Fastest index building
let config = EmergentConfig::fast_ingest();
```

## Priority Weights

Each preset uses different weights for the fitness function:

| Preset | Recall | Speed | Memory | Build |
|--------|--------|-------|--------|-------|
| `search_first()` | 50% | 40% | 5% | 5% |
| `balanced()` | 30% | 30% | 20% | 20% |
| `recall_first()` | 50% | 25% | 15% | 10% |
| `fast_ingest()` | 25% | 20% | 10% | 45% |

## Evolution Process

```
Generation 0: Seed with heuristic configurations
    ↓
Generation 1-N:
    1. Evaluate population on sample data
    2. Map each solution to behavior grid cell
    3. Keep best solution per cell (elite)
    4. Generate offspring via mutation/crossover
    ↓
Early termination if:
    • Fitness threshold reached (default: 0.92)
    • No improvement for 3 generations
    ↓
Return: Best elite from archive
```

## Key Features

### 99% Recall Floor
Configurations with recall < 99% are heavily penalized, ensuring the evolution never sacrifices accuracy for speed.

```rust
// Fitness function applies cubic penalty below 99% recall
let recall_penalty = if recall < 0.99 {
    (recall / 0.99).powi(3)  // Harsh penalty
} else {
    1.0
};
```

### ARM NEON Optimization
Native SIMD intrinsics for Apple M-series chips:
- 128-bit NEON registers (4x f32)
- Fused multiply-accumulate (vfmaq_f32)
- 4x loop unrolling for pipelining

```bash
# Build with M-series optimization
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Geometric Mean Fitness
Uses geometric mean instead of weighted sum, ensuring ALL metrics must be good:

```
fitness = (recall^w1 × speed^w2 × memory^w3 × build^w4)^(1/Σw)
```

## Architecture

```
crates/vector-core/
├── src/
│   ├── lib.rs              # Core types (Embedding, NodeId, etc.)
│   ├── simd.rs             # SIMD primitives + insert strategies
│   └── index/
│       ├── mod.rs          # VectorIndex trait
│       ├── flat.rs         # Brute-force index
│       ├── hnsw.rs         # Hierarchical NSW
│       ├── ivf.rs          # Inverted file index
│       ├── pq.rs           # Product quantization
│       └── emergent.rs     # MAP-Elites evolution system
└── examples/
    └── pdf_benchmark.rs    # Comparison benchmark
```

## Benchmarking

Run the full benchmark comparing EmergentDB vs LanceDB vs ChromaDB:

```bash
# Rust-only benchmark
cargo run --release --example pdf_benchmark

# Full comparison (requires Python deps)
python tests/db_comparison_benchmark.py
```

## Performance Tips

1. **Use `search_first()` for production** - Minimizes query latency
2. **Let evolution run** - The one-time cost finds optimal configs
3. **Batch inserts** - InsertQD will select the best strategy
4. **Build with `-C target-cpu=native`** - Enables full SIMD

## How MAP-Elites Differs from Traditional Optimization

| Traditional | MAP-Elites |
|------------|------------|
| Single best solution | Archive of diverse solutions |
| May get stuck in local optima | Explores entire behavior space |
| One config for all workloads | Multiple configs for different needs |
| Requires manual hyperparameter tuning | Self-discovers optimal parameters |

## License

MIT
