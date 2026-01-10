# EmergentDB Architecture

This document describes the internal architecture of EmergentDB, a self-optimizing vector database that uses quality-diversity evolution to automatically discover optimal index configurations.

## System Overview

```
                                 ┌─────────────────────────────────────┐
                                 │            EmergentDB               │
                                 └─────────────────────────────────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
                    ▼                              ▼                              ▼
          ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
          │   API Server    │            │  Vector Core    │            │   QD Engine     │
          │    (Axum)       │◀──────────▶│  (Rust SIMD)    │◀──────────▶│  (MAP-Elites)   │
          └─────────────────┘            └─────────────────┘            └─────────────────┘
                    │                              │                              │
                    │                              │                              │
          ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
          │ Research Client │            │  Context Graph  │            │  Elite Archive  │
          │   (HTTP API)    │            │  (Semantic DAG) │            │   (Grid-based)  │
          └─────────────────┘            └─────────────────┘            └─────────────────┘
```

## Core Architecture Layers

### Layer 1: Vector Core (Performance Foundation)

The vector-core crate provides the performance-critical vector operations with SIMD optimization.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   Vector Core                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │
│  │   Embedding  │   │   NodeId     │   │ SearchResult │   │ DistMetric   │          │
│  │   Vec<f32>   │   │     u64      │   │  (id, score) │   │ Cos/Euc/Dot  │          │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘          │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                            VectorIndex Trait                                   │ │
│  │  fn insert(&mut self, id: NodeId, emb: Embedding) -> Result<()>                │ │
│  │  fn remove(&mut self, id: NodeId) -> Result<bool>                              │ │
│  │  fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>>    │ │
│  │  fn len(&self) -> usize                                                        │ │
│  │  fn get(&self, id: NodeId) -> Option<&Embedding>                               │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                            │
│         ┌───────────────┬───────────────┼───────────────┬───────────────┐           │
│         ▼               ▼               ▼               ▼               ▼           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │    Flat    │  │    HNSW    │  │    IVF     │  │     PQ     │  │  Emergent  │    │
│  │   O(N)     │  │  O(log N)  │  │  O(N/k)    │  │ Compressed │  │  Adaptive  │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              SIMD Engine                                       │ │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐    │ │
│  │  │   ARM NEON (M1+)    │  │    x86 AVX2/SSE     │  │   Scalar Fallback   │    │ │
│  │  │   4xf32 vectors     │  │    8xf32 vectors    │  │   Standard loops    │    │ │
│  │  │   vfmaq_f32 FMA     │  │    wide crate       │  │   Auto-vectorize    │    │ │
│  │  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Layer 2: QD Engine (Evolution System)

The quality-diversity engine implements MAP-Elites for automatic configuration discovery.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    QD Engine                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                            MAP-Elites Loop                                     │ │
│  │                                                                                │ │
│  │   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐        │ │
│  │   │   Select   │───▶│   Mutate   │───▶│  Evaluate  │───▶│   Place    │        │ │
│  │   │   Elite    │    │  Offspring │    │  Fitness   │    │ in Archive │        │ │
│  │   └────────────┘    └────────────┘    └────────────┘    └────────────┘        │ │
│  │         ▲                                                      │               │ │
│  │         └──────────────────────────────────────────────────────┘               │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────────────────┐ │
│  │        Elite Archive          │  │         Behavior Space                       │ │
│  │                               │  │                                              │ │
│  │   Grid: 6 x 6 x 6 = 216 cells │  │   IndexQD (3D):                             │ │
│  │                               │  │     X: Recall@10 (0.0 - 1.0)                │ │
│  │   Each cell stores:           │  │     Y: Query Latency (1μs - 1000μs log)     │ │
│  │   - Best genome for that      │  │     Z: Memory per vector (3KB - 6KB)        │ │
│  │     behavior region           │  │                                              │ │
│  │   - Fitness score             │  │   InsertQD (2D):                            │ │
│  │   - Behavior descriptor       │  │     X: Throughput (100K - 10M vec/s log)    │ │
│  │                               │  │     Y: CPU Efficiency (0.0 - 1.0)           │ │
│  └──────────────────────────────┘  └──────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           Genome Representation                                │ │
│  │                                                                                │ │
│  │   IndexGenome {                    InsertGenome {                              │ │
│  │     index_type: Flat|Hnsw|Ivf        strategy: SimdSeq|Batch|Parallel|...     │ │
│  │     hnsw_m: 8..48                    batch_size: 32..4096                      │ │
│  │     hnsw_ef_construction: 50..400    use_parallelism: bool                     │ │
│  │     hnsw_ef_search: 10..150          prefetch_distance: 0..8                   │ │
│  │     ivf_partitions: 32..512       }                                            │ │
│  │     ivf_nprobe: 2..64                                                          │ │
│  │   }                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Layer 3: API Server (HTTP Interface)

RESTful API built with Axum for external access.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   API Server                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              Axum Router                                       │ │
│  │                                                                                │ │
│  │   GET  /health              → health_check()                                   │ │
│  │   POST /vectors/search      → search_vectors()                                 │ │
│  │   POST /vectors/insert      → insert_vector()                                  │ │
│  │   POST /vectors/batch_insert→ batch_insert()                                   │ │
│  │   POST /graph/concepts      → ontology_operations()                            │ │
│  │   POST /qd/evolve           → run_evolution()                                  │ │
│  │   POST /tools/call          → llm_tool_interface()                             │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────────────────┐ │
│  │       App State              │  │        Tool Schemas                          │ │
│  │                              │  │                                              │ │
│  │   Arc<RwLock<AppState>> {    │  │   OpenAI Format:                             │ │
│  │     index: Box<dyn VectorIdx>│  │     { "type": "function", ... }              │ │
│  │     dim: usize               │  │                                              │ │
│  │     config: EmergentConfig   │  │   Anthropic Format:                          │ │
│  │   }                          │  │     { "name": "...", "input_schema": ... }   │ │
│  └──────────────────────────────┘  └──────────────────────────────────────────────┘ │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Index Architecture Deep Dive

### Flat Index

Simple brute-force implementation for baseline and small datasets.

```
┌────────────────────────────────────────┐
│              Flat Index                │
├────────────────────────────────────────┤
│                                        │
│   HashMap<NodeId, Embedding>           │
│                                        │
│   Search: Iterate all, compute dist    │
│   Insert: O(1) hash insert             │
│   Memory: dim × 4 bytes per vector     │
│                                        │
│   Best for: < 10K vectors, exact       │
│                                        │
└────────────────────────────────────────┘
```

### HNSW Index

Hierarchical Navigable Small World graph for fast approximate search.

```
┌────────────────────────────────────────────────────────────────────┐
│                           HNSW Index                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Layer 2 (sparse):    ●─────────────────●                        │
│                        │                  │                        │
│   Layer 1 (medium):    ●───●───●──────●──●                        │
│                        │   │   │      │  │                        │
│   Layer 0 (dense):     ●─●─●─●─●─●─●─●─●─●                        │
│                                                                    │
│   Parameters:                                                      │
│   - M: neighbors per node (8-48)                                   │
│   - ef_construction: candidates during build (50-400)              │
│   - ef_search: candidates during query (10-150)                    │
│                                                                    │
│   Search: Start at top, greedy descend, expand at base             │
│   Complexity: O(log N) average                                     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### IVF Index

Inverted File Index using k-means clustering.

```
┌────────────────────────────────────────────────────────────────────┐
│                           IVF Index                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Centroids (k-means):    C1      C2      C3      C4    ...       │
│                            │       │       │       │              │
│                            ▼       ▼       ▼       ▼              │
│   Partitions:          [●●●●]  [●●●]   [●●●●●] [●●]   ...        │
│                                                                    │
│   Parameters:                                                      │
│   - num_partitions: cluster count (32-512)                         │
│   - nprobe: partitions to search (2-64)                            │
│   - kmeans_iterations: training steps (20)                         │
│                                                                    │
│   Search: Find nprobe nearest centroids, search those partitions   │
│   Complexity: O(N / num_partitions × nprobe)                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Product Quantization (PQ)

Subspace quantization for memory efficiency.

```
┌────────────────────────────────────────────────────────────────────┐
│                           PQ Index                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Original vector (768 dims):                                      │
│   [x1, x2, ..., x768]                                              │
│        ↓                                                           │
│   Split into M subvectors (e.g., M=96):                            │
│   [s1][s2][s3]...[s96]  (each 8 dims)                             │
│        ↓                                                           │
│   Quantize each subvector to nearest centroid:                     │
│   [c1][c2][c3]...[c96]  (each 1 byte = 256 centroids)             │
│        ↓                                                           │
│   Compressed: 96 bytes instead of 3072 bytes (32x compression)     │
│                                                                    │
│   Search: Lookup distance tables, sum subvector distances          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Emergent Index (Adaptive)

The flagship index that evolves its own configuration.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Emergent Index                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                   │
│  │  Initial State  │  Uses Flat index before evolution                 │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │    evolve()     │  Runs MAP-Elites on sample data                   │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ├──────────────────────────────────────────────┐              │
│           │                                              │              │
│           ▼                                              ▼              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   Index QD      │  │   Insert QD     │  │  Best Elite     │         │
│  │ Evolve type +   │  │ Evolve SIMD     │  │  Selection      │         │
│  │ hyperparameters │  │ insert strategy │  │                 │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┴────────────────────┘                   │
│                                │                                        │
│                                ▼                                        │
│                    ┌─────────────────────┐                              │
│                    │  Dynamic Index      │                              │
│                    │  (HNSW/Flat/IVF)    │                              │
│                    │  with optimal params │                              │
│                    └─────────────────────┘                              │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Evolution Algorithm Details

### MAP-Elites Quality-Diversity

```
Algorithm MAP-Elites:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. INITIALIZE                                                          │
│     archive ← empty 6×6×6 grid                                          │
│     seed_archive(10 heuristic configurations)                           │
│                                                                         │
│  2. EVOLVE (for each generation)                                        │
│     for i in 0..population_size:                                        │
│         parent ← random_select(archive)                                 │
│         offspring ← mutate(parent, 1-2 params)                          │
│         fitness, behavior ← evaluate(offspring, sample_vectors)         │
│         cell ← map_to_cell(behavior)                                    │
│                                                                         │
│         if archive[cell].empty OR fitness > archive[cell].fitness:      │
│             archive[cell] ← offspring                                   │
│                                                                         │
│  3. EARLY STOPPING                                                      │
│     if best_fitness > threshold: break                                  │
│     if no_improvement_for(3 generations): break                         │
│                                                                         │
│  4. RETURN                                                              │
│     best_elite ← max(archive, by=fitness)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Fitness Function

```
fitness = (recall^w_r × speed^w_s × memory^w_m × build^w_b)^(1/Σw)

Where:
- recall: Recall@10 (0.0 - 1.0)
- speed: Normalized query latency (higher = faster)
- memory: Normalized memory efficiency (higher = less memory)
- build: Normalized build time (higher = faster build)

99% Recall Floor:
if recall < 0.99:
    penalty = (recall / 0.99)^3
    fitness *= penalty
```

### Genome Mutation

```
Mutation operators for IndexGenome:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  1. SELECT 1-2 random parameters to mutate                         │
│                                                                    │
│  2. For each parameter, choose randomly from discrete set:         │
│                                                                    │
│     index_type: [Flat, Hnsw, Ivf]                                 │
│     hnsw_m: [8, 12, 16, 24, 32, 48]                               │
│     hnsw_ef_construction: [50, 100, 150, 200, 300, 400]           │
│     hnsw_ef_search: [10, 20, 40, 50, 80, 100, 150]               │
│     ivf_partitions: [32, 64, 128, 256, 512]                       │
│     ivf_nprobe: [2, 4, 8, 16, 32, 64]                             │
│                                                                    │
│  3. CROSSOVER (when two parents):                                  │
│     For each parameter: randomly select from parent A or B         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## SIMD Architecture

### Platform Detection and Dispatch

```
┌────────────────────────────────────────────────────────────────────────┐
│                          SIMD Dispatch                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  #[cfg(target_arch = "aarch64")]    → ARM NEON implementation          │
│  #[cfg(target_arch = "x86_64")]     → Wide crate (AVX2/SSE)            │
│  #[cfg(not(any(...)))]              → Scalar fallback                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      ARM NEON (Apple Silicon)                   │   │
│  │                                                                 │   │
│  │   // 4-way unrolled dot product with FMA                        │   │
│  │   unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {     │   │
│  │       let mut sum0 = vdupq_n_f32(0.0);                          │   │
│  │       let mut sum1 = vdupq_n_f32(0.0);                          │   │
│  │       let mut sum2 = vdupq_n_f32(0.0);                          │   │
│  │       let mut sum3 = vdupq_n_f32(0.0);                          │   │
│  │                                                                 │   │
│  │       for chunk in 0..n/16 {  // 16 floats per iteration        │   │
│  │           sum0 = vfmaq_f32(sum0, va0, vb0);  // FMA             │   │
│  │           sum1 = vfmaq_f32(sum1, va1, vb1);                     │   │
│  │           sum2 = vfmaq_f32(sum2, va2, vb2);                     │   │
│  │           sum3 = vfmaq_f32(sum3, va3, vb3);                     │   │
│  │       }                                                         │   │
│  │                                                                 │   │
│  │       vaddvq_f32(vadd(vadd(sum0,sum1), vadd(sum2,sum3)))       │   │
│  │   }                                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Insert Strategies

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Insert Strategy Variants                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SimdSequential:   Process one vector at a time with SIMD              │
│  SimdBatch:        Batch multiple vectors, normalize together          │
│  SimdParallel:     Rayon parallelism + SIMD per thread                 │
│  SimdChunked:      Process in L2 cache-sized chunks                    │
│  SimdUnrolled:     4-way loop unrolling for CPU pipelining             │
│  SimdInterleaved:  Two-pass (compute norms, then scale)                │
│  PreNormalized:    Skip normalization (user provides normalized)       │
│                                                                         │
│  Performance: ~5.6M vectors/second on modern CPUs                       │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Concurrency Model

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Concurrency Architecture                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                   │
│  │  Tokio Runtime  │  Async I/O for API server                         │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │ parking_lot     │  High-performance RwLock for index access         │
│  │ RwLock<Index>   │  - Multiple concurrent readers                    │
│  └────────┬────────┘  - Single writer with priority                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │     Rayon       │  Data parallelism for:                            │
│  │   Thread Pool   │  - Batch distance computations                    │
│  └─────────────────┘  - Parallel fitness evaluation                    │
│                                                                         │
│  ┌─────────────────┐                                                   │
│  │    DashMap      │  Concurrent HashMap for context graph             │
│  └─────────────────┘                                                   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Insert Flow

```
User Request                                                     
     │                                                           
     ▼                                                           
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ API Server  │───▶│  Validate   │───▶│  Normalize  │───▶│   Insert    │
│ /insert     │    │  Dimension  │    │  (SIMD)     │    │  to Index   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
                                                               ▼
                                                         ┌─────────────┐
                                                         │   Response  │
                                                         │  (NodeId)   │
                                                         └─────────────┘
```

### Search Flow

```
User Request                                                     
     │                                                           
     ▼                                                           
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐
│ API Server  │───▶│  Validate   │───▶│          Index Search           │
│ /search     │    │  Query      │    │                                 │
└─────────────┘    └─────────────┘    │  Flat: Linear scan all vectors  │
                                      │  HNSW: Greedy graph traversal   │
                                      │  IVF:  nprobe partition search  │
                                      └─────────────────────────────────┘
                                                               │
                                                               ▼
                                                         ┌─────────────┐
                                                         │  Top-k      │
                                                         │  Results    │
                                                         └─────────────┘
```

### Evolution Flow

```
┌─────────────┐                                                  
│   evolve()  │                                                  
└──────┬──────┘                                                  
       │                                                         
       ▼                                                         
┌─────────────┐    ┌─────────────┐    ┌─────────────┐           
│    Seed     │───▶│  Generation │───▶│   Evaluate  │           
│   Archive   │    │    Loop     │    │   Fitness   │           
└─────────────┘    └──────┬──────┘    └──────┬──────┘           
                          │                  │                   
                          ▼                  ▼                   
                   ┌─────────────┐    ┌─────────────┐           
                   │   Mutate    │    │ Map to Cell │           
                   │  Offspring  │    │  (Behavior) │           
                   └─────────────┘    └──────┬──────┘           
                                             │                   
                                             ▼                   
                                      ┌─────────────┐           
                                      │   Update    │           
                                      │   Archive   │           
                                      └──────┬──────┘           
                                             │                   
                                             ▼                   
                                      ┌─────────────┐           
                                      │ Best Elite  │           
                                      │  Selection  │           
                                      └─────────────┘           
```

## Memory Layout

### Vector Storage

```
Flat Index:
┌────────────────────────────────────────────────────────────────┐
│  HashMap<NodeId, Embedding>                                    │
│                                                                │
│  NodeId (8 bytes) → Embedding (dim × 4 bytes)                  │
│                                                                │
│  Example (768 dims): 8 + 3072 = 3080 bytes per vector          │
└────────────────────────────────────────────────────────────────┘

HNSW Index:
┌────────────────────────────────────────────────────────────────┐
│  Vectors: HashMap<NodeId, Embedding>    ~3080 bytes/vec        │
│  Graph:   Vec<Vec<NodeId>>              ~M×8 bytes/vec/level   │
│  Levels:  Vec<usize>                    ~8 bytes/vec           │
│                                                                │
│  Total: ~4000-5000 bytes per vector (depending on M)           │
└────────────────────────────────────────────────────────────────┘

PQ Index:
┌────────────────────────────────────────────────────────────────┐
│  Codes:      Vec<Vec<u8>>               M bytes per vector     │
│  Codebooks:  Vec<Codebook>              K×(dim/M)×4 bytes      │
│                                                                │
│  Example: M=96, K=256, dim=768                                 │
│  Per vector: 96 bytes (32x compression!)                       │
└────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Operation | Flat | HNSW | IVF | PQ |
|-----------|------|------|-----|-----|
| Insert | O(1) | O(log N × M) | O(1)* | O(M) |
| Search | O(N × dim) | O(log N × ef) | O(nprobe × N/k) | O(N × M) |
| Memory | 100% | 130-150% | 105% | 3-5% |
| Recall | 100% | 95-100% | 90-99% | 80-95% |

*IVF insert is O(1) after training; training is O(N × k × iters)

## Reliability Features

1. **99% Recall Floor**: Cubic penalty for low-recall configurations
2. **Heuristic Seeding**: 10 diverse initial configurations
3. **Archive Niching**: Prevents convergence to single optimum
4. **Ground Truth Validation**: Recall computed against Flat index
5. **Early Stopping**: Prevents overfitting to evolution samples

## Future Architecture Considerations

- **Persistence Layer**: RocksDB integration for durable storage
- **Distributed Mode**: Sharding and replication
- **Streaming Updates**: Real-time index updates without full rebuild
- **GPU Acceleration**: CUDA/Metal kernels for distance computation
- **Hybrid Indices**: Combining HNSW with PQ for large-scale deployment
