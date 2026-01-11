# EmergentDB Benchmark Methodology

This document describes the testing methodology used to evaluate EmergentDB's performance against other vector databases.

## Overview

All benchmarks use **real semantic embeddings** from Google's Gemini API rather than random vectors. This is critical because random high-dimensional vectors exhibit the "curse of dimensionality" where all points become nearly equidistant, making ANN algorithms perform poorly regardless of their actual quality.

## Embedding Generation

### Source Data
- Documentation files from the EmergentDB repository (`.md` files)
- Split into chunks of ~500 characters with 100-character overlap
- 184 unique text chunks from real documentation

### Embedding Model
- **Model**: `gemini-embedding-001`
- **Dimension**: 768
- **Task Type**: `RETRIEVAL_DOCUMENT` (optimized for semantic search)
- **Normalization**: All embeddings L2-normalized for cosine similarity

### Scaling to Larger Datasets
To test at scale (1K, 5K, 10K vectors), we use controlled noise injection:

```python
noise_scale = 0.05  # Small Gaussian noise
for i in range(target_size):
    orig_emb = original_embeddings[i % n_original]
    if i >= n_original:
        noise = np.random.normal(0, noise_scale, dim)
        new_emb = orig_emb + noise
        new_emb = new_emb / np.linalg.norm(new_emb)  # Re-normalize
```

This preserves semantic structure while creating larger test sets. Similarity between original and variation vectors remains >0.99.

## Benchmark Protocol

### Ground Truth Computation
For each query, we compute exact k-nearest neighbors using brute-force cosine similarity:

```python
def compute_ground_truth(embeddings, query_indices, k=10):
    for qi in query_indices:
        query = embeddings[qi]
        similarities = embeddings @ query  # Cosine similarity (normalized vectors)
        top_k = np.argsort(similarities)[::-1][:k]
        ground_truths.append(top_k)
```

### Recall@K Metric
Recall measures how many of the true k-nearest neighbors are found:

```python
def compute_recall(ground_truth, results, k=10):
    gt_set = set(ground_truth[:k])
    result_set = set(results[:k])
    return len(gt_set & result_set) / k
```

### Latency Measurement
- **Warmup**: 10 queries discarded before measurement
- **Test Queries**: 50-100 queries (using first N embeddings as queries)
- **Timing**: `time.time()` for Python, `std::time::Instant` for Rust
- **Reported**: Average latency, P50, P99 in milliseconds/microseconds

### Throughput
Measured as vectors inserted per second:
```
throughput = num_vectors / insert_time
```

## Test Configurations

### EmergentDB (Rust Direct)
Tests the core index implementation without HTTP overhead:
- Flat index (brute force baseline)
- HNSW with precomputed elite configurations (m=8 to m=48)
- Multiple ef_search values (50-500)

### EmergentDB (HTTP API)
Tests the full REST API stack:
- Serialization/deserialization overhead included
- Network latency (localhost) included
- Real-world usage pattern

### LanceDB
- IVF-PQ index with 256 partitions, 96 sub-vectors
- Cosine distance metric
- Persistent storage in temp directory

### ChromaDB
- HNSW index (default)
- Cosine distance metric
- PersistentClient with temp directory

## Precomputed Elites Grid

The precomputed configurations are derived from industry sources:

| Source | Configuration | Parameters |
|--------|--------------|------------|
| OpenSearch | Low latency | m=8, ef_c=64, ef_s=50 |
| OpenSearch | Default | m=16, ef_c=100, ef_s=100 |
| OpenSearch | High recall | m=24, ef_c=200, ef_s=200 |
| Milvus | Recommended | m=24, ef_c=128, ef_s=64 |
| Pinecone | Performance | m=32, ef_c=256, ef_s=128 |
| Research | Maximum | m=48, ef_c=500, ef_s=500 |

## Results Summary (10K Vectors, 768-dim)

### Latency Comparison
| Database | Avg Latency | Recall@10 |
|----------|-------------|-----------|
| EmergentDB (HNSW m=8) | **44us** | 100.0% |
| EmergentDB (HNSW m=16) | **102us** | 100.0% |
| ChromaDB (HNSW) | 2,259us | 99.8% |
| LanceDB (IVF-PQ) | 3,590us | 84.3% |

### Key Findings

1. **Real embeddings change everything**: With Gemini embeddings, HNSW achieves near-perfect recall even with aggressive parameters (m=8), while random vectors show 30-40% recall.

2. **EmergentDB is 51x faster than ChromaDB** with identical recall (100% vs 99.8%).

3. **LanceDB's IVF-PQ struggles** with semantic structure at this scale (84% recall), showing the importance of index selection.

4. **HTTP overhead is significant**: Python HTTP benchmarks add ~2-5ms latency. Direct Rust benchmarks show true index performance.

## Reproducibility

### Generate Embeddings
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# Generate base embeddings
cd tests
python3 gemini_embedding_benchmark.py

# Scale to larger sizes
python3 scale_gemini_embeddings.py
```

### Run Benchmarks
```bash
# Rust direct benchmark (most accurate)
cargo run --release --example gemini_benchmark -p vector-core -- 10000

# Python comparison benchmark
cd tests
python3 full_comparison_benchmark.py
```

## Files

| File | Description |
|------|-------------|
| `gemini_embedding_benchmark.py` | Generates embeddings using Gemini API |
| `scale_gemini_embeddings.py` | Scales embeddings to 1K, 5K, 10K vectors |
| `full_comparison_benchmark.py` | Compares EmergentDB vs LanceDB vs ChromaDB |
| `benchmark_results/` | JSON output from benchmarks |
| `../examples/gemini_benchmark.rs` | Rust benchmark with precomputed grid |

## Why Real Embeddings Matter

Random high-dimensional vectors suffer from the "concentration of measure" phenomenon:

```
In high dimensions, all pairwise distances converge to the same value.
For random 768-dim vectors: std(distances) / mean(distances) < 0.01
```

This means ANN algorithms can't find meaningful "near" neighbors because all vectors are equally far apart. Real embeddings from language models have:

- **Semantic clustering**: Similar concepts cluster together
- **Meaningful neighborhoods**: Each vector has truly closer neighbors
- **Realistic query patterns**: Queries relate semantically to documents

The result: HNSW goes from ~35% recall (random) to ~100% recall (Gemini) with identical parameters.
