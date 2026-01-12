# EmergentDB Grep Extension - Claude Code Context

This document provides context for Claude Code to understand and work with the grep-core extension effectively.

## Overview

The grep-core crate extends EmergentDB's Quality-Diversity optimization to **discrete code search strategies**. Instead of evolving continuous vector index parameters, we evolve discrete retrieval mechanisms (trigram, BM25, inverted index).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        grep-core                                 │
├─────────────────────────────────────────────────────────────────┤
│  LoadQD          │  SearchQD         │  EmergentSearch          │
│  ────────        │  ─────────        │  ───────────────         │
│  12 elites       │  32 elites        │  Auto-selector           │
│  File loading    │  Search strategy  │  Decision tree           │
│  strategies      │  configurations   │  100% optimal            │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Core types: `CodeIndex`, `FileEntry`, `SearchResult` |
| `src/load_qd.rs` | LoadQD: 12 precomputed file loading elites |
| `src/search_qd.rs` | SearchQD: 32 precomputed search elites + EmergentSearch |
| `src/index.rs` | Trigram and inverted index implementations |
| `src/search.rs` | Search engine with BM25/TF-IDF ranking |
| `examples/benchmark_qd.rs` | Benchmark runner for all strategies |

## LoadQD System

Evolves file loading strategies across a 2D behavior space: **Throughput x Latency**

### Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `mmap_sequential` | Mmap with sequential hints | Large sequential reads |
| `huge_files` | Prefaulted mmap | Very large files (>10MB) |
| `instant_access` | Zero-copy mmap | Random access patterns |
| `buffered_large` | Large buffer reads | Moderate files |
| `parallel_aggressive` | Many threads, small blocks | NVMe with mixed sizes |
| `conservative` | Safe defaults | HDD or unknown storage |

### Performance (652KB file)

```
mmap_sequential     59.3 GB/s  ← fastest
huge_files          50.2 GB/s
instant_access      46.6 GB/s
parallel_aggressive 12.5 GB/s  ← 4.7x slower
```

## SearchQD System

Evolves search strategies across a 3D behavior space: **Index Type x Ranking x Tokenizer**

### Index Types

| Type | Build Cost | Query Cost | Best For |
|------|------------|------------|----------|
| `None` | 0ms | O(N) | <1000 files |
| `Trigram` | ~30-70ms | O(1) lookup | Fuzzy matching |
| `Inverted` | ~15-35ms | O(1) lookup | Exact matching, repeated queries |

### Ranking Algorithms

| Algorithm | Description |
|-----------|-------------|
| `MatchCount` | Simple occurrence counting |
| `BM25` | Probabilistic relevance (best for natural text) |
| `TfIdf` | Term frequency-inverse document frequency |
| `PositionWeighted` | Boost matches near file start |

### Tokenizers

| Tokenizer | Handles |
|-----------|---------|
| `Whitespace` | Basic word splitting |
| `CamelCase` | `getUserName` → `get`, `User`, `Name` |
| `Code` | Identifiers, operators, strings |
| `Trigrams` | 3-character sliding window |

### Performance (98-360 files)

```
literal_only     49,000-60,000 files/sec  ← fastest
inverted_simple   6,000-9,000 files/sec
balanced          2,700-3,600 files/sec   ← 18x slower
```

## EmergentSearch Auto-Selector

Decision tree that picks optimal strategy based on workload characteristics:

```rust
impl EmergentSearch {
    pub fn select_elite(file_count: usize, pattern: &str, repeated_queries: bool) -> &'static SearchElite {
        match (file_count, is_regex, repeated_queries) {
            (0..=100, false, _) => "literal_only",      // Fast scan
            (0..=100, true, _) => "regex_heavy",        // Regex needs care
            (101..=1000, false, false) => "literal_only",
            (101..=1000, true, false) => "regex_heavy",
            (101..=1000, _, true) => "inverted_simple", // Index pays off
            (1001..=10000, _, false) => "balanced",     // Need indexing
            (1001..=10000, _, true) => "large_codebase",
            _ => "massive_parallel",                     // 10K+ files
        }
    }
}
```

### Efficiency

| Repo | Files | Emergent Choice | Efficiency |
|------|-------|-----------------|------------|
| ripgrep | 98 | literal_only | **100%** |
| python-genai | 360 | literal_only | **100%** |

## Usage Examples

### Basic Search

```rust
use grep_core::{CodeIndex, IndexConfig};

let mut index = CodeIndex::new(IndexConfig {
    root: "./src".into(),
    file_types: Some(vec!["rs".into(), "py".into()]),
    ..Default::default()
});
index.build()?;

let results = index.search("async fn", 10)?;
for result in results {
    println!("{}: {} matches", result.path.display(), result.matches.len());
}
```

### Using EmergentSearch

```rust
use grep_core::search_qd::EmergentSearch;

// Auto-select optimal strategy
let elite = EmergentSearch::select_elite(
    file_count,      // Number of files to search
    "fn main",       // Search pattern
    false,           // Is this a repeated query?
);

println!("Using strategy: {}", elite.name);
let engine = SearchEngine::new(elite.genome.clone());
let results = engine.search(&files, pattern)?;
```

### Running Benchmarks

```bash
# Build and run benchmark
cargo build --release -p grep-core --example benchmark_qd
./target/release/examples/benchmark_qd /path/to/repo "search pattern"

# Example output:
# literal_only: 49,000 files/sec
# balanced:      2,722 files/sec
# Emergent efficiency: 100%
```

## Key Insights

1. **Wrong strategy = 18x slower**: Default trigram+BM25 indexing is overkill for small repos
2. **No index beats indexing** for <1000 files: Index build time dominates
3. **EmergentSearch achieves 100% optimal** by matching strategy to workload
4. **44 precomputed elites** = no evolution wait time, instant deployment

## Extending the System

### Adding a New Load Strategy

```rust
// In load_qd.rs
LoadElite {
    name: "my_strategy",
    description: "Custom loading approach",
    genome: LoadGenome {
        strategy: LoadStrategy::Buffered,
        buffer_size: 128 * 1024,
        use_mmap: false,
        mmap_threshold: 10 * 1024 * 1024,
        parallel_threshold: 50 * 1024 * 1024,
        prefault: false,
    },
    behavior: (0.6, 0.4), // (throughput_percentile, latency_percentile)
    fitness: 0.75,
}
```

### Adding a New Search Strategy

```rust
// In search_qd.rs
SearchElite {
    name: "my_search",
    description: "Custom search approach",
    genome: SearchGenome {
        index_type: IndexType::Inverted,
        ranking: RankingAlgorithm::BM25,
        tokenizer: TokenizerType::Code,
        parallel: true,
        case_sensitive: false,
    },
    behavior: (0.7, 0.5, 0.6), // (speed, relevance, memory)
    fitness: 0.80,
}
```

## Comparison with Vector Search

| Aspect | grep-core (Discrete) | vector-core (Continuous) |
|--------|---------------------|-------------------------|
| Search Type | Exact/regex matching | Semantic similarity |
| Index | Trigram/Inverted | HNSW/IVF |
| Evolution | Discrete strategies | Continuous hyperparameters |
| Use Case | "class AuthError" | "authentication failure" |
| Speed | 49,000-60,000 files/sec | 42μs per query |

**Best of both worlds**: Combine discrete search for exact matches with vector search for semantic queries.
