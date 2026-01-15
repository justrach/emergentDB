# grep-core

Fast, embedding-free code search with Quality-Diversity optimization.

Part of [EmergentDB](https://github.com/justrach/emergentDB) - extends MAP-Elites evolution from vector indices to discrete search strategies.

## Performance

**18x faster** by automatically selecting the optimal search strategy:

```
Strategy Comparison (ripgrep repo, 98 files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EmergentSearch  ████████████████████████████████████████ 49,000 files/sec
literal_only    ████████████████████████████████████████ 49,000 files/sec
inverted_index  █████                                     6,125 files/sec
trigram+BM25    ██▎                                       2,722 files/sec

                └────────────────────────────────────────┘
                         18x faster than wrong choice!
```

| Repository | Files | Best Strategy | Speed | vs Default |
|------------|-------|---------------|-------|------------|
| ripgrep | 98 | literal_only | 49,000/sec | **18x faster** |
| python-genai | 360 | literal_only | 60,000/sec | **16.5x faster** |

## How It Works

### The Problem

Most code search tools use a fixed strategy (e.g., trigram indexing + BM25 ranking). But the optimal strategy depends on:
- Repository size (100 files vs 100,000 files)
- Query type (literal vs regex)
- Query frequency (one-off vs repeated)

Wrong strategy = 18x slower.

### The Solution: Quality-Diversity Optimization

We precompute **44 elite strategies** using MAP-Elites evolution across two systems:

**LoadQD** (12 elites) - File loading strategies:
- `mmap_sequential` - 59 GB/s for large files
- `buffered_large` - Safe default for mixed workloads
- `parallel_aggressive` - Max throughput on NVMe

**SearchQD** (32 elites) - Search strategies:
- `literal_only` - No index, SIMD scan (best for <1000 files)
- `trigram_tfidf` - Fuzzy matching with TF-IDF ranking
- `inverted_simple` - Fast exact matching for repeated queries
- `large_codebase` - Optimized for 10K+ files

**EmergentSearch** auto-selects the optimal strategy:

```rust
// Decision tree based on workload characteristics
let elite = EmergentSearch::select_elite(
    file_count,      // Repository size
    pattern,         // Search pattern
    is_repeated,     // Repeated query?
);
// Returns optimal strategy: 100% optimal on tested repos
```

## Usage

### As a Library

```rust
use grep_core::{CodeIndex, IndexConfig, search_qd::EmergentSearch};

// Build file index
let mut index = CodeIndex::new(IndexConfig {
    root: "./src".into(),
    file_types: Some(vec!["rs".into(), "py".into()]),
    ..Default::default()
});
index.build()?;

// Search with auto-selected optimal strategy
let results = index.search("async fn", 10)?;

for result in results {
    println!("{}: {} matches", result.path.display(), result.matches.len());
}
```

### As MCP Server

```bash
# Build
cargo build --release -p grep-mcp

# Run
./target/release/grep-mcp
```

Connect from Claude Code or any MCP client.

### Run Benchmarks

```bash
# Build benchmark
cargo build --release -p grep-core --example benchmark_qd

# Run against any repo
./target/release/examples/benchmark_qd /path/to/repo "search pattern"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        grep-core                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   LoadQD     │  │  SearchQD    │  │   EmergentSearch     │  │
│  │  ──────────  │  │  ──────────  │  │   ──────────────     │  │
│  │  12 elites   │  │  32 elites   │  │   Auto-selector      │  │
│  │  File I/O    │  │  Indexing    │  │   Decision tree      │  │
│  │  strategies  │  │  + Ranking   │  │   100% optimal       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Index Types        Ranking Algorithms     Tokenizers           │
│  ────────────       ──────────────────     ──────────           │
│  • None (scan)      • MatchCount           • Whitespace         │
│  • Trigram          • BM25                 • CamelCase          │
│  • Inverted         • TF-IDF               • Code-aware         │
│                     • PositionWeighted     • Trigrams           │
└─────────────────────────────────────────────────────────────────┘
```

## Benchmark Results

### Search Strategy Comparison

**ripgrep (98 Rust files, "fn search"):**

| Strategy | Index Time | Search Time | Files/sec | vs Optimal |
|----------|-----------|-------------|-----------|------------|
| literal_only | 0ms | 2ms | **49,000** | 100% |
| inverted_simple | 14ms | 2ms | 6,125 | 12.5% |
| balanced | 31ms | 5ms | 2,722 | 5.6% |

**python-genai (360 Python files, "async def"):**

| Strategy | Index Time | Search Time | Files/sec | vs Optimal |
|----------|-----------|-------------|-----------|------------|
| literal_only | 0ms | 6ms | **60,000** | 100% |
| inverted_match | 33ms | 7ms | 9,000 | 15% |
| balanced | 75ms | 24ms | 3,636 | 6.1% |

### Load Strategy Comparison (652KB file)

| Strategy | Throughput | vs Optimal |
|----------|------------|------------|
| mmap_sequential | **59.3 GB/s** | 100% |
| huge_files | 50.2 GB/s | 85% |
| parallel_aggressive | 12.5 GB/s | 21% |

### EmergentSearch Accuracy

| Repo | Files | Selected | Efficiency |
|------|-------|----------|------------|
| ripgrep | 98 | literal_only | **100%** |
| python-genai | 360 | literal_only | **100%** |

## Key Insight

For codebases under 1,000 files, **don't build an index**. SIMD-accelerated linear scan beats indexed search because:

1. Index build time dominates (30-75ms)
2. Search time is negligible either way (2-6ms)
3. Modern CPUs + NVMe = 60,000 files/sec scanning

EmergentSearch knows this and picks `literal_only` for small repos.

## Comparison with Vector Search

| Aspect | grep-core | vector-core |
|--------|-----------|-------------|
| Query | `"class AuthError"` | `"authentication failure"` |
| Matching | Exact/regex | Semantic similarity |
| Speed | 49,000-60,000 files/sec | 42μs per query |
| Use case | Find identifiers | Find concepts |

**Best of both:** Use grep-core for exact matches, vector-core for semantic search.

## Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Core types, exports |
| `src/load_qd.rs` | 12 precomputed file loading elites |
| `src/search_qd.rs` | 32 precomputed search elites + EmergentSearch |
| `src/index.rs` | Trigram and inverted index implementations |
| `src/search.rs` | Search engine with ranking algorithms |
| `examples/benchmark_qd.rs` | Benchmark runner |

## License

MIT
