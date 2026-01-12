# EmergentDB Grep Extension - Gemini Integration Guide

This document describes how grep-core integrates with Gemini embeddings for hybrid code search.

## Hybrid Search Architecture

```
User Query: "authentication error handling"
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  grep-core    │       │  vector-core  │
│  (Discrete)   │       │  (Semantic)   │
├───────────────┤       ├───────────────┤
│ "AuthError"   │       │ Gemini embed  │
│ "auth.*error" │       │ 768-dim vec   │
│ Exact match   │       │ Cosine sim    │
└───────┬───────┘       └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    ▼
            ┌───────────────┐
            │  Merged       │
            │  Results      │
            │  (RRF/Linear) │
            └───────────────┘
```

## Why Hybrid Search?

| Query Type | Best Approach | Example |
|------------|---------------|---------|
| Exact identifier | grep-core | `class AuthenticationError` |
| Semantic concept | vector-core + Gemini | "how does login work" |
| Regex pattern | grep-core | `fn test_.*async` |
| Natural language | vector-core + Gemini | "fix memory leak" |
| Mixed | Both | "AuthError message formatting" |

## Gemini Embedding Integration

EmergentDB uses Gemini's `text-embedding-004` model for semantic search:

```python
# From examples/ingestion/ingest.py
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.embed_content(
    model="text-embedding-004",
    content=text_chunk
)
embedding = response.embedding  # 768-dim vector
```

### Complementary Strengths

| Gemini Embeddings | grep-core |
|-------------------|-----------|
| "auth" ≈ "login" ≈ "signin" | "auth" = exact match only |
| Semantic similarity | Syntactic matching |
| ~100ms per embed | ~0.02ms per search |
| Best for concepts | Best for identifiers |

## Benchmark Results

### grep-core Performance (Discrete Search)

**Test: ripgrep repository (98 Rust files)**

| Strategy | Speed | vs Optimal |
|----------|-------|------------|
| literal_only | 49,000 files/sec | 100% |
| inverted_simple | 6,125 files/sec | 12.5% |
| balanced (trigram+BM25) | 2,722 files/sec | 5.6% |

**Key insight**: EmergentSearch auto-selector achieves **100% optimal** by picking the right strategy.

**Test: python-genai repository (360 Python files)**

| Strategy | Speed | vs Optimal |
|----------|-------|------------|
| literal_only | 60,000 files/sec | 100% |
| inverted_match_camel | 9,000 files/sec | 15% |
| balanced (trigram+BM25) | 3,636 files/sec | 6.1% |

### Combined Latency (Hybrid Search)

```
Discrete search:     0.02ms (grep-core)
Semantic search:     0.04ms (vector-core, in-memory)
Gemini embedding:  ~80-150ms (API call, cached)
                   ─────────
Total (cold):      ~80-150ms (embedding dominates)
Total (warm):       ~0.06ms (both from cache/memory)
```

## Quality-Diversity Optimization

### LoadQD: File Loading Strategies

12 precomputed elites covering:
- **mmap strategies**: Zero-copy, prefaulted, sequential hints
- **buffered strategies**: Various buffer sizes (4KB - 1MB)
- **parallel strategies**: Thread counts and block sizes

Best performer: `mmap_sequential` at **59.3 GB/s**

### SearchQD: Search Strategies

32 precomputed elites across 3 dimensions:

**Index Type** (4 options):
- None (inline scan)
- Trigram (fuzzy-capable)
- Inverted (exact lookup)

**Ranking Algorithm** (4 options):
- MatchCount
- BM25
- TF-IDF
- PositionWeighted

**Tokenizer** (4 options):
- Whitespace
- CamelCase
- Code-aware
- Trigrams

Coverage: **62.5%** of configuration space

## EmergentSearch Decision Tree

```
file_count ≤ 100?
├─ YES: is_regex?
│       ├─ YES: regex_heavy
│       └─ NO:  literal_only ← 49,000 files/sec
└─ NO:  file_count ≤ 1000?
        ├─ YES: repeated_queries?
        │       ├─ YES: inverted_simple (index pays off)
        │       └─ NO:  literal_only
        └─ NO:  file_count ≤ 10000?
                ├─ YES: balanced or large_codebase
                └─ NO:  massive_parallel
```

## API Usage

### Discrete Search Only

```rust
use grep_core::{CodeIndex, IndexConfig, search_qd::EmergentSearch};

// Build file index
let mut index = CodeIndex::new(IndexConfig {
    root: "./src".into(),
    file_types: Some(vec!["py".into()]),
    ..Default::default()
});
index.build()?;

// Auto-select optimal strategy
let elite = EmergentSearch::select_elite(
    index.files().len(),
    "async def",
    false
);

// Search
let results = index.search_with_strategy("async def", 10, &elite.genome)?;
```

### Hybrid Search (with Gemini)

```python
import httpx
from google import genai

# 1. Get Gemini embedding for semantic search
client = genai.Client(api_key=API_KEY)
embedding = client.models.embed_content(
    model="text-embedding-004",
    content=query
).embedding

# 2. Run both searches in parallel
async with httpx.AsyncClient() as client:
    # Semantic search (vector-core)
    semantic = await client.post(
        "http://localhost:3000/vectors/search",
        json={"query": embedding, "k": 10}
    )

    # Discrete search (grep-core)
    discrete = await client.post(
        "http://localhost:3001/search",
        json={"pattern": query, "k": 10}
    )

# 3. Merge results (Reciprocal Rank Fusion)
merged = rrf_merge(semantic.json(), discrete.json())
```

## Optimization Tips

### When to Use grep-core Only

- Searching for exact identifiers: `getUserById`
- Regex patterns: `fn test_.*`
- Small codebases (<1000 files)
- Latency-critical applications

### When to Use Gemini + vector-core Only

- Natural language queries: "how to handle errors"
- Concept search: "authentication flow"
- Cross-language semantic matching

### When to Use Hybrid

- Mixed queries: "AuthError message format"
- Unknown query intent
- Maximum recall requirements

## Performance Comparison

| Approach | Query Latency | Best For |
|----------|---------------|----------|
| grep-core only | 0.02ms | Exact matching |
| vector-core only | 0.04ms | Semantic search |
| Hybrid (cached) | 0.06ms | Mixed queries |
| Hybrid (cold) | ~100ms | First semantic query |

## Future: DPO-Based Selection

The next evolution will use Direct Preference Optimization (DPO) to learn strategy selection from user feedback:

```
User searches → Strategy selected → Results shown → User clicks
                                                         │
                                    ┌────────────────────┘
                                    ▼
                            Preference signal
                                    │
                                    ▼
                            DPO training
                                    │
                                    ▼
                            Better selection
```

This will enable:
- Personalized strategy selection per user/project
- Continuous improvement from implicit feedback
- Adaptation to new codebases and patterns
