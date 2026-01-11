#!/usr/bin/env python3
"""
Full Comparison Benchmark: EmergentDB vs LanceDB vs ChromaDB
Uses real Gemini embeddings to test semantic search performance.

Run with: python3 full_comparison_benchmark.py
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Scales to test
SCALES = [1000, 5000, 10000]
K = 10  # Top-K results
N_QUERIES = 100  # Number of search queries
N_WARMUP = 10  # Warmup queries

def load_embeddings(scale: int) -> Tuple[List[List[float]], List[str]]:
    """Load Gemini embeddings at specified scale."""
    if scale <= 184:
        path = Path(__file__).parent / "gemini_embeddings.json"
    else:
        path = Path(__file__).parent / f"gemini_embeddings_{scale}.json"

    if not path.exists():
        print(f"  Embeddings file not found: {path}")
        print(f"  Run: python3 scale_gemini_embeddings.py")
        return [], []

    with open(path) as f:
        data = json.load(f)

    embeddings = [e["vector"] for e in data["embeddings"]]
    texts = [e["text"] for e in data["embeddings"]]

    return embeddings, texts

def compute_recall(ground_truth: List[int], results: List[int], k: int = K) -> float:
    """Compute recall@k."""
    gt_set = set(ground_truth[:k])
    result_set = set(results[:k])
    return len(gt_set & result_set) / min(len(gt_set), k)

def compute_ground_truth(embeddings: List[List[float]], query_indices: List[int], k: int = K) -> List[List[int]]:
    """Compute ground truth using brute force."""
    embeddings_np = np.array(embeddings)
    ground_truths = []

    for qi in query_indices:
        query = embeddings_np[qi]
        # Cosine similarity (embeddings are normalized)
        similarities = embeddings_np @ query
        top_k = np.argsort(similarities)[::-1][:k].tolist()
        ground_truths.append(top_k)

    return ground_truths

# =============================================================================
# EmergentDB Benchmark
# =============================================================================

def benchmark_emergentdb(embeddings: List[List[float]], texts: List[str],
                         query_indices: List[int], ground_truths: List[List[int]],
                         config_name: str = "auto") -> Dict[str, Any]:
    """Benchmark EmergentDB with precomputed configurations."""
    import requests

    base_url = "http://localhost:3000"

    # Check if server is running
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code != 200:
            return {"error": "Server not responding"}
    except:
        return {"error": "Could not connect to server"}

    # Reset index
    try:
        requests.post(f"{base_url}/index/reset", timeout=5)
    except:
        pass

    # Insert vectors
    insert_data = {
        "vectors": [
            {"id": i, "vector": emb}
            for i, emb in enumerate(embeddings)
        ]
    }

    insert_start = time.time()
    resp = requests.post(f"{base_url}/vectors/batch_insert", json=insert_data, timeout=120)
    insert_time = time.time() - insert_start

    if resp.status_code != 200:
        return {"error": f"Insert failed: {resp.text}"}

    # Configure index based on config_name
    build_start = time.time()
    if config_name == "auto":
        # Use evolution
        resp = requests.post(f"{base_url}/qd/evolve", timeout=300)
        if resp.status_code == 200:
            result = resp.json()
            actual_config = result.get("index_type", "unknown")
        else:
            actual_config = "flat"
    elif config_name == "flat":
        actual_config = "flat"
    else:
        # Use precomputed config via API
        resp = requests.post(f"{base_url}/index/configure",
                           json={"index_type": config_name}, timeout=120)
        actual_config = config_name

    build_time = time.time() - build_start

    # Warmup
    for i in range(N_WARMUP):
        qi = query_indices[i % len(query_indices)]
        requests.post(f"{base_url}/vectors/search",
                     json={"query": embeddings[qi], "k": K})

    # Search benchmark
    latencies = []
    recalls = []

    for i, qi in enumerate(query_indices):
        query = embeddings[qi]

        start = time.time()
        resp = requests.post(f"{base_url}/vectors/search",
                           json={"query": query, "k": K})
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

        if resp.status_code == 200:
            results = [r["id"] for r in resp.json().get("results", [])]
            recall = compute_recall(ground_truths[i], results)
            recalls.append(recall)

    return {
        "database": f"EmergentDB ({actual_config})",
        "insert_time": insert_time,
        "build_time": build_time,
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_recall": np.mean(recalls) * 100,
        "throughput": len(embeddings) / insert_time
    }

# =============================================================================
# LanceDB Benchmark
# =============================================================================

def benchmark_lancedb(embeddings: List[List[float]], texts: List[str],
                      query_indices: List[int], ground_truths: List[List[int]]) -> Dict[str, Any]:
    """Benchmark LanceDB."""
    try:
        import lancedb
        import pyarrow as pa
    except ImportError:
        return {"error": "LanceDB not installed. Run: pip install lancedb"}

    # Create temp directory
    tmp_dir = tempfile.mkdtemp()

    try:
        db = lancedb.connect(tmp_dir)

        # Prepare data
        data = [
            {"id": i, "vector": emb, "text": texts[i] if i < len(texts) else ""}
            for i, emb in enumerate(embeddings)
        ]

        # Insert
        insert_start = time.time()
        table = db.create_table("vectors", data)
        insert_time = time.time() - insert_start

        # Build index
        build_start = time.time()
        table.create_index(metric="cosine", num_partitions=256, num_sub_vectors=96)
        build_time = time.time() - build_start

        # Warmup
        for i in range(N_WARMUP):
            qi = query_indices[i % len(query_indices)]
            table.search(embeddings[qi]).limit(K).to_list()

        # Search benchmark
        latencies = []
        recalls = []

        for i, qi in enumerate(query_indices):
            query = embeddings[qi]

            start = time.time()
            results = table.search(query).limit(K).to_list()
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            result_ids = [r["id"] for r in results]
            recall = compute_recall(ground_truths[i], result_ids)
            recalls.append(recall)

        return {
            "database": "LanceDB",
            "insert_time": insert_time,
            "build_time": build_time,
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_recall": np.mean(recalls) * 100,
            "throughput": len(embeddings) / insert_time
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# =============================================================================
# ChromaDB Benchmark
# =============================================================================

def benchmark_chromadb(embeddings: List[List[float]], texts: List[str],
                       query_indices: List[int], ground_truths: List[List[int]]) -> Dict[str, Any]:
    """Benchmark ChromaDB."""
    try:
        import chromadb
    except ImportError:
        return {"error": "ChromaDB not installed. Run: pip install chromadb"}

    # Create temp directory
    tmp_dir = tempfile.mkdtemp()

    try:
        # New ChromaDB API
        client = chromadb.PersistentClient(path=tmp_dir)

        collection = client.create_collection(
            name="vectors",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert in batches (ChromaDB has limits)
        batch_size = 5000
        insert_start = time.time()

        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_ids = [str(j) for j in range(i, batch_end)]
            batch_embeddings = embeddings[i:batch_end]
            batch_texts = texts[i:batch_end] if texts else None

            if batch_texts:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts
                )
            else:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )

        insert_time = time.time() - insert_start
        build_time = 0  # ChromaDB builds index during insert

        # Warmup
        for i in range(N_WARMUP):
            qi = query_indices[i % len(query_indices)]
            collection.query(query_embeddings=[embeddings[qi]], n_results=K)

        # Search benchmark
        latencies = []
        recalls = []

        for i, qi in enumerate(query_indices):
            query = embeddings[qi]

            start = time.time()
            results = collection.query(query_embeddings=[query], n_results=K)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            result_ids = [int(id) for id in results["ids"][0]]
            recall = compute_recall(ground_truths[i], result_ids)
            recalls.append(recall)

        return {
            "database": "ChromaDB",
            "insert_time": insert_time,
            "build_time": build_time,
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_recall": np.mean(recalls) * 100,
            "throughput": len(embeddings) / insert_time
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# =============================================================================
# Main Benchmark
# =============================================================================

def print_results_table(results: List[Dict[str, Any]], scale: int):
    """Print results in a nice table."""
    print(f"\n{'='*100}")
    print(f"  BENCHMARK RESULTS: {scale} vectors (768 dimensions, Gemini embeddings)")
    print(f"{'='*100}")

    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    if valid_results:
        print(f"\n  {'Database':<25} {'Avg Latency':>12} {'P99 Latency':>12} {'Recall':>10} {'Throughput':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

        # Sort by latency
        valid_results.sort(key=lambda x: x.get("avg_latency_ms", float('inf')))

        baseline_latency = valid_results[-1]["avg_latency_ms"] if valid_results else 1

        for r in valid_results:
            speedup = baseline_latency / r["avg_latency_ms"] if r["avg_latency_ms"] > 0 else 0
            speedup_str = f"({speedup:.0f}x)" if speedup > 1.5 else ""

            print(f"  {r['database']:<25} {r['avg_latency_ms']:>9.2f}ms {r['p99_latency_ms']:>9.2f}ms "
                  f"{r['avg_recall']:>8.1f}% {r['throughput']:>9.0f}/s {speedup_str}")

    if error_results:
        print(f"\n  Errors:")
        for r in error_results:
            print(f"    {r.get('database', 'Unknown')}: {r['error']}")

def main():
    print("="*100)
    print("  EmergentDB vs LanceDB vs ChromaDB - Full Comparison Benchmark")
    print("  Using Real Gemini Embeddings (768 dimensions)")
    print("="*100)

    all_results = {}

    for scale in SCALES:
        print(f"\n\n{'#'*100}")
        print(f"  SCALE: {scale} vectors")
        print(f"{'#'*100}")

        # Load embeddings
        print(f"\n  Loading embeddings...")
        embeddings, texts = load_embeddings(scale)

        if not embeddings:
            print(f"  Skipping scale {scale} - no embeddings available")
            continue

        print(f"  Loaded {len(embeddings)} embeddings")

        # Compute ground truth
        print(f"  Computing ground truth for {N_QUERIES} queries...")
        query_indices = list(range(min(N_QUERIES, len(embeddings))))
        ground_truths = compute_ground_truth(embeddings, query_indices)

        results = []

        # Benchmark EmergentDB (auto-evolved)
        print(f"\n  Benchmarking EmergentDB (auto)...")
        result = benchmark_emergentdb(embeddings, texts, query_indices, ground_truths, "auto")
        results.append(result)
        if "error" not in result:
            print(f"    Latency: {result['avg_latency_ms']:.2f}ms, Recall: {result['avg_recall']:.1f}%")
        else:
            print(f"    Error: {result['error']}")

        # Benchmark LanceDB
        print(f"\n  Benchmarking LanceDB...")
        result = benchmark_lancedb(embeddings, texts, query_indices, ground_truths)
        results.append(result)
        if "error" not in result:
            print(f"    Latency: {result['avg_latency_ms']:.2f}ms, Recall: {result['avg_recall']:.1f}%")
        else:
            print(f"    Error: {result['error']}")

        # Benchmark ChromaDB
        print(f"\n  Benchmarking ChromaDB...")
        result = benchmark_chromadb(embeddings, texts, query_indices, ground_truths)
        results.append(result)
        if "error" not in result:
            print(f"    Latency: {result['avg_latency_ms']:.2f}ms, Recall: {result['avg_recall']:.1f}%")
        else:
            print(f"    Error: {result['error']}")

        # Print results table
        print_results_table(results, scale)
        all_results[scale] = results

    # Save results
    output_path = Path(__file__).parent / "benchmark_results" / "full_comparison.json"
    output_path.parent.mkdir(exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*100)
    print("  SUMMARY")
    print("="*100)

    for scale, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            fastest = min(valid, key=lambda x: x["avg_latency_ms"])
            print(f"\n  {scale} vectors: {fastest['database']} fastest at {fastest['avg_latency_ms']:.2f}ms "
                  f"with {fastest['avg_recall']:.1f}% recall")

if __name__ == "__main__":
    main()
